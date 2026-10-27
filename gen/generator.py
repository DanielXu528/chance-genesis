"""Synthetic 25 Hz tracking generator.

The generator produces marking with a mechanism the detector never sees:
  * each defender owns the attackers whose position falls in its *anchor cell*
    (Voronoi cell of the formation anchors, not of the defenders themselves),
  * pick-up happens after a random reaction delay, release happens on a per-role leash,
  * neighbouring markers randomly swap,
  * a marker chases a goal-side point computed from the attacker's position
    0.3-0.6 s in the past, under speed and acceleration limits.
The detector (analysis/marking.py) instead solves a per-frame assignment with the
Hungarian algorithm on positional costs and smooths it over time. Ground-truth labels
are the generator's internal assignment state and are written to a separate file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from common.pitch import (
    DT, FPS, L, NOT_DEFENDING, PLAYER_IDS, W, ZONE, attack_sign, from_rel, keeper,
    opp_goal, outfield, own_goal, team_of, to_rel,
)

# Formation templates in the team's attacking frame (x from own goal line), for a mid block
# with the ball on the halfway line. Keyed by shirt number.
FORMATIONS = {
    "433": {
        1: ("GK", 4.0, 34.0), 2: ("RB", 34.0, 9.0), 4: ("RCB", 27.0, 25.0), 5: ("LCB", 27.0, 43.0),
        3: ("LB", 34.0, 59.0), 6: ("DM", 40.0, 34.0), 8: ("RCM", 50.0, 22.0), 10: ("LCM", 50.0, 46.0),
        7: ("RW", 66.0, 9.0), 9: ("ST", 70.0, 34.0), 11: ("LW", 66.0, 59.0),
    },
    "442": {
        1: ("GK", 4.0, 34.0), 2: ("RB", 32.0, 9.0), 4: ("RCB", 27.0, 26.0), 5: ("LCB", 27.0, 42.0),
        3: ("LB", 32.0, 59.0), 7: ("RM", 50.0, 10.0), 6: ("RCM", 46.0, 28.0), 8: ("LCM", 46.0, 40.0),
        11: ("LM", 50.0, 58.0), 9: ("RST", 66.0, 29.0), 10: ("LST", 66.0, 39.0),
    },
}
CB_ROLES = {"RCB", "LCB"}
# Off-ball movement: Ornstein-Uhlenbeck wander around the anchor and tracking gains.
# Calibrated so a full match lands near published physical outputs (see README).
MOVE = {"ou_tau": 8.0, "ou_att": 1.6, "ou_def": 0.5, "kp_off": 0.55, "kp_mark": 1.4, "anchor_shift": 0.42}
MID_ROLES = {"DM", "RCM", "LCM", "CM"}


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


@dataclass
class Flight:
    start: np.ndarray
    end: np.ndarray
    f0: int
    f1: int
    kind: str                     # pass | shot | intercept
    passer: int
    receiver: Optional[int] = None
    goal: bool = False


Override = Callable[["Sim"], tuple]  # -> (target, vmax, amax, feedforward_vel, kp)


class Sim:
    def __init__(self, rng: np.random.Generator, formations=("433", "442"), possession: int = 0,
                 carrier: Optional[int] = None, ball_rel_x: float = 35.0):
        self.rng = rng
        self.formations = formations
        self.base_rel = np.zeros((22, 2))
        self.role = [""] * 22
        for team in (0, 1):
            for shirt, (role, x, y) in FORMATIONS[formations[team]].items():
                i = team * 11 + shirt - 1
                self.base_rel[i] = (x, y)
                self.role[i] = role
        is_cb = np.array([r in CB_ROLES for r in self.role])
        # Per-player physical and behavioural parameters (hidden from the detector).
        self.delay_f = rng.integers(int(0.3 * FPS), int(0.6 * FPS) + 1, size=22)
        self.goal_side = rng.uniform(1.0, 2.5, size=22)
        self.vmax = rng.uniform(6.8, 7.4, size=22)
        self.amax = rng.uniform(3.6, 4.6, size=22)
        self.leash = np.where(is_cb, 22.0, 15.0)
        self.pick_r = np.where(is_cb, 13.0, 11.5)
        self.zonal_only: set[int] = set()
        for team in (0, 1):
            mids = [i for i in outfield(team) if self.role[i] in MID_ROLES]
            k = rng.integers(0, 2)
            self.zonal_only.update(rng.choice(mids, size=k, replace=False).tolist() if k else [])

        self.possession = possession
        self.frame = 0
        self.ball = from_rel(np.array([ball_rel_x, 34.0]), possession)
        anch = self.anchors(0), self.anchors(1)
        self.pos = np.vstack(anch) + rng.normal(0, 1.5, size=(22, 2))
        self.vel = np.zeros((22, 2))
        self.ou = rng.normal(0, 1, size=(22, 2))
        self.carrier = carrier
        if carrier is not None:
            self.ball = self.pos[carrier].copy()
        self.flight: Optional[Flight] = None
        self.dead = False
        self.assign = np.full(22, ZONE, dtype=int)
        self.pending: dict[int, tuple[int, int]] = {}
        self.locked: set[int] = set()
        self.zonal_locked: set[int] = set()
        self.unmarkable: set[int] = set()   # attackers no defender will pick up (script use)
        self.ov: dict[int, Override] = {}
        self.events: list[dict] = []
        self.director = None
        self.script = None
        self.director_enabled = True
        self.hold_until = 0
        self.dribble_dir = np.zeros(2)
        # recordings
        self.rec_pos: list[np.ndarray] = []
        self.rec_vel: list[np.ndarray] = []
        self.rec_ball: list[np.ndarray] = []
        self.rec_lab: list[np.ndarray] = []
        self._anchor_cache = [anch[0], anch[1]]

    # ------------------------------------------------------------------ geometry
    def anchors(self, team: int) -> np.ndarray:
        base = self.base_rel[team * 11:(team + 1) * 11]
        b = to_rel(self.ball, team)
        if self.possession == team:
            x = base[:, 0] + MOVE['anchor_shift'] * (b[0] - 52.5) + 8.0
            y = 34 + (base[:, 1] - 34) * 1.08 + 0.2 * (b[1] - 34)
        else:
            x = base[:, 0] * 0.9 + MOVE['anchor_shift'] * (b[0] - 52.5) + 1.0
            y = 34 + (base[:, 1] - 34) * 0.78 + 0.33 * (b[1] - 34)
        x[0] = 4.0 + 0.06 * max(b[0] - 20.0, 0.0)
        y[0] = 34 + 0.12 * (b[1] - 34)
        x = np.clip(x, 2.0, 100.0)
        y = np.clip(y, 2.0, 66.0)
        rel = np.stack([x, y], axis=1)
        if self.possession == team:
            # attackers stay roughly level with the second-last opponent
            opp = to_rel(self.pos[outfield(1 - team)], team)[:, 0] if hasattr(self, "pos") else None
            if opp is not None:
                line = np.sort(opp)[-1] if len(opp) else 100.0  # deepest outfield defender
                rel[1:, 0] = np.minimum(rel[1:, 0], max(line - 0.5, 52.5))
        return from_rel(rel, team)

    def pos_delayed(self, i: int, frames: int) -> tuple[np.ndarray, np.ndarray]:
        f = len(self.rec_pos) - frames
        if f < 0 or not self.rec_pos:
            return self.pos[i].copy(), self.vel[i].copy()
        return self.rec_pos[f][i], self.rec_vel[f][i]

    # ------------------------------------------------------------------ control API (scripts)
    def set_override(self, i: int, fn: Override):
        self.ov[i] = fn

    def clear_override(self, i: int):
        self.ov.pop(i, None)

    def force_assign(self, d: int, a: int, lock: bool = True):
        for o in outfield(team_of(d)):
            if self.assign[o] == a and o != d and o not in self.locked:
                self.assign[o] = ZONE
        self.assign[d] = a
        self.pending.pop(d, None)
        if lock:
            self.locked.add(d)

    def release(self, d: int, zonal_lock: bool = True):
        self.assign[d] = ZONE
        self.pending.pop(d, None)
        self.locked.discard(d)
        if zonal_lock:
            self.zonal_locked.add(d)

    def marker_of(self, a: int) -> Optional[int]:
        ds = [d for d in outfield(1 - team_of(a)) if self.assign[d] == a]
        return ds[0] if ds else None

    def t(self) -> float:
        return self.frame * DT

    def _event(self, etype: str, player: int, frm, to, **kw):
        ev = {"t": round(self.t(), 3), "frame": self.frame, "type": etype, "player": PLAYER_IDS[player],
              "from": [round(float(frm[0]), 2), round(float(frm[1]), 2)],
              "to": [round(float(to[0]), 2), round(float(to[1]), 2)]}
        ev.update(kw)
        self.events.append(ev)
        return ev

    def pass_to(self, receiver: int, point: Optional[np.ndarray] = None, speed: Optional[float] = None,
                lead: float = 0.0):
        c = self.carrier
        assert c is not None and receiver != c
        start = self.ball.copy()
        speed = speed or self.rng.uniform(13.0, 17.0)
        if point is None:
            d = np.linalg.norm(self.pos[receiver] - start)
            ft = d / speed
            point = self.pos[receiver] + self.vel[receiver] * ft
            if lead > 0:
                point = point + unit(self.vel[receiver]) * lead
        point = np.clip(point, [1.0, 1.0], [L - 1.0, W - 1.0])
        d = np.linalg.norm(point - start)
        nf = max(int(round(d / speed * FPS)), 5)
        self.flight = Flight(start, point, self.frame, self.frame + nf, "pass", c, receiver)
        self._event("pass", c, start, point, receiver=PLAYER_IDS[receiver])
        self.carrier = None

    def intercepted_pass(self, intended: int, interceptor: int, point: np.ndarray, speed: float):
        c = self.carrier
        start = self.ball.copy()
        d = np.linalg.norm(point - start)
        nf = max(int(round(d / speed * FPS)), 5)
        intended_pt = self.pos[intended]
        self.flight = Flight(start, point.copy(), self.frame, self.frame + nf, "intercept", c, interceptor)
        self._event("pass", c, start, intended_pt, receiver=PLAYER_IDS[intended])
        self.carrier = None

    def shoot(self, shooter: int, goal_prob: float = 0.35):
        team = team_of(shooter)
        start = self.ball.copy()
        g = opp_goal(team)
        on_target = self.rng.random() < 0.8
        y = g[1] + self.rng.uniform(-3.0, 3.0) * (1.0 if on_target else 1.8)
        end = np.array([g[0], y])
        is_goal = on_target and self.rng.random() < goal_prob / 0.8
        nf = max(int(round(np.linalg.norm(end - start) / 26.0 * FPS)), 3)
        self.flight = Flight(start, end, self.frame, self.frame + nf, "shot", shooter, None, is_goal)
        self._event("shot", shooter, start, end, outcome="goal" if is_goal else ("saved" if on_target else "off_target"))
        self.carrier = None

    def flip_possession(self, new_carrier: int):
        self.possession = team_of(new_carrier)
        self.carrier = new_carrier
        self.assign[:] = ZONE
        self.pending.clear()
        self.locked.clear()
        self.zonal_locked.clear()
        self.unmarkable.clear()
        self.hold_until = self.frame + int(self.rng.uniform(0.6, 1.6) * FPS)

    # ------------------------------------------------------------------ marking (generator side)
    def update_assignments(self):
        D = 1 - self.possession
        defs = outfield(D)
        atts = outfield(self.possession)
        anch = self._anchor_cache[D]
        # releases and hand-offs
        for d in defs:
            a = self.assign[d]
            if a < 0 or d in self.locked:
                continue
            da = np.linalg.norm(self.pos[a] - anch[d - D * 11])
            if da > self.leash[d]:
                cand = self._free_owner(a, D, anch, exclude=d)
                if cand is not None or da > 1.6 * self.leash[d]:
                    self.assign[d] = ZONE
        # random swaps between neighbouring markers
        marking = [d for d in defs if self.assign[d] >= 0 and d not in self.locked]
        for i in range(len(marking)):
            for j in range(i + 1, len(marking)):
                d1, d2 = marking[i], marking[j]
                a1, a2 = self.assign[d1], self.assign[d2]
                if np.linalg.norm(self.pos[a1] - self.pos[a2]) < 8.0 and self.rng.random() < 0.012:
                    self.assign[d1], self.assign[d2] = a2, a1
        # pick-ups
        marked = set(self.assign[defs][self.assign[defs] >= 0].tolist())
        order = sorted(atts, key=lambda a: np.linalg.norm(self.pos[a] - own_goal(D)))
        for a in order:
            if a in marked or a in self.unmarkable:
                continue
            d = self._free_owner(a, D, anch)
            if d is None:
                continue
            pend = self.pending.get(d)
            if pend and pend[0] == a:
                left = pend[1] - 5
                if left <= 0:
                    self.assign[d] = a
                    self.pending.pop(d)
                    marked.add(a)
                else:
                    self.pending[d] = (a, left)
            else:
                self.pending[d] = (a, int(self.rng.uniform(0.2, 0.8) * FPS))

    def _free_owner(self, a: int, D: int, anch: np.ndarray, exclude: Optional[int] = None) -> Optional[int]:
        dists = np.linalg.norm(anch - self.pos[a], axis=1)
        for k in np.argsort(dists):
            d = D * 11 + int(k)
            if k == 0 or d == exclude:
                continue
            if dists[k] > self.pick_r[d]:
                return None
            if d in self.zonal_only or d in self.zonal_locked or d in self.locked or self.assign[d] >= 0:
                continue
            return d
        return None

    # ------------------------------------------------------------------ simulation step
    def step(self):
        if self.script is not None:
            self.script.step(self)
        if self.director is not None and self.director_enabled and not self.dead:
            self.director.step(self)
        self._anchor_cache = [self.anchors(0), self.anchors(1)]
        if self.frame % 5 == 0 and not self.dead:
            self.update_assignments()

        A, D = self.possession, 1 - self.possession
        tgt = np.zeros((22, 2))
        vmax = self.vmax.copy()
        amax = self.amax.copy()
        ff = np.zeros((22, 2))
        kp = np.full(22, MOVE['kp_off'])

        tau, sig = MOVE['ou_tau'], 1.0
        self.ou += -self.ou / tau * DT + np.sqrt(2 * DT / tau) * self.rng.normal(0, 1, size=(22, 2))
        spread = np.where(np.array([team_of(i) for i in range(22)]) == A, MOVE['ou_att'], MOVE['ou_def'])
        tgt[:11] = self._anchor_cache[0]
        tgt[11:] = self._anchor_cache[1]
        tgt += self.ou * spread[:, None] * sig
        for k in (keeper(0), keeper(1)):
            tgt[k] = self._anchor_cache[team_of(k)][0]

        if not self.dead:
            for d in outfield(D):
                a = self.assign[d]
                if a < 0:
                    continue
                pa, va = self.pos_delayed(a, int(self.delay_f[d]))
                g = own_goal(D)
                tgt[d] = pa + unit(g - pa) * self.goal_side[d]
                ff[d] = va
                kp[d] = MOVE['kp_mark']
            if self.carrier is not None:
                c = self.carrier
                g = opp_goal(team_of(c))
                if self.rng.random() < 0.02 or not self.dribble_dir.any():
                    self.dribble_dir = unit(unit(g - self.pos[c]) + self.rng.normal(0, 0.8, 2))
                tgt[c] = self.pos[c] + self.dribble_dir * 3.0
                vmax[c] = 3.0
                kp[c] = 1.0
            if self.flight is not None and self.flight.receiver is not None:
                r = self.flight.receiver
                tgt[r] = self.flight.end
                kp[r] = 2.5
        for i, fn in list(self.ov.items()):
            tgt[i], vmax[i], amax[i], ff[i], kp[i] = fn(self)

        desired = kp[:, None] * (tgt - self.pos) + ff
        sp = np.linalg.norm(desired, axis=1)
        desired *= np.minimum(1.0, vmax / np.maximum(sp, 1e-9))[:, None]
        acc = (desired - self.vel) / DT
        an = np.linalg.norm(acc, axis=1)
        acc *= np.minimum(1.0, amax / np.maximum(an, 1e-9))[:, None]
        self.vel += acc * DT
        self.pos += self.vel * DT
        self.pos[:, 0] = np.clip(self.pos[:, 0], -1.0, L + 1.0)
        self.pos[:, 1] = np.clip(self.pos[:, 1], -1.0, W + 1.0)

        self._update_ball()
        self._record()
        self.frame += 1

    def _update_ball(self):
        f = self.flight
        if f is not None:
            k = min(1.0, (self.frame - f.f0) / max(f.f1 - f.f0, 1))
            self.ball = f.start + (f.end - f.start) * k
            if self.frame >= f.f1:
                self.flight = None
                if f.kind == "pass":
                    self.carrier = f.receiver
                    self.hold_until = self.frame + int(self.rng.uniform(0.8, 2.5) * FPS)
                elif f.kind == "intercept":
                    self._event("interception", f.receiver, self.ball, self.ball)
                    self.flip_possession(f.receiver)
                elif f.kind == "shot":
                    if f.goal:
                        self._event("goal", f.passer, self.ball, self.ball)
                    self.dead = True
                    self.carrier = None
                    self.assign[:] = ZONE
        elif self.carrier is not None:
            c = self.carrier
            sp = np.linalg.norm(self.vel[c])
            d = self.vel[c] / sp if sp > 0.5 else np.array([attack_sign(team_of(c)), 0.0])
            self.ball = self.pos[c] + 0.5 * d

    def _record(self):
        self.rec_pos.append(self.pos.copy())
        self.rec_vel.append(self.vel.copy())
        self.rec_ball.append(self.ball.copy())
        lab = np.full(22, NOT_DEFENDING, dtype=np.int16)
        if not self.dead and not (self.flight is not None and self.flight.kind == "shot"):
            D = 1 - self.possession
            for d in outfield(D):
                lab[d] = self.assign[d]
        self.rec_lab.append(lab)


class Director:
    """Filler possession: the team in possession circulates the ball, optionally steering it
    toward a target depth, and loses it with a tackle at a scheduled frame."""

    def __init__(self, rng: np.random.Generator, steer_x: Optional[float] = None,
                 turnover_frame: Optional[int] = None):
        self.rng = rng
        self.steer_x = steer_x
        self.turnover_frame = turnover_frame
        self.tackler: Optional[int] = None
        self.tackle_deadline = 0

    def cancel_tackle(self, sim: "Sim"):
        if self.tackler is not None:
            sim.clear_override(self.tackler)
        self.tackler = None
        self.turnover_frame = None

    def step(self, sim: Sim):
        if sim.flight is not None or sim.carrier is None:
            return
        c = sim.carrier
        T = team_of(c)
        if self.turnover_frame is not None and sim.frame >= self.turnover_frame:
            self._tackle(sim, c)
            return
        if sim.frame < sim.hold_until:
            return
        mates = [m for m in outfield(T) if m != c]
        P = sim.pos[mates]
        d = np.linalg.norm(P - sim.pos[c], axis=1)
        opp = sim.pos[outfield(1 - T)]
        openness = np.min(np.linalg.norm(P[:, None] - opp[None], axis=2), axis=1).clip(0, 8)
        seg = P - sim.pos[c]
        rel_o = opp[None] - sim.pos[c][None, None]
        tproj = np.clip(np.einsum("mk,mok->mo", seg, rel_o) / np.maximum(np.einsum("mk,mk->m", seg, seg), 1e-6)[:, None], 0, 1)
        closest = sim.pos[c][None, None] + tproj[..., None] * seg[:, None]
        lane = np.min(np.linalg.norm(closest - opp[None], axis=2), axis=1).clip(0, 6)
        rx = to_rel(P, T)[:, 0]
        cx = to_rel(sim.pos[c], T)[0]
        score = 0.35 * openness + 0.45 * lane - 0.04 * d + 0.03 * (rx - cx)
        if self.steer_x is not None:
            score -= 0.10 * np.abs(rx - self.steer_x)
        score[(d < 6) | (d > 38)] = -1e9
        if np.all(score < -1e8):
            sim.hold_until = sim.frame + 5
            return
        p = np.exp((score - score.max()) / 0.5)
        p /= p.sum()
        r = mates[int(self.rng.choice(len(mates), p=p))]
        sim.pass_to(r)

    def _tackle(self, sim: Sim, c: int):
        if self.tackler is None or team_of(self.tackler) == team_of(c) or sim.frame > self.tackle_deadline:
            if self.tackler is not None:
                sim.clear_override(self.tackler)
            opp = outfield(1 - team_of(c))
            self.tackler = int(opp[np.argmin(np.linalg.norm(sim.pos[opp] - sim.pos[c], axis=1))])
            self.tackle_deadline = sim.frame + 6 * FPS
        k = self.tackler
        # (re)asserted every frame: a script finishing elsewhere may have cleared overrides
        sim.set_override(k, lambda s: (s.pos[s.carrier] if s.carrier is not None else s.ball, 7.2, 5.0, np.zeros(2), 3.0))
        if np.linalg.norm(sim.pos[k] - sim.ball) < 1.2:
            sim._event("tackle", k, sim.ball, sim.ball)
            sim.clear_override(k)
            sim.flip_possession(k)
            self.tackler = None
            self.turnover_frame = None


def run_to(target: np.ndarray, vmax: float, amax: float = 5.0, kp: float = 2.5) -> Override:
    target = np.asarray(target, dtype=float)
    return lambda sim: (target, vmax, amax, np.zeros(2), kp)


def hold_ball(i: int) -> Override:
    """Carrier shields the ball, barely moving."""
    return lambda sim: (sim.pos[i] + sim.vel[i] * 0.2, 1.2, 3.0, np.zeros(2), 1.0)
