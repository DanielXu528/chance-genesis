"""Synthetic 25 Hz tracking generator.

Movement model (what a viewer sees)
  * Each team keeps a block shaped by its formation. The block follows a reference point
    that lags the ball by about 1.5 s (and leads it to where a pass is going), and the
    switch between the attacking and defending shape fades over about 2 s, so shapes shift
    smoothly instead of jumping with every pass.
  * Each player drifts slowly around his slot (two low-frequency sine waves) and has a
    small dead zone, so players stand still between shifts instead of jittering.
  * Speed depends on how far a player is from where he wants to be: walk, jog, run, sprint.

Defending is zonal with ball-side pressure
  * the defender nearest the ball presses it, the next one covers behind him;
  * everyone else holds his zone and only leans toward the attacker in it, lightly in
    midfield and tightly near his own box or when that attacker runs at goal (tracking).
Attacking
  * the carrier dribbles into space ahead of him, away from the nearest opponent;
  * the two nearest team-mates come short to offer angles, the rest look for pockets of
    space near their slot, and forwards now and then run in behind the line.
The ball
  * passes travel and slow down; if the receiver is not there when the ball arrives it
    rolls on as a loose ball until someone controls it; an opponent close to the line of a
    pass can cut it out. Nothing is teleported.

Ground truth (labels) is the generator's internal defending state: a defender is
"marking" attacker a when he presses a or leans on a with weight >= 0.6 (tight marking or
tracking a run); otherwise he is holding a zone. The detector (analysis/marking.py) never
sees these mechanics: it solves a Hungarian assignment on positional costs.
"""
from __future__ import annotations

from dataclasses import dataclass
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
MID_ROLES = {"DM", "RCM", "LCM", "CM"}
FWD_ROLES = {"ST", "RST", "LST", "RW", "LW", "RM", "LM"}

MOVE = {
    "shape_tau": 2.5,      # s, lag of the block's reference point behind the ball
    "poss_tau": 2.0,       # s, fade between attacking and defending shape
    "shift": 0.35,         # how far the block shifts per metre of ball movement
    "dead_zone": 1.0,      # m, off-ball players stand still this close to their target
    "gain": 0.45,          # 1/s, off-ball approach gain
    "urgent_gain": 2.4,    # 1/s, players going for the ball / tracking a man
    "press_r": 18.0,       # m, the nearest defender presses the ball inside this radius
    "contain_m": 3.0,      # m, pressing distance (goal side) unless the team wants the ball back
    "lean_mid": 0.30,      # weight toward the attacker in one's zone, midfield
    "lean_box": 0.6,       # ... near one's own box (close marking, 2-5 m off)
    "box_m": 26.0,         # m from own goal line counted as "near the box"
    "zone_r": 12.0,        # m, an attacker within this of a defender's slot is in his zone
    "run_rate": 0.25,      # runs in behind per second per forward (when the ball is high enough)
    "control_r": 1.3,      # m, a player within this of the ball can control it
    "friction": 3.2,       # m/s^2, rolling deceleration of a loose ball
}


HOLD = (1.2, 3.2)      # s on the ball before looking to pass (calibrated: about 10 passes a minute)
RUN_SPEED = 1.0        # share of top speed on a run in behind


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


@dataclass
class Flight:
    start: np.ndarray
    end: np.ndarray
    f0: int
    f1: int
    kind: str                     # pass | shot
    passer: int
    receiver: Optional[int] = None
    goal: bool = False
    safe: bool = False            # scripted pass: nobody else cuts it out
    interceptor: Optional[int] = None   # scripted interception: this player will cut it out
    lofted: bool = False
    saved: bool = False           # shot: the goalkeeper gets to it


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
        is_gk = np.array([r == "GK" for r in self.role])
        # Per-player physical and behavioural parameters (hidden from the detector).
        self.delay_f = rng.integers(int(0.3 * FPS), int(0.6 * FPS) + 1, size=22)
        self.goal_side = rng.uniform(1.0, 2.2, size=22)
        self.vmax = np.where(is_gk, 6.5, rng.uniform(7.8, 9.2, size=22))   # top speeds 28-33 km/h
        self.amax = rng.uniform(3.6, 4.8, size=22)
        self.wob_amp = rng.uniform(0.8, 2.2, size=(22, 2))
        self.wob_freq = rng.uniform(0.03, 0.09, size=(22, 2)) * 2 * np.pi
        self.wob_phase = rng.uniform(0, 2 * np.pi, size=(22, 2))
        self.zonal_only: set[int] = set()
        for team in (0, 1):
            mids = [i for i in outfield(team) if self.role[i] in MID_ROLES]
            k = rng.integers(0, 2)
            self.zonal_only.update(rng.choice(mids, size=k, replace=False).tolist() if k else [])

        self.possession = possession
        self.poss_w = np.array([1.0, 0.0]) if possession == 0 else np.array([0.0, 1.0])
        self.frame = 0
        self.ball = from_rel(np.array([ball_rel_x, 34.0]), possession)
        self.shape_ref = self.ball.copy()
        self.ball_v = np.zeros(2)
        self.pos = np.zeros((22, 2))
        anch = self.anchors(0), self.anchors(1)
        self.pos = np.vstack(anch) + rng.normal(0, 1.0, size=(22, 2))
        self.vel = np.zeros((22, 2))
        self.carrier = carrier
        if carrier is not None:
            self.ball = self.pos[carrier].copy()
        self.flight: Optional[Flight] = None
        self.loose = False
        self.loose_since = 0
        self.dead = False
        self.last_touch_team = possession
        # defending state (ground truth comes from here)
        self.assign = np.full(22, ZONE, dtype=int)     # label: attacker idx or ZONE
        self.mark = np.full(22, -1, dtype=int)         # attacker a defender leans on
        self.lean = np.zeros(22)                       # weight of that lean
        self.mode = ["zone"] * 22                      # zone | lean | press | cover
        self.pending: dict[int, tuple[int, int]] = {}
        self.locked: set[int] = set()
        self.zonal_locked: set[int] = set()
        self.unmarkable: set[int] = set()
        self.ov: dict[int, Override] = {}
        # attacking state
        self.run_until = np.zeros(22, dtype=int)
        self.run_target = np.zeros((22, 2))
        self.space_off = np.zeros((22, 2))
        self.space_goal = np.zeros((22, 2))
        self.carry_speed = 4.0
        self.carry_side = 0.0
        self.hold_until = 0
        self.want_ball = False          # defending team is going in for a tackle
        self.runs_enabled = True
        self.events: list[dict] = []
        self.director = None
        self.script = None
        self.director_enabled = True
        self._ball_off = np.array([0.5, 0.0])
        # recordings
        self.rec_pos: list[np.ndarray] = []
        self.rec_vel: list[np.ndarray] = []
        self.rec_ball: list[np.ndarray] = []
        self.rec_lab: list[np.ndarray] = []
        self._anchor_cache = [anch[0], anch[1]]

    # ------------------------------------------------------------------ geometry
    def anchors(self, team: int) -> np.ndarray:
        base = self.base_rel[team * 11:(team + 1) * 11]
        b = to_rel(self.shape_ref, team)
        s = MOVE["shift"]
        att_x = base[:, 0] + s * (b[0] - 52.5) + 8.0
        att_y = 34 + (base[:, 1] - 34) * 1.08 + 0.2 * (b[1] - 34)
        def_x = base[:, 0] * 0.9 + s * (b[0] - 52.5) + 1.0
        def_y = 34 + (base[:, 1] - 34) * 0.78 + 0.33 * (b[1] - 34)
        w = self.poss_w[team]
        x = w * att_x + (1 - w) * def_x
        y = w * att_y + (1 - w) * def_y
        t = self.frame * DT
        sl = slice(team * 11, (team + 1) * 11)
        wob = self.wob_amp[sl] * np.sin(self.wob_freq[sl] * t + self.wob_phase[sl])
        x = x + wob[:, 0]
        y = y + wob[:, 1]
        x[0] = 4.0 + 0.06 * max(b[0] - 20.0, 0.0)
        y[0] = 34 + 0.12 * (b[1] - 34)
        x = np.clip(x, 2.0, 100.0)
        y = np.clip(y, 2.0, 66.0)
        rel = np.stack([x, y], axis=1)
        if w > 0.5 and self.pos.any():
            # attackers stay onside: forwards on the last defender's shoulder, others deeper
            line = float(np.max(to_rel(self.pos[outfield(1 - team)], team)[:, 0]))
            gap = np.array([0.0 if self.role[team * 11 + k] == "GK" else
                            0.6 if self.role[team * 11 + k] in FWD_ROLES else
                            5.0 if self.role[team * 11 + k] in MID_ROLES else 12.0 for k in range(11)])
            rel[1:, 0] = np.minimum(rel[1:, 0], np.maximum(line - gap[1:], 52.5 - gap[1:] * 0.5))
        return from_rel(rel, team)

    def pos_delayed(self, i: int, frames: int) -> tuple[np.ndarray, np.ndarray]:
        f = len(self.rec_pos) - frames
        if f < 0 or not self.rec_pos:
            return self.pos[i].copy(), self.vel[i].copy()
        return self.rec_pos[f][i], self.rec_vel[f][i]

    def t(self) -> float:
        return self.frame * DT

    # ------------------------------------------------------------------ control API (scripts)
    def set_override(self, i: int, fn: Override):
        self.ov[i] = fn

    def clear_override(self, i: int):
        self.ov.pop(i, None)

    def force_assign(self, d: int, a: int, lock: bool = True):
        """Script: defender d tracks attacker a tightly (ground truth: marking a)."""
        for o in outfield(team_of(d)):
            if self.mark[o] == a and o != d and o not in self.locked:
                self.mark[o], self.lean[o], self.assign[o], self.mode[o] = -1, 0.0, ZONE, "zone"
        self.mark[d], self.lean[d], self.assign[d], self.mode[d] = a, 1.0, a, "lean"
        self.pending.pop(d, None)
        if lock:
            self.locked.add(d)

    def release(self, d: int, zonal_lock: bool = True):
        """Script: defender d holds his zone (ground truth: zone)."""
        self.mark[d], self.lean[d], self.assign[d], self.mode[d] = -1, 0.0, ZONE, "zone"
        self.pending.pop(d, None)
        self.locked.discard(d)
        if zonal_lock:
            self.zonal_locked.add(d)

    def marker_of(self, a: int) -> Optional[int]:
        ds = [d for d in outfield(1 - team_of(a)) if self.assign[d] == a]
        return ds[0] if ds else None

    def _event(self, etype: str, player: int, frm, to, **kw):
        ev = {"t": round(self.t(), 3), "frame": self.frame, "type": etype, "player": PLAYER_IDS[player],
              "from": [round(float(frm[0]), 2), round(float(frm[1]), 2)],
              "to": [round(float(to[0]), 2), round(float(to[1]), 2)]}
        ev.update(kw)
        self.events.append(ev)
        return ev

    def pass_to(self, receiver: int, point: Optional[np.ndarray] = None, speed: Optional[float] = None,
                lead: float = 0.0, safe: bool = False, interceptor: Optional[int] = None):
        """Release a pass toward `point` (default: where the receiver will be when it arrives)."""
        c = self.carrier
        assert c is not None and receiver != c
        start = self.ball.copy()
        if point is None:
            # aim where the receiver will be when the ball gets there (two refinements)
            point = self.pos[receiver].copy()
            for _ in range(2):
                d0 = float(np.linalg.norm(point - start))
                sp = speed or float(np.clip(9.0 + 0.3 * d0, 11.0, 19.0))
                ft = d0 / sp * 1.12
                point = self.pos[receiver] + self.vel[receiver] * ft * 0.9
            if lead > 0:
                point = point + unit(self.vel[receiver]) * lead
        point = np.clip(point, [1.0, 1.0], [L - 1.0, W - 1.0])
        d = float(np.linalg.norm(point - start))
        speed = speed or float(np.clip(9.0 + 0.3 * d, 11.0, 19.0))
        lofted = d > 32.0 and interceptor is None
        # the ball decelerates along the way: flight time a little longer than d / speed
        nf = max(int(round(d / speed * 1.12 * FPS)), 5)
        self.flight = Flight(start, point, self.frame, self.frame + nf, "pass", c, receiver,
                             safe=safe or lofted, interceptor=interceptor, lofted=lofted)
        self._event("pass", c, start, point, receiver=PLAYER_IDS[receiver])
        self.carrier = None
        self.last_touch_team = team_of(c)
        self.ov.pop(receiver, None)          # whoever the ball is played to goes for it
        self.run_until[receiver] = 0

    def intercepted_pass(self, intended: int, interceptor: int, point: np.ndarray, speed: float):
        """A pass meant for `intended` that `interceptor`, standing on its line at `point`, cuts out.
        The ball travels toward the intended receiver; if the interceptor is not quite there
        when it passes, it rolls on as a loose ball."""
        c = self.carrier
        start = self.ball.copy()
        aim = self.pos[intended].copy()
        d = float(np.linalg.norm(point - start))
        nf = max(int(round(d / speed * 1.12 * FPS)), 5)
        self.flight = Flight(start, np.asarray(point, float).copy(), self.frame, self.frame + nf, "pass", c, intended,
                             interceptor=interceptor)
        self._event("pass", c, start, aim, receiver=PLAYER_IDS[intended])
        self.carrier = None
        self.last_touch_team = team_of(c)
        return d

    def shoot(self, shooter: int, goal_prob: float = 0.35):
        team = team_of(shooter)
        start = self.ball.copy()
        g = opp_goal(team)
        out = 1.0 if g[0] > L / 2 else -1.0                  # +x is "behind the goal line" for this goal
        on_target = self.rng.random() < 0.8
        is_goal = on_target and self.rng.random() < goal_prob / 0.8
        saved = on_target and not is_goal
        if is_goal:
            end = np.array([g[0] + out * 1.2, g[1] + self.rng.uniform(-3.2, 3.2)])          # into the net
        elif saved:
            # struck within the keeper's reach: he gets across and holds it
            gk = keeper(1 - team)
            end = self.pos[gk] + np.array([0.0, self.rng.uniform(-1.3, 1.3)])
        else:
            side = 1.0 if self.rng.random() < 0.5 else -1.0
            end = np.array([g[0] + out * 3.0, g[1] + side * self.rng.uniform(4.5, 9.0)])   # wide, out of play
        nf = max(int(round(np.linalg.norm(end - start) / 26.0 * FPS)), 3)
        self.flight = Flight(start, end, self.frame, self.frame + nf, "shot", shooter, None, is_goal, safe=True,
                             saved=saved)
        self._event("shot", shooter, start, end, outcome="goal" if is_goal else ("saved" if on_target else "off_target"))
        self.carrier = None

    def flip_possession(self, new_carrier: int):
        self.possession = team_of(new_carrier)
        self.carrier = new_carrier
        self.loose = False
        self.flight = None
        self.last_touch_team = self.possession
        self.mark[:] = -1
        self.lean[:] = 0.0
        self.assign[:] = ZONE
        self.mode = ["zone"] * 22
        self.pending.clear()
        self.locked.clear()
        self.zonal_locked.clear()
        self.unmarkable.clear()
        self.run_until[:] = 0
        self.want_ball = False
        self.hold_until = self.frame + int(self.rng.uniform(0.6, 1.6) * FPS)

    def _claim(self, i: int):
        """Player i takes control of the ball (after a pass, a loose ball or an interception)."""
        off = self.ball - self.pos[i]
        n = np.linalg.norm(off)
        self._ball_off = off if n <= 1.5 else off / n * 1.5      # no jump: the ball settles into his stride
        if team_of(i) != self.possession:
            self._event("interception", i, self.ball, self.ball)
            self.flip_possession(i)
        else:
            self.carrier = i
            self.flight = None
            self.loose = False
            self.hold_until = self.frame + int(self.rng.uniform(*HOLD) * FPS)
            self.carry_side = float(self.rng.uniform(-1, 1))
            self.carry_speed = float(self.rng.uniform(3.2, 5.2))

    # ------------------------------------------------------------------ defending (generator side)
    def update_defense(self):
        D = 1 - self.possession
        A = self.possession
        defs = [int(d) for d in outfield(D)]
        anch = self._anchor_cache[D]
        free = [d for d in defs if d not in self.locked]
        # ball-side pressure: nearest presses, next covers
        ref = self.pos[self.carrier] if self.carrier is not None else (
            self.flight.end if self.flight is not None else self.ball)
        cand = sorted([d for d in free if d not in self.zonal_locked],
                      key=lambda d: np.linalg.norm(self.pos[d] - ref))
        presser = cand[0] if cand and np.linalg.norm(self.pos[cand[0]] - ref) < MOVE["press_r"] else None
        cover = cand[1] if presser is not None and len(cand) > 1 else None
        for d in free:
            self.mode[d] = "zone"
        if presser is not None:
            self.mode[presser] = "press"
            self.mark[presser] = self.carrier if self.carrier is not None else -1
            self.lean[presser] = 1.0
            self.assign[presser] = self.carrier if self.carrier is not None and team_of(self.carrier) == A else ZONE
        if cover is not None:
            self.mode[cover] = "cover"
            self.mark[cover], self.lean[cover], self.assign[cover] = -1, 0.0, ZONE
        # everyone else: hold the zone, lean on the most dangerous attacker inside it
        atts = [int(a) for a in outfield(A) if a not in self.unmarkable]
        if not atts:
            return
        P_att = self.pos[atts]
        own = own_goal(D)
        dist_anchor = np.linalg.norm(P_att[:, None, :] - anch[None, 1:, :], axis=2)   # (atts, 10)
        owner = np.argmin(dist_anchor, axis=1) + D * 11 + 1
        for d in free:
            if d in (presser, cover):
                continue
            if d in self.zonal_only or d in self.zonal_locked:
                self.mark[d], self.lean[d], self.assign[d] = -1, 0.0, ZONE
                continue
            k = d - D * 11 - 1
            mine = [j for j in range(len(atts)) if owner[j] == d and dist_anchor[j, k] < MOVE["zone_r"]]
            if not mine:
                self.mark[d], self.lean[d], self.assign[d] = -1, 0.0, ZONE
                self.pending.pop(d, None)
                continue
            j = min(mine, key=lambda j: np.linalg.norm(P_att[j] - own))
            a = atts[j]
            depth = to_rel(self.pos[a], D)[0]          # attacker's distance from D's goal line
            v_goal = float(self.vel[a] @ unit(own - self.pos[a]))
            if depth < MOVE["box_m"]:
                w = MOVE["lean_box"]
            elif v_goal > 3.5 and depth < 52.5:
                w = 1.0                                # tracking a run at goal
            else:
                w = MOVE["lean_mid"]
            if self.mark[d] != a:
                pend = self.pending.get(d)
                if pend and pend[0] == a:
                    left = pend[1] - 5
                    if left > 0:
                        self.pending[d] = (a, left)
                        continue
                    self.pending.pop(d)
                else:
                    self.pending[d] = (a, int(self.rng.uniform(0.2, 0.6) * FPS))
                    continue
            self.mark[d] = a
            self.lean[d] = w
            self.assign[d] = a if w >= 0.6 else ZONE

    # ------------------------------------------------------------------ attacking off the ball
    def update_attack(self):
        A = self.possession
        if self.carrier is None or team_of(self.carrier) != A:
            return
        opp = self.pos[outfield(1 - A)]
        ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        ring = np.stack([np.cos(ang), np.sin(ang)], axis=1) * 4.0
        for i in outfield(A):
            if i == self.carrier or self.run_until[i] > self.frame:
                continue
            base = self._anchor_cache[A][i - A * 11]
            cands = np.vstack([[0, 0], ring]) + base
            cands[:, 0] = np.clip(cands[:, 0], 2, L - 2)
            cands[:, 1] = np.clip(cands[:, 1], 2, W - 2)
            clear = np.min(np.linalg.norm(cands[:, None, :] - opp[None], axis=2), axis=1)
            score = np.minimum(clear, 8.0) - 0.15 * np.linalg.norm(cands - base, axis=1)
            self.space_goal[i] = cands[int(np.argmax(score))] - base
        self.space_off += (self.space_goal - self.space_off) * 0.25
        # occasional runs in behind the line by forwards
        if not self.runs_enabled:
            return
        bx = to_rel(self.ball, A)[0]
        if bx < 45:
            return
        line = float(np.max(to_rel(opp, A)[:, 0]))
        for i in outfield(A):
            if self.role[i] not in FWD_ROLES or i == self.carrier or self.run_until[i] > self.frame or i in self.ov:
                continue
            if self.rng.random() < MOVE["run_rate"] * 5 * DT:
                r = to_rel(self.pos[i], A)
                tgt = np.array([min(line + self.rng.uniform(3, 9), 97.0), np.clip(r[1] + self.rng.uniform(-8, 8), 6, 62)])
                self.run_target[i] = from_rel(tgt, A)
                self.run_until[i] = self.frame + int(self.rng.uniform(1.8, 3.0) * FPS)

    def carry_target(self, c: int) -> np.ndarray:
        A = team_of(c)
        r = to_rel(self.pos[c], A)
        if r[0] > 84:
            aim = np.array([min(r[0] + 4, 92.0), r[1] + (34 - r[1]) * 0.3])
        else:
            aim = np.array([r[0] + 12.0, r[1] + self.carry_side * 6.0 + (34 - r[1]) * 0.1])
        aim = from_rel(aim, A)
        opp = self.pos[outfield(1 - A)]
        d = np.linalg.norm(opp - self.pos[c], axis=1)
        q = int(np.argmin(d))
        if d[q] < 5.0:
            aim = aim + unit(self.pos[c] - opp[q]) * (5.0 - d[q]) * 1.5
        return np.clip(aim, [1.5, 1.5], [L - 1.5, W - 1.5])

    # ------------------------------------------------------------------ simulation step
    def step(self):
        if self.script is not None:
            self.script.step(self)
        if self.director is not None and self.director_enabled and not self.dead:
            self.director.step(self)
        ref = self.flight.end if self.flight is not None and self.flight.kind == "pass" else self.ball
        self.shape_ref += (ref - self.shape_ref) * min(1.0, DT / MOVE["shape_tau"])
        if not self.dead:
            tgt_w = np.array([1.0, 0.0]) if self.possession == 0 else np.array([0.0, 1.0])
            self.poss_w += (tgt_w - self.poss_w) * min(1.0, DT / MOVE["poss_tau"])
        self._anchor_cache = [self.anchors(0), self.anchors(1)]
        if self.frame % 5 == 0 and not self.dead:
            self.update_defense()
            self.update_attack()
        tgt, vcap, amax, ff, gain, urgent = self._targets()
        # players going for something use the fast approach gain unless they set their own
        gain = np.where(urgent & (gain == MOVE["gain"]), MOVE["urgent_gain"], gain)
        for i, fn in list(self.ov.items()):
            tgt[i], vcap[i], amax[i], ff[i], gain[i] = fn(self)
            urgent[i] = True
        self._move(tgt, vcap, amax, ff, gain, urgent)
        self._update_ball()
        self._record()
        self.frame += 1

    def _targets(self):
        A, D = self.possession, 1 - self.possession
        tgt = np.vstack(self._anchor_cache).copy()
        tgt[outfield(A)] += self.space_off[outfield(A)]
        # team-mates drifting into each other spread out
        for k in (0, 1):
            sl = slice(11 * k, 11 * k + 11)
            diff = self.pos[sl, None, :] - self.pos[None, sl, :]
            dd = np.linalg.norm(diff, axis=-1) + np.eye(11) * 1e3
            push = np.clip(3.0 - dd, 0, None) / np.maximum(dd, 0.1)
            tgt[sl] += (diff * push[..., None]).sum(axis=1) * 0.6
        gap = np.linalg.norm(tgt - self.pos, axis=1)
        vcap = np.select([gap > 18, gap > 10, gap > 4], [self.vmax * 0.85, 5.0, 3.2], 1.8)
        amax = self.amax.copy()
        ff = np.zeros((22, 2))
        gain = np.full(22, MOVE["gain"])
        urgent = np.zeros(22, dtype=bool)
        self.minspd = np.zeros(22)         # players going for the ball keep running into it
        if self.dead:
            return tgt, vcap, amax, ff, gain, urgent
        # attackers
        for i in outfield(A):
            if self.run_until[i] > self.frame:
                tgt[i] = self.run_target[i]
                vcap[i] = self.vmax[i] * RUN_SPEED
                urgent[i] = True
        c = self.carrier
        if c is not None and team_of(c) == A:
            tgt[c] = self.carry_target(c)
            pressed = np.min(np.linalg.norm(self.pos[outfield(D)] - self.pos[c], axis=1)) < 4.0
            vcap[c] = 2.4 if pressed else self.carry_speed
            urgent[c] = True
            gain[c] = 1.2
            if self.role[c] != "GK":
                fwd = unit(opp_goal(A) - self.ball)
                mates = sorted([j for j in outfield(A) if j != c and self.run_until[j] <= self.frame],
                               key=lambda j: np.linalg.norm(self.pos[j] - self.ball))[:2]
                for j, ang in zip(mates, (0.95, -0.95)):
                    if np.linalg.norm(self.pos[j] - self.ball) > 28:
                        continue
                    cs, sn = np.cos(ang), np.sin(ang)
                    offer = self.ball + np.array([fwd[0] * cs - fwd[1] * sn, fwd[0] * sn + fwd[1] * cs]) * 11.0
                    tgt[j] = np.clip(offer, [2, 2], [L - 2, W - 2])
                    vcap[j] = max(vcap[j], 4.5)
        # defenders
        own = own_goal(D)
        for d in outfield(D):
            m = self.mode[d]
            if m == "press":
                bp = self.ball
                contain = 0.6 if self.want_ball else MOVE["contain_m"]
                tgt[d] = bp + unit(own - bp) * contain
                vcap[d] = self.vmax[d] * (0.9 if self.want_ball else 0.75)
                urgent[d] = True
                if self.carrier is not None:
                    ff[d] = self.vel[self.carrier] * 0.8
            elif m == "cover":
                bp = self.ball
                tgt[d] = 0.5 * tgt[d] + 0.5 * (bp + unit(own - bp) * 7.0)
                vcap[d] = max(vcap[d], 4.5)
            elif self.mark[d] >= 0 and self.lean[d] > 0:
                a = int(self.mark[d])
                pa, va = self.pos_delayed(a, int(self.delay_f[d]))
                mpt = pa + unit(own - pa) * self.goal_side[d]
                w = self.lean[d]
                tgt[d] = (1 - w) * tgt[d] + w * mpt
                if w >= 0.6:
                    ff[d] = va * w
                    urgent[d] = True
                    gain[d] = MOVE["urgent_gain"] * 0.8
                    vcap[d] = self.vmax[d]
        # ball in flight or loose: the intended receiver goes to it, the nearest opponent contests
        f = self.flight
        if f is not None and f.kind == "pass":
            if f.receiver is not None and f.interceptor is None:
                r = f.receiver
                # meet the ball: run to where it is going, then step into it
                tgt[r] = f.end if np.linalg.norm(self.pos[r] - f.end) > 1.0 else self.ball
                vcap[r] = self.vmax[r]
                urgent[r] = True
                self.minspd[r] = 3.0
            if f.interceptor is not None:
                tgt[f.interceptor] = self._closest_on_flight(f.interceptor)
                vcap[f.interceptor] = self.vmax[f.interceptor]
                urgent[f.interceptor] = True
        elif f is not None and f.kind == "shot" and f.saved:
            gk = keeper(1 - team_of(f.passer))
            tgt[gk] = f.end
            vcap[gk] = self.vmax[gk]
            urgent[gk] = True
            gain[gk] = MOVE["urgent_gain"]
        elif self.loose:
            for k in (0, 1):
                idx = outfield(k)
                dist = np.linalg.norm(self.pos[idx] - self.ball, axis=1)
                n = int(idx[np.argmin(dist)])
                t_reach = float(dist.min()) / max(self.vmax[n], 1.0)
                tgt[n] = self.ball + self.ball_v * min(t_reach, 1.5)
                vcap[n] = self.vmax[n]
                urgent[n] = True
                self.minspd[n] = 5.0
        return tgt, vcap, amax, ff, gain, urgent

    def _closest_on_flight(self, i: int) -> np.ndarray:
        f = self.flight
        seg = f.end - f.start
        s = np.clip(np.dot(self.pos[i] - f.start, seg) / max(np.dot(seg, seg), 1e-6), 0.15, 0.95)
        return f.start + seg * s

    def _move(self, tgt, vcap, amax, ff, gain, urgent):
        delta = tgt - self.pos
        d = np.linalg.norm(delta, axis=1)
        speed = np.where(urgent, gain * d, gain * np.maximum(d - MOVE["dead_zone"], 0.0))
        ms = getattr(self, "minspd", None)
        if ms is not None:
            speed = np.where(d > 0.4, np.maximum(speed, ms), speed)
        desired = delta * (speed / np.maximum(d, 1e-6))[:, None] + ff
        sp = np.linalg.norm(desired, axis=1)
        desired *= np.minimum(1.0, vcap / np.maximum(sp, 1e-9))[:, None]
        dv = desired - self.vel
        m = np.linalg.norm(dv, axis=1)
        self.vel += dv * np.minimum(1.0, amax * DT / np.maximum(m, 1e-9))[:, None]
        self.pos += self.vel * DT
        self.pos[:, 0] = np.clip(self.pos[:, 0], -1.0, L + 1.0)
        self.pos[:, 1] = np.clip(self.pos[:, 1], -1.0, W + 1.0)

    def _update_ball(self):
        f = self.flight
        if self.carrier is not None:
            c = self.carrier
            sp = np.linalg.norm(self.vel[c])
            if sp > 0.4:
                self._ball_off += (self.vel[c] / sp * 0.6 - self._ball_off) * 0.15
            self.ball = self.pos[c] + self._ball_off
            self.ball_v = self.vel[c].copy()
            return
        if f is not None:
            s = min(1.0, (self.frame - f.f0) / max(f.f1 - f.f0, 1))
            se = s * (1.2 - 0.2 * s)                    # passes slow down a little
            prev = self.ball.copy()
            self.ball = f.start + (f.end - f.start) * se
            self.ball_v = (self.ball - prev) / DT
            if f.kind == "pass":
                travelled = np.linalg.norm(self.ball - f.start)
                if f.interceptor is not None and travelled > 2.0 and \
                        np.linalg.norm(self.pos[f.interceptor] - self.ball) < MOVE["control_r"] + 1.0:
                    self._claim(f.interceptor)
                    return
                if not f.safe and travelled > 3.0:
                    opp = outfield(1 - team_of(f.passer))
                    dd = np.linalg.norm(self.pos[opp] - self.ball, axis=1)
                    j = int(np.argmin(dd))
                    if dd[j] < 0.9 and self.rng.random() < 0.35:
                        self._claim(int(opp[j]))
                        return
                if f.receiver is not None and f.interceptor is None and s > 0.5 and \
                        np.linalg.norm(self.pos[f.receiver] - self.ball) < (2.4 if s > 0.7 else MOVE["control_r"] + 0.3):
                    self._claim(f.receiver)
                    return
            if s >= 1.0:
                self.flight = None
                if f.kind == "shot" and f.saved:
                    # the keeper holds it and restarts play from his hands
                    gk = keeper(1 - team_of(f.passer))
                    off = self.ball - self.pos[gk]
                    n = float(np.linalg.norm(off))
                    self._ball_off = off if n <= 1.0 else off / n * 1.0
                    self._event("restart", gk, self.ball, self.ball, how="save")
                    self.flip_possession(gk)
                    self.hold_until = self.frame + int(self.rng.uniform(3.0, 5.5) * FPS)
                    return
                if f.kind == "shot":
                    if f.goal:
                        self._event("goal", f.passer, self.ball, self.ball)
                    self.dead = True
                    self.carrier = None
                    self.assign[:] = ZONE
                    return
                # nobody there yet: the ball rolls on
                self.loose = True
                self.loose_since = self.frame
                # passes are weighted to the receiver: a missed one rolls on slowly
                self.ball_v = unit(f.end - f.start) * float(np.clip(np.linalg.norm(self.ball_v) * 0.6, 2.0, 5.0))
                self.loose_receiver = f.receiver
            return
        if self.loose:
            self.ball = self.ball + self.ball_v * DT
            sp = np.linalg.norm(self.ball_v)
            if sp > 1e-6:
                self.ball_v *= max(0.0, sp - MOVE["friction"] * DT) / sp
            self.ball = np.clip(self.ball, [0.5, 0.5], [L - 0.5, W - 0.5])
            if self.frame - self.loose_since > 3:
                d = np.linalg.norm(self.pos - self.ball, axis=1)
                d[[0, 11]] += 3.0                       # keepers stay home
                i = int(np.argmin(d))
                if d[i] < MOVE["control_r"]:
                    self._claim(i)

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
    toward a target depth, and loses it to a tackle after a scheduled frame."""

    def __init__(self, rng: np.random.Generator, steer_x: Optional[float] = None,
                 turnover_frame: Optional[int] = None):
        self.rng = rng
        self.steer_x = steer_x
        self.turnover_frame = turnover_frame
        self.tackle_deadline = 0

    def cancel_tackle(self, sim: "Sim"):
        sim.want_ball = False
        self.turnover_frame = None

    def step(self, sim: Sim):
        if sim.flight is not None or sim.carrier is None or sim.loose:
            return
        c = sim.carrier
        T = team_of(c)
        if self.turnover_frame is not None and sim.frame >= self.turnover_frame:
            self._tackle(sim, c)
            return
        pressed = np.min(np.linalg.norm(sim.pos[outfield(1 - T)] - sim.pos[c], axis=1)) < 2.0
        if sim.frame < sim.hold_until and not (pressed and sim.rng.random() < 0.06):
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
        score = 0.35 * openness + 0.5 * lane - 0.04 * d + 0.03 * (rx - cx)
        if self.steer_x is not None:
            score -= 0.10 * np.abs(rx - self.steer_x)
        score[(d < 6) | (d > 40) | (lane < 1.0)] = -1e9
        if np.all(score < -1e8):
            # nothing on: after a moment, take the safest option available
            if sim.frame < sim.hold_until + int(1.0 * FPS):
                return
            score = 0.6 * lane - 0.02 * d
            score[(d < 4) | (d > 45)] = -1e9
            if np.all(score < -1e8):
                return
        p = np.exp((score - score.max()) / 0.5)
        p /= p.sum()
        r = mates[int(self.rng.choice(len(mates), p=p))]
        running = sim.run_until[r] > sim.frame
        sim.pass_to(r, lead=2.0 if running else 0.0)

    def _tackle(self, sim: Sim, c: int):
        """The team out of possession goes in: the presser closes and wins the ball."""
        if not sim.want_ball:
            sim.want_ball = True
            self.tackle_deadline = sim.frame + 8 * FPS
        D = 1 - team_of(c)
        defs = outfield(D)
        d = np.linalg.norm(sim.pos[defs] - sim.pos[c], axis=1)
        k = int(np.argmin(d))
        if d[k] < 1.1:
            w = int(defs[k])
            off = sim.ball - sim.pos[w]
            n = float(np.linalg.norm(off))
            sim._ball_off = off if n <= 1.2 else off / n * 1.2      # ball stays where it was, no jump
            sim._event("tackle", w, sim.ball, sim.ball)
            sim.flip_possession(w)
            self.turnover_frame = None
        elif sim.frame > self.tackle_deadline:
            # could not get close enough: the carrier is hurried into a pass that gets cut out
            self._forced_pass(sim, c, defs)

    def _forced_pass(self, sim: Sim, c: int, defs):
        best = None
        for m in outfield(team_of(c)):
            if m == c:
                continue
            seg = sim.pos[m] - sim.pos[c]
            L2 = float(seg @ seg)
            if L2 < 25.0 or L2 > 900.0:
                continue
            for dd in defs:
                t = float(np.clip((sim.pos[dd] - sim.pos[c]) @ seg / L2, 0.25, 0.85))
                pt = sim.pos[c] + t * seg
                gap = float(np.linalg.norm(sim.pos[dd] - pt))
                if best is None or gap < best[0]:
                    best = (gap, int(m), int(dd), pt)
        sim.want_ball = False
        self.turnover_frame = None
        if best is None:
            return
        _, m, dd, pt = best
        sim.intercepted_pass(m, dd, pt, speed=12.0)


def run_to(target: np.ndarray, vmax: float, amax: float = 5.0, kp: float = 2.5) -> Override:
    target = np.asarray(target, dtype=float)
    return lambda sim: (target, vmax, amax, np.zeros(2), kp)


def hold_ball(i: int) -> Override:
    """Carrier waits for a team-mate's run: keeps the ball moving slowly forward, shielding it."""
    def fn(sim):
        A = team_of(i)
        fwd = unit(opp_goal(A) - sim.pos[i])
        opp = sim.pos[outfield(1 - A)]
        q = opp[int(np.argmin(np.linalg.norm(opp - sim.pos[i], axis=1)))]
        away = unit(sim.pos[i] - q) if np.linalg.norm(q - sim.pos[i]) < 4 else np.zeros(2)
        return sim.pos[i] + (fwd + away) * 3.0, 2.0, 3.0, np.zeros(2), 1.0
    return fn
