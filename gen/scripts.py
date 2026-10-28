"""Scripted chances. Each script overrides only a few players; everyone else keeps the
generator's normal formation / marking / physics behaviour. Every script records its
ground-truth cause, which is written to the label file and never shown to the analysis.

Kinds
  decoy_run      striker runs wide or deep, his marker follows, a team-mate attacks the vacated space
  lost_marker    attacker runs in behind, his marker stops tracking for 1.3-2.0 s
  press_turnover pressing team closes the passing lanes high up the pitch, intercepts, shoots fast
  neg_nofollow   same wide run, but the marker holds his zone (nobody dragged); shot from between the lines
  neg_outpaced   marker keeps chasing at full effort but is simply slower; attacker gets away
  neg_long       ordinary circulation ending in a long-range shot
  neg_unforced   the ball is won high up the pitch from a loose pass nobody forced (no one closes
                 the carrier down), then a quick shot
The neg_* kinds have ground-truth cause "none".

Scripts that need an established marking relation first go through a short "settle"
phase: the marker is locked onto the attacker and the carrier shields the ball until the
marker is within a few metres (or 2.5 s pass). The ground-truth start time is the moment
the decisive action begins (run start, switch-off, press trigger), after settling.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from common.pitch import DT, FPS, PLAYER_IDS, W, from_rel, opp_goal, outfield, team_of, to_rel
from gen.generator import Sim, hold_ball, run_to, unit

STRIKERS = {"ST", "RST", "LST"}
WIDE = {"RW", "LW", "RM", "LM"}
MIDS = {"DM", "RCM", "LCM"}
DEF_ROLES = {"RCB", "LCB", "RB", "LB"}
SETTLE_MAX_S = 4.0
GOAL_PROB = 0.15   # chance of a scripted chance ending in a goal (keeps scorelines plausible)
FREE_M = 5.0        # a beneficiary must start with no defender this close (clean ground truth)


class ScriptAbort(RuntimeError):
    """The current state cannot produce a clean instance of this scenario; retry the episode."""


class Script:
    kind = "base"
    cause = "none"

    def __init__(self, rng: np.random.Generator, team: int, start_frame: int):
        self.rng = rng
        self.team = team
        self.start_frame = start_frame
        self.phase = "wait"
        self.t0: Optional[int] = None
        self.shot_frame: Optional[int] = None
        self.done = False
        self.players: dict = {}
        self.extra: dict = {}
        self._saved: dict = {}

    def tune(self, sim: Sim, i: int, **attrs):
        """Temporarily change a player's per-player traits (restored when the script ends)."""
        for k, v in attrs.items():
            arr = getattr(sim, k)
            self._saved.setdefault((k, i), arr[i].copy() if hasattr(arr[i], "copy") else arr[i])
            arr[i] = v

    def restore(self, sim: Sim):
        for (k, i), v in self._saved.items():
            getattr(sim, k)[i] = v
        self._saved.clear()

    # helpers -------------------------------------------------------------
    def rel(self, xy):
        return to_rel(xy, self.team)

    def ab(self, rel_xy):
        return from_rel(np.asarray(rel_xy, dtype=float), self.team)

    def truth(self) -> dict:
        return {
            "kind": self.kind,
            "cause": self.cause,
            "team": "AB"[self.team],
            "players": self.players,
            "t_start": None if self.t0 is None else round(self.t0 * DT, 3),
            "t_shot": None if self.shot_frame is None else round(self.shot_frame * DT, 3),
            **self.extra,
        }

    def step(self, sim: Sim):
        if self.done:
            return
        if self.phase == "wait":
            if sim.frame >= self.start_frame and self.ready(sim):
                sim.director_enabled = False
                sim.runs_enabled = False            # no unscripted runs during a scripted chance
                sim.run_until[:] = 0
                if sim.director is not None:
                    sim.director.cancel_tackle(sim)
                self.begin(sim)
                self.settle_end = sim.frame + int(SETTLE_MAX_S * FPS)
                self.phase = "settle"
            return
        if self.t0 is not None and sim.frame - self.t0 > 15 * FPS:
            raise ScriptAbort(f"{self.kind}: did not finish within 15 s")
        if self.phase == "intercept" and sim.flight is None and not sim.loose and sim.possession != self.team:
            raise ScriptAbort(f"{self.kind}: the interception did not happen")
        if self.phase in ("running", "passed", "carry", "counter") and sim.possession != self.team \
                and not (self.kind in ("press_turnover", "neg_unforced") and self.phase in ("pressing", "intercept")):
            raise ScriptAbort(f"{self.kind}: the ball was lost before the chance")
        if self.phase == "settle":
            if self.settled(sim) or sim.frame >= self.settle_end:
                if self.can_go(sim):
                    self.t0 = sim.frame
                    self.go(sim)
                elif sim.frame >= self.settle_end + int(3.0 * FPS):
                    raise ScriptAbort(f"{self.kind}: no clean start")
            return
        self.update(sim)

    def ready(self, sim: Sim) -> bool:
        if sim.possession != self.team or sim.carrier is None or sim.flight is not None or sim.dead:
            return False
        c = sim.carrier
        cx = self.rel(sim.pos[c])[0]
        late = sim.frame > self.start_frame + 15 * FPS
        if not ((48.0 if late else 54.0) <= cx <= 76.0 and sim.role[c] in MIDS | WIDE):
            return False
        fw = [i for i in outfield(self.team) if sim.role[i] in STRIKERS and i != c]
        return min(np.linalg.norm(sim.pos[i] - sim.pos[c]) for i in fw) < 30.0

    def begin(self, sim: Sim):
        pass

    def settled(self, sim: Sim) -> bool:
        return True

    def can_go(self, sim: Sim) -> bool:
        return True

    def _free_cands(self, sim: Sim, cands) -> list:
        dfn = sim.pos[outfield(1 - self.team)]
        return [i for i in cands if sim.marker_of(i) is None
                and np.linalg.norm(dfn - sim.pos[i], axis=1).min() > FREE_M]

    def go(self, sim: Sim):
        raise NotImplementedError

    def update(self, sim: Sim):
        raise NotImplementedError

    # common finishing: receive -> carry -> shoot -------------------------
    def _finish(self, sim: Sim, shooter: int, carry_range=(0.3, 0.9)):
        if self.phase == "passed" and sim.carrier == shooter:
            self.phase = "carry"
            self.extra["t_reception"] = round(sim.frame * DT, 3)
            self._shoot_at = sim.frame + int(self.rng.uniform(*carry_range) * FPS)
            g = opp_goal(self.team)
            sim.set_override(shooter, run_to(g, 5.5, 4.0, 1.0))
        elif self.phase == "carry" and sim.frame >= self._shoot_at and sim.carrier == shooter:
            sim.clear_override(shooter)
            sim.shoot(shooter, goal_prob=GOAL_PROB)
            self.shot_frame = sim.frame
            self.phase = "shot"
            self._release_all(sim)
            sim.runs_enabled = True
            self.done = True

    def _release_all(self, sim: Sim):
        for i in list(sim.ov.keys()):
            sim.clear_override(i)
        sim.locked.clear()
        self.restore(sim)

    def _attackers(self, sim: Sim, roles, exclude=()):
        return [i for i in outfield(self.team) if sim.role[i] in roles and i not in exclude]

    def _nearest(self, sim: Sim, cands, point):
        cands = list(cands)
        d = [np.linalg.norm(sim.pos[i] - point) for i in cands]
        return cands[int(np.argmin(d))]

    def _free_choice(self, sim: Sim, cands, point) -> int:
        """Pick a beneficiary, preferring one nobody is marking, weighted toward `point`.
        A free beneficiary stays free; a marked one keeps his marker, who tracks him as normal
        (so the only defender who fails anyone in the scenario is the one the script says)."""
        free = self._free_cands(sim, cands)
        pool = free or list(cands)
        dist = np.array([np.linalg.norm(sim.pos[i] - point) for i in pool])
        w = np.exp(-dist / 8.0)
        S = pool[int(self.rng.choice(len(pool), p=w / w.sum()))]
        if S in free:
            sim.unmarkable.add(S)
        return S

    def _pick_marker(self, sim: Sim, R: int, prefer_roles=None) -> int:
        D = 1 - self.team
        Dm = sim.marker_of(R)
        if Dm is None:
            pool = [d for d in outfield(D) if prefer_roles is None or sim.role[d] in prefer_roles]
            Dm = self._nearest(sim, pool, sim.pos[R])
        sim.force_assign(Dm, R, lock=True)
        return Dm


class DecoyRun(Script):
    kind = "decoy_run"
    cause = "decoy_run"

    def begin(self, sim: Sim):
        c = sim.carrier
        strikers = self._attackers(sim, STRIKERS, exclude=(c,))
        self.R = max(strikers, key=lambda i: self.rel(sim.pos[i])[0])
        self.Dm = self._pick_marker(sim, self.R, prefer_roles={"RCB", "LCB"})
        self.c = c
        sim.set_override(c, hold_ball(c))

    def settled(self, sim: Sim) -> bool:
        steady = (np.linalg.norm(sim.pos[self.Dm] - sim.pos[self.R]) < 3.5
                  and np.linalg.norm(sim.vel[self.Dm] - sim.vel[self.R]) < 1.5)
        self._steady = getattr(self, "_steady", 0) + 1 if steady else 0
        return self._steady >= FPS

    def can_go(self, sim: Sim) -> bool:
        return getattr(self, "_steady", 0) >= FPS

    def _cands(self, sim: Sim):
        D0 = sim.pos[self.Dm]
        cx = self.rel(sim.pos[self.c])[0]
        cands = [i for i in outfield(self.team) if i not in (self.R, self.c) and sim.role[i] not in DEF_ROLES]
        near = [i for i in cands if self.rel(sim.pos[i])[0] >= cx - 4 and np.linalg.norm(sim.pos[i] - D0) < 30]
        return near or cands

    def go(self, sim: Sim):
        T = self.team
        R, Dm, c = self.R, self.Dm, self.c
        D0 = sim.pos[Dm].copy()
        D0r = self.rel(D0)
        S = self._free_choice(sim, self._cands(sim), D0)
        # a centre-back who reads the run tracks it closely (short reaction, tight goal-side gap)
        self.tune(sim, Dm, delay_f=int(self.rng.uniform(0.10, 0.22) * FPS),
                  goal_side=self.rng.uniform(0.9, 1.4), amax=max(float(sim.amax[Dm]), 4.8))

        Rr = self.rel(sim.pos[R])
        Sr = self.rel(sim.pos[S])
        side = -1.0 if Sr[1] > Rr[1] else 1.0       # run away from the beneficiary's side
        wide_y = float(np.clip(Rr[1] + side * self.rng.uniform(13, 19), 3, W - 3))
        if self.rng.random() < 0.7 and abs(wide_y - Rr[1]) >= 10.0:
            run_type = "wide"
            tgt_r = np.array([Rr[0] + self.rng.uniform(-2, 4), wide_y])
        else:
            run_type = "deep"
            tgt_r = np.array([Rr[0] - self.rng.uniform(11, 15), Rr[1] + self.rng.uniform(-4, 4)])
        tgt_r[0] = min(tgt_r[0], 97.0)
        tgt_r[1] = np.clip(tgt_r[1], 3, W - 3)
        vr = self.rng.uniform(7.0, 7.8)
        sim.set_override(R, run_to(self.ab(tgt_r), vr, 5.0))
        self.run_T = np.linalg.norm(tgt_r - Rr) / vr + 0.6

        s_tgt_r = D0r + np.array([self.rng.uniform(0, 3), self.rng.uniform(-3, 3)])
        s_tgt_r[0] = min(s_tgt_r[0], 94.0)
        self.s_tgt = self.ab(s_tgt_r)
        self.s_go = sim.frame + int(self.rng.uniform(0.6, 1.4) * FPS)
        self.min_pass = sim.frame + int(0.7 * self.run_T * FPS)
        self.max_pass = sim.frame + int(5.0 * FPS)
        self.S = S
        self.players = {"runner": PLAYER_IDS[R], "dragged": PLAYER_IDS[Dm], "beneficiary": PLAYER_IDS[S],
                        "passer": PLAYER_IDS[c]}
        self.extra = {"run_type": run_type}
        self.phase = "running"

    def update(self, sim: Sim):
        if self.phase == "running":
            if sim.frame == self.s_go:
                sim.set_override(self.S, run_to(self.s_tgt, 7.2, 5.0))
            if sim.frame >= self.s_go and sim.carrier == self.c and sim.frame >= self.min_pass:
                flight = np.linalg.norm(self.s_tgt - sim.pos[self.c]) / 16.0
                eta = np.linalg.norm(self.s_tgt - sim.pos[self.S]) / max(np.linalg.norm(sim.vel[self.S]), 3.0)
                if eta <= flight + 0.3 or sim.frame >= self.max_pass:
                    sim.clear_override(self.c)
                    sim.clear_override(self.S)
                    sim.pass_to(self.S, speed=self.rng.uniform(14, 18), lead=0.8, safe=True)
                    self.extra["t_pass"] = round(sim.frame * DT, 3)
                    self.phase = "passed"
            return
        self._finish(sim, self.S)


class LostMarker(Script):
    kind = "lost_marker"
    cause = "lost_marker"
    outpaced = False

    def begin(self, sim: Sim):
        c = sim.carrier
        cands = [i for i in outfield(self.team) if i != c and sim.role[i] in STRIKERS | WIDE]
        fwd = [i for i in cands if self.rel(sim.pos[i])[0] >= 60]
        cands = fwd or cands
        marked = [i for i in cands if sim.marker_of(i) is not None]
        self.R = int(self.rng.choice(marked if marked else cands))
        self.Dm = self._pick_marker(sim, self.R)
        self.c = c
        sim.set_override(c, hold_ball(c))

    def settled(self, sim: Sim) -> bool:
        # a real man-marking job first: within 2.5 m of his man for 1.5 s
        close = np.linalg.norm(sim.pos[self.Dm] - sim.pos[self.R]) < 2.5
        self._tight = getattr(self, "_tight", 0) + 1 if close else 0
        return self._tight >= int(1.5 * FPS)

    def can_go(self, sim: Sim) -> bool:
        return getattr(self, "_tight", 0) >= int(1.5 * FPS)   # otherwise the episode is redrawn

    def go(self, sim: Sim):
        T = self.team
        R, Dm, c = self.R, self.Dm, self.c
        tgt_r = np.array([self.rng.uniform(88, 94), 34 + self.rng.uniform(-9, 9)])
        if self.outpaced:
            vr, ar = 8.6, 6.5
        else:
            vr, ar = self.rng.uniform(7.2, 8.0), 5.5
        sim.set_override(R, run_to(self.ab(tgt_r), vr, ar))
        self.off_frame = sim.frame + 1 + int(self.rng.uniform(0.0, 0.4) * FPS)
        self.off_len = int(self.rng.uniform(1.3, 2.0) * FPS)
        self.off_v = self.rng.uniform(0.8, 2.0)
        self.pass_frame = sim.frame + int(self.rng.uniform(1.1, 1.6) * FPS)
        self.players = {"attacker": PLAYER_IDS[R], "defender": PLAYER_IDS[Dm], "passer": PLAYER_IDS[c]}
        self.phase = "running"
        if self.outpaced:
            dv = self.rng.uniform(5.9, 6.5)
            self.extra = {"defender_vmax": round(dv, 2)}
            gs = sim.goal_side[Dm]
            g = opp_goal(T)

            am = float(sim.amax[Dm])

            def chase(s, Dm=Dm, R=R, dv=dv, gs=gs, g=g, am=am):
                # full effort with normal acceleration, but a lower top speed than the attacker
                pa, va = s.pos_delayed(R, int(s.delay_f[Dm]))
                return pa + unit(g - pa) * gs, dv, am, np.zeros(2), 3.0
            sim.set_override(Dm, chase)

    def update(self, sim: Sim):
        if not self.outpaced:
            if sim.frame == self.off_frame:
                self.t0 = sim.frame  # ground-truth start = moment the defender switches off
                p0 = sim.pos[self.Dm].copy()
                v = self.off_v
                sim.set_override(self.Dm, lambda s, p0=p0, v=v: (p0 + unit(s.ball - p0) * 1.0, v, 3.0, np.zeros(2), 1.0))
                self.extra = {"switch_off_s": round(self.off_len * DT, 2), "switch_off_speed": round(v, 2)}
            if sim.frame == self.off_frame + self.off_len:
                sim.clear_override(self.Dm)
        if self.phase == "running":
            if sim.frame >= self.pass_frame and sim.carrier == self.c:
                sim.clear_override(self.c)
                sim.pass_to(self.R, speed=self.rng.uniform(15, 19), lead=2.0, safe=True)
                self.extra["t_pass"] = round(sim.frame * DT, 3)
                self.phase = "passed"
            return
        self._finish(sim, self.R, carry_range=(0.2, 0.6))


class Outpaced(LostMarker):
    kind = "neg_outpaced"
    cause = "none"
    outpaced = True


class NoFollow(Script):
    kind = "neg_nofollow"
    cause = "none"

    def begin(self, sim: Sim):
        c = sim.carrier
        strikers = self._attackers(sim, STRIKERS, exclude=(c,))
        self.R = max(strikers, key=lambda i: self.rel(sim.pos[i])[0])
        self.Dm = self._pick_marker(sim, self.R, prefer_roles={"RCB", "LCB"})
        self.c = c
        sim.set_override(c, hold_ball(c))

    def settled(self, sim: Sim) -> bool:
        return np.linalg.norm(sim.pos[self.Dm] - sim.pos[self.R]) < 4.0

    def _cands(self, sim: Sim):
        cx = self.rel(sim.pos[self.c])[0]
        cands = [i for i in outfield(self.team) if i not in (self.R, self.c) and sim.role[i] in MIDS | STRIKERS | WIDE]
        near = [i for i in cands if self.rel(sim.pos[i])[0] >= cx - 4]
        return near or cands

    def go(self, sim: Sim):
        T, D = self.team, 1 - self.team
        R, Dm, c = self.R, self.Dm, self.c
        line = np.max(self.rel(sim.pos[outfield(D)])[:, 0])
        s_tgt_r = np.array([min(line, 92) - self.rng.uniform(6, 10), 34 + self.rng.uniform(-8, 8)])
        S = self._free_choice(sim, self._cands(sim), self.ab(s_tgt_r))
        sim.release(Dm, zonal_lock=True)           # the marker holds his zone: nobody is dragged
        sim.unmarkable.add(S)
        Rr = self.rel(sim.pos[R])
        side = 1.0 if self.rng.random() < 0.5 else -1.0
        if not 3 <= Rr[1] + side * 13 <= W - 3:
            side = -side
        tgt_r = np.array([Rr[0] + self.rng.uniform(-2, 4), Rr[1] + side * self.rng.uniform(13, 19)])
        tgt_r[0] = min(tgt_r[0], 97.0)
        tgt_r[1] = np.clip(tgt_r[1], 3, W - 3)
        sim.set_override(R, run_to(self.ab(tgt_r), self.rng.uniform(7.0, 7.8), 5.0))
        self.s_tgt = self.ab(s_tgt_r)
        sim.set_override(S, run_to(self.s_tgt, 6.8, 4.5))
        self.min_pass = sim.frame + int(1.4 * FPS)
        self.max_pass = sim.frame + int(5.0 * FPS)
        self.S = S
        self.players = {"runner": PLAYER_IDS[R], "marker_held_zone": PLAYER_IDS[Dm], "shooter": PLAYER_IDS[S]}
        self.phase = "running"

    def update(self, sim: Sim):
        if self.phase == "running":
            if sim.carrier == self.c and sim.frame >= self.min_pass:
                flight = np.linalg.norm(self.s_tgt - sim.pos[self.c]) / 15.0
                eta = np.linalg.norm(self.s_tgt - sim.pos[self.S]) / max(np.linalg.norm(sim.vel[self.S]), 3.0)
                if eta <= flight + 0.3 or sim.frame >= self.max_pass:
                    sim.clear_override(self.c)
                    sim.clear_override(self.S)
                    sim.pass_to(self.S, speed=self.rng.uniform(13, 17), safe=True)
                    self.extra["t_pass"] = round(sim.frame * DT, 3)
                    self.phase = "passed"
            return
        self._finish(sim, self.S, carry_range=(0.4, 1.0))


class LongShot(Script):
    kind = "neg_long"
    cause = "none"

    def ready(self, sim: Sim) -> bool:
        if sim.possession != self.team or sim.carrier is None or sim.flight is not None or sim.dead:
            return False
        cx = self.rel(sim.pos[sim.carrier])[0]
        late = sim.frame > self.start_frame + 10 * FPS
        return cx >= (55.0 if late else 66.0) and sim.role[sim.carrier] not in DEF_ROLES | {"GK"}

    def go(self, sim: Sim):
        c = sim.carrier
        cr = self.rel(sim.pos[c])
        tgt = self.ab([max(cr[0], self.rng.uniform(74, 80)), np.clip(cr[1], 18, 50)])
        sim.set_override(c, run_to(tgt, 5.0, 4.0, 1.2))
        self.c = c
        self.players = {"shooter": PLAYER_IDS[c]}
        self.phase = "carry_long"

    def update(self, sim: Sim):
        if self.phase == "carry_long" and sim.carrier == self.c:
            if self.rel(sim.pos[self.c])[0] >= 73 or sim.frame > self.t0 + 4 * FPS:
                sim.clear_override(self.c)
                sim.shoot(self.c, goal_prob=0.08)
                self.shot_frame = sim.frame
                sim.runs_enabled = True
                self.done = True


class PressTurnover(Script):
    """`team` is the pressing team, which wins the ball and shoots."""
    kind = "press_turnover"
    cause = "press_turnover"

    def ready(self, sim: Sim) -> bool:
        Y = 1 - self.team
        if sim.possession != Y or sim.carrier is None or sim.flight is not None or sim.dead:
            return False
        c = sim.carrier
        cx = to_rel(sim.pos[c], Y)[0]
        if not (cx < 30.0 and sim.role[c] in DEF_ROLES):
            return False
        fw = [i for i in outfield(self.team) if sim.role[i] in STRIKERS | WIDE]
        return min(np.linalg.norm(sim.pos[i] - sim.pos[c]) for i in fw) < 18.0

    def go(self, sim: Sim):
        X, Y = self.team, 1 - self.team
        c = sim.carrier
        mates = [m for m in outfield(Y) if m != c]
        dist = np.array([np.linalg.norm(sim.pos[m] - sim.pos[c]) for m in mates])
        order = [mates[k] for k in np.argsort(dist)]
        Q1, Q2, Q3 = order[0], order[1], order[2]
        xs = list(outfield(X))
        fw = [i for i in xs if sim.role[i] in STRIKERS | WIDE]
        p1 = self._nearest(sim, fw, sim.pos[c])
        rest = [i for i in xs if i != p1]

        def lane_pt(s, q, frac):
            return s.pos[c] + (s.pos[q] - s.pos[c]) * frac

        f2 = self.rng.uniform(0.45, 0.6)
        p2 = self._nearest(sim, [i for i in rest if sim.role[i] not in DEF_ROLES], lane_pt(sim, Q2, f2))
        rest = [i for i in rest if i != p2]
        f3 = self.rng.uniform(0.4, 0.6)
        p3 = self._nearest(sim, [i for i in rest if sim.role[i] not in DEF_ROLES], lane_pt(sim, Q3, f3))
        sim.force_assign(p1, c, lock=True)
        for p in (p2, p3):
            sim.release(p, zonal_lock=True)
        sim.set_override(p1, lambda s: (s.pos[c] + unit(s.pos[Q1] - s.pos[c]) * 2.2, 7.6, 5.0, np.zeros(2), 3.0))
        sim.set_override(p2, lambda s: (lane_pt(s, Q2, f2), 7.3, 5.0, np.zeros(2), 3.0))
        sim.set_override(p3, lambda s: (lane_pt(s, Q3, f3), 7.0, 5.0, np.zeros(2), 3.0))
        sim.set_override(c, hold_ball(c))
        self.press_start = sim.frame
        self.release_frame = sim.frame + int(self.rng.uniform(1.4, 2.2) * FPS)
        self.c, self.p1, self.p2, self.p3, self.Q2 = c, p1, p2, p3, Q2
        self.players = {"ball_winner": PLAYER_IDS[p2], "pressers": [PLAYER_IDS[p1], PLAYER_IDS[p2], PLAYER_IDS[p3]],
                        "victim": PLAYER_IDS[c]}
        self.phase = "pressing"

    def update(self, sim: Sim):
        X = self.team
        if self.phase == "pressing":
            closed = np.linalg.norm(sim.pos[self.p1] - sim.pos[self.c]) < 3.0
            seg = sim.pos[self.Q2] - sim.pos[self.c]
            tp = np.clip(np.dot(sim.pos[self.p2] - sim.pos[self.c], seg) / max(np.dot(seg, seg), 1e-6), 0, 1)
            closed = closed and np.linalg.norm(sim.pos[self.c] + seg * tp - sim.pos[self.p2]) < 2.5
            if sim.frame > self.press_start + int(4.5 * FPS):
                raise ScriptAbort("press never reached the carrier")
            if sim.frame >= self.release_frame and closed and sim.carrier == self.c:
                c, p2, Q2 = self.c, self.p2, self.Q2
                seg = sim.pos[Q2] - sim.pos[c]
                tproj = np.clip(np.dot(sim.pos[p2] - sim.pos[c], seg) / max(np.dot(seg, seg), 1e-6), 0.2, 0.9)
                pt = sim.pos[c] + seg * tproj
                sim.clear_override(c)
                sim.intercepted_pass(Q2, p2, pt, self.rng.uniform(11, 14))
                sim.set_override(p2, run_to(pt, 7.0, 5.0, 3.0))
                self.phase = "intercept"
            return
        if self.phase == "intercept" and sim.carrier == self.p2 and sim.possession == X:
            self.extra["t_turnover"] = round(sim.frame * DT, 3)
            for p in (self.p1, self.p3):
                sim.clear_override(p)
            g = opp_goal(X)
            sim.director_enabled = False
            if np.linalg.norm(sim.pos[self.p2] - g) < 26:
                self.shooter = self.p2
                sim.set_override(self.p2, run_to(g, 5.5, 4.0, 1.0))
                self._shoot_at = sim.frame + int(self.rng.uniform(0.4, 0.9) * FPS)
                self.phase = "carry"
            else:
                sim.clear_override(self.p2)
                fw = [i for i in outfield(X) if i != self.p2 and sim.role[i] in STRIKERS | WIDE | MIDS]
                close = [i for i in fw if np.linalg.norm(sim.pos[i] - sim.pos[self.p2]) < 28]
                tgt = self._nearest(sim, close or fw, g)
                self.shooter = tgt
                self.run_pt = g + np.array([-self.rng.uniform(11, 15) * (1 if X == 0 else -1), self.rng.uniform(-6, 6)])
                sim.set_override(tgt, run_to(self.run_pt, 7.0, 5.0))
                self._pass_at = sim.frame + int(self.rng.uniform(0.3, 0.7) * FPS)
                self.phase = "counter"
            self.players["shooter"] = PLAYER_IDS[self.shooter]
            return
        if self.phase == "counter" and sim.frame >= self._pass_at and sim.carrier == self.p2:
            sim.clear_override(self.shooter)
            sim.pass_to(self.shooter, speed=self.rng.uniform(15, 18), lead=2.0, safe=True)
            self.phase = "passed"
            return
        if self.phase in ("passed", "carry"):
            self._finish(sim, self.shooter, carry_range=(0.2, 0.6))


class UnforcedTurnover(PressTurnover):
    """High turnover without a press: the carrier has time and space, plays a loose pass,
    and an opponent reading the lane intercepts it."""
    kind = "neg_unforced"
    cause = "none"

    def ready(self, sim: Sim) -> bool:
        Y = 1 - self.team
        if sim.possession != Y or sim.carrier is None or sim.flight is not None or sim.dead:
            return False
        c = sim.carrier
        if not (to_rel(sim.pos[c], Y)[0] < 36.0 and sim.role[c] in DEF_ROLES | MIDS):
            return False
        return min(np.linalg.norm(sim.pos[i] - sim.pos[c]) for i in outfield(self.team)) > 7.0

    def go(self, sim: Sim):
        X, Y = self.team, 1 - self.team
        c = sim.carrier
        for d in outfield(X):                     # nobody steps up to the carrier: the team holds its shape
            sim.release(d, zonal_lock=True)
        sim.unmarkable.add(c)
        mates = [m for m in outfield(Y) if m != c]
        dist = np.array([np.linalg.norm(sim.pos[m] - sim.pos[c]) for m in mates])
        order = [mates[k] for k in np.argsort(dist)]
        Q2 = order[min(1, len(order) - 1)]
        mid = sim.pos[c] + (sim.pos[Q2] - sim.pos[c]) * 0.55
        xs = [i for i in outfield(X) if sim.role[i] not in DEF_ROLES]
        p2 = self._nearest(sim, xs, mid)
        f2 = self.rng.uniform(0.45, 0.65)
        sim.set_override(p2, lambda s: (s.pos[c] + (s.pos[Q2] - s.pos[c]) * f2, 6.0, 4.0, np.zeros(2), 2.0))
        sim.set_override(c, hold_ball(c))
        self.press_start = sim.frame
        self.release_frame = sim.frame + int(self.rng.uniform(1.0, 1.8) * FPS)
        self.c, self.p1, self.p2, self.p3, self.Q2 = c, p2, p2, p2, Q2
        self.players = {"ball_winner": PLAYER_IDS[p2], "victim": PLAYER_IDS[c]}
        self.phase = "pressing"

    def update(self, sim: Sim):
        if self.phase == "pressing":
            if sim.frame > self.press_start + int(4.5 * FPS):
                raise ScriptAbort("unforced: no release")
            if sim.frame >= self.release_frame and sim.carrier == self.c:
                c, p2, Q2 = self.c, self.p2, self.Q2
                seg = sim.pos[Q2] - sim.pos[c]
                tproj = np.clip(np.dot(sim.pos[p2] - sim.pos[c], seg) / max(np.dot(seg, seg), 1e-6), 0.2, 0.9)
                pt = sim.pos[c] + seg * tproj
                sim.clear_override(c)
                sim.intercepted_pass(Q2, p2, pt, self.rng.uniform(11, 14))
                sim.set_override(p2, run_to(pt, 7.0, 5.0, 3.0))
                self.phase = "intercept"
            return
        super().update(sim)


SCRIPTS = {
    "decoy_run": DecoyRun,
    "lost_marker": LostMarker,
    "press_turnover": PressTurnover,
    "neg_nofollow": NoFollow,
    "neg_outpaced": Outpaced,
    "neg_long": LongShot,
    "neg_unforced": UnforcedTurnover,
}
KINDS = list(SCRIPTS)
CAUSE_OF_KIND = {k: v.cause for k, v in SCRIPTS.items()}
