"""Chance genesis: starting from a shot, trace back up to 10 s and decide how the chance
was created. Everything here is deterministic geometry on tracking data plus the
marking assignments inferred in analysis/marking.py. Ground-truth labels are never read.

Candidate causes
  decoy_run       a defender who was protecting the eventual reception point left it while
                  tracking the man he was marking; the shooter received in the space he left
  lost_marker     the shooter's own marker stopped tracking him: the gap grew while the
                  defender's speed stayed well below the attacker's
  press_turnover  the shooting team won the ball high up the pitch right after the
                  opponent's carrier had (almost) no open passing lane
The strongest valid candidate becomes the primary cause; the others are reported as
secondary factors. "outpaced" (the marker kept chasing but was slower) is reported as
an individual, non-tactical pattern and maps to no tactical cause.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from analysis.lanes import lanes_at
from analysis.preprocess import Track, reception_frame
from analysis.space import space_at
from common.pitch import FPS, ID_TO_IDX, L, PLAYER_IDS, fmt_clock, opp_goal, outfield, team_of, to_rel


@dataclass
class GenesisParams:
    lookback_s: float = 10.0
    # decoy run
    protect_r: float = 5.0       # dragged defender was standing where the shooter later received
    protect_rank: int = 2         # ... and among the k nearest defenders to it
    min_drag: float = 9.0         # metres he moved away from the reception point
    follow_frac: float = 0.3      # share of the drag spent assigned to the runner
    follow_max: float = 8.0       # median distance to the runner while being dragged (he trails a sprinter)
    run_speed_min: float = 5.5    # the runner actually sprints
    run_len_min: float = 6.0      # ... over at least this distance
    drag_recent_s: float = 6.0    # the drag starts at most this long before the reception
    # lost marker
    sep_start_max: float = 4.0    # marker was this close at the start
    mark_window_s: int = 5        # ... and assigned to the attacker for most of the 5 s before
    mark_frac: float = 0.5
    sep_end_min: float = 5.0      # gap at reception
    watch_def_speed: float = 2.5  # marker below jogging pace ...
    watch_att_speed: float = 4.5  # ... while his man is sprinting
    watch_min_s: float = 0.9      # for at least this long -> switched off
    tight_m: float = 2.5          # tracking his man this closely ...
    tight_min_s: float = 1.0      # ... for at least this long in the 3 s before the gap opened
    outpaced_ratio_min: float = 0.55  # marker covered >= this share of his man's distance
    # press
    press_window_s: float = 3.0
    press_contact_m: float = 4.0  # nearest presser this close to the carrier when he loses it
    press_closing_m: float = 4.0  # ... after closing in by at least this much
    press_tight_s: float = 1.0    # (or: already this tight for at least this long)
    press_unit_r: float = 20.0    # pressing unit: opponents within this radius of the carrier
    press_unit_n: int = 3         # ... at least this many
    high_turnover_m: float = 40.0  # ball lost within this distance of the victim's goal line
    fast_s: float = 10.0          # shot within this long of the turnover


CAUSE_LABELS = {"decoy_run": "Decoy run", "lost_marker": "Lost marker", "press_turnover": "Pressing trap",
                "none": "No tactical cause found"}
VERDICT = {"decoy_run": "pulled_by_design", "lost_marker": "switched_off", "press_turnover": "forced_error",
           "outpaced": "outpaced", "none": "unattributed"}


def _clock(f: int) -> str:
    return fmt_clock(f / FPS)


def _r(x, n=1):
    return round(float(x), n)


class ChanceAnalyzer:
    def __init__(self, track: Track, marking: dict, p: Optional[GenesisParams] = None):
        self.tr = track
        self.assign = marking["assign"]
        self.conf = marking["conf"]
        self.p = p or GenesisParams()

    # ------------------------------------------------------------------ context
    def context(self, shot_ev: dict) -> dict:
        tr = self.tr
        f_s = int(shot_ev["frame"])
        S = ID_TO_IDX[shot_ev["player"]]
        X = team_of(S)
        Y = 1 - X
        f_r = reception_frame(tr, S, f_s)
        f_o = f_s
        while f_o > 0 and tr.possession[f_o - 1] == X:
            f_o -= 1
        w0 = max(f_s - int(self.p.lookback_s * FPS), f_o)
        key_pass = None
        for ev in tr.events:
            if ev["type"] == "pass" and ev["frame"] < f_s and ev.get("receiver") == PLAYER_IDS[S] \
                    and team_of(ID_TO_IDX[ev["player"]]) == X and ev["frame"] >= f_o:
                key_pass = ev
        turnover = None
        for (f, nt, pl, typ) in tr.turnovers:
            if nt == X and f_s - int(self.p.lookback_s * FPS) <= f <= f_s:
                turnover = (f, pl, typ)
        return {"f_s": f_s, "S": S, "X": X, "Y": Y, "f_r": f_r, "P_r": tr.pos[f_r, S].copy(), "f_o": f_o,
                "w0": w0, "key_pass": key_pass, "turnover": turnover, "shot": shot_ev}

    # ------------------------------------------------------------------ decoy run
    def decoy(self, c: dict) -> list[dict]:
        tr, p = self.tr, self.p
        f_r, P_r, S, X, Y = c["f_r"], c["P_r"], c["S"], c["X"], c["Y"]
        w0 = c["w0"]
        last = f_r - int(0.4 * FPS)
        w0 = max(w0, f_r - int(p.drag_recent_s * FPS))
        if last - w0 < FPS:
            return []
        Ydef = outfield(Y)
        dist_all = np.linalg.norm(tr.pos[w0:f_r + 1][:, Ydef] - P_r, axis=2)     # (F, 10)
        out = []
        for k, d in enumerate(Ydef):
            ds = dist_all[:, k]
            kn = int(np.argmin(ds[: last - w0 + 1]))
            f_near = w0 + kn
            near, at_rec = ds[kn], ds[-1]
            drag = at_rec - near
            if near > p.protect_r or drag < p.min_drag:
                continue
            rank = int((dist_all[kn] < near - 1e-9).sum())
            if rank >= p.protect_rank:
                continue
            seg = self.assign[f_near:f_r + 1, d]
            cand = seg[(seg >= 0) & (seg != S)]
            cand = cand[np.array([team_of(int(a)) == X for a in cand], dtype=bool)] if len(cand) else cand
            if not len(cand):
                continue
            R = int(np.bincount(cand).argmax())
            frac = float((seg == R).mean())
            if frac < p.follow_frac:
                continue
            follow = float(np.median(np.linalg.norm(tr.pos[f_near:f_r + 1, d] - tr.pos[f_near:f_r + 1, R], axis=1)))
            if follow > p.follow_max:
                continue
            disp_d = tr.pos[f_r, d] - tr.pos[f_near, d]
            disp_R = tr.pos[f_r, R] - tr.pos[f_near, R]
            cosv = float(disp_d @ disp_R / max(np.linalg.norm(disp_d) * np.linalg.norm(disp_R), 1e-6))
            if cosv < 0.3:
                continue
            r0 = max(w0, f_near - FPS)
            run_peak = float(tr.speed[r0:f_r + 1, R].max())
            run_len = float(np.linalg.norm(np.diff(tr.pos[r0:f_r + 1, R], axis=0), axis=1).sum())
            if run_peak < p.run_speed_min or run_len < p.run_len_min:
                continue
            # run onset: back from the half-way point of the drag to where the runner was still slow
            half = np.nonzero(ds[kn:] >= near + drag / 2)[0]
            f_mid = f_near + int(half[0]) if len(half) else f_near
            f_on = f_mid
            while f_on > w0 and tr.speed[f_on - 1, R] >= 3.0:
                f_on -= 1
            sp_near = space_at(P_r, tr.pos[f_near], tr.vel[f_near], X)
            sp_rec = space_at(P_r, tr.pos[f_r], tr.vel[f_r], X)
            pos_cf = tr.pos[f_r].copy()
            vel_cf = tr.vel[f_r].copy()
            pos_cf[d] = tr.pos[f_near, d]
            vel_cf[d] = 0.0
            sp_cf = space_at(P_r, pos_cf, vel_cf, X)
            gain = sp_rec["area_m2"] - sp_cf["area_m2"]
            strength = (drag / p.min_drag) * frac * float(np.clip(0.3 + gain / 60.0, 0.3, 1.5))
            rel = disp_R * (1 if X == 0 else -1)          # displacement in the attacking frame
            direction = "wide" if abs(rel[1]) > abs(rel[0]) else ("deep" if rel[0] < 0 else "in behind")
            out.append({
                "cause": "decoy_run", "strength": strength, "runner": R, "dragged": int(d), "beneficiary": S,
                "f_on": f_on, "f_near": f_near, "f_r": f_r, "drag_m": drag, "near_m": near, "frac": frac,
                "follow_m": follow, "direction": direction, "space_near": sp_near, "space_rec": sp_rec,
                "space_cf": sp_cf, "marker_conf": float(np.mean(self.conf[f_near:f_r + 1, d])),
            })
        return sorted(out, key=lambda x: -x["strength"])

    # ------------------------------------------------------------------ lost marker
    def lost(self, c: dict) -> list[dict]:
        tr, p = self.tr, self.p
        f_r, S, X, Y, w0 = c["f_r"], c["S"], c["X"], c["Y"], c["w0"]
        last = f_r - int(0.5 * FPS)
        if last - w0 < FPS // 2:
            return []
        out = []
        for d in outfield(Y):
            ds = np.linalg.norm(tr.pos[w0:f_r + 1, d] - tr.pos[w0:f_r + 1, S], axis=1)
            d_rec = float(ds[-1])
            if d_rec < p.sep_end_min:
                continue
            close = np.nonzero(ds[: last - w0 + 1] <= p.sep_start_max)[0]
            if not len(close):
                continue
            f_close = w0 + int(close[-1])       # last moment the two were still close
            if f_r - f_close > int(5.0 * FPS):
                continue
            k = f_close - w0
            look = self.assign[max(0, f_close - p.mark_window_s * FPS):f_close + 1, d]
            frac = float((look == S).mean())
            if frac < p.mark_frac:
                continue
            # a man-marking job before the gap opened: within 2.5 m of his man for long enough
            a3 = max(w0 - 2 * FPS, f_close - 3 * FPS, 0)
            tight_s = float((np.linalg.norm(tr.pos[a3:f_close + 1, d] - tr.pos[a3:f_close + 1, S], axis=1)
                             <= p.tight_m).sum()) / FPS
            if tight_s < p.tight_min_s:
                continue
            d_min = float(ds[max(0, k - FPS):k + 1].min())
            f_min = w0 + max(0, k - FPS) + int(np.argmin(ds[max(0, k - FPS):k + 1]))
            # "watching": the marker below jogging pace while his man sprints
            a = max(w0, f_close - FPS)
            watch = (tr.speed[a:f_r + 1, d] < p.watch_def_speed) & (tr.speed[a:f_r + 1, S] > p.watch_att_speed)
            best, run, best_end = 0, 0, a
            for i, wv in enumerate(watch):
                run = run + 1 if wv else 0
                if run > best:
                    best, best_end = run, a + i
            watch_s = best / FPS
            f_on = best_end - best + 1 if best else f_close
            seg_d = np.linalg.norm(np.diff(tr.pos[f_close:f_r + 1, d], axis=0), axis=1).sum()
            seg_a = np.linalg.norm(np.diff(tr.pos[f_close:f_r + 1, S], axis=0), axis=1).sum()
            ratio = float(seg_d / max(seg_a, 1.0))
            pattern = None
            if watch_s >= p.watch_min_s:
                pattern = "lost_marker"
            elif p.outpaced_ratio_min <= ratio < 0.95:
                pattern = "outpaced"
                f_on = f_close
            if pattern is None:
                continue
            f_end = min(f_on + int(1.5 * FPS), f_r)
            v_d = float(tr.speed[f_on:f_end + 1, d].mean())
            v_a = float(tr.speed[f_on:f_end + 1, S].mean())
            if pattern == "outpaced" and v_a < v_d + 0.5:
                continue        # "outran" needs the attacker to actually be the faster one
            P_r = c["P_r"]
            sp_rec = space_at(P_r, tr.pos[f_r], tr.vel[f_r], X)
            # counterfactual: the marker keeps the offset he had to his man when the run began
            pos_cf = tr.pos[f_r].copy()
            vel_cf = tr.vel[f_r].copy()
            pos_cf[d] = tr.pos[f_r, S] + (tr.pos[f_close, d] - tr.pos[f_close, S])
            vel_cf[d] = tr.vel[f_r, S]
            sp_cf = space_at(P_r, pos_cf, vel_cf, X)
            sep = d_rec - d_min
            out.append({
                "cause": pattern,
                "strength": (sep / p.sep_end_min) * min(watch_s / p.watch_min_s, 2.0) if pattern == "lost_marker" else 0.0,
                "attacker": S, "defender": int(d), "f_min": f_min, "f_close": f_close, "f_on": f_on, "f_r": f_r,
                "d_min": d_min, "d_rec": d_rec, "v_def": v_d, "v_att": v_a, "ratio": ratio, "watch_s": watch_s,
                "tight_s": tight_s,
                "space_rec": sp_rec, "space_cf": sp_cf,
                "marker_conf": float(np.mean(self.conf[max(w0, f_close - FPS):f_close + 1, d])),
            })
        return sorted(out, key=lambda x: (x["cause"] != "lost_marker", -x["strength"]))

    # ------------------------------------------------------------------ pressing trap
    def press(self, c: dict) -> list[dict]:
        """High turnover after the carrier was closed down by a pressing unit."""
        tr, p = self.tr, self.p
        if c["turnover"] is None:
            return []
        f_o, winner, typ = c["turnover"]
        X, Y, f_s = c["X"], c["Y"], c["f_s"]
        if (f_s - f_o) / FPS > p.fast_s:
            return []
        loc = tr.ball[f_o]
        # the victim's last spell on the ball before losing it
        f_lost = f_o
        while f_lost > 0 and tr.carrier[f_lost - 1] < 0 and tr.possession[f_lost - 1] == Y:
            f_lost -= 1
        if f_o - f_lost > 2 * FPS:
            return []
        dist_goal_line = float(min(to_rel(tr.ball[max(f_lost - 1, 0)], Y)[0], to_rel(loc, Y)[0]))
        if dist_goal_line > p.high_turnover_m:
            return []
        f_a = max(f_lost - int(p.press_window_s * FPS), 0)
        frames = [f for f in range(f_a, f_lost) if tr.carrier[f] >= 0 and team_of(int(tr.carrier[f])) == Y]
        if len(frames) < 10:
            return []
        Xo = outfield(X)
        contact = np.array([np.linalg.norm(tr.pos[f, Xo] - tr.pos[f, int(tr.carrier[f])], axis=1).min() for f in frames])
        closing = float(contact[: max(1, len(contact) // 3)].max() - contact[-max(1, FPS // 2):].min())
        f_end = frames[-1]
        victim = int(tr.carrier[f_end])
        unit_d = np.linalg.norm(tr.pos[f_end, Xo] - tr.pos[f_end, victim], axis=1)
        unit = [int(j) for j in Xo[np.argsort(unit_d)] if np.linalg.norm(tr.pos[f_end, j] - tr.pos[f_end, victim]) <= p.press_unit_r]
        tight_s = float((contact <= p.press_contact_m).sum()) / FPS
        if contact[-max(1, FPS // 2):].min() > p.press_contact_m or len(unit) < p.press_unit_n:
            return []
        if closing < p.press_closing_m and tight_s < p.press_tight_s:
            return []
        # lane evidence over the closing phase
        per_opp = np.zeros(22, dtype=int)
        n_open, n_tot = [], []
        for f in frames[-FPS:]:
            r = lanes_at(tr.pos[f], int(tr.carrier[f]))
            n_open.append(int((~r["blocked"]).sum()))
            n_tot.append(int(len(r["mates"])))
            per_opp[r["opp"]] += r["blocks"].sum(axis=0)
        n_open = np.array(n_open)
        f_peak = frames[-FPS:][int(np.argmin(n_open))]
        r_peak = lanes_at(tr.pos[f_peak], int(tr.carrier[f_peak]))
        blocks_peak = {int(j): int(n) for j, n in zip(r_peak["opp"], r_peak["blocks"].sum(axis=0)) if n > 0}
        pressers = [j for j in unit if per_opp[j] > 0 or j == unit[0]][:4]
        if winner not in pressers and winner in unit:
            pressers.append(int(winner))
        top = max(pressers, key=lambda j: per_opp[j])
        # which receivers the pressing unit cut off at the peak
        pcols = [k for k, j in enumerate(r_peak["opp"]) if int(j) in pressers]
        cut_to = [int(m) for m, row in zip(r_peak["mates"], r_peak["blocks"]) if row[pcols].any()] if pcols else []
        r_cf = lanes_at(tr.pos[f_peak], int(tr.carrier[f_peak]), exclude_def=(top,))
        # press start: when the nearest presser began closing in (contact still > 2x final)
        k0 = len(contact) - 1
        while k0 > 0 and contact[k0 - 1] < max(2.0 * contact[-1], contact[-1] + 4.0):
            k0 -= 1
        frac_blocked = 1 - n_open.mean() / max(np.mean(n_tot), 1)
        strength = max(closing / p.press_closing_m, tight_s / p.press_tight_s) * (0.5 + frac_blocked) \
            * min(len(unit) / p.press_unit_n, 1.5)
        return [{
            "cause": "press_turnover", "strength": float(strength), "winner": int(winner), "victim": victim,
            "pressers": pressers, "lanes_blocked": {int(j): int(per_opp[j]) for j in pressers},
            "blocks_peak": blocks_peak, "f_press": frames[k0], "f_peak": f_peak, "f_lost": f_lost, "f_o": f_o,
            "turnover_type": typ, "dist_goal_line": dist_goal_line, "open_min": int(n_open.min()),
            "open_peak": int((~r_peak["blocked"]).sum()), "open_cf": int((~r_cf["blocked"]).sum()),
            "low_s": float(len(frames) - k0) / FPS, "lanes_total": int(len(r_peak["mates"])), "turnover_loc": loc.copy(),
            "closing_m": closing, "contact_m": float(contact[-max(1, FPS // 2):].min()), "unit": unit,
            "contact_start_m": float(contact[: max(1, len(contact) // 3)].max()), "cut_to": cut_to,
        }]

    # ------------------------------------------------------------------ decision
    def analyze(self, shot_ev: dict) -> dict:
        c = self.context(shot_ev)
        cands = self.press(c) + self.decoy(c) + self.lost(c)
        tactical = [x for x in cands if x["cause"] in ("decoy_run", "lost_marker", "press_turnover")]
        tactical.sort(key=lambda x: -x["strength"])
        outpaced = [x for x in cands if x["cause"] == "outpaced"]
        primary = tactical[0] if tactical else None
        return build_evidence(self.tr, c, primary, tactical[1:], outpaced[0] if outpaced and not primary else None,
                              self.assign, self.conf)


# ---------------------------------------------------------------------- evidence pack
def build_evidence(tr: Track, c: dict, primary, secondary, outpaced, assign, conf) -> dict:
    S, X, f_s, f_r = c["S"], c["X"], c["f_s"], c["f_r"]
    shot = c["shot"]
    outcome = shot.get("outcome", "shot")
    goal_dist = float(np.linalg.norm(tr.pos[f_s, S] - opp_goal(X)))
    cause = primary["cause"] if primary else "none"
    verdict = VERDICT["outpaced"] if outpaced else VERDICT[cause]
    chain: list[dict] = []
    cf: dict = {}
    frames: list[int] = []
    marker_role = None
    kp = c["key_pass"]
    if primary and cause == "decoy_run":
        R, D = primary["runner"], primary["dragged"]
        chain.append({"t": f"{_clock(primary['f_on'])}-{_clock(primary['f_r'])}", "actor": PLAYER_IDS[R],
                      "action": f"decoy run ({primary['direction']})", "dragged": PLAYER_IDS[D],
                      "dragged_m": _r(primary["drag_m"]),
                      "space_m2": [_r(primary["space_near"]["area_m2"], 0), _r(primary["space_rec"]["area_m2"], 0)]})
        if kp is not None:
            chain.append({"t": _clock(kp["frame"]), "actor": kp["player"], "action": "pass into the vacated space",
                          "receiver": PLAYER_IDS[S]})
        chain.append({"t": _clock(f_r), "actor": PLAYER_IDS[S], "action": "received in vacated space",
                      "nearest_defender_m": _r(primary["space_rec"]["nearest_def_m"]),
                      "counterfactual_space_m2": _r(primary["space_cf"]["area_m2"], 0),
                      "counterfactual_nearest_defender_m": _r(primary["space_cf"]["nearest_def_m"])})
        cf = {"assumption": f"{PLAYER_IDS[D]} stays where he was before {PLAYER_IDS[R]}'s run",
              "space_m2_actual": _r(primary["space_rec"]["area_m2"], 0),
              "space_m2_counterfactual": _r(primary["space_cf"]["area_m2"], 0),
              "space_m2_created": _r(primary["space_rec"]["area_m2"] - primary["space_cf"]["area_m2"], 0),
              "nearest_defender_m_actual": _r(primary["space_rec"]["nearest_def_m"]),
              "nearest_defender_m_counterfactual": _r(primary["space_cf"]["nearest_def_m"])}
        marker_role = {"player": PLAYER_IDS[D], "assignment": PLAYER_IDS[R], "confidence": _r(primary["marker_conf"], 2),
                       "share_of_run_assigned": _r(primary["frac"], 2), "median_gap_m": _r(primary["follow_m"])}
        frames = [primary["f_on"], primary["f_near"], f_r, f_s]
        players = {"runner": PLAYER_IDS[R], "dragged": PLAYER_IDS[D], "beneficiary": PLAYER_IDS[S]}
        t_start = primary["f_on"]
    elif primary and cause == "lost_marker":
        D = primary["defender"]
        chain.append({"t": _clock(primary["f_min"]), "actor": PLAYER_IDS[D], "action": "marking",
                      "marking": PLAYER_IDS[S], "gap_m": _r(primary["d_min"])})
        chain.append({"t": f"{_clock(primary['f_on'])}-{_clock(f_r)}", "actor": PLAYER_IDS[D],
                      "action": "stopped tracking", "watching_s": _r(primary["watch_s"]),
                      "defender_speed_ms": _r(primary["v_def"]),
                      "attacker_speed_ms": _r(primary["v_att"]), "gap_m": [_r(primary["d_min"]), _r(primary["d_rec"])]})
        if kp is not None:
            chain.append({"t": _clock(kp["frame"]), "actor": kp["player"], "action": "pass into the gap",
                          "receiver": PLAYER_IDS[S]})
        chain.append({"t": _clock(f_r), "actor": PLAYER_IDS[S], "action": "received unmarked",
                      "nearest_defender_m": _r(primary["space_rec"]["nearest_def_m"])})
        cf = {"assumption": f"{PLAYER_IDS[D]} keeps the {_r(primary['d_min'])} m gap he had when the run began",
              "nearest_defender_m_actual": _r(primary["space_rec"]["nearest_def_m"]),
              "nearest_defender_m_counterfactual": _r(primary["space_cf"]["nearest_def_m"]),
              "space_m2_actual": _r(primary["space_rec"]["area_m2"], 0),
              "space_m2_counterfactual": _r(primary["space_cf"]["area_m2"], 0)}
        marker_role = {"player": PLAYER_IDS[D], "assignment": PLAYER_IDS[S], "confidence": _r(primary["marker_conf"], 2)}
        frames = [primary["f_min"], primary["f_on"], f_r, f_s]
        players = {"attacker": PLAYER_IDS[S], "defender": PLAYER_IDS[D]}
        t_start = primary["f_on"]
    elif primary and cause == "press_turnover":
        pr = primary
        chain.append({"t": f"{_clock(pr['f_press'])}-{_clock(pr['f_lost'])}",
                      "actor": ", ".join(PLAYER_IDS[j] for j in pr["pressers"]), "action": "press",
                      "victim": PLAYER_IDS[pr["victim"]], "open_lanes_min": pr["open_min"],
                      "lanes_total": pr["lanes_total"], "press_duration_s": _r(pr["low_s"]),
                      "nearest_presser_m": [_r(pr["contact_start_m"]), _r(pr["contact_m"])],
                      "lanes_cut_to": [PLAYER_IDS[j] for j in pr["cut_to"]],
                      "lanes_blocked_by": {PLAYER_IDS[j]: n for j, n in pr["blocks_peak"].items()}})
        chain.append({"t": _clock(pr["f_o"]), "actor": PLAYER_IDS[pr["winner"]], "action": f"won the ball ({pr['turnover_type']})",
                      "metres_from_their_goal_line": _r(pr["dist_goal_line"])})
        chain.append({"t": _clock(f_s), "actor": PLAYER_IDS[S], "action": "shot",
                      "seconds_after_turnover": _r((f_s - pr["f_o"]) / FPS)})
        top = pr["pressers"][0]
        cf = {"assumption": f"{PLAYER_IDS[top]} is removed from the press at its peak",
              "open_lanes_actual": pr["open_peak"], "open_lanes_counterfactual": pr["open_cf"]}
        frames = [pr["f_press"], pr["f_peak"], pr["f_o"], f_s]
        players = {"ball_winner": PLAYER_IDS[pr["winner"]], "pressers": [PLAYER_IDS[j] for j in pr["pressers"]]}
        t_start = pr["f_press"]
    else:
        sp = space_at(tr.pos[f_s, S], tr.pos[f_s], tr.vel[f_s], X)
        if outpaced:
            D = outpaced["defender"]
            chain.append({"t": f"{_clock(outpaced['f_on'])}-{_clock(f_r)}", "actor": PLAYER_IDS[S],
                          "action": "outran his marker", "marker": PLAYER_IDS[D],
                          "defender_speed_ms": _r(outpaced["v_def"]), "attacker_speed_ms": _r(outpaced["v_att"]),
                          "gap_m": [_r(outpaced["d_min"]), _r(outpaced["d_rec"])]})
            marker_role = {"player": PLAYER_IDS[D], "assignment": PLAYER_IDS[S], "confidence": _r(outpaced["marker_conf"], 2)}
            frames = [outpaced["f_min"], outpaced["f_on"], f_r, f_s]
            players = {"attacker": PLAYER_IDS[S], "defender": PLAYER_IDS[D]}
            t_start = outpaced["f_on"]
        else:
            frames = [max(c["w0"], f_r - FPS), f_r, f_s]
            players = {"shooter": PLAYER_IDS[S]}
            t_start = None
        chain.append({"t": _clock(f_s), "actor": PLAYER_IDS[S], "action": "shot",
                      "distance_to_goal_m": _r(goal_dist), "nearest_defender_m": _r(sp["nearest_def_m"])})
    if cause != "none" and not any(st.get("action") == "shot" for st in chain):
        chain.append({"t": _clock(f_s), "actor": PLAYER_IDS[S], "action": "shot", "distance_to_goal_m": _r(goal_dist)})
    strength = primary["strength"] if primary else 0.0
    ev = {
        "moment_id": f"{'goal' if outcome == 'goal' else 'shot'}_{PLAYER_IDS[S]}_{_clock(f_s).replace(':', '_')}",
        "type": "goal" if outcome == "goal" else "shot",
        "outcome": outcome,
        "t": _clock(f_s), "t_s": _r(f_s / FPS, 2), "frame": f_s,
        "team": "AB"[X], "shooter": PLAYER_IDS[S],
        "cause": cause, "cause_label": CAUSE_LABELS[cause],
        "confidence": _r(1 - np.exp(-strength), 2),
        "verdict_type": verdict,
        "players": players,
        "t_start": None if t_start is None else _r(t_start / FPS, 2),
        "chain": chain,
        "counterfactual": cf,
        "marker_role": marker_role,
        "secondary": [{"cause": s["cause"], "strength": _r(s["strength"], 2),
                       **({"runner": PLAYER_IDS[s["runner"]], "dragged": PLAYER_IDS[s["dragged"]]} if s["cause"] == "decoy_run" else {}),
                       **({"defender": PLAYER_IDS[s["defender"]]} if s["cause"] == "lost_marker" else {}),
                       **({"ball_winner": PLAYER_IDS[s["winner"]]} if s["cause"] == "press_turnover" else {})}
                      for s in secondary[:2]],
        "evidence_frames": [int(f) for f in frames],
        "clip": {"from_frame": int(max(0, (min(frames) if frames else f_s) - 2 * FPS)), "to_frame": int(min(tr.T - 1, f_s + FPS))},
        "shot_distance_m": _r(goal_dist),
    }
    return ev
