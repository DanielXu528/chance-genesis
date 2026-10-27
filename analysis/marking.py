"""Marking inference: who is marking whom, from positions alone.

Per frame, a cost matrix between the defending team's outfield players and the attacking
team's outfield players is solved with the Hungarian algorithm
(scipy.optimize.linear_sum_assignment). Ten dummy "zone" columns let any defender be
assigned to no one. Raw per-frame matches flicker, so a hysteresis rule confirms a
change only when the new match has persisted for HOLD frames and is clearly cheaper.

Cost of defender d marking attacker a
  |p_d - m_a|                 distance to the goal-side marking point m_a
  + W_ANG * angle term        penalty for standing on the wrong side of the attacker
  + W_VEL * |v_d - v_a|       a marker moves with his man
  - W_CLOSE * closing speed   a defender sprinting onto a man is picking him up
  gated to BIG beyond GATE metres.

Defaults were tuned on development seeds (100-199) only; evaluation uses seeds >= 10000.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from analysis.preprocess import Track
from common.pitch import FPS, NOT_DEFENDING, ZONE, outfield, own_goal


@dataclass
class MarkingParams:
    goal_side: float = 1.5
    w_ang: float = 2.0
    w_vel: float = 0.2
    w_close: float = 0.6        # reward for closing in on the marking point
    zone_cost: float = 6.5
    gate: float = 12.0
    hold_frames: int = 5        # 0.2 s
    min_gain: float = 0.3       # metres of cost improvement needed to switch
    # HMM / Viterbi smoother (offline: uses frames after t as well as before)
    hmm_sigma: float = 1.5      # emission log-likelihood = -cost / sigma
    hmm_switch: float = 0.01    # per-frame probability of changing assignment
    hmm_raw_bonus: float = 2.0  # extra log-likelihood for the state the Hungarian picked


BIG = 1e4


def cost_tensor(track: Track, def_team: int, p: MarkingParams, frames=None) -> np.ndarray:
    """(F, 10, 10) costs for def_team's outfield (rows) vs the other team's outfield (cols)."""
    sl = slice(None) if frames is None else frames
    D = outfield(def_team)
    A = outfield(1 - def_team)
    Pd, Pa = track.pos[sl][:, D], track.pos[sl][:, A]
    Vd, Va = track.vel[sl][:, D], track.vel[sl][:, A]
    g = own_goal(def_team)
    ua = g[None, None, :] - Pa
    ua /= np.maximum(np.linalg.norm(ua, axis=2, keepdims=True), 1e-6)
    Ma = Pa + ua * p.goal_side
    diff = Pd[:, :, None, :] - Ma[:, None, :, :]
    d_ideal = np.linalg.norm(diff, axis=3)
    rel = Pd[:, :, None, :] - Pa[:, None, :, :]
    raw = np.linalg.norm(rel, axis=3)
    cos = np.einsum("fdak,fak->fda", rel, ua) / np.maximum(raw, 1e-6)
    ang = (1 - cos) / 2 * np.minimum(raw, 6.0) / 6.0
    rv = Vd[:, :, None, :] - Va[:, None, :, :]
    dv = np.linalg.norm(rv, axis=3).clip(0, 6)
    closing = (-np.einsum("fdak,fdak->fda", diff, rv) / np.maximum(d_ideal, 1e-6)).clip(0, 6)
    cost = d_ideal + p.w_ang * ang + p.w_vel * dv - p.w_close * closing * (d_ideal > 2.0)
    cost[raw > p.gate] = BIG
    return cost


def infer_marking(track: Track, defending: np.ndarray, p: MarkingParams | None = None,
                  method: str = "viterbi") -> dict:
    """Returns per-frame raw and smoothed assignments (T, 22) and confidence (T, 22).
    Values: attacker player index, ZONE (-1), or NOT_DEFENDING (-2).

    method="hysteresis": causal, a change is confirmed after hold_frames (live use).
    method="viterbi":    HMM smoothing over each defending spell (retrospective use,
                         e.g. explaining a chance after the shot)."""
    p = p or MarkingParams()
    T = track.T
    raw_out = np.full((T, 22), NOT_DEFENDING, dtype=np.int16)
    out = np.full((T, 22), NOT_DEFENDING, dtype=np.int16)
    cost_out = np.full((T, 22), np.nan, dtype=np.float32)
    costs = {t: np.concatenate([cost_tensor(track, t, p, slice(a, min(a + 20000, T)))
                                for a in range(0, T, 20000)]) for t in (0, 1)}
    zone_block = np.full((10, 10), p.zone_cost)
    state = None
    prev_team = -1
    for f in range(T):
        dt = int(defending[f])
        if dt < 0:
            prev_team = -1
            continue
        C = costs[dt][f]
        full = np.hstack([C, zone_block])
        rows, cols = linear_sum_assignment(full)
        D = outfield(dt)
        A = outfield(1 - dt)
        raw = np.where(cols < 10, A[np.minimum(cols, 9)], ZONE)
        raw_out[f, D] = raw
        if dt != prev_team:
            state = {"cur": raw.copy(), "pend": raw.copy(), "cnt": np.zeros(10, dtype=int)}
            prev_team = dt
        cur, pend, cnt = state["cur"], state["pend"], state["cnt"]
        for k in range(10):
            r = raw[k]
            if r == cur[k]:
                cnt[k] = 0
                continue
            if r == pend[k]:
                cnt[k] += 1
            else:
                pend[k] = r
                cnt[k] = 1
            if cnt[k] >= p.hold_frames:
                c_cur = p.zone_cost if cur[k] < 0 else C[k, cur[k] - A[0]]
                c_new = p.zone_cost if r < 0 else C[k, r - A[0]]
                if c_cur - c_new > p.min_gain or c_cur >= BIG:
                    cur[k] = r
                    cnt[k] = 0
        # a confirmed man-assignment that has become physically impossible is dropped at once
        for k in range(10):
            if cur[k] >= 0 and C[k, cur[k] - A[0]] >= BIG:
                cur[k] = ZONE
        out[f, D] = cur
        cost_out[f, D] = [p.zone_cost if c < 0 else C[k, c - A[0]] for k, c in enumerate(cur)]
    hyst = out.copy()
    if method == "viterbi":
        out = _viterbi_all(raw_out, costs, defending, p)
        cost_out = _cost_of(out, costs, defending, p)
    stab = _stability(raw_out, out)
    conf = np.where(out >= 0, 0.5 * np.clip((p.zone_cost - cost_out) / p.zone_cost, 0, 1) + 0.5 * stab,
                    np.where(out == ZONE, stab, 0.0)).astype(np.float32)
    return {"raw": raw_out, "assign": out, "hysteresis": hyst, "conf": conf, "cost": cost_out}


def _segments(defending: np.ndarray):
    T = len(defending)
    f = 0
    while f < T:
        if defending[f] < 0:
            f += 1
            continue
        g = f
        while g < T and defending[g] == defending[f]:
            g += 1
        yield f, g, int(defending[f])
        f = g


def _viterbi_all(raw_out, costs, defending, p: MarkingParams) -> np.ndarray:
    out = np.full_like(raw_out, NOT_DEFENDING)
    for f0, f1, dt in _segments(defending):
        D = outfield(dt)
        A0 = outfield(1 - dt)[0]
        C = costs[dt][f0:f1]                                   # (F, 10, 10)
        F = f1 - f0
        em = np.empty((F, 10, 11))
        em[:, :, :10] = -np.minimum(C, BIG) / p.hmm_sigma
        em[:, :, 10] = -p.zone_cost / p.hmm_sigma
        r = raw_out[f0:f1][:, D]
        ridx = np.where(r >= 0, r - A0, 10)
        np.put_along_axis(em, ridx[:, :, None], np.take_along_axis(em, ridx[:, :, None], 2) + p.hmm_raw_bonus, 2)
        ls, lw = np.log(1 - p.hmm_switch), np.log(p.hmm_switch / 10)
        dp = em[0].copy()
        ptr = np.zeros((F, 10, 11), dtype=np.int8)
        ar = np.arange(11)
        for t in range(1, F):
            best = dp.argmax(axis=1)
            bval = dp.max(axis=1, keepdims=True) + lw
            stay = dp + ls
            use_stay = stay >= bval
            ptr[t] = np.where(use_stay, ar[None, :], best[:, None])
            dp = np.where(use_stay, stay, bval) + em[t]
        path = np.zeros((F, 10), dtype=np.int64)
        path[-1] = dp.argmax(axis=1)
        for t in range(F - 1, 0, -1):
            path[t - 1] = ptr[t][np.arange(10), path[t]]
        out[f0:f1][:, D] = np.where(path < 10, path + A0, ZONE)
    return out


def _cost_of(out, costs, defending, p: MarkingParams) -> np.ndarray:
    cost_out = np.full(out.shape, np.nan, dtype=np.float32)
    for f0, f1, dt in _segments(defending):
        D = outfield(dt)
        A0 = outfield(1 - dt)[0]
        a = out[f0:f1][:, D]
        C = costs[dt][f0:f1]
        idx = np.where(a >= 0, a - A0, 0)
        c = np.take_along_axis(C, idx[:, :, None], 2)[:, :, 0]
        cost_out[f0:f1, D[0]:D[-1] + 1] = np.where(a >= 0, c, p.zone_cost)
    return cost_out


def _stability(raw: np.ndarray, out: np.ndarray, win: int = FPS) -> np.ndarray:
    eq = (raw == out).astype(np.float32)
    c = np.cumsum(np.vstack([np.zeros((1, eq.shape[1]), np.float32), eq]), axis=0)
    idx = np.arange(eq.shape[0])
    lo = np.maximum(idx - win + 1, 0)
    return (c[idx + 1] - c[lo]) / (idx + 1 - lo)[:, None]


def marker_of(assign: np.ndarray, f: int, attacker: int) -> int:
    ds = np.nonzero(assign[f] == attacker)[0]
    return int(ds[0]) if len(ds) else -1
