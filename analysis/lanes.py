"""Passing-lane availability under pressure.

For the ball carrier and each team-mate, sample N_PTS points along the straight lane.
The ball reaches point k after |p_k - carrier| / BALL_SPEED seconds. Opponent j reaches it
after REACT + |p_k - p_j| / RUN_SPEED. If any opponent can be at any point before the
ball, the lane is blocked, and that opponent is credited with blocking it.
"""
from __future__ import annotations

import numpy as np

from analysis.preprocess import Track
from common.pitch import team_of

BALL_SPEED = 15.0
REACT = 0.3
RUN_SPEED = 7.0
N_PTS = 20
MIN_LANE = 1.5      # ignore the first 1.5 m: a pass leaves the foot faster than anyone reacts
MAX_LANE = 30.0     # only short/medium options count as passing lanes under pressure


def lanes_at(pos: np.ndarray, carrier: int, exclude_def: tuple = ()) -> dict:
    """Lane state for one frame. Returns team-mates, blocked mask, and a (mates, opponents)
    matrix saying which opponent blocks which lane."""
    team = team_of(carrier)
    mates = np.array([i for i in (range(11) if team == 0 else range(11, 22))
                      if i != carrier and np.linalg.norm(pos[i] - pos[carrier]) <= MAX_LANE])
    if len(mates) == 0:
        opp = np.array([j for j in (range(11, 22) if team == 0 else range(11)) if j not in exclude_def])
        return {"mates": mates, "opp": opp, "blocked": np.zeros(0, bool), "blocks": np.zeros((0, len(opp)), bool)}
    opp = np.array([j for j in (range(11, 22) if team == 0 else range(11)) if j not in exclude_def])
    pc = pos[carrier]
    s = np.linspace(1.0 / N_PTS, 1.0, N_PTS)
    seg = pos[mates] - pc
    pts = pc + seg[:, None, :] * s[None, :, None]                      # (M, N, 2)
    dist_c = np.linalg.norm(seg, axis=1)[:, None] * s[None, :]          # (M, N)
    tb = dist_c / BALL_SPEED
    td = REACT + np.linalg.norm(pts[:, :, None, :] - pos[opp][None, None, :, :], axis=3) / RUN_SPEED
    can = (td < tb[:, :, None]) & (dist_c[:, :, None] > MIN_LANE)
    blocks = can.any(axis=1)                                            # (M, O)
    return {"mates": mates, "opp": opp, "blocked": blocks.any(axis=1), "blocks": blocks}


def lanes_series(track: Track, frames: np.ndarray) -> dict:
    """Open-lane count and per-opponent blocked-lane counts for the given frames
    (only frames with a known carrier are evaluated; others get -1)."""
    F = len(frames)
    n_open = np.full(F, -1, dtype=np.int16)
    per_opp = np.zeros((F, 22), dtype=np.int16)
    open_mask = np.zeros((F, 22), dtype=bool)
    for k, f in enumerate(frames):
        c = int(track.carrier[f])
        if c < 0:
            continue
        r = lanes_at(track.pos[f], c)
        n_open[k] = int((~r["blocked"]).sum())
        per_opp[k, r["opp"]] = r["blocks"].sum(axis=0)
        open_mask[k, r["mates"]] = ~r["blocked"]
    return {"frames": frames, "n_open": n_open, "per_opp": per_opp, "open": open_mask}
