"""Offside line: the second-last defender (keeper included), or the ball if it is nearer
the goal line; never inside the attacking team's own half."""
from __future__ import annotations

import numpy as np

from common.pitch import L, team_slice


def offside_line(pos: np.ndarray, ball: np.ndarray, att_team: int) -> float:
    """Absolute x of the offside line for att_team at one frame."""
    dfn = pos[team_slice(1 - att_team)][:, 0]
    if att_team == 0:
        line = max(np.sort(dfn)[-2], ball[0], L / 2)
    else:
        line = min(np.sort(dfn)[1], ball[0], L / 2)
    return float(line)


def offside_series(pos: np.ndarray, ball: np.ndarray, possession: np.ndarray) -> np.ndarray:
    out = np.full(len(possession), np.nan)
    for t in (0, 1):
        m = possession == t
        if not m.any():
            continue
        d = np.sort(pos[m][:, (slice(11, 22) if t == 0 else slice(0, 11)), 0], axis=1)
        if t == 0:
            out[m] = np.maximum.reduce([d[:, -2], ball[m, 0], np.full(m.sum(), L / 2)])
        else:
            out[m] = np.minimum.reduce([d[:, 1], ball[m, 0], np.full(m.sum(), L / 2)])
    return out
