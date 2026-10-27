"""Phase of play per frame, from the in-possession team's point of view."""
from __future__ import annotations

import numpy as np

from analysis.preprocess import Track
from common.pitch import FPS, ID_TO_IDX, L

DEAD, ESTABLISHED, TRANSITION, SET_PIECE = 0, 1, 2, 3
PHASE_NAMES = {DEAD: "dead_ball", ESTABLISHED: "established", TRANSITION: "transition", SET_PIECE: "set_piece"}
THIRD_NAMES = {0: "defensive_third", 1: "middle_third", 2: "final_third"}
TRANSITION_S = 5.0


def phases(track: Track) -> dict:
    T = track.T
    poss = track.possession
    phase = np.where(poss >= 0, ESTABLISHED, DEAD).astype(np.int8)
    # transition: within 5 s after a turnover
    for f, _team, _p, _typ in track.turnovers:
        phase[f:min(T, f + int(TRANSITION_S * FPS))] = np.where(poss[f:min(T, f + int(TRANSITION_S * FPS))] >= 0,
                                                                 TRANSITION, DEAD)
    for ev in track.events:
        if ev["type"] == "restart":
            f = int(ev["frame"])
            sl = slice(f, min(T, f + int(TRANSITION_S * FPS)))
            phase[sl] = np.where(poss[sl] >= 0, SET_PIECE, DEAD)
    bx = track.ball[:, 0]
    rel_x = np.where(poss == 1, L - bx, bx)
    third = np.digitize(rel_x, [35.0, 70.0]).astype(np.int8)
    third[poss < 0] = -1
    defending = np.where(poss >= 0, 1 - poss, -1).astype(np.int8)
    return {"phase": phase, "third": third, "defending": defending}
