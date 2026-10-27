"""Pitch geometry and player indexing shared by every layer.

Coordinates are metres on a 105 x 68 pitch. Team A attacks +x, team B attacks -x.
Players are indexed 0..21: A1..A11 are 0..10, B1..B11 are 11..21 (shirt 1 is the keeper).
"""
from __future__ import annotations

import numpy as np

L = 105.0
W = 68.0
FPS = 25
DT = 1.0 / FPS
GOAL_HALF_WIDTH = 3.66
N_PLAYERS = 22
TEAM_NAMES = ("A", "B")

PLAYER_IDS = [f"A{i}" for i in range(1, 12)] + [f"B{i}" for i in range(1, 12)]
ID_TO_IDX = {pid: i for i, pid in enumerate(PLAYER_IDS)}

# Label codes used in per-frame marking arrays.
ZONE = -1          # defender is holding a zone, not marking a player
NOT_DEFENDING = -2  # player's team is in possession, or player is a goalkeeper


def team_of(idx: int) -> int:
    return 0 if idx < 11 else 1


def team_slice(team: int) -> slice:
    return slice(0, 11) if team == 0 else slice(11, 22)


def outfield(team: int) -> np.ndarray:
    """Indices of a team's ten outfield players."""
    base = 0 if team == 0 else 11
    return np.arange(base + 1, base + 11)


def keeper(team: int) -> int:
    return 0 if team == 0 else 11


def attack_sign(team: int) -> float:
    return 1.0 if team == 0 else -1.0


def to_rel(xy: np.ndarray, team: int) -> np.ndarray:
    """Absolute coordinates -> the team's attacking frame (x grows toward the opponent goal)."""
    xy = np.asarray(xy, dtype=float)
    if team == 0:
        return xy.copy()
    out = xy.copy()
    out[..., 0] = L - xy[..., 0]
    out[..., 1] = W - xy[..., 1]
    return out


from_rel = to_rel  # the transform is its own inverse


def opp_goal(team: int) -> np.ndarray:
    return np.array([L, W / 2]) if team == 0 else np.array([0.0, W / 2])


def own_goal(team: int) -> np.ndarray:
    return opp_goal(1 - team)


def fmt_clock(t: float) -> str:
    """Seconds -> 'mm:ss'."""
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{int(t % 60):02d}"
