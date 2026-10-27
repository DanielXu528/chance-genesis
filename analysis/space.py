"""Space around a point: who can get there first.

Time to reach a grid cell = reaction time + distance from the player's position projected
forward by its velocity over the reaction time, at a fixed top speed. A cell is controlled
by whichever team arrives first. This is a simple time-to-intercept dominant region, a
cheap stand-in for a full pitch-control model.
"""
from __future__ import annotations

import numpy as np

from common.pitch import L, W

REACT = 0.3
VMAX = 7.0
RADIUS = 9.0
RES = 0.75


def arrival_times(points: np.ndarray, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    start = pos + vel * REACT
    return REACT + np.linalg.norm(points[:, None, :] - start[None, :, :], axis=2) / VMAX


def disk(center: np.ndarray, radius: float = RADIUS, res: float = RES) -> np.ndarray:
    r = np.arange(-radius, radius + 1e-9, res)
    gx, gy = np.meshgrid(r, r)
    m = gx ** 2 + gy ** 2 <= radius ** 2
    pts = np.stack([gx[m], gy[m]], axis=1) + center
    inside = (pts[:, 0] >= 0) & (pts[:, 0] <= L) & (pts[:, 1] >= 0) & (pts[:, 1] <= W)
    return pts[inside]


def space_at(center: np.ndarray, pos: np.ndarray, vel: np.ndarray, att_team: int,
             radius: float = RADIUS) -> dict:
    """Attacking-team-controlled area (m^2) within `radius` of `center`, and the nearest
    defender's distance to `center`."""
    pts = disk(np.asarray(center, float), radius)
    t = arrival_times(pts, pos, vel)
    att = slice(0, 11) if att_team == 0 else slice(11, 22)
    dfn = slice(11, 22) if att_team == 0 else slice(0, 11)
    ctrl = t[:, att].min(axis=1) < t[:, dfn].min(axis=1)
    nd = np.linalg.norm(pos[dfn] - center, axis=1)
    return {"area_m2": float(ctrl.sum() * RES * RES), "nearest_def_m": float(nd.min()),
            "nearest_def": int(np.argmin(nd)) + (11 if att_team == 0 else 0)}


def control_grid(pos: np.ndarray, vel: np.ndarray, center: np.ndarray, radius: float = 14.0,
                 res: float = 1.0) -> dict:
    """Cells around `center` with the team that controls each (for drawing)."""
    pts = disk(np.asarray(center, float), radius, res)
    t = arrival_times(pts, pos, vel)
    team = (t[:, 11:].min(axis=1) < t[:, :11].min(axis=1)).astype(int)
    return {"points": np.round(pts, 2).tolist(), "team": team.tolist(), "res": res}
