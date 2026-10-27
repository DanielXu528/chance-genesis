"""Preprocessing: smoothing, velocities, and ball possession reconstructed from the event
feed plus tracking. Works on observations only (positions, ball, events, lineup)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter

from common.pitch import FPS, ID_TO_IDX, team_of

POS_WIN = 13   # 0.52 s Savitzky-Golay window for positions
VEL_WIN = 25   # 1.0 s window for velocities (noise on 25 Hz positions is large)


@dataclass
class Track:
    pos: np.ndarray          # (T, 22, 2) smoothed
    vel: np.ndarray          # (T, 22, 2)
    speed: np.ndarray        # (T, 22)
    ball: np.ndarray         # (T, 2)
    possession: np.ndarray   # (T,) team in possession 0/1, -1 when the ball is dead
    carrier: np.ndarray      # (T,) player index on the ball, -1 in flight / dead
    events: list
    meta: dict
    fps: int = FPS
    turnovers: list = field(default_factory=list)   # [(frame, new_team, player_idx, type)]

    @property
    def T(self) -> int:
        return self.pos.shape[0]


def _smooth(x: np.ndarray, win: int, deriv: int = 0) -> np.ndarray:
    T = x.shape[0]
    w = min(win, T if T % 2 == 1 else T - 1)
    if w < 5:
        return x.copy() if deriv == 0 else np.gradient(x, axis=0) * FPS
    return savgol_filter(x, w, 2, deriv=deriv, delta=1.0 / FPS, axis=0)


def preprocess(obs: dict) -> Track:
    raw = np.asarray(obs["pos"], dtype=np.float64)
    T = raw.shape[0]
    pos = _smooth(raw, POS_WIN)
    vel = _smooth(raw, VEL_WIN, deriv=1)
    ball = _smooth(np.asarray(obs["ball"], dtype=np.float64), 5)
    events = sorted(obs["events"], key=lambda e: e["frame"])
    possession, carrier, turnovers = _possession(events, pos, ball, T)
    return Track(pos=pos, vel=vel, speed=np.linalg.norm(vel, axis=2), ball=ball, possession=possession,
                 carrier=carrier, events=events, meta=obs["meta"], turnovers=turnovers)


def _possession(events, pos, ball, T):
    """Possession is driven by on-ball events. A pass leaves the ball in flight until the
    next on-ball actor is within 1.5 m of it (or acts)."""
    possession = np.full(T, -1, dtype=np.int8)
    carrier = np.full(T, -1, dtype=np.int16)
    turnovers = []
    onball = [e for e in events if e["type"] in ("pass", "shot", "tackle", "interception", "restart")]
    team = -1
    for k, ev in enumerate(onball):
        f = int(ev["frame"])
        p = ID_TO_IDX[ev["player"]]
        f_next = int(onball[k + 1]["frame"]) if k + 1 < len(onball) else T
        new_team = team_of(p)
        if ev["type"] in ("tackle", "interception", "restart") or team == -1:
            if new_team != team and team != -1 and ev["type"] != "restart":
                turnovers.append((f, new_team, p, ev["type"]))
            team = new_team
        if ev["type"] == "shot":
            # ball is dead from the shot until the next restart / on-ball event
            possession[f:f_next] = team
            carrier[f:f_next] = -1
            nxt = onball[k + 1] if k + 1 < len(onball) else None
            dead_from = min(f + int(1.5 * FPS), f_next)
            possession[dead_from:f_next] = -1
            team = -1 if nxt is None or nxt["type"] == "restart" else team
            continue
        possession[f:f_next] = team
        if ev["type"] == "pass":
            carrier[f:f_next] = -1
            if k + 1 < len(onball):
                nxt = onball[k + 1]
                q = ID_TO_IDX[nxt["player"]]
                if team_of(q) != team:
                    # lost after reception (tackle): the intended receiver had it in between
                    q = ID_TO_IDX[ev["receiver"]] if nxt["type"] == "tackle" and ev.get("receiver") else -1
                if q >= 0:
                    d = np.linalg.norm(ball[f + 3:f_next] - pos[f + 3:f_next, q], axis=1) if f + 3 < f_next else np.array([])
                    hit = np.nonzero(d < 1.5)[0]
                    f_rec = f + 3 + int(hit[0]) if len(hit) else f_next
                    carrier[f_rec:f_next] = q
        else:
            carrier[f:f_next] = p
    # before the first event: assume the first actor's team had the ball
    if onball:
        f0 = int(onball[0]["frame"])
        possession[:f0] = team_of(ID_TO_IDX[onball[0]["player"]])
        carrier[:f0] = ID_TO_IDX[onball[0]["player"]]
    return possession, carrier, turnovers


def reception_frame(track: Track, player: int, before: int) -> int:
    """First frame of the player's current spell on the ball, searching back from `before`."""
    f = before
    while f > 0 and track.carrier[f - 1] == player:
        f -= 1
    return f
