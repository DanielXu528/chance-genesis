"""Chance Genesis API + replay server (FastAPI).

  GET  /                      single-page studio UI
  GET  /api/info              match metadata, model backend, list of chances
  GET  /api/moment/{id}       evidence pack + verified narration (runs the agent workflow once, then cached)
  GET  /api/clip/{id}         frames + overlays for the replay of one chance
  GET  /api/frames            frames for a time range (from, to, step)
  GET  /api/match/meta|track|state   the whole match as compact binary arrays (scrubbable match view)
  GET  /api/summary           Tactical Analyst note for a time window (same verify/fallback workflow)
  GET  /api/eval              evaluation results (eval/results/results.json)
  WS   /ws/live               streams the match; pushes each chance's evidence + narration as it happens
  GET  /healthz               liveness probe for Azure Container Apps
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agents.narrator import backend_name, default_writer
from agents.workflow import narrate, summary_pack
from analysis.lanes import lanes_at
from analysis.space import control_grid
from common.pitch import FPS, ID_TO_IDX, PLAYER_IDS, fmt_clock

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"
WEB = ROOT / "web"
TEAMS = {"A": {"name": "Northbridge Athletic", "short": "NBA", "color": "#e4572e", "formation": "4-3-3"},
         "B": {"name": "Harbour Vale", "short": "HVL", "color": "#2e86de", "formation": "4-4-2"}}

app = FastAPI(title="Chance Genesis", version="0.2.0")
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
app.add_middleware(GZipMiddleware, minimum_size=2000)
STATE: dict = {}


def _load():
    if STATE:
        return STATE
    z = np.load(DEMO / "analysis.npz")
    meta = json.loads((DEMO / "obs" / "match.json").read_text())
    STATE.update({k: z[k] for k in z.files})
    STATE["meta"] = meta["meta"]
    STATE["events"] = meta["events"]
    STATE["moments"] = json.loads((DEMO / "moments.json").read_text())
    STATE["by_id"] = {m["moment_id"]: m for m in STATE["moments"]}
    STATE["narr"] = {}
    STATE["locks"] = {}
    truth_file = DEMO / "truth" / "match.json"
    # Ground truth is used ONLY for the optional "audit" badge in the UI, never by the analysis.
    STATE["truth"] = json.loads(truth_file.read_text())["truth"] if truth_file.exists() else []
    STATE["writer"] = default_writer()
    # live cause tracing: rebuild the analysis state so a shot can be explained the moment it happens
    from analysis.genesis import ChanceAnalyzer
    from analysis.preprocess import Track
    pos = STATE["pos"].astype(np.float64)
    vel = STATE["vel"].astype(np.float64)
    turnovers = [tuple(t) for t in json.loads((DEMO / "turnovers.json").read_text())]
    events = sorted(STATE["events"], key=lambda e: e["frame"])
    track = Track(pos=pos, vel=vel, speed=np.linalg.norm(vel, axis=2), ball=STATE["ball"].astype(np.float64),
                  possession=STATE["possession"], carrier=STATE["carrier"], events=events, meta=STATE["meta"],
                  turnovers=turnovers)
    STATE["analyzer"] = ChanceAnalyzer(track, {"assign": STATE["assign"], "conf": STATE["conf"].astype(np.float32)})
    STATE["shot_events"] = {int(e["frame"]): e for e in events if e["type"] == "shot"}
    return STATE


@app.on_event("startup")
async def _startup():
    _load()


def _summary(m: dict) -> dict:
    return {k: m.get(k) for k in ("moment_id", "type", "t", "t_s", "frame", "team", "shooter", "cause", "cause_label",
                                  "confidence", "verdict_type", "players")}


def _audit(m: dict) -> Optional[dict]:
    for tr in STATE["truth"]:
        if tr.get("t_shot") is not None and abs(tr["t_shot"] - m["t_s"]) < 0.5:
            from gen.scripts import CAUSE_OF_KIND
            true_cause = CAUSE_OF_KIND[tr["kind"]]
            return {"scenario": tr["kind"], "true_cause": true_cause, "match": true_cause == m["cause"],
                    "true_players": tr.get("players")}
    return None


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/api/info")
async def info():
    s = _load()
    return {"teams": TEAMS, "fps": FPS, "n_frames": int(s["pos"].shape[0]), "minutes": s["pos"].shape[0] / FPS / 60,
            "backend": backend_name(), "players": PLAYER_IDS,
            "moments": [_summary(m) for m in s["moments"]],
            "disclaimer": "Fully synthetic data. Fictional league, clubs and players."}


async def _narration(mid: str, language: str = "en") -> dict:
    s = _load()
    key = (mid, language)
    if key in s["narr"]:
        return s["narr"][key]
    lock = s["locks"].setdefault(key, asyncio.Lock())
    async with lock:
        if key not in s["narr"]:
            n = await narrate(s["by_id"][mid], s["writer"], language)
            s["narr"][key] = n.to_dict()
    return s["narr"][key]


@app.get("/api/moment/{mid}")
async def moment(mid: str, language: str = "en"):
    s = _load()
    if mid not in s["by_id"]:
        raise HTTPException(404, "unknown moment")
    m = s["by_id"][mid]
    return {"evidence": m, "narration": await _narration(mid, language), "audit": _audit(m), "backend": backend_name()}


def _frame(f: int, with_lanes: bool = False) -> dict:
    s = STATE
    pos = s["pos"][f]
    dp = s["disp"][f] if "disp" in s else pos
    db = s["disp_ball"][f] if "disp_ball" in s else s["ball"][f]
    out = {"f": int(f), "t": round(f / FPS, 2), "clock": fmt_clock(f / FPS),
           "b": [round(float(db[0]), 2), round(float(db[1]), 2)],
           "p": np.round(dp, 2).tolist(), "c": int(s["carrier"][f]), "poss": int(s["possession"][f]),
           "m": s["assign"][f].astype(int).tolist()}
    off = s["offside"][f]
    out["off"] = None if np.isnan(off) else round(float(off), 2)
    if with_lanes and out["c"] >= 0:
        r = lanes_at(pos.astype(np.float64), out["c"])
        out["lanes"] = {"open": [int(x) for x in r["mates"][~r["blocked"]]],
                        "blocked": [int(x) for x in r["mates"][r["blocked"]]]}
    return out


@app.get("/api/clip/{mid}")
async def clip(mid: str, step: int = 1):
    s = _load()
    if mid not in s["by_id"]:
        raise HTTPException(404, "unknown moment")
    m = s["by_id"][mid]
    f0, f1 = m["clip"]["from_frame"], m["clip"]["to_frame"]
    frames = [_frame(f, with_lanes=True) for f in range(f0, f1 + 1, max(1, step))]
    overlays = _overlays(m)
    return {"moment_id": mid, "from": f0, "to": f1, "frames": frames, "overlays": overlays,
            "evidence_frames": m["evidence_frames"]}


def _overlays(m: dict) -> dict:
    """Static drawings for the replay: vacated space, counterfactual ghost, key players."""
    s = STATE
    pos, ev = s["pos"], m["evidence_frames"]
    o: dict = {"key_players": [], "ghost": None, "space": None, "space_cf": None, "focus": None}
    p = m.get("players", {})
    for role in ("runner", "dragged", "beneficiary", "attacker", "defender", "ball_winner", "shooter"):
        if role in p:
            o["key_players"].append({"id": p[role], "idx": ID_TO_IDX[p[role]], "role": role})
    for pid in p.get("pressers", []) if isinstance(p.get("pressers"), list) else []:
        o["key_players"].append({"id": pid, "idx": ID_TO_IDX[pid], "role": "presser"})
    shooter = ID_TO_IDX[m["shooter"]]
    if m["cause"] == "decoy_run":
        f_near, f_r = ev[1], ev[2]
        d = ID_TO_IDX[p["dragged"]]
        focus = pos[f_r, shooter].astype(float)
        o["ghost"] = {"idx": d, "id": p["dragged"], "x": round(float(pos[f_near, d, 0]), 2), "y": round(float(pos[f_near, d, 1]), 2),
                      "from_frame": int(f_near), "label": f"{p['dragged']} if he had stayed"}
        vel = np.gradient(pos[max(f_r - 2, 0):f_r + 3].astype(float), axis=0)[2] * FPS
        o["space"] = control_grid(pos[f_r].astype(float), vel, focus, radius=12.0)
        pcf = pos[f_r].astype(float).copy()
        vcf = vel.copy()
        pcf[d] = pos[f_near, d]
        vcf[d] = 0
        o["space_cf"] = control_grid(pcf, vcf, focus, radius=12.0)
        o["focus"] = {"x": round(float(focus[0]), 2), "y": round(float(focus[1]), 2), "frame": int(f_r)}
    elif m["cause"] == "lost_marker":
        f_close, f_r = ev[0], ev[2]
        d = ID_TO_IDX[p["defender"]]
        gp = pos[f_r, shooter] + (pos[f_close, d] - pos[f_close, shooter])
        o["ghost"] = {"idx": d, "id": p["defender"], "x": round(float(gp[0]), 2), "y": round(float(gp[1]), 2),
                      "from_frame": int(f_r), "label": f"{p['defender']} keeping his gap"}
        o["focus"] = {"x": round(float(pos[f_r, shooter, 0]), 2), "y": round(float(pos[f_r, shooter, 1]), 2), "frame": int(f_r)}
    elif m["cause"] == "press_turnover":
        f_peak = ev[1]
        o["focus"] = {"x": round(float(s["ball"][f_peak, 0]), 2), "y": round(float(s["ball"][f_peak, 1]), 2), "frame": int(f_peak)}
    return o


@app.get("/api/frames")
async def frames(start: int = 0, end: int = 250, step: int = 2):
    s = _load()
    end = min(end, s["pos"].shape[0] - 1)
    if end - start > 25 * 120:
        raise HTTPException(400, "at most 120 s per request")
    return {"frames": [_frame(f) for f in range(start, end + 1, max(1, step))]}


STEP_MATCH = 2   # full-match download at 12.5 frames per second


def _match_blobs() -> dict:
    """Whole match as compact binary arrays, built once: the browser scrubs it locally."""
    s = _load()
    if "blobs" in s:
        return s["blobs"]
    sl = slice(0, None, STEP_MATCH)
    dp = s["disp"][sl] if "disp" in s else s["pos"][sl]
    db = s["disp_ball"][sl] if "disp_ball" in s else s["ball"][sl]
    n = dp.shape[0]
    track = np.empty((n, 23, 2), dtype=np.int16)                  # decimetres
    track[:, :22] = np.round(dp * 10).astype(np.int16)
    track[:, 22] = np.round(db * 10).astype(np.int16)
    off = s["offside"][sl]
    state = np.empty((n, 26), dtype=np.int8)                       # carrier, possession, 22 marks, offside (dm, 2 bytes)
    state[:, 0] = s["carrier"][sl]
    state[:, 1] = s["possession"][sl]
    state[:, 2:24] = s["assign"][sl]
    offi = np.where(np.isnan(off), -1, np.round(off * 10)).astype(np.int16)
    state[:, 24:26] = offi.view(np.int8).reshape(n, 2)
    s["blobs"] = {"track": track.tobytes(), "state": state.tobytes(), "n": n}
    return s["blobs"]


@app.get("/api/match/meta")
async def match_meta():
    b = _match_blobs()
    return {"frames": b["n"], "fps": FPS / STEP_MATCH, "step": STEP_MATCH, "track_bytes": len(b["track"]),
            "state_bytes": len(b["state"]), "layout": {"track": "int16 [frame][23][x,y] decimetres, row 22 = ball",
                                                       "state": "int8 [frame][26]: carrier, possession, 22 marks, offside int16 dm"}}


@app.get("/api/match/track")
async def match_track():
    from fastapi.responses import Response
    return Response(_match_blobs()["track"], media_type="application/octet-stream")


@app.get("/api/match/state")
async def match_state():
    from fastapi.responses import Response
    return Response(_match_blobs()["state"], media_type="application/octet-stream")


@app.get("/api/summary")
async def summary(start_s: float = 0, end_s: float = 45 * 60, language: str = "en"):
    s = _load()
    pack = summary_pack(s["moments"], start_s, end_s)
    n = await narrate(pack, s["writer"], language)
    return {"evidence": pack, "narration": n.to_dict()}


@app.get("/api/eval")
async def eval_results():
    p = ROOT / "eval" / "results" / "results.json"
    if not p.exists():
        raise HTTPException(404, "run python -m eval.run_eval first")
    r = json.loads(p.read_text())
    return JSONResponse({k: r[k] for k in r if k != "rows"})


@app.websocket("/ws/live")
async def live(ws: WebSocket):
    """Streams the match at `speed`x (frames sent at 12.5 Hz of match time). When a shot
    happens, the chance's evidence pack is pushed immediately and the verified narration
    follows as soon as the agent workflow finishes (event-driven, no polling)."""
    s = _load()
    await ws.accept()
    params = ws.query_params
    speed = float(params.get("speed", 4))
    f = int(params.get("start", 0))
    step = 2
    T = s["pos"].shape[0]
    try:
        while f < T:
            t0 = time.perf_counter()
            await ws.send_text(json.dumps({"type": "frame", "frame": _frame(f)}))
            for g in range(f, min(f + step, T)):
                if g in s["shot_events"]:
                    # trace the cause now, from the tracking state up to this shot
                    a = time.perf_counter()
                    m = s["analyzer"].analyze(s["shot_events"][g])
                    genesis_ms = (time.perf_counter() - a) * 1000
                    s["by_id"].setdefault(m["moment_id"], m)
                    await ws.send_text(json.dumps({"type": "chance", "evidence": m, "audit": _audit(m),
                                                   "genesis_ms": round(genesis_ms, 2)}, default=float))
                    asyncio.create_task(_push_narration(ws, m["moment_id"]))
            f += step
            await asyncio.sleep(max(0.0, step / FPS / speed - (time.perf_counter() - t0)))
        await ws.send_text(json.dumps({"type": "end"}))
        await asyncio.sleep(0.5)          # let pending narrations go out
        await ws.close()
    except WebSocketDisconnect:
        return


async def _push_narration(ws: WebSocket, mid: str):
    try:
        n = await _narration(mid)
        await ws.send_text(json.dumps({"type": "narration", "moment_id": mid, "narration": n}, default=float))
    except Exception:
        pass


@app.get("/")
async def index():
    return FileResponse(WEB / "index.html")


if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")
