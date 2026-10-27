"""Build episodes (a build-up followed by one scripted chance) and full synthetic matches,
and split each into an observation (what the analysis may read) and the ground truth.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from common.pitch import DT, FPS, PLAYER_IDS, keeper, outfield, team_of
from gen.generator import Director, Sim
from gen.scripts import KINDS, SCRIPTS, ScriptAbort

MAX_EPISODE_S = 70.0


class EpisodeFailed(RuntimeError):
    pass


def _new_sim(rng, kind: str, team: int) -> Sim:
    if kind in ("press_turnover", "neg_unforced"):
        Y = 1 - team
        sim = Sim(rng, possession=Y, ball_rel_x=rng.uniform(12, 22) if kind == "press_turnover" else rng.uniform(22, 32))
        cbs = [i for i in outfield(Y) if sim.role[i] in ("RCB", "LCB")]
        sim.carrier = int(rng.choice(cbs))
        sim.director = Director(rng, steer_x=rng.uniform(14, 24) if kind == "press_turnover" else rng.uniform(24, 34))
    else:
        sim = Sim(rng, possession=team, ball_rel_x=rng.uniform(28, 42))
        backs = [i for i in outfield(team) if sim.role[i] in ("RCB", "LCB", "DM", "RCM", "LCM")]
        sim.carrier = int(rng.choice(backs))
        steer = rng.uniform(70, 78) if kind == "neg_long" else rng.uniform(60, 70)
        sim.director = Director(rng, steer_x=steer)
    sim.ball = sim.pos[sim.carrier].copy()
    sim.hold_until = int(rng.uniform(0.5, 1.5) * FPS)
    return sim


def make_episode(kind: str, seed: int, team: Optional[int] = None, tries: int = 20) -> dict:
    """Simulate one episode. Retries with derived seeds if the script never fires."""
    for attempt in range(tries):
        rng = np.random.default_rng([seed, attempt])
        tm = int(rng.integers(2)) if team is None else team
        sim = _new_sim(rng, kind, tm)
        start = int(rng.uniform(9, 16) * FPS)
        sim.script = SCRIPTS[kind](rng, tm, start)
        end = None
        aborted = False
        while sim.frame < MAX_EPISODE_S * FPS:
            try:
                sim.step()
            except ScriptAbort:
                aborted = True
                break
            if sim.script.done and end is None:
                end = sim.frame + int(1.6 * FPS)
            if end is not None and sim.frame >= end:
                break
        if end is not None and not aborted:
            return _pack(sim, {"episode_kind": kind, "seed": seed, "attempt": attempt, "attacking_team": "AB"[tm]},
                         [sim.script.truth()])
    raise EpisodeFailed(f"{kind} seed={seed}")


def make_match(seed: int, minutes: float = 90.0, n_chances: int = 24) -> dict:
    """One continuous synthetic match: filler possession with turnovers, scripted chances spread
    through it, restarts from the goalkeeper after each shot."""
    rng = np.random.default_rng(seed)
    n_frames = int(minutes * 60 * FPS)
    sim = Sim(rng, possession=0, ball_rel_x=30.0)
    sim.carrier = 4
    sim.ball = sim.pos[4].copy()
    sim.director = Director(rng, steer_x=None, turnover_frame=int(rng.uniform(15, 35) * FPS))
    kinds = [KINDS[i % len(KINDS)] for i in range(n_chances)]
    rng.shuffle(kinds)
    plan = [(int((k + 0.5) * n_frames / n_chances + rng.uniform(-40, 10) * FPS), kind, int(rng.integers(2)))
            for k, kind in enumerate(kinds)]
    truths: list[dict] = []
    pending = list(plan)
    dead_since = None
    while sim.frame < n_frames:
        if sim.script is None and pending and sim.frame >= pending[0][0] and not sim.dead:
            _, kind, tm = pending.pop(0)
            sim.script = SCRIPTS[kind](rng, tm, sim.frame)
            want = 1 - tm if kind in ("press_turnover", "neg_unforced") else tm
            sim.director.steer_x = rng.uniform(14, 24) if kind in ("press_turnover", "neg_unforced") else (
                rng.uniform(70, 78) if kind == "neg_long" else rng.uniform(60, 70))
            sim.director.turnover_frame = None if sim.possession == want else sim.frame + int(rng.uniform(2, 5) * FPS)
            sim.script.waiting_since = sim.frame
        sc = sim.script
        if sc is not None and sc.phase == "wait":
            want = 1 - sc.team if sc.kind in ("press_turnover", "neg_unforced") else sc.team
            if sim.possession != want and sim.director.turnover_frame is None and sim.carrier is not None:
                sim.director.turnover_frame = sim.frame + int(rng.uniform(1, 3) * FPS)
            if sim.frame - sc.waiting_since > 60 * FPS:   # give up on this chance
                sim.script = None
                sim.director_enabled = True
        try:
            sim.step()
        except ScriptAbort:
            for i in list(sim.ov.keys()):
                sim.clear_override(i)
            sim.locked.clear()
            sim.unmarkable.clear()
            sim.zonal_locked.clear()
            sim.script = None
            sim.director_enabled = True
            sim.frame += 0
            sc = None
            continue
        if sc is not None and sc.done:
            truths.append(sc.truth())
            sim.script = None
        if sim.dead:
            dead_since = sim.frame if dead_since is None else dead_since
            if sim.frame - dead_since >= int(rng.uniform(3, 5) * FPS):
                shooter_team = team_of(_last_shooter(sim))
                gk = keeper(1 - shooter_team)
                sim.dead = False
                sim.flip_possession(gk)
                sim.ball = sim.pos[gk].copy()
                sim._event("restart", gk, sim.ball, sim.ball)
                sim.director_enabled = True
                sim.director.steer_x = None
                sim.director.turnover_frame = sim.frame + int(rng.uniform(15, 40) * FPS)
                dead_since = None
        elif sim.script is None and sim.director.turnover_frame is None and sim.carrier is not None:
            sim.director.turnover_frame = sim.frame + int(rng.uniform(15, 40) * FPS)
            sim.director.steer_x = None
    return _pack(sim, {"episode_kind": "match", "seed": seed, "minutes": minutes}, truths)


def _last_shooter(sim: Sim) -> int:
    from common.pitch import ID_TO_IDX
    for ev in reversed(sim.events):
        if ev["type"] == "shot":
            return ID_TO_IDX[ev["player"]]
    return 0


def _pack(sim: Sim, meta: dict, truths: list[dict]) -> dict:
    pos = np.asarray(sim.rec_pos, dtype=np.float32)
    meta = {**meta, "fps": FPS, "n_frames": int(pos.shape[0]), "formations": list(sim.formations),
            "players": PLAYER_IDS, "roles": sim.role}
    return {
        "meta": meta,
        "pos": pos,
        "ball": np.asarray(sim.rec_ball, dtype=np.float32),
        "events": sim.events,
        "labels": np.asarray(sim.rec_lab, dtype=np.int16),
        "truth": truths,
    }


def observe(ep: dict, noise_m: float, seed: int = 0, kind: str = "iid", corr_s: float = 1.0) -> dict:
    """The observation layer: tracking noise on positions, small noise on event locations.
    Returns only what a real data feed would contain (no labels, no scenario kind).

    kind="iid": independent Gaussian error per frame (std noise_m per coordinate).
    kind="ar1": temporally correlated error (AR(1), correlation time corr_s), same std; this is
                the harder case because smoothing cannot average it away."""
    rng = np.random.default_rng(seed)
    if noise_m > 0 and kind == "ar1":
        from scipy.signal import lfilter
        phi = float(np.exp(-1.0 / (FPS * corr_s)))
        e = rng.normal(0, noise_m * np.sqrt(1 - phi ** 2), size=ep["pos"].shape)
        e[0] = rng.normal(0, noise_m, size=e[0].shape)
        err = lfilter([1.0], [1.0, -phi], e, axis=0)
        pos = (ep["pos"] + err).astype(np.float32)
    elif noise_m > 0:
        pos = ep["pos"] + rng.normal(0, noise_m, size=ep["pos"].shape).astype(np.float32)
    else:
        pos = ep["pos"].copy()
    ball = ep["ball"] + rng.normal(0, min(noise_m, 0.15), size=ep["ball"].shape).astype(np.float32) if noise_m > 0 else ep["ball"].copy()
    events = []
    for ev in ep["events"]:
        e = dict(ev)
        if noise_m > 0:
            e["from"] = [round(v + float(rng.normal(0, 0.3)), 2) for v in ev["from"]]
            e["to"] = [round(v + float(rng.normal(0, 0.3)), 2) for v in ev["to"]]
        events.append(e)
    meta = {k: ep["meta"][k] for k in ("fps", "n_frames", "players", "formations")}
    meta["noise_m"] = noise_m
    meta["noise_kind"] = kind
    return {"meta": meta, "pos": pos, "ball": ball, "events": events}


# ---------------------------------------------------------------------- disk IO
def save_episode(ep: dict, root: Path, name: str, noise_m: float, obs_seed: int):
    """Observation and truth go to different directories; analysis only ever reads obs/."""
    obs = observe(ep, noise_m, obs_seed)
    (root / "obs").mkdir(parents=True, exist_ok=True)
    (root / "truth").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "obs" / f"{name}.npz", pos=obs["pos"], ball=obs["ball"])
    (root / "obs" / f"{name}.json").write_text(json.dumps({"meta": obs["meta"], "events": obs["events"]}))
    np.savez_compressed(root / "truth" / f"{name}.npz", pos=ep["pos"], ball=ep["ball"], labels=ep["labels"])
    (root / "truth" / f"{name}.json").write_text(json.dumps({"meta": ep["meta"], "truth": ep["truth"]}, indent=1))


def load_obs(root: Path, name: str) -> dict:
    z = np.load(root / "obs" / f"{name}.npz")
    j = json.loads((root / "obs" / f"{name}.json").read_text())
    return {"meta": j["meta"], "pos": z["pos"], "ball": z["ball"], "events": j["events"]}


def load_truth(root: Path, name: str) -> dict:
    z = np.load(root / "truth" / f"{name}.npz")
    j = json.loads((root / "truth" / f"{name}.json").read_text())
    return {"meta": j["meta"], "truth": j["truth"], "pos": z["pos"], "ball": z["ball"], "labels": z["labels"]}
