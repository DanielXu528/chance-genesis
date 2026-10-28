"""Build the demo match served by the app: a 90-minute synthetic match between two
fictional clubs. Observations and ground truth are written to separate files; the
analysis cache is computed from observations only.

Usage: python -m server.prepare [--minutes 90] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from analysis.pipeline import analyze
from gen.episodes import load_obs, make_match, save_episode

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"
NAME = "match"
DEMO_SEED = 27


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=DEMO_SEED)
    ap.add_argument("--noise", type=float, default=0.15, help="feed error std (m): AR(1) drift + 5 cm flicker")
    args = ap.parse_args()
    t0 = time.time()
    match = make_match(seed=args.seed, minutes=args.minutes, n_chances=max(4, int(round(24 * args.minutes / 90))))
    print(f"generated {match['meta']['n_frames']} frames, {len(match['truth'])} scripted chances in {time.time() - t0:.1f}s")
    save_episode(match, DEMO, NAME, noise_m=args.noise, obs_seed=args.seed)
    obs = load_obs(DEMO, NAME)
    t1 = time.time()
    an = analyze(obs)
    print(f"analysed in {time.time() - t1:.1f}s, {len(an.moments)} shots")
    # display track: one more smoothing pass so players glide on screen (analysis keeps its own)
    from scipy.signal import savgol_filter
    disp = savgol_filter(an.track.pos, 25, 3, axis=0).astype(np.float32)
    disp_ball = savgol_filter(an.track.ball, 7, 2, axis=0).astype(np.float32)
    np.savez_compressed(DEMO / "analysis.npz", pos=an.track.pos.astype(np.float32), ball=an.track.ball.astype(np.float32),
                        disp=disp, disp_ball=disp_ball,
                        vel=an.track.vel.astype(np.float32),
                        possession=an.track.possession, carrier=an.track.carrier, assign=an.marking["assign"],
                        conf=an.marking["conf"].astype(np.float16), offside=an.offside.astype(np.float32),
                        phase=an.phases["phase"], third=an.phases["third"])
    (DEMO / "moments.json").write_text(json.dumps(an.moments, indent=1, default=float))
    (DEMO / "timings.json").write_text(json.dumps(an.timings, indent=1, default=float))
    (DEMO / "turnovers.json").write_text(json.dumps([[int(a), int(b), int(c), d] for a, b, c, d in an.track.turnovers]))


if __name__ == "__main__":
    main()
