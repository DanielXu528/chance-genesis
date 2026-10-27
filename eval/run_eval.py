"""Offline evaluation against ground truth. Every number in the README comes from here.

Protocol
  * Thresholds were tuned on development seeds 100-299 only.
  * Test episodes use seeds >= 10000 (never seen during development).
  * The analysis receives only observations (noisy tracking + events). Ground truth is read
    here, after the analysis has run, and only to score it.

Usage: python -m eval.run_eval [--n 40] [--quick]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analysis.genesis import GenesisParams
from analysis.lanes import lanes_at
from analysis.marking import MarkingParams, cost_tensor
from analysis.pipeline import analyze
from analysis.preprocess import preprocess
from analysis.phases import phases
from common.pitch import FPS, ID_TO_IDX, outfield
from gen.episodes import EpisodeFailed, make_episode, make_match, observe
from gen.scripts import CAUSE_OF_KIND, KINDS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "results"
CAUSES = ["decoy_run", "lost_marker", "press_turnover", "none"]
TEST_SEED0 = 10000
NOISES = [0.0, 0.3, 0.6, 1.0]
MAIN_NOISE = 0.3


def build_test_set(n: int) -> list[dict]:
    eps, failed = [], 0
    for k, kind in enumerate(KINDS):
        got, i = 0, 0
        while got < n and i < n * 3:
            seed = TEST_SEED0 + k * 1000 + i
            i += 1
            try:
                ep = make_episode(kind, seed)
            except EpisodeFailed:
                failed += 1
                continue
            ep["meta"]["test_seed"] = seed
            eps.append(ep)
            got += 1
    return eps, failed


# ---------------------------------------------------------------------- marking metrics
def marking_metrics(ep: dict, mk: dict) -> dict:
    lab = ep["labels"]
    out = {}
    for name, pred in (("raw", mk["raw"]), ("hysteresis", mk["hysteresis"]), ("viterbi", mk["assign"])):
        ok = (lab >= -1) & (pred >= -1)
        out[name] = {"n": int(ok.sum()), "correct": int(((lab == pred) & ok).sum()),
                     "man_n": int((ok & (lab >= 0)).sum()), "man_correct": int((ok & (lab >= 0) & (lab == pred)).sum()),
                     "zone_n": int((ok & (lab == -1)).sum()), "zone_correct": int((ok & (lab == -1) & (pred == -1)).sum()),
                     "changes": int(((pred[1:] != pred[:-1]) & (pred[1:] >= -1) & (pred[:-1] >= -1)).sum()),
                     "frames": int((pred >= -1).sum())}
    out["truth_changes"] = int(((lab[1:] != lab[:-1]) & (lab[1:] >= -1) & (lab[:-1] >= -1)).sum())
    # switch latency: truth changes man->man; time until the prediction shows the new man
    lat = {"hysteresis": [], "viterbi": [], "raw": []}
    T = lab.shape[0]
    for d in range(22):
        col = lab[:, d]
        ch = np.nonzero((col[1:] != col[:-1]) & (col[1:] >= 0) & (col[:-1] >= 0))[0] + 1
        for f in ch:
            new = col[f]
            hi = min(T, f + 4 * FPS)
            if not np.all(col[f:hi] == new) and hi - f < 2 * FPS:
                continue
            for name, pred in (("raw", mk["raw"]), ("hysteresis", mk["hysteresis"]), ("viterbi", mk["assign"])):
                hit = np.nonzero(pred[f:hi, d] == new)[0]
                lat[name].append(float(hit[0]) / FPS if len(hit) else None)
    out["switch_latency"] = lat
    return out


def agg_marking(rows: list[dict]) -> dict:
    res = {}
    for name in ("raw", "hysteresis", "viterbi"):
        s = defaultdict(int)
        for r in rows:
            for k, v in r[name].items():
                s[k] += v
        lat = [x for r in rows for x in r["switch_latency"][name]]
        found = [x for x in lat if x is not None]
        res[name] = {
            "frame_accuracy": s["correct"] / max(s["n"], 1),
            "man_marking_recall": s["man_correct"] / max(s["man_n"], 1),
            "zone_recall": s["zone_correct"] / max(s["zone_n"], 1),
            "changes_per_defender_min": s["changes"] / max(s["frames"] / FPS / 60, 1e-9),
            "switches_found_within_4s": len(found) / max(len(lat), 1),
            "switch_latency_median_s": float(np.median(found)) if found else None,
            "switch_latency_p90_s": float(np.percentile(found, 90)) if found else None,
            "n_frames": s["n"], "n_switches": len(lat),
        }
    tc = sum(r["truth_changes"] for r in rows)
    fr = sum(r["viterbi"]["frames"] for r in rows)
    res["truth_changes_per_defender_min"] = tc / max(fr / FPS / 60, 1e-9)
    return res


# ---------------------------------------------------------------------- genesis metrics
def score_moment(ep: dict, moments: list[dict]) -> dict:
    tr = ep["truth"][0]
    kind = tr["kind"]
    true_cause = CAUSE_OF_KIND[kind]
    m = min(moments, key=lambda m: abs(m["t_s"] - tr["t_shot"])) if moments else None
    pred = m["cause"] if m else "none"
    r = {"kind": kind, "true": true_cause, "pred": pred, "seed": ep["meta"]["test_seed"],
         "players_ok": None, "t_err": None, "verdict": m["verdict_type"] if m else None}
    if m and pred == true_cause and pred != "none":
        tp, pp = tr["players"], m["players"]
        if pred == "decoy_run":
            r["players_ok"] = pp.get("runner") == tp.get("runner") and pp.get("dragged") == tp.get("dragged")
        elif pred == "lost_marker":
            r["players_ok"] = pp.get("defender") == tp.get("defender") and pp.get("attacker") == tp.get("attacker")
        elif pred == "press_turnover":
            r["players_ok"] = pp.get("ball_winner") == tp.get("ball_winner")
            r["pressers_overlap"] = len(set(pp.get("pressers", [])) & set(tp.get("pressers", []))) / 3
        if m.get("t_start") is not None and tr.get("t_start") is not None:
            r["t_err"] = abs(m["t_start"] - tr["t_start"])
    return r


def agg_genesis(rows: list[dict]) -> dict:
    conf = Counter((r["kind"], r["pred"]) for r in rows)
    n = len(rows)
    correct = sum(r["pred"] == r["true"] for r in rows)
    pos = [r for r in rows if r["true"] != "none"]
    neg = [r for r in rows if r["true"] == "none"]
    per = {}
    for c in CAUSES[:3]:
        tp = sum(1 for r in rows if r["pred"] == c and r["true"] == c)
        fp = sum(1 for r in rows if r["pred"] == c and r["true"] != c)
        fn = sum(1 for r in rows if r["pred"] != c and r["true"] == c)
        both = [r for r in rows if r["pred"] == c and r["true"] == c]
        per[c] = {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                  "players_correct_given_cause": sum(1 for r in both if r["players_ok"]) / max(len(both), 1),
                  "t_start_err_median_s": float(np.median([r["t_err"] for r in both if r["t_err"] is not None])) if both else None}
    full = sum(1 for r in pos if r["pred"] == r["true"] and r["players_ok"])
    return {
        "n": n, "accuracy": correct / n,
        "positives_cause_and_players_correct": full / max(len(pos), 1),
        "positives_cause_correct": sum(r["pred"] == r["true"] for r in pos) / max(len(pos), 1),
        "negatives_false_positive_rate": sum(r["pred"] != "none" for r in neg) / max(len(neg), 1),
        "per_cause": per,
        "confusion": {k: {c: conf[(k, c)] for c in CAUSES} for k in KINDS},
        "t_start_err_median_s_all": float(np.median([r["t_err"] for r in rows if r["t_err"] is not None]))
        if any(r["t_err"] is not None for r in rows) else None,
    }


# ---------------------------------------------------------------------- performance
def per_frame_benchmark(obs: dict, n_frames: int = 20000) -> dict:
    """Streaming-style cost of the per-frame stage: marking cost matrix + Hungarian +
    hysteresis update + offside line + lanes for the carrier, one frame at a time."""
    from scipy.optimize import linear_sum_assignment
    from analysis.offside import offside_line
    tr = preprocess(obs)
    ph = phases(tr)
    p = MarkingParams()
    zone = np.full((10, 10), p.zone_cost)
    times = []
    T = min(tr.T, n_frames)
    for f in range(T):
        dt = int(ph["defending"][f])
        t0 = time.perf_counter()
        if dt >= 0:
            C = cost_tensor(tr, dt, p, slice(f, f + 1))[0]
            linear_sum_assignment(np.hstack([C, zone]))
            offside_line(tr.pos[f], tr.ball[f], 1 - dt)
            c = int(tr.carrier[f])
            if c >= 0:
                lanes_at(tr.pos[f], c)
        times.append(time.perf_counter() - t0)
    t = np.array(times) * 1000
    return {"frames": T, "p50_ms": float(np.percentile(t, 50)), "p99_ms": float(np.percentile(t, 99)),
            "mean_ms": float(t.mean()), "realtime_budget_ms": 1000 / FPS}


def realism(match: dict) -> dict:
    pos = match["pos"].astype(np.float64)
    T = pos.shape[0]
    minutes = T / FPS / 60
    v = np.linalg.norm(np.diff(pos, axis=0), axis=2) * FPS
    outf = [i for i in range(22) if i not in (0, 11)]
    dist_km = (v[:, outf].sum(axis=0) / FPS / 1000) * (90 / minutes)
    sprints = []
    for i in outf:
        fast = v[:, i] > 7.0
        starts = np.nonzero(fast[1:] & ~fast[:-1])[0]
        # a sprint lasts at least 1 s
        cnt = sum(1 for s in starts if fast[s + 1:s + 1 + FPS].all())
        sprints.append(cnt * 90 / minutes)
    hi_m = (np.clip(v[:, outf], 0, None) * (v[:, outf] > 5.5)).sum(axis=0) / FPS * (90 / minutes)
    widths, depths = [], []
    for t in (0, 1):
        o = outfield(t)
        widths.append(np.ptp(pos[:, o, 1], axis=1).mean())
        depths.append(np.ptp(pos[:, o, 0], axis=1).mean())
    return {"minutes": minutes, "distance_km_per90_mean": float(dist_km.mean()),
            "distance_km_per90_range": [float(dist_km.min()), float(dist_km.max())],
            "high_intensity_m_per90_mean": float(hi_m.mean()),
            "sprints_per90_mean": float(np.mean(sprints)),
            "team_width_m_mean": float(np.mean(widths)), "team_length_m_mean": float(np.mean(depths)),
            "max_speed_ms": float(v[:, outf].max())}


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="episodes per scenario kind")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    n = 8 if args.quick else args.n
    eps, failed = build_test_set(n)
    print(f"test set: {len(eps)} episodes ({failed} seeds skipped: script could not start cleanly)")
    results = {"protocol": {"test_seed0": TEST_SEED0, "episodes": len(eps), "per_kind": n, "skipped_seeds": failed,
                            "main_noise_m": MAIN_NOISE, "noise_levels_m": NOISES}}
    by_noise = {}
    for k, noise in enumerate(NOISES):
        mrows, grows = [], []
        for ep in eps:
            obs = observe(ep, noise, seed=ep["meta"]["test_seed"] * 10 + k)
            an = analyze(obs)
            mrows.append(marking_metrics(ep, an.marking))
            grows.append(score_moment(ep, an.moments))
        by_noise[noise] = {"marking": agg_marking(mrows), "genesis": agg_genesis(grows), "rows": grows}
        g, m = by_noise[noise]["genesis"], by_noise[noise]["marking"]
        print(f"noise {noise:.1f} m | genesis acc {g['accuracy']:.3f}  cause+players {g['positives_cause_and_players_correct']:.3f}"
              f"  FP {g['negatives_false_positive_rate']:.3f} | marking raw {m['raw']['frame_accuracy']:.3f}"
              f" hyst {m['hysteresis']['frame_accuracy']:.3f} vit {m['viterbi']['frame_accuracy']:.3f}")
    results["by_noise"] = {str(k): {"marking": v["marking"], "genesis": v["genesis"]} for k, v in by_noise.items()}
    # temporally correlated tracking error (drift), the harder case
    by_corr = {}
    for k, noise in enumerate([0.3, 0.6, 1.0]):
        mrows, grows = [], []
        for ep in eps:
            obs = observe(ep, noise, seed=ep["meta"]["test_seed"] * 10 + 7 + k, kind="ar1", corr_s=1.0)
            an = analyze(obs)
            mrows.append(marking_metrics(ep, an.marking))
            grows.append(score_moment(ep, an.moments))
        by_corr[str(noise)] = {"marking": agg_marking(mrows), "genesis": agg_genesis(grows)}
        g, m = by_corr[str(noise)]["genesis"], by_corr[str(noise)]["marking"]
        print(f"AR1 noise {noise:.1f} m | genesis acc {g['accuracy']:.3f}  cause+players {g['positives_cause_and_players_correct']:.3f}"
              f"  FP {g['negatives_false_positive_rate']:.3f} | marking vit {m['viterbi']['frame_accuracy']:.3f}")
    results["by_noise_correlated"] = by_corr
    (OUT / "genesis_rows.json").write_text(json.dumps(by_noise[MAIN_NOISE]["rows"], indent=1))

    # full match: integration + throughput + realism
    t0 = time.time()
    match = make_match(seed=7, minutes=10.0 if args.quick else 90.0, n_chances=4 if args.quick else 24)
    gen_s = time.time() - t0
    obs = observe(match, MAIN_NOISE, seed=7)
    t0 = time.time()
    an = analyze(obs)
    total = time.time() - t0
    lat = an.timings["genesis_per_chance_ms"]
    rows = []
    for tr in match["truth"]:
        ms = [m for m in an.moments if abs(m["t_s"] - tr["t_shot"]) < 0.5]
        pred = ms[0]["cause"] if ms else "missed"
        rows.append({"kind": tr["kind"], "true": CAUSE_OF_KIND[tr["kind"]], "pred": pred})
    results["full_match"] = {
        "minutes": match["meta"]["n_frames"] / FPS / 60, "frames": match["meta"]["n_frames"],
        "scripted_chances": len(match["truth"]), "shots_total": len(an.moments),
        "scripted_cause_correct": sum(r["pred"] == r["true"] for r in rows) / max(len(rows), 1),
        "rows": rows, "generate_s": gen_s, "analysis_total_s": total,
        "x_realtime": (match["meta"]["n_frames"] / FPS) / total,
        "stage_s": {k: v for k, v in an.timings.items() if k != "genesis_per_chance_ms"},
        "genesis_per_chance_ms_median": float(np.median(lat)) if lat else None,
        "genesis_per_chance_ms_max": float(np.max(lat)) if lat else None,
    }
    results["per_frame"] = per_frame_benchmark(obs, 5000 if args.quick else 20000)
    results["realism"] = realism(match)
    results["wall_s"] = time.time() - t_all
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))
    write_markdown(results)
    plot_noise(results)
    print(json.dumps({k: results[k] for k in ("full_match", "per_frame", "realism")}, indent=1, default=float)[:3000])


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def write_markdown(res: dict):
    main = res["by_noise"][str(MAIN_NOISE)]
    g, m = main["genesis"], main["marking"]
    L = []
    L.append(f"# Evaluation results\n\nGenerated by `python -m eval.run_eval`. Test set: {res['protocol']['episodes']} held-out "
             f"episodes ({res['protocol']['per_kind']} per scenario kind, seeds >= {res['protocol']['test_seed0']}), "
             f"main tracking noise {MAIN_NOISE} m (Gaussian, per player per frame).\n")
    L.append("## Chance genesis (cause attribution)\n")
    L.append(f"- Overall accuracy ({len(KINDS)} scenario kinds, 4 classes): **{pct(g['accuracy'])}**")
    L.append(f"- Real causes (decoy / lost marker / press): cause correct **{pct(g['positives_cause_correct'])}**, "
             f"cause and key players correct **{pct(g['positives_cause_and_players_correct'])}**")
    L.append(f"- Hard negatives (no tactical cause): false attribution rate **{pct(g['negatives_false_positive_rate'])}**")
    if g["t_start_err_median_s_all"] is not None:
        L.append(f"- Median error of the detected start time: **{g['t_start_err_median_s_all']:.2f} s**\n")
    L.append("| Cause | Precision | Recall | Key players correct (when cause correct) | Median start-time error |")
    L.append("|---|---|---|---|---|")
    for c, v in g["per_cause"].items():
        te = "n/a" if v["t_start_err_median_s"] is None else f"{v['t_start_err_median_s']:.2f} s"
        L.append(f"| {c} | {pct(v['precision'])} | {pct(v['recall'])} | {pct(v['players_correct_given_cause'])} | {te} |")
    L.append("\nConfusion matrix (rows: scenario kind, columns: predicted cause)\n")
    L.append("| kind | " + " | ".join(CAUSES) + " |")
    L.append("|---" * (len(CAUSES) + 1) + "|")
    for k, row in g["confusion"].items():
        L.append(f"| {k} | " + " | ".join(str(row[c]) for c in CAUSES) + " |")
    L.append("\n## Marking inference\n")
    L.append("| Smoother | Frame accuracy | Man-marking recall | Zone recall | Assignment changes / defender-min | Switches found ≤4 s | Median switch latency |")
    L.append("|---|---|---|---|---|---|---|")
    for name in ("raw", "hysteresis", "viterbi"):
        v = m[name]
        lat = "n/a" if v["switch_latency_median_s"] is None else f"{v['switch_latency_median_s']:.2f} s"
        L.append(f"| {name} | {pct(v['frame_accuracy'])} | {pct(v['man_marking_recall'])} | {pct(v['zone_recall'])} | "
                 f"{v['changes_per_defender_min']:.1f} | {pct(v['switches_found_within_4s'])} | {lat} |")
    L.append(f"\nGround truth itself changes {m['truth_changes_per_defender_min']:.1f} times per defender-minute.\n")
    L.append("## Robustness to tracking noise\n")
    L.append("| Noise (m) | Genesis accuracy | Cause + players (real causes) | False attribution on negatives | Marking frame accuracy (Viterbi) |")
    L.append("|---|---|---|---|---|")
    for nz, v in res["by_noise"].items():
        L.append(f"| {float(nz):.1f} | {pct(v['genesis']['accuracy'])} | {pct(v['genesis']['positives_cause_and_players_correct'])} | "
                 f"{pct(v['genesis']['negatives_false_positive_rate'])} | {pct(v['marking']['viterbi']['frame_accuracy'])} |")
    if res.get("by_noise_correlated"):
        L.append("\nTemporally correlated error (AR(1), 1 s correlation time, same per-coordinate std):\n")
        L.append("| Noise (m) | Genesis accuracy | Cause + players (real causes) | False attribution on negatives | Marking frame accuracy (Viterbi) |")
        L.append("|---|---|---|---|---|")
        for nz, v in res["by_noise_correlated"].items():
            L.append(f"| {float(nz):.1f} | {pct(v['genesis']['accuracy'])} | {pct(v['genesis']['positives_cause_and_players_correct'])} | "
                     f"{pct(v['genesis']['negatives_false_positive_rate'])} | {pct(v['marking']['viterbi']['frame_accuracy'])} |")
    fm, pf = res["full_match"], res["per_frame"]
    L.append("\n## Performance (single CPU core, Python + NumPy/SciPy)\n")
    L.append(f"- Full {fm['minutes']:.0f}-minute match ({fm['frames']:,} frames at 25 Hz): whole pipeline in "
             f"**{fm['analysis_total_s']:.1f} s** ({fm['x_realtime']:.0f}x faster than real time)")
    L.append(f"- Per-frame live stage (marking cost + Hungarian + offside + passing lanes): p50 **{pf['p50_ms']:.2f} ms**, "
             f"p99 **{pf['p99_ms']:.2f} ms** (budget at 25 Hz: {pf['realtime_budget_ms']:.0f} ms)")
    if fm["genesis_per_chance_ms_median"] is not None:
        L.append(f"- Chance genesis after a shot: median **{fm['genesis_per_chance_ms_median']:.1f} ms** per chance "
                 f"(max {fm['genesis_per_chance_ms_max']:.1f} ms)")
    L.append(f"- In the full match, {fm['scripted_chances']} scripted chances among {fm['shots_total']} shots: "
             f"cause correct for {pct(fm['scripted_cause_correct'])}")
    r = res["realism"]
    L.append("\n## Generator realism (full match)\n")
    L.append(f"- Distance per outfield player: {r['distance_km_per90_mean']:.1f} km per 90 "
             f"(range {r['distance_km_per90_range'][0]:.1f}-{r['distance_km_per90_range'][1]:.1f})")
    L.append(f"- Running above 5.5 m/s: {r['high_intensity_m_per90_mean']:.0f} m per player per 90; "
             f"sprints above 7 m/s lasting at least 1 s: {r['sprints_per90_mean']:.1f} per player per 90")
    L.append(f"- Mean team width {r['team_width_m_mean']:.1f} m, length {r['team_length_m_mean']:.1f} m; top speed {r['max_speed_ms']:.1f} m/s")
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n")


def plot_noise(res: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nz = [float(k) for k in res["by_noise"]]
    g = [res["by_noise"][k]["genesis"]["accuracy"] * 100 for k in res["by_noise"]]
    cp = [res["by_noise"][k]["genesis"]["positives_cause_and_players_correct"] * 100 for k in res["by_noise"]]
    mk = [res["by_noise"][k]["marking"]["viterbi"]["frame_accuracy"] * 100 for k in res["by_noise"]]
    fp = [res["by_noise"][k]["genesis"]["negatives_false_positive_rate"] * 100 for k in res["by_noise"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    ax.plot(nz, g, "o-", label="Genesis accuracy", color="#2a6fdb")
    ax.plot(nz, cp, "s-", label="Cause + players (real causes)", color="#1b9e77")
    ax.plot(nz, mk, "^-", label="Marking frame accuracy", color="#7a5195")
    ax.plot(nz, fp, "v--", label="False attribution (negatives)", color="#d95f02")
    if res.get("by_noise_correlated"):
        cz = [float(k) for k in res["by_noise_correlated"]]
        cg = [res["by_noise_correlated"][k]["genesis"]["accuracy"] * 100 for k in res["by_noise_correlated"]]
        ax.plot(cz, cg, "o:", label="Genesis accuracy, correlated error", color="#2a6fdb", alpha=0.6)
    ax.set_xlabel("Tracking noise, std per coordinate (m)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.set_xticks(nz)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="center left")
    ax.set_title("Held-out test set vs tracking noise", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "noise_curve.png")


if __name__ == "__main__":
    main()
