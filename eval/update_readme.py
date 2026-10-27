"""Copy the headline numbers from eval/results/results.json into README.md (between the
RESULTS markers), so the README never drifts from the evaluation output.

Usage: python -m eval.update_readme
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pct(x):
    return f"{100 * x:.1f}%"


def block(r: dict) -> str:
    main = r["by_noise"][str(r["protocol"]["main_noise_m"])]
    g, m = main["genesis"], main["marking"]
    fm, pf, re_ = r["full_match"], r["per_frame"], r["realism"]
    corr = r.get("by_noise_correlated", {})
    lines = [
        f"{r['protocol']['episodes']} held-out scenarios ({r['protocol']['per_kind']} per kind, seeds never used for tuning), "
        f"{r['protocol']['main_noise_m']} m tracking noise. Full tables: [eval/results/RESULTS.md](eval/results/RESULTS.md).",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Cause attribution accuracy (7 scenario kinds, 4 classes) | **{pct(g['accuracy'])}** |",
        f"| Real causes: cause **and** key players correct | **{pct(g['positives_cause_and_players_correct'])}** |",
        f"| Hard negatives wrongly given a tactical cause | **{pct(g['negatives_false_positive_rate'])}** |",
        f"| Median error of the detected start of the cause | {g['t_start_err_median_s_all']:.2f} s |",
    ]
    for c, label in (("decoy_run", "Decoy run"), ("lost_marker", "Lost marker"), ("press_turnover", "Pressing trap")):
        v = g["per_cause"][c]
        lines.append(f"| {label}: precision / recall | {pct(v['precision'])} / {pct(v['recall'])} |")
    lines += [
        f"| Marking inference, frame accuracy (Viterbi) | {pct(m['viterbi']['frame_accuracy'])} |",
        f"| Accuracy under correlated tracking error 0.3 / 0.6 / 1.0 m | "
        + " / ".join(pct(corr[k]["genesis"]["accuracy"]) for k in sorted(corr, key=float)) + " |" if corr else "",
        f"| Full 90-minute match analysed in | {fm['analysis_total_s']:.1f} s ({fm['x_realtime']:.0f}× real time) |",
        f"| Per-frame live stage, p50 / p99 | {pf['p50_ms']:.2f} / {pf['p99_ms']:.2f} ms (budget 40 ms) |",
        f"| Cause tracing after a shot, median | {fm['genesis_per_chance_ms_median']:.1f} ms |",
        f"| Generator: distance per player per 90 / sprints per 90 | {re_['distance_km_per90_mean']:.1f} km / {re_['sprints_per90_mean']:.0f} |",
        "",
        "![Accuracy vs tracking noise](eval/results/noise_curve.png)",
    ]
    return "\n".join(x for x in lines if x is not None)


def main():
    r = json.loads((ROOT / "eval" / "results" / "results.json").read_text())
    readme = ROOT / "README.md"
    s = readme.read_text()
    new = "<!-- RESULTS:START -->\n## Results on a held-out test set\n\n" + block(r) + "\n<!-- RESULTS:END -->"
    s = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda _: new, s, flags=re.S)
    readme.write_text(s)
    print(block(r))


if __name__ == "__main__":
    main()
