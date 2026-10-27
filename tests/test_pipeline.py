"""Pipeline tests: determinism, truth isolation, and end-to-end attribution on fixed seeds."""
import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.lanes import lanes_at
from analysis.offside import offside_line
from analysis.pipeline import analyze
from gen.episodes import make_episode, observe


def test_analysis_never_imports_generator():
    """The analysis layer must work on real data: it may not import the generator or labels."""
    for f in (ROOT / "analysis").glob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("gen"), f"{f.name} imports {node.module}"
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith("gen") for a in node.names), f.name


def test_observation_has_no_truth():
    ep = make_episode("decoy_run", 101)
    obs = observe(ep, 0.3, seed=1)
    assert set(obs) == {"meta", "pos", "ball", "events"}
    assert set(obs["meta"]) <= {"fps", "n_frames", "players", "formations", "noise_m", "noise_kind"}
    assert "decoy" not in str(obs)


def test_generator_is_deterministic():
    a = make_episode("lost_marker", 123)
    b = make_episode("lost_marker", 123)
    assert np.array_equal(a["pos"], b["pos"]) and a["truth"] == b["truth"]


def test_offside_and_lanes_sanity():
    pos = np.zeros((22, 2))
    pos[:11, 0] = np.linspace(10, 60, 11)
    pos[11:, 0] = np.linspace(40, 100, 11)
    pos[:, 1] = 34
    assert offside_line(pos, np.array([50.0, 34.0]), 0) == np.sort(pos[11:, 0])[-2]
    r = lanes_at(pos, 5)
    assert len(r["mates"]) == len(r["blocked"]) > 0


def test_end_to_end_attribution():
    expect = {("decoy_run", 10001): "decoy_run", ("lost_marker", 11000): "lost_marker",
              ("press_turnover", 12000): "press_turnover", ("neg_long", 15000): "none"}
    for (kind, seed), cause in expect.items():
        ep = make_episode(kind, seed)
        m = analyze(observe(ep, 0.3, seed=seed)).moments[-1]
        assert m["cause"] == cause, (kind, seed, m["cause"])
        assert m["chain"], kind
