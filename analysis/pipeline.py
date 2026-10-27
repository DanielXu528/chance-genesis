"""End-to-end analysis of one observation (tracking + events), no ground truth involved."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from analysis.genesis import ChanceAnalyzer, GenesisParams
from analysis.marking import MarkingParams, infer_marking
from analysis.offside import offside_series
from analysis.phases import phases
from analysis.preprocess import Track, preprocess


@dataclass
class Analysis:
    track: Track
    phases: dict
    marking: dict
    offside: np.ndarray
    moments: list
    timings: dict = field(default_factory=dict)


def analyze(obs: dict, marking_params: MarkingParams | None = None, genesis_params: GenesisParams | None = None,
            method: str = "viterbi") -> Analysis:
    t0 = time.perf_counter()
    tr = preprocess(obs)
    t1 = time.perf_counter()
    ph = phases(tr)
    mk = infer_marking(tr, ph["defending"], marking_params, method=method)
    t2 = time.perf_counter()
    off = offside_series(tr.pos, tr.ball, tr.possession)
    ca = ChanceAnalyzer(tr, mk, genesis_params)
    moments = []
    lat = []
    for ev in tr.events:
        if ev["type"] == "shot":
            a = time.perf_counter()
            moments.append(ca.analyze(ev))
            lat.append(time.perf_counter() - a)
    t3 = time.perf_counter()
    return Analysis(tr, ph, mk, off, moments,
                    {"preprocess_s": t1 - t0, "marking_s": t2 - t1, "genesis_s": t3 - t2,
                     "genesis_per_chance_ms": [x * 1000 for x in lat]})
