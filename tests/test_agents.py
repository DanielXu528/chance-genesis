"""Workflow tests with offline writers (no model endpoint needed)."""
import asyncio
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.templates import template_sentences
from agents.verifier import verify
from agents.workflow import narrate
from analysis.pipeline import analyze
from gen.episodes import make_episode, observe


def _packs():
    out = []
    for kind, seed in [("decoy_run", 201), ("lost_marker", 201), ("press_turnover", 201), ("neg_long", 201), ("neg_outpaced", 200)]:
        ep = make_episode(kind, seed)
        out.append(analyze(observe(ep, 0.3, seed=seed)).moments[-1])
    return out


class FaultyThenGood:
    """First draft invents a number; the second is clean."""
    source = "llm"

    def __init__(self):
        self.calls = 0

    async def __call__(self, pack, feedback, language="en"):
        self.calls += 1
        s = template_sentences(pack)
        if self.calls == 1:
            s = copy.deepcopy(s)
            s[0]["text"] += " He was 99.9 m clear."
        return s


class AlwaysBad:
    source = "llm"

    async def __call__(self, pack, feedback, language="en"):
        return [{"text": f"{pack['shooter']} made a terrible mistake, 123 m out at 99:99.", "cites": ["nope"]}]


class Crashes:
    source = "llm"

    async def __call__(self, pack, feedback, language="en"):
        raise TimeoutError("model endpoint timed out")


def test_templates_verify():
    for p in _packs():
        r = verify(p, template_sentences(p))
        assert r.ok, (p["cause"], r.problems)


def test_retry_then_pass():
    p = _packs()[0]
    w = FaultyThenGood()
    n = asyncio.run(narrate(p, w))
    assert n.verified and n.source == "llm" and n.attempts == 2 and w.calls == 2
    assert any("99.9" in prob for prob in n.history[0]["problems"])


def test_fallback_after_two_failures():
    p = _packs()[1]
    n = asyncio.run(narrate(p, AlwaysBad()))
    assert n.verified and n.source == "template" and n.attempts == 2
    probs = " ".join(n.history[0]["problems"])
    assert "123" in probs and "99:99" in probs and "blame" in probs and "nope" in probs


def test_fallback_on_crash():
    p = _packs()[2]
    n = asyncio.run(narrate(p, Crashes()))
    assert n.verified and n.source == "template" and "timed out" in n.history[0]["problems"][0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
