"""Fan and analyst views: same facts, different writing; both must pass the verifier."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.templates import join, template_sentences
from agents.verifier import verify
from analysis.pipeline import analyze
from gen.episodes import make_episode, observe

TEAMS = {"A": "Northbridge Athletic", "B": "Harbour Vale"}
CASES = [("decoy_run", 10001), ("lost_marker", 11000), ("press_turnover", 12000), ("neg_long", 15000),
         ("neg_outpaced", 14000)]


def _pack(kind, seed):
    m = analyze(observe(make_episode(kind, seed), 0.3, seed=seed)).moments[-1]
    return m


def test_both_views_verify_and_fan_uses_shirt_numbers():
    for kind, seed in CASES:
        m = _pack(kind, seed)
        assert m["key_player"]["id"] and m["stats"]["players"], kind
        for aud in ("analyst", "fan"):
            pack = {**m, "audience": aud, "teams": TEAMS}
            sents = template_sentences(pack)
            rep = verify(pack, sents)
            assert rep.ok, (kind, aud, rep.problems, join(sents))
        fan = join(template_sentences({**m, "audience": "fan", "teams": TEAMS}))
        assert fan.startswith("The key player in this chance was")
        assert "No. " in fan and m["key_player"]["id"] not in fan        # shirt numbers, not ids


def test_verifier_rejects_unknown_shirt_number():
    m = _pack("decoy_run", 10001)
    pack = {**m, "audience": "fan", "teams": TEAMS}
    sents = template_sentences(pack)
    import json
    import re
    ids = {int(x[1:]) for x in re.findall(r"\b[AB](?:1[01]|[1-9])\b", json.dumps(m))}
    ghost = next(n for n in range(11, 0, -1) if n not in ids)
    bad = [dict(sents[0], text=sents[0]["text"] + f" No. {ghost} watched.")] + sents[1:]
    rep = verify(pack, bad)
    assert not rep.ok and any("shirt number" in p for p in rep.problems)
