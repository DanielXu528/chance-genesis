"""Exercises the real Agent Framework path of the Narrator (Agent.run with structured output)
using a local fake chat client, so the LLM wiring is tested without a model endpoint."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_framework import BaseChatClient, ChatResponse, Message

from agents.narrator import LLMWriter
from agents.templates import template_sentences
from agents.workflow import narrate
from analysis.pipeline import analyze
from gen.episodes import make_episode, observe


class FakeModel(BaseChatClient):
    """Answers like a model asked for NarrationModel JSON. First answer invents a number."""

    def __init__(self, pack):
        super().__init__()
        self.pack = pack
        self.calls = 0
        self.seen_options = []

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        self.calls += 1
        self.seen_options.append(options)
        sents = template_sentences(self.pack)
        if self.calls == 1:
            sents = [dict(s) for s in sents]
            sents[0]["text"] += " It was 77.7 m out."
        text = json.dumps({"sentences": sents})
        return ChatResponse(messages=[Message(role="assistant", contents=[text])], response_id=f"fake-{self.calls}")


def test_llm_writer_through_agent_framework():
    ep = make_episode("decoy_run", 10001)
    pack = analyze(observe(ep, 0.3, seed=1)).moments[-1]
    model = FakeModel(pack)
    n = asyncio.run(narrate(pack, LLMWriter(model)))
    assert model.calls == 2, model.calls                     # rejected once, accepted on retry
    assert n.verified and n.source == "llm" and n.attempts == 2
    assert "77.7" in " ".join(n.history[0]["problems"])
    # the second prompt carried the verifier's feedback
    assert any("response_format" in (o or {}) for o in model.seen_options)


class ReasoningModel(FakeModel):
    """Behaves like a reasoning deployment behind an opaque name: rejects `temperature`."""

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        if (options or {}).get("temperature") is not None:
            self.seen_options.append(options)
            raise RuntimeError("Unsupported parameter: 'temperature' is not supported with this model.")
        self.calls += 1
        self.seen_options.append(options)
        text = json.dumps({"sentences": template_sentences(self.pack)})
        return ChatResponse(messages=[Message(role="assistant", contents=[text])], response_id=f"fake-{self.calls}")


def test_llm_writer_drops_unsupported_temperature(monkeypatch):
    monkeypatch.setenv("FOUNDRY_MODEL", "narrator-prod")      # name does not reveal a reasoning model
    ep = make_episode("decoy_run", 10001)
    pack = analyze(observe(ep, 0.3, seed=1)).moments[-1]
    model = ReasoningModel(pack)
    writer = LLMWriter(model)
    n = asyncio.run(narrate(pack, writer))
    assert n.verified and n.source == "llm" and n.attempts == 1
    assert writer._opts_index == 1                           # remembered: no temperature for this deployment
    n2 = asyncio.run(narrate(pack, writer, "es"))
    assert n2.source == "llm"
    assert all((o or {}).get("temperature") is None for o in model.seen_options[2:])


if __name__ == "__main__":
    test_llm_writer_through_agent_framework()
    print("ok")
