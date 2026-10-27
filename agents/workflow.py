"""Narration workflow (Microsoft Agent Framework).

    evidence pack ──► Narrator ──► Verifier ──ok──────────────► verified narration
                         ▲            │
                         └─ rejected, │ attempt < MAX_ATTEMPTS (problems fed back)
                                      └─ still rejected ──► Fallback (template) ──► verified narration

Roles: the Narrator (LLM agent) writes; the Verifier (deterministic code) checks every
number, time and player id against the evidence pack and the fields each sentence cites;
the Fallback writes a template narration that is guaranteed to verify. The evidence pack
is the shared state every executor reads; nothing downstream can add facts to it.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from typing_extensions import Never

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler

from agents.templates import join, template_sentences
from agents.verifier import verify

MAX_ATTEMPTS = 2


@dataclass
class NarrationRequest:
    pack: dict
    language: str = "en"
    attempt: int = 1
    feedback: list = field(default_factory=list)
    history: list = field(default_factory=list)
    t0: float = 0.0


@dataclass
class Draft:
    req: NarrationRequest
    sentences: list
    source: str
    error: Optional[str] = None


@dataclass
class FallbackRequest:
    req: NarrationRequest
    problems: list


@dataclass
class Narration:
    moment_id: str
    text: str
    sentences: list
    verified: bool
    source: str            # "llm" | "template"
    attempts: int
    history: list          # [{attempt, source, problems}] for every rejected draft
    checked: dict
    language: str
    latency_ms: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class NarratorExecutor(Executor):
    def __init__(self, writer, id: str = "narrator"):
        super().__init__(id=id)
        self.writer = writer

    @handler
    async def write(self, req: NarrationRequest, ctx: WorkflowContext[Draft]) -> None:
        try:
            sentences = await self.writer(req.pack, req.feedback, req.language)
            await ctx.send_message(Draft(req, sentences, getattr(self.writer, "source", "llm")))
        except Exception as e:  # model outage, timeout, malformed output: the verifier routes it on
            await ctx.send_message(Draft(req, [], getattr(self.writer, "source", "llm"), error=f"{type(e).__name__}: {e}"))


class VerifierExecutor(Executor):
    def __init__(self, id: str = "verifier"):
        super().__init__(id=id)

    @handler
    async def check(self, d: Draft, ctx: WorkflowContext[NarrationRequest | FallbackRequest, Narration]) -> None:
        req = d.req
        if d.error:
            problems = [f"narrator failed: {d.error}"]
            rep = None
        else:
            rep = verify(req.pack, d.sentences)
            problems = rep.problems
        if rep is not None and rep.ok:
            await ctx.yield_output(Narration(
                moment_id=req.pack.get("moment_id", req.pack.get("window", "")), text=join(d.sentences),
                sentences=d.sentences, verified=True, source=d.source, attempts=req.attempt, history=req.history,
                checked=rep.checked, language=req.language, latency_ms=(time.perf_counter() - req.t0) * 1000))
            return
        history = req.history + [{"attempt": req.attempt, "source": d.source, "problems": problems,
                                  "draft": join(d.sentences) if d.sentences else None}]
        if req.attempt < MAX_ATTEMPTS and not d.error:
            await ctx.send_message(NarrationRequest(req.pack, req.language, req.attempt + 1, problems, history, req.t0),
                                   target_id="narrator")
        else:
            await ctx.send_message(FallbackRequest(NarrationRequest(req.pack, req.language, req.attempt, problems,
                                                                    history, req.t0), problems),
                                   target_id="fallback")


class FallbackExecutor(Executor):
    def __init__(self, id: str = "fallback"):
        super().__init__(id=id)

    @handler
    async def fallback(self, r: FallbackRequest, ctx: WorkflowContext[Never, Narration]) -> None:
        req = r.req
        sentences = template_sentences(req.pack)
        rep = verify(req.pack, sentences)
        await ctx.yield_output(Narration(
            moment_id=req.pack.get("moment_id", req.pack.get("window", "")), text=join(sentences), sentences=sentences,
            verified=rep.ok, source="template", attempts=req.attempt, history=req.history, checked=rep.checked,
            language="en", latency_ms=(time.perf_counter() - req.t0) * 1000))


def build_workflow(writer):
    narrator = NarratorExecutor(writer)
    verifier = VerifierExecutor()
    fallback = FallbackExecutor()
    return (WorkflowBuilder(start_executor=narrator, name="chance-narration",
                            description="Narrator -> Verifier -> (retry | template fallback)")
            .add_edge(narrator, verifier)
            .add_edge(verifier, narrator)
            .add_edge(verifier, fallback)
            .build())


async def narrate(pack: dict, writer, language: str = "en") -> Narration:
    wf = build_workflow(writer)
    result = await wf.run(NarrationRequest(pack, language, t0=time.perf_counter()))
    outs = result.get_outputs()
    return outs[-1]


def narrate_sync(pack: dict, writer, language: str = "en") -> Narration:
    return asyncio.run(narrate(pack, writer, language))


# ---------------------------------------------------------------------- tactical analyst
def summary_pack(moments: list[dict], t_from: float, t_to: float) -> dict:
    """Shared state for the Tactical Analyst: the verified moments in a time window."""
    from collections import Counter
    from common.pitch import fmt_clock
    sel = [m for m in moments if t_from <= m["t_s"] < t_to]
    by_team: dict = {}
    for m in sel:
        by_team.setdefault(m["team"], Counter())[m["cause"]] += 1
    defenders = Counter()
    for m in sel:
        p = m.get("players", {})
        for role in ("dragged", "defender"):
            if role in p:
                defenders[p[role]] += 1
    return {
        "type": "period_summary", "window": f"{fmt_clock(t_from)}-{fmt_clock(t_to)}", "n_chances": len(sel),
        "by_team": {t: dict(c) for t, c in sorted(by_team.items())},
        "team_totals": {t: sum(c.values()) for t, c in sorted(by_team.items())},
        "repeat_defenders": [{"player": p, "count": n} for p, n in defenders.most_common() if n >= 2],
        "moments": [{"t": m["t"], "team": m["team"], "shooter": m["shooter"], "cause": m["cause"],
                     "players": m.get("players", {})} for m in sel],
        "cause": "none",
    }
