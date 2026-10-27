"""Narrator and Tactical Analyst agents (Microsoft Agent Framework).

The LLM only turns an evidence pack into words. It never sees tracking data and never
decides causes; every sentence must cite the evidence fields it uses, and the Verifier
checks every number, time and player id against the pack.

Model endpoint (first one configured wins)
  FOUNDRY_PROJECT_ENDPOINT + FOUNDRY_MODEL      Microsoft Foundry project (Entra ID auth)
  AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT [+ AZURE_OPENAI_API_KEY]   Azure OpenAI
  none of the above                              offline: deterministic template writer
"""
from __future__ import annotations

import json
import os
from typing import Optional

from pydantic import BaseModel, Field

from agents.templates import template_sentences

LANG_NAMES = {"en": "English", "es": "Spanish", "zh": "Simplified Chinese", "fr": "French", "pt": "Portuguese"}

NARRATOR_INSTRUCTIONS = """You are the replay analyst in a football broadcast studio. While the replay of a chance
runs, you explain in a few seconds HOW the chance was created, using ONLY the evidence pack you are given.
The pack was produced by deterministic tracking analysis; treat it as the only source of truth.

Rules
- 3 to 5 short sentences, at most 90 words, broadcast tone, present the cause first.
- Use player ids exactly as written (e.g. A9, B4). Use clock times exactly as written (e.g. 63:12).
- Every number you write must appear in the pack (you may round to 0 or 1 decimals). Do not compute new
  numbers such as differences or percentages.
- For every sentence, list in `cites` the JSON paths of the pack fields it uses, e.g. "chain[0].dragged_m",
  "counterfactual.space_m2_counterfactual". Paths use dots for keys and [i] for list items.
- Explain the counterfactual when the pack has one ("if B4 had stayed ... the space would have been ...").
- Never blame or judge players (no "mistake", "error", "lazy"). Say what the tracking model shows.
  For verdict_type "pulled_by_design" say the defender followed his assignment; for "switched_off" say the
  model shows him stopping tracking; for "forced_error" credit the press.
- If cause is "none", say no tactical cause was found and describe the shot.
- For a pack of type "period_summary", write a studio note about patterns across the listed chances.
- Write in {language}. Keep player ids and clock times unchanged.
"""


class Sentence(BaseModel):
    text: str = Field(description="One sentence of narration.")
    cites: list[str] = Field(description="JSON paths of the evidence-pack fields this sentence uses.")


class NarrationModel(BaseModel):
    sentences: list[Sentence]


def make_chat_client():
    """Returns an Agent Framework chat client, or None for offline mode."""
    if os.getenv("FOUNDRY_PROJECT_ENDPOINT"):
        from agent_framework.foundry import FoundryChatClient
        from azure.identity.aio import DefaultAzureCredential
        return FoundryChatClient(project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
                                 model=os.getenv("FOUNDRY_MODEL", "gpt-4.1-mini"),
                                 credential=DefaultAzureCredential())
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        from agent_framework.openai import OpenAIChatCompletionClient
        kw = {"model": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"),
              "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
              "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")}
        if os.getenv("AZURE_OPENAI_API_KEY"):
            kw["api_key"] = os.environ["AZURE_OPENAI_API_KEY"]
        else:
            from azure.identity.aio import DefaultAzureCredential
            kw["credential"] = DefaultAzureCredential()
        return OpenAIChatCompletionClient(**kw)
    return None


def backend_name() -> str:
    if os.getenv("FOUNDRY_PROJECT_ENDPOINT"):
        return f"Microsoft Foundry ({os.getenv('FOUNDRY_MODEL', 'gpt-4.1-mini')})"
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return f"Azure OpenAI ({os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4.1-mini')})"
    return "offline template writer"


class LLMWriter:
    """Narrator backed by an Agent Framework `Agent` with structured output."""
    source = "llm"

    def __init__(self, client, name: str = "Narrator"):
        from agent_framework import Agent
        self.client = client
        self.name = name
        self._agents: dict[str, object] = {}
        self._Agent = Agent

    def _agent(self, language: str):
        if language not in self._agents:
            self._agents[language] = self._Agent(
                self.client, NARRATOR_INSTRUCTIONS.replace("{language}", LANG_NAMES.get(language, "English")),
                name=self.name)
        return self._agents[language]

    async def __call__(self, pack: dict, feedback: list[str], language: str = "en") -> list[dict]:
        prompt = "Evidence pack:\n" + json.dumps(pack, ensure_ascii=False)
        if feedback:
            prompt += ("\n\nYour previous draft was rejected by the verifier for these reasons. Fix every one:\n- "
                       + "\n- ".join(feedback))
        resp = await self._agent(language).run(prompt, options={"response_format": NarrationModel, "temperature": 0.3})
        value = resp.value
        if value is None:
            value = NarrationModel.model_validate_json(resp.text)
        return [s.model_dump() for s in value.sentences]


class TemplateWriter:
    """Offline writer: the deterministic templates (always verifiable)."""
    source = "template"

    async def __call__(self, pack: dict, feedback: list[str], language: str = "en") -> list[dict]:
        return template_sentences(pack)


def default_writer() -> "LLMWriter | TemplateWriter":
    client = make_chat_client()
    return LLMWriter(client) if client is not None else TemplateWriter()
