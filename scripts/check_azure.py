"""Smoke test for the Azure setup: narrates one demo chance through the real workflow
(Narrator agent -> Verifier -> retry / fallback) and reports which backend answered.

Usage:  python -m scripts.check_azure [--moment <moment_id>] [--language en|es|fr|zh|pt]
Needs the demo match (python -m server.prepare) and, for the model path, either
FOUNDRY_PROJECT_ENDPOINT + FOUNDRY_MODEL (after `az login`) or AZURE_OPENAI_* variables.
Exit code 0 only when a language-model draft passed the verifier.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from agents.narrator import backend_name, default_writer
from agents.telemetry import setup_tracing
from agents.workflow import narrate

ROOT = Path(__file__).resolve().parents[1]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moment", default=None)
    ap.add_argument("--language", default="en")
    args = ap.parse_args()
    f = (Path(os.environ["CG_DATA_DIR"]) if os.environ.get("CG_DATA_DIR") else ROOT / "data" / "demo") / "moments.json"
    if not f.exists():
        print("No demo match yet: run  python -m server.prepare  first.")
        return 2
    moments = json.loads(f.read_text())
    if args.moment:
        pack = next((m for m in moments if m["moment_id"] == args.moment), None)
        if pack is None:
            print(f"Unknown moment {args.moment}. Available: {', '.join(m['moment_id'] for m in moments)}")
            return 2
    else:
        pack = next((m for m in moments if m["cause"] == "decoy_run"), moments[0])
    tracing = setup_tracing("chance-genesis-check")
    print(f"Backend:  {backend_name()}")
    print(f"Tracing:  {'Application Insights' if tracing else 'off (set APPLICATIONINSIGHTS_CONNECTION_STRING)'}")
    print(f"Moment:   {pack['moment_id']} ({pack['cause']})\n")
    n = await narrate(pack, default_writer(), args.language)
    print(f"Source:   {n.source}   verified: {n.verified}   attempts: {n.attempts}   {n.latency_ms:.0f} ms")
    for h in n.history:
        print(f"  rejected draft {h.get('attempt')}: {'; '.join(h.get('problems', []))[:300]}")
    print(f"Checked:  {n.checked}\n")
    print(n.text)
    if n.source != "llm":
        print("\nThe language model did not produce the accepted narration (offline mode, model error, or two "
              "rejected drafts). Check the variables above and that your identity has the Foundry User role.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
