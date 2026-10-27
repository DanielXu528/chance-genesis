# Submission text

## Project name
Chance Genesis — why every chance happened, explained while the replay runs

## Short description (one line)
Traces every shot back to its tactical cause from 25 Hz tracking data, quantifies it with a
counterfactual, and hands the studio a verified explanation in milliseconds, with accuracy measured
against ground truth.

## What it does
When a goal goes in, the studio's question is not *what* happened but *why it was open*. Chance Genesis
watches synthetic 25 Hz tracking data and, the moment a shot is taken, goes back ten seconds to find the
cause: a decoy run that dragged a defender out of the space, a marker who stopped tracking his runner, or
a pressing trap that forced a turnover. It separates a defender who was pulled out by design from one who
switched off — two situations that look identical on the replay — and quantifies each with a
counterfactual ("had A4 held his position instead of following B10, the space around the receiver would
have been 54 m², not 128 m²").

Unlike tactic dashboards, it reports how often it is right. A synthetic match generator with hidden
labels (built with a different marking mechanism from the detector) lets us score the system on 280
held-out scenarios, including four kinds of hard negatives designed to fool it, and under correlated
tracking noise: it names the right cause in 90.4% of chances (cause and key players right in 80.8% of
real causes) and wrongly assigns a tactical cause to only 3.1% of the hard negatives.

## How we built it
- Deterministic analysis in Python/NumPy/SciPy: Hungarian-algorithm marking inference with HMM/Viterbi
  smoothing, interception-time passing-lane model, time-to-intercept space model, rule-based cause tracing
  with counterfactuals. A full 90-minute match is analysed in about 11 s (≈470× real time); the per-frame
  live stage takes 0.2 ms (p99 0.6 ms); tracing a chance after the shot takes about 3 ms.
- **Microsoft Agent Framework** workflow: a Narrator agent (Microsoft Foundry / Azure OpenAI, structured
  output with field citations) → a code Verifier that checks every number, time and player id against the
  evidence pack → retry once with the problems → template fallback. A Tactical Analyst agent writes
  half-time studio notes through the same verified workflow.
- FastAPI + WebSocket live feed and a canvas studio UI with telestrator overlays, deployed to **Azure
  Container Apps** with managed identity to Foundry.

## Microsoft / Azure technologies
Microsoft Agent Framework (Agent, structured output, WorkflowBuilder/executors), Microsoft Foundry and
Azure OpenAI, Azure Container Apps, Azure Identity (managed identity).

## Problem it solves
Broadcasters and streaming platforms get an explanation of every chance during the replay instead of
after the break, with numbers that can be trusted: the language model only writes, code supplies and
checks every fact, and the detection itself has a published accuracy.

## Data statement
All data is synthetic. The league, clubs (Northbridge Athletic, Harbour Vale) and players are fictional.
No real match data, club names or trademarks are used.
