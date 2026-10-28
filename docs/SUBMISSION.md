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
counterfactual ("had B4 held his position instead of following A9, the space around the receiver would
have been 33 m², not 127 m²").

Unlike tactic dashboards, it reports how often it is right. A synthetic match generator with hidden
labels (built with a different marking mechanism from the detector) lets us score the system on 280
held-out scenarios, including four kinds of hard negatives designed to fool it, and under correlated
tracking noise: it names the right cause in 94.3% of chances (cause and key players right in 88.3% of
real causes) and wrongly assigns a tactical cause to only 1.2% of the hard negatives.

The same verified facts are personalised for two audiences: an **Analyst** view with the full tactical
breakdown and measured numbers (speeds, distances, shot speed and angle, key pass), and a **Fan** view
that names the key player in plain words ("The key player in this chance was Northbridge Athletic's No. 9.
His decoy run wide dragged Harbour Vale's No. 4 out of position…"), in English, Spanish, French or Chinese.

## How we built it
- Deterministic analysis in Python/NumPy/SciPy: Hungarian-algorithm marking inference with HMM/Viterbi
  smoothing, interception-time passing-lane model, time-to-intercept space model, rule-based cause tracing
  with counterfactuals. A full 90-minute match is analysed in about 8 s (≈670× real time); the per-frame
  live stage takes 0.2 ms (p99 0.4 ms); tracing a chance after the shot takes about 1.5 ms.
- **Microsoft Agent Framework** workflow: a Narrator agent (analyst view) and a Fan Commentator agent (Microsoft Foundry / Azure OpenAI, structured
  output with field citations) → a code Verifier that checks every number, time and player id against the
  evidence pack → retry once with the problems → template fallback. A Tactical Analyst agent writes
  half-time studio notes through the same verified workflow.
- A synthetic match generator built to look like football: zonal defending with ball-side pressure,
  players who find space and dribble into it, a simulated ball (passes that can be cut out or run loose),
  goal kicks and kick-offs.
- FastAPI and a canvas studio UI: the full match on a scrubbable timeline, replays with telestrator
  overlays, and a WebSocket path for a live feed. Deployed to **Azure App Service** with managed
  identity to Foundry.

## Microsoft / Azure technologies
Microsoft Agent Framework (Agent, structured output, WorkflowBuilder/executors), Microsoft Foundry and
Azure OpenAI, Azure App Service (Azure Container Apps script included), Azure Monitor / Application
Insights (OpenTelemetry tracing), Azure Identity (managed identity).

## Problem it solves
Broadcasters and streaming platforms get an explanation of every chance during the replay instead of
after the break, with numbers that can be trusted: the language model only writes, code supplies and
checks every fact, and the detection itself has a published accuracy.

## Data statement
All data is synthetic. The league, clubs (Northbridge Athletic, Harbour Vale) and players are fictional.
No real match data, club names or trademarks are used.
