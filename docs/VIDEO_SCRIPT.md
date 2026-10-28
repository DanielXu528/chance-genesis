# Demo video script (target 1:55, hard limit 2:00)

Record the browser at 1920×1080 on the live site (https://chance-genesis-live-837.azurewebsites.net), full
screen, no music, no third-party logos. Voice-over in quotes (about 260 words, speak briskly); on-screen
actions in brackets. Clock times refer to the demo match (seed 27).

Before recording: open the site 3 minutes early so every narration is pre-written, and check that the
line under the narration says "written by Microsoft Foundry (gpt-5-mini)".

**0:00–0:08 — the question**
[`/?match=1&t=4590`: full match view, paused at 76:30.]
"Every goal starts the same studio argument: why was he free? The replay shows what happened. It doesn't
show why."

**0:08–0:22 — the match, then the explanation**
[Press Play (4×). At 76:37 the banner appears; click "Explain this chance". The replay starts at 0.5×.]
"Chance Genesis watches 25 hertz tracking data. When a shot happens, it traces the previous ten seconds,
finds the cause, and has a verified explanation ready before the replay starts."

**0:22–0:45 — decoy run**
[Let the replay run; drag to the reception frame. Point at the yellow area, the dashed circle and the
"What if" boxes.]
"Here, the striker's run drags his marker fifteen metres wide. The yellow area is the space that opened.
The dashed circle is where the defender would have been: the receiver would have had 33 square metres
instead of 127. And the verdict: he followed his man. Pulled out by design, not out of position."

**0:45–0:53 — fan view**
[Click **Fan** at the top right: the key-player card for No. 9 appears.]
"For fans at home, the same checked facts become a story about the key player."

**0:53–1:06 — same picture, different story**
[Click **Analyst**. Click the orange dot at 05:52 on the timeline, then chain step 2.]
"This chance looks the same on replay, but the tracking disagrees: the marker stopped for two seconds
while his man sprinted. That's a lost marker, not a decoy."

**1:06–1:16 — pressing trap**
[Click the purple dot at 48:03 (below the line). Passing lanes are on; red dashed lanes are blocked.]
"Here the press closed the defender down from fifteen metres to two, cut his passing lanes, and won the
ball 35 metres from goal."

**1:16–1:36 — why you can trust it**
[Scroll the side panel to "✓ Verified … written by Microsoft Foundry (gpt-5-mini)". Then switch to the
GitHub README results table.]
"The language model only writes the words. Every number, time and player comes from code, and a verifier
rejects any draft that adds a fact. On 280 held-out test scenarios, including traps designed to fool it,
it names the right cause 94 percent of the time, with one percent false alarms."

**1:36–1:52 — how it's built**
[README architecture diagram.]
"It's built with Microsoft Agent Framework: a narrator, a fan commentator and a tactical analyst, checked
by a verifier with retry and fallback, running on Microsoft Foundry and Azure App Service."

**1:52–2:00 — close**
[Repo URL on screen: github.com/DanielXu528/chance-genesis]
"Chance Genesis. The why, before the replay ends."
