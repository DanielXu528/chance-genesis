# Demo video script (target 1:50, hard limit 2:00)

Record the browser at 1920×1080 with the app running locally or on Azure Container Apps. No music (or a
royalty-free track at low volume); no third-party logos. Voice-over lines are in quotes; on-screen actions
in brackets. Times are cumulative. Clock times refer to the default demo match (seed 27).

**0:00–0:10 — the question**
[Full match view, paused at 76:30, the attack building down the right.]
"Every goal starts the same argument in the studio: why was he free? The replay shows what happened. It
doesn't show why."

**0:10–0:28 — the match, then the replay**
[Open `/?match=1&t=4580` (76:20) or drag the bar there. Press Play at 4×. The goal goes in at 76:37: the
banner names the cause; click "Explain this chance". The replay starts at 0.5×.]
"Chance Genesis watches 25 Hz tracking data. The moment a shot is taken it goes back ten seconds, finds
the cause, and has a checked explanation ready in milliseconds, before the replay even starts."

**0:28–0:55 — decoy run**
[Decoy goal at 76:37. Let the yellow arrows draw A9's wide run and B4 following him. Pause on the
reception frame: the yellow vacated space and the dashed ghost defender are visible. Point at the cause
card and the "What if" boxes.]
"Here the striker's run drags his marker 15 metres wide. The yellow area is the space that opened. The
dashed circle is where the defender would have been if he'd stayed: the space around the receiver would
have been 33 square metres instead of 127, with a defender a metre away.
And the verdict: he followed his man. Pulled out by design, not out of position."

**0:55–1:10 — same picture, different story**
[Open the lost-marker chance at 05:52. Click chain step 2 to jump to the frame where the marker stops.]
"This one looks the same on the replay, but the tracking says otherwise: the marker stayed below jogging
pace for two seconds while his man sprinted, and the gap grew from 1.3 to 15 metres. That's a lost
marker, not a decoy."

**1:10–1:20 — pressing trap**
[Open the pressing-trap chance at 48:03 with Passing lanes on. Red dashed lanes are blocked.]
"And here the press closed the defender down from 15 metres to 2, cut his passing lanes, and won the ball
35 metres from his goal. The shot came one second later."

**1:20–1:40 — why you can trust it**
[Expand "Evidence pack" briefly; point at "✓ Verified: N numbers, N player ids…". Then show
eval/results/RESULTS.md or the README results table and the noise curve.]
"The language model only writes the sentence. Every number, time and player comes from code, and a
verifier rejects any draft that adds a fact. And we measured it: on 280 held-out scenarios, including
hard negatives designed to fool it, it names the right cause in 94 percent of chances, with 1 percent
false alarms."

**1:40–1:55 — how it's built**
[Show the README architecture diagram.]
"Microsoft Agent Framework orchestrates the narrator, verifier and fallback on Foundry, the app runs on
Azure Container Apps, and a full match is analysed more than 600 times faster than real time."

**1:55–2:00 — close**
[Repo URL on screen.]
"Chance Genesis. The why, before the replay ends."
