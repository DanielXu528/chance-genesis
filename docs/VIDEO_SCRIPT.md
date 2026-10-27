# Demo video script (target 1:50, hard limit 2:00)

Record the browser at 1920×1080 with the app running locally or on Azure Container Apps. No music (or a
royalty-free track at low volume); no third-party logos. Voice-over lines are in quotes; on-screen actions
in brackets. Times are cumulative.

**0:00–0:10 — the question**
[Replays mode, the demo goal paused just before the shot.]
"Every goal starts the same argument in the studio: why was he free? The replay shows what happened. It
doesn't show why."

**0:10–0:28 — live, then replay time**
[Open `/?live=1&start=125150` (about 30 s of match time before the 83:56 chance) or click Live. The match streams at 6×. A shot: the banner shows the cause, then "Explanation ready … ms
after the shot. Rolling the replay." The replay starts at 0.5×.]
"Chance Genesis watches 25 Hz tracking data. The moment a shot is taken it goes back ten seconds, finds
the cause, and has a checked explanation ready before the replay even starts."

**0:28–0:55 — decoy run**
[Decoy chance at 83:56. Let the yellow arrows draw the striker's run and the dragged centre-back. Pause on the
reception frame: the yellow vacated space and the dashed ghost defender are visible. Point at the cause
card and the "What if" boxes.]
"Here the striker's run drags his marker wide. The yellow area is the space that opened. The dashed
circle is where the defender would have been if he'd stayed: the space around the receiver would have
been 54 square metres instead of 128.
And the verdict: he followed his man. Pulled out by design, not out of position."

**0:55–1:10 — same picture, different story**
[Open the lost-marker chance at 42:52. Click chain step 2 to jump to the frame where the marker stops.]
"This one looks the same on the replay, but the tracking says otherwise: the marker stayed below jogging
pace for 2.2 seconds while his man sprinted, and the gap grew to 14.5 metres. That's a lost marker, not a
decoy."

**1:10–1:20 — pressing trap**
[Open the pressing-trap goal at 46:47 with Passing lanes on. Red dashed lanes are blocked.]
"And here three players pressed the centre-back and cut his passing lanes to four team-mates before the
interception, 16 metres from his own goal."

**1:20–1:40 — why you can trust it**
[Expand "Evidence pack" briefly; point at "✓ Verified: N numbers, N player ids…". Then show
eval/results/RESULTS.md or the README results table and the noise curve.]
"The language model only writes the sentence. Every number, time and player comes from code, and a
verifier rejects any draft that adds a fact. And we measured it: on 280 held-out scenarios, including
hard negatives designed to fool it, it names the right cause in 90 percent of chances, with 3 percent false alarms."

**1:40–1:55 — how it's built**
[Show the README architecture diagram.]
"Microsoft Agent Framework orchestrates the narrator, verifier and fallback on Foundry, the app runs on
Azure Container Apps, and a full match is analysed more than 400 times faster than real time."

**1:55–2:00 — close**
[Repo URL on screen.]
"Chance Genesis. The why, before the replay ends."
