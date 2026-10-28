"""Verifier: deterministic checks that a narration says nothing the evidence pack does not.

Checks
  1. every number, clock time and player id (or shirt number, in the fan view) in the text exists in
     the evidence pack
  2. every number in a sentence is found among the fields that sentence cites
  3. every cited field path exists
  4. no blame language (the system reports what the tracking model shows; it does not judge)
  5. the primary cause and its key players are actually mentioned
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

NUM_RE = re.compile(r"(?<![A-Za-z0-9:])(-?\d+(?:\.\d+)?)(?![\d:])")
CLOCK_RE = re.compile(r"\b(\d{1,3}:\d{2})\b")
ID_RE = re.compile(r"\b([AB](?:1[01]|[1-9]))\b")
# shirt-number references used in the fan view: "No. 9", "Nº 9", "#9", "9号"
SHIRT_RE = re.compile(r"(?:\bNo\.?\s?|\bN[º°]\s?|\bn\.º\s?|#)(1[01]|[1-9])\b|(1[01]|[1-9])\s?号")
BLAME = ["mistake", "blunder", "fault", "lazy", "careless", "terrible", "awful", "error", "useless", "asleep",
         "失误", "犯错", "懒"]
CAUSE_WORDS = {  # any one of these must appear (en / es / fr / zh)
    "decoy_run": ["decoy", "dragged", "pulled", "señuelo", "desmarque", "arrastr", "leurre", "appel", "entraîn",
                  "假跑", "拉扯", "带走"],
    "lost_marker": ["lost", "stopped tracking", "switched off", "gap", "marca", "marcaje", "marquage", "perdu",
                    "脱离", "盯丢", "停止"],
    "press_turnover": ["press", "forced", "presión", "presion", "pressing", "压迫", "逼抢"],
    "none": [],
}


@dataclass
class Report:
    ok: bool
    problems: list[str] = field(default_factory=list)
    checked: dict = field(default_factory=dict)


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """{'chain[0].dragged_m': 8.0, ...}"""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _numbers_in(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        if CLOCK_RE.fullmatch(value.strip()):
            return []
        return [float(x) for x in NUM_RE.findall(value)]
    return []


def _strings_in(value: Any) -> list[str]:
    return [value] if isinstance(value, str) else []


def _num_match(x: float, pool: list[float]) -> bool:
    for v in pool:
        if abs(x - v) < 1e-6:
            return True
        # allow the writer to round: 6.3 -> 6, 210.4 -> 210, 0.86 -> 86 (%)
        for d in (0, 1):
            if abs(round(v, d) - x) < 1e-6:
                return True
        if 0 < abs(v) <= 1 and abs(round(v * 100) - x) < 1e-6:
            return True
    return False


def _pack_index(pack: dict):
    flat = flatten(pack)
    nums, ids, clocks = [], set(), set()
    for v in flat.values():
        nums.extend(_numbers_in(v))
        for s in _strings_in(v):
            ids.update(ID_RE.findall(s))
            clocks.update(CLOCK_RE.findall(s))
    return flat, nums, ids, clocks


def verify(pack: dict, sentences: list[dict]) -> Report:
    """`sentences`: [{"text": str, "cites": [field paths]}]"""
    flat, nums, ids, clocks = _pack_index(pack)
    shirts = {int(pid[1:]) for pid in ids}
    problems: list[str] = []
    n_num = n_id = n_clock = 0
    full_text = " ".join(s.get("text", "") for s in sentences)
    for i, s in enumerate(sentences):
        text = s.get("text", "")
        cites = s.get("cites", []) or []
        for c in cites:
            if c not in flat and not any(k.startswith(c + ".") or k.startswith(c + "[") for k in flat):
                problems.append(f"sentence {i + 1}: cites a field that does not exist: {c}")
        cited_vals: list[float] = []
        for c in cites:
            for k, v in flat.items():
                if k == c or k.startswith(c + ".") or k.startswith(c + "["):
                    cited_vals.extend(_numbers_in(v))
        clean = CLOCK_RE.sub(" ", text)
        clean = ID_RE.sub(" ", clean)
        for m in SHIRT_RE.finditer(clean):
            n_id += 1
            num = int(m.group(1) or m.group(2))
            if num not in shirts:
                problems.append(f"sentence {i + 1}: shirt number {num} is not a player in the evidence")
        clean = SHIRT_RE.sub(" ", clean)
        for clk in CLOCK_RE.findall(text):
            n_clock += 1
            if clk not in clocks:
                problems.append(f"sentence {i + 1}: time {clk} is not in the evidence")
        for pid in ID_RE.findall(text):
            n_id += 1
            if pid not in ids:
                problems.append(f"sentence {i + 1}: player {pid} is not in the evidence")
        for raw in NUM_RE.findall(clean):
            x = float(raw)
            n_num += 1
            if not _num_match(x, nums):
                problems.append(f"sentence {i + 1}: number {raw} is not in the evidence")
            elif cites and not _num_match(x, cited_vals):
                problems.append(f"sentence {i + 1}: number {raw} is not in the fields it cites ({', '.join(cites)})")
    low = full_text.lower()
    for w in BLAME:
        if w in low:
            problems.append(f"blame language is not allowed: '{w}'")
    cause = pack.get("cause", "none")
    words = CAUSE_WORDS.get(cause, [])
    if words and not any(w in low for w in words):
        problems.append(f"the narration does not name the primary cause ({pack.get('cause_label')})")
    named_shirts = {int(m.group(1) or m.group(2)) for m in SHIRT_RE.finditer(full_text)}
    for role, pid in (pack.get("players") or {}).items():
        if isinstance(pid, str) and role in ("runner", "dragged", "defender", "ball_winner") \
                and pid not in full_text and int(pid[1:]) not in named_shirts:
            problems.append(f"key player {pid} ({role}) is not mentioned")
    return Report(ok=not problems, problems=problems, checked={"numbers": n_num, "player_ids": n_id, "clock_times": n_clock})
