"""Deterministic template narration. Used as the fallback when the LLM draft fails
verification twice, and as the offline writer when no model endpoint is configured.
Every sentence cites the evidence fields it uses, exactly like the LLM output."""
from __future__ import annotations


def _chain(pack: dict, action_prefix: str):
    for i, st in enumerate(pack.get("chain", [])):
        if str(st.get("action", "")).startswith(action_prefix):
            return i, st
    return None, None


def template_sentences(pack: dict) -> list[dict]:
    if pack.get("type") == "period_summary":
        return _summary(pack)
    cause = pack.get("cause", "none")
    what = "Goal" if pack.get("type") == "goal" else "Shot"
    S = [{"text": f"{pack['t']} {what} by {pack['shooter']}.", "cites": ["t", "shooter"]}]
    cf = pack.get("counterfactual") or {}
    if cause == "decoy_run":
        i, run = _chain(pack, "decoy run")
        j, rec = _chain(pack, "received")
        direction = run["action"].split("(")[-1].rstrip(")")
        S.append({"text": f"It was created by {run['actor']}'s decoy run: between {run['t']} he went {direction} and "
                          f"{run['dragged']}, who was marking him, was dragged {run['dragged_m']} m away.",
                  "cites": [f"chain[{i}]"]})
        S.append({"text": f"The space around the spot where the shot was set up grew from {run['space_m2'][0]:.0f} m² "
                          f"to {run['space_m2'][1]:.0f} m², and {rec['actor']} received there at {rec['t']} with the "
                          f"nearest defender {rec['nearest_defender_m']} m away.",
                  "cites": [f"chain[{i}].space_m2", f"chain[{j}]"]})
        S.append({"text": f"If {run['dragged']} had stayed where he was, the model puts that space at "
                          f"{cf['space_m2_counterfactual']:.0f} m² and the nearest defender "
                          f"{cf['nearest_defender_m_counterfactual']} m away.",
                  "cites": ["counterfactual"]})
        S.append({"text": f"The tracking model shows {run['dragged']} following his man rather than losing him: "
                          f"he was pulled out by design.", "cites": ["marker_role", "verdict_type"]})
    elif cause == "lost_marker":
        i, mk = _chain(pack, "marking")
        j, st = _chain(pack, "stopped tracking")
        k, rec = _chain(pack, "received")
        S.append({"text": f"It came from a lost marker: at {mk['t']} {mk['actor']} was {mk['gap_m']} m from "
                          f"{mk['marking']}, then stopped tracking him.", "cites": [f"chain[{i}]"]})
        S.append({"text": f"Between {st['t']} {st['actor']} averaged {st['defender_speed_ms']} m/s against "
                          f"{st['attacker_speed_ms']} m/s for {pack['shooter']}, stayed below jogging pace for "
                          f"{st['watching_s']} s, and the gap grew from {st['gap_m'][0]} m to {st['gap_m'][1]} m.",
                  "cites": [f"chain[{j}]"]})
        S.append({"text": f"{rec['actor']} received at {rec['t']} with the nearest defender {rec['nearest_defender_m']} m away; "
                          f"had {st['actor']} kept his starting gap, the model puts the nearest defender "
                          f"{cf['nearest_defender_m_counterfactual']} m away.",
                  "cites": [f"chain[{k}]", "counterfactual"]})
    elif cause == "press_turnover":
        i, pr = _chain(pack, "press")
        j, won = _chain(pack, "won the ball")
        k, sh = _chain(pack, "shot")
        cut = pr.get("lanes_cut_to") or []
        near = pr["nearest_presser_m"]
        if near[0] - near[1] >= 2.0:
            txt = (f"It started with a pressing trap: {pr['actor']} closed down {pr['victim']}, the nearest presser "
                   f"coming in from {near[0]} m to {near[1]} m")
        else:
            txt = (f"It started with a pressing trap: {pr['actor']} pressed {pr['victim']} for {pr['press_duration_s']} s "
                   f"with the nearest presser within {near[1]} m")
        if cut:
            names = cut[0] if len(cut) == 1 else ", ".join(cut[:-1]) + " and " + cut[-1]
            txt += f", while the press cut his passing lanes to {names}"
        S.append({"text": txt + ".", "cites": [f"chain[{i}]"]})
        S.append({"text": f"{won['actor']} won the ball at {won['t']}, {won['metres_from_their_goal_line']} m from "
                          f"their goal line, and the shot came {sh['seconds_after_turnover']} s later.",
                  "cites": [f"chain[{j}]", f"chain[{k}]"]})
        S.append({"text": f"Without the busiest presser, the model says {cf['open_lanes_counterfactual']} lanes would "
                          f"have been open instead of {cf['open_lanes_actual']}.", "cites": ["counterfactual"]})
    else:
        i, out = _chain(pack, "outran")
        k, sh = _chain(pack, "shot")
        if out is not None:
            S.append({"text": f"No tactical cause: {out['actor']} simply outran {out['marker']}, who kept chasing at "
                              f"{out['defender_speed_ms']} m/s against {out['attacker_speed_ms']} m/s.",
                      "cites": [f"chain[{i}]"]})
        else:
            S.append({"text": "The tracking model finds no decoy run, lost marker or pressing trap behind this chance.",
                      "cites": ["cause"]})
        S.append({"text": f"The shot came from {sh['distance_to_goal_m']} m with the nearest defender "
                          f"{sh['nearest_defender_m']} m away.", "cites": [f"chain[{k}]"]})
    return S


NAMES = {"decoy_run": ("decoy run", "decoy runs"), "lost_marker": ("lost marker", "lost markers"),
         "press_turnover": ("pressing trap", "pressing traps"), "none": ("with no tactical cause", "with no tactical cause")}


def _summary(pack: dict) -> list[dict]:
    S = [{"text": f"Studio note for {pack['window']}: {pack['n_chances']} chances.", "cites": ["window", "n_chances"]}]
    for team, cnt in pack.get("by_team", {}).items():
        parts = []
        for c in ("press_turnover", "decoy_run", "lost_marker", "none"):
            n = cnt.get(c, 0)
            if n:
                one, many = NAMES[c]
                parts.append(f"{n} {one if n == 1 else many}" if c != "none" else f"{n} {one}")
        if parts:
            S.append({"text": f"Team {team} had {pack['team_totals'][team]}: " + ", ".join(parts) + ".",
                      "cites": [f"by_team.{team}", f"team_totals.{team}"]})
    for k, rep in enumerate(pack.get("repeat_defenders", [])):
        S.append({"text": f"{rep['player']} was involved in {rep['count']} of them as the defender the model points to.",
                  "cites": [f"repeat_defenders[{k}]"]})
    return S


def join(sentences: list[dict]) -> str:
    return " ".join(s["text"] for s in sentences)
