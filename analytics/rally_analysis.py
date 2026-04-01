"""
Rally segmentation and AI analysis via Claude API.

Requires ball_x, ball_y columns in the dataframe (Phase B — pipeline re-run needed).
The AI analysis step requires ANTHROPIC_API_KEY to be set in the environment.
"""

import json
import os
from typing import Optional

import numpy as np
import pandas as pd


# ── Rally segmentation ────────────────────────────────────────────────────────


def segment_rallies(df: pd.DataFrame, min_gap: int = 30) -> list[dict]:
    """
    Segment rallies using ball visibility (non-null ball_x/ball_y).

    A rally starts when ball becomes visible after a gap of >= min_gap frames.
    A rally ends when ball is invisible for >= min_gap consecutive frames.

    Args:
        df: analytics dataframe with ball_x column
        min_gap: minimum number of consecutive no-ball frames to end/start a rally

    Returns:
        List of rally dicts with keys: start_frame, end_frame, start_time, end_time
    """
    if "ball_x" not in df.columns:
        return []

    df = df.reset_index(drop=True)
    ball_visible = df["ball_x"].notna().astype(int)

    rallies: list[dict] = []
    in_rally = False
    rally_start_i = None
    last_visible_i = None
    gap_count = 0

    for i in range(len(df)):
        if ball_visible.iloc[i]:
            if not in_rally:
                in_rally = True
                rally_start_i = i
            last_visible_i = i
            gap_count = 0
        else:
            if in_rally:
                gap_count += 1
                if gap_count >= min_gap:
                    # Close rally at last visible frame
                    end_i = last_visible_i
                    rallies.append(_make_rally(df, rally_start_i, end_i))
                    in_rally = False
                    gap_count = 0
                    rally_start_i = None

    # Close open rally at end of video
    if in_rally and rally_start_i is not None and last_visible_i is not None:
        rallies.append(_make_rally(df, rally_start_i, last_visible_i))

    return rallies


def _make_rally(df: pd.DataFrame, start_i: int, end_i: int) -> dict:
    return {
        "start_frame": int(df.loc[start_i, "frame"]),
        "end_frame": int(df.loc[end_i, "frame"]),
        "start_time": float(df.loc[start_i, "time"]) if "time" in df.columns else None,
        "end_time": float(df.loc[end_i, "time"]) if "time" in df.columns else None,
    }


# ── Rally enrichment ──────────────────────────────────────────────────────────


def enrich_rallies(
    rallies: list[dict],
    df: pd.DataFrame,
    hits: list[dict],
) -> list[dict]:
    """
    Attach per-rally statistics (duration, hit sequence, player positions).

    Args:
        rallies: list from segment_rallies
        df: analytics dataframe
        hits: list from detect_hits (may contain shot_type after classify_hits)

    Returns:
        Enriched rallies list (mutates and returns in-place).
    """
    for rally in rallies:
        sf = rally["start_frame"]
        ef = rally["end_frame"]

        rally_df = df[(df["frame"] >= sf) & (df["frame"] <= ef)]
        rally_hits = [h for h in hits if sf <= h["frame"] <= ef]

        st = rally.get("start_time") or 0.0
        et = rally.get("end_time") or 0.0
        rally["duration_s"] = round(et - st, 2)
        rally["total_hits"] = len(rally_hits)
        rally["hit_sequence"] = [h["player_id"] for h in rally_hits]
        rally["shot_types"] = [h.get("shot_type", "unknown") for h in rally_hits]

        # Average player positions during the rally
        avg_pos: dict[str, dict] = {}
        for pid in (1, 2, 3, 4):
            xc = f"player{pid}_x"
            yc = f"player{pid}_y"
            if xc in rally_df.columns:
                mx = pd.to_numeric(rally_df[xc], errors="coerce").mean()
                my = pd.to_numeric(rally_df[yc], errors="coerce").mean()
                avg_pos[str(pid)] = {
                    "x": round(float(mx), 2) if not np.isnan(mx) else None,
                    "y": round(float(my), 2) if not np.isnan(my) else None,
                }
        rally["avg_positions"] = avg_pos

    return rallies


# ── AI rally analysis ─────────────────────────────────────────────────────────


def analyse_rallies_with_claude(
    rallies: list[dict],
    zone_data: dict,
    sync_data: dict,
    kpi_data: Optional[dict] = None,
    player_names: Optional[dict] = None,
) -> dict:
    """
    Send match data to Claude (claude-sonnet-4-6) for coaching analysis.

    Returns a dict with:
        player_feedback:     {str(player_id): str}
        training_drills:     {str(player_id): list[str]}  (3 drills per player)
        overall_patterns:    list[str]

    Requires ANTHROPIC_API_KEY environment variable.
    """
    try:
        import anthropic
    except ImportError:
        return {
            "error": "anthropic package not installed — add it to requirements.txt",
            "player_feedback": {},
            "training_drills": {},
            "overall_patterns": [],
        }

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "error": "ANTHROPIC_API_KEY environment variable not set",
            "player_feedback": {},
            "training_drills": {},
            "overall_patterns": [],
        }

    if player_names is None:
        player_names = {1: "Player 1", 2: "Player 2", 3: "Player 3", 4: "Player 4"}

    kpi_section = ""
    if kpi_data:
        kpi_section = (
            f"\nPer-player coaching KPIs:\n{json.dumps(kpi_data, indent=2)}\n\n"
            "KPI reference:\n"
            "- net_approach_count: how many times player crossed into net zone (|y|<2m)\n"
            "- time_in_nomansland_pct: % of time in transition zone (3-6m) — high = poor positioning\n"
            "- change_of_direction_count: lateral direction reversals — low = passive/linear movement\n"
            "- lateral_bias: mean Vx (positive=right, negative=left) — near zero = balanced coverage\n"
            "- recovery_speed: avg speed (m/s) when retreating from net — low = slow recovery\n"
            "- peak_sprint_count: frames above 5.5 m/s — physical conditioning indicator\n"
        )

    client = anthropic.Anthropic(api_key=api_key)

    prompt = (
        "You are an expert padel coach analysing intermediate-level match tracking data.\n"
        "Based on the data below, provide for each player:\n"
        "1. Short, actionable coaching feedback (2-3 sentences).\n"
        "2. Exactly 3 specific training drill recommendations based on their weaknesses.\n"
        "   Each drill should be a concrete exercise name + 1-sentence description.\n"
        "Also provide 3-5 overall tactical patterns observed across the match.\n\n"
        f"Total rallies detected: {len(rallies)}\n"
        f"Court zone breakdown (% time in front/transition/back per player):\n"
        f"{json.dumps(zone_data, indent=2)}\n\n"
        f"Partner synchrony scores (Pearson r, -1 to +1):\n"
        f"{json.dumps(sync_data, indent=2)}\n"
        f"{kpi_section}"
        f"Rally summaries (first 20 rallies):\n"
        f"{json.dumps(rallies[:20], indent=2)}\n\n"
        "Respond with valid JSON only — no markdown, no prose outside the JSON:\n"
        "{\n"
        '  "player_feedback": {"1": "...", "2": "...", "3": "...", "4": "..."},\n'
        '  "training_drills": {\n'
        '    "1": ["Drill name: description", "Drill name: description", "Drill name: description"],\n'
        '    "2": [...], "3": [...], "4": [...]\n'
        "  },\n"
        '  "overall_patterns": ["...", "...", "..."]\n'
        "}"
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text.strip()

    # Strip markdown code fences if present
    if response_text.startswith("```"):
        parts = response_text.split("```")
        for part in parts:
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            if stripped.startswith("{"):
                response_text = stripped
                break

    return json.loads(response_text)
