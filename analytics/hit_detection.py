"""
Hit detection from ball trajectory reversals + player proximity.

Requires ball_x, ball_y columns in the dataframe (Phase B — pipeline re-run needed).
"""

import numpy as np
import pandas as pd


def detect_hits(df: pd.DataFrame, proximity_threshold: float = 2.5) -> list[dict]:
    """
    Detect ball hit events using two signals:
      1. Sign change in ball_Vy (trajectory reversal)
      2. Nearest player within `proximity_threshold` metres of the ball

    Args:
        df: analytics dataframe with ball_x, ball_y and player{1-4}_x/y columns
        proximity_threshold: maximum distance (m) from player feet to ball for a
            valid hit attribution

    Returns:
        List of hit event dicts with keys:
            frame, time, player_id, ball_x, ball_y, player_x, player_y, proximity
    """
    if "ball_x" not in df.columns or "ball_y" not in df.columns:
        return []

    ball_y = pd.to_numeric(df["ball_y"], errors="coerce")
    ball_x = pd.to_numeric(df["ball_x"], errors="coerce")

    ball_vy = ball_y.diff()

    sign_prev = np.sign(ball_vy.shift(1))
    sign_curr = np.sign(ball_vy)

    reversals = (
        (sign_prev != sign_curr)
        & (sign_prev != 0)
        & (sign_curr != 0)
        & ball_x.notna()
        & ball_y.notna()
    )

    hits = []
    for idx in df.index[reversals]:
        bx = ball_x.loc[idx]
        by = ball_y.loc[idx]

        min_dist = float("inf")
        nearest_player = None
        for pid in (1, 2, 3, 4):
            px = pd.to_numeric(df.loc[idx, f"player{pid}_x"], errors="coerce")
            py = pd.to_numeric(df.loc[idx, f"player{pid}_y"], errors="coerce")
            if pd.isna(px) or pd.isna(py):
                continue
            dist = np.sqrt((bx - px) ** 2 + (by - py) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_player = pid

        if nearest_player is not None and min_dist <= proximity_threshold:
            hits.append(
                {
                    "frame": int(df.loc[idx, "frame"]),
                    "time": float(df.loc[idx, "time"]) if "time" in df.columns else None,
                    "player_id": nearest_player,
                    "ball_x": float(bx),
                    "ball_y": float(by),
                    "player_x": float(df.loc[idx, f"player{nearest_player}_x"]),
                    "player_y": float(df.loc[idx, f"player{nearest_player}_y"]),
                    "proximity": round(float(min_dist), 3),
                }
            )

    return hits
