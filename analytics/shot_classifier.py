"""
Rule-based padel shot type classifier.

Shot rules (verified against padel coaching sources):

| Shot       | Player |y|  | Post-hit direction | Notes                                    |
|------------|----------|--------------------|------------------------------------------|
| volley     | < 2 m    | any                | Ball not bounced before contact at net   |
| smash      | < 5 m    | toward net, fast   | Finishing overhead at net zone           |
| vibora     | < 5 m    | toward net, mid    | Overhead with sidespin (faster bandeja)  |
| bandeja    | < 5 m    | toward net, slow   | Controlled overhead with backspin        |
| bajada     | >= 7 m   | toward net         | After ball bounces off back wall         |
| chiquita   | >= 5 m   | toward net, soft   | Low soft touch to opponents' feet        |
| globo      | any      | away from net      | Lob to opponent's back court             |
| unknown    | —        | —                  | Insufficient data                        |
"""

import numpy as np
import pandas as pd


def classify_shot(hit_event: dict, df: pd.DataFrame) -> str:
    """
    Classify a single padel hit event using court position and ball trajectory.

    Args:
        hit_event: dict returned by detect_hits (must contain player_y, frame, ball_x/y)
        df: analytics dataframe

    Returns:
        Shot type string: one of volley, smash, vibora, bandeja, bajada, chiquita, globo, unknown
    """
    py = hit_event["player_y"]
    frame = hit_event["frame"]
    abs_py = abs(float(py))

    if "ball_y" not in df.columns:
        return "unknown"

    # Look ahead 3 frames to estimate post-hit ball trajectory
    post_frames = df[df["frame"] > frame].head(3)
    if len(post_frames) < 2:
        return "unknown"

    post_by = pd.to_numeric(post_frames["ball_y"], errors="coerce").dropna()
    post_bx = pd.to_numeric(post_frames["ball_x"], errors="coerce").dropna()

    if len(post_by) < 2:
        return "unknown"

    ball_dy = float(post_by.diff().mean())
    ball_dx = float(post_bx.diff().mean()) if len(post_bx) >= 2 else 0.0

    if np.isnan(ball_dy):
        ball_dy = 0.0
    if np.isnan(ball_dx):
        ball_dx = 0.0

    ball_speed = np.sqrt(ball_dy ** 2 + ball_dx ** 2)

    # "toward net" means |y| of ball is decreasing, i.e. ball moves toward y=0.
    # Player on y>0 side: toward net → ball_dy < 0
    # Player on y<0 side: toward net → ball_dy > 0
    if float(py) > 0:
        toward_net = ball_dy < -0.05
        away_from_net = ball_dy > 0.05
    else:
        toward_net = ball_dy > 0.05
        away_from_net = ball_dy < -0.05

    # ── Classification rules ──────────────────────────────────────────────────

    # Volley: player right at the net (|y| < 2 m)
    if abs_py < 2:
        return "volley"

    # Net / mid-zone overhead shots (|y| < 5 m)
    if abs_py < 5:
        if away_from_net:
            # Defensive lob from mid-zone
            return "globo"
        if toward_net:
            if ball_speed > 1.5:
                return "smash"
            elif ball_speed > 0.8:
                return "vibora"
            else:
                return "bandeja"

    # Back-court shots (|y| >= 5 m)
    if away_from_net:
        return "globo"

    if toward_net:
        if abs_py >= 7:
            # Deep back court, attacking toward net after wall bounce
            return "bajada"
        if ball_speed < 0.6:
            return "chiquita"
        return "chiquita"  # default soft back-court shot

    return "unknown"


def classify_hits(hits: list[dict], df: pd.DataFrame) -> list[dict]:
    """
    Classify all hit events and attach a `shot_type` field to each.
    """
    for hit in hits:
        hit["shot_type"] = classify_shot(hit, df)
    return hits
