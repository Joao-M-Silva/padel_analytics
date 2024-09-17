from pathlib import Path
import timeit
import cv2

from utils import (
    read_video, 
    save_video,
)
from trackers import PlayerTracker, BallTracker, KeypointsTracker
from mini_court import MiniCourt

CACHE_SAVE_PATH = Path("cache/")
PLAYERS_DETECTIONS_LOAD_PATH = "cache/player_detections.json"
BALL_DETECTIONS_LOAD_PATH = "cache/ball_detections.json"
KEYPOINTS_DETECTIONS_LOAD_PATH = "cache/keypoints_detections.json"


INPUT_VIDEO_PATH = "./videos/trimmed_padel.mp4"
# INPUT_VIDEO_PATH = "./videos/FINAL A1PADEL MARBELLA MASTER  Tolito Aguirre  Alfonso vs Allemandi  Pereyra HIGHLIGHTS.mp4"
PLAYERS_TRACKER_MODEL = "yolov8m.pt"
BALL_TRACKER_MODEL = "./weights/ball_detection/TrackNet_best.pt"
BALL_TRACKER_INPAINT_MODEL = "./weights/ball_detection/InpaintNet_best.pt"
KEYPOINTS_TRACKER_MODEL = "./runs/keypoints/train2/weights/best.pt"
OUTPUT_VIDEO_PATH = "test_all_detections.mp4"

if __name__ == "__main__":

    t1 = timeit.default_timer()

    frames, fps, w, h = read_video(INPUT_VIDEO_PATH, max_frames=1200)

    # FILTER FRAMES OF INTEREST

    # Track players
    player_tracker = PlayerTracker(PLAYERS_TRACKER_MODEL)
    if CACHE_SAVE_PATH is not None:
        player_detections_save_path = CACHE_SAVE_PATH / "player_detections.json"
    else:
        player_detections_save_path = None

    # Probably need to implement the tracking myself
    player_detections = player_tracker.detect_frames(
        frames, 
        save_path=player_detections_save_path,
        load_path=PLAYERS_DETECTIONS_LOAD_PATH,
    )

    # TRACK THE BALL (use TrackNetV3)
    ball_tracker = BallTracker(BALL_TRACKER_MODEL, BALL_TRACKER_INPAINT_MODEL)
    if CACHE_SAVE_PATH is not None:
        ball_detections_save_path = CACHE_SAVE_PATH / "ball_detections.json"
    else:
        ball_detections_save_path = None

    ball_detections = ball_tracker.detect_frames(
        frames,
        width=w,
        height=h,
        batch_size=8,
        save_path=ball_detections_save_path,
        load_path=BALL_DETECTIONS_LOAD_PATH,
    )

    # DETECT KEYPOINTS
    keypoints_tracker = KeypointsTracker(
        model_path=KEYPOINTS_TRACKER_MODEL,
        model_type="yolo",
    )
    if CACHE_SAVE_PATH is not None:
        keypoints_detections_save_path = CACHE_SAVE_PATH / "keypoints_detections.json"
    else:
        keypoints_detections_save_path = None

    # MIGH ONLY DETECT IN WIDE STEPS (OR EVEN ON A SINGLE FRAME DUE TO FIXED CAMERA)
    keypoints_detections = keypoints_tracker.detect_frames(
        frames,
        save_path=keypoints_detections_save_path,
        load_path=KEYPOINTS_DETECTIONS_LOAD_PATH,
        frequency=1,
    )

    # Draw players detections
    output_frames = player_tracker.draw_multiple_frames(
        frames,
        player_detections,
    )

    # Draw ball detections
    output_frames = ball_tracker.draw_multiple_frames(
        output_frames,
        ball_detections=ball_detections,
        traj_len=8,
    )

    # Draw keypoints detections
    output_frames = keypoints_tracker.draw_multiple_frames(
        output_frames, 
        keypoints_detections,
    )

    # 2D PROJECTION
    mini_court = MiniCourt(frames[0])
    output_frames = mini_court.draw_minicourt_with_projections(
        output_frames,
        keypoints_detections,
        player_detections,
        ball_detections,
    )

    # Draw frame number on top left corner
    for i, frame in enumerate(output_frames):
        cv2.putText(
            frame,
            f"Frame: {i}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    save_video(
        output_frames,
        OUTPUT_VIDEO_PATH,
        fps=fps,
    )

    t2 = timeit.default_timer()

    print("Duration (min): ", (t2 - t1) / 60)