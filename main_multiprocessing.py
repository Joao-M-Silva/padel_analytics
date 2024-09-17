from pathlib import Path
import concurrent.futures
import numpy as np
import timeit

from utils import (
    read_video, 
    save_video,
)
from trackers import PlayerTracker, Player, BallTracker, Ball


CACHE_SAVE_PATH = None
PLAYERS_DETECTIONS_LOAD_PATH = None
BALL_DETECTIONS_LOAD_PATH = None


INPUT_VIDEO_PATH = "./videos/trimmed_padel.mp4"
PLAYERS_TRACKER_MODEL = "yolov8m.pt"
BALL_TRACKER_MODEL = "./weights/ball_detection/TrackNet_best.pt"
BALL_TRACKER_INPAINT_MODEL = "./weights/ball_detection/InpaintNet_best.pt"
OUTPUT_VIDEO_PATH = "test_multiprocessing.mp4"


def players_tracking(player_tracker: PlayerTracker, frames) -> list[list[Player]]:
    
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

    return player_detections

def ball_tracking(ball_tracker: BallTracker, frames, h, w) -> list[Ball]:

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

    return ball_detections


def divide_frames(frames: list[list[np.ndarray]], size: int):
    for i in range(0, len(frames), size): 
        yield frames[i:i + size]


if __name__ == "__main__":

    t1 = timeit.default_timer()

    TRACKING_RESULTS = {
        "player_detections": [],
        "ball_detections": [],
    }

    MAX_FRAMES = 2000
    CHUNK_SIZE = 300

    frames, fps, w, h = read_video(INPUT_VIDEO_PATH, max_frames=MAX_FRAMES)
    if MAX_FRAMES is None or MAX_FRAMES < 800:
        chunks = [frames]
    else:
        chunks = divide_frames(frames, size=CHUNK_SIZE)

    player_tracker = PlayerTracker(PLAYERS_TRACKER_MODEL)
    ball_tracker = BallTracker(BALL_TRACKER_MODEL, BALL_TRACKER_INPAINT_MODEL)

    for i, chunk in enumerate(chunks):

        with concurrent.futures.ProcessPoolExecutor() as executor:
            
            print(f"PROCESSING CHUNK {i+1} with length {len(chunk)}")
            
            f1 = executor.submit(players_tracking, player_tracker, chunk)
            f2 = executor.submit(ball_tracking, ball_tracker, chunk, h, w)

            player_detections_chunk = f1.result()
            ball_detections_chunk = f2.result()

            TRACKING_RESULTS["player_detections"] += player_detections_chunk
            TRACKING_RESULTS["ball_detections"] += ball_detections_chunk

    assert len(TRACKING_RESULTS["player_detections"]) == len(frames)
    assert len(TRACKING_RESULTS["ball_detections"]) == len(frames)
    
    output_frames = player_tracker.draw_multiple_frames(
        frames,
        TRACKING_RESULTS["player_detections"],
    )

    output_frames = ball_tracker.draw_multiple_frames(
        output_frames,
        ball_detections=TRACKING_RESULTS["ball_detections"],
        traj_len=8,
    )

    save_video(
        output_frames,
        OUTPUT_VIDEO_PATH,
        fps=fps,
    )

    t2 = timeit.default_timer()

    print("Duration (min): ", (t2 - t1) / 60)