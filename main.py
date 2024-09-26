from pathlib import Path
import timeit
import json
import cv2
import numpy as np
import supervision as sv

from utils import (
    read_video, 
    save_video,
)
from trackers import (
    PlayerTracker, 
    BallTracker, 
    KeypointsTracker, 
    Keypoint,
    PlayersPoseTracker,
)
from analytics import MiniCourt, DataAnalytics


SELECTED_KEYPOINTS = []


def click_event(event, x, y, flags, params): 
  
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        SELECTED_KEYPOINTS.append((x, y))
  
        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (255, 0, 0), 2) 
        cv2.imshow('image', img) 


CACHE_SAVE_PATH = Path("cache/")
PLAYERS_DETECTIONS_LOAD_PATH = None # "cache/player_detections.json"
BALL_DETECTIONS_LOAD_PATH = None # "cache/ball_detections.json"
KEYPOINTS_DETECTIONS_LOAD_PATH = None # "cache/keypoints_detections.json"
PLAYERS_KEYPOINTS_DETECTIONS_LOAD_PATH = None

INPUT_VIDEO_PATH = "./videos/trimmed_padel.mp4"
# INPUT_VIDEO_PATH = "./videos/trimmed_esposende_padel.mp4"
PLAYERS_TRACKER_MODEL = "yolov8m.pt"
BALL_TRACKER_MODEL = "./weights/ball_detection/TrackNet_best.pt"
BALL_TRACKER_INPAINT_MODEL = "./weights/ball_detection/InpaintNet_best.pt"
KEYPOINTS_TRACKER_MODEL = "./runs/keypoints/train2/weights/best.pt"
PLAYERS_KEYPOINTS_TRACKER_MODEL = "./runs/pose/train3/weights/best.pt"
OUTPUT_VIDEO_PATH = "test_all_detections.mp4"

if __name__ == "__main__":
    
    t1 = timeit.default_timer()

    video_info = sv.VideoInfo.from_video_path(video_path=INPUT_VIDEO_PATH)
    fps, w, h, total_frames = (
        video_info.fps, 
        video_info.width,
        video_info.height,
        video_info.total_frames,
    )
    frame_generator = sv.get_video_frames_generator(
        INPUT_VIDEO_PATH,
        start=0,
        stride=1,
        end=300,
    )

    """SUBOPTIMAL"""
    frames = list(frame_generator)

    ##### frames, fps, w, h = read_video(INPUT_VIDEO_PATH, max_frames=300)
    
    first_frame_generator = sv.get_video_frames_generator(
        INPUT_VIDEO_PATH,
        start=0,
        stride=1,
        end=1,
    )
    img = next(first_frame_generator)
    cv2.imshow('image', img)
 
    cv2.setMouseCallback('image', click_event) 

    # wait for a key to be pressed to exit 
    cv2.waitKey(0) 
  
    # close the window 
    cv2.destroyAllWindows() 

    keypoints_array = np.array(SELECTED_KEYPOINTS)
    # Polygon to filter person detections inside padel court
    polygon_zone = sv.PolygonZone(
        np.concatenate(
            (
                np.expand_dims(keypoints_array[0], axis=0), 
                np.expand_dims(keypoints_array[1], axis=0), 
                np.expand_dims(keypoints_array[-1], axis=0), 
                np.expand_dims(keypoints_array[-2], axis=0),
            ),
            axis=0
        ),
        frame_resolution_wh=video_info.resolution_wh,
    )

    fixed_keypoints_detection = [
        Keypoint(
            id=i,
            xy=tuple(float(x) for x in v)
        )
        for i, v in enumerate(SELECTED_KEYPOINTS)
    ]

    if CACHE_SAVE_PATH is not None:
        fixed_keypoints_detection_path = CACHE_SAVE_PATH / "fixed_keypoints_detection.json"
        with open(fixed_keypoints_detection_path, "w") as f:
            json.dump(SELECTED_KEYPOINTS, f)

    # FILTER FRAMES OF INTEREST

    # Track players
    player_tracker = PlayerTracker(
        PLAYERS_TRACKER_MODEL, 
        video_info,
        polygon_zone,
    )
    if CACHE_SAVE_PATH is not None:
        player_detections_save_path = CACHE_SAVE_PATH / "player_detections.json"
    else:
        player_detections_save_path = None

    """SUBOPTIMAL"""
    frame_generator = sv.get_video_frames_generator(
        INPUT_VIDEO_PATH,
        start=0,
        stride=1,
        end=300,
    )

    player_detections = player_tracker.detect_frames(
        frame_generator, 
        save_path=player_detections_save_path,
        load_path=PLAYERS_DETECTIONS_LOAD_PATH,
    )

    # Track players keypoints
    players_pose_tracker = PlayersPoseTracker(
        PLAYERS_KEYPOINTS_TRACKER_MODEL,
        train_image_size=1280,
    )
    if CACHE_SAVE_PATH is not None:
        players_keypoints_detections_save_path = CACHE_SAVE_PATH / "players_keypoints_detections.json"
    else:
        players_keypoints_detections_save_path = None

    """SUBOPTIMAL"""
    frame_generator = sv.get_video_frames_generator(
        INPUT_VIDEO_PATH,
        start=0,
        stride=1,
        end=300,
    )

    players_keypoints_detections = players_pose_tracker.detect_frames(
        frame_generator,
        save_path=players_keypoints_detections_save_path,
        load_path=PLAYERS_KEYPOINTS_DETECTIONS_LOAD_PATH,
    )

    # TRACK THE BALL (use TrackNetV3)
    ball_tracker = BallTracker(BALL_TRACKER_MODEL, BALL_TRACKER_INPAINT_MODEL)
    if CACHE_SAVE_PATH is not None:
        ball_detections_save_path = CACHE_SAVE_PATH / "ball_detections.json"
    else:
        ball_detections_save_path = None

    ball_detections = ball_tracker.detect_frames(
        frames,  # OPTIMIZE THIS
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
        fixed_keypoints_detection=fixed_keypoints_detection,
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
        use_extra_model=False,
    )

    frame_generator = sv.get_video_frames_generator(
        INPUT_VIDEO_PATH,
        start=0,
        stride=1,
        end=300,
    )
    
    # Draw players detections
    output_frames = player_tracker.draw_multiple_frames(
        frame_generator,
        player_detections,
    )

    output_frames = players_pose_tracker.draw_multiple_frames(
        output_frames,
        players_keypoints_detections,
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
    output_frames, data_analytics = mini_court.draw_minicourt_with_projections(
        output_frames,
        keypoints_detections,
        player_detections,
        ball_detections,
        data_analytics=DataAnalytics(),
    )

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
