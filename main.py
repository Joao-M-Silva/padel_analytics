import timeit
import json
import cv2
import numpy as np
import supervision as sv

from trackers import (
    PlayerTracker, 
    BallTracker, 
    KeypointsTracker, 
    Keypoint,
    Keypoints,
    PlayerKeypointsTracker,
    TrackingRunner,
)
from analytics import MiniCourt, DataAnalytics
from config import *


SELECTED_KEYPOINTS = []

"""
PADEL COURT KEYPOINTS 

-> To be selected using the image pop-up

        k11--------------------k12
        |                       |
        k8-----------k9--------k10
        |            |          |
        |            |          |
        |            |          |
        k6----------------------k7
        |            |          |
        |            |          |
        |            |          |
        k3-----------k4---------k5
        |                       |
        k1----------------------k2
"""

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
        cv2.imshow('frame', img) 


if __name__ == "__main__":
    
    t1 = timeit.default_timer()

    video_info = sv.VideoInfo.from_video_path(video_path=INPUT_VIDEO_PATH)
    fps, w, h, total_frames = (
        video_info.fps, 
        video_info.width,
        video_info.height,
        video_info.total_frames,
    )

    first_frame_generator = sv.get_video_frames_generator(
        INPUT_VIDEO_PATH,
        start=0,
        stride=1,
        end=1,
    )

    img = next(first_frame_generator)

    if FIXED_COURT_KEYPOINTS_LOAD_PATH is not None:
        with open(FIXED_COURT_KEYPOINTS_LOAD_PATH, "r") as f:
            SELECTED_KEYPOINTS = json.load(f)
    else:
        cv2.imshow('frame', img)
        cv2.setMouseCallback('frame', click_event) 
        # wait for a key to be pressed to exit 
        cv2.waitKey(0) 
        # close the window 
        cv2.destroyAllWindows() 

    if FIXED_COURT_KEYPOINTS_SAVE_PATH is not None:
        with open(FIXED_COURT_KEYPOINTS_SAVE_PATH, "w") as f:
            json.dump(SELECTED_KEYPOINTS, f)

    fixed_keypoints_detection = Keypoints(
        [
            Keypoint(
                id=i,
                xy=tuple(float(x) for x in v)
            )
            for i, v in enumerate(SELECTED_KEYPOINTS)
        ]
    )

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

    # FILTER FRAMES OF INTEREST (TODO)


    # Instantiate trackers
    players_tracker = PlayerTracker(
        PLAYERS_TRACKER_MODEL,
        polygon_zone,
        batch_size=PLAYERS_TRACKER_BATCH_SIZE,
        annotator=PLAYERS_TRACKER_ANNOTATOR,
        show_confidence=True,
        load_path=PLAYERS_TRACKER_LOAD_PATH,
        save_path=PLAYERS_TRACKER_SAVE_PATH,
    )

    player_keypoints_tracker = PlayerKeypointsTracker(
        PLAYERS_KEYPOINTS_TRACKER_MODEL,
        train_image_size=PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE,
        batch_size=PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE,
        load_path=PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH,
        save_path=PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH,
    )
  
    ball_tracker = BallTracker(
        BALL_TRACKER_MODEL,
        BALL_TRACKER_INPAINT_MODEL,
        batch_size=BALL_TRACKER_BATCH_SIZE,
        median_max_sample_num=BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM,
        median=None,
        load_path=BALL_TRACKER_LOAD_PATH,
        save_path=BALL_TRACKER_SAVE_PATH,
    )

    keypoints_tracker = KeypointsTracker(
        model_path=KEYPOINTS_TRACKER_MODEL,
        batch_size=KEYPOINTS_TRACKER_BATCH_SIZE,
        model_type=KEYPOINTS_TRACKER_MODEL_TYPE,
        fixed_keypoints_detection=fixed_keypoints_detection,
        load_path=KEYPOINTS_TRACKER_LOAD_PATH,
        save_path=KEYPOINTS_TRACKER_SAVE_PATH,
    )

    runner = TrackingRunner(
        trackers=[
            players_tracker, 
            player_keypoints_tracker, 
            ball_tracker,
            keypoints_tracker,    
        ],
        video_path=INPUT_VIDEO_PATH,
        inference_path=OUTPUT_VIDEO_PATH,
        start=0,
        end=MAX_FRAMES,
    )

    # OPTIMIZE MINI COURT INTEGRATION

    mini_court = MiniCourt(img)

    runner.run(mini_court=mini_court, fixed_keypoints_detection=fixed_keypoints_detection)

    t2 = timeit.default_timer()

    print("Duration (min): ", (t2 - t1) / 60)
