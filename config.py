""" General configurations for main.py """

import os
from dotenv import load_dotenv

load_dotenv()

# Input video path
INPUT_VIDEO_PATH = os.getenv("INPUT_VIDEO_PATH", "./examples/videos/rally.mp4")

# Inference video path
OUTPUT_VIDEO_PATH = os.getenv("OUTPUT_VIDEO_PATH", "results.mp4")

# True to collect 2d projection data
COLLECT_DATA = os.getenv("COLLECT_DATA", "true").lower() == "true"
# Collected data path
COLLECT_DATA_PATH = os.getenv("COLLECT_DATA_PATH", "data.csv")

# Maximum number of frames to be analysed (None = all frames)
_max_frames = os.getenv("MAX_FRAMES")
MAX_FRAMES = int(_max_frames) if _max_frames else None

# Fixed court keypoints
FIXED_COURT_KEYPOINTS_LOAD_PATH = os.getenv(
    "FIXED_COURT_KEYPOINTS_LOAD_PATH", "./cache/fixed_keypoints_detection.json"
)
FIXED_COURT_KEYPOINTS_SAVE_PATH = os.getenv("FIXED_COURT_KEYPOINTS_SAVE_PATH", None)

# Players tracker
PLAYERS_TRACKER_MODEL = os.getenv(
    "PLAYERS_TRACKER_MODEL", "./weights/players_detection/yolov8m.pt"
)
PLAYERS_TRACKER_BATCH_SIZE = int(os.getenv("PLAYERS_TRACKER_BATCH_SIZE", "8"))
PLAYERS_TRACKER_ANNOTATOR = os.getenv(
    "PLAYERS_TRACKER_ANNOTATOR", "rectangle_bounding_box"
)
PLAYERS_TRACKER_LOAD_PATH = os.getenv(
    "PLAYERS_TRACKER_LOAD_PATH", "./cache/players_detections.json"
)
PLAYERS_TRACKER_SAVE_PATH = os.getenv(
    "PLAYERS_TRACKER_SAVE_PATH", "./cache/players_detections.json"
)

# Players keypoints tracker
PLAYERS_KEYPOINTS_TRACKER_MODEL = os.getenv(
    "PLAYERS_KEYPOINTS_TRACKER_MODEL", "./weights/players_keypoints_detection/best.pt"
)
PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE = int(
    os.getenv("PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE", "1280")
)
PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE = int(
    os.getenv("PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE", "8")
)
PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH = os.getenv(
    "PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH", "./cache/players_keypoints_detections.json"
)
PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH = os.getenv(
    "PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH", "./cache/players_keypoints_detections.json"
)

# Ball tracker
BALL_TRACKER_MODEL = os.getenv(
    "BALL_TRACKER_MODEL", "./weights/ball_detection/TrackNet_best.pt"
)
BALL_TRACKER_INPAINT_MODEL = os.getenv(
    "BALL_TRACKER_INPAINT_MODEL", "./weights/ball_detection/InpaintNet_best.pt"
)
BALL_TRACKER_BATCH_SIZE = int(os.getenv("BALL_TRACKER_BATCH_SIZE", "8"))
BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM = int(
    os.getenv("BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM", "400")
)
BALL_TRACKER_LOAD_PATH = os.getenv(
    "BALL_TRACKER_LOAD_PATH", "./cache/ball_detections.json"
)
BALL_TRACKER_SAVE_PATH = os.getenv(
    "BALL_TRACKER_SAVE_PATH", "./cache/ball_detections.json"
)

# Court keypoints tracker
KEYPOINTS_TRACKER_MODEL = os.getenv(
    "KEYPOINTS_TRACKER_MODEL", "./weights/court_keypoints_detection/best.pt"
)
KEYPOINTS_TRACKER_BATCH_SIZE = int(os.getenv("KEYPOINTS_TRACKER_BATCH_SIZE", "8"))
KEYPOINTS_TRACKER_MODEL_TYPE = os.getenv("KEYPOINTS_TRACKER_MODEL_TYPE", "yolo")
KEYPOINTS_TRACKER_CONF = float(os.getenv("KEYPOINTS_TRACKER_CONF", "0.3"))
KEYPOINTS_TRACKER_LOAD_PATH = os.getenv(
    "KEYPOINTS_TRACKER_LOAD_PATH", "./cache/court_keypoints_detections.json"
)
KEYPOINTS_TRACKER_SAVE_PATH = os.getenv(
    "KEYPOINTS_TRACKER_SAVE_PATH", "./cache/court_keypoints_detections.json"
)

# Results storage directory (for video history)
RESULTS_DIR = os.getenv("RESULTS_DIR", "./cache/results")

# Anthropic API key (for AI rally analysis)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
