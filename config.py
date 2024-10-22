""" General configurations for main.py """

# Input video path
INPUT_VIDEO_PATH = "./examples/videos/rally.mp4"

# Inference video path
OUTPUT_VIDEO_PATH = "results.mp4"

# True to collect 2d projection data
COLLECT_DATA = True
# Collected data path
COLLECT_DATA_PATH = "data.csv"

# Maximum number of frames to be analysed
MAX_FRAMES = 30

# Fixed court keypoints´
FIXED_COURT_KEYPOINTS_LOAD_PATH = "./cache/fixed_keypoints_detection.json"
FIXED_COURT_KEYPOINTS_SAVE_PATH = "./cache/fixed_keypoints_detection.json"

# Players tracker
PLAYERS_TRACKER_MODEL = "./weights/players_detection/yolov8m.pt"
PLAYERS_TRACKER_BATCH_SIZE = 8
PLAYERS_TRACKER_ANNOTATOR = "rectangle_bounding_box"
PLAYERS_TRACKER_LOAD_PATH = None # "./cache/players_detections.json"
PLAYERS_TRACKER_SAVE_PATH = "./cache/players_detections.json"

# Players keypoints tracker
PLAYERS_KEYPOINTS_TRACKER_MODEL = "./weights/players_keypoints_detection/best.pt"
PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE = 1280
PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE = 8
PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH = None #"./cache/players_keypoints_detections.json"
PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH = "./cache/players_keypoints_detections.json"

# Ball tracker
BALL_TRACKER_MODEL = "./weights/ball_detection/TrackNet_best.pt"
BALL_TRACKER_INPAINT_MODEL = "./weights/ball_detection/InpaintNet_best.pt"
BALL_TRACKER_BATCH_SIZE = 8
BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM = 400
BALL_TRACKER_LOAD_PATH = None # "./cache/ball_detections.json"
BALL_TRACKER_SAVE_PATH = "./cache/ball_detections.json"

# Court keypoints tracker
KEYPOINTS_TRACKER_MODEL = "./weights/court_keypoints_detection/best.pt"
KEYPOINTS_TRACKER_BATCH_SIZE = 8
KEYPOINTS_TRACKER_MODEL_TYPE = "yolo"
KEYPOINTS_TRACKER_LOAD_PATH = None
KEYPOINTS_TRACKER_SAVE_PATH = "./cache/keypoints_detections.json"

