import json
from pathlib import Path

from _pytest.fixtures import fixture

from trackers.ball_tracker.court_3d_model import Court3DModel
from trackers.keypoints_tracker.keypoints_tracker import Keypoints, Keypoint


@fixture
def court_keypoints() -> Keypoints:
    with open(Path(__file__).parent / "resources/fixed_keypoints_detection.json", "r") as f:
        kp_data = json.load(f)

    kp = Keypoints(
        [
            Keypoint(
                id=i,
                xy=(float(v[0]), float(v[1]))
            )
            for i, v in enumerate(kp_data)
        ]
    )

    yield kp


@fixture
def ball_detections():
    with open(Path(__file__).parent / "resources/full_ball_detections.json", "r") as f:
        ball_detections = json.load(f)

    yield ball_detections


@fixture
def ballistic_detections():
    with open(Path(__file__).parent / "resources/ballistic_detections1.json", "r") as f:
        ballistic_detections = json.load(f)

    yield ballistic_detections


@fixture
def court_model(court_keypoints) -> Court3DModel:
    model = Court3DModel(keypoints=court_keypoints)

    yield model
