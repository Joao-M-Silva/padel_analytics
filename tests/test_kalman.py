import json
from pathlib import Path

import numpy as np
from pytest import fixture

from trackers.ball_tracker.kalman import KalmanTracker, compute_regression_line, find_intersection
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
def kalman_tracker(court_keypoints) -> KalmanTracker:
    kt = KalmanTracker(keypoints=court_keypoints)

    yield kt


class TestKalman:

    def test_init(self, kalman_tracker):
        assert kalman_tracker.keypoints is not None
        assert kalman_tracker.depth_vanishing_point is not None
        assert kalman_tracker.projection_matrix.shape == (3, 4)

    def test_load_detections(self, kalman_tracker):
        kalman_tracker.load_data(Path(__file__).parent / "resources/ball_detections.json")

        assert kalman_tracker.xy is not None

    def test_vanishing_point(self):
        # Example sets of points
        points_set1 = np.array([[1, 2], [2, 3], [3, 5], [4, 6], [5, 8]])
        points_set2 = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]])

        # Compute the regression lines
        slope1, intercept1 = compute_regression_line(points_set1)
        slope2, intercept2 = compute_regression_line(points_set2)

        # Find the intersection
        intersection = find_intersection(slope1, intercept1, slope2, intercept2)

        assert intersection[0] > 2
        assert intersection[0] < 3
        assert intersection[1] > 3
        assert intersection[1] < 4
