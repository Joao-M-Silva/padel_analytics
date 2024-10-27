import json
from pathlib import Path

import numpy as np
import pandas as pd
from pytest import fixture

from trackers.ball_tracker.court_3d_model import Court3DModel, compute_regression_line, find_intersection
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
    with open(Path(__file__).parent / "resources/ball_detections.json", "r") as f:
        ball_detections = json.load(f)

    yield ball_detections


@fixture
def court_model(court_keypoints) -> Court3DModel:
    model = Court3DModel(keypoints=court_keypoints)

    yield model


class TestCourtModel:

    def test_init(self, court_model):
        assert court_model.keypoints is not None
        assert court_model.depth_vanishing_point is not None
        assert court_model.projection_matrix.shape == (3, 4)

    def test_vanishing_point(self):
        """
        Tests that the vanishing point calculation works as expected
        """
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

    def test_projection_matrix(self, court_model):
        """
        Ensures that the projection matrix for the example video is nondegenerate
        """
        assert np.linalg.matrix_rank(court_model.projection_matrix) > 0

    def test_projection(self):
        """

        """
        keypoints = Keypoints([
            Keypoint(id=id, xy=tuple(xy))
            for id, xy in enumerate([[0, 0], [1, 0],
                                     [-1, -1], [-1, -1], [-1, -1],
                                     [0.1, .5], [.9, .5],
                                     [-1, -1], [-1, -1], [-1, -1],
                                     [.2, 1], [.8, 1],
                                     [-.1, 1], [.1, 1]])
        ])

        model = Court3DModel(keypoints=keypoints)

        assert model.projection_matrix is not None
        assert np.linalg.matrix_rank(model.projection_matrix) > 0

    def test_filter(self, court_model, ball_detections):
        court_model.track(ball_detections)
        x, y, z, vx, vy, vz, _ = zip(*court_model.kf.states)

        print(pd.DataFrame(dict(x=x, y=y, z=z)))
        print(pd.DataFrame(dict(vx=vx, vy=vy, vz=vz)))
        assert len(x) == len(ball_detections)

        fig = court_model.kf.plot()

        fig.show()
        # Save to file
        with open("../render.html", "w") as f:
            f.write(fig.to_html())
