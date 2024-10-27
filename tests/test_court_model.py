import numpy as np
import numpy.testing as npt

from trackers.ball_tracker.court_3d_model import Court3DModel, compute_regression_line, find_intersection
from trackers.keypoints_tracker.keypoints_tracker import Keypoints, Keypoint


class TestCourtModel:

    def test_init(self, court_keypoints):

        court_model = Court3DModel(keypoints=court_keypoints)

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

    def test_projection_matrix(self, court_keypoints):
        """
        Ensures that the projection matrix for the example video is nondegenerate
        """
        court_model = Court3DModel(keypoints=court_keypoints)
        assert np.linalg.matrix_rank(court_model.projection_matrix) > 0

    def test_projection_nonsingular(self):
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

    def test_keypoints(self, court_keypoints):
        court_model = Court3DModel(keypoints=court_keypoints)

        # check that keypoints are mapped as expected
        for i, xyz in court_model.keypoint_correspondence.items():
            npt.assert_almost_equal(
                court_model.world2image(xyz)/1000, np.array(court_model.keypoints.keypoints_by_id[i].xy)/1000,
                decimal=2
            )

        court_model.plot_2D_court().show()
