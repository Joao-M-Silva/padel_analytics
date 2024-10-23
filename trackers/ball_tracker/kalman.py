import json
from pathlib import Path

import numpy as np
from scipy import stats

from filterpy.kalman import KalmanFilter

from trackers.keypoints_tracker.keypoints_tracker import Keypoints


class KalmanTracker:
    def __init__(self, keypoints: Keypoints, court_width=10., court_length=20.):
        """
        Initialize the Kalman tracker with the court keypoints and dimensions.
        :param keypoints: Keypoints object with the court keypoints
        :param court_width: Width of the court in meters (standard is 10 meters)
        :param court_length: Length of the court in meters (standard is 20 meters)
        """
        self.keypoints = keypoints
        self.court_width = court_width
        self.court_length = court_length
        self.depth_vanishing_point = self._determine_vanishing_point()
        self.projection_matrix = self._determine_projection_matrix()

        self.xy = None

    def _determine_vanishing_point(self):
        keypoints = self.keypoints
        left_side_kp = [
            keypoints.keypoints_by_id[0].xy,
            keypoints.keypoints_by_id[2].xy,
            keypoints.keypoints_by_id[5].xy,
            keypoints.keypoints_by_id[7].xy,
            keypoints.keypoints_by_id[10].xy
        ]

        right_side_kp = [
            keypoints.keypoints_by_id[1].xy,
            keypoints.keypoints_by_id[4].xy,
            keypoints.keypoints_by_id[6].xy,
            keypoints.keypoints_by_id[9].xy,
            keypoints.keypoints_by_id[11].xy
        ]

        # Compute the regression line of the left side of the court
        slope1, intercept1 = compute_regression_line(points=left_side_kp)

        # Compute the regression line of the right side of the court
        slope2, intercept2 = compute_regression_line(points=right_side_kp)

        # Find the intersection of the two lines
        intersection = find_intersection(
            slope1, intercept1, slope2, intercept2
        )

        return intersection

    def _determine_projection_matrix(self):
        # World coordinates of the corner points
        world_points = np.array([
            [0, 0, 0],  # k1
            [0, self.court_width, 0],  # k2
            [self.court_length/2, 0, 0],  # k6
            [self.court_length/2, self.court_width, 0],  # k7
            [self.court_length, 0, 0],  # k11
            [self.court_length, self.court_width, 0]  # k12
        ])

        # Image coordinates of the corner points
        image_points = np.array([
            [self.keypoints.keypoints_by_id[0].xy[0], self.keypoints.keypoints_by_id[0].xy[1]],
            [self.keypoints.keypoints_by_id[1].xy[0], self.keypoints.keypoints_by_id[1].xy[1]],
            [self.keypoints.keypoints_by_id[5].xy[0], self.keypoints.keypoints_by_id[5].xy[1]],
            [self.keypoints.keypoints_by_id[6].xy[0], self.keypoints.keypoints_by_id[6].xy[1]],
            [self.keypoints.keypoints_by_id[10].xy[0], self.keypoints.keypoints_by_id[10].xy[1]],
            [self.keypoints.keypoints_by_id[11].xy[0], self.keypoints.keypoints_by_id[11].xy[1]]
        ])

        # Vanishing point constraints
        # (for the moment we only use depth vanishing point)
        vanishing_points = np.array([self.depth_vanishing_point])
        vanishing_directions = np.array([[0, 1, 0]])

        # Estimate the projection matrix
        P = estimate_projection_matrix_with_vanishing_points(
            world_points=world_points,
            image_points=image_points,
            vanishing_points=vanishing_points,
            vanishing_directions=vanishing_directions
        )

        return P

    def load_data(self, file: Path):
        with open(file, 'r') as f:
            ball_detections = json.load(f)

        self.xy = [bd['xy'] for bd in ball_detections]


# Function to compute the line equation (slope and intercept) from a set of points
def compute_regression_line(points):
    x, y = zip(*points)

    # Compute the slope and intercept using linear regression
    slope, intercept, _, _, _ = stats.linregress(x, y)

    return slope, intercept


# Function to find the intersection of two lines
def find_intersection(slope1, intercept1, slope2, intercept2):
    # If the slopes are the same, the lines are parallel and don't intersect
    if slope1 == slope2:
        return None  # Parallel lines

    # x-coordinate of the intersection point
    x_intersect = (intercept2 - intercept1) / (slope1 - slope2)

    # y-coordinate of the intersection point
    y_intersect = slope1 * x_intersect + intercept1

    return x_intersect, y_intersect


def estimate_projection_matrix_with_vanishing_points(
        world_points,
        image_points,
        vanishing_points,
        vanishing_directions
):
    """
    Estimate the 3x4 projection matrix using both world-image point correspondences and vanishing points.

    :param world_points: Nx3 array of world coordinates (X, Y, Z)
    :param image_points: Nx2 array of image coordinates (x, y)
    :param vanishing_points: Mx2 array of image coordinates for vanishing points (x_v, y_v)
    :param vanishing_directions: Mx3 array of vanishing directions in world coordinates (dX, dY, dZ)
    :return: 3x4 projection matrix
    """

    # Create the matrix A and vector b for the system of equations A * p = b
    A = []
    b = []

    # Add the point correspondences to the system
    for world_point, image_point in zip(world_points, image_points):
        X, Y, Z = world_point
        x, y = image_point

        # Two rows per point
        A.append([-X, -Y, -Z, -1, 0, 0, 0, 0, x * X, x * Y, x * Z, x])
        A.append([0, 0, 0, 0, -X, -Y, -Z, -1, y * X, y * Y, y * Z, y])
        b.append(0)
        b.append(0)

    # Add the vanishing point constraints to the system
    for vp, vd in zip(vanishing_points, vanishing_directions):
        x_v, y_v = vp
        d_X, d_Y, d_Z = vd

        # Vanishing point constraint equation for the x & y coordinates
        A.append([-d_X, -d_Y, -d_Z, 0, 0, 0, 0, 0, x_v * d_X, x_v * d_Y, x_v * d_Z, 0])
        A.append([0, 0, 0, 0, -d_X, -d_Y, -d_Z, 0, y_v * d_X, y_v * d_Y, y_v * d_Z, 0])
        b.append(0)
        b.append(0)

    A = np.array(A)
    b = np.array(b)

    assert np.linalg.matrix_rank(A, tol=1e-6) == 11, "Matrix A must have rank 12"

    # Solve the system A * p = b using least squares
    P, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

    print("Least Squares fit of the projection matrix:")
    print("Residuals:", residuals)
    print("Rank:", rank)
    print("Singular values:", sv)

    # Reshape the solution into the 3x4 projection matrix
    P = P.reshape(3, 4)

    return P
