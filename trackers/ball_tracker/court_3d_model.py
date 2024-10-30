from scipy import stats
import plotly.graph_objects as go
import numpy as np

from trackers.keypoints_tracker.keypoints_tracker import Keypoints


class Court3DModel:
    def __init__(self, keypoints: Keypoints, court_width=10., court_length=20., court_height=4.):
        """
        Creates a 3D model of the court and determines the mapping between the world coordinates and the
        camera coordinates.
        :param keypoints: Keypoints object with the court keypoints
        :param court_width: Width of the court in meters (standard is 10 meters)
        :param court_length: Length of the court in meters (standard is 20 meters)
        """
        self.width = court_width
        self.length = court_length
        self.height = court_height

        self.keypoint_correspondence = {
            0: [0, 0, 0],  # k1
            1: [self.width, 0, 0],  # k2
            5: [0, self.length / 2, 0],  # k6
            6: [self.width, self.length / 2, 0],  # k7
            10: [0, self.length, 0],  # k11
            11: [self.width, self.length, 0],  # k12
            12: [0, 0, self.height],  # k13
            13: [self.width, 0, self.height],  # k14
        }

        self.keypoints = keypoints
        self.depth_vanishing_point = self._determine_vanishing_point(indexes=[[0, 5, 10], [1, 6, 11]])
        self.height_vanishing_point = self._determine_vanishing_point(indexes=[[0, 12], [1, 13]])
        self.projection_matrix = self._determine_projection_matrix()

    def _determine_vanishing_point(self, indexes):
        line1_keypoints = [self.keypoints.keypoints_by_id[i].xy for i in indexes[0]]
        line2_keypoints = [self.keypoints.keypoints_by_id[i].xy for i in indexes[1]]

        # Compute the regression line of the left side of the court
        slope1, intercept1 = compute_regression_line(points=line1_keypoints)

        # Compute the regression line of the right side of the court
        slope2, intercept2 = compute_regression_line(points=line2_keypoints)

        # Find the intersection of the two lines
        intersection = find_intersection(
            slope1, intercept1, slope2, intercept2
        )

        return intersection

    def _determine_projection_matrix(self):
        # This determines the correspondence between keypoints and their expected world coordinates

        image_points_idx, world_points = zip(*self.keypoint_correspondence.items())

        image_points = np.array([
            [self.keypoints.keypoints_by_id[idx].xy[0], self.keypoints.keypoints_by_id[idx].xy[1]]
            for idx in image_points_idx
        ])

        # Vanishing point constraints
        vanishing_points = np.array([self.depth_vanishing_point, self.height_vanishing_point])
        vanishing_directions = np.array([[0, 1, 0], [0, 0, -1]])

        # Estimate the projection matrix
        P = estimate_projection_matrix_with_vanishing_points(
            world_points=world_points,
            image_points=image_points,
            vanishing_points=vanishing_points,
            vanishing_directions=vanishing_directions
        )

        return P

    def world2image(self, x) -> np.ndarray:
        """
        Compute the image of a world vector x
        :param x: 3D world vector
        :return: 2D image vector
        """
        # Project the current 3D state to the 2D image space using the projection matrix
        observation_homogeneous = np.dot(self.projection_matrix, np.append(x, 1))  # Append 1 for homogeneous coords

        # Normalize homogeneous coordinates (x', y', w) to (x/w, y/w)
        x_img = observation_homogeneous[0] / observation_homogeneous[2]
        y_img = observation_homogeneous[1] / observation_homogeneous[2]

        # Construct the predicted 2D measurement from the projected 3D point
        return np.array([x_img, y_img])

    def plot_2D_court(self):
        """
        Plot the court, with the ground lines and the height lines in 2D using matplotlib
        using the perspective projection matrix
        """
        fig = go.Figure()

        # Make y axis upside-down
        fig.update_yaxes(autorange="reversed")

        # Use the keypoint correspondences to obtain their expected 2D locations
        image_coords = {}
        for keypoint_id, world_coords in self.keypoint_correspondence.items():
            image_coords[keypoint_id] = self.world2image(world_coords)
            fig.add_trace(go.Scatter(x=[image_coords[keypoint_id][0]], y=[image_coords[keypoint_id][1]], mode='markers', name=f'k{keypoint_id}'))

        court_lines = [[0, 1], [1, 6], [6, 11], [11, 10], [10, 5], [5, 0], [0, 12], [1, 13], [12, 13]]
        for line in court_lines:
            fig.add_trace(go.Line(x=[image_coords[line[0]][0], image_coords[line[1]][0]], y=[image_coords[line[0]][1], image_coords[line[1]][1]], mode='lines'))

        return fig

    def plot_3D_court(self):
        """
        Plot the court, with the ground lines and the height lines in 3D using plotly
        """
        import plotly.graph_objects as go

        fig = go.Figure()

        # Add the court lines
        fig.add_trace(go.Scatter3d(
            x=[0, self.width, self.width, 0, 0],
            y=[0, 0, self.length, self.length, 0],
            z=[0, 0, 0, 0, 0],
            mode='lines',
            name='Ground lines'
        ))

        fig.add_trace(go.Scatter3d(
            x=[0, self.width, self.width, 0, 0],
            y=[0, 0, 0, 0, 0],
            z=[0, 0, self.height, self.height, 0],
            mode='lines',
            name='Height lines'
        ))

        # Add the vanishing points
        fig.add_trace(go.Scatter3d(
            x=[self.depth_vanishing_point[0]],
            y=[self.depth_vanishing_point[1]],
            z=[self.depth_vanishing_point[2]],
            mode='markers',
            name='Depth vanishing point'
        ))

        fig.add_trace(go.Scatter3d(
            x=[self.height_vanishing_point[0]],
            y=[self.height_vanishing_point[1]],
            z=[self.height_vanishing_point[2]],
            mode='markers',
            name='Height vanishing point'
        ))

        fig.show()



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
    #
    # # Normalizing 3D world points
    # mean_world = np.mean(world_points, axis=0)
    # std_world = np.std(world_points, axis=0)
    # world_points_normalized = (world_points - mean_world) / std_world
    #
    # # Normalizing 2D image points
    # mean_image = np.mean(image_points, axis=0)
    # std_image = np.std(image_points, axis=0)
    # image_points_normalized = (image_points - mean_image) / std_image

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

    assert np.linalg.matrix_rank(A, tol=1e-6) == 12, \
        "Matrix A must have rank 12. Choose at least 6 independent points"

    # Solve the system A * p = b using least squares
    # P, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    # P = P.reshape(3, 4)

    # Perform SVD on A
    U, S, Vt = np.linalg.svd(A)

    # The solution is the last column of V (Vt is the transpose)
    P = Vt[-1, :].reshape(3, 4)  # Reshape it into a 3x4 matrix

    condition_number = np.linalg.cond(A)
    print("Least Squares fit of the projection matrix:")
    print("Condition Number of A:", condition_number)
    print("Projection matrix P:", P)

    # Reshape the solution into the 3x4 projection matrix

    return P
