import numpy as np
import plotly.graph_objs as go

from trackers.ball_tracker.ekf import ExtendedKalmanFilter


class KalmanFilter3DTracking(ExtendedKalmanFilter):
    """
    Implements the physics model for tracking a ball in 3D space using an Extended Kalman filter.
    """

    def __init__(self, projection_matrix, g=9.81, width=10., length=20., height=4., q=0.1, r=.01):

        # Process noise covariance
        # larger for V_y since we expect players hit the ball in this direction
        Q = np.diag(np.power([.01, .01, .01, .01, .1, .01, 0], 2))
        R = np.eye(2) * r  # Measurement noise covariance
        # Initial state (position and velocity)
        # Assume ball starts in the middle of the court
        x0 = np.array([width / 2, length / 2, height / 2, 0, 0, 0, 1])
        P = np.diag([width, length, height, width / 10, width / 10, width / 10, 0])  # Initial state uncertainty
        super().__init__(P, Q, R, x0)
        self.H = projection_matrix
        self.g = g
        self.width = width
        self.length = length
        self.height = height

    def observation_function(self, x):
        # Project the current 3D state to the 2D image space using the projection matrix
        observation_homogeneous = np.dot(self.H, x[[0, 1, 2, 6]])  # Append 1 for homogeneous coords

        # Normalize homogeneous coordinates (x', y', w) to (x/w, y/w)
        x_img = observation_homogeneous[0] / observation_homogeneous[2]
        y_img = observation_homogeneous[1] / observation_homogeneous[2]

        # Construct the predicted 2D measurement from the projected 3D point
        return np.array([x_img, y_img])

    def transition_function(self, x, dt=1. / 30):
        """
        Transition function for the state space model. This function predicts the next state given the current state.
        """
        # State transition matrices
        F = np.array(
            [
                [1, 0, 0, dt, 0, 0, 0],
                [0, 1, 0, 0, dt, 0, 0],
                [0, 0, 1, 0, 0, dt, -0.5 * self.g * dt ** 2],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, -self.g * dt],
                [0, 0, 0, 0, 0, 0, 1]
            ]
        )

        new_state = np.dot(F, x)

        # Bounce off floor
        if new_state[2] < 0:
            new_state[2] = -new_state[2]
            new_state[5] = -new_state[5]

        # Bounce off walls
        # width
        if new_state[0] < 0:
            new_state[0] = -new_state[0]
            new_state[3] = -new_state[3]
        elif new_state[0] > self.width:
            new_state[0] = 2 * self.width - new_state[0]
            new_state[3] = -new_state[3]

        # length
        if new_state[1] < 0:
            new_state[1] = -new_state[1]
            new_state[4] = -new_state[4]
        elif new_state[1] > self.length:
            new_state[1] = 2 * self.length - new_state[1]
            new_state[4] = -new_state[4]

        return new_state

    def plot(self, projection_matrix=None):
        if projection_matrix is not None:
            # Get video perspective
            K, R, C = decompose_projection_matrix(projection_matrix)

            target = np.array([self.width, self.length, 0]) / 2
            scene = dict(
                camera=dict(
                    eye=dict(x=C[0], y=C[1], z=C[2]),  # Camera position
                    center=dict(x=target[0], y=target[1], z=target[2]),  # Target point in scene
                ),
                aspectmode='manual',  # Set to manual to control each aspect ratio individually
                aspectratio=dict(x=1, y=1, z=1)
            )
        else:
            scene = {}

        x_positions, y_positions, z_positions = np.array(self.states).T[[0, 1, 2]]

        x_range = [0, self.width]
        y_range = [0, self.length]
        z_range = [z_positions.min(), z_positions.max()]

        # Create a base figure with the full trajectory as a static line in light gray
        fig = go.Figure()

        # Add the static line representing the entire path
        fig.add_trace(go.Scatter3d(
            x=x_positions,
            y=y_positions,
            z=z_positions,
            mode='lines+markers',
            line=dict(color='lightgray', width=2),
            marker=dict(size=5, color='lightgray')
        ))

        # Add a frame for each time step to animate the ball's position
        fig.frames = [
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=x_positions,
                        y=y_positions,
                        z=z_positions,
                        mode="lines+markers",
                        marker=dict(
                            color=["lightgray" if i != j else "red" for j in range(len(x_positions))],
                            size=5
                        )
                    )
                ],
                layout=dict(
                    scene=dict(
                        xaxis=dict(range=x_range, autorange=False),
                        yaxis=dict(range=y_range, autorange=False),
                        zaxis=dict(range=z_range, autorange=False)
                    )
                ),
                name=f'frame{i}'
            )
            for i in range(len(x_positions))
        ]

        # Set up animation controls with Play button and Slider
        fig.update_layout(
            title="3D Animation of Ball Position with Static Trajectory",
            scene=dict(
                xaxis=dict(title="X Position", range=[0, self.width], autorange=False),
                yaxis=dict(title="Y Position", range=y_range, autorange=False),
                zaxis=dict(title="Z Position", range=z_range, autorange=False)
            ) | scene,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None, {"frame": {"duration": 1. / 30, "redraw": True},
                                           "fromcurrent": True, "mode": "immediate"}])])
            ],
            sliders=[dict(
                steps=[dict(method="animate",
                            args=[[f"frame{i}"], {"mode": "immediate", "frame": {"duration": 1. / 30, "redraw": True},
                                                  "transition": {"duration": 0}}],
                            label=f"{i}") for i in range(len(x_positions))],
                active=0,
                x=0.1,
                y=0,
                len=0.9
            )]

        )


        return fig


def decompose_projection_matrix(P):
    # Separate P into intrinsic (K) and extrinsic (R and t) components
    M = P[:, :3]  # Left 3x3 part of P for intrinsic and rotation matrix
    K, R = np.linalg.qr(np.linalg.inv(M))  # RQ decomposition for K and R
    K = K / K[-1, -1]  # Normalize K to make the bottom right entry 1

    # Extract translation vector t
    t = np.linalg.inv(K) @ P[:, 3]

    # Calculate camera position (world coordinates)
    C = -R.T @ t
    return K, R, C
