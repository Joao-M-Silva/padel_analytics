import numpy as np
import plotly.graph_objs as go
from scipy.optimize import least_squares

from trackers.ball_tracker.court_3d_model import Court3DModel
from trackers.ball_tracker.ekf import ExtendedKalmanFilter


class KalmanFilter3DTracking(ExtendedKalmanFilter):
    """
    Implements the physics model for tracking a ball in 3D space using an Extended Kalman filter.
    """

    def __init__(self, court_model: Court3DModel, x0=None, g=9.81, q=0.1, r=100):
        self.court_model = court_model
        self.width = court_model.width
        self.length = court_model.length
        self.height = court_model.height

        # Process noise covariance
        # larger for V_y since we expect players hit the ball in this direction
        Q = np.diag(np.power([.01, .01, .01, .01, .1, .01, 0], 2))
        R = np.eye(2) * r  # Measurement noise covariance
        # Initial state (position and velocity)
        # Assume ball starts in the middle of the court
        x0 = x0 if x0 is not None else np.array([self.width / 2, self.length / 2, self.height / 2, 0, 0, 0, 1])
        P = np.diag([self.width, self.length, self.height, self.width / 10, self.width / 10, self.width / 10,
                     0])  # Initial state uncertainty
        super().__init__(P, Q, R, x0)
        self.g = g

    def estimate_initial_state(self, observations):
        def project_to_2d(points_3d):
            return np.array([self.court_model.world2image(x) for x in points_3d])

        def parabolic_trajectory(t, x0, y0, z0, vx, vy, vz, g=9.81):
            """
            Compute the 3D position of the ball at time t for a parabolic trajectory.
            """
            x = x0 + vx * t
            y = y0 + vy * t
            z = z0 + vz * t - 0.5 * g * t ** 2
            return np.stack([x, y, z], axis=-1)

        def residuals(params, t_values, observed_2d):
            """
            Compute residuals between observed 2D points and projected 3D points.
            """
            x0, y0, z0, vx, vy, vz = params
            points_3d = parabolic_trajectory(np.array(t_values), x0, y0, z0, vx, vy, vz)
            projected_2d = project_to_2d(points_3d)
            return (projected_2d - observed_2d).ravel()

        # Initial guess for the parameters

        # Define known values: time values and observed 2D points
        u, v, t_values = zip(*observations)
        observed_2d = np.array(list(zip(u, v)))  # Observed 2D points (Nx2 array)

        # Initial guess for [x0, y0, z0, vx, vy, vz]
        initial_guess = [self.width / 2, self.length / 2, self.height / 2, 0, 0, 0]

        bounds = [
            [0, 0, 0, -np.inf, -np.inf, -np.inf],
            [self.width, self.length, np.inf, np.inf, np.inf, np.inf]
        ]

        # Perform least-squares fitting
        result = least_squares(
            residuals,
            initial_guess,
            args=(t_values, observed_2d),
            bounds=bounds,
            loss='cauchy',
            method='dogbox'
        )

        # Extract optimized parameters
        x0, y0, z0, vx, vy, vz = result.x
        print("Optimized initial position:", (x0, y0, z0))
        print("Optimized initial velocity:", (vx, vy, vz))
        print("R_squared:", 1 - result.cost / np.linalg.norm(observed_2d - observed_2d.mean(axis=0)) ** 2)

        estimated_initial_state = [x0, y0, z0, vx, vy, vz, 1]

        return estimated_initial_state

    def observation_function(self, x):
        return self.court_model.world2image(x[:3])

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

    def dump(self, filename):
        fig = self.plot()
        fig.show()
        # Save to file
        with open(filename, "w") as f:
            f.write(fig.to_html())

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
        z_range = [min(0, z_positions.min()), max(z_positions.max(), self.height)]

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
                        aspectmode='data',
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
                aspectmode='data',
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
