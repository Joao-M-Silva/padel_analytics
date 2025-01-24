from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pytest import fixture

from trackers.ball_tracker.ekf import ExtendedKalmanFilter


class MockExtendedKalmanFilter(ExtendedKalmanFilter):
    """
    Implement a simple KalmanFilter for testing purposes.
    The ball lives in 2D and just bounces on the floor.
    """

    def __init__(self, x0, p=.1, q=.1, r=.1):
        """
        Initialize the Kalman filter with
        :param x0: Initial state
        :param p: Initial state covariance matrix
        :param q: Process noise covariance
        :param r: Measurement noise covariance
        """
        self.x0 = x0
        self.p = p
        self.q = q
        self.r = r

        P = np.diag([p, p, p ** 2, p ** 2, 0])
        Q = np.diag([0, 0, q, q, 0])
        R = np.diag([r, r])
        self.g = 10

        super().__init__(P, Q, R, x0)

    def transition_function(self, state, dt=1. / 30):
        # State transition matrices
        F = np.array(
            [
                [1, 0, dt, 0, 0],
                [0, 1, 0, dt, -0.5 * self.g * dt ** 2],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, - self.g * dt],
                [0, 0, 0, 0, 1]
            ]
        )

        new_state = np.dot(F, state)

        # Bounce off floor
        if new_state[1] < 0:
            new_state[1] = -new_state[1]  # reflect position over Z = 0
            new_state[3] = -new_state[3] * 0.9  # reflect velocity




        return new_state

    def observation_function(self, x):
        # Assume we observe the position undistorted full
        return np.array([x[0], x[1]])

    def reset(self):
        return MockExtendedKalmanFilter(self.x0, self.p, self.q, self.r)


@fixture
def ekf():
    """
    Initialize the Kalman Filter in order to track the ball location
    """
    ekf_instance = MockExtendedKalmanFilter(
        x0=np.array([0, 1, 1, 0, 1]),
        p=.0001,
        q=3,
        r=0.05
    )
    yield ekf_instance


@fixture
def true_states(ekf):
    """
    Generate a sequence of true ball locations, to be used as ground truth to test the Kalman filter
    """
    ekf_copy = ekf.reset()
    # Generate true state values
    ts = []
    for i in range(500):
        ekf_copy.predict()
        # Randomly kick the state
        if np.random.rand() < 0.1:
            dvx, dvy = np.random.multivariate_normal(np.zeros(2), np.eye(2) * ekf_copy.q)
            ekf_copy.x += np.array([0, 0, dvx, dvy, 0])
        ts.append(ekf_copy.x)

    yield np.array(ts)


@fixture
def noisy_measurements(true_states, ekf):
    """
    Simulate noisy measurements of true ball locations, to be fed to the Kalman filter
    """
    # Generate noisy measurements
    measurements = []
    for state in true_states:
        # Use the noise matrix R to generate the obervation noise
        noise = np.random.multivariate_normal(np.zeros(2), ekf.R)
        measurements.append(ekf.observation_function(state) + noise)

    yield np.array(measurements)


class TestExtendedKalman:
    def test_predict(self, ekf):
        """
        Test that predictions are tracked by the filter
        Test that prediction uncertainties are tracked by the filter
        """
        for i in range(10):
            ekf.predict()

        assert len(ekf.states) == 10
        assert len(ekf.state_uncertainties) == 10

    def test_bounce(self, ekf):
        """
        Make sure that the prediction never goes below the zero line
        """
        states = []
        for i in range(100):
            ekf.predict()

        states = np.array(ekf.states)

        assert all(states[:, 0] >= 0)

    def test_kalman_gain(self, ekf, true_states, noisy_measurements):
        """
        Assert that the update method takes the measurement noise into account
        - Smaller R means (larger Gain) means new state more sensitive to measurements
        - Larger R means (smaller Gain) means new state less sensitive to measurements
        """

        # Track the ball for 10 steps
        for i in range(10):
            ekf.predict()
            ekf.update(noisy_measurements[i])

        # Deepcopy the filter
        ekf_copy_larger_r = deepcopy(ekf)
        ekf_copy_smaller_r = deepcopy(ekf)

        # Set a much larger measurement noise for one of the filters
        ekf_copy_larger_r.R = ekf.R * 2
        ekf_copy_smaller_r.R = ekf.R / 2

        # Predict and update all filters with the next measurement
        ekf.predict()
        ekf.update(noisy_measurements[10])
        ekf_copy_larger_r.predict()
        ekf_copy_larger_r.update(noisy_measurements[10])
        ekf_copy_smaller_r.predict()
        ekf_copy_smaller_r.update(noisy_measurements[10])

        # Check that the filter with less noise predicts a state closer to the true state
        ekf_state = ekf.x
        ekf_copy_larger_r_state = ekf_copy_larger_r.x
        ekf_copy_smaller_r_state = ekf_copy_smaller_r.x

        # Measure the baseline distance from the internal state to the last measurement
        reference_distance_to_measurement = np.linalg.norm(ekf_state[:2] - noisy_measurements[10])

        # With smaller gain, the state should be further to the measurement
        assert np.linalg.norm(ekf_copy_larger_r_state[:2] - noisy_measurements[10]) > reference_distance_to_measurement

        # With larger gian, the state should be closer to the measurement
        assert np.linalg.norm(ekf_copy_smaller_r_state[:2] - noisy_measurements[10]) < reference_distance_to_measurement

    def test_plot(self, ekf, true_states, noisy_measurements):

        # Update the filter with the noisy measurements
        states = ekf.track([{'xy': nm} for nm in noisy_measurements])

        df = pd.DataFrame(
            dict(
                x_true=true_states[:, 0],
                y_true=true_states[:, 1],
                vx_true=true_states[:, 2],
                vy_true=true_states[:, 3],
                x_inf=states[:, 0],
                y_inf=states[:, 1],
                vx_inf=states[:, 2],
                vy_inf=states[:, 3],
                x_obs=noisy_measurements[:, 0],
                y_obs=noisy_measurements[:, 1]
            )
        )
