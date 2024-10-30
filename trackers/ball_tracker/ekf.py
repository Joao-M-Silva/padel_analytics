import numpy as np
from numdifftools import Jacobian

"""
Implements the Extended Kalman Filter for 3D tracking.
"""


class ExtendedKalmanFilter:
    """
    Implements the Extended Kalman Filter with homogeneous coordinates for 3D tracking.
    """

    def __init__(self, P, Q, R, x0):
        self.P = P  # Initial state covariance matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = x0  # Initial state estimate (3D position and velocity)
        self.states = []
        self.state_uncertainties = []

    def transition_function(self, state, dt=1. / 30):
        # To be implemented
        pass

    def observation_function(self, x):
        # To be implemented
        pass

    def predict(self):
        # Predict the state and covariance
        self.x = self.transition_function(self.x)
        F = self._jacob_F(self.x)
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

        self.states.append(self.x)
        self.state_uncertainties.append(self.P)

    def _jacob_F(self, state):
        return Jacobian(lambda s: self.transition_function(s))(state)

    def _jacob_H(self, state):
        return Jacobian(lambda s: self.observation_function(s))(state)

    def update(self, z):
        # Compute Jacobians
        H = self._jacob_H(self.x)

        # Construct the predicted measurement from the state
        predicted_measurement = self.observation_function(self.x)

        # Measurement residual
        y = z - predicted_measurement

        # Compute the innovation covariance S (2x2)
        S = np.dot(H, np.dot(self.P, H.T))[:2, :2] + self.R  # H P H^T + R

        # Compute the Kalman gain K (7x2)
        K = np.dot(np.dot(self.P, H[:2].T), np.linalg.inv(S))  # P H^T S^{-1}

        # Update state estimate (position and velocity) and covariance
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, H[:2]), self.P)

    def track(self, measurements):
        states = []
        for measurement in measurements:
            self.update(measurement['xy'])
            self.predict()
            states.append(self.get_state())

        return np.array(states)

    def get_state(self):
        # Return the 3D position and velocity state estimate
        return self.x
