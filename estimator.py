# estimator.py
"""
State Estimators — YOUR IMPLEMENTATION GOES HERE
=================================================

State vector: x = [h, s, v, gamma]

The simulation loop calls two methods on your estimator:
    estimator.predict(dt)        — called every time step
    estimator.update(z, H, R)    — called when measurements arrive

Both should update self.x (state) and self.P (covariance) in place.

Helper functions from dynamics.py:
    propagate_state(x, dt)       — RK4 state propagation
    numerical_jacobian(x, dt)    — discrete-time Jacobian (F matrix)
"""
import numpy as np
from dynamics import propagate_state, numerical_jacobian


class Baseline :
    """
    Baseline: propagates dynamics, ignores all measurements.
    ─────────────────────────────────────────────────────────
    Study predict() below — you'll reuse the same logic in your EKF.
    """

    def __init__(self, x0, P0, Q):
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q.copy()

    def predict(self, dt):
        F = numerical_jacobian(self.x, dt)       # linearised state transition
        self.x = propagate_state(self.x, dt)      # nonlinear propagation
        self.P = F @ self.P @ F.T + self.Q        # covariance propagation

    def update(self, z, H, R):
        pass   # intentionally ignores measurements


class ExtendedKalmanFilter:

    def __init__(self, x0, P0, Q):
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q.copy()
    
    def predict(self, dt):
        F = numerical_jacobian(self.x, dt)       # linearised state transition
        self.x = propagate_state(self.x, dt)      # nonlinear propagation
        self.P = F @ self.P @ F.T + self.Q        # covariance propagation

    def update(self, z, H, R):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        A = I - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T

