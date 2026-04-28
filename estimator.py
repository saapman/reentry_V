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


class DeadReckoning:
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
    """
    Extended Kalman Filter — TODO
    ──────────────────────────────

    PREDICT STEP  (identical to DeadReckoning.predict):
        1.  F  = numerical_jacobian(x, dt)
        2.  x⁻ = propagate_state(x, dt)
        3.  P⁻ = F  P  Fᵀ + Q

    UPDATE STEP:
        1.  Innovation:            y = z  −  H x⁻
        2.  Innovation covariance: S = H P⁻ Hᵀ + R
        3.  Kalman gain:           K = P⁻ Hᵀ S⁻¹
        4.  State update:          x⁺ = x⁻ + K y
        5.  Covariance update:     P⁺ = (I − K H) P⁻

    Useful numpy:
        A @ B           matrix multiply
        A.T             transpose
        np.eye(n)       identity matrix
        np.linalg.inv(S)
    """

    def __init__(self, x0, P0, Q):
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q.copy()

    def predict(self, dt):
        # ── TODO ──────────────────────────────────
        pass

    def update(self, z, H, R):
        # ── TODO ──────────────────────────────────
        # Note: z and H change size depending on GPS availability.
        # Your code should work for any size — the linear algebra
        # is identical, numpy handles the dimensions automatically.
        pass