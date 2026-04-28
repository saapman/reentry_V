# dynamics.py
"""
Planar re-entry dynamics over a spherical, non-rotating Earth.
"""
import numpy as np
from config import MASS, CD, CL, A_REF, R_EARTH, G0, RHO_0, H_SCALE


def atmosphere(h):
    """Exponential atmosphere model → density (kg/m³)."""
    return RHO_0 * np.exp(-np.clip(h, 0, None) / H_SCALE)


def gravity(h):
    """Gravitational acceleration at altitude h (m/s²)."""
    return G0 * (R_EARTH / (R_EARTH + h)) ** 2


def derivatives(state):
    """
    Continuous-time equations of motion.

    Parameters
    ----------
    state : [h, s, v, gamma]

    Returns
    -------
    [dh/dt, ds/dt, dv/dt, dgamma/dt]
    """
    h, s, v, gamma = state

    rho = atmosphere(h)
    g   = gravity(h)
    q   = 0.5 * rho * v ** 2          # dynamic pressure
    D   = q * CD * A_REF              # drag
    L   = q * CL * A_REF              # lift

    dh     = v * np.sin(gamma)
    ds     = v * np.cos(gamma) * R_EARTH / (R_EARTH + h)
    dv     = -D / MASS - g * np.sin(gamma)
    dgamma = (1.0 / v) * (L / MASS - g * np.cos(gamma)
              + v ** 2 * np.cos(gamma) / (R_EARTH + h))

    return np.array([dh, ds, dv, dgamma])


def propagate_state(state, dt):
    """Advance state by dt using RK4 integration."""
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def numerical_jacobian(state, dt, eps=1e-6):
    """
    Jacobian of the discrete-time state transition (propagate_state)
    via central finite differences.

    Returns
    -------
    F : (4, 4) ndarray
        F[i, j] = ∂f_i / ∂x_j   where x_{k+1} = f(x_k)
    """
    n = len(state)
    F = np.zeros((n, n))
    for j in range(n):
        x_plus  = state.copy();  x_plus[j]  += eps
        x_minus = state.copy();  x_minus[j] -= eps
        F[:, j] = (propagate_state(x_plus, dt)
                    - propagate_state(x_minus, dt)) / (2.0 * eps)
    return F