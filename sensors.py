# sensors.py
"""
Sensor models: barometric altimeter + GPS (with plasma blackout).
"""
import numpy as np
from config import (SIGMA_BARO_BASE, SIGMA_GPS_POS, SIGMA_GPS_VEL,
                    GPS_BLACKOUT_UPPER, GPS_BLACKOUT_LOWER)


def gps_available(h):
    """Returns False inside the plasma-blackout altitude band."""
    return not (GPS_BLACKOUT_LOWER <= h <= GPS_BLACKOUT_UPPER)


def baro_noise_sigma(h):
    """Barometer noise degrades exponentially above 40 km."""
    return SIGMA_BARO_BASE * np.exp(max(0.0, h - 40_000.0) / 25_000.0)


def generate_measurements(true_state):
    """
    Sample noisy measurements from the true state.

    Returns
    -------
    z : measurement vector
    H : observation matrix
    R : measurement noise covariance
    """
    h, s, v, gamma = true_state
    sigma_b = baro_noise_sigma(h)

    # Barometric altimeter (always available)
    z_baro = h + np.random.randn() * sigma_b

    if gps_available(h):
        # GPS gives independent h, s, and v measurements
        z = np.array([
            z_baro,
            h + np.random.randn() * SIGMA_GPS_POS,
            s + np.random.randn() * SIGMA_GPS_POS,
            v + np.random.randn() * SIGMA_GPS_VEL,
        ])
        H = np.array([
            [1, 0, 0, 0],   # baro  → h
            [1, 0, 0, 0],   # gps   → h
            [0, 1, 0, 0],   # gps   → s
            [0, 0, 1, 0],   # gps   → v
        ])
        R = np.diag([
            sigma_b ** 2,
            SIGMA_GPS_POS ** 2,
            SIGMA_GPS_POS ** 2,
            SIGMA_GPS_VEL ** 2,
        ])
    else:
        # Baro only
        z = np.array([z_baro])
        H = np.array([[1, 0, 0, 0]])
        R = np.array([[sigma_b ** 2]])

    return z, H, R