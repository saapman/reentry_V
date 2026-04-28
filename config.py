# config.py
"""
Re-entry State Estimation — Configuration
==========================================
State vector: x = [h, s, v, gamma]
  h     — altitude (m)
  s     — downrange distance (m)
  v     — speed (m/s)
  gamma — flight path angle (rad, negative = descending)
"""
import numpy as np

# ── Vehicle (Apollo-like capsule) ──────────────────────
MASS  = 5000.0     # kg
CD    = 1.2        # drag coefficient
CL    = 0.36       # lift coefficient (L/D ≈ 0.3)
A_REF = 10.0       # heat-shield reference area, m²

# ── Planet ─────────────────────────────────────────────
R_EARTH = 6_371_000.0    # m
G0      = 9.81           # surface gravity, m/s²
RHO_0   = 1.225          # sea-level air density, kg/m³
H_SCALE = 7500.0         # atmospheric scale height, m

# ── Simulation timing ─────────────────────────────────
DT      = 0.1      # integration step, s
T_FINAL = 350.0    # total time, s
MEAS_DT = 1.0      # measurement interval, s

# ── Reproducibility ───────────────────────────────────
RANDOM_SEED = 42

# ── True initial state ────────────────────────────────
X0_TRUE = np.array([
    120_000.0,            # h:     120 km altitude
    0.0,                  # s:     0 downrange
    7500.0,               # v:     7.5 km/s entry speed
    np.radians(-2.0),     # gamma: -2° (shallow descent)
])

# ── Estimator initial conditions (deliberately wrong) ─
X0_EST = X0_TRUE + np.array([800.0, 400.0, -40.0, np.radians(0.4)])

P0 = np.diag([
    1000.0**2,            # h uncertainty
    1000.0**2,            # s uncertainty
    50.0**2,              # v uncertainty
    np.radians(1.0)**2,   # gamma uncertainty
])

# ── Process noise (per time step) ─────────────────────
# Accounts for atmospheric / aero modelling errors.
# Tune these! Too small → filter ignores measurements.
#              Too large → estimate is noisy.
Q = np.diag([
    25.0**2,
    25.0**2,
    5.0**2,
    np.radians(0.15)**2,
]) * DT

# ── Sensor noise (1-sigma) ────────────────────────────
SIGMA_BARO_BASE = 30.0      # barometer altitude noise at low alt (m)
SIGMA_GPS_POS   = 15.0      # GPS position noise (m)
SIGMA_GPS_VEL   = 0.5       # GPS velocity noise (m/s)

# ── GPS blackout zone (plasma sheath) ─────────────────
GPS_BLACKOUT_UPPER = 85_000.0   # m
GPS_BLACKOUT_LOWER = 30_000.0   # m