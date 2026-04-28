# simulation.py
"""
Simulation loop: propagate truth, generate measurements, run estimator.
"""
import numpy as np
from config import (DT, T_FINAL, MEAS_DT, X0_TRUE, X0_EST, P0, Q,
                    RANDOM_SEED)
from dynamics import propagate_state
from sensors import generate_measurements
from estimator import DeadReckoning, ExtendedKalmanFilter


def generate_truth_and_measurements():
    """
    Propagate the true trajectory and pre-generate all sensor data.
    This ensures every estimator sees exactly the same measurements.
    """
    np.random.seed(RANDOM_SEED)

    n_steps   = int(T_FINAL / DT)
    meas_every = max(1, round(MEAS_DT / DT))

    times  = [0.0]
    truths = [X0_TRUE.copy()]
    measurements = {}          # step → (z, H, R)

    state = X0_TRUE.copy()
    for step in range(1, n_steps + 1):
        state = propagate_state(state, DT)
        if state[0] <= 0:
            print(f"  Ground impact at t = {step * DT:.1f} s")
            break

        times.append(step * DT)
        truths.append(state.copy())

        if step % meas_every == 0:
            z, H, R = generate_measurements(state)
            measurements[step] = (z, H, R)

    return {
        'time':         np.array(times),
        'truth':        np.array(truths),
        'measurements': measurements,
    }


def run_estimator(truth_data, estimator_type='ekf'):
    """Run an estimator against the pre-generated truth + measurements."""
    time         = truth_data['time']
    truth        = truth_data['truth']
    measurements = truth_data['measurements']

    # Build estimator
    if estimator_type == 'dead_reckoning':
        est = DeadReckoning(X0_EST.copy(), P0.copy(), Q.copy())
    elif estimator_type == 'ekf':
        est = ExtendedKalmanFilter(X0_EST.copy(), P0.copy(), Q.copy())
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")

    n = len(time)
    estimates   = np.zeros((n, 4))
    covariances = np.zeros((n, 4))

    estimates[0]   = est.x.copy()
    covariances[0] = np.diag(est.P)

    for i in range(1, n):
        est.predict(DT)

        if i in measurements:
            z, H, R = measurements[i]
            est.update(z, H, R)

        estimates[i]   = est.x.copy()
        covariances[i] = np.diag(est.P)

    # Collect measurement data for plotting
    baro_meas    = []
    gps_pos_meas = []
    gps_vel_meas = []
    for step, (z, H, R) in sorted(measurements.items()):
        t = step * DT
        baro_meas.append((t, z[0]))
        if len(z) > 1:
            gps_pos_meas.append((t, z[1], z[2]))
            gps_vel_meas.append((t, z[3]))

    return {
        'time':          time,
        'truth':         truth,
        'estimate':      estimates,
        'covariance':    covariances,
        'baro_meas':     baro_meas,
        'gps_pos_meas':  gps_pos_meas,
        'gps_vel_meas':  gps_vel_meas,
    }