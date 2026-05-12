import numpy as np

from config import DT, P0, Q, X0_EST, X0_TRUE
from estimator import DeadReckoning, ExtendedKalmanFilter
from sensors import generate_measurements


def assert_valid_estimator_state(estimator):
    assert estimator.x.shape == (4,)
    assert estimator.P.shape == (4, 4)
    assert np.allclose(estimator.P, estimator.P.T, atol=1e-9)

    eigvals = np.linalg.eigvalsh(estimator.P)
    assert np.all(eigvals >= -1e-9)


def test_dead_reckoning_predict_preserves_valid_shapes_and_covariance():
    estimator = DeadReckoning(X0_EST, P0, Q)

    estimator.predict(DT)

    assert_valid_estimator_state(estimator)


def test_ekf_predict_preserves_valid_shapes_and_covariance():
    estimator = ExtendedKalmanFilter(X0_EST, P0, Q)

    estimator.predict(DT)

    assert_valid_estimator_state(estimator)


def test_ekf_update_preserves_valid_shapes_and_covariance():
    estimator = ExtendedKalmanFilter(X0_EST, P0, Q)
    estimator.predict(DT)

    z, H, R = generate_measurements(X0_TRUE)
    estimator.update(z, H, R)

    assert_valid_estimator_state(estimator)
