# Re-entry Vehicle State Estimation

Python simulation of a planar re-entry vehicle with noisy sensor measurements,
GPS blackout, and estimator comparison. Intended as a
GNC learning project. 
Vehicle dynamics are modelled, simulated
measurements generated and
covariance matrices propagated.
As a baseline comparison a dead-reckoning scheme which doesnt take any measurements into account is compared with an
Extended Kalman Filter (EKF) method.

## Project Goals

- Model simplified re-entry trajectory over a spherical, non-rotating Earth.
- Simulate navigation challenges like sensor noise and GPS blackout.
- Compare dead reckoning with Kalman-filter state estimation.
- Visualise state estimates, measurement availability, estimation error, and
  covariance bounds.
- Build a clear foundation for later Monte Carlo analysis and model validation.

## State Vector

The vehicle state is represented as:

```text
x = [h, s, v, gamma]
```

where:

- `h`: altitude above Earth surface, m
- `s`: distance from atmosphere entry point, m
- `v`: vehicle speed, m/s
- `gamma`: flight-path angle, rad

## Dynamics Model

The simulation uses a nonlinear point-mass re-entry model with:

- exponential atmosphere
- altitude-dependent gravity
- aerodynamic drag
- aerodynamic lift
- spherical Earth geometry
- fourth-order Runge-Kutta integration

The dynamics are implemented in `dynamics.py`, and the state is
propagated using RK4. A numerical finite-difference Jacobian is used to
linearise the discrete-time dynamics for covariance propagation.

## Sensors

The measurement model in `sensors.py` includes:

- barometric altitude
- GPS-like altitude, distance, and velocity measurements
- barometer noise
- GPS blackout between configured altitude limits

During GPS blackout, only the barometric altitude measurement is available.
This removes distance, speed, and flight-path angle, from the measurement matrix.

## Estimators

- `DeadReckoning`: propagates the nonlinear dynamics and covariance, ignoring
  measurements. This is used as the baseline comparison.

- `ExtendedKalmanFilter`: implements predict and update steps, including a
  Joseph-form covariance update for numerical robustness.

The EKF structure is:

```text
Prediction:
    x_minus = f(x)
    P_minus = F P F^T + Q

Update:
    y = z - H x_minus
    S = H P_minus H^T + R
    K = P_minus H^T S^-1
    x_plus = x_minus + K y
    P_plus = (I - K H) P_minus (I - K H)^T + K R K^T
```

The final covariance line is the Joseph form. It is algebraically equivalent to
the simpler covariance update in ideal arithmetic, but is preferred here because
it better preserves covariance symmetry and positive semi-definiteness in
finite-precision numerical calculations.

## Repository Structure

```text
.
├── config.py          # Vehicle, sensor, simulation, and estimator settings
├── dynamics.py        # Re-entry dynamics, RK4 propagation, Jacobian
├── estimator.py       # Dead reckoning and EKF estimator classes
├── main.py            # Runs truth generation, estimators, and plots
├── sensors.py         # Barometer/GPS measurement generation and blackout
├── simulation.py      # Truth trajectory and estimator execution loop
└── visualization.py   # Overview and estimation-error plots
```

## Running The Simulation

Install dependencies:

```bash
pip install numpy matplotlib
```

Run:

```bash
python main.py
```

The script generates a truth trajectory, simulated sensor measurements, and
estimator outputs. It then plots:

- truth versus estimated altitude, speed, flight-path angle, and downrange
- sensor measurements
- GPS blackout region
- estimation error with covariance bounds

## Current Development Status

This is an active learning project. The dynamics, sensor simulation, dead-reckoning baseline
estimator, EKF predict/update logic, Joseph-form covariance update, and plotting
workflow are implemented.

Near-term development items:

- compare EKF performance against dead reckoning
- add Monte Carlo runs over initial condition, sensor noise, and vehicle model
  uncertainty
- save representative plots for documentation
- add simple tests for estimator dimensions and covariance symmetry

## Engineering Notes

This model is intentionally simplified. It is useful for studying estimation
logic, sensor availability, and uncertainty propagation, but it is not a
high-fidelity flight-dynamics model. Important omitted effects include Earth
rotation, winds, full 3D motion, vehicle attitude dynamics, changing aerodynamic
coefficients, heating/ablation effects, and real sensor error calibration.

## Development Note

This is an AI-assisted learning project. AI tools were used to accelerate
implementation and documentation while I worked through the underlying dynamics,
control, and estimation concepts. The project is intended to demonstrate my
learning process, engineering judgement, and ability to build and validate
simulation tools.
