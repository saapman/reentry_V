# main.py
"""
Re-entry State Estimation — Entry Point
========================================
Run:  python main.py
"""
from simulation import generate_truth_and_measurements, run_estimator
from visualization import plot_overview, plot_errors


def main():
    print("=" * 55)
    print("  Re-entry Vehicle — State Estimation")
    print("=" * 55)

    # Generate truth trajectory + sensor data (shared by all estimators)
    print("\nGenerating truth trajectory and measurements...")
    truth_data = generate_truth_and_measurements()

    # Baseline — dead reckoning (always works)
    print("Running dead reckoning (baseline)...")
    res_dr = run_estimator(truth_data, 'dead_reckoning')

    # EKF — will show flat estimate until you implement it
    print("Running EKF...")
    res_ekf = run_estimator(truth_data, 'ekf')

    # ── Plots ──────────────────────────────────────
    print("\nPlotting...")
    plot_overview([
        ('Dead Reckoning', 'orangered',  res_dr),
        ('EKF',            'dodgerblue', res_ekf),
    ])
    plot_errors(res_ekf, title='EKF — Estimation Errors with ±2σ Bounds')


if __name__ == "__main__":
    main()