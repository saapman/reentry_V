# visualization.py
"""
Plotting: overview comparison and estimation-error analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from config import GPS_BLACKOUT_LOWER, GPS_BLACKOUT_UPPER

def _blackout_ranges(time, truth):
    """Return list of (t_start, t_end) for GPS blackout periods."""
    ranges = []
    inside = False
    t0 = 0.0
    for i, t in enumerate(time):
        h = truth[i, 0]
        blacked = GPS_BLACKOUT_LOWER <= h <= GPS_BLACKOUT_UPPER
        if blacked and not inside:
            t0 = t;  inside = True
        elif not blacked and inside:
            ranges.append((t0, t));  inside = False
    if inside:
        ranges.append((t0, time[-1]))
    return ranges


def _shade(ax, ranges, first_label=True):
    for j, (a, b) in enumerate(ranges):
        ax.axvspan(a, b, alpha=0.12, color='red',
                   label='GPS blackout' if (j == 0 and first_label) else '')


def plot_overview(results_list):
    """
    Overlay truth + measurements + one or more estimator results.

    Parameters
    ----------
    results_list : list of (label, colour, results_dict)
    """
    _, _, r0 = results_list[0]
    time  = r0['time']
    truth = r0['truth']
    baro  = r0['baro_meas']
    gps_p = r0['gps_pos_meas']
    gps_v = r0['gps_vel_meas']
    bo    = _blackout_ranges(time, truth)

    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    fig.suptitle('Re-entry Vehicle — State Estimation', fontsize=14,
                 fontweight='bold')

    for ax in axes:
        _shade(ax, bo, first_label=(ax is axes[0]))

    # ── Altitude ──────────────────────────────
    ax = axes[0]
    ax.plot(time, truth[:, 0] / 1e3, 'k-', lw=2, label='Truth', zorder=5)
    for lbl, clr, res in results_list:
        ax.plot(res['time'], res['estimate'][:, 0] / 1e3, '--',
                color=clr, lw=1.5, label=lbl, zorder=4)
    if baro:
        ax.scatter([m[0] for m in baro], [m[1] / 1e3 for m in baro],
                   s=3, c='orange', alpha=0.4, label='Baro', zorder=2)
    if gps_p:
        ax.scatter([m[0] for m in gps_p], [m[1] / 1e3 for m in gps_p],
                   s=8, c='limegreen', alpha=0.7, label='GPS', zorder=3)
    ax.set_ylabel('Altitude (km)')
    ax.legend(loc='upper right', fontsize=8);  ax.grid(True, alpha=0.3)

    # ── Speed ─────────────────────────────────
    ax = axes[1]
    ax.plot(time, truth[:, 2], 'k-', lw=2, label='Truth', zorder=5)
    for lbl, clr, res in results_list:
        ax.plot(res['time'], res['estimate'][:, 2], '--',
                color=clr, lw=1.5, label=lbl, zorder=4)
    if gps_v:
        ax.scatter([m[0] for m in gps_v], [m[1] for m in gps_v],
                   s=8, c='limegreen', alpha=0.7, label='GPS', zorder=3)
    ax.set_ylabel('Speed (m/s)')
    ax.legend(loc='upper right', fontsize=8);  ax.grid(True, alpha=0.3)

    # ── Flight-path angle ─────────────────────
    ax = axes[2]
    ax.plot(time, np.degrees(truth[:, 3]), 'k-', lw=2, label='Truth', zorder=5)
    for lbl, clr, res in results_list:
        ax.plot(res['time'], np.degrees(res['estimate'][:, 3]), '--',
                color=clr, lw=1.5, label=lbl, zorder=4)
    ax.set_ylabel('Flight-path angle (°)')
    ax.legend(loc='lower left', fontsize=8);  ax.grid(True, alpha=0.3)

    # ── Downrange ─────────────────────────────
    ax = axes[3]
    ax.plot(time, truth[:, 1] / 1e3, 'k-', lw=2, label='Truth', zorder=5)
    for lbl, clr, res in results_list:
        ax.plot(res['time'], res['estimate'][:, 1] / 1e3, '--',
                color=clr, lw=1.5, label=lbl, zorder=4)
    if gps_p:
        ax.scatter([m[0] for m in gps_p], [m[2] / 1e3 for m in gps_p],
                   s=8, c='limegreen', alpha=0.7, label='GPS', zorder=3)
    ax.set_ylabel('Downrange (km)')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper left', fontsize=8);  ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig("assets/overview.png", dpi=200, bbox_inches="tight")
    plt.show()

def plot_errors(results, title='Estimation Errors'):
    """
    Plot estimation error for each state with ±2σ covariance bounds.
    """
    time  = results['time']
    truth = results['truth']
    est   = results['estimate']
    cov   = results['covariance']
    sigma = np.sqrt(np.maximum(cov, 0))

    err = est - truth
    bo  = _blackout_ranges(time, truth)

    labels = ['Altitude error (m)', 'Downrange error (m)',
              'Speed error (m/s)',  'FPA error (°)']
    scales = [1.0, 1.0, 1.0, np.degrees(1.0)]

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for i, ax in enumerate(axes):
        _shade(ax, bo, first_label=(i == 0))
        e = err[:, i] * scales[i]
        s = sigma[:, i] * scales[i]
        ax.plot(time, e, 'b-', lw=1, label='Error')
        ax.fill_between(time, -2 * s, 2 * s, alpha=0.2, color='blue',
                        label='±2σ bound')
        ax.axhline(0, color='k', ls='--', lw=0.5, alpha=0.5)
        ax.set_ylabel(labels[i])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    fig.savefig("assets/ekf_errors.png", dpi=200, bbox_inches="tight")
    plt.show()