"""Plotting and animation for WMS optimization results."""

from pathlib import Path

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np

from .wms import C_LIGHT_M_PER_S, DEFAULT_CENTER_HZ, DEFAULT_HWHM_HZ, get_hitran_profile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
ANIMATIONS_DIR = OUTPUT_DIR / "animations"

ALLOWED_ODD_HARMONICS = (1, 3, 5, 7, 9)


def _detuning_profile_norm(harmonics, amplitudes, phases_rad, n_points=2000):
    """Compute normalized detuning waveform over one period of theta."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    m_theta = np.zeros_like(theta)
    for h in harmonics:
        phi = 0.0 if h == 1 else phases_rad.get(h, 0.0)
        m_theta = m_theta + amplitudes[h] * np.sin(h * theta + phi)
    return theta, m_theta


def _bang_bang_dwell_profile(max_abs_m, n_points=2000):
    """Generate theoretical bang-bang dwell target waveform."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    m_bang = np.where(np.cos(2.0 * theta) >= 0.0, 0.0, max_abs_m * np.sign(np.sin(theta)))
    return theta, m_bang


def plot_vs_bang_bang(result, output_path=PLOTS_DIR / "waveform_vs_bang_bang.png", n_points=2000):
    """Plot optimized waveform against the bang-bang dwell target."""
    theta, m_opt = _detuning_profile_norm(result.harmonics, result.amplitudes, result.phases_rad, n_points=n_points)
    _, m_bang = _bang_bang_dwell_profile(float(np.max(np.abs(m_opt))), n_points=n_points)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    plt.plot(theta / np.pi, m_opt, label="Optimized odd-harmonic waveform", lw=2.0)
    plt.plot(theta / np.pi, m_bang, label="Theoretical bang-bang dwell target", lw=2.0, alpha=0.9)
    plt.xlabel(r"Phase $\theta / \pi$")
    plt.ylabel("Detuning / HWHM")
    plt.title("Optimized Waveform vs Bang-Bang Dwell Target")
    plt.xlim(0.0, 2.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_waveform_periods_and_amplitudes(
    result, compare_result=None,
    output_path=PLOTS_DIR / "waveform_3periods_and_amps.png",
    n_periods=3, points_per_period=1200,
):
    """Plot waveform over multiple periods with harmonic amplitude bar chart."""
    theta = np.linspace(0.0, 2.0 * np.pi * n_periods, n_periods * points_per_period, endpoint=False)

    m_theta = np.zeros_like(theta)
    for h in result.harmonics:
        phi = 0.0 if h == 1 else result.phases_rad.get(h, 0.0)
        m_theta = m_theta + result.amplitudes[h] * np.sin(h * theta + phi)

    m_theta_cmp = None
    if compare_result is not None:
        m_theta_cmp = np.zeros_like(theta)
        for h in compare_result.harmonics:
            phi = 0.0 if h == 1 else compare_result.phases_rad.get(h, 0.0)
            m_theta_cmp = m_theta_cmp + compare_result.amplitudes[h] * np.sin(h * theta + phi)

    harms = list(ALLOWED_ODD_HARMONICS)
    amps = [result.amplitudes.get(h, 0.0) for h in harms]
    amps_cmp = [compare_result.amplitudes.get(h, 0.0) for h in harms] if compare_result is not None else None

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), height_ratios=(2.3, 1.0))
    axes[0].plot(theta / (2.0 * np.pi), m_theta, lw=2.0, label=f"Optimized {result.harmonics}")
    if m_theta_cmp is not None:
        axes[0].plot(theta / (2.0 * np.pi), m_theta_cmp, lw=2.0, label=f"Optimized {compare_result.harmonics}")
    axes[0].set_xlim(0.0, float(n_periods))
    axes[0].set_xlabel("Time [periods]")
    axes[0].set_ylabel("Detuning / HWHM")
    axes[0].set_title(f"Optimized Waveform Over {n_periods} Periods")
    axes[0].grid(alpha=0.25)
    if m_theta_cmp is not None:
        axes[0].legend()

    x = np.arange(len(harms))
    if amps_cmp is None:
        axes[1].bar(x, amps, width=0.65, label=f"{result.harmonics}")
    else:
        width = 0.38
        axes[1].bar(x - width / 2, amps, width=width, label=f"{result.harmonics}")
        axes[1].bar(x + width / 2, amps_cmp, width=width, label=f"{compare_result.harmonics}")
    axes[1].set_xlabel("Odd Harmonic")
    axes[1].set_ylabel("Amplitude m_h")
    axes[1].set_title("Optimized Harmonic Amplitudes")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(h) for h in harms])
    axes[1].grid(alpha=0.25, axis="y")
    if amps_cmp is not None:
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_waveform_periods_nm(
    result, output_path=PLOTS_DIR / "waveform_3periods_nm.png",
    n_periods=3, points_per_period=1200,
    hwhm_hz=DEFAULT_HWHM_HZ, center_hz=DEFAULT_CENTER_HZ,
):
    """Plot waveform in wavelength (nm) over multiple periods."""
    theta = np.linspace(0.0, 2.0 * np.pi * n_periods, n_periods * points_per_period, endpoint=False)
    m_theta = np.zeros_like(theta)
    for h in result.harmonics:
        phi = 0.0 if h == 1 else result.phases_rad.get(h, 0.0)
        m_theta = m_theta + result.amplitudes[h] * np.sin(h * theta + phi)

    det_hz = m_theta * hwhm_hz
    f_hz = center_hz + det_hz
    lam_nm = (C_LIGHT_M_PER_S / f_hz) * 1e9

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    plt.plot(theta / (2.0 * np.pi), lam_nm, lw=2.0)
    plt.xlim(0.0, float(n_periods))
    plt.xlabel("Time [periods]")
    plt.ylabel("Wavelength [nm]")
    plt.title(f"Optimized Waveform Over {n_periods} Periods (nm)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def animate_scan_on_line(
    result, output_path=ANIMATIONS_DIR / "scan_on_line.gif", *,
    hwhm_hz=DEFAULT_HWHM_HZ, center_hz=DEFAULT_CENTER_HZ,
    n_periods=2, frames_per_period=100, fps=25,
    line_xlim_nm=(1650.0, 1652.0),
):
    """Create a GIF showing the scan cursor moving on the HITRAN line."""
    x_nm, y_norm, _, _, _ = get_hitran_profile()
    n_frames = max(2, int(n_periods * frames_per_period))

    theta = np.linspace(0.0, 2.0 * np.pi * n_periods, n_frames, endpoint=False)
    m_theta = np.zeros_like(theta)
    for h in result.harmonics:
        phi = 0.0 if h == 1 else result.phases_rad.get(h, 0.0)
        m_theta = m_theta + result.amplitudes[h] * np.sin(h * theta + phi)

    det_hz = m_theta * hwhm_hz
    f_hz = center_hz + det_hz
    lam_nm = (C_LIGHT_M_PER_S / f_hz) * 1e9
    y_scan = np.interp(lam_nm, x_nm, y_norm, left=0.0, right=0.0)

    t_periods = theta / (2.0 * np.pi)
    y2_min = float(np.min(lam_nm))
    y2_max = float(np.max(lam_nm))
    pad = 0.02 * (y2_max - y2_min if y2_max > y2_min else 1.0)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), height_ratios=(1.6, 1.0))
    ax0, ax1 = axes

    ax0.plot(x_nm, y_norm, color="C0", lw=2.0)
    (scan_pt,) = ax0.plot([], [], "o", color="C3", ms=7)
    ax0.set_xlim(line_xlim_nm[0], line_xlim_nm[1])
    ax0.set_ylim(-0.02, 1.05)
    ax0.set_xlabel("Wavelength [nm]")
    ax0.set_ylabel("Normalized Line Strength")
    ax0.set_title("Scan Position on HITRAN Methane Line")
    ax0.grid(alpha=0.25)

    ax1.plot(t_periods, lam_nm, color="C1", lw=2.0)
    (wave_pt,) = ax1.plot([], [], "o", color="C3", ms=6)
    cursor = ax1.axvline(0.0, color="0.2", lw=1.2, alpha=0.8)
    ax1.set_xlim(0.0, float(n_periods))
    ax1.set_ylim(y2_min - pad, y2_max + pad)
    ax1.set_xlabel("Time [periods]")
    ax1.set_ylabel("Wavelength [nm]")
    ax1.set_title("")
    ax1.grid(alpha=0.25)

    def lam_nm_to_m(lam):
        return (C_LIGHT_M_PER_S / (lam * 1e-9) - center_hz) / hwhm_hz

    def m_to_lam_nm(m):
        return (C_LIGHT_M_PER_S / (center_hz + m * hwhm_hz)) * 1e9

    ax1_right = ax1.secondary_yaxis("right", functions=(lam_nm_to_m, m_to_lam_nm))
    ax1_right.set_ylabel("m")
    ax1_right.set_yticks([-10.4, 0.0, 10.4])
    ax1_right.set_yticklabels(["-10.4", "0", "+10.4"])

    def _init():
        scan_pt.set_data([], [])
        wave_pt.set_data([], [])
        cursor.set_xdata([0.0, 0.0])
        return scan_pt, wave_pt, cursor

    def _update(i):
        scan_pt.set_data([lam_nm[i]], [y_scan[i]])
        wave_pt.set_data([t_periods[i]], [lam_nm[i]])
        cursor.set_xdata([t_periods[i], t_periods[i]])
        return scan_pt, wave_pt, cursor

    anim = FuncAnimation(fig, _update, frames=n_frames, init_func=_init, blit=True, interval=1000 / fps)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out
