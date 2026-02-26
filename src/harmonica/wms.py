import jax.numpy as jnp
import numpy as np
from pathlib import Path

PI2 = 2.0 * jnp.pi
C_LIGHT_M_PER_S = 299_792_458.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_hitran_line():
    """Load methane line profile from data/hitran.npz."""
    data_path = PROJECT_ROOT / "data" / "hitran.npz"
    if not data_path.exists():
        data_path = Path(__file__).with_name("hitran.npz")
    data = np.load(data_path)
    x = np.asarray(data["x"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        raise ValueError("hitran.npz must contain non-empty x/y arrays with matching lengths.")

    if x[0] > x[-1]:
        x = x[::-1]
        y = y[::-1]

    i_peak = int(np.argmax(y))
    x0_nm = float(x[i_peak])
    y_peak = float(y[i_peak])
    if y_peak <= 0.0:
        raise ValueError("hitran.npz y array must have a positive peak.")

    y_norm = y / y_peak
    f0_hz = C_LIGHT_M_PER_S / (x0_nm * 1e-9)
    return x, y_norm, i_peak, x0_nm, f0_hz


def _estimate_hwhm_hz_from_profile(x_nm, y_norm, i_peak):
    """Estimate HWHM in Hz from half-maximum crossings around the peak."""
    il = int(i_peak)
    while il > 0 and y_norm[il] >= 0.5:
        il -= 1

    ir = int(i_peak)
    while ir < y_norm.size - 1 and y_norm[ir] >= 0.5:
        ir += 1

    if il == 0 or ir == y_norm.size - 1:
        raise ValueError("Could not bracket half-maximum crossings for HITRAN line.")

    def crossing(xa, xb, ya, yb, target=0.5):
        if yb == ya:
            return 0.5 * (xa + xb)
        t = (target - ya) / (yb - ya)
        return xa + t * (xb - xa)

    x_left = crossing(x_nm[il], x_nm[il + 1], y_norm[il], y_norm[il + 1])
    x_right = crossing(x_nm[ir - 1], x_nm[ir], y_norm[ir - 1], y_norm[ir])
    f_left = C_LIGHT_M_PER_S / (x_left * 1e-9)
    f_right = C_LIGHT_M_PER_S / (x_right * 1e-9)
    return abs(f_right - f_left) * 0.5


_HITRAN_X_NM, _HITRAN_Y_NORM, _HITRAN_I_PEAK, _HITRAN_CENTER_NM, _HITRAN_CENTER_HZ = _load_hitran_line()
_HITRAN_HWHM_HZ = _estimate_hwhm_hz_from_profile(_HITRAN_X_NM, _HITRAN_Y_NORM, _HITRAN_I_PEAK)
DEFAULT_HWHM_HZ = float(_HITRAN_HWHM_HZ)
DEFAULT_CENTER_NM = float(_HITRAN_CENTER_NM)
DEFAULT_CENTER_HZ = float(_HITRAN_CENTER_HZ)


def make_timebase(fm_hz, fs_hz, n_periods=1):
    """Return (t_s, dt_s, n_per_period) spanning an integer number of modulation periods."""
    if fm_hz <= 0.0:
        raise ValueError("fm_hz must be > 0.")
    if fs_hz <= 0.0:
        raise ValueError("fs_hz must be > 0.")
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1.")

    dt_s = 1.0 / fs_hz
    n_per_period_f = fs_hz / fm_hz
    n_per_period = int(round(n_per_period_f))

    if abs(n_per_period - n_per_period_f) > 1e-12:
        raise ValueError(
            f"fs_hz/fm_hz must be (very nearly) an integer. "
            f"Got fs_hz/fm_hz = {n_per_period_f}."
        )

    t_s = dt_s * jnp.arange(n_per_period * n_periods)
    return t_s, dt_s, n_per_period


def detuning_hz_sine(t_s, fm_hz, m1, hwhm_hz):
    """Detuning for pure sine: dv(t) = m1 * HWHM * sin(2*pi*fm*t)."""
    theta = PI2 * fm_hz * t_s
    return (m1 * hwhm_hz) * jnp.sin(theta)


def detuning_hz_1f3f(t_s, fm_hz, m1, m3, phi3_rad, hwhm_hz):
    """Detuning with 1st and 3rd harmonics."""
    theta = PI2 * fm_hz * t_s
    return hwhm_hz * (m1 * jnp.sin(theta) + m3 * jnp.sin(3.0 * theta + phi3_rad))


def detuning_hz_1f3f5f(t_s, fm_hz, m1, m3, phi3_rad, m5, phi5_rad, hwhm_hz):
    """Detuning with 1st, 3rd, and 5th harmonics."""
    theta = PI2 * fm_hz * t_s
    return hwhm_hz * (
        m1 * jnp.sin(theta)
        + m3 * jnp.sin(3.0 * theta + phi3_rad)
        + m5 * jnp.sin(5.0 * theta + phi5_rad)
    )


def detuning_hz_1f3f5f7f9f(t_s, fm_hz, m1, m3, phi3_rad, m5, phi5_rad, m7, phi7_rad, m9, phi9_rad, hwhm_hz):
    """Detuning with odd harmonics through 9f."""
    theta = PI2 * fm_hz * t_s
    return hwhm_hz * (
        m1 * jnp.sin(theta)
        + m3 * jnp.sin(3.0 * theta + phi3_rad)
        + m5 * jnp.sin(5.0 * theta + phi5_rad)
        + m7 * jnp.sin(7.0 * theta + phi7_rad)
        + m9 * jnp.sin(9.0 * theta + phi9_rad)
    )


def transmission_placeholder(detuning_hz, peak_abs, hwhm_hz):
    """Methane transmission from HITRAN profile, scaled to peak_abs at line center."""
    det_np = np.asarray(detuning_hz, dtype=np.float64)
    f_hz = _HITRAN_CENTER_HZ + det_np
    lam_nm = (C_LIGHT_M_PER_S / f_hz) * 1e9
    profile = np.interp(lam_nm, _HITRAN_X_NM, _HITRAN_Y_NORM, left=0.0, right=0.0)
    absorbance = peak_abs * profile
    transmission = np.exp(-absorbance)
    return jnp.asarray(transmission, dtype=detuning_hz.dtype)


def get_hitran_profile():
    """Return (wavelength_nm, normalized_strength, center_nm, center_hz, hwhm_hz)."""
    return (
        _HITRAN_X_NM.copy(),
        _HITRAN_Y_NORM.copy(),
        DEFAULT_CENTER_NM,
        DEFAULT_CENTER_HZ,
        DEFAULT_HWHM_HZ,
    )


def lockin_amps(signal, t_s, f_ref_hz, harmonics=(1, 2, 3)):
    """Return lock-in amplitudes |c_h| for requested harmonics."""
    hs = jnp.asarray(harmonics, dtype=jnp.float32)
    w = PI2 * f_ref_hz * hs[:, None]
    ref = jnp.exp(-1j * w * t_s[None, :])
    c = (2.0 / signal.size) * jnp.sum(signal[None, :] * ref, axis=1)
    return jnp.abs(c)


def simulate_wms(
    fm_hz=20_000.0,
    fs_hz=2_000_000.0,
    n_periods=1,
    waveform="sine",
    m1=2.25,
    m3=0.0,
    phi3_rad=0.0,
    m5=0.0,
    phi5_rad=0.0,
    m7=0.0,
    phi7_rad=0.0,
    m9=0.0,
    phi9_rad=0.0,
    peak_abs=0.5,
    hwhm_hz=DEFAULT_HWHM_HZ,
    noise_std_time=1e-3,
    harmonics=(2,),
    return_trace=False,
):
    """Run a WMS toy simulation and return (amplitudes, SNRs, slew_rms, slew_max, dt_s, n_per_period)."""
    if hwhm_hz <= 0.0:
        raise ValueError("hwhm_hz must be > 0.")
    if noise_std_time < 0.0:
        raise ValueError("noise_std_time must be >= 0.")

    t_s, dt_s, n_per_period = make_timebase(fm_hz, fs_hz, n_periods=n_periods)

    if waveform == "sine":
        det_hz = detuning_hz_sine(t_s, fm_hz=fm_hz, m1=m1, hwhm_hz=hwhm_hz)
    elif waveform == "1f3f":
        det_hz = detuning_hz_1f3f(t_s, fm_hz=fm_hz, m1=m1, m3=m3, phi3_rad=phi3_rad, hwhm_hz=hwhm_hz)
    elif waveform == "1f3f5f":
        det_hz = detuning_hz_1f3f5f(
            t_s, fm_hz=fm_hz, m1=m1, m3=m3, phi3_rad=phi3_rad,
            m5=m5, phi5_rad=phi5_rad, hwhm_hz=hwhm_hz,
        )
    elif waveform == "1f3f5f7f9f":
        det_hz = detuning_hz_1f3f5f7f9f(
            t_s, fm_hz=fm_hz, m1=m1, m3=m3, phi3_rad=phi3_rad,
            m5=m5, phi5_rad=phi5_rad, m7=m7, phi7_rad=phi7_rad,
            m9=m9, phi9_rad=phi9_rad, hwhm_hz=hwhm_hz,
        )
    else:
        raise ValueError(f"Unknown waveform='{waveform}'.")

    transmission = transmission_placeholder(det_hz, peak_abs=peak_abs, hwhm_hz=hwhm_hz)
    amps_all = lockin_amps(transmission, t_s, f_ref_hz=fm_hz, harmonics=harmonics)

    ddt = jnp.diff(det_hz) / dt_s
    slew_rms = jnp.sqrt(jnp.mean(ddt * ddt))
    slew_max = jnp.max(jnp.abs(ddt))

    N = t_s.size
    sigma_amp = 2.0 * noise_std_time / jnp.sqrt(N)
    snrs = amps_all / sigma_amp

    if return_trace:
        return amps_all, snrs, slew_rms, slew_max, dt_s, n_per_period, det_hz, transmission
    return amps_all, snrs, slew_rms, slew_max, dt_s, n_per_period


if __name__ == "__main__":
    (amp2_sin,), (snr2_sin,), rms_sin, max_sin, dt_s, npp = simulate_wms(
        waveform="sine", m1=2.25, harmonics=(2,)
    )

    (amp2_13,), (snr2_13,), rms_13, max_13, *_ = simulate_wms(
        waveform="1f3f", m1=4.375, m3=1.88125, phi3_rad=jnp.pi, harmonics=(2,)
    )

    fs_hz = 1.0 / float(dt_s)
    print(f"fm_hz = 20 kHz, fs_hz = {fs_hz/1e6:.3f} MHz, n_per_period = {npp}")
    print("2f amp (sine):   ", float(amp2_sin))
    print("2f amp (1f+3f):  ", float(amp2_13))
    print("2f gain:         ", float(amp2_13 / amp2_sin))
    print("2f SNR (sine):   ", float(snr2_sin))
    print("2f SNR (1f+3f):  ", float(snr2_13))
    print("slew_rms [Hz/s] (sine):  ", float(rms_sin), "slew_max [Hz/s]:", float(max_sin))
    print("slew_rms [Hz/s] (1f+3f): ", float(rms_13), "slew_max [Hz/s]:", float(max_13))
