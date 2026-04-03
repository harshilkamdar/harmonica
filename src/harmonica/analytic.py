"""Analytic optimal waveform for Lorentzian line profile.

g(t) such that transmission through a Lorentzian produces a pure sin(2t),
concentrating all modulated power into the 2f harmonic.

From: 1 - A * L(g(t)) = 1 - A*(1 + sin(2t))/2
=> g(t) = gamma * sqrt(2/(1+sin(2t)) - 1) * sign(cos(t + pi/4))
"""

import numpy as np

from .waveforms import analytic_lorentzian as analytic_optimal_m
from .wms import (
    DEFAULT_HWHM_HZ,
    lockin_amps,
    make_timebase,
    simulate_wms,
    transmission,
)


def lorentzian_transmission(x_hz_arr, peak_abs, hwhm_hz=DEFAULT_HWHM_HZ):
    """Pure Lorentzian transmission: T = 1 - A / (1 + (x/gamma)^2)."""
    x = np.asarray(x_hz_arr, dtype=np.float64)
    return 1.0 - peak_abs / (1.0 + (x / hwhm_hz) ** 2)


def eval_analytic_2f(
    m_clip=None, *,
    fm_hz=20_000.0, fs_hz=2_000_000.0, n_periods=1,
    peak_abs=0.5, hwhm_hz=DEFAULT_HWHM_HZ, noise_std_time=1e-3,
    use_lorentzian=False, line_profile=None,
):
    """Evaluate 2f amplitude of the analytic waveform through a line profile.

    Parameters
    ----------
    use_lorentzian : bool
        If True, use a pure Lorentzian instead of the HITRAN profile.
    """
    t_s, dt_s, n_per_period = make_timebase(fm_hz, fs_hz, n_periods=n_periods)
    theta = 2.0 * np.pi * fm_hz * np.asarray(t_s, dtype=np.float64)

    m = analytic_optimal_m(theta, m_clip=m_clip)
    x = hwhm_hz * m

    if use_lorentzian:
        trans = lorentzian_transmission(x, peak_abs=peak_abs, hwhm_hz=hwhm_hz)
        import jax.numpy as jnp
        trans = jnp.asarray(trans)
    else:
        trans = transmission(x, peak_abs=peak_abs, line_profile=line_profile)

    (amp2,) = lockin_amps(trans, t_s, f_ref_hz=fm_hz, harmonics=(2,))
    amp2 = float(amp2)

    sigma_amp = 2.0 * noise_std_time / np.sqrt(t_s.size)
    snr2 = amp2 / sigma_amp

    # sine baseline
    (amp2_sine,), *_ = simulate_wms(
        amplitudes={1: 2.25}, harmonics=(2,),
        fm_hz=fm_hz, fs_hz=fs_hz, n_periods=n_periods,
        peak_abs=peak_abs, hwhm_hz=hwhm_hz, noise_std_time=noise_std_time,
        line_profile=line_profile,
    )
    amp2_sine = float(amp2_sine)

    return {
        "amp2": amp2,
        "snr2": snr2,
        "amp2_sine_baseline": amp2_sine,
        "gain_vs_sine": amp2 / amp2_sine,
        "m_clip": m_clip,
    }
