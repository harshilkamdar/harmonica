"""Objective functions: waveform array m(θ) → scalar fitness (amp2).

SimObjective: uses simulated transmission + lock-in.
HardwareObjective: placeholder for Red Pitaya hardware-in-the-loop.
"""

import numpy as np

from .wms import lockin_amps, make_timebase, transmission


class SimObjective:
    """Evaluate 2f amplitude via simulated transmission through a line profile."""

    def __init__(self, fm_hz=20_000.0, fs_hz=2_000_000.0, n_periods=1,
                 peak_abs=0.5, hwhm_hz=None, noise_std_time=1e-3,
                 line_profile=None):
        from .wms import DEFAULT_HWHM_HZ
        self.fm_hz = fm_hz
        self.fs_hz = fs_hz
        self.n_periods = n_periods
        self.peak_abs = peak_abs
        self.hwhm_hz = hwhm_hz or DEFAULT_HWHM_HZ
        self.noise_std_time = noise_std_time
        self.line_profile = line_profile

        # precompute timebase and theta
        t_s, self.dt_s, self.n_per_period = make_timebase(fm_hz, fs_hz, n_periods)
        self.t_s = t_s
        self.theta = 2.0 * np.pi * fm_hz * np.asarray(t_s, dtype=np.float64)

    def __call__(self, m_theta):
        """m_theta: normalized waveform array (same length as self.theta). Returns amp2."""
        x = self.hwhm_hz * m_theta
        trans = transmission(x, peak_abs=self.peak_abs, line_profile=self.line_profile)
        (amp2,) = lockin_amps(trans, self.t_s, f_ref_hz=self.fm_hz, harmonics=(2,))
        return float(amp2)


class HardwareObjective:
    """Placeholder for hardware-in-the-loop evaluation via Red Pitaya."""

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "HardwareObjective is a placeholder. "
            "Implement RP send/receive to use hardware-in-the-loop optimization."
        )

    def __call__(self, m_theta):
        raise NotImplementedError
