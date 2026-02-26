"""Phase-distorted sine optimization for WMS-2f."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from .wms import DEFAULT_HWHM_HZ, lockin_amps, make_timebase, simulate_wms, transmission

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"


@dataclass
class PhaseDistortedResult:
    m_max: float
    p_terms: tuple
    amp2: float
    snr2: float
    amp2_sine_baseline: float
    gain_vs_sine: float
    success: bool
    message: str
    nfev: int


def phase_distorted_m(theta, m_max, p_terms):
    """Normalized modulation: m(theta) = m_max * sin(theta + sum_k p_k * sin(k*theta))."""
    phase_warp = np.zeros_like(theta)
    for k, pk in enumerate(p_terms, start=1):
        phase_warp += pk * np.sin(k * theta)
    return m_max * np.sin(theta + phase_warp)


def _amp2_and_snr2_phase(
    m_max, p_terms, *,
    fm_hz=20_000.0, fs_hz=2_000_000.0, n_periods=1,
    peak_abs=0.5, hwhm_hz=DEFAULT_HWHM_HZ, noise_std_time=1e-3,
):
    """Evaluate 2f amplitude and SNR for a phase-distorted waveform."""
    t_s, _, _ = make_timebase(fm_hz, fs_hz, n_periods=n_periods)
    theta = 2.0 * np.pi * fm_hz * np.asarray(t_s, dtype=np.float64)

    m_theta = phase_distorted_m(theta, m_max, p_terms)
    x = hwhm_hz * m_theta
    trans = transmission(x, peak_abs=peak_abs)

    (amp2,) = lockin_amps(trans, t_s, f_ref_hz=fm_hz, harmonics=(2,))
    amp2 = float(amp2)

    sigma_amp = 2.0 * noise_std_time / np.sqrt(t_s.size)
    snr2 = amp2 / sigma_amp
    return amp2, float(snr2)


def optimize_phase_distorted_amp2(
    *, n_phase_terms=4,
    m_max_bounds=(0.0, 8.0), p_bounds=(-1.5, 1.5),
    sine_m1_baseline=2.25,
    fm_hz=20_000.0, fs_hz=2_000_000.0, n_periods=1,
    peak_abs=0.5, hwhm_hz=DEFAULT_HWHM_HZ, noise_std_time=1e-3,
    de_seed=0, de_maxiter=220, de_popsize=20,
):
    """Maximize 2f amplitude by optimizing m_max and phase warp coefficients p_k."""
    if n_phase_terms < 1:
        raise ValueError("n_phase_terms must be >= 1.")

    sim_kwargs = dict(
        fm_hz=fm_hz, fs_hz=fs_hz, n_periods=n_periods,
        peak_abs=peak_abs, hwhm_hz=hwhm_hz, noise_std_time=noise_std_time,
    )

    bounds = [m_max_bounds] + [p_bounds] * n_phase_terms

    def objective(x):
        m_max = float(x[0])
        p = tuple(float(v) for v in x[1:])
        amp2, _ = _amp2_and_snr2_phase(m_max, p, **sim_kwargs)
        return -amp2

    result = differential_evolution(
        objective, bounds=bounds,
        seed=de_seed, maxiter=de_maxiter, popsize=de_popsize,
        tol=1e-4, polish=True, updating="deferred",
    )

    m_max_opt = float(result.x[0])
    p_opt = tuple(float(v) for v in result.x[1:])
    amp2_opt, snr2_opt = _amp2_and_snr2_phase(m_max_opt, p_opt, **sim_kwargs)

    (amp2_sine,), *_ = simulate_wms(
        amplitudes={1: sine_m1_baseline}, harmonics=(2,), **sim_kwargs,
    )
    amp2_sine = float(amp2_sine)

    return PhaseDistortedResult(
        m_max=m_max_opt,
        p_terms=p_opt,
        amp2=amp2_opt,
        snr2=snr2_opt,
        amp2_sine_baseline=amp2_sine,
        gain_vs_sine=amp2_opt / amp2_sine,
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
    )


def save_result_json(result, output_path, *, n_phase_terms=None):
    """Serialize a PhaseDistortedResult to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_phase_terms": n_phase_terms or len(result.p_terms),
        "m_max": result.m_max,
        "p_terms": list(result.p_terms),
        "amp2": result.amp2,
        "snr2": result.snr2,
        "amp2_sine_baseline": result.amp2_sine_baseline,
        "gain_vs_sine": result.gain_vs_sine,
        "success": result.success,
        "message": result.message,
        "nfev": result.nfev,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    best = optimize_phase_distorted_amp2(n_phase_terms=4)
    print("success:", best.success)
    print("message:", best.message)
    print("nfev:", best.nfev)
    print("m_max:", best.m_max)
    print("p_terms:", best.p_terms)
    print("2f amp:", best.amp2)
    print("2f SNR:", best.snr2)
    print("2f amp (sine baseline):", best.amp2_sine_baseline)
    print("gain_vs_sine:", best.gain_vs_sine)
