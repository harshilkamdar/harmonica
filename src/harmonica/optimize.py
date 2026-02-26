"""Optimize odd-harmonic WMS waveforms (1f..9f) for maximum 2f amplitude."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .wms import DEFAULT_CENTER_HZ, DEFAULT_HWHM_HZ, simulate_wms

ALLOWED_ODD_HARMONICS = (1, 3, 5, 7, 9)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"


@dataclass
class OptimizationResult:
    harmonics: tuple
    amplitudes: dict
    phases_rad: dict
    amp2: float
    snr2: float
    amp2_sine_baseline: float
    gain_vs_sine: float
    success: bool
    message: str
    nfev: int


def _validate_harmonics(harmonics):
    """Validate and sort harmonics, ensuring they're from ALLOWED_ODD_HARMONICS."""
    h = tuple(sorted(set(harmonics)))
    if not h:
        raise ValueError("harmonics cannot be empty.")
    if any(x not in ALLOWED_ODD_HARMONICS for x in h):
        raise ValueError(f"harmonics must be chosen from {ALLOWED_ODD_HARMONICS}.")
    return h


def _vector_layout(harmonics):
    """Return [(harmonic, 'amp'|'phi'), ...] describing the optimization vector layout."""
    layout = []
    for harm in harmonics:
        layout.append((harm, "amp"))
        if harm != 1:
            layout.append((harm, "phi"))
    return layout


def _decode_params(x, layout):
    """Unpack optimization vector into (amplitudes, phases) dicts."""
    amplitudes = {}
    phases = {}
    for value, (harm, kind) in zip(x, layout, strict=True):
        if kind == "amp":
            amplitudes[harm] = float(value)
        else:
            phases[harm] = float(value)
    return amplitudes, phases


def _amp2_and_snr2(amplitudes, phases, **sim_kwargs):
    """Evaluate 2f amplitude and SNR for given harmonic parameters."""
    (amp2,), (snr2,), *_ = simulate_wms(
        amplitudes=amplitudes, phases=phases, harmonics=(2,), **sim_kwargs,
    )
    return float(amp2), float(snr2)


def optimize_harmonics_amp2(
    harmonics, *,
    method="differential_evolution",
    amplitude_bounds=(0.0, 8.0),
    phase_bounds=(-float(np.pi), float(np.pi)),
    initial_guess=None,
    sine_m1_baseline=2.25,
    fm_hz=20_000.0, fs_hz=2_000_000.0, n_periods=1,
    peak_abs=0.5, hwhm_hz=DEFAULT_HWHM_HZ, noise_std_time=1e-3,
    de_seed=0, de_maxiter=120, de_popsize=16,
):
    """Maximize 2f amplitude over harmonic amplitudes and phases via differential evolution."""
    harmonics = _validate_harmonics(harmonics)
    layout = _vector_layout(harmonics)

    sim_kwargs = dict(
        fm_hz=fm_hz, fs_hz=fs_hz, n_periods=n_periods,
        peak_abs=peak_abs, hwhm_hz=hwhm_hz, noise_std_time=noise_std_time,
    )

    bounds = []
    x0 = []
    for harm, kind in layout:
        if kind == "amp":
            bounds.append(amplitude_bounds)
            if initial_guess and harm in initial_guess:
                x0.append(float(initial_guess[harm][0]))
            else:
                x0.append(1.5 if harm == 1 else 0.25)
        else:
            bounds.append(phase_bounds)
            if initial_guess and harm in initial_guess:
                x0.append(float(initial_guess[harm][1]))
            else:
                x0.append(0.0)

    def objective(x):
        amplitudes, phases = _decode_params(x, layout)
        amp2, _ = _amp2_and_snr2(amplitudes, phases, **sim_kwargs)
        return -amp2

    if method == "differential_evolution":
        result = differential_evolution(
            objective, bounds=bounds,
            seed=de_seed, maxiter=de_maxiter, popsize=de_popsize,
            tol=1e-4, polish=True, updating="deferred",
        )
    else:
        result = minimize(
            objective, x0=np.asarray(x0, dtype=float),
            method=method, bounds=bounds,
            options={"maxiter": 300, "xtol": 1e-4, "ftol": 1e-8},
        )

    amplitudes, phases = _decode_params(np.asarray(result.x, dtype=float), layout)
    amp2_opt, snr2_opt = _amp2_and_snr2(amplitudes, phases, **sim_kwargs)

    (amp2_sine,), *_ = simulate_wms(
        amplitudes={1: sine_m1_baseline}, harmonics=(2,), **sim_kwargs,
    )
    amp2_sine = float(amp2_sine)

    return OptimizationResult(
        harmonics=harmonics,
        amplitudes=amplitudes,
        phases_rad=phases,
        amp2=amp2_opt,
        snr2=snr2_opt,
        amp2_sine_baseline=amp2_sine,
        gain_vs_sine=amp2_opt / amp2_sine,
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
    )


def save_result_json(result, output_path, *, hwhm_hz=DEFAULT_HWHM_HZ, center_hz=DEFAULT_CENTER_HZ):
    """Serialize an OptimizationResult to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "harmonics": list(result.harmonics),
        "amplitudes": {str(k): float(v) for k, v in result.amplitudes.items()},
        "phases_rad": {str(k): float(v) for k, v in result.phases_rad.items()},
        "amp2": float(result.amp2),
        "snr2": float(result.snr2),
        "amp2_sine_baseline": float(result.amp2_sine_baseline),
        "gain_vs_sine": float(result.gain_vs_sine),
        "success": bool(result.success),
        "message": result.message,
        "nfev": int(result.nfev),
        "hwhm_hz": float(hwhm_hz),
        "center_hz": float(center_hz),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def _format_result(result):
    """Format an OptimizationResult as a human-readable string."""
    parts = [f"m1={result.amplitudes.get(1, 0.0):.6f}"]
    for h in result.harmonics:
        if h == 1:
            continue
        parts.append(f"m{h}={result.amplitudes[h]:.6f}")
        parts.append(f"phi{h}_rad={result.phases_rad[h]:.6f}")
    coeffs = ", ".join(parts)
    lines = [
        f"harmonics: {result.harmonics}",
        f"success: {result.success}",
        f"message: {result.message}",
        coeffs,
        f"2f amp: {result.amp2}",
        f"2f SNR: {result.snr2}",
        f"2f amp (sine baseline): {result.amp2_sine_baseline}",
        f"2f gain vs sine: {result.gain_vs_sine}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    best = optimize_harmonics_amp2((1, 3, 5, 7, 9))
    print(_format_result(best))
