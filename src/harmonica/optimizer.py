"""Unified optimizer: strategy + objective → result."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time

import numpy as np
from scipy.optimize import differential_evolution

logger = logging.getLogger(__name__)


@dataclass
class OptResult:
    strategy_name: str
    params: dict            # decoded params (amplitudes/phases, m_max/p_terms, etc.)
    amp2: float
    snr2: float
    amp2_sine_baseline: float
    gain_vs_sine: float
    success: bool
    message: str
    nfev: int
    fitness_history: list = field(default_factory=list)  # [(gen, best_amp2), ...]

    # convenience accessors for odd-harmonic results (used by plotting)
    @property
    def amplitudes(self):
        return self.params.get("amplitudes", {})

    @property
    def phases_rad(self):
        return self.params.get("phases", {})

    @property
    def harmonics(self):
        return tuple(sorted(self.amplitudes.keys()))


def optimize(strategy, objective, *,
             sine_m1_baseline=2.25,
             de_seed=0, de_maxiter=200, de_popsize=20,
             on_generation=None):
    """Maximize 2f amplitude using differential evolution.

    Args:
        strategy: has .bounds, .decode(x) -> dict, .generate(theta, **params) -> m(θ)
        objective: callable, takes m(θ) array, returns amp2 float
        on_generation: optional callback(gen, best_amp2, best_params, best_waveform)
    """
    theta = objective.theta

    logger.info("starting optimization: %s, ndim=%d, maxiter=%d, popsize=%d",
                type(strategy).__name__, len(strategy.bounds), de_maxiter, de_popsize)

    t0 = time.perf_counter()
    best_neg = [float("inf")]
    best_params = [None]
    eval_count = [0]
    gen_count = [0]
    history = []

    def cost(x):
        params = strategy.decode(x)
        m = strategy.generate(theta, **params)
        amp2 = objective(m)
        eval_count[0] += 1
        neg = -amp2
        if neg < best_neg[0]:
            best_neg[0] = neg
            best_params[0] = params
        return neg

    def callback(xk, convergence):
        gen_count[0] += 1
        amp2 = -best_neg[0]
        elapsed = time.perf_counter() - t0
        history.append((gen_count[0], amp2))
        logger.info("gen %d | amp2=%.6f | conv=%.4e | %.1fs | nfev=%d",
                     gen_count[0], amp2, convergence, elapsed, eval_count[0])
        if on_generation:
            m = strategy.generate(theta, **best_params[0])
            on_generation(gen_count[0], amp2, best_params[0], m)

    result = differential_evolution(
        cost, bounds=strategy.bounds,
        seed=de_seed, maxiter=de_maxiter, popsize=de_popsize,
        tol=1e-4, polish=True, updating="deferred",
        callback=callback,
    )

    elapsed = time.perf_counter() - t0
    logger.info("done in %.1fs | nfev=%d | success=%s", elapsed, result.nfev, result.success)

    # final evaluation
    final_params = strategy.decode(result.x)
    m_final = strategy.generate(theta, **final_params)
    amp2 = objective(m_final)

    # sine baseline
    from .waveforms import odd_harmonic
    m_sine = odd_harmonic(theta, {1: sine_m1_baseline}, {})
    amp2_sine = objective(m_sine)

    sigma_amp = 2.0 * objective.noise_std_time / np.sqrt(len(theta))
    snr2 = amp2 / sigma_amp

    return OptResult(
        strategy_name=type(strategy).__name__,
        params=final_params,
        amp2=amp2,
        snr2=snr2,
        amp2_sine_baseline=amp2_sine,
        gain_vs_sine=amp2 / amp2_sine,
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
        fitness_history=history,
    )


def save_result_json(result, output_path):
    """Save an OptResult to JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": result.strategy_name,
        "params": result.params,
        "amp2": result.amp2,
        "snr2": result.snr2,
        "amp2_sine_baseline": result.amp2_sine_baseline,
        "gain_vs_sine": result.gain_vs_sine,
        "success": result.success,
        "message": result.message,
        "nfev": result.nfev,
        "fitness_history": result.fitness_history,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
