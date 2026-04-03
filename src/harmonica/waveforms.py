"""Waveform strategies: params → normalized modulation array m(θ).

Each strategy defines:
  - bounds: list of (lo, hi) for each parameter
  - decode(x) → dict of named params
  - generate(theta, **params) → m(θ) array
"""

import numpy as np
from scipy.interpolate import CubicSpline


# ── Waveform generation functions ──────────────────────────────────────

def odd_harmonic(theta, amplitudes, phases):
    """Sum of odd-harmonic sines: m(θ) = Σ_h m_h sin(h θ + φ_h)."""
    m = np.zeros_like(theta)
    for h, a in amplitudes.items():
        m += a * np.sin(h * theta + phases.get(h, 0.0))
    return m


def phase_distorted(theta, m_max, p_terms):
    """Phase-warped sine: m(θ) = m_max sin(θ + Σ_k p_k sin(kθ))."""
    warp = np.zeros_like(theta)
    for k, pk in enumerate(p_terms, start=1):
        warp += pk * np.sin(k * theta)
    return m_max * np.sin(theta + warp)


def interp_periodic(theta, control_values):
    """Periodic cubic-spline through N uniformly spaced control points."""
    n = len(control_values)
    knots = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    vals = np.asarray(control_values, dtype=np.float64)
    knots = np.append(knots, 2.0 * np.pi)
    vals = np.append(vals, vals[0])
    cs = CubicSpline(knots, vals, bc_type="periodic", extrapolate=True)
    return cs(np.asarray(theta) % (2.0 * np.pi))


def analytic_lorentzian(theta, m_clip=None):
    """Analytically optimal waveform for a pure Lorentzian line."""
    s = np.sin(2.0 * theta)
    inner = np.where(np.abs(1.0 + s) < 1e-12, 1e12, 2.0 / (1.0 + s) - 1.0)
    inner = np.clip(inner, 0.0, None)
    m = np.sqrt(inner) * np.sign(np.cos(theta + np.pi / 4.0))
    if m_clip is not None:
        m = np.clip(m, -m_clip, m_clip)
    return m


def fit_profile_gaussians(profile_x, profile_y, n_gaussians=10, n_tries=10):
    """Fit a line profile as a sum of Gaussians. Returns (params, fitted_fn, rms)."""
    from scipy.optimize import curve_fit

    peak_idx = np.argmax(profile_y)
    xc = profile_x - profile_x[peak_idx]
    mask = np.abs(xc) < 12
    xc_fit, yf_fit = xc[mask], profile_y[mask]

    def model(x, *params):
        y = np.zeros_like(x)
        for i in range(len(params) // 3):
            y += params[3*i] * np.exp(-0.5 * ((x - params[3*i+1]) / params[3*i+2])**2)
        return y

    best_rms, best_popt = np.inf, None
    for seed in range(n_tries):
        rng = np.random.RandomState(seed)
        p0 = []
        for _ in range(n_gaussians):
            p0 += [rng.uniform(0.05, 1.0), rng.uniform(-1.5, 1.5), rng.uniform(0.3, 6.0)]
        try:
            popt, _ = curve_fit(model, xc_fit, yf_fit, p0=p0, maxfev=50000,
                                bounds=([0, -10, 0.05]*n_gaussians, [10, 10, 20]*n_gaussians))
            rms = np.sqrt(np.mean((yf_fit - model(xc_fit, *popt))**2))
            if rms < best_rms:
                best_rms, best_popt = rms, popt
        except RuntimeError:
            pass

    return best_popt, lambda x: model(x, *best_popt), best_rms


def profile_inversion(theta, profile_x, profile_y, kappa=1.0, m_clip=200.0,
                      fitted_fn=None, n_gaussians=10, n_inv_points=50000):
    """Numerically inverted waveform for an arbitrary line profile.

    Constructs g(t) such that S(g(t)) follows a tunable target shape.
    kappa → 0: sinusoidal (spectrally pure), kappa → ∞: square wave (max |c2|).
    """
    if fitted_fn is None:
        _, fitted_fn, _ = fit_profile_gaussians(profile_x, profile_y, n_gaussians)

    x_half = np.linspace(0, m_clip, n_inv_points)
    s_half = np.clip(fitted_fn(x_half), 0, None)
    s_mono = np.minimum.accumulate(s_half) - np.arange(n_inv_points) * 1e-15
    s_flip, x_flip = s_mono[::-1], x_half[::-1]
    s_inv = CubicSpline(s_flip, x_flip, extrapolate=True)

    if kappa < 0.01:
        target = (1.0 + np.sin(2.0 * theta)) / 2.0
    else:
        target = (1.0 + np.tanh(kappa * np.sin(2.0 * theta)) / np.tanh(kappa)) / 2.0

    g_mag = s_inv(np.clip(target, s_flip[0] + 1e-10, s_flip[-1] - 1e-10))
    return np.clip(g_mag, 0, m_clip) * np.sign(np.cos(theta + np.pi / 4.0))


# ── Strategy wrappers ──────────────────────────────────────────────────

ALLOWED_ODD_HARMONICS = (1, 3, 5, 7, 9)


class OddHarmonicStrategy:
    """Optimize amplitudes + phases of odd harmonics (1f..9f)."""

    def __init__(self, harmonics, amplitude_bounds=(0.0, 8.0),
                 phase_bounds=(-np.pi, np.pi)):
        h = tuple(sorted(set(harmonics)))
        if not h or any(x not in ALLOWED_ODD_HARMONICS for x in h):
            raise ValueError(f"harmonics must be from {ALLOWED_ODD_HARMONICS}")
        self.harmonics = h
        self._layout = []
        for harm in h:
            self._layout.append((harm, "amp"))
            if harm != 1:
                self._layout.append((harm, "phi"))
        self.bounds = []
        for _, kind in self._layout:
            self.bounds.append(amplitude_bounds if kind == "amp" else phase_bounds)

    def decode(self, x):
        amps, phases = {}, {}
        for val, (harm, kind) in zip(x, self._layout):
            if kind == "amp":
                amps[harm] = float(val)
            else:
                phases[harm] = float(val)
        return dict(amplitudes=amps, phases=phases)

    def generate(self, theta, amplitudes, phases):
        return odd_harmonic(theta, amplitudes, phases)


class PhaseDistortedStrategy:
    """Optimize m_max and phase-warp coefficients p_k."""

    def __init__(self, n_terms, m_max_bounds=(0.0, 8.0), p_bounds=(-1.5, 1.5)):
        if n_terms < 1:
            raise ValueError("n_terms must be >= 1")
        self.n_terms = n_terms
        self.bounds = [m_max_bounds] + [p_bounds] * n_terms

    def decode(self, x):
        return dict(m_max=float(x[0]), p_terms=tuple(float(v) for v in x[1:]))

    def generate(self, theta, m_max, p_terms):
        return phase_distorted(theta, m_max, p_terms)


class InterpStrategy:
    """Optimize N free control-point values for a periodic cubic spline."""

    def __init__(self, n_points=64, value_bounds=(-8.0, 8.0)):
        if n_points < 4:
            raise ValueError("n_points must be >= 4")
        self.n_points = n_points
        self.bounds = [value_bounds] * n_points

    def decode(self, x):
        return dict(control_values=tuple(float(v) for v in x))

    def generate(self, theta, control_values):
        return interp_periodic(theta, control_values)
