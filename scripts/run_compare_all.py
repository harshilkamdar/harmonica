#!/usr/bin/env python3
"""Compare waveforms: 1f+3f, 1f+3f+5f+7f+9f, arbitrary (interpolated), and analytic."""

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harmonica.analytic import eval_analytic_2f
from harmonica.waveforms import (
    OddHarmonicStrategy, InterpStrategy,
    odd_harmonic, interp_periodic, analytic_lorentzian,
)
from harmonica.objectives import SimObjective
from harmonica.optimizer import OptResult, optimize

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)


def load_interp_result(path):
    """Load an interp result from JSON into an OptResult."""
    data = json.loads(Path(path).read_text())
    return OptResult(
        strategy_name="InterpStrategy",
        params=data["params"] if "params" in data else {
            "control_values": tuple(data["control_values"]),
        },
        amp2=data["amp2"],
        snr2=data["snr2"],
        amp2_sine_baseline=data["amp2_sine_baseline"],
        gain_vs_sine=data["gain_vs_sine"],
        success=data["success"],
        message=data["message"],
        nfev=data["nfev"],
    )


def main():
    results_dir = ROOT / "outputs" / "results"
    plots_dir = ROOT / "outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    objective = SimObjective()

    # --- 1f+3f ---
    print("=== 1f+3f ===")
    res_13 = optimize(OddHarmonicStrategy((1, 3)), objective,
                      de_maxiter=200, de_popsize=24, de_seed=0)
    print(f"  gain: {res_13.gain_vs_sine:.4f}x")

    # --- 1f+3f+5f+7f+9f ---
    print("=== 1f+3f+5f+7f+9f ===")
    res_13579 = optimize(OddHarmonicStrategy((1, 3, 5, 7, 9)), objective,
                         de_maxiter=500, de_popsize=30, de_seed=0)
    print(f"  gain: {res_13579.gain_vs_sine:.4f}x")

    # --- arbitrary (load from file or run) ---
    interp_path = results_dir / "result_interp_n64.json"
    if not interp_path.exists():
        interp_path = results_dir / "result_interp_n128.json"
    if not interp_path.exists():
        print("No interp result found, running optimization...")
        res_interp = optimize(InterpStrategy(64), objective,
                              de_maxiter=500, de_popsize=40, de_seed=0)
        n_points = 64
    else:
        print(f"=== arbitrary (from {interp_path.name}) ===")
        res_interp = load_interp_result(interp_path)
        n_points = len(res_interp.params.get("control_values", []))
    print(f"  gain: {res_interp.gain_vs_sine:.4f}x  (n_points={n_points})")

    # --- analytic optimal (scan clip values) ---
    print("=== analytic optimal (Lorentzian-derived) ===")
    clip_values = [4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 50.0, None]
    best_analytic = None
    best_analytic_lorentz = None
    print("  clip  | HITRAN gain | Lorentzian gain")
    print("  ------+-------------+----------------")
    for mc in clip_values:
        res_hitran = eval_analytic_2f(m_clip=mc, use_lorentzian=False)
        res_lorentz = eval_analytic_2f(m_clip=mc, use_lorentzian=True)
        label = f"{mc:.0f}" if mc is not None else "inf"
        print(f"  {label:>5s} | {res_hitran['gain_vs_sine']:.4f}x      | {res_lorentz['gain_vs_sine']:.4f}x")
        if best_analytic is None or res_hitran["gain_vs_sine"] > best_analytic["gain_vs_sine"]:
            best_analytic = res_hitran
        if best_analytic_lorentz is None or res_lorentz["gain_vs_sine"] > best_analytic_lorentz["gain_vs_sine"]:
            best_analytic_lorentz = res_lorentz

    print(f"  best HITRAN gain:     {best_analytic['gain_vs_sine']:.4f}x (clip={best_analytic['m_clip']})")
    print(f"  best Lorentzian gain: {best_analytic_lorentz['gain_vs_sine']:.4f}x (clip={best_analytic_lorentz['m_clip']})")

    # --- build waveforms over one period ---
    n_pts = 2000
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)

    m_13 = odd_harmonic(theta, res_13.amplitudes, res_13.phases_rad)
    m_13579 = odd_harmonic(theta, res_13579.amplitudes, res_13579.phases_rad)
    cv = res_interp.params.get("control_values", ())
    m_interp = interp_periodic(theta, cv) if cv else np.zeros_like(theta)
    m_analytic_clip = analytic_lorentzian(theta, m_clip=best_analytic["m_clip"])

    # bang-bang dwell reference
    max_m = max(np.max(np.abs(m_13579)), np.max(np.abs(m_interp)))
    m_bang = np.where(np.cos(2.0 * theta) >= 0.0, 0.0, max_m * np.sign(np.sin(theta)))

    # ── PLOT 1: waveform comparison + gain bars ──
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), height_ratios=(2.5, 1.0))
    phase = theta / np.pi

    ax = axes[0]
    ax.plot(phase, m_bang, color="0.75", lw=1.2, ls="--", label="Bang-bang dwell", zorder=1)
    ax.plot(phase, m_13, lw=2.0, label=f"1f+3f  ({res_13.gain_vs_sine:.3f}x)", zorder=2)
    ax.plot(phase, m_13579, lw=2.0, label=f"1f\u20139f  ({res_13579.gain_vs_sine:.3f}x)", zorder=3)
    ax.plot(phase, m_interp, lw=2.0, label=f"Arbitrary n={n_points}  ({res_interp.gain_vs_sine:.3f}x)", zorder=4)
    ax.plot(phase, m_analytic_clip, lw=2.0, ls="-.",
            label=f"Analytic (clip={best_analytic['m_clip']})  ({best_analytic['gain_vs_sine']:.3f}x)", zorder=5)
    ax.set_xlabel(r"Phase $\theta / \pi$")
    ax.set_ylabel("x / HWHM")
    ax.set_xlim(0.0, 2.0)
    ax.set_title("Waveform Comparison: All Approaches")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(alpha=0.25)

    ax2 = axes[1]
    labels = ["1f+3f", "1f\u20139f", f"Arbitrary\nn={n_points}", f"Analytic\nclip={best_analytic['m_clip']}"]
    gains = [res_13.gain_vs_sine, res_13579.gain_vs_sine, res_interp.gain_vs_sine, best_analytic["gain_vs_sine"]]
    colors = ["C0", "C1", "C2", "C3"]
    bars = ax2.bar(labels, gains, color=colors, width=0.5, edgecolor="k", linewidth=0.5)
    ax2.axhline(1.0, color="0.4", ls="--", lw=1.0, label="Sine baseline")
    ax2.set_ylabel("Gain vs Sine")
    ax2.set_title("2f Amplitude Gain")
    ax2.grid(alpha=0.25, axis="y")
    for bar, g in zip(bars, gains):
        ax2.text(bar.get_x() + bar.get_width() / 2, g + 0.01, f"{g:.3f}x",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylim(0.0, max(gains) * 1.15)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out1 = plots_dir / "compare_all_waveforms.png"
    plt.savefig(out1, dpi=180)
    plt.close(fig)
    print(f"\nPlot saved: {out1}")

    # ── PLOT 2: analytic waveform detail ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for mc, ls in [(None, "-"), (20.0, "--"), (10.0, "-."), (6.0, ":")]:
        m = analytic_lorentzian(theta, m_clip=mc)
        clabel = f"clip={mc:.0f}" if mc is not None else "unclipped"
        ax.plot(phase, m, lw=2.0, ls=ls, label=clabel)
    ax.set_xlabel(r"Phase $\theta / \pi$")
    ax.set_ylabel("m (x / HWHM)")
    ax.set_xlim(0.0, 2.0)
    ax.set_title("Analytic Optimal Waveform (Lorentzian-derived)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    clips = np.arange(2.0, 52.0, 1.0)
    gains_hitran = []
    gains_lorentz = []
    for mc in clips:
        rh = eval_analytic_2f(m_clip=float(mc), use_lorentzian=False)
        rl = eval_analytic_2f(m_clip=float(mc), use_lorentzian=True)
        gains_hitran.append(rh["gain_vs_sine"])
        gains_lorentz.append(rl["gain_vs_sine"])
    ax.plot(clips, gains_hitran, lw=2.0, label="Through HITRAN profile")
    ax.plot(clips, gains_lorentz, lw=2.0, label="Through pure Lorentzian")
    ax.axhline(res_13579.gain_vs_sine, color="C1", ls="--", lw=1.0, alpha=0.7,
               label=f"1f\u20139f optimized ({res_13579.gain_vs_sine:.3f}x)")
    ax.axhline(res_interp.gain_vs_sine, color="C2", ls="--", lw=1.0, alpha=0.7,
               label=f"Arbitrary n={n_points} ({res_interp.gain_vs_sine:.3f}x)")
    ax.set_xlabel("Clip value (max |m|)")
    ax.set_ylabel("Gain vs Sine")
    ax.set_title("Analytic Waveform: Gain vs Clip Level")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    out2 = plots_dir / "analytic_waveform_detail.png"
    plt.savefig(out2, dpi=180)
    plt.close(fig)
    print(f"Plot saved: {out2}")


if __name__ == "__main__":
    main()
