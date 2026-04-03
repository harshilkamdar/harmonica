#!/usr/bin/env python3
"""Run 1f+3f optimization on HITRAN vs QPC line profile and compare."""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harmonica.waveforms import OddHarmonicStrategy
from harmonica.objectives import SimObjective
from harmonica.optimizer import optimize
from harmonica.wms import load_profile

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)

DE_KWARGS = dict(de_maxiter=200, de_popsize=24, de_seed=0)


def main():
    strategy = OddHarmonicStrategy((1, 3))

    # --- HITRAN ---
    print("=" * 60)
    print("1f+3f optimization on HITRAN line")
    print("=" * 60)
    res_hitran = optimize(strategy, SimObjective(), **DE_KWARGS)
    print(f"  gain: {res_hitran.gain_vs_sine:.4f}x")
    print(f"  amps: {res_hitran.amplitudes}")

    # --- QPC experimental line ---
    qpc_x, qpc_y, qpc_center_nm, qpc_center_hz, qpc_hwhm = load_profile(
        ROOT / "data" / "qpc_interp.npz"
    )
    qpc_profile = (qpc_x, qpc_y, qpc_center_hz)

    print()
    print("=" * 60)
    print("1f+3f optimization on QPC experimental line")
    print(f"  QPC center: {qpc_center_nm:.4f} nm  HWHM: {qpc_hwhm:.0f} Hz")
    print("=" * 60)
    res_qpc = optimize(strategy,
                       SimObjective(hwhm_hz=qpc_hwhm, line_profile=qpc_profile),
                       **DE_KWARGS)
    print(f"  gain: {res_qpc.gain_vs_sine:.4f}x")
    print(f"  amps: {res_qpc.amplitudes}")

    # --- Summary ---
    print()
    print("=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'':30s} {'HITRAN':>12s} {'QPC':>12s}")
    print(f"{'2f amp (sine baseline)':30s} {res_hitran.amp2_sine_baseline:12.6f} {res_qpc.amp2_sine_baseline:12.6f}")
    print(f"{'2f amp (1f+3f optimized)':30s} {res_hitran.amp2:12.6f} {res_qpc.amp2:12.6f}")
    print(f"{'2f gain vs sine':30s} {res_hitran.gain_vs_sine:12.4f} {res_qpc.gain_vs_sine:12.4f}")
    print(f"{'m1':30s} {res_hitran.amplitudes[1]:12.4f} {res_qpc.amplitudes[1]:12.4f}")
    print(f"{'m3':30s} {res_hitran.amplitudes[3]:12.4f} {res_qpc.amplitudes[3]:12.4f}")
    print(f"{'phi3 [rad]':30s} {res_hitran.phases_rad[3]:12.4f} {res_qpc.phases_rad[3]:12.4f}")


if __name__ == "__main__":
    main()
