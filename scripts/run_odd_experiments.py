#!/usr/bin/env python3
"""Run odd-harmonic optimization experiments and save outputs."""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harmonica.waveforms import OddHarmonicStrategy
from harmonica.objectives import SimObjective
from harmonica.optimizer import optimize, save_result_json
from harmonica.plotting import (
    animate_scan_on_line,
    plot_vs_bang_bang,
    plot_waveform_periods_and_amplitudes,
    plot_waveform_periods_nm,
)

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)


def main():
    objective = SimObjective()

    res_13 = optimize(OddHarmonicStrategy((1, 3)), objective,
                      de_maxiter=200, de_popsize=24, de_seed=0)
    res_13579 = optimize(OddHarmonicStrategy((1, 3, 5, 7, 9)), objective,
                         de_maxiter=500, de_popsize=30, de_seed=0)

    save_result_json(res_13, ROOT / "outputs" / "results" / "result_1f3f.json")
    save_result_json(res_13579, ROOT / "outputs" / "results" / "result_1f3f5f7f9f.json")

    # plotting still expects the old-style result with .amplitudes/.phases_rad
    # adapt via a thin wrapper
    plot_vs_bang_bang(res_13579, ROOT / "outputs" / "plots" / "waveform_vs_bang_bang.png")
    plot_waveform_periods_and_amplitudes(
        res_13579,
        output_path=ROOT / "outputs" / "plots" / "waveform_3periods_and_amps.png",
    )
    plot_waveform_periods_and_amplitudes(
        res_13579, compare_result=res_13,
        output_path=ROOT / "outputs" / "plots" / "waveform_compare_3periods_and_amps.png",
    )
    plot_waveform_periods_nm(
        res_13579,
        output_path=ROOT / "outputs" / "plots" / "waveform_3periods_nm.png",
    )
    animate_scan_on_line(
        res_13,
        output_path=ROOT / "outputs" / "animations" / "scan_on_line_1f3f.gif",
        n_periods=2, frames_per_period=120, fps=24,
        line_xlim_nm=(1650.0, 1652.0),
    )
    animate_scan_on_line(
        res_13579,
        output_path=ROOT / "outputs" / "animations" / "scan_on_line_1f3f5f7f9f.gif",
        n_periods=2, frames_per_period=120, fps=24,
        line_xlim_nm=(1650.0, 1652.0),
    )

    print("1f+3f gain:", res_13.gain_vs_sine)
    print("1f+3f+5f+7f+9f gain:", res_13579.gain_vs_sine)


if __name__ == "__main__":
    main()
