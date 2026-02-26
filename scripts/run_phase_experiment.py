#!/usr/bin/env python3
"""Run phase-distorted sine optimization."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harmonica.optimize_phase import optimize_phase_distorted_amp2, save_result_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terms", type=int, default=6, help="Number of phase-warp terms (p_k).")
    parser.add_argument("--maxiter", type=int, default=260, help="DE max iterations.")
    parser.add_argument("--popsize", type=int, default=22, help="DE population size multiplier.")
    parser.add_argument("--seed", type=int, default=0, help="DE random seed.")
    args = parser.parse_args()

    res = optimize_phase_distorted_amp2(
        n_phase_terms=args.terms,
        de_maxiter=args.maxiter,
        de_popsize=args.popsize,
        de_seed=args.seed,
    )

    out_path = ROOT / "outputs" / "results" / f"result_phase_k{args.terms}.json"
    save_result_json(res, out_path)

    print("success:", res.success)
    print("message:", res.message)
    print("gain_vs_sine:", res.gain_vs_sine)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
