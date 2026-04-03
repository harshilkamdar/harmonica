#!/usr/bin/env python3
"""Run arbitrary-waveform (interpolated control-point) optimization."""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harmonica.waveforms import InterpStrategy
from harmonica.objectives import SimObjective
from harmonica.optimizer import optimize, save_result_json

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bounds", type=float, default=8.0)
    args = parser.parse_args()

    res = optimize(
        InterpStrategy(args.points, value_bounds=(-args.bounds, args.bounds)),
        SimObjective(),
        de_maxiter=args.maxiter, de_popsize=args.popsize, de_seed=args.seed,
    )

    out_path = ROOT / "outputs" / "results" / f"result_interp_n{args.points}.json"
    save_result_json(res, out_path)

    print(f"n_points:      {res.params['n_points'] if 'n_points' in res.params else args.points}")
    print(f"gain vs sine:  {res.gain_vs_sine:.4f}x")
    print(f"saved:         {out_path}")


if __name__ == "__main__":
    main()
