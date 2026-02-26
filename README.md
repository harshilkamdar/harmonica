## Harmonica

Repository layout:

- `src/harmonica/`: core simulation + optimization modules
- `scripts/`: runnable experiment entrypoints
- `data/`: static input data (`hitran.npz`)
- `outputs/plots/`: generated plot images
- `outputs/results/`: saved optimization result JSONs
- `outputs/animations/`: generated GIFs

Run experiments:

- Odd-harmonic experiments (1f+3f and 1f+3f+5f+7f+9f):
  - `python scripts/run_odd_experiments.py`
- Phase-distorted sine (Option 5B):
  - `python scripts/run_phase_experiment.py --terms 6 --maxiter 260 --popsize 22 --seed 0`
