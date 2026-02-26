## Harmonica

Optimizes modulation waveforms for wavelength modulation spectroscopy (WMS). Uses differential evolution to find odd-harmonic (1f+3f, up to 9f) and phase-distorted sine waveforms that maximize the 2f lock-in signal against arbitrary line profile.

Once you install uv, cd into the directory and run:

```
uv sync
uv run python -m unittest -q tests/test_wms_1f3f.py
```

For running the different experiments: 
```
uv run python scripts/run_odd_experiments.py
uv run python scripts/run_phase_experiment.py --terms 6
uv run python scripts/run_phase_experiment.py --terms 9 --maxiter 400 --popsize 30 --seed 42
```
