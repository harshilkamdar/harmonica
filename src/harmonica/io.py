"""Load and calibrate experimental QPC spectrometer data."""

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_qpc(path=None):
    """Load QPC multi-pass cell scan, returning (wavelength_nm, absorbance).

    Applies basic current-to-wavelength calibration and removes
    the power variation envelope before computing absorbance.
    """
    if path is None:
        path = PROJECT_ROOT / "data" / "QPCmulti (1).npy"
    qpc = np.load(path)

    x = np.flip(qpc[:, 1])  # laser current
    y = np.flip(qpc[:, 0])  # photodiode

    wavelength_nm = x * 4.8 + 1648.2
    baseline = x * 0.48 + 0.159 - 0.6 * (x - 0.35) ** 2
    absorbance = -np.log(y / baseline)

    return wavelength_nm, absorbance


def save_qpc_on_hitran_grid(output_path=None):
    """Interpolate QPC absorbance onto the HITRAN wavelength grid and save as .npz.

    Returns (x_nm, y_norm) — the peak-normalized profile on the HITRAN grid.
    """
    from .wms import get_hitran_profile

    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "qpc_interp.npz"

    wl_qpc, abs_qpc = load_qpc()
    wl_hitran, _, *_ = get_hitran_profile()

    y_interp = np.interp(wl_hitran, wl_qpc, abs_qpc, left=0.0, right=0.0)
    y_interp = np.clip(y_interp, 0.0, None)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, x=wl_hitran, y=y_interp)
    return out, wl_hitran, y_interp
