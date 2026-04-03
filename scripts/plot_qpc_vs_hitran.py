#!/usr/bin/env python3
"""Plot experimental QPC line scan vs HITRAN reference, both peak-normalized."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harmonica.io import load_qpc
from harmonica.wms import get_hitran_profile


def main():
    wl_qpc, abs_qpc = load_qpc()
    abs_qpc_norm = abs_qpc / abs_qpc.max()

    wl_hitran, y_hitran, *_ = get_hitran_profile()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(wl_qpc, abs_qpc_norm, label="QPC measurement", lw=1.8)
    ax.plot(wl_hitran, y_hitran, label="HITRAN", lw=1.8, alpha=0.85)
    ax.set_xlim(1650.8, 1651.2)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Normalized absorbance")
    ax.set_title("QPC Methane Line Scan vs HITRAN")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out = ROOT / "outputs" / "plots" / "qpc_vs_hitran.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
