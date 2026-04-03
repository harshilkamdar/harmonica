import unittest
import sys
from pathlib import Path

import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harmonica.wms import simulate_wms


class TestWms1f3f(unittest.TestCase):

    def test_1f3f_beats_sine_at_fixed_reference_settings(self):
        (amp2_sine,), *_ = simulate_wms(amplitudes={1: 2.25}, harmonics=(2,))
        (amp2_13,), *_ = simulate_wms(
            amplitudes={1: 4.375, 3: 1.88125}, phases={3: jnp.pi}, harmonics=(2,),
        )
        gain = float(amp2_13 / amp2_sine)
        self.assertGreater(gain, 1.55, f"unexpected 1f+3f gain: {gain}")
        self.assertLess(gain, 1.65, f"unexpected 1f+3f gain: {gain}")

    def test_1f9f_optimized_coefficients(self):
        """Verify 2f gain from DE-optimized 1f+3f+5f+7f+9f coefficients (seed=0, maxiter=500, popsize=30)."""
        (amp2_sine,), *_ = simulate_wms(amplitudes={1: 2.25}, harmonics=(2,))
        (amp2_19,), *_ = simulate_wms(
            amplitudes={1: 7.999, 3: 4.111, 5: 0.098, 7: 1.250, 9: 0.586},
            phases={3: -3.136, 5: 0.591, 7: -0.049, 9: 3.086},
            harmonics=(2,),
        )
        gain = float(amp2_19 / amp2_sine)
        self.assertGreater(gain, 1.70, f"unexpected 1f-9f gain: {gain}")
        self.assertLess(gain, 1.80, f"unexpected 1f-9f gain: {gain}")


if __name__ == "__main__":
    unittest.main()
