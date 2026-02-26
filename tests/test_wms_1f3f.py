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


if __name__ == "__main__":
    unittest.main()
