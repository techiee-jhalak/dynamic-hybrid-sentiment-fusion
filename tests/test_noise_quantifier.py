"""Unit tests for Multi-Dimensional Noise Quantification."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from configs.config import NoiseWeights
from src.features.noise_quantifier import (
    NoiseQuantifier,
    NoiseFeatures,
    calculate_emoji_density,
    calculate_repetition_score,
    calculate_code_mixing_ratio,
    calculate_symbol_density,
    calculate_noise_index,
)


class TestNoiseQuantifier(unittest.TestCase):
    """Thorough unit test suite for noise features (E, R, C, S) and composite noise score N."""

    def setUp(self):
        self.quantifier = NoiseQuantifier()

    def test_noise_formula_exactness(self):
        """Verify N = 0.25E + 0.25R + 0.30C + 0.20S."""
        e = 0.40
        r = 0.60
        c = 0.80
        s = 0.50

        # Expected: 0.25*0.40 + 0.25*0.60 + 0.30*0.80 + 0.20*0.50
        # = 0.10 + 0.15 + 0.24 + 0.10 = 0.59
        n = calculate_noise_index(e, r, c, s)
        self.assertAlmostEqual(n, 0.59, places=5)

    def test_bounds_and_clamping(self):
        """All features and composite N must strictly reside within [0, 1]."""
        extreme_cases = [
            "",
            "   ",
            None,
            "Clean standard English sentence.",
            "😍🔥❤️🎉👏💯🙌✨",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "ye toh bohot hi zyada bakwaas hai bhai",
            "?!?!?!?!?!?!?!!!!!!!!!!!!!!!!!!",
            "Bhaaaaiiiiii kya movie thi yaar 😍🔥❤️ #SuperHit #Mast 100/100!!!!!! :D",
        ]

        for text in extreme_cases:
            feats = self.quantifier.extract_features(text)
            self.assertIsInstance(feats, NoiseFeatures)
            self.assertGreaterEqual(feats.emoji_density, 0.0)
            self.assertLessEqual(feats.emoji_density, 1.0)

            self.assertGreaterEqual(feats.repetition_score, 0.0)
            self.assertLessEqual(feats.repetition_score, 1.0)

            self.assertGreaterEqual(feats.code_mixing_ratio, 0.0)
            self.assertLessEqual(feats.code_mixing_ratio, 1.0)

            self.assertGreaterEqual(feats.symbol_density, 0.0)
            self.assertLessEqual(feats.symbol_density, 1.0)

            self.assertGreaterEqual(feats.noise_score, 0.0)
            self.assertLessEqual(feats.noise_score, 1.0)

    def test_empty_and_zero_token_handling(self):
        """Safe handling of empty inputs without crashing or division by zero."""
        for empty_val in ["", "   ", "\t\n", None]:
            feats = self.quantifier.extract_features(empty_val)
            self.assertEqual(feats.emoji_density, 0.0)
            self.assertEqual(feats.repetition_score, 0.0)
            self.assertEqual(feats.code_mixing_ratio, 0.0)
            self.assertEqual(feats.symbol_density, 0.0)
            self.assertEqual(feats.noise_score, 0.0)

    def test_emoji_density_calculation(self):
        """Test E calculation on clean vs emoji-rich text."""
        clean_text = "This movie is completely devoid of emojis."
        self.assertEqual(calculate_emoji_density(clean_text), 0.0)

        emoji_text = "Good movie 😍🔥"
        e = calculate_emoji_density(emoji_text)
        self.assertGreater(e, 0.0)
        self.assertLessEqual(e, 1.0)

        pure_emojis = "😍 🔥 ❤️"
        e_pure = calculate_emoji_density(pure_emojis)
        self.assertAlmostEqual(e_pure, 1.0, places=3)

    def test_repetition_score_calculation(self):
        """Test R calculation on clean vs elongated text."""
        clean_text = "Standard words with no elongation."
        self.assertEqual(calculate_repetition_score(clean_text), 0.0)

        elongated = "sooooo goooooddddd achhaaaaa"
        r = calculate_repetition_score(elongated)
        self.assertGreater(r, 0.4)
        self.assertLessEqual(r, 1.0)

        pure_repetition = "aaaaaaaaaa"
        r_pure = calculate_repetition_score(pure_repetition)
        self.assertAlmostEqual(r_pure, 1.0, places=3)

    def test_code_mixing_ratio_calculation(self):
        """Test C calculation on pure English vs Hinglish."""
        pure_english = "The cinematography and acting were absolutely superb."
        c_eng = calculate_code_mixing_ratio(pure_english)
        self.assertEqual(c_eng, 0.0)

        hinglish = "Ye movie bohot mast hai bhai sach me"
        c_hinglish = calculate_code_mixing_ratio(hinglish)
        self.assertGreater(c_hinglish, 0.7)

    def test_symbol_density_calculation(self):
        """Test S calculation on plain text vs symbol-dense text."""
        plain_text = "Hello world"
        self.assertEqual(calculate_symbol_density(plain_text), 0.0)

        symbolic = "What?!?!?! Wow!!!!! #Epic"
        s = calculate_symbol_density(symbolic)
        self.assertGreater(s, 0.3)

        pure_symbols = "!!!!????$$$$"
        s_pure = calculate_symbol_density(pure_symbols)
        self.assertAlmostEqual(s_pure, 1.0, places=3)

    def test_noise_features_container_and_dict(self):
        """Verify NoiseFeatures conversion to dictionary."""
        feats = self.quantifier.extract_features("Movie was goooood! 😍 #Hit")
        d = feats.to_dict()
        self.assertIn("emoji_density", d)
        self.assertIn("repetition_score", d)
        self.assertIn("code_mixing_ratio", d)
        self.assertIn("symbol_density", d)
        self.assertIn("noise_score", d)
        self.assertEqual(d["noise_score"], feats.noise_score)

    def test_determinism(self):
        """Multiple runs must produce identical floating point values."""
        text = "Kya mast film thi yaar! 🔥🔥 Bilkul recommended hai... 10/10!!!"
        f1 = self.quantifier.extract_features(text)
        f2 = self.quantifier.extract_features(text)

        self.assertEqual(f1.emoji_density, f2.emoji_density)
        self.assertEqual(f1.repetition_score, f2.repetition_score)
        self.assertEqual(f1.code_mixing_ratio, f2.code_mixing_ratio)
        self.assertEqual(f1.symbol_density, f2.symbol_density)
        self.assertEqual(f1.noise_score, f2.noise_score)


if __name__ == "__main__":
    unittest.main()
