"""Unit tests for conservative text preprocessing."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from src.data.preprocessor import TextPreprocessor, PreprocessingResult


class TestTextPreprocessor(unittest.TestCase):
    """Test suite verifying conservative preprocessing on code-mixed and noisy social media text."""

    def setUp(self):
        self.preprocessor = TextPreprocessor()

    def test_clean_english(self):
        """Test 1: Standard clean English text."""
        raw = "The acting and direction were absolutely fantastic throughout."
        res = self.preprocessor.preprocess(raw)

        self.assertIsInstance(res, PreprocessingResult)
        self.assertEqual(res.raw_text, raw)
        self.assertEqual(res.processed_text, raw)
        self.assertIn("fantastic", res.tokens)
        self.assertEqual(res.token_length, len(res.tokens))
        self.assertGreater(res.token_length, 5)

    def test_hinglish_preservation(self):
        """Test 2: Hinglish code-mixed text preservation."""
        raw = "Ye movie bohot mast hai bhai, full paisa vasool!"
        res = self.preprocessor.preprocess(raw)

        self.assertEqual(res.processed_text, raw)
        # Hinglish tokens must not be translated, stemmed, or removed
        for word in ["Ye", "movie", "bohot", "mast", "hai", "bhai"]:
            self.assertIn(word, res.tokens)
        self.assertIn("paisa", res.tokens)
        self.assertIn("vasool", res.tokens)

    def test_emoji_heavy_text(self):
        """Test 3: Preservation of single and clustered emojis."""
        raw = "Loved it! 😍🔥❤️ Best performance ever 🎉👏"
        res = self.preprocessor.preprocess(raw)

        # Check emojis in processed text
        for emoji_char in ["😍", "🔥", "❤️", "🎉", "👏"]:
            self.assertIn(emoji_char, res.processed_text)
            self.assertIn(emoji_char, res.tokens)

    def test_repeated_characters(self):
        """Test 4: Preservation of elongated words and character repetition."""
        raw = "sooooo goooooddddd achhaaaaaa yaaaarrrr"
        res = self.preprocessor.preprocess(raw)

        # Character repetitions must NOT be truncated or normalized away
        self.assertIn("sooooo", res.tokens)
        self.assertIn("goooooddddd", res.tokens)
        self.assertIn("achhaaaaaa", res.tokens)
        self.assertIn("yaaaarrrr", res.tokens)
        self.assertEqual(res.processed_text, raw)

    def test_punctuation_heavy_text(self):
        """Test 5: Preservation of expressive punctuation and emoticons."""
        raw = "What an ending?!?!?! Seriously.... wow!!!! :D :("
        res = self.preprocessor.preprocess(raw)

        self.assertIn("?!?!?!", res.processed_text)
        self.assertIn("....", res.processed_text)
        self.assertIn("!!!!", res.processed_text)
        self.assertIn(":D", res.tokens)
        self.assertIn(":(", res.tokens)

    def test_mixed_language_and_social_entities(self):
        """Test 6: English + Hinglish + Hashtags + Mentions + URLs."""
        raw = "Check this review https://t.co/xyz123 @director Great direction lekin storyline thodi weak thi #SuperHit"
        res = self.preprocessor.preprocess(raw)

        # URL should be cleanly handled (removed without breaking adjacent tokens)
        self.assertNotIn("https://t.co/xyz123", res.processed_text)
        # Hashtags and code-mixed tokens preserved
        self.assertIn("#SuperHit", res.tokens)
        self.assertIn("lekin", res.tokens)
        self.assertIn("storyline", res.tokens)
        self.assertIn("thodi", res.tokens)
        self.assertIn("weak", res.tokens)
        self.assertIn("thi", res.tokens)

    def test_empty_and_whitespace_text(self):
        """Test 7: Empty, whitespace-only, and None text inputs."""
        empty_cases = ["", "   ", "\t\n  \r", None]

        for case in empty_cases:
            res = self.preprocessor.preprocess(case)
            self.assertEqual(res.processed_text, "")
            self.assertEqual(res.tokens, [])
            self.assertEqual(res.token_length, 0)


if __name__ == "__main__":
    unittest.main()
