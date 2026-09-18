"""Multi-Dimensional Noise Quantification Module.

Extracts normalized noise features according to PROJECT_SPEC.md:
- E: Emoji Density in [0, 1]
- R: Character Repetition Ratio in [0, 1]
- C: Code-Mixing Intensity in [0, 1]
- S: Symbol Density in [0, 1]

Computes the Composite Noise Score:
N = 0.25 * E + 0.25 * R + 0.30 * C + 0.20 * S  where N in [0, 1]
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Set
import re
import string
import emoji

from configs.config import NoiseWeights, config
from src.data.preprocessor import TextPreprocessor

# Exact noise group bands specified by research requirements
NOISE_GROUP_LOW = "LOW"
NOISE_GROUP_MODERATE = "MODERATE"
NOISE_GROUP_HIGH = "HIGH"
NOISE_GROUP_EXTREME = "EXTREME"

ALL_NOISE_GROUPS = [
    NOISE_GROUP_LOW,
    NOISE_GROUP_MODERATE,
    NOISE_GROUP_HIGH,
    NOISE_GROUP_EXTREME,
]


def assign_noise_group(noise_score: float) -> str:
    """Map continuous noise score N in [0, 1] to one of four research bands.

    - LOW:       0.0 <= N <= 0.2
    - MODERATE:  0.2 <  N <= 0.5
    - HIGH:      0.5 <  N <= 0.8
    - EXTREME:   0.8 <  N <= 1.0
    """
    n = float(min(1.0, max(0.0, noise_score)))
    if n <= 0.20:
        return NOISE_GROUP_LOW
    elif n <= 0.50:
        return NOISE_GROUP_MODERATE
    elif n <= 0.80:
        return NOISE_GROUP_HIGH
    else:
        return NOISE_GROUP_EXTREME


# Unambiguous Romanized Hindi/Hinglish lexical markers
HINGLISH_MARKERS: Set[str] = {
    "ye", "yeh", "wo", "woh", "kya", "kyu", "kyun", "hai", "hain", "hoon",
    "tha", "thi", "mein", "mai", "bhai", "yaar", "mast", "bohot", "bahut",
    "achha", "achhi", "achhe", "accha", "acchi", "acche", "sach", "bilkul", "matlab",
    "lekin", "magar", "wala", "wali", "wale", "karo", "karna", "kiya", "gaya", "gayi",
    "raha", "rahi", "rahe", "hota", "hoti", "hote", "sabse", "apna", "apne", "apni",
    "aisa", "aisi", "aise", "waisa", "waisi", "waise", "nahi", "nahin", "dekha", "dekho",
    "paisa", "vasool", "bakwaas", "bakwas", "zabardast", "dhamaal", "jhakaas", "dhamaka",
    "chutiyapa", "ghanta", "thodi", "thoda", "thode", "kripya", "dhanyawad",
}


# Global cache for English vocabulary to ensure fast, deterministic code-mixing detection
_ENGLISH_VOCAB: Optional[Set[str]] = None



def get_english_vocabulary() -> Set[str]:

    """Lazy-load and cache the standard English vocabulary from NLTK."""
    global _ENGLISH_VOCAB
    if _ENGLISH_VOCAB is None:
        try:
            import nltk
            try:
                from nltk.corpus import words
                _ENGLISH_VOCAB = set(w.lower() for w in words.words())
            except LookupError:
                nltk.download("words", quiet=True)
                from nltk.corpus import words
                _ENGLISH_VOCAB = set(w.lower() for w in words.words())
        except Exception:
            # Fallback minimal English lexicon in case of offline environment issues
            _ENGLISH_VOCAB = {
                "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
                "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
                "this", "but", "his", "by", "from", "they", "we", "say", "her",
                "she", "or", "an", "will", "my", "one", "all", "would", "there",
                "their", "what", "so", "up", "out", "if", "about", "who", "get",
                "which", "go", "me", "when", "make", "can", "like", "time", "no",
                "just", "him", "know", "take", "people", "into", "year", "your",
                "good", "some", "could", "them", "see", "other", "than", "then",
                "now", "look", "only", "come", "its", "over", "think", "also",
                "back", "after", "use", "two", "how", "our", "work", "first",
                "well", "way", "even", "new", "want", "because", "any", "these",
                "give", "day", "most", "us", "great", "movie", "film", "bad",
                "acting", "story", "director", "actor", "best", "worst", "amazing",
            }
    return _ENGLISH_VOCAB


@dataclass(frozen=True)
class NoiseFeatures:
    """Container for intermediate and composite noise quantification scores."""
    emoji_density: float      # E in [0, 1]
    repetition_score: float   # R in [0, 1]
    code_mixing_ratio: float  # C in [0, 1]
    symbol_density: float     # S in [0, 1]
    noise_score: float        # N in [0, 1]

    def to_dict(self) -> Dict[str, float]:
        """Convert features to dictionary."""
        return {
            "emoji_density": self.emoji_density,
            "repetition_score": self.repetition_score,
            "code_mixing_ratio": self.code_mixing_ratio,
            "symbol_density": self.symbol_density,
            "noise_score": self.noise_score,
        }


# Regular expressions for noise feature extraction
REPETITION_PATTERN = re.compile(r"(\S)\1{2,}", re.IGNORECASE)
DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")
PUNCTUATION_SYMBOLS = set(string.punctuation + "–—…•·«»“”‘’¿¡₹$€£¥%‰")


def calculate_emoji_density(text: Optional[str], tokens: Optional[List[str]] = None) -> float:
    """Compute Emoji Density E in [0, 1].

    Formula: E = Nemoji / Ntokens
    Returns 0.0 for empty or zero-token inputs.
    """
    if text is None:
        return 0.0
    text_str = str(text).strip()
    if not text_str:
        return 0.0

    if tokens is None:
        tokens = TextPreprocessor().tokenize(text_str)

    total_tokens = len(tokens)
    if total_tokens == 0:
        return 0.0

    num_emojis = emoji.emoji_count(text_str)
    if num_emojis == 0:
        return 0.0

    density = num_emojis / total_tokens
    return min(1.0, max(0.0, float(density)))


def calculate_repetition_score(text: Optional[str], tokens: Optional[List[str]] = None) -> float:
    """Compute Character Repetition Ratio R in [0, 1].

    Formula: R = Nrepeat / Ntokens
    where Nrepeat is the number of characters repeated consecutively more than twice.
    Returns 0.0 for empty or zero-token inputs.
    """
    if text is None:
        return 0.0
    text_str = str(text).strip()
    if not text_str:
        return 0.0

    if tokens is None:
        tokens = TextPreprocessor().tokenize(text_str)

    total_tokens = len(tokens)
    if total_tokens == 0:
        return 0.0

    matches = [m.group(0) for m in REPETITION_PATTERN.finditer(text_str)]
    if not matches:
        return 0.0

    repeated_chars_count = sum(len(m) for m in matches)
    ratio = repeated_chars_count / total_tokens
    return min(1.0, max(0.0, float(ratio)))


def calculate_code_mixing_ratio(text: Optional[str], tokens: Optional[List[str]] = None) -> float:
    """Compute Code-Mixing Intensity C in [0, 1].

    Formula:
        C = min(Tokens_eng, Tokens_hin) / (max(Tokens_eng, Tokens_hin) + 1)

    Measures code-mixing between English and Hindi/Hinglish word tokens.
    Returns 0.0 for empty text, zero-token text, or monolingual text.
    """
    if text is None:
        return 0.0
    text_str = str(text).strip()
    if not text_str:
        return 0.0

    if tokens is None:
        tokens = TextPreprocessor().tokenize(text_str)

    # Filter for word tokens containing alphanumeric characters, excluding pure punctuation and pure emojis
    word_tokens = [
        t for t in tokens
        if not all(c in PUNCTUATION_SYMBOLS for c in t) and emoji.emoji_count(t) == 0 and any(c.isalnum() for c in t)
    ]

    if not word_tokens:
        return 0.0

    english_vocab = get_english_vocabulary()
    tokens_eng = 0
    tokens_hin = 0

    for token in word_tokens:
        clean_tok = token.lower().strip(string.punctuation)
        if not clean_tok:
            continue

        # Check Devanagari script presence or known Hinglish markers
        if DEVANAGARI_PATTERN.search(clean_tok) or clean_tok in HINGLISH_MARKERS:
            tokens_hin += 1
        elif clean_tok in english_vocab:
            tokens_eng += 1
        else:
            # Token is an out-of-vocabulary / transliterated Hinglish word
            tokens_hin += 1

    if tokens_eng == 0 and tokens_hin == 0:
        return 0.0

    c = min(tokens_eng, tokens_hin) / (max(tokens_eng, tokens_hin) + 1.0)
    return min(1.0, max(0.0, float(c)))


def calculate_symbol_density(text: Optional[str], tokens: Optional[List[str]] = None) -> float:
    """Compute Symbol Density S in [0, 1].

    Formula: S = Nsymbol / Ntokens
    where Nsymbol is the number of non-alphanumeric tokens excluding emojis.
    Returns 0.0 for empty or zero-token inputs.
    """
    if text is None:
        return 0.0
    text_str = str(text).strip()
    if not text_str:
        return 0.0

    if tokens is None:
        tokens = TextPreprocessor().tokenize(text_str)

    total_tokens = len(tokens)
    if total_tokens == 0:
        return 0.0

    # Count non-alphanumeric tokens excluding emojis
    symbol_count = sum(
        1 for t in tokens
        if not any(c.isalnum() for c in t) and emoji.emoji_count(t) == 0 and t.strip()
    )
    density = symbol_count / total_tokens
    return min(1.0, max(0.0, float(density)))


def calculate_noise_index(
    emoji_density: float,
    repetition_score: float,
    code_mixing_ratio: float,
    symbol_density: float,
    weights: Optional[NoiseWeights] = None,
) -> float:
    """Compute the Composite Noise Score N in [0, 1].

    Formula: N = 0.25 * E + 0.25 * R + 0.30 * C + 0.20 * S
    """
    w = weights or config.noise
    n = (
        w.w_emoji * emoji_density +
        w.w_repetition * repetition_score +
        w.w_code_mix * code_mixing_ratio +
        w.w_symbol * symbol_density
    )
    return min(1.0, max(0.0, float(n)))


class NoiseQuantifier:
    """Multi-Dimensional Noise Quantification Engine."""

    def __init__(self, weights: NoiseWeights = config.noise) -> None:
        self.weights = weights
        self.preprocessor = TextPreprocessor()

    def compute_emoji_density(self, text: str, tokens: Optional[List[str]] = None) -> float:
        return calculate_emoji_density(text, tokens)

    def compute_repetition_ratio(self, text: str, tokens: Optional[List[str]] = None) -> float:
        return calculate_repetition_score(text, tokens)

    def compute_codemix_intensity(self, text: str, tokens: Optional[List[str]] = None) -> float:
        return calculate_code_mixing_ratio(text, tokens)

    def compute_symbol_density(self, text: str, tokens: Optional[List[str]] = None) -> float:
        return calculate_symbol_density(text, tokens)

    def extract_features(self, text: Optional[str]) -> NoiseFeatures:
        """Extract all multi-dimensional noise features and compute composite noise score N."""
        prep_res = self.preprocessor.preprocess(text)
        cleaned_text = prep_res.processed_text
        tokens = prep_res.tokens

        e = calculate_emoji_density(cleaned_text, tokens)
        r = calculate_repetition_score(cleaned_text, tokens)
        c = calculate_code_mixing_ratio(cleaned_text, tokens)
        s = calculate_symbol_density(cleaned_text, tokens)
        n = calculate_noise_index(e, r, c, s, self.weights)

        return NoiseFeatures(
            emoji_density=e,
            repetition_score=r,
            code_mixing_ratio=c,
            symbol_density=s,
            noise_score=n,
        )
