"""Text preprocessing module for noisy, code-mixed social media text.

Maintains structural and semantic fidelity by preserving:
- Emojis (e.g. 🔥, 😍, 😡, ❤️)
- Punctuation & sentiment symbols (e.g. !, ?, ..., :), :()
- Character repetitions (e.g. sooooo, achhaaaaa)
- Hinglish code-mixed lexical tokens and hashtags
- Mentions and URLs handled safely without corrupting sentiment signals
"""

from dataclasses import dataclass, field
from typing import List, Optional
import re
import unicodedata
from nltk.tokenize import TweetTokenizer


@dataclass(frozen=True)
class PreprocessingResult:
    """Structured container for preprocessed text and tokenization output."""
    raw_text: str
    processed_text: str
    tokens: List[str] = field(default_factory=list)
    token_length: int = 0


class TextPreprocessor:
    """Conservative preprocessor tailored for code-mixed social media sentiment analysis.

    Designed to preserve all cues required for Multi-Dimensional Noise Quantification
    (Emoji Density, Repetition Ratio, Code-Mixing Intensity, Symbol Density).
    """

    # URL regex pattern
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

    # Whitespace normalization pattern
    WHITESPACE_PATTERN = re.compile(r"\s+")

    def __init__(
        self,
        remove_urls: bool = True,
        normalize_mentions: bool = False,
        preserve_case: bool = True,
    ) -> None:
        self.remove_urls = remove_urls
        self.normalize_mentions = normalize_mentions
        self.preserve_case = preserve_case
        # NLTK TweetTokenizer preserves emojis, hashtags, and social media morphology
        self.tokenizer = TweetTokenizer(
            preserve_case=preserve_case,
            reduce_len=False,  # DO NOT truncate character repetitions (essential for noise score R)
            strip_handles=False,
        )

    def clean_text(self, text: Optional[str]) -> str:
        """Conservative cleaning: normalize unicode, handle URLs/mentions, normalize whitespace."""
        if text is None:
            return ""

        if not isinstance(text, str):
            text = str(text)

        if not text.strip():
            return ""

        # Normalize unicode to NFC (canonical decomposition followed by canonical composition)
        cleaned = unicodedata.normalize("NFC", text)

        # Handle URLs safely: remove or replace with whitespace to keep token boundaries intact
        if self.remove_urls:
            cleaned = self.URL_PATTERN.sub(" ", cleaned)

        # Optionally normalize user handles (e.g. @username) without affecting #hashtags
        if self.normalize_mentions:
            cleaned = re.sub(r"@\w+", "@user", cleaned)

        # Normalize internal whitespace (convert tabs, multiple spaces, newlines into a single space)
        cleaned = self.WHITESPACE_PATTERN.sub(" ", cleaned).strip()

        return cleaned

    def tokenize(self, text: str) -> List[str]:
        """Tokenize preprocessed text while preserving emojis, symbols, and code-mixing."""
        if not text:
            return []
        raw_tokens = self.tokenizer.tokenize(text)
        tokens: List[str] = []
        for t in raw_tokens:
            if not t.strip():
                continue
            # Merge emoji variation selectors / tone modifiers with preceding token
            if t in ("\ufe0f", "\ufe0e") and tokens:
                tokens[-1] = tokens[-1] + t
            else:
                tokens.append(t)
        return tokens


    def preprocess(self, text: Optional[str]) -> PreprocessingResult:
        """Full conservative preprocessing pipeline returning structured result."""
        raw = "" if text is None else str(text)
        cleaned = self.clean_text(raw)
        tokens = self.tokenize(cleaned)
        token_len = len(tokens)

        return PreprocessingResult(
            raw_text=raw,
            processed_text=cleaned,
            tokens=tokens,
            token_length=token_len,
        )
