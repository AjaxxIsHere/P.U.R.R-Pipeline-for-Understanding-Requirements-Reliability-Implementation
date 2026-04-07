import re
from typing import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


DOMAIN_STOPWORDS = {
    "shall",
    "must",
    "system",
    "application",
    "ensure",
    "provide",
    "able",
    "user",
    "users",
    "app",
    "data",
    "network",
}
ALL_STOPWORDS = set(ENGLISH_STOP_WORDS).union(DOMAIN_STOPWORDS)


def _strip_numeric_bullets(text: str) -> str:
    return re.sub(r"^\s*(?:\d+[\.\)]\s*)+", "", text)


def _replace_numbers(text: str) -> str:
    return re.sub(r"\b\d+(?:\.\d+)?\b", "numtoken", text)


def _remove_non_alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text)


def clean_text_base(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _strip_numeric_bullets(text)
    text = _replace_numbers(text)
    text = _remove_non_alnum(text)
    tokens = [tok for tok in text.split() if tok and tok not in ALL_STOPWORDS]
    return " ".join(tokens)


def clean_text_ambiguity(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _strip_numeric_bullets(text)
    text = _replace_numbers(text)
    text = _remove_non_alnum(text)
    return " ".join(text.split())


def keep_min_words(texts: Iterable[str], min_words: int = 3) -> list[bool]:
    return [len(str(t).split()) >= min_words for t in texts]
