import re
from typing import List, Set


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)?")


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return TOKEN_PATTERN.findall(text.lower())


def unique_tokens(text: str) -> Set[str]:
    return set(tokenize(text))


def estimate_tokens(text: str) -> int:
    # Rough estimate suitable for runtime telemetry without extra tokenizer dependency.
    words = tokenize(text)
    if not words:
        return 0
    return max(1, int(len(words) * 1.35))
