"""Number matching shared by verification and evaluation, so the two cannot disagree.

Spoken numbers normalize in more than one way: "twelve fifty" -> "1250", "twelve euros fifty"
-> "12" "50", "four point two percent" -> "4.2". A value matches a text if any spoken form of
the value appears among the text's normalized numbers.
"""

from __future__ import annotations

import re

from whisper_normalizer.english import EnglishTextNormalizer

_NORM = EnglishTextNormalizer()
_NUM = re.compile(r"\d+(?:\.\d+)?")
_REPEAT = re.compile(r"\b(\w+)(?:[\s,]+\1\b)+", re.IGNORECASE)  # "eight eight" -> "eight"
_HALF = re.compile(r"\b(\d+) and a half\b")


def numbers_in(text: str) -> list[str]:
    """Numbers in `text` after normalization ("four point two percent" -> ["4.2"]).

    Disfluent repeats are collapsed first ("twelve twelve and a half" would otherwise normalize
    to one merged number), and "12 and a half" becomes 12.5.
    """
    text = _REPEAT.sub(r"\1", text)
    text = _HALF.sub(lambda m: f"{m.group(1)}.5", text)
    normalized = _HALF.sub(lambda m: f"{m.group(1)}.5", _NORM(text))
    return _NUM.findall(normalized)


def _canon(n: str) -> str:
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n.lstrip("0") or "0"


def value_in(value: str, text_numbers: list[str] | set[str]) -> bool:
    """True if every number in `value` appears in `text_numbers` in some spoken form."""
    have = {_canon(n) for n in text_numbers}
    for n in numbers_in(value):
        forms = {_canon(n)}
        if "." in n:
            whole, frac = n.split(".", 1)
            forms.add(_canon(whole + frac))                 # "12.50" said "twelve fifty"
            if _canon(whole) in have and _canon(frac) in have:  # "twelve euros fifty"
                continue
        if not forms & have:
            return False
    return True
