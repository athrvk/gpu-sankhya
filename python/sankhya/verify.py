"""Span verification: is every token of a decoded span independently
justified by the exported lexicon?

A span is **verified** when each of its (class, text) tokens is explained by
the lexicon or by a purely structural rule (separator / digits / punctuation /
range marker). The value of a verified span is then a pure function of the
lexicon plus the arithmetic core, both unit-tested -- so strict mode
("the right number or nothing") can drop everything else.

This mirrors `src/verify.ts` 1:1; see the verified-tier contract.

    from sankhya.verify import verify_tokens
    verify_tokens([("PFX_SAADHE", "sava"), ("SEP", " "), ("UNIT_LAKH", "lakh")])
"""
from __future__ import annotations

import json
import os
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = ["load_lexicon", "verify_tokens", "union_forms", "union_range_words"]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEXICON_PATH = os.path.join(_REPO_ROOT, "src", "data", "lexicon.json")

# structural range markers, in addition to the packs' alphabetic range words
RANGE_SYMBOLS = {"-", "–", "—", "/"}

_DEVA_DIGITS = set("०१२३४५६७८९")

# module-level caches of the union maps
_LEXICON: Optional[dict] = None
_FORMS: Optional[Dict[str, Set[str]]] = None
_RANGE_WORDS: Optional[Set[str]] = None


def load_lexicon(path: str = None) -> dict:
    """Load (and cache) the exported lexicon JSON.

    Reads `src/data/lexicon.json` relative to the repo; if that file is
    missing, builds the same object in-process from the registered language
    packs via `export_lexicon.build()`.
    """
    global _LEXICON, _FORMS, _RANGE_WORDS
    if path is None and _LEXICON is not None:
        return _LEXICON
    p = path or LEXICON_PATH
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except FileNotFoundError:
        from .export_lexicon import build
        obj = build()
    if path is None:
        _LEXICON = obj
        _FORMS = None
        _RANGE_WORDS = None
    return obj


def _build_union(lex: dict):
    forms: Dict[str, Set[str]] = {}
    range_words: Set[str] = set()
    for pack in lex.get("packs", {}).values():
        for surface, cls in pack.get("forms", {}).items():
            forms.setdefault(_norm(surface), set()).add(cls)
        for w in pack.get("range_words", []):
            range_words.add(_norm(w))
    return forms, range_words


def union_forms(lex: dict = None) -> Dict[str, Set[str]]:
    """surface(lower, NFC) -> set of classes, across every pack."""
    global _FORMS, _RANGE_WORDS
    if lex is not None:
        return _build_union(lex)[0]
    if _FORMS is None:
        _FORMS, _RANGE_WORDS = _build_union(load_lexicon())
    return _FORMS


def union_range_words(lex: dict = None) -> Set[str]:
    global _FORMS, _RANGE_WORDS
    if lex is not None:
        return _build_union(lex)[1]
    if _RANGE_WORDS is None:
        _FORMS, _RANGE_WORDS = _build_union(load_lexicon())
    return _RANGE_WORDS


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).lower()


def _is_digit_char(c: str) -> bool:
    return ("0" <= c <= "9") or (c in _DEVA_DIGITS)


def _verify_token(cls: str, text: str, forms, range_words) -> bool:
    if cls == "SEP":
        return len(text) > 0 and text.strip() == ""
    if cls == "DOT":
        return text == "."
    if cls == "COMMA":
        return text == ","
    if cls == "RANGE":
        t = text.strip()
        if t in RANGE_SYMBOLS:
            return True
        return _norm(t) in range_words
    if cls == "DIGITS":
        return len(text) > 0 and all(_is_digit_char(c) for c in text)
    if cls == "O":
        return "O" in forms.get(_norm(text), ())
    # PFX_* / CARD_* / UNIT_*
    return cls in forms.get(_norm(text), ())


def _is_content(cls: str) -> bool:
    return cls == "DIGITS" or cls.startswith("PFX_") or cls.startswith("CARD_") or cls.startswith("UNIT_")


def verify_tokens(tokens: Sequence[Tuple[str, str]], lex: dict = None) -> bool:
    """True when every token of the span is justified by the lexicon.

    `tokens` are the span's (class, text) pairs exactly as passed to
    `core.evaluate`. `lex` overrides the cached lexicon (tests).
    """
    if not tokens:
        return False
    if lex is None:
        forms, range_words = union_forms(), union_range_words()
    else:
        forms, range_words = _build_union(lex)
    has_content = False
    for cls, text in tokens:
        if not _verify_token(cls, text, forms, range_words):
            return False
        if _is_content(cls):
            has_content = True
    return has_content
