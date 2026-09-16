"""Devanagari-script noise transforms for the hi_deva language pack.

Mirrors the intent of noise_latn.py's stacked char-level noise, but the
transformations are specific to Devanagari orthography (nukta, matras,
chandrabindu/anusvara, conjunct simplification). Collision rejection against
the pack's vocabulary happens in the caller (generator._safe_word), same as
for hi_latn.
"""
from __future__ import annotations
import random

# nukta drop/add pairs (bidirectional)
NUKTA_PAIRS = [("ड़", "ड"), ("ढ़", "ढ"), ("ज़", "ज"), ("फ़", "फ"), ("क़", "क")]

# long/short matra swap pairs (bidirectional)
MATRA_PAIRS = [("ि", "ी"), ("ी", "ि"), ("ु", "ू"), ("ू", "ु")]

VIRAMA = "्"
CHANDRABINDU = "ँ"
ANUSVARA = "ं"

# Devanagari digits 0-9
_DEVA_DIGITS = "०१२३४५६७८९"


def to_deva_digits(s: str) -> str:
    """Render an ASCII digit string using Devanagari digit glyphs."""
    out = []
    for ch in s:
        if ch.isdigit():
            out.append(_DEVA_DIGITS[int(ch)])
        else:
            out.append(ch)
    return "".join(out)


def n_nukta(word, rng):
    candidates = []
    for a, b in NUKTA_PAIRS:
        if a in word:
            candidates.append((a, b))
        if b in word and (b, a) not in candidates:
            candidates.append((b, a))
    if not candidates:
        return word
    a, b = rng.choice(candidates)
    idx = word.find(a)
    return word[:idx] + b + word[idx + len(a):]


def n_chandrabindu(word, rng):
    if CHANDRABINDU in word:
        return word.replace(CHANDRABINDU, ANUSVARA, 1)
    if ANUSVARA in word:
        return word.replace(ANUSVARA, CHANDRABINDU, 1)
    return word


def n_matra(word, rng):
    candidates = [(a, b) for a, b in MATRA_PAIRS if a in word]
    if not candidates:
        return word
    a, b = rng.choice(candidates)
    idx = word.find(a)
    return word[:idx] + b + word[idx + len(a):]


def n_conjunct_simplify(word, rng):
    """Drop a virama-joined doubled consonant: X + '्' + X -> X
    (e.g. अट्ठाईस -> अठाईस, इक्कीस -> इकीस)."""
    for i in range(len(word) - 2):
        if word[i + 1] == VIRAMA and word[i + 2] == word[i]:
            return word[:i + 1] + word[i + 3:]
    return word


def n_typo(word, rng):
    """Matra deletion / adjacent transposition, only for words with >= 4
    code points."""
    if len(word) < 4:
        return word
    op = rng.choice(["del_matra", "transpose"])
    matras = "ािीुूेैोौंँः़्"
    if op == "del_matra":
        idxs = [i for i, c in enumerate(word) if c in matras]
        if idxs:
            i = rng.choice(idxs)
            return word[:i] + word[i + 1:]
        return word
    i = rng.randrange(len(word) - 1)
    lst = list(word)
    lst[i], lst[i + 1] = lst[i + 1], lst[i]
    return "".join(lst)


def n_join_hyphen(word, rng):
    """Not a per-word op; handled at the phrase level by the generator when
    joining prefix+unit or number+unit tokens (see hi_deva usage)."""
    return word


_STACKABLE = [
    (n_nukta, 0.25),
    (n_chandrabindu, 0.30),
    (n_matra, 0.08),
    (n_conjunct_simplify, 0.10),
    (n_typo, 0.04),
]


def apply_noise(word: str, rng: random.Random) -> str:
    out = word
    applied = 0
    for fn, p in _STACKABLE:
        if applied >= 2:
            break
        if rng.random() < p:
            new = fn(out, rng)
            if new != out:
                applied += 1
            out = new
    return out
