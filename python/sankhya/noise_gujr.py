"""Gujarati-script noise transforms for the gu_gujr language pack.

Same stacked char-level noise model as noise_deva.py, with the
transformations that Gujarati orthography and phone typing actually
produce. The families, all observed in the seed corpus and notes:

  * anusvara drop / chandrabindu swap -- નેવું/નેવુ, પંચાણું/પંચાણુ,
    ચોસઠ/ચોંસઠ, એંસી/અંસી. Much the commonest phone-typing variance.
  * long/short matra swap -- ઓગણીસ/ઓગણિસ, પિસ્તાલીસ/પીસ્તાલીસ.
  * ડ / ઢ confusion (adjacent retroflexes, adjacent on most keyboards).
  * ય / ઇ / ઈ in the glide and vowel-initial families.
  * doubled-consonant simplification across virama -- પચ્ચીસ/પચીસ,
    છવ્વીસ/છવીસ, અઠ્ઠાવીસ/અઠાવીસ.

Collision rejection against the pack's vocabulary happens in the caller
(generator._safe_word), exactly as for hi_deva/mr_deva.
"""
from __future__ import annotations
import random

# long/short matra swap pairs (bidirectional)
MATRA_PAIRS = [("િ", "ી"), ("ી", "િ"), ("ુ", "ૂ"), ("ૂ", "ુ")]

# adjacent-retroflex and glide/vowel confusions
LETTER_PAIRS = [("ઢ", "ડ"), ("ડ", "ઢ"), ("ય", "ઇ"), ("ઇ", "ઈ"), ("ઈ", "ઇ")]

VIRAMA = "્"
CHANDRABINDU = "ઁ"
ANUSVARA = "ં"

# Gujarati digits 0-9 (U+0AE6..U+0AEF)
_GUJR_DIGITS = "૦૧૨૩૪૫૬૭૮૯"


def to_gujr_digits(s: str) -> str:
    """Render an ASCII digit string using Gujarati digit glyphs."""
    return "".join(_GUJR_DIGITS[int(ch)] if ch.isdigit() else ch for ch in s)


def n_anusvara(word, rng):
    """Drop an anusvara -- the single most common Gujarati phone-typing
    variant (નેવું -> નેવુ, પંચાણું -> પંચાણુ).

    Deliberately one-directional. noise_deva swaps anusvara against
    chandrabindu, but modern Gujarati barely uses ઁ: no form in this pack
    contains one, so producing one would emit a character outside the
    charset built from the pack's own vocabulary. A chandrabindu that
    does turn up in real input is still normalised and handled as an
    unknown char, exactly like any other out-of-vocabulary mark.
    """
    if ANUSVARA in word:
        return word.replace(ANUSVARA, "", 1)
    if CHANDRABINDU in word:
        return word.replace(CHANDRABINDU, ANUSVARA, 1)
    return word


def n_matra(word, rng):
    candidates = [(a, b) for a, b in MATRA_PAIRS if a in word]
    if not candidates:
        return word
    a, b = rng.choice(candidates)
    idx = word.find(a)
    return word[:idx] + b + word[idx + len(a):]


def n_letter(word, rng):
    candidates = [(a, b) for a, b in LETTER_PAIRS if a in word]
    if not candidates:
        return word
    a, b = rng.choice(candidates)
    idx = word.find(a)
    return word[:idx] + b + word[idx + len(a):]


def n_conjunct_simplify(word, rng):
    """Drop a virama-joined doubled consonant: X + '્' + X -> X
    (પચ્ચીસ -> પચીસ, છવ્વીસ -> છવીસ, અઠ્ઠાવીસ -> અઠાવીસ)."""
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
    matras = "ાિીુૂૃૅેૈૉોૌંઁઃ્"
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


_STACKABLE = [
    (n_anusvara, 0.30),
    (n_conjunct_simplify, 0.14),
    (n_matra, 0.10),
    (n_letter, 0.06),
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
