"""Latin-script noise transforms N2-N12 from DATA_GRAMMAR §4.

Reusable by any Latin-script language pack. N1 (variant pick) is the pack's
own job since it needs the lexicon; this module handles the character-level
noise applied on top of a chosen surface form.
"""
from __future__ import annotations
import random

VOWEL_SWAPS = [("aa", "a"), ("a", "aa"), ("ee", "i"), ("i", "ee"), ("u", "oo"), ("oo", "u"), ("e", "ai"), ("o", "au")]
ASPIRATION = [("dh", "d"), ("bh", "b"), ("kh", "k"), ("th", "t"), ("chh", "ch")]
CONSONANT_ALT = [("z", "j"), ("j", "z"), ("w", "v"), ("v", "w"), ("f", "ph"), ("sh", "s"), ("rh", "dh")]


def _swap_one(word: str, pairs, rng: random.Random) -> str:
    candidates = []
    for a, b in pairs:
        idx = word.find(a)
        if idx != -1:
            candidates.append((idx, a, b))
    if not candidates:
        return word
    idx, a, b = rng.choice(candidates)
    return word[:idx] + b + word[idx + len(a):]


def n2_vowel(word, rng):
    return _swap_one(word, VOWEL_SWAPS, rng)


def n3_aspiration(word, rng):
    return _swap_one(word, ASPIRATION, rng)


def n4_consonant(word, rng):
    return _swap_one(word, CONSONANT_ALT, rng)


def n5_final_vowel(word, rng):
    if not word:
        return word
    if word[-1] in "aei" and rng.random() < 0.5:
        return word[:-1]
    return word + rng.choice("aei")


def n6_doubling(word, rng):
    # find a doubled letter to de-double, or a letter to double
    doubled = [i for i in range(len(word) - 1) if word[i] == word[i + 1]]
    if doubled and rng.random() < 0.5:
        i = rng.choice(doubled)
        return word[:i] + word[i + 1:]
    if len(word) > 2:
        i = rng.randrange(1, len(word) - 1)
        return word[:i] + word[i] + word[i:]
    return word


def n7_typo(word, rng):
    if len(word) < 4:
        return word
    op = rng.choice(["del", "ins", "transpose", "adjacent"])
    i = rng.randrange(len(word))
    if op == "del":
        return word[:i] + word[i + 1:]
    if op == "ins":
        return word[:i] + rng.choice("aeiou") + word[i:]
    if op == "transpose" and i < len(word) - 1:
        lst = list(word)
        lst[i], lst[i + 1] = lst[i + 1], lst[i]
        return "".join(lst)
    if op == "adjacent":
        keys = "qwertyuiopasdfghjklzxcvbnm"
        ki = keys.find(word[i])
        if ki != -1:
            nb = keys[max(0, ki - 1):ki + 2].replace(word[i], "")
            if nb:
                return word[:i] + rng.choice(nb) + word[i + 1:]
    return word


def n8_casing(word, rng):
    r = rng.random()
    if r < 0.70:
        return word.lower()
    if r < 0.85:
        return word.title()
    if r < 0.90:
        return word.upper()
    return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in word)


def n11_elongation(word, rng):
    if not word:
        return word
    n = rng.randint(2, 4)
    return word + word[-1] * (n - 1)


def n12_plural(word, rng):
    if word.endswith("s"):
        return word
    return word + "s"


_STACKABLE = [n2_vowel, n3_aspiration, n4_consonant, n5_final_vowel, n6_doubling, n7_typo, n11_elongation]
_PROBS = {
    n2_vowel: 0.15, n3_aspiration: 0.08, n4_consonant: 0.10, n5_final_vowel: 0.08,
    n6_doubling: 0.05, n7_typo: 0.04, n11_elongation: 0.02,
}


def apply_noise(word: str, rng: random.Random, apply_casing: bool = True, apply_plural: bool = False) -> str:
    """Apply stacked (<=2) char-level noise ops, then N12 plural last, then casing.

    N5 (final vowel add/drop) and N11 (elongation) both touch the final
    letter(s), so if N12 has already appended the plural "s" they must not
    fire afterwards on top of it (and N12 itself must come after them, not
    before, so it appends to the real word ending, not a noised one).
    """
    out = word
    applied = 0
    plural_fired = apply_plural and rng.random() < 0.15
    # a word already ending in a plural "s" (N12, or a curated "lakhs"/"lacs"
    # style variant) must not also get a final-vowel or elongation op stacked
    # on top of that "s" - that produces junk like "lAKHsA".
    already_plural = plural_fired or out.endswith("s")
    for fn in _STACKABLE:
        if applied >= 2:
            break
        if fn in (n5_final_vowel, n11_elongation) and already_plural:
            continue
        if rng.random() < _PROBS[fn]:
            out = fn(out, rng)
            applied += 1
    if plural_fired:
        out = n12_plural(out, rng)
    if apply_casing:
        out = n8_casing(out, rng)
    return out
