"""Export the language packs' surface-form lexicon to a JSON the JS runtime
can use for span verification (see docs in VERIFY contract / README).

    python -m sankhya.export_lexicon --out ../src/data/lexicon.json

Format (version 1):
    {"version": 1,
     "packs": {"hi_latn": {"forms": {surface_lower: CLASS, ...},
                           "range_words": ["to", "se", ...],
                           "bound_forms": {surface_lower: {"cls": CLASS,
                                                           "before": [...]}}},
               ...}}
`forms` is LanguagePack.all_forms() (lexicon + symbol_units +
english_fraction_phrases + indefinite_plurals as "O"). `range_words` are the
alphabetic range connectors (stripped, lowercased).

`word_suffixes` / `word_oblique_endings` (optional, absent for packs
that declare none) are the pack's case endings and the oblique stems that
may sit under them (Marathi "लाखांचं" = लाख + oblique "ां" + ending "चं").
Both verifiers strip ONE declared ending from a PFX_*/CARD_*/UNIT_* token
whose exact surface is unknown, and accept the token when the remaining
head is a lexicon form of the token's own class -- see verify.py.

`ambiguous_forms` (optional) are lexicon surfaces that are also ordinary
words in that language ("so", "sath", "arab", "अरब"); R12 drops a span in
which such a surface stands with no independently justified number word
beside it. `blocked_surfaces` (optional) are surfaces that must never
verify or decode as a number word at all ("हजारे", "अरबी", "સવાઈ") -- R14.
Both are SURFACE lists, not classes, and both are exported so the JS
runtime sees exactly what the packs declare.

`bound_forms` (optional, absent for packs that have none) holds surfaces
that are ONLY valid immediately before one of the listed unit surfaces --
Gujarati CARD_2 "બ", which exists in બસો (200) and nowhere else. They are
kept out of `forms` on purpose: a verifier that merged them in would
happily verify a standalone "બ" as 2. Both verifiers therefore accept a
bound form only when the very next token's text is one of its `before`
surfaces (see verify.py / src/verify.ts).
"""
from __future__ import annotations

import argparse
import json
import unicodedata

from .langs import base  # noqa: F401
from .langs.base import load_all

load_all()  # registers every known pack


def _is_letters(s: str) -> bool:
    return bool(s) and all(c.isalpha() or unicodedata.category(c).startswith("M") for c in s)


def build() -> dict:
    packs = {}
    for pid, pack in sorted(base.load_all().items()):
        forms = dict(sorted(pack.all_forms().items()))
        range_words = sorted({w.strip().lower() for w in pack.range_connectors if _is_letters(w.strip())})
        entry = {"forms": forms, "range_words": range_words}
        bound = {
            surface.lower(): {
                "cls": spec["cls"],
                "before": sorted(u.lower() for u in spec.get("before", ())),
            }
            for surface, spec in sorted(pack.bound_number_forms.items())
        }
        if bound:
            entry["bound_forms"] = bound
        suffixes = sorted({s.lower() for s in (pack.word_suffixes or [])})
        if suffixes:
            entry["word_suffixes"] = suffixes
        obliques = sorted({o.lower() for o in (pack.word_oblique_endings or [])})
        if obliques:
            entry["word_oblique_endings"] = obliques
        ambiguous = sorted({unicodedata.normalize("NFC", a).lower()
                            for a in (pack.ambiguous_forms or [])})
        if ambiguous:
            entry["ambiguous_forms"] = ambiguous
        blocked = sorted({unicodedata.normalize("NFC", b).lower()
                          for b in (pack.blocked_surfaces or [])})
        if blocked:
            entry["blocked_surfaces"] = blocked
        packs[pid] = entry
    return {"version": 1, "packs": packs}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sankhya.export_lexicon")
    ap.add_argument("--out", default="../src/data/lexicon.json")
    args = ap.parse_args(argv)
    obj = build()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=0, sort_keys=True)
        f.write("\n")
    n = sum(len(p["forms"]) for p in obj["packs"].values())
    print(f"wrote {args.out}: {len(obj['packs'])} packs, {n} forms")


if __name__ == "__main__":
    main()
