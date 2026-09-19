"""Export the language packs' surface-form lexicon to a JSON the JS runtime
can use for span verification (see docs in VERIFY contract / README).

    python -m sankhya.export_lexicon --out ../src/data/lexicon.json

Format (version 1):
    {"version": 1,
     "packs": {"hi_latn": {"forms": {surface_lower: CLASS, ...},
                           "range_words": ["to", "se", ...]}, ...}}
`forms` is LanguagePack.all_forms() (lexicon + symbol_units +
english_fraction_phrases + indefinite_plurals as "O"). `range_words` are the
alphabetic range connectors (stripped, lowercased).
"""
from __future__ import annotations

import argparse
import json
import unicodedata

from .langs import base  # noqa: F401
from .langs import hi_latn, hi_deva  # noqa: F401  (registers packs)


def _is_letters(s: str) -> bool:
    return bool(s) and all(c.isalpha() or unicodedata.category(c).startswith("M") for c in s)


def build() -> dict:
    packs = {}
    for pid, pack in sorted(base.registry.items()):
        forms = dict(sorted(pack.all_forms().items()))
        range_words = sorted({w.strip().lower() for w in pack.range_connectors if _is_letters(w.strip())})
        packs[pid] = {"forms": forms, "range_words": range_words}
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
