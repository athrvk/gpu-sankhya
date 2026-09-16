#!/usr/bin/env python3
"""Mine romanised Hindi number-word spellings from the Google Dakshina v1.0
Hindi romanization lexicon (hi.translit.sampled.{train,dev,test}.tsv).

Dakshina: https://github.com/google-research-datasets/dakshina
Licence: CC BY-SA 4.0 (attribution required for any redistributed derived
list — see docs/DATA_GRAMMAR.md).

For every Devanagari surface form used by hi_deva's cardinal (1-99), unit
and prefix tables (plus common inflections: plural unit forms, and
with/without-nukta variants of ज़/ज, ड़/ड, ढ़/ढ), this script collects every
attested romanization + its count from the three Dakshina TSVs, normalises
to lowercase, drops anything already present in hi_latn's lexicon, and
writes a JSON report:

    {class: {devanagari: [[roman, count], ...]}}

Usage:
    python mine_dakshina.py --lexicon-dir /path/to/dakshina/hi/lexicons \
        --out python/scripts/dakshina_variants.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sankhya.langs import hi_deva as HD  # noqa: E402
from sankhya.langs import hi_latn as HL  # noqa: E402

# nukta <-> no-nukta pairs to also try when looking up a Devanagari surface.
_NUKTA_PAIRS = [("ज़", "ज"), ("ड़", "ड"), ("ढ़", "ढ")]

# Devanagari plural suffixes for units (लाखों, करोड़ों, हज़ारों, ...) that
# should map to the same unit class as the singular/bare form.
_PLURAL_SUFFIXES = ["ों", "ो"]


def _nukta_variants(s: str) -> set:
    """Return {s} plus versions with each nukta pair swapped, both ways."""
    out = {s}
    for with_nukta, without_nukta in _NUKTA_PAIRS:
        new_out = set()
        for cand in out:
            new_out.add(cand)
            if with_nukta in cand:
                new_out.add(cand.replace(with_nukta, without_nukta))
            if without_nukta in cand:
                new_out.add(cand.replace(without_nukta, with_nukta))
        out = new_out
    return out


def build_lookup_keys():
    """Map class -> set of Devanagari surface forms to search for in Dakshina,
    built from hi_deva's cardinal/unit/prefix tables plus inflections."""
    keys = defaultdict(set)

    # cardinals 1..99
    for n in range(1, 100):
        cls = f"CARD_{n}"
        for form in HD.CARD_WORDS[n]:
            form = unicodedata.normalize("NFC", form)
            keys[cls].update(_nukta_variants(form))

    # units and prefixes
    for cls, forms in HD.LEXICON.items():
        if cls.startswith("CARD_"):
            continue
        for form in forms:
            form = unicodedata.normalize("NFC", form)
            variants = _nukta_variants(form)
            keys[cls].update(variants)
            # plural inflections (लाख -> लाखों, लाखो)
            for v in list(variants):
                for suf in _PLURAL_SUFFIXES:
                    keys[cls].add(v + suf)

    return keys


def load_tsvs(lexicon_dir: str):
    """Yield (devanagari, roman, count) rows from the three Dakshina TSVs."""
    files = [
        "hi.translit.sampled.train.tsv",
        "hi.translit.sampled.dev.tsv",
        "hi.translit.sampled.test.tsv",
    ]
    found_any = False
    for fname in files:
        path = os.path.join(lexicon_dir, fname)
        if not os.path.isfile(path):
            print(f"WARNING: missing {path}", file=sys.stderr)
            continue
        found_any = True
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                deva, roman, count = parts[0], parts[1], parts[2]
                deva = unicodedata.normalize("NFC", deva)
                roman = unicodedata.normalize("NFC", roman).strip().lower()
                try:
                    count = int(count)
                except ValueError:
                    continue
                yield deva, roman, count
    if not found_any:
        raise FileNotFoundError(
            f"None of the Dakshina lexicon TSVs were found under {lexicon_dir}"
        )


def existing_latn_forms():
    """Set of lowercase romanizations already present in hi_latn per class."""
    out = defaultdict(set)
    for cls, forms in HL.LEXICON.items():
        for f in forms:
            out[cls].add(f.lower())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lexicon-dir",
        default=os.path.join(
            "/tmp/claude-0/-home-user-gpu-sankhya/336d5702-9f4a-5916-ba5c-3720c33965e0",
            "scratchpad/dakshina/dakshina_dataset_v1.0/hi/lexicons",
        ),
        help="Directory containing hi.translit.sampled.{train,dev,test}.tsv",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "dakshina_variants.json"),
    )
    args = ap.parse_args(argv)

    lookup_keys = build_lookup_keys()
    # reverse index: devanagari surface -> class(es)
    deva_to_cls = defaultdict(set)
    for cls, forms in lookup_keys.items():
        for f in forms:
            deva_to_cls[f].add(cls)

    # class -> devanagari -> roman -> count
    collected = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    n_rows = 0
    n_matched = 0
    for deva, roman, count in load_tsvs(args.lexicon_dir):
        n_rows += 1
        if deva not in deva_to_cls:
            continue
        n_matched += 1
        for cls in deva_to_cls[deva]:
            collected[cls][deva][roman] += count

    existing = existing_latn_forms()

    report = {}
    total_new = 0
    for cls in sorted(collected.keys()):
        cls_out = {}
        for deva, romans in collected[cls].items():
            new_romans = []
            for roman, count in sorted(romans.items(), key=lambda kv: -kv[1]):
                if roman in existing.get(cls, set()):
                    continue
                if not roman:
                    continue
                new_romans.append([roman, count])
            if new_romans:
                cls_out[deva] = new_romans
                total_new += len(new_romans)
        if cls_out:
            report[cls] = cls_out

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Dakshina rows scanned: {n_rows}, matched a lookup key: {n_matched}")
    print(f"Report written to {args.out}: {total_new} new (class,deva,roman) variants "
          f"across {len(report)} classes")
    print()
    print(f"{'class':<12}{'devanagari':<14}{'top new romanizations (count)':<50}")
    for cls in sorted(report.keys()):
        for deva, romans in report[cls].items():
            top = ", ".join(f"{r}({c})" for r, c in romans[:6])
            print(f"{cls:<12}{deva:<14}{top}")


if __name__ == "__main__":
    main()
