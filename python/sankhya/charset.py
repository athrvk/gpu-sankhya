"""Build a stable char vocab from a language pack's forms + digits + punctuation."""
from __future__ import annotations

import argparse
import json
import string

from .langs import base as langbase

EXTRA_PUNCT = list(" .,-/+()%:'\"!?@#₹~")


def build_charset(pack) -> list:
    chars = set()
    forms = pack.all_forms()
    for surface in forms.keys():
        chars.update(surface.lower())
    for digit in string.digits:
        chars.add(digit)
    chars.update(EXTRA_PUNCT)
    chars.update(string.ascii_lowercase)
    ordered = ["<pad>", "<unk>"] + sorted(chars)
    return ordered


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi_latn")
    ap.add_argument("--out", default="data/charset.json")
    args = ap.parse_args(argv)
    pack = langbase.get_pack(args.lang)
    vocab = build_charset(pack)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(vocab)} chars to {args.out}")


if __name__ == "__main__":
    main()
