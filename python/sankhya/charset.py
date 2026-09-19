"""Build a stable char vocab from a language pack's forms + digits + punctuation."""
from __future__ import annotations

import argparse
import json
import string
import unicodedata

from .langs import base as langbase

EXTRA_PUNCT = list(" .,-/+()%:'\"!?@#₹~")

# Indic decimal digits -> ASCII 0-9, 1:1, offsets preserved. Every block
# here is a contiguous run of ten code points in 0..9 order, so the map is
# a pure per-character substitution and every text offset is unchanged.
# Kept in sync with core._DIGIT_MAP and src/charset.ts INDIC_DIGIT_RE.
NATIVE_DIGIT_BLOCKS = {
    "deva": 0x0966,  # Devanagari U+0966 '०' .. U+096F '९'
    "gujr": 0x0AE6,  # Gujarati   U+0AE6 '૦' .. U+0AEF '૯'
}
_NATIVE_DIGIT_MAP = {
    chr(base + i): str(i) for base in NATIVE_DIGIT_BLOCKS.values() for i in range(10)
}


def normalize_text(s: str) -> str:
    """Shared normalization spec (identical to the JS runtime):
    1) Unicode NFC normalize
    2) Indic digits -> ASCII 0-9 (1:1, offsets preserved): Devanagari
       U+0966-U+096F and Gujarati U+0AE6-U+0AEF
    3) lowercase
    """
    s = unicodedata.normalize("NFC", s)
    s = "".join(_NATIVE_DIGIT_MAP.get(ch, ch) for ch in s)
    s = s.lower()
    return s


def build_charset(pack) -> list:
    chars = set()
    forms = pack.all_forms()
    for surface in forms.keys():
        chars.update(normalize_text(surface))
    for tmpl_list in getattr(pack, "templates", {}).values():
        for t in tmpl_list:
            chars.update(normalize_text(t))
    for w in getattr(pack, "filler_words", []) or []:
        chars.update(normalize_text(w))
    for digit in string.digits:
        chars.add(digit)
    chars.update(EXTRA_PUNCT)
    chars.update(string.ascii_lowercase)
    ordered = ["<pad>", "<unk>"] + sorted(chars)
    return ordered


def build_charset_multi(packs) -> list:
    """Union of build_charset() over several packs (used for multi-lang runs)."""
    chars = set()
    for pack in packs:
        chars.update(build_charset(pack))
    chars.discard("<pad>")
    chars.discard("<unk>")
    return ["<pad>", "<unk>"] + sorted(chars)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi_latn",
                    help="comma-separated pack ids (see sankhya.langs.base.KNOWN_PACKS)")
    ap.add_argument("--out", default="data/charset.json")
    args = ap.parse_args(argv)
    lang_ids = [x.strip() for x in args.lang.split(",") if x.strip()]
    packs = [langbase.get_pack(l) for l in lang_ids]
    vocab = build_charset_multi(packs) if len(packs) > 1 else build_charset(packs[0])
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(vocab)} chars to {args.out}")


if __name__ == "__main__":
    main()
