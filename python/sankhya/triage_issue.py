"""Triage a "wrong parse" report against the shipped weights.

Given the text from a wrong-parse report and what the reporter says the
correct answer should be, this prints what the currently shipped int8
weights (`models/default/sankhya.weights.int8.json`) actually return for
that text, says whether the report reproduces against those weights, and
-- when it does -- prints the exact JSONL line to append to the matching
`tests/gold_<lang>.jsonl` file.

Usage::

    python -m sankhya.triage_issue --text "sava lakh ka phone" \\
        --lang hi_latn --expect-value 125000

    python -m sankhya.triage_issue --text "paune do lakh" \\
        --lang hi_deva --expect-range 175000 175000

    python -m sankhya.triage_issue --text "lakhon log aaye" \\
        --lang hi_latn --negative
"""
from __future__ import annotations

import argparse
import json
import os

from . import core
from .charset import normalize_text
from .decode import decode_spans
from .eval_gold import _apply_bare_digits_gate, run_json_weights

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_WEIGHTS = os.path.join(REPO_ROOT, "models", "default", "sankhya.weights.int8.json")

# maps --lang to the gold file it belongs in, mirroring eval_gold's
# _GOLD_SUFFIX_ALIASES (inverted) and python/README.md's "Evaluate on gold".
LANG_TO_GOLD_FILE = {
    "hi_latn": "gold.jsonl",
    "hi_deva": "gold_deva.jsonl",
    "mr_deva": "gold_mr.jsonl",
    "gu_gujr": "gold_gu.jsonl",
}


def _decode_current(weights_path, text):
    """Run the shipped weights on `text` and return (decoded_spans, classes).

    `decoded_spans` is the gated decode_spans() output -- exactly what
    eval_gold/the JS runtime would score.
    """
    norm = normalize_text(text)
    preds, classes = run_json_weights(weights_path, [{"text": norm}], int8=True)
    _, bio_pred, cls_pred, bio_probs = preds[0]
    decoded = decode_spans(norm, bio_pred, cls_pred, bio_probs=bio_probs)
    decoded = _apply_bare_digits_gate(norm, decoded, classes)
    return norm, decoded, classes


def _span_results(text, decoded, classes):
    """Return a list of {start, end, value, range, span} for each decoded span."""
    out = []
    for d in decoded:
        toks = [(classes[cid], sub) for cid, sub in d["tokens"]]
        res = core.evaluate(toks)
        out.append({
            "start": d["start"],
            "end": d["end"],
            "span": text[d["start"]:d["end"]],
            "value": res.value,
            "range": list(res.range) if res.range else None,
        })
    return out


def triage(text, lang, expect_value=None, expect_range=None, negative=False,
           weights_path=DEFAULT_WEIGHTS):
    """Run the report's text through the shipped weights and decide whether
    it reproduces. Returns a dict with the current output, whether it
    reproduces, and (if it does) the gold line + target file to append it to.
    """
    norm, decoded, classes = _decode_current(weights_path, text)
    current_spans = _span_results(norm, decoded, classes)

    if negative:
        reproduces = len(current_spans) > 0
    elif expect_range is not None:
        lo, hi = expect_range
        reproduces = not any(
            sp["range"] == [lo, hi] for sp in current_spans
        )
    else:
        reproduces = not any(sp["value"] == expect_value for sp in current_spans)

    gold_line = None
    gold_file = LANG_TO_GOLD_FILE.get(lang)
    if reproduces and gold_file is not None:
        if negative:
            gold_spans = []
        else:
            # use the offsets of the (first) span the model actually found,
            # if any -- that's the best guess for where the true span is.
            # Otherwise fall back to the whole normalized text.
            if current_spans:
                start, end = current_spans[0]["start"], current_spans[0]["end"]
            else:
                start, end = 0, len(norm)
            span = {"start": start, "end": end}
            if expect_range is not None:
                span["value"] = expect_range[1]
                span["range"] = list(expect_range)
            else:
                span["value"] = expect_value
            gold_spans = [span]
        gold_line = json.dumps({"text": norm, "lang": lang, "spans": gold_spans}, ensure_ascii=False)

    return {
        "text": norm,
        "lang": lang,
        "current_spans": current_spans,
        "reproduces": reproduces,
        "gold_file": gold_file,
        "gold_line": gold_line,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True, help="the exact text from the report")
    ap.add_argument("--lang", required=True, choices=sorted(LANG_TO_GOLD_FILE), help="language pack")
    ap.add_argument("--expect-value", type=float, default=None, help="the correct numeric value")
    ap.add_argument("--expect-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                     help="the correct [low, high] range instead of a single value")
    ap.add_argument("--negative", action="store_true",
                     help="report claims this text contains no amount (false positive)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="path to int8 JSON weights to test against")
    args = ap.parse_args(argv)

    if not args.negative and args.expect_value is None and args.expect_range is None:
        ap.error("one of --expect-value, --expect-range, or --negative is required")

    result = triage(
        text=args.text,
        lang=args.lang,
        expect_value=args.expect_value,
        expect_range=tuple(args.expect_range) if args.expect_range else None,
        negative=args.negative,
        weights_path=args.weights,
    )

    print(f"text (normalized): {result['text']!r}")
    print(f"lang: {result['lang']}")
    print("current output (shipped int8 weights):")
    if result["current_spans"]:
        for sp in result["current_spans"]:
            print(f"  [{sp['start']}:{sp['end']}] {sp['span']!r} -> value={sp['value']} range={sp['range']}")
    else:
        print("  (no spans)")

    print()
    if result["reproduces"]:
        print("REPRODUCES: the shipped weights do not match the reported correct answer.")
        if result["gold_file"]:
            print(f"\nAppend this line to tests/{result['gold_file']}:")
            print(result["gold_line"])
            print(
                "\nThen run the gates:\n"
                "  cd python && python -m pytest tests/test_gates.py -q\n"
                "and, if coverage/value_acc regress, retrain "
                "(see python/README.md's 'Train' section)."
            )
    else:
        print("DOES NOT REPRODUCE: the shipped weights already agree with the reported correct answer.")

    return 0 if True else 1


if __name__ == "__main__":
    raise SystemExit(main())
