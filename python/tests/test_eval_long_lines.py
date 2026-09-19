"""Long gold lines: `eval_gold` must window, not truncate.

`run_json_weights` used to predict on `normalize_text(text)[:MAX_LEN]` and
then hand `decode_spans` the ORIGINAL text, so any line over 128 characters
made the decoder index past the end of the prediction and raise IndexError.
The hand-written gold is all short, so nothing caught it; 11-27% of the real
sentences `sankhya.wild` samples are longer (data_wild/REPORT.md failure
#10). Both runtimes now slide the same MAX_LEN window over the text
(charset.make_windows / src/charset.ts makeWindows).
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sankhya.charset import make_windows, normalize_text
from sankhya.train import MAX_LEN

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "..", "..", "models", "default", "sankhya.weights.int8.json")

# 200 characters, with a real amount well past the 128-char cut-off.
_PREFIX = ("yah ek lamba vakya hai jo sirf isliye likha gaya hai taki yeh model ki "
           "window se lamba ho jaye aur decoder ko poora text dekhna pade, theek hai na, "
           "to phir baat ye hai ki ab suniye ")
LONG_TEXT = _PREFIX + "salary 12 lakh hai."
LONG_SPAN_START = LONG_TEXT.index("12 lakh")
LONG_SPAN_END = LONG_SPAN_START + len("12 lakh")


def test_the_fixture_line_is_actually_past_the_window():
    assert len(LONG_TEXT) == 200
    assert LONG_SPAN_START > MAX_LEN, "the amount must sit past the first window"


def test_make_windows_matches_the_js_scheme():
    assert make_windows(10) == [(0, 10)]
    assert make_windows(128) == [(0, 128)]
    # stride = MAX_LEN - WINDOW_OVERLAP = 112
    assert make_windows(200) == [(0, 128), (112, 88)]
    for n in (129, 200, 500, 1000):
        wins = make_windows(n)
        assert all(length <= MAX_LEN for _o, length in wins)
        assert wins[0][0] == 0
        assert wins[-1][0] + wins[-1][1] == n, "windows must cover the whole text"


def test_normalisation_is_one_to_one_per_character_on_gold():
    """The offsets gold records against the raw text only carry over to the
    normalised text the predictions index into if normalisation never
    changes length. Checked over every shipped gold file."""
    names = ["gold.jsonl", "gold_deva.jsonl", "gold_mr.jsonl", "gold_gu.jsonl"] + [
        f"gold_wild_{l}.jsonl" for l in ("hi_latn", "hi_deva", "mr_deva", "gu_gujr")
    ]
    checked = 0
    for name in names:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                text = json.loads(line)["text"]
                assert len(normalize_text(text)) == len(text), repr(text)
                checked += 1
    assert checked > 0


@pytest.mark.skipif(not os.path.exists(WEIGHTS), reason="shipped int8 weights not found")
def test_a_200_char_gold_line_evaluates_instead_of_raising(tmp_path):
    from sankhya.eval_gold import run_eval

    gold = tmp_path / "gold_long.jsonl"
    gold.write_text(json.dumps({
        "text": LONG_TEXT,
        "lang": "hi_latn",
        "spans": [{"start": LONG_SPAN_START, "end": LONG_SPAN_END, "value": 1200000}],
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    # The regression this pins: this call used to raise IndexError.
    out = run_eval([str(gold)], weights_json=WEIGHTS, int8=True, quiet=True)
    combined = out["combined"]
    assert combined["examples"] == 1
    assert combined["spans"] == 1
    assert 0.0 <= combined["value_acc"] <= 1.0


@pytest.mark.skipif(not os.path.exists(WEIGHTS), reason="shipped int8 weights not found")
def test_a_long_line_is_predicted_over_its_whole_length():
    import json as _json

    import numpy as np  # noqa: F401  (np_infer needs it)

    from sankhya import np_infer
    from sankhya.eval_gold import predict_example
    from sankhya.train import build_char_to_id

    with open(WEIGHTS, encoding="utf-8") as f:
        obj = _json.load(f)
    weights = np_infer.load_weights_int8_json(obj)
    char_to_id = build_char_to_id(obj["charset"])
    unk = char_to_id.get("<unk>", 1)

    text, bio, cls, probs = predict_example(LONG_TEXT, weights, char_to_id, unk)
    assert len(text) == len(LONG_TEXT) == 200
    assert len(bio) == len(cls) == len(probs) == 200
    assert all(p is not None for p in probs), "every character must be covered"
