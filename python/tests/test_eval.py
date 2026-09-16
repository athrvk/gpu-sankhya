"""Plain-assert tests for sankhya.eval_gold's categorize_span and
sankhya.eval_matrix's matrix-winner selection, runnable via pytest or directly."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sankhya.eval_gold import categorize_span
from sankhya.eval_matrix import select_matrix_winner
from sankhya.langs import base as langs_base

pack = langs_base.get_pack("hi_latn")
pack_deva = langs_base.get_pack("hi_deva")


def test_digits_category():
    text = "kharcha 50000 tha"
    span = {"start": 8, "end": 13, "value": 50000}
    cats = categorize_span(text, span, pack)
    assert "digits" in cats
    assert "words" not in cats


def test_words_category():
    text = "bhai sava lakh mein ho jayega kya"
    span = {"start": 5, "end": 14, "value": 125000}
    cats = categorize_span(text, span, pack)
    assert "words" in cats
    assert "digits" not in cats


def test_prefix_category():
    text = "bhai sava lakh mein ho jayega kya"
    span = {"start": 5, "end": 14, "value": 125000}
    cats = categorize_span(text, span, pack)
    assert "prefix" in cats


def test_range_category():
    text = "bees se pachees hazaar tak"
    span = {"start": 0, "end": 26, "value": 20000, "range": [20000, 25000]}
    cats = categorize_span(text, span, pack)
    assert "range" in cats


def test_mixed_script_category():
    text = "usne paanch लाख maange"
    start = text.index("paanch")
    end = text.index("maange") - 1
    span = {"start": start, "end": end, "value": 500000}
    cats = categorize_span(text, span, pack)
    assert "mixed_script" in cats


def test_symbol_unit_category():
    # find an actual symbol form from the pack lexicon so this stays valid
    # even if the symbol table changes.
    sym = None
    for forms in pack.symbol_units.values():
        if forms:
            sym = forms[0]
            break
    assert sym, "hi_latn pack has no symbol_units to test against"
    text = f"kharcha 5{sym} tha"
    start = text.index(f"5{sym}")
    span = {"start": start, "end": start + 1 + len(sym), "value": 500000}
    cats = categorize_span(text, span, pack)
    assert "symbol_unit" in cats


def test_negatives_false_positive_counting():
    # simulate the fp-rate bookkeeping used in eval_gold: an example with
    # zero gold spans counts as a negative; it's a false positive iff any
    # span was predicted.
    negatives = [
        {"spans": [], "pred_spans": []},
        {"spans": [], "pred_spans": [(0, 3)]},
        {"spans": [], "pred_spans": []},
    ]
    neg_total = sum(1 for ex in negatives if not ex["spans"])
    neg_fp = sum(1 for ex in negatives if not ex["spans"] and ex["pred_spans"])
    assert neg_total == 3
    assert neg_fp == 1
    assert abs(neg_fp / neg_total - (1 / 3)) < 1e-9


def test_matrix_winner_selects_best_mean_config():
    entries = [
        {"arch": "v1", "channels": 32, "seed": 0, "val_value_acc": 0.90, "gold_int8_value_acc": 0.80},
        {"arch": "v1", "channels": 32, "seed": 1, "val_value_acc": 0.95, "gold_int8_value_acc": 0.82},
        {"arch": "v2", "channels": 32, "seed": 0, "val_value_acc": 0.85, "gold_int8_value_acc": 0.90},
        {"arch": "v2", "channels": 32, "seed": 1, "val_value_acc": 0.99, "gold_int8_value_acc": 0.92},
    ]
    winner, label = select_matrix_winner(entries)
    # v2:32 has mean gold_int8_value_acc = 0.91 > v1:32's 0.81
    assert label == "v2:32"
    # within v2:32, seed 1 has the better val_value_acc
    assert winner["seed"] == 1


def test_matrix_winner_tie_breaks_to_first():
    entries = [
        {"arch": "v1", "channels": 32, "seed": 0, "val_value_acc": 0.80, "gold_int8_value_acc": 0.85},
        {"arch": "v2", "channels": 32, "seed": 0, "val_value_acc": 0.99, "gold_int8_value_acc": 0.85},
    ]
    winner, label = select_matrix_winner(entries)
    assert label == "v1:32"
    assert winner["seed"] == 0


def test_matrix_winner_seed_tie_breaks_to_first():
    entries = [
        {"arch": "v1", "channels": 32, "seed": 0, "val_value_acc": 0.90, "gold_int8_value_acc": 0.85},
        {"arch": "v1", "channels": 32, "seed": 1, "val_value_acc": 0.90, "gold_int8_value_acc": 0.85},
    ]
    winner, label = select_matrix_winner(entries)
    assert winner["seed"] == 0


def test_matrix_winner_val_tie_band_breaks_by_f1():
    # 0.9343 / 0.9338 / 0.9346 are within the 0.005 tie band -- practically tied on val
    # value_acc, so the seed with the best gold F1 should win, not the raw
    # max val_value_acc.
    entries = [
        {"arch": "v2", "channels": 48, "seed": 0, "val_value_acc": 0.9343,
         "gold_int8_value_acc": 0.90, "gold_int8_f1": 0.952, "negatives_fp_rate": 0.07},
        {"arch": "v2", "channels": 48, "seed": 1, "val_value_acc": 0.9338,
         "gold_int8_value_acc": 0.90, "gold_int8_f1": 0.956, "negatives_fp_rate": 0.055},
        {"arch": "v2", "channels": 48, "seed": 2, "val_value_acc": 0.9346,
         "gold_int8_value_acc": 0.90, "gold_int8_f1": 0.950, "negatives_fp_rate": 0.091},
    ]
    winner, label = select_matrix_winner(entries)
    assert label == "v2:48"
    assert winner["seed"] == 1, "seed with best F1 among near-tied val_value_acc should win"


def test_matrix_winner_tie_band_is_not_a_rounding_bucket():
    # Real sweep: 0.9351 vs 0.9340 straddle a 2-decimal rounding boundary
    # but are within the band, so the better-F1 seed must still win.
    entries = [
        {"arch": "v2", "channels": 48, "seed": 0, "val_value_acc": 0.9340,
         "gold_int8_value_acc": 0.95, "gold_int8_f1": 0.956, "negatives_fp_rate": 0.018},
        {"arch": "v2", "channels": 48, "seed": 4, "val_value_acc": 0.9351,
         "gold_int8_value_acc": 0.94, "gold_int8_f1": 0.924, "negatives_fp_rate": 0.036},
    ]
    winner, _ = select_matrix_winner(entries)
    assert winner["seed"] == 0


def test_matrix_winner_outside_band_val_acc_wins():
    entries = [
        {"arch": "v2", "channels": 48, "seed": 0, "val_value_acc": 0.920,
         "gold_int8_value_acc": 0.95, "gold_int8_f1": 0.99, "negatives_fp_rate": 0.0},
        {"arch": "v2", "channels": 48, "seed": 1, "val_value_acc": 0.935,
         "gold_int8_value_acc": 0.94, "gold_int8_f1": 0.90, "negatives_fp_rate": 0.1},
    ]
    winner, _ = select_matrix_winner(entries)
    assert winner["seed"] == 1


def test_matrix_winner_f1_tie_breaks_by_fp_rate():
    entries = [
        {"arch": "v1", "channels": 32, "seed": 0, "val_value_acc": 0.90,
         "gold_int8_value_acc": 0.85, "gold_int8_f1": 0.95, "negatives_fp_rate": 0.10},
        {"arch": "v1", "channels": 32, "seed": 1, "val_value_acc": 0.90,
         "gold_int8_value_acc": 0.85, "gold_int8_f1": 0.95, "negatives_fp_rate": 0.02},
    ]
    winner, label = select_matrix_winner(entries)
    assert winner["seed"] == 1, "lower negatives fp_rate should win an F1 tie"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all tests passed")
