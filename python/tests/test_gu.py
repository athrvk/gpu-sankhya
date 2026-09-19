"""Tests for the Gujarati (gu_gujr) language pack: lexicon sanity,
cross-pack collisions, generator round-trips, the fused-hundreds token
boundary and the two irregular bound cardinals (બ / છસ્), corpus
verification, and the hand-written gold set.

Mirrors tests/test_mr.py."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sankhya import classes as C
from sankhya import core
from sankhya import charset as CS
from sankhya import generator as G
from sankhya import llm_corpus as L
from sankhya import verify as V
from sankhya.langs import base

HERE = os.path.dirname(os.path.abspath(__file__))
PY_ROOT = os.path.dirname(HERE)

pack = base.get_pack("gu_gujr")


# --------------------------------------------------------------------------
# lexicon sanity
# --------------------------------------------------------------------------

def test_pack_registered_and_complete():
    assert "gu_gujr" in base.registry
    assert pack.script == "gujr"
    for n in range(1, 100):
        forms = pack.lexicon[f"CARD_{n}"]
        assert forms and all(f.strip() for f in forms), n
    for cls in ("PFX_PAAV", "PFX_AADHA", "PFX_PAUNE", "PFX_SAVA", "PFX_DEDH",
                "PFX_DHAI", "PFX_SAADHE", "UNIT_SAU", "UNIT_HAZAAR",
                "UNIT_LAKH", "UNIT_CRORE", "UNIT_ARAB"):
        assert pack.lexicon.get(cls), cls
    assert "સો" in pack.lexicon["UNIT_SAU"]
    assert "હજાર" in pack.lexicon["UNIT_HAZAAR"]
    assert "કરોડ" in pack.lexicon["UNIT_CRORE"]
    assert "અબજ" in pack.lexicon["UNIT_ARAB"]
    assert "મિલિયન" in pack.lexicon["UNIT_MILLION"]
    assert "બિલિયન" in pack.lexicon["UNIT_BILLION"]
    assert pack.native_digits == "૦૧૨૩૪૫૬૭૮૯"
    assert pack.native_digit_prob > 0


def test_every_surface_form_is_unambiguous_within_the_pack():
    """No surface may mean two different things inside one pack."""
    seen = {}
    for cls, forms in pack.lexicon.items():
        for f in forms:
            key = f.lower()
            assert key not in seen or seen[key] == cls, f"{f!r}: {seen.get(key)} vs {cls}"
            seen[key] = cls


def test_no_cross_pack_conflicts():
    """A surface shared by two packs must map to the SAME class - the
    charset/training data unions every pack. Gujarati is a different
    script from the other three, so in practice this only constrains the
    shared Latin symbol forms (k/L/lac/LPA/cr/bn), which must agree."""
    packs = base.all_packs()
    conflicts = []
    for i, p in enumerate(packs):
        a = p.all_forms()
        for q in packs[i + 1:]:
            b = q.all_forms()
            for surface, cls in a.items():
                if surface in b and b[surface] != cls:
                    conflicts.append((surface, p.id, cls, q.id, b[surface]))
        for w in list(p.filler_words) + list(p.duration_nouns):
            for q in packs:
                if q is p:
                    continue
                other = q.all_forms().get(w.lower())
                if other is not None and other != "O":
                    conflicts.append((w, p.id, "O(filler)", q.id, other))
    assert not conflicts, conflicts


def test_latin_symbol_units_agree_with_the_other_packs():
    assert pack.symbol_units["UNIT_HAZAAR"] == ["k", "K"]
    assert "cr" in pack.symbol_units["UNIT_CRORE"]
    assert "L" in pack.symbol_units["UNIT_LAKH"]
    assert "LPA" in pack.symbol_units["UNIT_LAKH"]
    assert "bn" in pack.symbol_units["UNIT_BILLION"]
    for other_id in ("hi_latn", "hi_deva", "mr_deva"):
        other = base.get_pack(other_id).symbol_units
        for cls, forms in pack.symbol_units.items():
            for f in forms:
                for ocls, oforms in other.items():
                    if f in oforms:
                        assert ocls == cls, (f, cls, other_id, ocls)


def test_range_and_conjunction_connectors():
    assert " થી " in pack.range_connectors
    assert "-" in pack.range_connectors
    assert " કે " in pack.range_connectors
    assert " to " in pack.range_connectors
    assert " અને " in pack.conj_connectors


def test_approximators_and_currency():
    for w in ("લગભગ", "આશરે", "અંદાજે"):
        assert w in pack.approximators, w
    for m in ("₹", "રૂ", "રૂ.", "Rs."):
        assert m in pack.currency_markers_before, m
    for w in ("રૂપિયા", "રૂપિયો"):
        assert w in pack.currency_words_after, w


def test_indefinite_plurals_are_not_quantities():
    for w in ("લાખો", "હજારો", "કરોડો", "લખપતિ", "કરોડપતિ"):
        assert w in pack.indefinite_plurals
        toks = L._word_tokens(pack, w)
        assert toks == [("O", w)], (w, toks)


def test_templates_cover_every_register():
    for family in ("casual", "classifieds", "news", "salary", "ranges",
                   "two_spans", "bare", "short_context", "negatives"):
        assert pack.templates.get(family), family
    for family in ("casual", "classifieds", "news", "salary", "ranges", "two_spans"):
        assert len(pack.templates[family]) >= 20, family
    assert len(pack.templates["negatives"]) >= 40
    negs = " ".join(pack.templates["negatives"])
    for marker in ("લાખો", "કરોડપતિ", "વાગ્યે", "9876543210"):
        assert marker in negs, marker


# --------------------------------------------------------------------------
# hundreds: the glue unit, and the two bound cardinals
# --------------------------------------------------------------------------

def test_fused_hundreds_tokenize_and_evaluate():
    cases = {
        "ત્રણસો": 300, "ચારસો": 400, "પાંચસો": 500, "સાતસો": 700,
        "આઠસો": 800, "નવસો": 900, "એકસો": 100, "બારસો": 1200,
        "અઢારસો": 1800, "બત્રીસસો": 3200, "દોઢસો": 150, "અઢીસો": 250,
        "સાડાત્રણસો": 350, "સાડાબારસો": 1250,
    }
    for word, value in cases.items():
        toks = L._word_tokens(pack, word)
        assert toks is not None, word
        assert toks[-1][0] == "UNIT_SAU" and toks[-1][1] == "સો", (word, toks)
        assert core.evaluate(toks).value == value, (word, toks)


def test_the_hundreds_word_is_also_a_free_standalone_word():
    """Unlike Marathi's bound "शे", Gujarati "સો" is an ordinary word on
    its own ("સો રૂપિયા" = 100), so it is in glue_unit_forms_standalone
    and bound_units() is empty."""
    assert pack.glue_unit_forms == ["સો"]
    assert pack.glue_unit_forms_standalone == ["સો"]
    assert pack.bound_units() == []
    assert L._word_tokens(pack, "સો") == [("UNIT_SAU", "સો")]
    assert core.evaluate(L.tokenize_span_text(pack, "સો રૂપિયા")).value == 100


def test_irregular_hundreds_are_bound_cardinal_forms():
    """બસો = 200 (CARD_2 "બ", never *બેસો) and છસ્સો = 600 (CARD_6 "છસ્"),
    both declared in bound_number_forms as valid only before "સો"."""
    assert pack.bound_number_forms["બ"]["cls"] == "CARD_2"
    assert pack.bound_number_forms["બ"]["before"] == ["સો"]
    assert pack.bound_number_forms["છસ્"]["cls"] == "CARD_6"
    for word, toks_expected, value in (
        ("બસો", [("CARD_2", "બ"), ("UNIT_SAU", "સો")], 200),
        ("છસ્સો", [("CARD_6", "છસ્"), ("UNIT_SAU", "સો")], 600),
        ("છસો", [("CARD_6", "છ"), ("UNIT_SAU", "સો")], 600),
    ):
        toks = L._word_tokens(pack, word)
        assert toks == toks_expected, (word, toks)
        assert core.evaluate(toks).value == value, word


def test_bound_cardinal_forms_never_resolve_standalone():
    """"બ" and "છસ્" exist ONLY glued before "સો". On their own they are
    not words at all, and they must not be reachable through the ordinary
    lexicon, the collision map, or a case-suffix strip."""
    assert L._word_tokens(pack, "બ") is None
    assert L._word_tokens(pack, "છસ્") is None
    assert L._word_tokens(pack, "બની") is None or all(
        not c.startswith("CARD_") for c, _ in L._word_tokens(pack, "બની"))
    forms = pack.all_forms()
    assert "બ" not in forms and "છસ્" not in forms
    # ... and restricted to the unit they precede
    assert pack.bound_number_map(before="સો") == {"બ": "CARD_2", "છસ્": "CARD_6"}
    assert pack.bound_number_map(before="હજાર") == {}


def test_bound_cardinal_verifies_only_before_the_hundreds_word():
    """The verified tier must accept ("CARD_2","બ") inside બસો but never a
    standalone one - otherwise strict mode would report a lone "બ" as 2."""
    assert V.verify_tokens([("CARD_2", "બ"), ("UNIT_SAU", "સો")])
    assert V.verify_tokens([("CARD_6", "છસ્"), ("UNIT_SAU", "સો")])
    assert not V.verify_tokens([("CARD_2", "બ")])
    assert not V.verify_tokens([("CARD_2", "બ"), ("SEP", " "), ("UNIT_SAU", "સો")])
    assert not V.verify_tokens([("CARD_2", "બ"), ("UNIT_HAZAAR", "હજાર")])
    # and the ordinary free form still verifies on its own
    assert V.verify_tokens([("CARD_2", "બે"), ("SEP", " "), ("UNIT_LAKH", "લાખ")])


def test_exported_lexicon_keeps_bound_forms_out_of_forms():
    lex = V.load_lexicon()
    gu = lex["packs"]["gu_gujr"]
    assert "બ" not in gu["forms"] and "છસ્" not in gu["forms"]
    assert gu["bound_forms"]["બ"] == {"cls": "CARD_2", "before": ["સો"]}
    assert gu["bound_forms"]["છસ્"] == {"cls": "CARD_6", "before": ["સો"]}
    # no other pack grew a bound_forms section
    for pid in ("hi_latn", "hi_deva", "mr_deva"):
        assert "bound_forms" not in lex["packs"][pid]


def test_fused_and_split_prefixes_both_parse():
    for word, value in (("સાડાત્રણ", 3.5), ("સાડાપાંચ", 5.5), ("સાડાઆઠ", 8.5)):
        assert core.evaluate(L._word_tokens(pack, word)).value == value, word
    split = L.tokenize_span_text(pack, "સાડા ત્રણ લાખ")
    fused = L.tokenize_span_text(pack, "સાડાત્રણ લાખ")
    assert core.evaluate(split).value == core.evaluate(fused).value == 350000


def test_case_suffixes_resolve_to_the_head_word():
    for word, cls, value in (
        ("હજારનું", "UNIT_HAZAAR", 1000),
        ("હજારમાં", "UNIT_HAZAAR", 1000),
        ("હજારેય", "UNIT_HAZAAR", 1000),
        ("લાખની", "UNIT_LAKH", 100000),
        ("કરોડનો", "UNIT_CRORE", 10000000),
        ("પચાસમાં", "CARD_50", 50),
        ("ત્રીસમાં", "CARD_30", 30),
    ):
        toks = L._word_tokens(pack, word)
        assert toks == [(cls, word)], (word, toks)
        assert core.evaluate(toks).value == value


def test_gold_of_words_that_only_look_like_quantities():
    """"સોનાનો"/"સોનાની" (of gold) must not strip down to "સો" - that is
    why "ના" is not a word suffix in this pack."""
    for w in ("સોના", "સોનાનો", "સોનાની"):
        toks = L._word_tokens(pack, w)
        assert toks is None or all(not c.startswith("UNIT_") for c, _ in toks), (w, toks)
    for text in ("સોનાનો ભાવ વધ્યો છે", "સોનાની વીંટી ખોવાઈ"):
        assert not L._negative_has_quantity(pack, CS.normalize_text(text)), text


def test_fused_hundreds_label_boundaries_survive_decoding():
    """The char labels for "ત્રણસો" put a token boundary inside the word;
    decode_spans must keep the unit tail instead of smoothing it into the
    cardinal (which would value the span at 3, not 300)."""
    from sankhya.decode import decode_spans

    for text, toks, value in (
        ("ત્રણસો", [("CARD_3", "ત્રણ"), ("UNIT_SAU", "સો")], 300),
        ("અઢારસો", [("CARD_18", "અઢાર"), ("UNIT_SAU", "સો")], 1800),
        ("સાડાત્રણસો", [("PFX_SAADHE", "સાડા"), ("CARD_3", "ત્રણ"), ("UNIT_SAU", "સો")], 350),
        ("દોઢસો", [("PFX_DEDH", "દોઢ"), ("UNIT_SAU", "સો")], 150),
        ("છસ્સો", [("CARD_6", "છસ્"), ("UNIT_SAU", "સો")], 600),
    ):
        bio, cls = [], []
        for cl, t in toks:
            for _ in t:
                bio.append(1 if not bio else 2)
                cls.append(C.CLASS_TO_ID[cl])
        decoded = decode_spans(text, bio, cls)
        assert len(decoded) == 1, (text, decoded)
        got = [(C.CLASSES[cid], sub) for cid, sub in decoded[0]["tokens"]]
        assert core.evaluate(got).value == value, (text, got)


# --------------------------------------------------------------------------
# digits
# --------------------------------------------------------------------------

def test_gujarati_digits_normalize_and_verify():
    assert CS.normalize_text("૧,૨૫,૦૦૦") == "1,25,000"
    assert len(CS.normalize_text("૨૭,૪૮,૯૪૫")) == len("૨૭,૪૮,૯૪૫")
    assert V._is_digit_char("૭") and V._is_digit_char("૦")
    assert V.verify_tokens([("DIGITS", "૭૫૦૦")])
    assert core.evaluate([("DIGITS", "૭૫૦૦")]).value == 7500


def test_native_digit_rendering_is_per_pack():
    assert G.to_native_digits("1,25,000.5", pack.native_digits) == "૧,૨૫,૦૦૦.૫"
    deva = base.get_pack("hi_deva")
    assert G.to_native_digits("2025", deva.native_digits) == "२०२५"
    # a pack with no table is a no-op
    assert G.to_native_digits("2025", "") == "2025"


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------

def test_generate_roundtrip_gu_gujr():
    """Same round-trip checks test_generator uses: labels must be
    consistent with the text, and re-evaluating the span's tokens must
    reproduce the recorded value."""
    examples = G.generate(pack, 2000, seed=21)
    assert len(examples) == 2000
    n_pos = 0
    for ex in examples:
        text = ex["text"]
        assert len(ex["bio"]) == len(text) == len(ex["cls"])
        assert len(CS.normalize_text(text)) == len(text)
        for sp in ex["spans"]:
            n_pos += 1
            assert 0 <= sp["start"] < sp["end"] <= len(text)
            assert ex["bio"][sp["start"]] == 1
            assert all(ex["bio"][i] == 2 for i in range(sp["start"] + 1, sp["end"]))
            assert any(ex["cls"][i] != C.CLASS_TO_ID["O"] for i in range(sp["start"], sp["end"]))
            toks = []
            i = sp["start"]
            while i < sp["end"]:
                cid = ex["cls"][i]
                j = i
                while j < sp["end"] and ex["cls"][j] == cid:
                    j += 1
                toks.append((C.CLASSES[cid], CS.normalize_text(text[i:j])))
                i = j
            assert core.evaluate(toks).value == sp["value"], (text, toks, sp)
    assert n_pos > 1000


def test_generator_only_emits_bound_cardinals_glued():
    """"બ"/"છસ્" must never appear standalone or space-separated: every
    occurrence inside a span must be immediately followed by "સો"."""
    examples = G.generate(pack, 4000, seed=22)
    seen_glued = 0
    for ex in examples:
        text = ex["text"]
        for sp in ex["spans"]:
            i = sp["start"]
            while i < sp["end"]:
                cid = ex["cls"][i]
                j = i
                while j < sp["end"] and ex["cls"][j] == cid:
                    j += 1
                sub = text[i:j]
                if sub in ("બ", "છસ્"):
                    assert C.CLASSES[cid].startswith("CARD_"), (text, sub)
                    assert text[j:j + 2] == "સો", (text, sp, repr(text[j:j + 4]))
                    seen_glued += 1
                i = j
    assert seen_glued > 5, seen_glued


def test_generated_bound_cardinals_never_stand_alone_in_text():
    """Belt and braces on the surface text itself: a space-delimited word
    equal to a bound form must not occur anywhere in the generated data."""
    bound = set(pack.bound_number_forms)
    for ex in G.generate(pack, 2000, seed=23):
        for word in ex["text"].split():
            assert word.strip(",.!?/-") not in bound, ex["text"]


def test_generated_examples_are_in_charset():
    vocab = set(CS.build_charset(pack))
    unk_chars = set("".join(G._UNK_POOL))
    examples = G.generate(pack, 300, seed=24)
    for ex in examples:
        for ch in CS.normalize_text(ex["text"]):
            assert ch in vocab or ch in unk_chars, (ch, ex["text"])


def test_charset_covers_the_gujarati_block():
    """The charset is built from the pack's own forms/templates/fillers,
    so registering the pack is what puts the Gujarati block into the
    vocab. Gujarati digits are not there because normalize_text maps
    them to ASCII first."""
    vocab = set(CS.build_charset(pack))
    for ch in "સોહજારલાખકરોડબછણ":
        assert ch in vocab, ch
    # every non-ASCII letter the pack contributes is in the Gujarati block
    for ch in vocab:
        if len(ch) == 1 and not ch.isascii() and ch.isalpha():
            assert "઀" <= ch <= "૿", repr(ch)
    # Gujarati digits never reach the vocab: normalize_text maps them to ASCII
    for d in "૦૧૨૩૪૫૬૭૮૯":
        assert d not in vocab, d
    multi = set(CS.build_charset_multi(base.all_packs()))
    assert vocab <= multi


def test_multi_pack_generation_includes_gu():
    packs = [base.get_pack(x) for x in ("hi_latn", "hi_deva", "mr_deva", "gu_gujr")]
    examples = G.generate_multi(packs, n=400, seed=25, cross=0.1)
    langs = {ex["lang"] for ex in examples}
    assert langs == {"hi_latn", "hi_deva", "mr_deva", "gu_gujr"}


# --------------------------------------------------------------------------
# corpus verification
# --------------------------------------------------------------------------

def test_corpus_verify_accepts_representative_lines():
    lines = [
        {"text": "ક્લાસની ફી સાડા ચાર હજાર થઈ ગઈ છે હવે",
         "lang": "gu_gujr", "phrases": [{"phrase": "સાડા ચાર હજાર", "value": 4500}]},
        {"text": "રિક્ષાવાળાને બસો રૂપિયા આપવા પડ્યા આજે",
         "lang": "gu_gujr", "phrases": [{"phrase": "બસો રૂપિયા", "value": 200}]},
        {"text": "પગાર સવા લાખ અને બોનસ પચાસ હજાર મળ્યું",
         "lang": "gu_gujr", "phrases": [{"phrase": "સવા લાખ", "value": 125000},
                                        {"phrase": "પચાસ હજાર", "value": 50000}]},
        {"text": "બે થી ત્રણ લાખમાં ગાડી મળી જશે તને",
         "lang": "gu_gujr", "phrases": [{"phrase": "બે થી ત્રણ લાખમાં", "value": 200000,
                                         "range": [200000, 300000]}]},
        {"text": "લાખો લોકો ભેગા થયા હતા એ સભામાં", "lang": "gu_gujr", "phrases": []},
    ]
    seen = set()
    for obj in lines:
        ex = L.verify_line(pack, obj, set(), seen)
        seen.add(CS.normalize_text(ex["text"]))
        assert ex["lang"] == "gu_gujr"
        assert len(ex["spans"]) == len(obj["phrases"])
        for sp, claimed in zip(ex["spans"], obj["phrases"]):
            assert sp["value"] == claimed["value"]


def test_corpus_verify_rejects_a_wrong_value():
    obj = {"text": "ભાડું સાડા ત્રણ હજાર છે મહિનાનું", "lang": "gu_gujr",
           "phrases": [{"phrase": "સાડા ત્રણ હજાર", "value": 3000}]}
    try:
        L.verify_line(pack, obj, set(), set())
    except L.RejectError as e:
        assert "value_mismatch" in e.args[0]
    else:
        raise AssertionError("expected a value_mismatch reject")


def test_verified_corpus_file_is_consistent():
    path = os.path.join(PY_ROOT, "data_llm", "gu_gujr.jsonl")
    if not os.path.isfile(path):
        return
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            n += 1
            assert ex["lang"] == "gu_gujr"
            assert len(ex["bio"]) == len(ex["text"]) == len(ex["cls"])
    assert n > 400


# --------------------------------------------------------------------------
# gold set
# --------------------------------------------------------------------------

def _load_gold():
    path = os.path.join(HERE, "gold_gu.jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_gold_file_loads_and_is_well_formed():
    rows = _load_gold()
    assert len(rows) >= 150
    negatives = [r for r in rows if not r["spans"]]
    assert 0.2 <= len(negatives) / len(rows) <= 0.4
    texts = [r["text"] for r in rows]
    assert len(set(texts)) == len(texts)
    for r in rows:
        assert r["lang"] == "gu_gujr"
        norm = CS.normalize_text(r["text"])
        assert len(norm) == len(r["text"])
        for sp in r["spans"]:
            assert 0 <= sp["start"] < sp["end"] <= len(norm)


def test_every_gold_value_reevaluates():
    """Each gold span's text, tokenized with the gu_gujr lexicon and run
    through core.evaluate, must reproduce the recorded value/range."""
    for r in _load_gold():
        norm = CS.normalize_text(r["text"])
        for sp in r["spans"]:
            toks = L.tokenize_span_text(pack, norm[sp["start"]:sp["end"]])
            res = core.evaluate(toks)
            assert res.value == sp["value"], (r["text"], toks, res.value, sp["value"])
            got = list(res.range) if res.range else None
            assert got == sp.get("range"), (r["text"], got, sp.get("range"))


def test_gold_covers_every_prefix_and_shape():
    rows = _load_gold()
    blob = " ".join(r["text"] for r in rows)
    for marker in ("પા ", "અડધ", "પોણા", "સવા", "દોઢ", "અઢી", "સાડા",
                   "સો", "બસો", "છસ્સો", " થી ", " કે ", "-", "₹", "રૂ.",
                   "રૂપિયા", "k", "LPA", "cr", "લાખો", "કરોડપતિ", "વાગ્યે",
                   "૫", "અબજ", "મિલિયન"):
        assert marker in blob, marker
    assert any("સો" in r["text"] and r["spans"] for r in rows)
    # both irregular hundreds appear as real, valued spans
    for word in ("બસો", "છસ્સો"):
        assert any(word in CS.normalize_text(r["text"])[sp["start"]:sp["end"]]
                   for r in rows for sp in r["spans"]), word


def test_gold_spans_end_at_word_boundaries():
    """Per the pack convention (gu_gujr.py word_suffixes, python/README.md),
    a Gujarati postposition attached to a number/unit word stays INSIDE the
    span. So every gold span must end at a word boundary - never mid-word -
    and the char right after a span must never be a Gujarati letter or
    combining mark, which would mean a case ending got cut off."""
    combining = "ઁંઃ઼ાિીુૂૃૄૅેૈૉોૌ્ૢૣ"
    for r in _load_gold():
        text = r["text"]
        for sp in r["spans"]:
            end = sp["end"]
            if end < len(text):
                nxt = text[end]
                assert nxt in " ,.।!?)('\"" or not ("઀" <= nxt <= "૿"), \
                    (r["text"], sp, repr(nxt))
                assert nxt not in combining, (r["text"], sp, repr(nxt))


def test_gold_is_disjoint_from_the_raw_corpus():
    raw = os.path.join(PY_ROOT, "data_llm", "raw", "gu_gujr.sonnet.jsonl")
    if not os.path.isfile(raw):
        return
    corpus = set()
    with open(raw, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus.add(CS.normalize_text(json.loads(line)["text"]))
    for r in _load_gold():
        assert CS.normalize_text(r["text"]) not in corpus, r["text"]


# --------------------------------------------------------------------------
# exported lexicon is in sync with the packs
# --------------------------------------------------------------------------

def test_exported_lexicon_json_matches_the_registered_packs():
    """src/data/lexicon.json is checked in and consumed by the JS
    verifier, so it must match what export_lexicon would write today."""
    from sankhya.export_lexicon import build

    on_disk = V.load_lexicon()
    assert on_disk == build()
    assert set(on_disk["packs"]) == set(base.KNOWN_PACKS)


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
