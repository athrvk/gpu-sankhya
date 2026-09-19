"""Tests for the Devanagari Marathi (mr_deva) language pack: lexicon
sanity, cross-pack collisions, generator round-trips, the fused-hundreds
token boundary, corpus verification, and the hand-written gold set."""
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
from sankhya.langs import base

HERE = os.path.dirname(os.path.abspath(__file__))
PY_ROOT = os.path.dirname(HERE)

pack = base.get_pack("mr_deva")


# --------------------------------------------------------------------------
# lexicon sanity
# --------------------------------------------------------------------------

def test_pack_registered_and_complete():
    assert "mr_deva" in base.registry
    assert pack.script == "deva"
    for n in range(1, 100):
        forms = pack.lexicon[f"CARD_{n}"]
        assert forms and all(f.strip() for f in forms), n
    for cls in ("PFX_PAAV", "PFX_AADHA", "PFX_PAUNE", "PFX_SAVA", "PFX_DEDH",
                "PFX_DHAI", "PFX_SAADHE", "UNIT_SAU", "UNIT_HAZAAR",
                "UNIT_LAKH", "UNIT_CRORE", "UNIT_ARAB"):
        assert pack.lexicon.get(cls), cls
    assert "शंभर" in pack.lexicon["UNIT_SAU"]
    assert "शे" in pack.lexicon["UNIT_SAU"]
    assert "कोटी" in pack.lexicon["UNIT_CRORE"] and "करोड" in pack.lexicon["UNIT_CRORE"]
    assert "अब्ज" in pack.lexicon["UNIT_ARAB"]
    assert pack.native_digits == "०१२३४५६७८९"
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
    charset/training data unions every pack, so a surface that is CARD_13
    in one pack and an O-labelled filler in another teaches the model two
    contradictory labels."""
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


def test_symbol_units_present():
    assert pack.symbol_units["UNIT_HAZAAR"] == ["k", "K"]
    assert "cr" in pack.symbol_units["UNIT_CRORE"]
    assert "L" in pack.symbol_units["UNIT_LAKH"]


def test_range_and_conjunction_connectors():
    assert " ते " in pack.range_connectors
    assert "-" in pack.range_connectors
    assert " किंवा " in pack.range_connectors
    assert " आणि " in pack.conj_connectors


def test_indefinite_plurals_are_not_quantities():
    for w in ("लाखो", "हजारो", "करोडो", "कोट्यवधी", "लखपती", "करोडपती"):
        assert w in pack.indefinite_plurals
        toks = L._word_tokens(pack, w)
        assert toks == [("O", w)], (w, toks)


def test_templates_cover_every_register():
    for family in ("casual", "classifieds", "news", "salary", "ranges",
                   "two_spans", "bare", "short_context", "negatives"):
        assert pack.templates.get(family), family
    for family in ("casual", "classifieds", "news", "salary", "ranges", "two_spans"):
        assert len(pack.templates[family]) >= 20, family
    negs = " ".join(pack.templates["negatives"])
    for marker in ("लाखो", "करोडपती", "वाजता", "9876543210"):
        assert marker in negs, marker


# --------------------------------------------------------------------------
# fused hundreds / fused prefixes / case suffixes
# --------------------------------------------------------------------------

def test_fused_hundreds_tokenize_and_evaluate():
    cases = {
        "दोनशे": 200, "तीनशे": 300, "नऊशे": 900, "अठराशे": 1800,
        "बाराशे": 1200, "बत्तीसशे": 3200, "दीडशे": 150, "अडीचशे": 250,
        "साडेतीनशे": 350, "साडेबाराशे": 1250,
    }
    for word, value in cases.items():
        toks = L._word_tokens(pack, word)
        assert toks is not None, word
        assert toks[-1][0] == "UNIT_SAU" and toks[-1][1] == "शे", (word, toks)
        assert core.evaluate(toks).value == value, (word, toks)


def test_bound_hundreds_suffix_is_never_a_standalone_word():
    """"शे" is a bound morpheme: on its own the run is some other word
    (e.g. "शेत" = field), never UNIT_SAU."""
    assert L._word_tokens(pack, "शे") is None
    toks = L._word_tokens(pack, "शेत")
    assert toks is None or all(not c.startswith("UNIT_") for c, _ in toks)


def test_fused_and_split_prefixes_both_parse():
    for word, value in (("साडेतीन", 3.5), ("साडेपाच", 5.5), ("पावणेदोन", 1.75)):
        assert core.evaluate(L._word_tokens(pack, word)).value == value, word
    split = L.tokenize_span_text(pack, "साडे तीन लाख")
    fused = L.tokenize_span_text(pack, "साडेतीन लाख")
    assert core.evaluate(split).value == core.evaluate(fused).value == 350000


def test_case_suffixes_resolve_to_the_head_word():
    for word, cls, value in (
        ("हजारात", "UNIT_HAZAAR", 1000),
        ("लाखांचा", "UNIT_LAKH", 100000),
        ("कोटींची", "UNIT_CRORE", 10000000),
        ("तीसच", "CARD_30", 30),
        ("पन्नासमध्ये", "CARD_50", 50),
    ):
        toks = L._word_tokens(pack, word)
        assert toks == [(cls, word)], (word, toks)
        assert core.evaluate(toks).value == value


def test_fused_hundreds_label_boundaries_survive_decoding():
    """The char labels for "दोनशे" put a token boundary inside the word;
    decode_spans must keep the 2-char unit tail instead of smoothing it
    into the cardinal (which would value the span at 2, not 200)."""
    from sankhya.decode import decode_spans

    for text, toks, value in (
        ("दोनशे", [("CARD_2", "दोन"), ("UNIT_SAU", "शे")], 200),
        ("आठशे", [("CARD_8", "आठ"), ("UNIT_SAU", "शे")], 800),
        ("साडेतीनशे", [("PFX_SAADHE", "साडे"), ("CARD_3", "तीन"), ("UNIT_SAU", "शे")], 350),
        ("दीडशे", [("PFX_DEDH", "दीड"), ("UNIT_SAU", "शे")], 150),
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
# generator
# --------------------------------------------------------------------------

def test_generate_roundtrip_mr_deva():
    """Same round-trip checks test_generator uses: labels must be
    consistent with the text, and re-evaluating the span's tokens must
    reproduce the recorded value."""
    examples = G.generate(pack, 2000, seed=11)
    assert len(examples) == 2000
    n_pos = 0
    for ex in examples:
        text = ex["text"]
        assert len(ex["bio"]) == len(text) == len(ex["cls"])
        assert CS.normalize_text(text) == CS.normalize_text(text)
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


def test_generator_only_emits_the_bound_hundreds_form_glued():
    """"शे" must never appear standalone or space-separated in generated
    text; when the shape doesn't allow gluing, शंभर is used instead."""
    examples = G.generate(pack, 3000, seed=12)
    seen_glued = 0
    for ex in examples:
        text = ex["text"]
        for sp in ex["spans"]:
            for i in range(sp["start"], sp["end"] - 1):
                if C.CLASSES[ex["cls"][i]] != "UNIT_SAU" or text[i:i + 2] != "शे":
                    continue
                # the bound hundreds form is always glued to the CARD_*/PFX_*
                # word before it - never standalone, never after a space
                assert i > sp["start"], (text, sp)
                prev = C.CLASSES[ex["cls"][i - 1]]
                assert text[i - 1] != " " and (prev.startswith("CARD_") or prev.startswith("PFX_")), \
                    (text, sp, prev)
                seen_glued += 1
    assert seen_glued > 20, seen_glued


def test_generated_examples_are_in_charset():
    vocab = set(CS.build_charset(pack))
    # the out-of-vocab "unk noise" runs are deliberately outside the charset
    unk_chars = set("".join(G._UNK_POOL))
    examples = G.generate(pack, 300, seed=13)
    for ex in examples:
        for ch in CS.normalize_text(ex["text"]):
            assert ch in vocab or ch in unk_chars, (ch, ex["text"])


def test_multi_pack_generation_includes_mr():
    packs = [base.get_pack(x) for x in ("hi_latn", "hi_deva", "mr_deva")]
    examples = G.generate_multi(packs, n=300, seed=14, cross=0.1)
    langs = {ex["lang"] for ex in examples}
    assert langs == {"hi_latn", "hi_deva", "mr_deva"}


# --------------------------------------------------------------------------
# corpus verification
# --------------------------------------------------------------------------

def test_corpus_verify_accepts_representative_lines():
    lines = [
        {"text": "क्लासची फी साडेचार हजार झाली आहे आता",
         "lang": "mr_deva", "phrases": [{"phrase": "साडेचार हजार", "value": 4500}]},
        {"text": "रिक्षाला अठराशे रुपये झाले या महिन्यात",
         "lang": "mr_deva", "phrases": [{"phrase": "अठराशे रुपये", "value": 1800}]},
        {"text": "पगार सव्वा लाख आणि बोनस पन्नास हजार मिळाला",
         "lang": "mr_deva", "phrases": [{"phrase": "सव्वा लाख", "value": 125000},
                                        {"phrase": "पन्नास हजार", "value": 50000}]},
        {"text": "दोन ते तीन लाखात मिळेल गाडी",
         "lang": "mr_deva", "phrases": [{"phrase": "दोन ते तीन लाखात", "value": 200000,
                                         "range": [200000, 300000]}]},
        {"text": "लाखो लोक जमले होते त्या सभेला", "lang": "mr_deva", "phrases": []},
    ]
    seen = set()
    for obj in lines:
        ex = L.verify_line(pack, obj, set(), seen)
        seen.add(CS.normalize_text(ex["text"]))
        assert ex["lang"] == "mr_deva"
        assert len(ex["spans"]) == len(obj["phrases"])
        for sp, claimed in zip(ex["spans"], obj["phrases"]):
            assert sp["value"] == claimed["value"]


def test_corpus_verify_rejects_a_wrong_value():
    obj = {"text": "भाडं साडेतीन हजार आहे महिन्याचं", "lang": "mr_deva",
           "phrases": [{"phrase": "साडेतीन हजार", "value": 3000}]}
    try:
        L.verify_line(pack, obj, set(), set())
    except L.RejectError as e:
        assert "value_mismatch" in e.args[0]
    else:
        raise AssertionError("expected a value_mismatch reject")


def test_verified_corpus_file_is_consistent():
    path = os.path.join(PY_ROOT, "data_llm", "mr_deva.jsonl")
    if not os.path.isfile(path):
        return
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            n += 1
            assert ex["lang"] == "mr_deva"
            assert len(ex["bio"]) == len(ex["text"]) == len(ex["cls"])
    assert n > 400


# --------------------------------------------------------------------------
# gold set
# --------------------------------------------------------------------------

def _load_gold():
    path = os.path.join(HERE, "gold_mr.jsonl")
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
        assert r["lang"] == "mr_deva"
        norm = CS.normalize_text(r["text"])
        assert len(norm) == len(r["text"])
        for sp in r["spans"]:
            assert 0 <= sp["start"] < sp["end"] <= len(norm)


def test_every_gold_value_reevaluates():
    """Each gold span's text, tokenized with the mr_deva lexicon and run
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
    for marker in ("पाव ", "अर्ध", "पावणे", "सव्वा", "दीड", "अडीच", "साडे",
                   "शे", "शंभर", " ते ", "किंवा", "-", "₹", "रु.", "रुपये",
                   "k", "LPA", "cr", "लाखो", "करोडपती", "वाजता"):
        assert marker in blob, marker
    assert any("शे" in r["text"] and r["spans"] for r in rows)


def test_gold_spans_end_at_word_boundaries():
    """Per the pack convention (mr_deva.py word_suffixes /
    word_oblique_endings, python/README.md), a Marathi case ending
    attached to a number/unit word stays INSIDE the span. So every gold
    span must end at a word boundary (space, punctuation, or end of
    text) - never mid-word - and the char right after a span must never
    be a Devanagari letter or combining mark, which would mean a case
    ending got cut off outside the span."""
    combining = "ऀँंऺऻ़ाि" \
        "ीुूृॄॅॆेै" \
        "ॉॊोौ्ॎॏ॒॑" \
        "॓॔ॕॖॗॢॣ"
    for r in _load_gold():
        text = r["text"]
        for sp in r["spans"]:
            end = sp["end"]
            if end < len(text):
                nxt = text[end]
                assert nxt in " ,.।!?)('\"" or not (
                    "ऀ" <= nxt <= "ॿ"
                ), (r["text"], sp, repr(nxt))
                assert nxt not in combining, (r["text"], sp, repr(nxt))


def test_gold_is_disjoint_from_the_raw_corpus():
    raw = os.path.join(PY_ROOT, "data_llm", "raw", "mr_deva.sonnet.jsonl")
    if not os.path.isfile(raw):
        return
    corpus = set()
    with open(raw, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus.add(CS.normalize_text(json.loads(line)["text"]))
    for r in _load_gold():
        assert CS.normalize_text(r["text"]) not in corpus, r["text"]


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
