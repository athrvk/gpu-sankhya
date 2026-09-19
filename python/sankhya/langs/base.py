"""Base LanguagePack interface. Language-specific packs subclass this."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass
class LanguagePack:
    id: str
    script: str
    lexicon: Dict[str, List[str]] = field(default_factory=dict)
    symbol_units: Dict[str, List[str]] = field(default_factory=dict)
    english_fraction_phrases: Dict[str, List[str]] = field(default_factory=dict)
    indefinite_plurals: List[str] = field(default_factory=list)
    currency_markers_before: List[str] = field(default_factory=list)
    currency_words_after: List[str] = field(default_factory=list)
    approximators: List[str] = field(default_factory=list)
    range_connectors: List[str] = field(default_factory=list)
    templates: Dict[str, List[str]] = field(default_factory=dict)
    noise_fn: Callable = None  # noise(word, rng) -> str
    # surface forms that must never be emitted even if noise produces them
    # (e.g. they collide with a real-world proper noun / place name), and
    # which the VERIFIER and the decoder gate also refuse: a blocked surface
    # never verifies as a number word, and a span carrying one is dropped
    # (R14). Motivated by real text: "अण्णा हजारे" / "सवाई तुकोजीराव" /
    # "अरबी समुद्र" are names, never amounts -- see data_wild/REPORT.md §4.1.
    blocked_surfaces: List[str] = field(default_factory=list)
    # Lexicon SURFACES (not classes) that are real number words in this
    # language AND common ordinary words: hi_latn "so" (English "so" /
    # UNIT_SAU), "sath" ("with" / CARD_60), "mil" ("meet"/"mile" /
    # UNIT_MILLION), "arab"/"अरब"/"અરબ" (the ethnonym / UNIT_ARAB). They
    # stay in the lexicon -- "ek so", "das arab" are genuine amounts -- but
    # R12 drops a span in which such a surface stands with no independently
    # justified number word beside it. See data_wild/REPORT.md §4.2/§4.3.
    ambiguous_forms: List[str] = field(default_factory=list)
    # common chat/English words used as O-labelled filler around spans, so
    # the model learns unknown words outside a span are not part of it
    filler_words: List[str] = field(default_factory=list)
    # duration/time/count nouns for "CARD/DIGITS + noun" negatives
    # ("teen din baad", "das minute mein") - bare numbers with no unit/currency
    duration_nouns: List[str] = field(default_factory=list)
    # conjunction words that join TWO (sometimes three) independent quantity
    # spans in one text ("sava lakh aur dedh lakh") - the connector itself is
    # O, each side is its own span. Do not confuse with range_connectors
    # ("se lekar ... tak" etc): a range is ONE span, this is several.
    conj_connectors: List[str] = field(default_factory=list)
    # spelled-out UNIT_* words that also have a common non-numeric meaning
    # in this language ("kharab" = 10^11 unit word but also "broken";
    # "mil" = million but also "meet"). A lone occurrence of one of these
    # in negative-quantity scanning only counts as a quantity marker when
    # immediately preceded by a number-ish token (CARD_*/DIGITS/PFX_* or a
    # digit run) - see _negative_has_quantity in llm_corpus.py.
    ambiguous_units: List[str] = field(default_factory=list)
    # UNIT_* surface forms that are BOUND morphemes: they only ever occur
    # glued (no space) to the number word before them. Marathi's fused
    # hundreds suffix "शे" (दोनशे = CARD_2 + UNIT_SAU) is the motivating
    # case. The generator never emits one standalone or space-separated
    # (generator._apply_pack_glue), and the corpus tokenizer splits such a
    # word back into its two tokens (llm_corpus._word_tokens).
    glue_unit_forms: List[str] = field(default_factory=list)
    # Subset of `glue_unit_forms` that ALSO exists as a free standalone
    # word. Marathi's "शे" is purely bound, so its list is empty; Gujarati's
    # "સો" is glued in બસો/અઢારસો but is equally a word on its own
    # ("સો રૂપિયા" = 100, "સાડા સો" = 150). Forms listed here keep the
    # gluing behaviour but are not refused when they stand alone.
    glue_unit_forms_standalone: List[str] = field(default_factory=list)
    # CARD_*/PFX_* surface forms that exist ONLY in the glued position
    # immediately before one of `glue_unit_forms`. Gujarati's 200 is બસો,
    # not *બેસો: the bound-position form of CARD_2 is "બ", which is not a
    # word on its own and must never be labelled or verified standalone.
    # Shape: {surface: {"cls": CLASS, "before": [unit surfaces], "p": prob}}
    # where `p` is how often the generator prefers it over the free form.
    # These forms are deliberately NOT part of `all_forms()`, so they never
    # enter the cross-pack collision map, the standalone lexicon lookup, or
    # the verifier's context-free surface table -- see `bound_number_map()`.
    bound_number_forms: Dict[str, dict] = field(default_factory=dict)
    # Per-pack native digit glyphs (Devanagari "०१२...", Gujarati "૦૧૨...").
    # `native_digit_prob` is how often the generator renders a DIGITS token
    # with them instead of ASCII. Token CLASSES are unaffected: the shared
    # normalization maps these back to ASCII before labels are computed.
    native_digits: str = ""
    native_digit_prob: float = 0.0
    # PFX_* surface forms commonly typed as ONE word with the cardinal that
    # follows ("साडेतीन", "पावणेदोन"): surface -> probability the generator
    # glues them. Both spellings stay valid input for the tokenizer.
    fused_prefix_forms: Dict[str, float] = field(default_factory=dict)
    # Case endings that attach directly to a unit/cardinal word in this
    # language ("हजारात", "लाखांचा", "तीसच"). The corpus tokenizer strips
    # them (longest first, repeatedly) to recognise the head word; the
    # suffix characters stay part of that word's token.
    word_suffixes: List[str] = field(default_factory=list)
    # Oblique stem endings tried after a suffix strip ("हजारा-", "कोटीं-").
    word_oblique_endings: List[str] = field(default_factory=list)

    def noise(self, word: str, rng) -> str:
        if self.noise_fn is not None:
            return self.noise_fn(word, rng)
        return word

    def bound_units(self) -> List[str]:
        """Glue unit forms that are BOUND: never valid on their own."""
        free = {f.lower() for f in self.glue_unit_forms_standalone}
        return [f for f in self.glue_unit_forms if f.lower() not in free]

    def bound_number_map(self, before: str = None) -> Dict[str, str]:
        """surface(lower) -> class for the pack's bound number forms.

        With `before` (a glue unit surface), restricted to the forms that
        may precede exactly that unit. Callers must only consult this in
        the glued position -- that is what keeps "બ" from ever resolving,
        or verifying, as a standalone CARD_2.
        """
        out = {}
        for surface, spec in self.bound_number_forms.items():
            if before is not None:
                allowed = [u.lower() for u in spec.get("before", ())]
                if before.lower() not in allowed:
                    continue
            out[surface.lower()] = spec["cls"]
        return out

    def all_forms(self) -> Dict[str, str]:
        """Map surface_lower -> class, across every table, for collision checks."""
        out = {}

        def add(forms, cls):
            for f in forms:
                out.setdefault(f.lower(), cls)

        for cls, forms in self.lexicon.items():
            add(forms, cls)
        for cls, forms in self.symbol_units.items():
            add(forms, cls)
        for cls, forms in self.english_fraction_phrases.items():
            add(forms, cls)
        for w in self.indefinite_plurals:
            out.setdefault(w.lower(), "O")
        return out


registry: Dict[str, LanguagePack] = {}


def register(pack: LanguagePack) -> None:
    registry[pack.id] = pack


# every pack module shipped in this package, in a stable order; importing
# one registers its PACK. `load_all()` is what lets callers iterate
# `registry` instead of hard-coding pack ids.
KNOWN_PACKS = ("hi_latn", "hi_deva", "mr_deva", "gu_gujr")


def load_all() -> Dict[str, LanguagePack]:
    """Import every known pack module and return the registry."""
    import importlib

    for lang_id in KNOWN_PACKS:
        if lang_id not in registry:
            importlib.import_module(f"{__package__}.{lang_id}")
    return registry


def all_packs() -> List[LanguagePack]:
    """Every registered pack, in KNOWN_PACKS order (loading them first)."""
    load_all()
    return [registry[k] for k in KNOWN_PACKS if k in registry]


def get_pack(lang_id: str) -> LanguagePack:
    if lang_id not in registry and lang_id in KNOWN_PACKS:
        import importlib

        importlib.import_module(f"{__package__}.{lang_id}")
    return registry[lang_id]


GOLD_SUFFIX_ALIASES = {"": "hi_latn", "deva": "hi_deva", "latn": "hi_latn",
                        "mr": "mr_deva", "gu": "gu_gujr"}


def resolve_pack(lang_id: str):
    """Resolve an example's `lang` (or a `gold_<suffix>.jsonl` suffix) to a
    pack. Tries the id itself, then the gold-suffix aliases, then any
    registered pack whose id ends/starts with that suffix - so a new pack is
    picked up with no code change here. Falls back to the first registered
    pack (best-effort, never fatal). Pure: safe to call from tooling that
    must not import numpy/torch (hf_push)."""
    packs = load_all()
    if lang_id in packs:
        return packs[lang_id]
    alias = GOLD_SUFFIX_ALIASES.get(lang_id or "")
    if alias and alias in packs:
        return packs[alias]
    if lang_id:
        matches = [p for k, p in packs.items() if k.endswith(f"_{lang_id}") or k.startswith(f"{lang_id}_")]
        if len(matches) == 1:
            return matches[0]
    ordered = all_packs()
    return ordered[0] if ordered else None
