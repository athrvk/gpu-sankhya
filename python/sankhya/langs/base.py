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
    # (e.g. they collide with a real-world proper noun / place name)
    blocked_surfaces: List[str] = field(default_factory=list)

    def noise(self, word: str, rng) -> str:
        if self.noise_fn is not None:
            return self.noise_fn(word, rng)
        return word

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


def get_pack(lang_id: str) -> LanguagePack:
    if lang_id not in registry:
        # lazy-load known packs
        if lang_id == "hi_latn":
            from . import hi_latn  # noqa: F401
    return registry[lang_id]
