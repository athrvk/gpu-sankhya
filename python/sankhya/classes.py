"""Language-independent semantic class inventory for gpu-sankhya.

Nothing here references a specific language / script.
"""
from __future__ import annotations

PREFIXES = ["PFX_PAAV", "PFX_AADHA", "PFX_PAUNE", "PFX_SAVA", "PFX_DEDH", "PFX_DHAI", "PFX_SAADHE"]

CARDINALS = [f"CARD_{i}" for i in range(1, 100)]

UNITS = [
    "UNIT_SAU", "UNIT_HAZAAR", "UNIT_LAKH", "UNIT_MILLION",
    "UNIT_CRORE", "UNIT_BILLION", "UNIT_ARAB", "UNIT_KHARAB",
]

MISC = ["O", "SEP", "RANGE", "DIGITS", "DOT", "COMMA"]

# Stable order, O first.
CLASSES = ["O"] + [c for c in MISC if c != "O"] + PREFIXES + CARDINALS + UNITS
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

_UNIT_VALUE = {
    "UNIT_SAU": 1e2,
    "UNIT_HAZAAR": 1e3,
    "UNIT_LAKH": 1e5,
    "UNIT_MILLION": 1e6,
    "UNIT_CRORE": 1e7,
    "UNIT_BILLION": 1e9,
    "UNIT_ARAB": 1e9,
    "UNIT_KHARAB": 1e11,
}

UNIT_NAME = {
    "UNIT_SAU": "sau",
    "UNIT_HAZAAR": "hazaar",
    "UNIT_LAKH": "lakh",
    "UNIT_MILLION": "million",
    "UNIT_CRORE": "crore",
    "UNIT_BILLION": "billion",
    "UNIT_ARAB": "arab",
    "UNIT_KHARAB": "kharab",
}

# standalone value, and semantics with a following/preceding number.
# op: how to combine with N (None -> never combines, standalone only)
PREFIX_INFO = {
    "PFX_PAAV": {"standalone": 0.25, "op": None},
    "PFX_AADHA": {"standalone": 0.5, "op": None},
    "PFX_PAUNE": {"standalone": 0.75, "op": "sub25", "min_n": 2},
    "PFX_SAVA": {"standalone": 1.25, "op": "add25", "min_n": 2},
    "PFX_DEDH": {"standalone": 1.5, "op": None},
    "PFX_DHAI": {"standalone": 2.5, "op": None},
    "PFX_SAADHE": {"standalone": 0.5, "op": "add5", "min_n": 3},
}

PREFIX = PREFIX_INFO  # alias requested in spec


def is_prefix(cls: str) -> bool:
    return cls in PREFIX_INFO


def is_card(cls: str) -> bool:
    return cls.startswith("CARD_")


def is_unit(cls: str) -> bool:
    return cls in _UNIT_VALUE


def card_value(cls: str) -> int:
    return int(cls.split("_", 1)[1])


def unit_value(cls: str) -> float:
    return _UNIT_VALUE[cls]
