"""Hinglish (Latin-script romanised Hindi) language pack."""
from __future__ import annotations

from . import base
from .. import noise_latn as NL

# --- 1..99 canonical Hindi cardinals (romanised) -----------------------
_BASE99 = [
    "ek", "do", "teen", "chaar", "paanch", "chhe", "saat", "aath", "nau", "das",
    "gyarah", "baarah", "terah", "chaudah", "pandrah", "solah", "satrah", "atharah", "unnees", "bees",
    "ikkis", "baais", "teis", "chaubis", "pachchis", "chhabbis", "sattais", "atthais", "untis", "tees",
    "iktis", "battis", "taintis", "chauntis", "paintis", "chhattis", "saintis", "adtis", "untalis", "chalis",
    "iktalis", "bayalis", "taintalis", "chauwalis", "paintalis", "chhiyalis", "saintalis", "adtalis", "unchas", "pachas",
    "ikyavan", "bavan", "tirpan", "chauwan", "pachpan", "chhappan", "sattavan", "atthavan", "unsath", "saath",
    "iksath", "basath", "tirsath", "chaunsath", "painsath", "chhiyasath", "sadsath", "adsath", "unhattar", "sattar",
    "ikhattar", "bahattar", "tihattar", "chauhattar", "pachhattar", "chhihattar", "sathattar", "athhattar", "unyasi", "assi",
    "ikyasi", "bayasi", "tirasi", "chaurasi", "pachasi", "chhiyasi", "sattasi", "atthasi", "navasi", "nabbe",
    "ikyanve", "banve", "tiranve", "chauranve", "pachanve", "chhiyanve", "sattanve", "atthanve", "ninyanve",
]
assert len(_BASE99) == 99

CARD_WORDS = {i + 1: [w] for i, w in enumerate(_BASE99)}

_EXTRA_VARIANTS = {
    1: ["ek", "1"],
    2: ["do"],
    3: ["teen", "tin"],
    5: ["paanch", "panch", "paach"],
    6: ["chhe", "che", "chhah", "chah"],
    20: ["bees", "bis", "biss"],
    25: ["pachchis", "pachees", "pachchees"],
    30: ["tees", "tis"],
    50: ["pachas", "pachaas"],
    60: ["saath", "sath"],
    90: ["nabbe", "nabbey"],
}
for n, variants in _EXTRA_VARIANTS.items():
    CARD_WORDS[n] = list(dict.fromkeys(CARD_WORDS[n] + variants))

_ENGLISH_CARD = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
    9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
    30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
}
for n, w in _ENGLISH_CARD.items():
    CARD_WORDS[n].append(w)

LEXICON = {}
for n in range(1, 100):
    LEXICON[f"CARD_{n}"] = CARD_WORDS[n]

LEXICON.update({
    "PFX_PAAV": ["paav", "pav", "pau", "paw", "pauv"],
    "PFX_AADHA": ["aadha", "adha", "aadhaa", "adhaa", "aadh", "aadhe", "adhe", "aadhi", "adhi"],
    "PFX_PAUNE": ["paune", "pone", "pauna", "paun", "pona", "pawne", "powne", "paunay"],
    "PFX_SAVA": ["sava", "sawa", "savaa", "sawaa", "swa", "savva"],
    "PFX_DEDH": ["dedh", "derh", "ded", "dhed", "dhedh", "dedhh", "dedhi"],
    "PFX_DHAI": ["dhai", "dhaai", "dhaee", "dhayi", "dhay", "adhai", "adhaai", "arhai", "arhaai", "dhaii"],
    "PFX_SAADHE": ["saadhe", "sadhe", "sade", "saade", "sarhe", "sadhey", "saadhey"],

    "UNIT_SAU": ["sau", "so", "sao", "sou", "hundred"],
    "UNIT_HAZAAR": ["hazaar", "hazar", "hajaar", "hajar", "hazzar", "thousand", "thou"],
    "UNIT_LAKH": ["lakh", "lac", "lack", "laakh", "lakhs", "lacs", "peti"],
    "UNIT_MILLION": ["million", "mil"],
    "UNIT_CRORE": ["crore", "karod", "karor", "karore", "crores", "karodh", "khokha"],
    "UNIT_BILLION": ["billion"],
    "UNIT_ARAB": ["arab"],
    "UNIT_KHARAB": ["kharab"],
})

SYMBOL_UNITS = {
    "UNIT_HAZAAR": ["k", "K", "hzr"],
    "UNIT_LAKH": ["L", "l", "lk", "lkh", "LPA", "lpa"],
    "UNIT_MILLION": ["M", "mn"],
    "UNIT_CRORE": ["cr", "Cr", "CR", "crs"],
    "UNIT_BILLION": ["B", "bn"],
}

ENGLISH_FRACTION_PHRASES = {
    "PFX_SAADHE": ["and a half", "and half", "n half"],
    "PFX_SAVA": ["and a quarter"],
    "PFX_AADHA": ["half a", "half"],
}

INDEFINITE_PLURALS = [
    "lakhon", "lakho", "laakhon", "hazaaron", "hazaron", "karodon", "crodon",
    "croron", "sainkdon",
]

CURRENCY_MARKERS_BEFORE = ["₹", "Rs", "Rs.", "rs", "INR", "Re"]
CURRENCY_WORDS_AFTER = ["rupaye", "rupay", "rupees", "rupiya", "rupya", "/-"]
APPROXIMATORS = ["karib", "kareeb", "lagbhag", "takreeban", "around", "approx", "about", "roughly", "~", "koi", "kuch"]
RANGE_CONNECTORS = [" - ", "-", " to ", " se ", " ya ", " / ", " or "]

TEMPLATES = {
    "casual": [
        "bhai {A} {P} mein ho jayega kya",
        "usne {C}{P} maange yaar",
        "{P} toh bahut zyada hai",
        "mera budget {P} tak ka hai",
        "yaar {P} mein deal ho gayi",
        "abhi {C}{P} hi hai mere paas",
        "kal {P} udha liya tha",
        "itna {P} kaha se launga",
        "usko {P} chahiye the",
        "{A} {P} lagega isme",
        "mujhe {P} udhaar dedo",
        "paapa ne {P} diye the",
        "shaadi mein {P} kharch ho gaya",
        "{P} ka jugaad ho gaya bhai",
        "wo {P} maang raha hai mujhse",
    ],
    "classifieds": [
        "Price: {C}{P} negotiable",
        "2019 model, {P} final",
        "rent {P}/month, 2bhk",
        "MRP {C}{P} only",
        "asking price {C}{P}",
        "for sale {C}{P} fixed",
        "brand new, {P} onwards",
        "urgent sale {C}{P} slightly negotiable",
        "flat for {C}{P}, ready to move",
        "bike {P} only, genuine buyers",
        "{P} best price guaranteed",
        "laptop {P} negotiable slightly",
        "resale value {C}{P}",
        "cash price {C}{P}",
        "final price {C}{P} no bargain",
    ],
    "news": [
        "{P} logon ne attend kiya",
        "startup ne {P} ka funding raise kiya",
        "{P} se zyada views aa gaye",
        "sarkar ne {P} ka package announce kiya",
        "company ne {P} ka nuksaan bataya",
        "video ko {P} baar dekha gaya",
        "iss saal {P} logon ne vote diya",
        "unhone {P} ka daan diya",
        "prashasan ne {P} ka bajat rakha",
        "{P} ki lagat aayi hai",
        "report ke anusaar {P} kharch hua",
        "yojana mein {P} kharch honge",
        "{P} ka ghotala samne aaya",
        "in logon ne {P} kamaye",
        "{P} ka mahasangram hua",
    ],
    "salary": [
        "CTC {P} per annum",
        "package {P}",
        "EMI {C}{P} monthly",
        "loan {P} ka liya tha",
        "salary {P} mahine ki hai",
        "in hand {P} milta hai",
        "increment ke baad {P} ho gaya",
        "bonus {C}{P} mila",
        "stipend {P} milta hai",
        "hike {P} ka mila",
        "annual package {P} hai",
        "monthly kharch {P} hai",
        "rent {P} deta hoon",
        "fee {P} lagti hai",
        "tax {C}{P} banta hai",
    ],
    "ranges": [
        "{P} mein aa jayega",
        "{P} ki deal hai",
        "{P} ka phone hai",
        "{P} ke beech mein",
        "{P} tak jayega",
        "{P} lagenge",
        "budget {P} hai mera",
        "range {P} tak hai",
        "kimat {P} ke beech hogi",
        "{P} ka scope hai",
        "{P} ke beech deal hogi",
        "rent {P} tak hoga",
        "salary {P} milegi",
        "package {P} ke beech",
        "cost {P} tak",
    ],
    "two_spans": [
        "{P1} se badhkar {P2} ho gaya",
        "pehle {P1} tha ab {P2}",
        "{P1} + {P2} = ?",
        "pichle saal {P1} tha is saal {P2}",
        "{P1} se {P2} tak pahunch gaya",
        "budget {P1} tha ab {P2} ho gaya",
        "kal {P1} tha aaj {P2}",
        "{P1} khatam hokar {P2} bacha",
        "usne {P1} diya maine {P2} liya",
        "{P1} lagaya {P2} kamaya",
    ],
    "bare": [
        "{P}",
        "{A} {P}",
        "{C}{P}",
        "{P}!!",
        "{P}?",
        "{C}{P}/-",
        "{P} rupaye",
        "bas {P}",
        "sirf {P}",
        "kul {P}",
    ],
    "negatives": [
        "Pune mein flat lena hai",
        "Lakhan bhai ka number 9876543210 hai",
        "KBC crorepati bann gaya",
        "lakhon log aaye the",
        "do din mein aa jaunga",
        "pin 400001 hai mera",
        "2.30 baje milte hain",
        "so what yaar",
        "chaar dost the hum",
        "15 August ko chhutti hai",
        "sauda pakka ho gaya",
        "crorepati banna hai mujhe",
        "hazaaron log the wahan",
        "9123456780 pe call karo",
        "date 12/05 hai meeting ki",
        "unnees so pachasi mein hua tha",
    ],
}


def _hi_latn_noise(word: str, rng) -> str:
    return NL.apply_noise(word, rng, apply_casing=True, apply_plural=False)


BLOCKED_SURFACES = ["pune"]

PACK = base.LanguagePack(
    id="hi_latn",
    script="latn",
    lexicon=LEXICON,
    symbol_units=SYMBOL_UNITS,
    english_fraction_phrases=ENGLISH_FRACTION_PHRASES,
    indefinite_plurals=INDEFINITE_PLURALS,
    currency_markers_before=CURRENCY_MARKERS_BEFORE,
    currency_words_after=CURRENCY_WORDS_AFTER,
    approximators=APPROXIMATORS,
    range_connectors=RANGE_CONNECTORS,
    templates=TEMPLATES,
    noise_fn=_hi_latn_noise,
    blocked_surfaces=BLOCKED_SURFACES,
)

base.register(PACK)
