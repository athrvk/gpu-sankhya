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

# Curated Hinglish-chat spellings for every 1..99 cardinal (2-4 each), so
# common non-canonical spellings like "unnasi" (79) are recognised directly
# instead of relying on the ~4%-rate N7 char-noise to happen to produce them.
# Checked for collisions: no variant here collides (case-insensitively) with
# a different cardinal, or with any prefix/unit/symbol/english-fraction form
# (see python/tests/test_generator.py::test_cardinal_variants_map_correctly).
_EXTRA_VARIANTS = {
    1: ["ek", "1"],
    2: ["do"],
    3: ["teen", "tin"],
    4: ["char"],
    5: ["paanch", "panch", "paach"],
    6: ["chhe", "che", "chhah", "chah", "cheh"],
    10: ["dus"],
    11: ["gyarah", "gyaarah", "gyara"],
    12: ["baarah", "barah", "bara"],
    13: ["terah", "tera", "terh"],
    14: ["chaudah", "chauda", "choudah"],
    15: ["pandrah", "pandra", "pandreh"],
    16: ["solah", "sola", "soleh"],
    17: ["satrah", "satra", "satreh"],
    18: ["atharah", "athara", "attharah", "atarah", "athaarah"],
    19: ["unnees", "unnis", "unees"],
    20: ["bees", "bis", "biss"],
    21: ["ikkis", "ikkees", "ekkis"],
    22: ["baais", "bais", "bayees"],
    25: ["pachchis", "pachees", "pachchees", "pachis"],
    26: ["chhabees"],
    27: ["sattais", "sattaees", "satais"],
    29: ["untis", "unattis", "unnatis", "untees", "unatees"],
    30: ["tees", "tis"],
    31: ["iktis", "ektis", "ikattis"],
    35: ["paintis", "paintees", "pentis", "pantis"],
    37: ["saintis", "saitis", "saintees"],
    38: ["adtis", "artis", "athtis"],
    39: ["untalis", "unchalis", "untaalis"],
    41: ["iktalis", "ektalis", "iktaalis"],
    42: ["bayalis", "bayaalis", "byalis"],
    45: ["paintalis", "pentalis", "paintaalis"],
    47: ["saintalis", "saitalis"],
    48: ["adtalis", "artalis", "athtalis"],
    49: ["unchas", "unnchas", "unanchas"],
    50: ["pachas", "pachaas"],
    51: ["ikyavan", "ikkyavan", "ekyavan"],
    52: ["bavan", "baavan", "bawan"],
    53: ["tirpan", "trepan", "tirepan"],
    54: ["chauwan", "chauvan", "chawwan"],
    55: ["pachpan", "pachpann"],
    56: ["chhappan", "chappan", "chhapan"],
    57: ["sattavan", "satavan", "sattawan"],
    58: ["atthavan", "athavan", "atthawan"],
    59: ["unsath", "unsatth", "unnsath"],
    60: ["saath", "sath"],
    61: ["iksath", "eksath", "ikhsath"],
    62: ["basath", "baasath", "baseth"],
    63: ["tirsath", "tresath", "tirseth"],
    64: ["chaunsath", "chausath", "chonsath"],
    65: ["painsath", "pensath", "painsatth"],
    66: [
        "chhiyasath", "chiyasath", "chhiasath",
        # user-reported colloquial spellings, not attested in Dakshina
        "chanchat", "chanchatt", "chaachat", "chhachat", "chhasath",
        "chansath", "chhiyasat", "chiyasat",
    ],
    67: ["sadsath", "sarsath", "sarhsath"],
    68: ["adsath", "arsath", "athsath"],
    69: ["unhattar", "unhatar", "unnhattar"],
    71: ["ikhattar", "ikhatar", "ekhattar"],
    72: ["bahattar", "bahatar"],
    73: ["tihattar", "tihatar", "tehattar"],
    74: ["chauhattar", "chauhatar", "chohattar"],
    75: ["pachhattar", "pachattar", "pachhatar"],
    76: ["chhihattar", "chihattar", "chhiyattar", "chihttr"],
    77: ["sathattar", "satattar", "sattahattar"],
    78: ["athhattar", "athattar", "atthattar"],
    79: ["unyasi", "unasi", "unnasi", "unnyasi", "unnaasi"],
    81: ["ikyasi", "ikkyasi", "ikyaasi", "ekyasi"],
    82: ["bayasi", "bayaasi", "byasi"],
    83: ["tirasi", "tiraasi", "terasi"],
    84: ["chaurasi", "chauraasi", "chorasi"],
    85: ["pachasi", "pachaasi", "pichasi"],
    86: ["chhiyasi", "chiyasi", "chhiasi"],
    87: ["sattasi", "sataasi", "satasi"],
    88: ["atthasi", "athasi", "atthaasi"],
    89: ["navasi", "nawasi", "navaasi", "nawaasi"],
    90: ["nabbe", "nabbey"],
    91: ["ikyanve", "ikkyanve", "ikyaanve", "ekyanve"],
    92: ["banve", "baanve", "banave"],
    93: ["tiranve", "tiraanve", "teranve"],
    94: ["chauranve", "chauraanve", "choranve"],
    95: ["pachanve", "pachaanve", "panchanve"],
    96: ["chhiyanve", "chiyanve", "chhianve"],
    97: ["sattanve", "sataanve", "satanve"],
    98: ["atthanve", "athanve", "atthaanve"],
    99: ["ninyanve", "ninnyanve", "ninanve", "ninyaanve"],
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
    # "lkah"/"lakhh" are curated typo spellings (see DATA_GRAMMAR §4 N7) kept
    # in the lexicon so they are always recognised inside a quantity context
    # ("do lkah") rather than only occasionally produced by random N7 noise.
    "UNIT_LAKH": ["lakh", "lac", "lack", "laakh", "lakhs", "lacs", "peti", "lkah", "lakhh"],
    "UNIT_MILLION": ["million", "mil", "millium"],
    # "croer"/"crorr" are curated typo spellings, same rationale as above.
    "UNIT_CRORE": [
        "crore", "karod", "karor", "karore", "crores", "karodh", "khokha",
        "croer", "crorr", "crode", "khokhaa",
    ],
    "UNIT_BILLION": ["billion"],
    "UNIT_ARAB": ["arab", "araba"],
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

DURATION_NOUNS = [
    "din", "dino", "ghante", "ghanta", "minute", "min", "second", "saal",
    "saalon", "mahine", "hafte", "baje", "tareekh", "baar",
]

INDEFINITE_PLURALS = [
    "lakhon", "lakho", "laakhon", "hazaaron", "hazaron", "karodon", "crodon",
    "croron", "sainkdon", "hajaron", "krodo", "karoron", "krodon", "arabon",
]

FILLER_WORDS = [
    # Hinglish chat / social
    "yaar", "bhai", "acha", "theek", "guzara", "salary", "offer", "negotiable",
    "urgent", "genuine", "buyer", "seller", "scooty", "laptop", "flat", "plot",
    "sofa", "event", "subscribers", "views", "package", "announce", "survey",
    "nuksaan", "loan", "bonus", "mahina", "mahine", "milte", "hain", "hota",
    "nahi", "already", "chuka", "hoon", "abhi", "kal", "aaj", "bhi", "toh",
    "sirf", "pura", "bilkul", "sach", "jhooth", "dekho", "suno", "batao",
    "kitna", "kitne", "kya", "kyun", "kaise", "kahan", "wala", "wali",
    "jaldi", "dhire", "sasta", "mehenga", "naya", "purana", "model",
    "condition", "warranty", "delivery", "location", "contact", "whatsapp",
    "call", "dm", "pls", "plz", "bro", "dude", "sir", "madam", "ji",
    "matlab", "waise", "vaise", "chalo", "chal", "arre", "are", "oye",
    "seriously", "sach mein", "sahi", "galat", "problem", "issue", "solve",
    "hogaya", "karna", "karo", "karte", "hoga", "hogi", "milega", "milegi",
    "chahiye", "chahiyega", "dena", "lena", "diya", "liya", "bola", "bolo",
    "suna", "dekha", "dikha", "dikhao", "pasand", "accha", "bekaar",
    "zabardast", "mast", "shaandaar", "bindaas", "faltu", "waste", "time",
    "jaana", "aana", "gaya", "aaya", "jayenge", "ayenge", "hoga na",
    "sahi hai", "koi baat nahi", "dekhte hain", "sochta hoon", "samajh",
    "gaya", "clear", "confirm", "pakka", "shayad", "maybe", "definitely",
    "surely", "obviously", "honestly", "basically", "actually", "literally",
    "exactly", "totally", "fully", "half", "part", "extra", "less", "more",
    "zyada", "kam", "bahut", "thoda", "thodi", "bohot", "kaafi", "itna",
    "itni", "jitna", "jitni", "sab", "sabko", "sabse", "koi", "kisi",
    "kuch", "kuchh", "sabko", "hume", "humein", "unko", "usko", "mujhe",
    "tumhe", "apko", "aapko", "unhe", "hamara", "tumhara", "unka", "mera",
    "meri", "tera", "teri", "uska", "uski", "iska", "iski", "yeh", "woh",
    "ye", "wo", "is", "us", "ye sab", "wo sab", "sabhi", "kai", "bohot saare",
    "log", "logon", "insaan", "banda", "aadmi", "aurat", "ladka", "ladki",
    "dost", "yaaron", "family", "ghar", "office", "school", "college",
    "university", "company", "business", "startup", "shop", "dukaan",
    "market", "bazaar", "showroom", "outlet", "branch", "store", "mall",
    "product", "item", "goods", "stock", "inventory", "order", "booking",
    "advance", "payment", "cash", "online", "transfer", "neft", "upi",
    "gpay", "phonepe", "paytm", "bank", "account", "cheque", "draft",
    "invoice", "bill", "receipt", "tax", "gst", "discount", "sale",
    "clearance", "festival", "diwali", "holi", "eid", "christmas",
    "newyear", "wedding", "shaadi", "party", "function", "ceremony",
    "birthday", "anniversary", "celebration", "guest", "invite", "invitation",
    "card", "gift", "present", "surprise", "special", "regular", "daily",
    "weekly", "monthly", "yearly", "annual", "quarterly", "season",
    "summer", "winter", "monsoon", "rain", "sun", "hot", "cold", "weather",
    "traffic", "signal", "road", "highway", "bridge", "station", "airport",
    "train", "bus", "auto", "rickshaw", "taxi", "cab", "uber", "ola",
    "bike", "scooter", "car", "vehicle", "petrol", "diesel", "mileage",
    "service", "repair", "mechanic", "garage", "showroom", "insurance",
    "policy", "premium", "claim", "renewal", "expiry", "valid", "invalid",
    "document", "papers", "proof", "id", "aadhar", "pan", "license",
    "registration", "number", "plate", "address", "pincode", "city",
    "state", "country", "area", "sector", "colony", "society", "building",
    "apartment", "room", "hall", "kitchen", "bathroom", "balcony", "parking",
    "lift", "floor", "square", "feet", "sqft", "acre", "bigha", "carpet",
    "furnished", "unfurnished", "semi", "fully", "vacant", "occupied",
    "tenant", "owner", "broker", "agent", "commission", "deposit", "rent",
    "lease", "agreement", "contract", "clause", "terms", "conditions",
    "policy", "rule", "regulation", "guideline", "instruction", "manual",
    "guide", "help", "support", "assistance", "query", "question", "answer",
    "doubt", "clarify", "clarification", "explain", "explanation", "example",
    "sample", "demo", "trial", "test", "testing", "checking", "verify",
    "verification", "validation", "approve", "approval", "reject",
    "rejection", "accept", "acceptance", "decline", "cancel", "cancellation",
    "refund", "return", "exchange", "replace", "replacement", "warranty",
    "guarantee", "assurance", "promise", "commitment", "trust", "reliable",
    "trustworthy", "honest", "fair", "reasonable", "affordable", "budget",
    "friendly", "convenient", "comfortable", "spacious", "compact", "small",
    "big", "large", "huge", "tiny", "medium", "average", "standard",
    "premium", "luxury", "basic", "advance", "advanced", "modern", "old",
    "used", "fresh", "brand", "branded", "local", "imported", "foreign",
    "domestic", "national", "international", "global", "worldwide",
    "everywhere", "somewhere", "anywhere", "nowhere", "here", "there",
    "everywhere", "nearby", "far", "close", "distance", "kilometer", "meter",
    "the", "and", "for", "with", "only", "best", "deal", "price", "rate",
    "cost", "total", "amount", "per", "month", "year", "week", "day",
    "please", "thanks", "thank you", "welcome", "sorry", "excuse me",
    "hello", "hi", "bye", "goodbye", "see you", "take care", "regards",
]
FILLER_WORDS = list(dict.fromkeys(FILLER_WORDS))

CURRENCY_MARKERS_BEFORE = ["₹", "Rs", "Rs.", "rs", "INR", "Re"]
CURRENCY_WORDS_AFTER = ["rupaye", "rupay", "rupees", "rupiya", "rupya", "/-"]
APPROXIMATORS = ["karib", "kareeb", "lagbhag", "takreeban", "around", "approx", "about", "roughly", "~", "koi", "kuch"]
RANGE_CONNECTORS = [" - ", "-", " to ", " se ", " ya ", " / ", " or "]

# joins TWO/THREE independent quantity spans in one text ("sava lakh aur
# dedh lakh", "50k ya 60k"). Distinct from RANGE_CONNECTORS: "se lekar ... tak"
# is a RANGE (one span, two amounts) and is never used here.
CONJ_CONNECTORS = [" aur ", " ya ", " ya phir ", ", ", " nahi to "]

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
        "{P1} mein {P2} flat, har flat alag price",
        "society mein {P1} ghar hain, average {P2} ka",
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
    "short_context": [
        "{P} only",
        "{P} ka",
        "price {P}",
        "{P} final",
        "{C}{P}",
        "budget {P}",
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
        "OTP 4521 hai tumhara",
        "2019 mein shuru hua tha",
        "pin code 110001 daalna",
        "OTP 8823 aaya hai check karo",
        "2024 mein plan hai humara",
        "rupaye chahiye abhi",
        "paise nahi hain mere paas",
        "paisa nahi hai bilkul",
        "flat #1 dekh liya tha",
        "item #12 pasand aaya",
        "room number 4 mein hai",
        "flat number 4 khali hai",
        "seat #23 book kar di",
        "order #45 aa gaya",
        "table number 9 reserve hai",
        # -pati / indefinite-plural hard negatives (§(c)): net still tags these
        "lakhpati bann gaya wo",
        "karodpati keh diya sabne",
        "crore-pati nikla akhir",
        "laakhon mein bik gaya",
        "karodon ka nuksaan hua",
        # unit-noun abbreviations after digits, not scale units
        "10 km door hai ghar",
        "5 kg aata chahiye",
        "100 kmph se gaya",
        "10kb ki file hai",
        "5mb ka video hai",
        # ordinals
        "1st prize jeeta usne",
        "2nd floor pe rehta hoon",
        "3rd prize mila mujhe",
        "4th floor tak jao",
        # bare id-like numbers
        "route 66 pe gaya tha",
        "mobile 98765 43210 hai mera",
        "flat no 302 hai mera",
        # duration/time phrases that use prefix words - must stay O, not a span
        "sava ghanta lag gaya",
        "dedh ghante mein aa jaunga",
        "saadhe teen baje milte hain",
        "paune paanch baje nikalna",
    ],
}


def _hi_latn_noise(word: str, rng) -> str:
    return NL.apply_noise(word, rng, apply_casing=True, apply_plural=False)


BLOCKED_SURFACES = [
    # place/person names that collide with a number word after noising
    "pune",
    # real-text proper nouns built on number words (data_wild/REPORT.md §4.1)
    "hazare", "hajare", "sawai", "savai",
]

# Lexicon surfaces that are also ordinary Hinglish/English words. See
# LanguagePack.ambiguous_forms / R12. Every entry here IS a real form in
# LEXICON/SYMBOL_UNITS above -- this list only says "needs an independently
# justified number word beside it to count".
AMBIGUOUS_FORMS = [
    "k",        # UNIT_HAZAAR symbol, but also the ke/ki clitic
    "so",       # UNIT_SAU, but also English "so"
    "sath", "saath",   # CARD_60, but also "saath" = with
    "sade", "sad",     # PFX_SAADHE, but also "sad"/"sade" (rotten)
    "peti",     # UNIT_LAKH slang, but also "peti" = box/belt
    "mil",      # UNIT_MILLION, but also "mil" = mile / "milna" stem
    "kharab",   # UNIT_KHARAB, but also "kharab" = bad/broken
    "arab",     # UNIT_ARAB, but also the ethnonym/place
    "char",     # CARD_4, but also English "char"
    "bees",     # CARD_20, but also English "bees"
    "lac",      # UNIT_LAKH symbol, but also English "lac"/"lack"
    "l", "m", "b",     # single-letter symbol units
]

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
    ambiguous_forms=AMBIGUOUS_FORMS,
    filler_words=FILLER_WORDS,
    duration_nouns=DURATION_NOUNS,
    conj_connectors=CONJ_CONNECTORS,
    ambiguous_units=["kharab", "mil"],
)

# drop any filler word that collides with a real vocabulary surface form
# (e.g. "half" is also an english-fraction-phrase form).
_forms_for_filter = PACK.all_forms()
PACK.filler_words = [
    w for w in PACK.filler_words
    if w.lower() not in _forms_for_filter and w.lower() not in BLOCKED_SURFACES
]
PACK.duration_nouns = [
    w for w in PACK.duration_nouns
    if w.lower() not in _forms_for_filter and w.lower() not in BLOCKED_SURFACES
]

base.register(PACK)
