"""Gujarati language pack in the Gujarati script (gu_gujr).

Mirrors mr_deva's structure. Four things are specific to Gujarati and are
expressed through generic LanguagePack hooks rather than one-off code:

1. HUNDREDS. Gujarati glues the hundreds word onto the number before it
   with no space (બારસો, અઢારસો, ત્રણસો, દોઢસો, અઢીસો, સાડાત્રણસો), so
   "સો" is listed in `glue_unit_forms`. Unlike Marathi's "शे" it is ALSO
   an ordinary free word ("સો રૂપિયા" = 100), so it is additionally listed
   in `glue_unit_forms_standalone`: the generator glues it where the shape
   allows and otherwise leaves it standing alone, and the tokenizer
   accepts both.

2. બસો (200), છસ્સો (600) -- the irregular hundreds. THE LABELLING
   DECISION. 200 is બસો, never *બેસો: in the bound position CARD_2 is "બ",
   not "બે". 600 is written both છસો and છસ્સો, the latter geminating the
   sibilant across the boundary. Rather than give UNIT_SAU extra surface
   forms (which would let *બાવીસસ્સો be generated and would make a bare
   "સ્સો" a unit word), both are modelled as BOUND CARD forms in
   `bound_number_forms`: "બ" -> CARD_2 and "છસ્" -> CARD_6, each declared
   valid only immediately before "સો". Consequences, all enforced by
   tests in tests/test_gu.py:

     * the generator emits them ONLY glued -- the swap happens inside
       generator._apply_pack_glue after the separator has been deleted, so
       a standalone or space-separated "બ" cannot be produced;
     * `llm_corpus._word_tokens` consults `bound_number_map(before="સો")`
       only in the glue-split branch, so "બસો" splits to CARD_2 + UNIT_SAU
       while a lone "બ" stays an unknown token;
     * they are deliberately NOT in `all_forms()`, so they never enter the
       cross-pack collision map nor the verifier's context-free surface
       table. `export_lexicon` writes them to a separate `bound_forms`
       section, and both verifiers accept one only when the NEXT token's
       text is "સો". A standalone ("CARD_2", "બ") never verifies.

   The class is unchanged by the swap, so char labels and core.evaluate()
   are identical either way: બસો = CARD_2 + UNIT_SAU = 200.

3. FUSED PREFIX. સાડા is usually written apart from its cardinal
   ("સાડા ત્રણ લાખ") but is also seen fused ("સાડાત્રણ"), so it is in
   `fused_prefix_forms` at a low probability. પોણા keeps its space in
   every sample seen and is not fused.

4. CASE SUFFIXES. Gujarati postpositions attach directly to a unit or
   cardinal word (હજારનું, લાખની, કરોડનો, હજારમાં, પચાસમાં, હજારેય).
   `word_suffixes` (plus `word_oblique_endings`) let the corpus tokenizer
   strip them to find the head word; the suffix characters stay part of
   that word's token, so the whole word carries the unit/cardinal class.

KNOWN DECODE CAVEAT (no Gujarati weights exist yet). `decode._is_bound_unit_tail`
only preserves a word-final unit sub-run when the CARD head before it is at
least 2 characters long. "બ" and "છસ્"'s head evidence is 1 and 3 chars, so
once weights are trained, બસો may decode as a single UNIT_SAU run (100
instead of 200) while છસ્સો is fine. Relaxing that length bound is a
decode-tuning question that belongs with the retrain, since loosening it
would also change how the three shipped languages smooth stray one-char
mispredictions; it is deliberately not changed here.
"""
from __future__ import annotations

from . import base
from .. import noise_gujr as NG

# --- 1..99 canonical Gujarati cardinals ----------------------------------
# Transcribed from data_llm/prompts/lexicon_notes_mr_gu.md (Gujarati section).
_BASE99 = [
    "એક", "બે", "ત્રણ", "ચાર", "પાંચ", "છ", "સાત", "આઠ", "નવ", "દસ",
    "અગિયાર", "બાર", "તેર", "ચૌદ", "પંદર", "સોળ", "સત્તર", "અઢાર", "ઓગણીસ", "વીસ",
    "એકવીસ", "બાવીસ", "તેવીસ", "ચોવીસ", "પચ્ચીસ", "છવ્વીસ", "સત્તાવીસ", "અઠ્ઠાવીસ", "ઓગણત્રીસ", "ત્રીસ",
    "એકત્રીસ", "બત્રીસ", "તેત્રીસ", "ચોત્રીસ", "પાંત્રીસ", "છત્રીસ", "સડત્રીસ", "અડત્રીસ", "ઓગણચાલીસ", "ચાલીસ",
    "એકતાલીસ", "બેતાલીસ", "ત્રેતાલીસ", "ચુમાલીસ", "પિસ્તાલીસ", "છેતાલીસ", "સુડતાલીસ", "અડતાલીસ", "ઓગણપચાસ", "પચાસ",
    "એકાવન", "બાવન", "ત્રેપન", "ચોપન", "પંચાવન", "છપ્પન", "સત્તાવન", "અઠ્ઠાવન", "ઓગણસાઠ", "સાઠ",
    "એકસઠ", "બાસઠ", "ત્રેસઠ", "ચોસઠ", "પાંસઠ", "છાસઠ", "સડસઠ", "અડસઠ", "ઓગણસિત્તેર", "સિત્તેર",
    "એકોતેર", "બોતેર", "તોતેર", "ચુમોતેર", "પંચોતેર", "છોતેર", "સિત્યોતેર", "ઇઠ્યોતેર", "ઓગણ્યાસી", "એંસી",
    "એક્યાસી", "બ્યાસી", "ત્યાસી", "ચોર્યાસી", "પંચ્યાસી", "છ્યાસી", "સિત્યાસી", "ઈઠ્યાસી", "નેવ્યાસી", "નેવું",
    "એકાણું", "બાણું", "ત્રાણું", "ચોરાણું", "પંચાણું", "છન્નું", "સત્તાણું", "અઠ્ઠાણું", "નવ્વાણું",
]
assert len(_BASE99) == 99

CARD_WORDS = {i + 1: [w] for i, w in enumerate(_BASE99)}

# Common phone-typed spelling variants. Checked for collisions against every
# other class by tests/test_gu.py (no variant may mean two different things).
# The recurring families are:
#   * anusvara dropped on phones (નેવું / નેવુ, પંચાણું / પંચાણુ,
#     એંસી / અંસી) - by far the commonest
#   * anusvara ADDED where the standard spelling has none
#     (ચોસઠ / ચોંસઠ, બોતેર / બોંતેર, બેતાલીસ / બેંતાલીસ - all three in the
#     seed corpus)
#   * ી / િ in the -વીસ / -ત્રીસ / -તાલીસ families (બાવીસ / બાવિસ)
#   * doubled-consonant simplification across the virama
#     (પચ્ચીસ / પચીસ, છવ્વીસ / છવીસ, અઠ્ઠાવીસ / અઠાવીસ)
#   * ઇ / ઈ in the ઇઠ્યોતેર / ઈઠ્યાસી pair, which the sources spell both ways
_EXTRA_VARIANTS = {
    5: ["પાચ"],
    10: ["દશ"],
    11: ["અગીયાર"],
    14: ["ચોદ"],
    16: ["સોલ"],
    17: ["સતર"],
    19: ["ઓગણિસ", "ઓગણીશ"],
    20: ["વિસ"],
    21: ["એકવિસ"],
    22: ["બાવિસ"],
    23: ["તેવિસ"],
    24: ["ચોવિસ"],
    25: ["પચીસ", "પચ્ચિસ"],
    26: ["છવીસ", "છવ્વિસ"],
    27: ["સત્તાવિસ", "સતાવીસ"],
    28: ["અઠાવીસ", "અઠ્ઠાવિસ"],
    29: ["ઓગણત્રિસ"],
    30: ["ત્રિસ"],
    35: ["પસ્તીસ", "પાંત્રિસ"],
    37: ["સડત્રિસ", "સાડત્રીસ"],
    39: ["ઓગણચાલિસ"],
    40: ["ચાળીસ", "ચાલિસ"],
    41: ["એકતાલિસ"],
    42: ["બેંતાલીસ", "બેતાલિસ"],
    43: ["ત્રેતાલિસ"],
    44: ["ચુમ્માલીસ", "ચુમાલિસ"],
    45: ["પીસ્તાલીસ", "પિસ્તાલિસ"],
    46: ["છેતાલિસ"],
    47: ["સડતાલીસ", "સુડતાલિસ"],
    48: ["અડતાલિસ"],
    49: ["ઓગણપંચાસ"],
    51: ["એક્કાવન"],
    56: ["છપન"],
    57: ["સતાવન"],
    58: ["અઠાવન"],
    64: ["ચોંસઠ"],
    66: ["છાંસઠ"],
    69: ["ઓગણસીત્તેર"],
    70: ["સીત્તેર"],
    71: ["એકોંતેર"],
    72: ["બોંતેર"],
    73: ["તોંતેર"],
    74: ["ચુમ્મોતેર"],
    75: ["પંચ્યોતેર"],
    76: ["છોંતેર"],
    77: ["સીત્યોતેર"],
    78: ["ઈઠ્યોતેર", "અઠ્યોતેર"],
    79: ["ઓગણ્યાંસી"],
    80: ["અંસી", "એંશી", "એસી"],
    81: ["એક્યાંસી"],
    82: ["બ્યાંસી"],
    83: ["ત્યાંસી"],
    84: ["ચોર્યાંસી", "ચોરાસી"],
    85: ["પંચ્યાંસી", "પચ્યાસી"],
    86: ["છ્યાંસી", "છયાસી"],
    87: ["સીત્યાસી"],
    88: ["ઇઠ્યાસી", "અઠ્યાસી"],
    89: ["નેવ્યાંસી"],
    90: ["નેવુ"],
    91: ["એકાણુ"],
    92: ["બાણુ"],
    93: ["ત્રાણુ"],
    94: ["ચોરાણુ"],
    95: ["પંચાણુ"],
    96: ["છન્નુ", "છનું"],
    97: ["સત્તાણુ", "સતાણું"],
    98: ["અઠ્ઠાણુ", "અઠાણું"],
    99: ["નવ્વાણુ", "નવાણું"],
}
for n, variants in _EXTRA_VARIANTS.items():
    CARD_WORDS[n] = list(dict.fromkeys(CARD_WORDS[n] + [v for v in variants if v]))

LEXICON = {}
for n in range(1, 100):
    LEXICON[f"CARD_{n}"] = CARD_WORDS[n]

LEXICON.update({
    "PFX_PAAV": ["પા"],
    "PFX_AADHA": ["અડધો", "અડધા", "અડધી", "અડધું"],
    "PFX_PAUNE": ["પોણા", "પોણો", "પોણી", "પોણું"],
    "PFX_SAVA": ["સવા"],
    "PFX_DEDH": ["દોઢ"],
    "PFX_DHAI": ["અઢી"],
    "PFX_SAADHE": ["સાડા", "સાડાં"],

    # "સો" is both the free hundred word ("સો રૂપિયા" = 100) and the form
    # that glues onto the number before it (બારસો, ત્રણસો) - see
    # GLUE_UNIT_FORMS / GLUE_UNIT_FORMS_STANDALONE below.
    "UNIT_SAU": ["સો"],
    "UNIT_HAZAAR": ["હજાર", "હઝાર"],
    "UNIT_LAKH": ["લાખ"],
    "UNIT_MILLION": ["મિલિયન"],
    "UNIT_CRORE": ["કરોડ", "ક્રોડ"],
    "UNIT_BILLION": ["બિલિયન"],
    "UNIT_ARAB": ["અબજ", "અરબ"],
    "UNIT_KHARAB": ["ખર્વ", "ખરબ"],
})

# Latin symbol units are typed after digits in Gujarati lines exactly as in
# Hindi and Marathi ones (2.5L, 20k, 1.5cr, 12 LPA). These Latin surfaces
# are shared with the other packs and must agree with them class-for-class
# (tests/test_gu.py::test_no_cross_pack_conflicts).
SYMBOL_UNITS = {
    "UNIT_HAZAAR": ["k", "K"],
    "UNIT_LAKH": ["L", "l", "lac", "LPA", "lpa"],
    "UNIT_MILLION": ["M"],
    "UNIT_CRORE": ["cr", "Cr", "CR"],
    "UNIT_BILLION": ["bn"],
}

ENGLISH_FRACTION_PHRASES = {}  # none for gu_gujr

# The hundreds word glues onto the number before it (બારસો, અઢારસો)...
GLUE_UNIT_FORMS = ["સો"]
# ... but unlike Marathi's "शे" it is equally a free word on its own.
GLUE_UNIT_FORMS_STANDALONE = ["સો"]

# The two irregular hundreds. See point 2 of the module docstring: these
# surfaces exist ONLY immediately before "સો" and are kept out of
# all_forms() so that a standalone "બ" can never resolve or verify.
BOUND_NUMBER_FORMS = {
    # 200 is બસો, never *બેસો - the bound form of CARD_2 is "બ".
    "બ": {"cls": "CARD_2", "before": ["સો"], "p": 0.9},
    # 600 is written છસો and છસ્સો; the geminated spelling puts the extra
    # sibilant on the cardinal side of the boundary.
    "છસ્": {"cls": "CARD_6", "before": ["સો"], "p": 0.5},
}

# સાડા is usually written apart ("સાડા ત્રણ લાખ") but also fused
# ("સાડાત્રણ"). પોણા keeps its space in every sample seen, so it is not
# listed. Keyed by class so a noised spelling of સાડા fuses too.
FUSED_PREFIX_FORMS = {"PFX_SAADHE": 0.3}

# Gujarati postpositions that attach directly to a unit/cardinal word
# ("હજારનું", "લાખની", "કરોડનો", "હજારમાં", "પચાસમાં", "હજારેય"). Longest
# first; the tokenizer strips them repeatedly.
#
# "ના" is deliberately NOT here. It would strip "સોનાની"/"સોનાનો" (of
# gold) down to "સો" and report three genuine negatives as quantities,
# and the only unit form that needs it in the whole corpus is
# "રૂપિયાના", which CURRENCY_WORDS_AFTER already lists in full.
WORD_SUFFIXES = [
    "સુધીનું", "સુધીની", "સુધીનો", "નુંય", "માંય", "નું", "ની", "નો",
    "ને", "માં", "થી", "એ", "ેય", "ે", "ય", "જ",
]

# Gujarati postpositions attach to the DIRECT stem of every unit and
# cardinal this pack cares about (હજાર + નું, લાખ + ની, કરોડ + નો), so
# unlike Marathi there is no oblique stem to try. Deliberately empty:
# allowing an "-ા" oblique made "સોનાની"/"સોનાનો" (of gold) strip down to
# "સો" and flag three perfectly good negatives as containing a quantity.
WORD_OBLIQUE_ENDINGS = []

DURATION_NOUNS = [
    "દિવસ", "દિવસમાં", "કલાક", "કલાકમાં", "મિનિટ", "મિનિટમાં", "વર્ષ",
    "વરસ", "મહિના", "મહિનામાં", "અઠવાડિયા", "વાગ્યે", "તારીખે", "વખત",
    "કિલોમીટર", "કિલો", "જણ", "માણસો", "ટકા",
]

INDEFINITE_PLURALS = [
    "લાખો", "લાખોનું", "લાખોની", "હજારો", "હજારોનું", "કરોડો", "કરોડોનું",
    "કરોડોની", "લખપતિ", "કરોડપતિ", "અબજોપતિ", "સેંકડો",
]

FILLER_WORDS = [
    "અરે", "યાર", "ભાઈ", "ભાઉ", "બહેન", "કાકા", "મામા", "સારું", "બરાબર",
    "નક્કી", "ખરેખર", "એટલે", "એવું", "હવે", "કાલે", "ગઈકાલે", "આજે",
    "ફક્ત", "આખું", "નવું", "જૂની", "જૂનો", "સસ્તું", "મોંઘું", "કિંમત",
    "ભાવ", "બજેટ", "પગાર", "ઓફર", "ફ્લેટ", "ગાડી", "ફોન", "લેપટોપ",
    "જમીન", "ઘર", "ભાડું", "લોન", "બોનસ", "પેકેજ", "કંપની", "સરકાર",
    "બજાર", "દુકાન", "લોકો", "લોકોએ", "મિત્ર", "પરિવાર", "ઓફિસ", "શાળા",
    "કોલેજ", "લગ્ન", "પાર્ટી", "તહેવાર", "દિવાળી", "નવરાત્રી", "રજા",
    "ટિકિટ", "બુકિંગ", "એડવાન્સ", "પેમેન્ટ", "કેશ", "ઓનલાઈન", "ટ્રાન્સફર",
    "બેંક", "એકાઉન્ટ", "ચેક", "બિલ", "ટેક્સ", "ડિસ્કાઉન્ટ", "સેલ", "સ્ટોક",
    "ઓર્ડર", "ડિલિવરી", "ક્વોલિટી", "વોરંટી", "ગેરંટી", "સર્વિસ",
    "રિપેર", "ઇન્શ્યોરન્સ", "પોલિસી", "પ્રીમિયમ", "ક્લેમ", "ડોક્યુમેન્ટ",
    "આધાર", "પાન", "લાઇસન્સ", "સરનામું", "પિનકોડ", "શહેર", "ગામ",
    "વિસ્તાર", "સોસાયટી", "બિલ્ડિંગ", "રૂમ", "કિચન", "બાલ્કની", "પાર્કિંગ",
    "લિફ્ટ", "માળ", "ફર્નિચર", "માલિક", "ભાડૂત", "બ્રોકર", "એજન્ટ",
    "કમિશન", "ડિપોઝિટ", "કરાર", "શરત", "નિયમ", "મદદ", "સપોર્ટ", "સવાલ",
    "જવાબ", "શંકા", "ઉદાહરણ", "ટેસ્ટ", "ચેકિંગ", "એપ્રૂવલ", "રિજેક્ટ",
    "કેન્સલ", "રિફંડ", "રિટર્ન", "એક્સચેન્જ", "વિશ્વાસ", "પ્રામાણિક",
    "પોસાય", "મોટું", "મોટો", "નાનું", "નાનો", "મીડિયમ", "સ્ટાન્ડર્ડ",
    "લક્ઝરી", "બેઝિક", "મોડર્ન", "લોકલ", "ઇમ્પોર્ટેડ", "નજીક", "દૂર",
    "કેટલું", "કેટલામાં", "શું", "કેવું", "ક્યાં", "જલદી", "ધીમે",
    "મોડેલ", "કન્ડિશન", "લોકેશન", "કોન્ટેક્ટ", "વોટ્સએપ", "કોલ", "સર",
    "મેડમ", "ચાલો", "જો", "જુઓ", "કહે", "કહો", "સાંભળ", "સાંભળો",
    "ગમ્યું", "મસ્ત", "જબરદસ્ત", "બેકાર", "સમય", "જશે", "આવશે", "ગયો",
    "આવ્યો", "થશે", "મળશે", "જોઈએ", "આપો", "લો", "આપ્યું", "લીધું",
    "બોલ્યો", "બોલો", "સાંભળ્યું", "જોયું", "બતાવો", "કદાચ", "વધારે",
    "ઓછું", "ખૂબ", "થોડું", "થોડો", "બધું", "બધા", "કોઈ", "કંઈક",
    "અમને", "તેમને", "મને", "તને", "આપણને", "અમારો", "તમારો", "તેમનો",
    "મારો", "મારી", "તારો", "તારી", "તેનો", "તેની", "આ", "તે", "છે",
    "હતું", "હતો", "નથી", "પણ", "એટલા", "તોય", "પણે",
    "budget", "salary", "offer", "final", "ok", "bro", "deal", "rent",
]
FILLER_WORDS = list(dict.fromkeys(FILLER_WORDS))

CURRENCY_MARKERS_BEFORE = ["₹", "રૂ", "રૂ.", "Rs", "Rs.", "INR"]
CURRENCY_WORDS_AFTER = [
    # "રૂપિયા" is the stem every case-inflected form starts with
    # (રૂપિયાનું, રૂપિયાની, રૂપિયામાં, રૂપિયાથી ...), so listing it lets
    # both core.detect_currency and the corpus tokenizer handle the family.
    "રૂપિયા", "રૂપિયો", "રૂપિયાનું", "રૂપિયાની", "રૂપિયાનો", "રૂપિયાના",
    "રૂપિયામાં", "રૂપિયાથી", "રૂપિયે", "રૂપિયા/-",
]
APPROXIMATORS = ["લગભગ", "આશરે", "અંદાજે", "આસપાસ", "આશરેનું"]
RANGE_CONNECTORS = [" થી ", "-", " કે ", " to "]

# joins TWO/THREE independent quantity spans in one text ("સવા લાખ અને
# દોઢ લાખ"). Distinct from RANGE_CONNECTORS: "... થી ..." is a RANGE (one
# span, two amounts) and is never used here.
CONJ_CONNECTORS = [" અને ", " કે ", " અથવા ", ", "]

TEMPLATES = {
    "casual": [
        "અરે {A} {P} માં થઈ જશે કે",
        "એણે {C}{P} માગ્યા યાર",
        "{P} એટલે બહુ વધારે થઈ ગયું",
        "મારું બજેટ {P} સુધીનું જ છે",
        "{P} માં ડીલ થઈ ગઈ ભાઈ",
        "અત્યારે મારી પાસે {C}{P} જ છે",
        "કાલે {P} ઉધાર લીધા હતા",
        "આટલા {P} ક્યાંથી લાવું હું",
        "એને {P} જોઈતા હતા",
        "{A} {P} લાગશે આમાં",
        "મને {P} ઉછીના આપ ને",
        "પપ્પાએ {P} આપ્યા હતા",
        "લગ્નમાં {P} ખર્ચ થયો",
        "એ {P} માગે છે મારી પાસે",
        "{P} ભેગા કર્યા માંડ માંડ",
        "ભાઈએ {P} મોકલ્યા કાલે",
        "{P} ભર્યા પછી મળ્યું",
        "જો, {P} થી વધારે નહીં આપું",
        "{P} માં લીધી હતી ગાડી",
        "આટલામાં {P} ખર્ચ થઈ ગયા",
        "મિત્ર પાસેથી {P} લીધા ઉધાર",
        "{P} માં શું આવશે કહે",
        "મમ્મીને {P} મોકલવાના છે",
        "{P} બાકી છે મહિનાના અંતે",
        "ગયા વખતે {C}{P} આપ્યા હતા",
    ],
    "classifieds": [
        "કિંમત: {C}{P} વાટાઘાટ શક્ય",
        "2019 મોડેલ, {P} ફાઇનલ",
        "ભાડું {P} મહિનો, 2બીએચકે",
        "એમઆરપી {C}{P} ઓન્લી",
        "માગેલી કિંમત {C}{P}",
        "વેચવાનું છે {C}{P} ફિક્સ",
        "એકદમ નવું, {P} થી શરૂ",
        "અર્જન્ટ સેલ {C}{P} થોડી વાટાઘાટ",
        "ફ્લેટ {C}{P} માં, તરત શિફ્ટ",
        "બાઇક {P} ફક્ત, સિરિયસ બાયર",
        "{P} બેસ્ટ પ્રાઇસ ગેરંટી",
        "લેપટોપ {P} થોડું ઓછું થશે",
        "રિસેલ વેલ્યુ {C}{P}",
        "કેશ પ્રાઇસ {C}{P}",
        "છેલ્લી કિંમત {C}{P} ભાવતાલ નહીં",
        "સોફા સેટ {P} માં વેચવાનો છે",
        "પ્લોટ {C}{P} માં, ક્લિયર ટાઇટલ",
        "સ્કૂટી {P}, સિંગલ ઓનર",
        "જગ્યા {P} માં આપું છું, જોઈ જાવ",
        "ટીવી {C}{P}, બિલ સાથે છે",
        "ફ્રિજ {P} માં, બે વર્ષ વાપરેલું",
        "દુકાન ભાડે {P} મહિનો",
        "{P} માં આપું છું, ફોન કરો",
        "ગાડી {C}{P}, ઇન્શ્યોરન્સ છે",
        "રૂમ શેરિંગ {P} દરેકના",
    ],
    "news": [
        "{P} લોકોએ ભાગ લીધો",
        "સ્ટાર્ટઅપે {P} નું ભંડોળ ઊભું કર્યું",
        "{P} થી વધુ વ્યૂઝ આવ્યા",
        "સરકારે {P} નું પેકેજ જાહેર કર્યું",
        "કંપનીએ {P} ની ખોટ જણાવી",
        "વીડિયો {P} વખત જોવાયો",
        "આ વર્ષે {P} લોકોએ મતદાન કર્યું",
        "તેમણે {P} નું દાન આપ્યું",
        "વહીવટીતંત્રે {P} નું બજેટ રાખ્યું",
        "અહેવાલ મુજબ {P} ખર્ચ થયો",
        "યોજનામાં {P} ખર્ચ થશે",
        "{P} નું કૌભાંડ સામે આવ્યું",
        "મહાનગરપાલિકાએ {P} મંજૂર કર્યા",
        "પ્રોજેક્ટની કિંમત {P} થઈ",
        "{P} નું રોકાણ જાહેર",
        "ખેડૂતોને {P} ની મદદ મળી",
        "{P} નું ટર્નઓવર થયું આ વર્ષે",
        "મેટ્રો માટે {P} ખર્ચ અપેક્ષિત",
        "વિક્રમી {P} ભંડોળમાં જમા થયા",
        "{P} નું દેવું માફ કર્યું",
        "રસ્તા માટે {P} મંજૂર",
        "{P} નો નફો નોંધાવ્યો કંપનીએ",
        "બેંકે {P} ની લોન આપી",
        "{P} ની સહાય જાહેર કરી",
        "ચૂંટણીમાં {P} ખર્ચ થયાનો અંદાજ",
    ],
    "salary": [
        "સીટીસી {P} વાર્ષિક",
        "પેકેજ {P} છે",
        "ઈએમઆઈ {C}{P} મહિનો",
        "લોન {P} ની લીધી હતી",
        "પગાર {P} મહિનાનો છે",
        "ઇન હેન્ડ {P} મળે છે",
        "ઇન્ક્રીમેન્ટ પછી {P} થયો",
        "બોનસ {C}{P} મળ્યું",
        "સ્ટાઇપેન્ડ {P} મળે છે",
        "હાઇક {P} નો મળ્યો",
        "વાર્ષિક પેકેજ {P} છે",
        "મહિનાનો ખર્ચ {P} છે",
        "ભાડું {P} આપું છું હું",
        "ફી {P} લાગે છે",
        "ટેક્સ {C}{P} થાય છે",
        "ફ્રેશરને {P} મળે છે અહીં",
        "ઓફર લેટરમાં {P} લખ્યું છે",
        "પીએફ કાપીને {P} હાથમાં આવે છે",
        "{P} પર સેટલ થયો છેવટે",
        "નોટિસ પિરિયડના {P} કપાયા",
        "ઓવરટાઇમના {P} અલગ",
        "કોન્ટ્રાક્ટ પર {P} મળે છે",
        "ઇન્ટર્નશિપમાં {P} હતા",
        "{P} નું એપ્રાઇઝલ થયું",
        "સાઇડ ઇન્કમ {P} છે મહિને",
    ],
    "ranges": [
        "{P} માં આવી જશે",
        "{P} ની ડીલ છે",
        "{P} નો ફોન છે",
        "{P} ની વચ્ચે",
        "{P} સુધી જશે",
        "{P} લાગશે",
        "બજેટ {P} છે મારું",
        "રેન્જ {P} સુધીની છે",
        "કિંમત {P} ની વચ્ચે હશે",
        "{P} નો સ્કોપ છે",
        "ભાડું {P} સુધી થશે",
        "પગાર {P} મળશે",
        "પેકેજ {P} ની વચ્ચે",
        "ખર્ચ {P} સુધી",
        "{P} માં કંઈક જો",
        "{P} ની રેન્જમાં બતાવ",
        "{P} જેટલું તો લાગશે",
        "રોકાણ {P} નું કર",
        "{P} દરમિયાન ક્યાંક",
        "અંદાજ {P} નો છે",
        "{P} માં મળી જશે",
        "ટોટલ {P} થશે કદાચ",
        "{P} માં ખરીદી શકાય",
        "ફી {P} ની આસપાસ છે",
        "{P} જેટલો ખર્ચ અપેક્ષિત",
    ],
    "two_spans": [
        "{P1} થી વધીને {P2} થયું",
        "પહેલાં {P1} હતું હવે {P2}",
        "{P1} + {P2} = ?",
        "ગયા વર્ષે {P1} હતું આ વર્ષે {P2}",
        "{P1} થી {P2} સુધી પહોંચ્યું",
        "બજેટ {P1} હતું હવે {P2} થયું",
        "કાલે {P1} હતું આજે {P2}",
        "{P1} પૂરા થઈને {P2} બચ્યા",
        "એણે {P1} આપ્યા મેં {P2} લીધા",
        "{P1} રોક્યા {P2} કમાયા",
        "{P1} માં {P2} ફ્લેટ, દરેકની કિંમત અલગ",
        "સોસાયટીમાં {P1} ઘર છે, સરેરાશ {P2} ના",
        "{P1} પગાર હતો હવે {P2} છે",
        "{P1} ભર્યા {P2} બાકી છે",
        "એકના {P1} બીજાના {P2}",
        "{P1} ડાઉન પેમેન્ટ {P2} લોન",
        "{P1} ખર્ચ થયા {P2} બાકી",
        "{P1} ની ગાડી {P2} માં વેચી",
        "{P1} હતા તે {P2} થઈ ગયા જોતજોતામાં",
        "જૂનું {P1} નવું {P2}",
        "{P1} અને {P2} એમ બે હપ્તા",
        "{P1} માગે છે એ, મેં {P2} કહ્યું",
        "{P1} ટોટલ, {P2} એડવાન્સ",
        "ઓનલાઈન {P1} દુકાનમાં {P2}",
        "{P1} ની ઓફર હતી {P2} માં લીધું",
    ],
    "bare": [
        "{P}",
        "{A} {P}",
        "{C}{P}",
        "{P}!!",
        "{P}?",
        "{C}{P}/-",
        "{P} રૂપિયા",
        "ફક્ત {P}",
        "કુલ {P}",
        "લગભગ {P}",
    ],
    "short_context": [
        "{P} ઓન્લી",
        "{P} નો",
        "કિંમત {P}",
        "{P} ફાઇનલ",
        "{C}{P}",
        "બજેટ {P}",
    ],
    "negatives": [
        "રાજકોટમાં ફ્લેટ લેવો છે",
        "જયેશભાઈનો નંબર 9876543210 છે",
        "કેબીસીમાં કરોડપતિ થયો એ",
        "લાખો લોકો આવ્યા હતા",
        "બે દિવસમાં આવું છું",
        "પિન 380001 છે મારો",
        "2:30 વાગ્યે મળીએ",
        "15 ઓગસ્ટે રજા છે",
        "સોદો પાક્કો થઈ ગયો",
        "કરોડપતિ થવું છે મારે",
        "હજારો લોકો હતા ત્યાં",
        "9123456780 પર કોલ કર",
        "તારીખ 12/05 છે મીટિંગની",
        "ઓટીપી 4521 છે તારો",
        "2019 માં શરૂ થયું હતું",
        "પિન કોડ 395001 નાખ",
        "2024 માં પ્લાન છે અમારો",
        "રૂપિયા જોઈએ છે અત્યારે",
        "પૈસા નથી મારી પાસે",
        "ત્રણ દિવસ પછી કહું છું",
        "ફ્લેટ #1 જોયો હતો",
        "આઇટમ #12 ગમી",
        "ફ્લેટ નંબર 4 ખાલી છે",
        "રૂમ નંબર 9 રિઝર્વ છે",
        "ઓર્ડર #45 આવ્યો",
        "સીટ નંબર 23 બુક કરી",
        "લખપતિ થયો એ અચાનક",
        "કરોડપતિ થયો છેવટે",
        "કરોડોનું નુકસાન થયું",
        "લાખોની વાતો ના કર",
        "10 કિમી દૂર છે ઘર",
        "5 કિલો લોટ જોઈએ",
        "100 કિમી/કલાકની ઝડપે ગયો",
        "પહેલું ઇનામ મળ્યું એને",
        "બીજા માળે રહું છું હું",
        "ત્રીજો નંબર આવ્યો મને",
        "ચોથા માળ સુધી જા",
        "રૂટ 66 પર ગયો હતો",
        "મોબાઈલ 98765 43210 છે મારો",
        "ફ્લેટ નંબર 302 છે મારો",
        "સવા કલાક લાગ્યો",
        "દોઢ કલાકમાં આવું છું",
        "સાડા ત્રણ વાગ્યે મળીએ",
        "પોણા પાંચ વાગ્યે નીકળવાનું",
        "અઢી વર્ષ થયા અહીં",
        "ઉંમર 35 છે એની",
        "ચાર માણસો હતા ઘરમાં",
        "ગાડી નંબર GJ01 AB 1234",
        "સાત વાગ્યે ટ્રેન છે",
        "બારમામાં 92 ટકા આવ્યા",
        "દસમામાં 88 ટકા હતા",
    ],
}


def _gu_gujr_noise(word: str, rng) -> str:
    return NG.apply_noise(word, rng)


BLOCKED_SURFACES = ["રાજકોટ", "સુરત"]

PACK = base.LanguagePack(
    id="gu_gujr",
    script="gujr",
    lexicon=LEXICON,
    symbol_units=SYMBOL_UNITS,
    english_fraction_phrases=ENGLISH_FRACTION_PHRASES,
    indefinite_plurals=INDEFINITE_PLURALS,
    currency_markers_before=CURRENCY_MARKERS_BEFORE,
    currency_words_after=CURRENCY_WORDS_AFTER,
    approximators=APPROXIMATORS,
    range_connectors=RANGE_CONNECTORS,
    templates=TEMPLATES,
    noise_fn=_gu_gujr_noise,
    blocked_surfaces=BLOCKED_SURFACES,
    filler_words=FILLER_WORDS,
    duration_nouns=DURATION_NOUNS,
    conj_connectors=CONJ_CONNECTORS,
    glue_unit_forms=GLUE_UNIT_FORMS,
    glue_unit_forms_standalone=GLUE_UNIT_FORMS_STANDALONE,
    bound_number_forms=BOUND_NUMBER_FORMS,
    fused_prefix_forms=FUSED_PREFIX_FORMS,
    word_suffixes=WORD_SUFFIXES,
    word_oblique_endings=WORD_OBLIQUE_ENDINGS,
)

_forms_for_filter = PACK.all_forms()
PACK.filler_words = [
    w for w in PACK.filler_words
    if w.lower() not in _forms_for_filter and w.lower() not in BLOCKED_SURFACES
]
PACK.duration_nouns = [
    w for w in PACK.duration_nouns
    if w.lower() not in _forms_for_filter and w.lower() not in BLOCKED_SURFACES
]

# 25% of DIGITS tokens are rendered with this script's own digit glyphs.
PACK.native_digits = "૦૧૨૩૪૫૬૭૮૯"  # Gujarati U+0AE6-U+0AEF
PACK.native_digit_prob = 0.25

base.register(PACK)
