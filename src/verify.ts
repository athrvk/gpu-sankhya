// Verified-tier predicate: a span is "verified" when every (class, text)
// token it decoded to is independently justified by the lexicon (or, for
// SEP/DOT/COMMA/RANGE/DIGITS, by a structural rule). Mirrors
// python/sankhya/verify.py exactly -- see VERIFY_CONTRACT.md.
import lexiconJson from "./data/lexicon.json" with { type: "json" };

interface LexiconBoundForm {
  cls: string;
  before: string[];
}
interface LexiconPack {
  forms: Record<string, string>;
  range_words: string[];
  /** Surfaces that are ONLY valid immediately before one of `before`'s
   * unit surfaces (Gujarati CARD_2 "બ", which exists in બસો = 200 and
   * nowhere else). Kept out of `forms` on purpose -- a verifier that
   * merged them in would verify a standalone "બ" as 2. */
  bound_forms?: Record<string, LexiconBoundForm>;
  /** Case endings that attach directly to a unit/cardinal word in this
   * language ("लाखांचं", "કરોડનો"), and the oblique stem endings that may
   * sit under one ("लाख" -> "लाखां-"). Absent for packs with none. */
  word_suffixes?: string[];
  word_oblique_endings?: string[];
  /** Lexicon surfaces that are also ordinary words in this language
   * ("so", "sath", "arab", "अरब", "એક") -- R12. */
  ambiguous_forms?: string[];
  /** Surfaces that must never verify or decode as a number word: proper
   * nouns built on number words ("हजारे", "अरबी", "સવાઈ") -- R14. */
  blocked_surfaces?: string[];
}
interface LexiconJson {
  version: number;
  packs: Record<string, LexiconPack>;
}

const lexicon = lexiconJson as unknown as LexiconJson;

/** NFC + lowercase, the single normalisation every lexicon lookup uses --
 * mirrors python/sankhya/verify.py::_norm. */
function norm(s: string): string {
  return s.normalize("NFC").toLowerCase();
}

// Union of all packs' forms (surface -> class) and range_words.
const FORMS: Map<string, string> = new Map();
// surface -> EVERY class any pack gives it (FORMS keeps only the first, which
// is enough for verification but not for "is this surface O and nothing
// else?" -- see isLexiconOOnly).
const ALL_CLASSES: Map<string, Set<string>> = new Map();
const RANGE_WORDS: Set<string> = new Set();
// surface -> class -> the unit surfaces that surface may precede.
const BOUND_FORMS: Map<string, Map<string, Set<string>>> = new Map();
// Union of every pack's declared case endings / oblique stem endings.
const WORD_SUFFIXES: Set<string> = new Set();
const WORD_OBLIQUE_ENDINGS: Set<string> = new Set();
// R12/R14: surfaces that are also ordinary words, and surfaces that must
// never count as a number word at all.
const AMBIGUOUS_FORMS: Set<string> = new Set();
const BLOCKED_SURFACES: Set<string> = new Set();
/** Per-pack view of forms + that pack's OWN case endings. R15: a suffix
 * strip must stay inside one language -- a union of every pack's endings
 * lets Marathi morphology justify a Hindi-only head. Mirrors
 * python/sankhya/verify.py::pack_views. */
interface PackView {
  forms: Map<string, Set<string>>;
  suffixes: string[];
  obliques: string[];
}
const PACK_VIEWS: PackView[] = [];
for (const pack of Object.values(lexicon.packs)) {
  for (const [surface, cls] of Object.entries(pack.forms)) {
    if (!FORMS.has(surface)) FORMS.set(surface, cls);
    const key = norm(surface);
    let set = ALL_CLASSES.get(key);
    if (!set) ALL_CLASSES.set(key, (set = new Set()));
    set.add(cls);
  }
  for (const w of pack.range_words) RANGE_WORDS.add(w.toLowerCase());
  for (const [surface, spec] of Object.entries(pack.bound_forms ?? {})) {
    const key = surface.toLowerCase();
    let byClass = BOUND_FORMS.get(key);
    if (!byClass) BOUND_FORMS.set(key, (byClass = new Map()));
    let units = byClass.get(spec.cls);
    if (!units) byClass.set(spec.cls, (units = new Set()));
    for (const u of spec.before) units.add(u.toLowerCase());
  }
  for (const suf of pack.word_suffixes ?? []) WORD_SUFFIXES.add(norm(suf));
  for (const obl of pack.word_oblique_endings ?? []) WORD_OBLIQUE_ENDINGS.add(norm(obl));
  for (const a of pack.ambiguous_forms ?? []) AMBIGUOUS_FORMS.add(norm(a));
  for (const b of pack.blocked_surfaces ?? []) BLOCKED_SURFACES.add(norm(b));
}

const byLengthDesc = (a: string, b: string): number => b.length - a.length || (a < b ? -1 : a > b ? 1 : 0);

for (const pack of Object.values(lexicon.packs)) {
  const forms = new Map<string, Set<string>>();
  for (const [surface, cls] of Object.entries(pack.forms)) {
    const key = norm(surface);
    let set = forms.get(key);
    if (!set) forms.set(key, (set = new Set()));
    set.add(cls);
  }
  PACK_VIEWS.push({
    forms,
    suffixes: [...new Set((pack.word_suffixes ?? []).map(norm))].sort(byLengthDesc),
    obliques: [...new Set((pack.word_oblique_endings ?? []).map(norm))].sort(byLengthDesc),
  });
}

/** R15: the minimum length a suffix-stripped head must still have before it
 * may be accepted as an inflected lexicon form. Mirrors
 * python/sankhya/verify.py::MIN_SUFFIX_HEAD_LEN. */
const MIN_SUFFIX_HEAD_LEN = 3;
/** R16: ambiguous SYMBOL unit surfaces at most this long ("k", "l", "m",
 * "b") only count when glued straight onto their digits ("20k"). Mirrors
 * python/sankhya/verify.py::MAX_GLUED_SYMBOL_LEN. */
const MAX_GLUED_SYMBOL_LEN = 2;

/** A PFX_, CARD_ or UNIT_ token whose exact surface is unknown may still be
 * an inflected form of a known head: Marathi "लाखांचं" is UNIT_LAKH (लाख +
 * oblique "ां" + ending "चं"), Gujarati "કરોડનો" is UNIT_CRORE. Exactly ONE
 * declared ending is removed (longest first), optionally with one oblique
 * stem ending under it, and the head must carry the token's own class. A
 * surface that is itself a full lexicon form is never stripped. Mirrors
 * python/sankhya/verify.py::_verify_suffixed. */
function isSuffixedForm(cls: string, text: string): boolean {
  const surface = norm(text);
  if (ALL_CLASSES.has(surface)) return false;
  // R14: a blocked surface is never explained away as an inflected number
  // word -- "हजारे" (the surname) does not become हजार + "-े".
  if (BLOCKED_SURFACES.has(surface)) return false;
  for (const view of PACK_VIEWS) {
    if (view.suffixes.length === 0) continue;
    for (const suf of view.suffixes) {
      if (surface.length <= suf.length || !surface.endsWith(suf)) continue;
      const stem = surface.slice(0, surface.length - suf.length);
      const heads = [stem];
      for (const obl of view.obliques) {
        if (stem.length > obl.length && stem.endsWith(obl)) {
          heads.push(stem.slice(0, stem.length - obl.length));
        }
      }
      for (const head of heads) {
        // R15: per-pack strip, and the head must survive at a real length.
        if (head.length >= MIN_SUFFIX_HEAD_LEN && view.forms.get(head)?.has(cls)) return true;
      }
    }
  }
  return false;
}

/** True when `text` is an exact lexicon form of `cls`, or an inflected one
 * (R15 per-pack suffix strip). A blocked surface never is. Mirrors
 * python/sankhya/verify.py::is_known_form. */
export function isKnownForm(cls: string, text: string): boolean {
  const surface = norm(text);
  if (BLOCKED_SURFACES.has(surface)) return false;
  if (ALL_CLASSES.get(surface)?.has(cls)) return true;
  return isSuffixedForm(cls, text);
}

/** Edit distance between `a` and `b` is at most `k`. A tiny bounded
 * implementation -- the lexicon is a few hundred surfaces per class.
 * Mirrors python/sankhya/verify.py::levenshtein_le. */
export function levenshteinLE(a: string, b: string, k = 1): boolean {
  if (Math.abs(a.length - b.length) > k) return false;
  if (a === b) return true;
  let prev = Array.from({ length: b.length + 1 }, (_v, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    let best = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      const v = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
      cur.push(v);
      if (v < best) best = v;
    }
    if (best > k) return false;
    prev = cur;
  }
  return prev[b.length] <= k;
}

/** R17: `text` is a known form of `cls`, or within `maxDistance` edits of
 * one -- the typo tolerance that keeps a noised unit word working after
 * digits. Mirrors python/sankhya/verify.py::is_near_known_form. */
export function isNearKnownForm(cls: string, text: string, maxDistance = 1): boolean {
  if (isKnownForm(cls, text)) return true;
  const surface = norm(text);
  if (BLOCKED_SURFACES.has(surface)) return false;
  for (const [form, classes] of ALL_CLASSES) {
    if (classes.has(cls) && levenshteinLE(surface, form, maxDistance)) return true;
  }
  return false;
}

/** R12: `text` is a surface some pack declares an ambiguous form. Mirrors
 * python/sankhya/verify.py::is_ambiguous_form. */
export function isAmbiguousForm(text: string): boolean {
  return AMBIGUOUS_FORMS.has(norm(text));
}

/** R14: `text` is a surface some pack declares blocked. Mirrors
 * python/sankhya/verify.py::is_blocked_surface. */
export function isBlockedSurface(text: string): boolean {
  return BLOCKED_SURFACES.has(norm(text));
}

/** A BOUND number form is justified only by what follows it: the very next
 * token's text must be one of the unit surfaces it attaches to. Mirrors
 * python/sankhya/verify.py _verify_token. */
function isBoundFormInContext(cls: string, text: string, nextText: string | null): boolean {
  if (nextText === null) return false;
  const units = BOUND_FORMS.get(norm(text))?.get(cls);
  return units !== undefined && units.has(norm(nextText));
}

/** True when `text` is a declared BOUND number form of class `cls` that may
 * attach to `nextText`. Exported for the decoder's fused-word partition
 * (decode.ts boundKeepIndices), so both the verifier and the decoder agree
 * on which 1-character sub-runs stand on lexicon evidence. Mirrors
 * python/sankhya/verify.py::is_bound_form. */
export function isBoundForm(cls: string, text: string, nextText: string | null): boolean {
  return isBoundFormInContext(cls, text, nextText);
}

/** R9: true when `text` appears in the lexicon union with class "O" and NO
 * other class -- an ordinary word (an indefinite plural such as
 * "karodon"/"करोडो") a pack has explicitly declared a non-number. A surface
 * that ALSO carries a real number class in some other pack (a cross-pack
 * conflict) is NOT O-only and is left alone. Mirrors
 * python/sankhya/verify.py::is_lexicon_o_only. */
export function isLexiconOOnly(text: string): boolean {
  const classes = ALL_CLASSES.get(norm(text));
  return classes !== undefined && classes.size === 1 && classes.has("O");
}

const RANGE_SYMBOLS = new Set(["-", "–", "—", "/"]);

function isDigitsOnly(text: string): boolean {
  if (text.length === 0) return false;
  for (const ch of text) {
    const cp = ch.codePointAt(0)!;
    const isAscii = ch >= "0" && ch <= "9";
    // Indic digit glyphs the arithmetic core normalises to ASCII:
    // Devanagari U+0966-U+096F, Gujarati U+0AE6-U+0AEF. Mirrors
    // python/sankhya/verify.py _NATIVE_DIGITS.
    const isDeva = cp >= 0x0966 && cp <= 0x096f;
    const isGujr = cp >= 0x0ae6 && cp <= 0x0aef;
    if (!isAscii && !isDeva && !isGujr) return false;
  }
  return true;
}

/** Verify a decoded span's token list against the lexicon + structural
 * rules. Pure function of (tokens, lexicon) -- see VERIFY_CONTRACT.md. */
export function verifyTokens(tokens: Array<[string, string]>): boolean {
  if (tokens.length === 0) return false;

  let hasContentToken = false;

  for (let i = 0; i < tokens.length; i++) {
    const [cls, text] = tokens[i];
    const nextText = i + 1 < tokens.length ? tokens[i + 1][1] : null;
    switch (cls) {
      case "SEP": {
        if (text.length === 0 || /\S/.test(text)) return false;
        break;
      }
      case "DOT": {
        if (text !== ".") return false;
        break;
      }
      case "COMMA": {
        if (text !== ",") return false;
        break;
      }
      case "RANGE": {
        // Mirrors python verify._verify_token: strip surrounding whitespace,
        // then symbol match or NFC-lowercased range-word match.
        const stripped = text.trim();
        const lower = stripped.normalize("NFC").toLowerCase();
        if (!RANGE_SYMBOLS.has(stripped) && !RANGE_WORDS.has(lower)) return false;
        break;
      }
      case "DIGITS": {
        if (!isDigitsOnly(text)) return false;
        hasContentToken = true;
        break;
      }
      case "O": {
        if (FORMS.get(text.toLowerCase()) !== "O") return false;
        break;
      }
      default: {
        // R14: a surface a pack declares BLOCKED (a proper noun built on a
        // number word: "हजारे", "अरबी", "સવાઈ") never verifies as a number.
        if (BLOCKED_SURFACES.has(norm(text))) return false;
        // PFX_*/CARD_*/UNIT_*, or a bound form justified by the next token
        if (
          !ALL_CLASSES.get(norm(text))?.has(cls) &&
          !isBoundFormInContext(cls, text, nextText) &&
          !isSuffixedForm(cls, text)
        ) {
          return false;
        }
        hasContentToken = true;
        break;
      }
    }
  }

  return hasContentToken;
}

// --- span-level drop rules that need the lexicon -------------------------
//
// Each is mirrored 1:1 by python/sankhya/verify.py and applied by the same
// gate in both runtimes (Parser.decodeForward /
// eval_gold._apply_bare_digits_gate). They exist because the tagger, on real
// text, happily labels ordinary words as number words -- see
// python/data_wild/REPORT.md section 4.

type Tok = [string, string];

function isMeaningfulClass(cls: string): boolean {
  return cls === "DIGITS" || cls.startsWith("PFX_") || cls.startsWith("CARD_") || cls.startsWith("UNIT_");
}

/** (index, cls, text) for the PFX_/CARD_/UNIT_/DIGITS tokens of a span. */
function meaningfulTokens(tokens: Tok[]): Array<[number, string, string]> {
  const out: Array<[number, string, string]> = [];
  for (let i = 0; i < tokens.length; i++) {
    if (isMeaningfulClass(tokens[i][0])) out.push([i, tokens[i][0], tokens[i][1]]);
  }
  return out;
}

/** R14: the span carries a meaningful token whose surface a pack declares
 * blocked -- a proper noun built on a number word ("अण्णा हजारे",
 * "सवाई तुकोजीराव", "अरबी समुद्र"). Dropped outright. Mirrors
 * python/sankhya/verify.py::has_blocked_surface. */
export function hasBlockedSurface(tokens: Tok[]): boolean {
  return meaningfulTokens(tokens).some(([, , text]) => BLOCKED_SURFACES.has(norm(text)));
}

/** R12: the span rests entirely on surfaces that are also ordinary words.
 *
 * A meaningful token is *justified* when it is DIGITS or a known lexicon
 * form of its own class. If the span has at least one justified token and
 * EVERY justified token's surface is an ambiguous form, nothing but an
 * ordinary-word reading is holding the span up: "so", "sath", "arab",
 * "अरब", "peti" standing alone, or "सऊदी अरब" where the only lexicon-backed
 * word is the ethnonym. One non-ambiguous justified token anywhere keeps
 * the span ("ek arab", "100 अरब", "do peti", "unnis sau sath"). Callers
 * exempt a span with a currency marker. Mirrors
 * python/sankhya/verify.py::should_drop_ambiguous_form. */
export function shouldDropAmbiguousForm(tokens: Tok[]): boolean {
  const justified = meaningfulTokens(tokens).filter(
    ([, cls, text]) => cls === "DIGITS" || isKnownForm(cls, text),
  );
  if (justified.length === 0) return false;
  return justified.every(([, cls, text]) => cls !== "DIGITS" && isAmbiguousForm(text));
}

/** R16: a 1-2 character ambiguous SYMBOL unit ("k", "l", "m", "b") only
 * counts when GLUED to the digits it scales. "20k"/"2.5L" keep their unit;
 * "1996 k" (the ke/ki clitic after a year) and "33 k. m." (kilometres) lose
 * theirs and the span goes. Longer symbol forms ("lac", "cr", "bn", "LPA")
 * are commonly written with a space and are untouched. Mirrors
 * python/sankhya/verify.py::should_drop_loose_symbol_unit. */
export function shouldDropLooseSymbolUnit(tokens: Tok[]): boolean {
  for (const [i, cls, text] of meaningfulTokens(tokens)) {
    if (!cls.startsWith("UNIT_")) continue;
    const s = norm(text);
    if (s.length > MAX_GLUED_SYMBOL_LEN || !isAmbiguousForm(s)) continue;
    if (!isKnownForm(cls, s)) continue;
    const prevCls = i > 0 ? tokens[i - 1][0] : null;
    if (prevCls === "DIGITS" || prevCls === "DOT" || prevCls === "COMMA") continue;
    return true;
  }
  return false;
}

/** R13b: a span whose ONLY meaningful token is a spelled cardinal is kept
 * only when its surface is an EXACT lexicon form of that class. "पचास" (50)
 * and "બાવીસ" (22) are real bare-cardinal gold spans; "aavesh", "chaahie",
 * "barabar", "इ" are ordinary words the tagger guessed a cardinal for. No
 * suffix stripping and no edit-distance tolerance: a bare cardinal has
 * nothing else in the span to corroborate it. Mirrors
 * python/sankhya/verify.py::should_drop_unsupported_bare_cardinal. */
export function shouldDropUnsupportedBareCardinal(tokens: Tok[]): boolean {
  const meaningful = meaningfulTokens(tokens);
  if (meaningful.length !== 1) return false;
  const [, cls, text] = meaningful[0];
  if (!cls.startsWith("CARD_")) return false;
  return !ALL_CLASSES.get(norm(text))?.has(cls);
}

/** R17: a unit word that is not a unit word, next to bare digits.
 *
 * "3 hours" -> 3,000, "25वे" (an ordinal) -> 2,500,000, "1980ના" -> 1.98e8,
 * "chori"/"thought"/"oooh"/"खोड" alone: the tagger invents a unit class for
 * whatever letters sit next to (or instead of) a number. When a span's
 * UNIT_* token's surface is neither a lexicon form of that class (after one
 * declared case ending, R15) nor within edit distance 1 of one, and the
 * span has no independently justified CARD_ / PFX_ coefficient to
 * corroborate it, the unit is a guess and the span is dropped. The
 * edit-distance-1 tolerance keeps the synthetic set's noised unit words
 * working after digits. Mirrors
 * python/sankhya/verify.py::should_drop_unknown_unit. */
export function shouldDropUnknownUnit(tokens: Tok[]): boolean {
  const meaningful = meaningfulTokens(tokens);
  const hasJustifiedCoef = meaningful.some(
    ([, cls, text]) => (cls.startsWith("CARD_") || cls.startsWith("PFX_")) && isKnownForm(cls, text),
  );
  if (hasJustifiedCoef) return false;
  return meaningful.some(([, cls, text]) => cls.startsWith("UNIT_") && !isNearKnownForm(cls, text, 1));
}
