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
}

const byLengthDesc = (a: string, b: string): number => b.length - a.length || (a < b ? -1 : a > b ? 1 : 0);
const SUFFIXES_SORTED: string[] = [...WORD_SUFFIXES].sort(byLengthDesc);
const OBLIQUES_SORTED: string[] = [...WORD_OBLIQUE_ENDINGS].sort(byLengthDesc);

/** A PFX_, CARD_ or UNIT_ token whose exact surface is unknown may still be
 * an inflected form of a known head: Marathi "लाखांचं" is UNIT_LAKH (लाख +
 * oblique "ां" + ending "चं"), Gujarati "કરોડનો" is UNIT_CRORE. Exactly ONE
 * declared ending is removed (longest first), optionally with one oblique
 * stem ending under it, and the head must carry the token's own class. A
 * surface that is itself a full lexicon form is never stripped. Mirrors
 * python/sankhya/verify.py::_verify_suffixed. */
function isSuffixedForm(cls: string, text: string): boolean {
  const surface = norm(text);
  if (SUFFIXES_SORTED.length === 0) return false;
  if (ALL_CLASSES.has(surface)) return false;
  for (const suf of SUFFIXES_SORTED) {
    if (surface.length <= suf.length || !surface.endsWith(suf)) continue;
    const stem = surface.slice(0, surface.length - suf.length);
    const heads = [stem];
    for (const obl of OBLIQUES_SORTED) {
      if (stem.length > obl.length && stem.endsWith(obl)) {
        heads.push(stem.slice(0, stem.length - obl.length));
      }
    }
    for (const head of heads) {
      if (ALL_CLASSES.get(head)?.has(cls)) return true;
    }
  }
  return false;
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
