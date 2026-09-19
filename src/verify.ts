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
}
interface LexiconJson {
  version: number;
  packs: Record<string, LexiconPack>;
}

const lexicon = lexiconJson as unknown as LexiconJson;

// Union of all packs' forms (surface -> class) and range_words.
const FORMS: Map<string, string> = new Map();
const RANGE_WORDS: Set<string> = new Set();
// surface -> class -> the unit surfaces that surface may precede.
const BOUND_FORMS: Map<string, Map<string, Set<string>>> = new Map();
for (const pack of Object.values(lexicon.packs)) {
  for (const [surface, cls] of Object.entries(pack.forms)) {
    if (!FORMS.has(surface)) FORMS.set(surface, cls);
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
}

/** A BOUND number form is justified only by what follows it: the very next
 * token's text must be one of the unit surfaces it attaches to. Mirrors
 * python/sankhya/verify.py _verify_token. */
function isBoundFormInContext(cls: string, text: string, nextText: string | null): boolean {
  if (nextText === null) return false;
  const units = BOUND_FORMS.get(text.toLowerCase())?.get(cls);
  return units !== undefined && units.has(nextText.toLowerCase());
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
        if (FORMS.get(text.toLowerCase()) !== cls && !isBoundFormInContext(cls, text, nextText)) {
          return false;
        }
        hasContentToken = true;
        break;
      }
    }
  }

  return hasContentToken;
}
