// Verified-tier predicate: a span is "verified" when every (class, text)
// token it decoded to is independently justified by the lexicon (or, for
// SEP/DOT/COMMA/RANGE/DIGITS, by a structural rule). Mirrors
// python/sankhya/verify.py exactly -- see VERIFY_CONTRACT.md.
import lexiconJson from "./data/lexicon.json" with { type: "json" };

interface LexiconPack {
  forms: Record<string, string>;
  range_words: string[];
}
interface LexiconJson {
  version: number;
  packs: Record<string, LexiconPack>;
}

const lexicon = lexiconJson as unknown as LexiconJson;

// Union of all packs' forms (surface -> class) and range_words.
const FORMS: Map<string, string> = new Map();
const RANGE_WORDS: Set<string> = new Set();
for (const pack of Object.values(lexicon.packs)) {
  for (const [surface, cls] of Object.entries(pack.forms)) {
    if (!FORMS.has(surface)) FORMS.set(surface, cls);
  }
  for (const w of pack.range_words) RANGE_WORDS.add(w.toLowerCase());
}

const RANGE_SYMBOLS = new Set(["-", "–", "—", "/"]);

function isDigitsOnly(text: string): boolean {
  if (text.length === 0) return false;
  for (const ch of text) {
    const cp = ch.codePointAt(0)!;
    const isAscii = ch >= "0" && ch <= "9";
    const isDeva = cp >= 0x0966 && cp <= 0x096f;
    if (!isAscii && !isDeva) return false;
  }
  return true;
}

/** Verify a decoded span's token list against the lexicon + structural
 * rules. Pure function of (tokens, lexicon) -- see VERIFY_CONTRACT.md. */
export function verifyTokens(tokens: Array<[string, string]>): boolean {
  if (tokens.length === 0) return false;

  let hasContentToken = false;

  for (const [cls, text] of tokens) {
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
        // PFX_*/CARD_*/UNIT_*
        if (FORMS.get(text.toLowerCase()) !== cls) return false;
        hasContentToken = true;
        break;
      }
    }
  }

  return hasContentToken;
}
