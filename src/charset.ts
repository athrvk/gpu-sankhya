// Lower-case + char-id mapping, with a sliding window for text longer than
// the model's effective max length (128, matching python train.MAX_LEN),
// and mandatory right-padding to match training-time padding (see PAD_TAIL
// below -- mirrors python/sankhya/np_infer.py's pad_ids()/PAD_TAIL exactly).

export const MAX_LEN = 128;
export const WINDOW_OVERLAP = 16;

// Training always right-pads every example to MAX_LEN with pad id 0 before
// running the conv stack, so every real character saw a long run of pad-id
// context to its right. Running inference on a short, tightly-cropped input
// (no padding) puts the last few real characters at the literal edge of the
// array, where conv1d's zero-padding (numeric 0.0, not the id-0 embedding)
// kicks in instead -- a distribution mismatch that measurably corrupts
// predictions on short/bare inputs (e.g. "2.5L" mistags "L" as a non-lakh
// unit when run tight). The fix: always right-pad the char-id array with at
// least PAD_TAIL copies of pad id 0 before calling forward(), run forward()
// on the padded array, then only take argmax over the first (real) text.length
// positions -- the padded tail's outputs are discarded, only used to give the
// real tokens correct right-context. Python's pad_ids() pads by exactly
// PAD_TAIL; here we additionally round the total up to a multiple of 8
// (buffer-size friendly for the JS backends), which only ever adds a few
// more pad chars -- harmless, since PAD_TAIL=24 already comfortably exceeds
// every arch preset's receptive field (see CONTRACT.md section 1; v2's
// receptive field is 17 chars each side).
export const PAD_TAIL = 24;

function roundUp8(n: number): number {
  return Math.ceil(n / 8) * 8;
}

/** Total (padded) inference length for a window/text of `realLen` real
 * characters: realLen + at least PAD_TAIL pad chars, rounded up to a
 * multiple of 8. */
export function paddedLength(realLen: number): number {
  return roundUp8(realLen + PAD_TAIL);
}

/** Upper bound on paddedLength() for any single window (window text is
 * capped at MAX_LEN chars) -- the size Scratch buffers are allocated to. */
export const PADDED_MAX = paddedLength(MAX_LEN);

// Indic decimal digits -> ASCII "0".."9": Devanagari U+0966 ('०') ..
// U+096F ('९') and Gujarati U+0AE6 ('૦') .. U+0AEF ('૯'). Each block is
// ten contiguous code points in 0..9 order, so the mapping is a pure
// per-character substitution that preserves every offset. Kept in sync
// with python/sankhya/charset.py NATIVE_DIGIT_BLOCKS.
const INDIC_DIGIT_RE = /[\u0966-\u096F\u0AE6-\u0AEF]/g;
const INDIC_DIGIT_BASES = [0x0966, 0x0ae6];

/** Normalize input text exactly as the Python side does, before char
 * encoding: (1) Unicode NFC normalization, (2) Indic digits mapped 1:1
 * to ASCII "0"-"9" (Devanagari U+0966-U+096F, Gujarati U+0AE6-U+0AEF),
 * (3) lowercasing. Must stay byte-identical to Python's `s.normalize("NFC")` + digit map + `.lower()`.
 *
 * NFC normalization can change string length (e.g. composing a base +
 * combining mark into fewer code units, or occasionally more). `parse()`
 * returns spans/offsets into `normalizeText(input)`, not the original
 * input string -- for the overwhelming majority of real-world inputs,
 * which already arrive in NFC form, normalization is a no-op on length
 * and offsets, and the `span` field is safe to treat as a substring of
 * the original input too (this is guaranteed to be true when the two
 * strings have equal length). Callers who need to map offsets back onto
 * the original input for non-NFC input should first call
 * `normalizeText()` themselves and index into its result. */
export function normalizeText(s: string): string {
  return s
    .normalize("NFC")
    .replace(INDIC_DIGIT_RE, (d) => {
      const cp = d.charCodeAt(0);
      for (const base of INDIC_DIGIT_BASES) {
        if (cp >= base && cp <= base + 9) return String(cp - base);
      }
      return d;
    })
    .toLowerCase();
}

export function buildCharToId(charset: string[]): Map<string, number> {
  const m = new Map<string, number>();
  for (let i = 0; i < charset.length; i++) m.set(charset[i], i);
  return m;
}

/** Encode `text` into char ids, right-padded with pad id 0 up to `padLen`
 * total entries (default: no padding, i.e. exactly text.length). If `out`
 * is given (must be >= padLen), writes into it and returns a subarray view
 * (no allocation beyond the view itself); otherwise allocates a fresh
 * Int32Array. Use `paddedLength(text.length)` as `padLen` before calling
 * forward() -- see PAD_TAIL above. */
// NOTE: `text` passed here must already be normalizeText()'d by the caller
// (Parser.parse / parseBatch normalize once up front) -- this function does
// not re-normalize, so its indexing matches the caller's offset bookkeeping
// exactly.
//
// Code-point vs. code-unit semantics: Python strings are sequences of
// Unicode code points, so decode.py's per-character labelling (and the
// offsets it reports) operate on code points. JS strings are UTF-16 code
// units, which agree with code points for every BMP character -- all of
// ASCII, Latin, and Devanagari (U+0000-U+FFFF) included, so hi_latn and
// hi_deva offsets are unaffected. They disagree for astral characters
// (e.g. most emoji, U+10000+), which JS represents as a surrogate *pair*
// (two UTF-16 code units) where Python counts one code point. This
// package deliberately keeps plain index-based (UTF-16 code unit)
// iteration here -- consistent with every other index/slice in this
// codebase (windowing in makeWindows/parse, decode.ts's run-splitting and
// text.slice()) -- rather than switching to `Array.from`-based code-point
// iteration, because doing so would require re-threading code-point
// offsets through windowing, decode.ts and Parser.parse as well; a
// partial fix in this function alone would only worsen the mismatch. The
// documented, tested consequence (see charset.test astral-emoji case):
// for input containing an astral character, every offset gpu-sankhya
// reports *after* that character is shifted by +1 (one extra UTF-16 code
// unit) relative to the equivalent Python (code-point-indexed) offset.
// Devanagari and Latin/ASCII input -- the supported scripts -- never hit
// this; it only affects stray astral characters such as emoji.
export function encodeChars(text: string, charToId: Map<string, number>, out?: Int32Array, padLen?: number): Int32Array {
  const unk = charToId.get("<unk>") ?? 1;
  const padId = charToId.get("<pad>") ?? 0;
  const total = padLen ?? text.length;
  const ids = out ? out.subarray(0, total) : new Int32Array(total);
  for (let i = 0; i < text.length; i++) {
    ids[i] = charToId.get(text[i]) ?? unk;
  }
  for (let i = text.length; i < total; i++) {
    ids[i] = padId;
  }
  return ids;
}

export interface Window {
  offset: number;
  length: number;
}

/** Split a length `n` string into windows of at most MAX_LEN chars with
 * WINDOW_OVERLAP overlap between consecutive windows. Single window if
 * n <= MAX_LEN. Each window's text length is <= MAX_LEN; padding (PAD_TAIL)
 * is applied separately at encode time, on top of this. */
export function makeWindows(n: number): Window[] {
  if (n <= MAX_LEN) return [{ offset: 0, length: n }];
  const windows: Window[] = [];
  const stride = MAX_LEN - WINDOW_OVERLAP;
  let offset = 0;
  while (offset < n) {
    const length = Math.min(MAX_LEN, n - offset);
    windows.push({ offset, length });
    if (offset + length >= n) break;
    offset += stride;
  }
  return windows;
}
