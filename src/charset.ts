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
// more pad chars -- harmless, since PAD_TAIL=16 already comfortably exceeds
// the network's receptive field.
export const PAD_TAIL = 16;

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
export function encodeChars(text: string, charToId: Map<string, number>, out?: Int32Array, padLen?: number): Int32Array {
  const lower = text.toLowerCase();
  const unk = charToId.get("<unk>") ?? 1;
  const padId = charToId.get("<pad>") ?? 0;
  const total = padLen ?? lower.length;
  const ids = out ? out.subarray(0, total) : new Int32Array(total);
  for (let i = 0; i < lower.length; i++) {
    ids[i] = charToId.get(lower[i]) ?? unk;
  }
  for (let i = lower.length; i < total; i++) {
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
