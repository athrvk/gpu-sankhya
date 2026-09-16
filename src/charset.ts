// Lower-case + char-id mapping, with a sliding window for text longer than
// the model's effective max length (128, matching python train.MAX_LEN).

export const MAX_LEN = 128;
export const WINDOW_OVERLAP = 16;

export function buildCharToId(charset: string[]): Map<string, number> {
  const m = new Map<string, number>();
  for (let i = 0; i < charset.length; i++) m.set(charset[i], i);
  return m;
}

/** Encode `text` into char ids. If `out` is given (must be >= text.length),
 * writes into it and returns a subarray view (no allocation beyond the view
 * itself); otherwise allocates a fresh Int32Array. */
export function encodeChars(text: string, charToId: Map<string, number>, out?: Int32Array): Int32Array {
  const lower = text.toLowerCase();
  const unk = charToId.get("<unk>") ?? 1;
  const ids = out ? out.subarray(0, lower.length) : new Int32Array(lower.length);
  for (let i = 0; i < lower.length; i++) {
    ids[i] = charToId.get(lower[i]) ?? unk;
  }
  return ids;
}

export interface Window {
  offset: number;
  length: number;
}

/** Split a length `n` string into windows of at most MAX_LEN chars with
 * WINDOW_OVERLAP overlap between consecutive windows. Single window if
 * n <= MAX_LEN. */
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
