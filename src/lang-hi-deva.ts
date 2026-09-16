// Currency marker lists ported from python/sankhya/langs/hi_deva.py.
// Only the pieces core.detectCurrency needs are mirrored here.

import type { LangPack } from "./core.ts";

export const HI_DEVA: LangPack = {
  currency_markers_before: ["₹", "रु", "रु.", "Rs", "Rs.", "INR"],
  currency_words_after: ["रुपये", "रुपए", "रूपये", "रुपया", "/-"],
};
