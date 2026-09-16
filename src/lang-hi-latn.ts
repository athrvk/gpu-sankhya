// Currency marker lists ported from python/sankhya/langs/hi_latn.py.
// Only the pieces core.detectCurrency needs are mirrored here.

import type { LangPack } from "./core.ts";

export const HI_LATN: LangPack = {
  currency_markers_before: ["₹", "Rs", "Rs.", "rs", "INR", "Re"],
  currency_words_after: ["rupaye", "rupay", "rupees", "rupiya", "rupya", "/-"],
};
