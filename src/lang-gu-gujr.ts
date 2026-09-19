// Currency marker lists ported from python/sankhya/langs/gu_gujr.py.
// Only the pieces core.detectCurrency needs are mirrored here.
//
// "રૂપિયા" is the stem every case-inflected Gujarati form starts with
// (રૂપિયાનું, રૂપિયાની, રૂપિયામાં, રૂપિયાથી ...), so the family is
// listed out in full alongside it.

import type { LangPack } from "./core.ts";

export const GU_GUJR: LangPack = {
  currency_markers_before: ["₹", "રૂ", "રૂ.", "Rs", "Rs.", "INR"],
  currency_words_after: [
    "રૂપિયા",
    "રૂપિયો",
    "રૂપિયાનું",
    "રૂપિયાની",
    "રૂપિયાનો",
    "રૂપિયાના",
    "રૂપિયામાં",
    "રૂપિયાથી",
    "રૂપિયે",
    "/-",
  ],
};
