// Currency marker lists ported from python/sankhya/langs/mr_deva.py.
// Only the pieces core.detectCurrency needs are mirrored here.
//
// "रुपयां" is the oblique stem every case-inflected Marathi form starts
// with (रुपयांवर, रुपयांची, रुपयांनी ...), so it covers the whole family.

import type { LangPack } from "./core.ts";

export const MR_DEVA: LangPack = {
  currency_markers_before: ["₹", "रु", "रु.", "Rs", "Rs.", "INR"],
  currency_words_after: [
    "रुपये",
    "रुपयां",
    "रुपयांचे",
    "रुपयांचा",
    "रुपया",
    "रूपये",
    "रुपयांना",
    "रुपयात",
    "/-",
  ],
};
