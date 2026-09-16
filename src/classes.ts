// Port of python/sankhya/classes.py — language-independent semantic class inventory.

export const PREFIXES = [
  "PFX_PAAV",
  "PFX_AADHA",
  "PFX_PAUNE",
  "PFX_SAVA",
  "PFX_DEDH",
  "PFX_DHAI",
  "PFX_SAADHE",
];

export const CARDINALS = Array.from({ length: 99 }, (_, i) => `CARD_${i + 1}`);

export const UNITS = [
  "UNIT_SAU",
  "UNIT_HAZAAR",
  "UNIT_LAKH",
  "UNIT_MILLION",
  "UNIT_CRORE",
  "UNIT_BILLION",
  "UNIT_ARAB",
  "UNIT_KHARAB",
];

export const MISC = ["O", "SEP", "RANGE", "DIGITS", "DOT", "COMMA"];

export const CLASSES: string[] = ["O", ...MISC.filter((c) => c !== "O"), ...PREFIXES, ...CARDINALS, ...UNITS];
export const CLASS_TO_ID: Record<string, number> = Object.fromEntries(CLASSES.map((c, i) => [c, i]));

const _UNIT_VALUE: Record<string, number> = {
  UNIT_SAU: 1e2,
  UNIT_HAZAAR: 1e3,
  UNIT_LAKH: 1e5,
  UNIT_MILLION: 1e6,
  UNIT_CRORE: 1e7,
  UNIT_BILLION: 1e9,
  UNIT_ARAB: 1e9,
  UNIT_KHARAB: 1e11,
};

export const UNIT_NAME: Record<string, Sankhya_Unit> = {
  UNIT_SAU: "sau",
  UNIT_HAZAAR: "hazaar",
  UNIT_LAKH: "lakh",
  UNIT_MILLION: "million",
  UNIT_CRORE: "crore",
  UNIT_BILLION: "billion",
  UNIT_ARAB: "arab",
  UNIT_KHARAB: "kharab",
};

type Sankhya_Unit = "sau" | "hazaar" | "lakh" | "crore" | "million" | "billion" | "arab" | "kharab";

export interface PrefixInfo {
  standalone: number;
  op: "sub25" | "add25" | "add5" | null;
  min_n?: number;
}

export const PREFIX_INFO: Record<string, PrefixInfo> = {
  PFX_PAAV: { standalone: 0.25, op: null },
  PFX_AADHA: { standalone: 0.5, op: null },
  PFX_PAUNE: { standalone: 0.75, op: "sub25", min_n: 2 },
  PFX_SAVA: { standalone: 1.25, op: "add25", min_n: 2 },
  PFX_DEDH: { standalone: 1.5, op: null },
  PFX_DHAI: { standalone: 2.5, op: null },
  PFX_SAADHE: { standalone: 0.5, op: "add5", min_n: 3 },
};

export function isPrefix(cls: string): boolean {
  return cls in PREFIX_INFO;
}

export function isCard(cls: string): boolean {
  return cls.startsWith("CARD_");
}

export function isUnit(cls: string): boolean {
  return cls in _UNIT_VALUE;
}

export function cardValue(cls: string): number {
  return parseInt(cls.split("_", 2)[1], 10);
}

export function unitValue(cls: string): number {
  return _UNIT_VALUE[cls];
}
