# LLM corpus generation brief (shared by every language generator)

Goal: realistic, chat-register sentences in which people mention money or
quantities the way they actually type them, plus a share of sentences with
NO quantity (negatives). These lines become training data for a character
tagger; the labels are produced by our own lexicon + arithmetic core, so
what matters here is AUTHENTIC surface text, not the label.

## Output format (one JSON object per line, UTF-8, no trailing commas)

{"text": "<the line>", "lang": "<pack id>", "phrases": [{"phrase": "<exact substring that is the quantity>", "value": <number>}], "scenario": "<tag>", "gen": "<agent tag>"}

- `phrases` is [] for negatives.
- `phrase` must be an EXACT substring of `text` (same characters, same case).
- `value` is the plain number the phrase means (sava lakh -> 125000; 2-3 lakh -> 200000 with the range noted as "range": [200000, 300000]).
- Multiple quantities in a line -> multiple entries, in text order.
- Do NOT normalise: keep the casing, emoji, punctuation, typos the writer would use.

## Coverage targets per 1,000 lines

- 55% one quantity, 15% two or more, 30% negatives (numbers that are not amounts: times, dates, phone numbers, ages, counts of people, distances, ordinals; and lines with no numbers at all but money-ish words like "lakhpati", "crorepati", "lakhon log").
- Registers: WhatsApp chat, OLX/classifieds listings, salary/CTC talk, news headlines, family budgeting, loans/EMI, stock/crypto chatter, jokes/exaggeration, bargaining, rent, wedding budgets, gig-work rates, gaming/in-app purchases.
- Forms: prefix words (sava/dedh/dhai/saadhe/paune/aadha/paav), spelled cardinals incl. regional and doubled-letter spellings, digits + symbol (2.5L, 20k, 1.5cr, 12 LPA), digits + word (5 lakh, 50 hazaar), grouped digits (2,50,000), ranges (2-3 lakh, 20-25k, das se pandrah hazaar), multiplicative chains (das hazaar crore), additive chains (ek lakh dus hazaar), currency markers (₹, Rs, Rs., INR, rupaye, rupees, /-), plurals and possessives (lakhs, crore's), mixed script within a line, emoji around and inside phrases, no-space forms (2lakh, sawalakh), ALL CAPS, sentence-initial and sentence-final phrases, phrases split across "aur"/"ya".
- Length: 3 to 25 words; ~10% one- to three-word lines.
- Diversity: no template reuse; vary names, cities, objects, verbs. Never repeat a line.

## Quality bar

- Would a native speaker believe a real person typed this? If not, drop it.
- Numbers must be plausible for the scenario (a phone costs 15k-1.5L, not 5 crore).
- For Devanagari packs, write Devanagari the way people type it on phones (some Latin words mixed in is fine and realistic).
- Do not invent number words that do not exist; unusual spellings of real words are welcome.

## Process rule (learned the hard way)

Write every line yourself, directly. Do NOT write a generator script, slot-filling
template, or loop that assembles sentences from lists — that is what
`python/sankhya/generator.py` already does, and the point of this corpus is
LLM-authored phrasing the grammar lacks. Code is allowed only to append lines and
to self-check (JSON validity, phrase-in-text, uniqueness, and a sliding 5-gram
overlap check: no two lines may share more than 4 consecutive words).
