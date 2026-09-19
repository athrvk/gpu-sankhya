# Attribution for `gold_wild_<lang>.jsonl`

Unlike the other `tests/gold*.jsonl` files (which we wrote ourselves), the
`gold_wild_*` files contain **real sentences sampled from openly licensed
corpora** and hand-labelled by us. The labels are ours; the sentences are not.

| gold file | sentences from | licence |
| --- | --- | --- |
| `gold_wild_hi_latn.jsonl` | Google **Dakshina** v1.0, `hi/romanized/hi.romanized.rejoined.tsv` (native-speaker romanisations of Hindi Wikipedia sentences), and **CMU DoG Hinglish** (`festvox/cmu_hinglish_dog`, romanised-Hindi chat) | Dakshina: CC BY-SA 4.0. CMU DoG Hinglish: CC BY-SA 3.0 / GFDL. |
| `gold_wild_hi_deva.jsonl` | Dakshina v1.0 `hi/native_script_wikipedia` (Hindi Wikipedia sentences) | CC BY-SA 4.0 (underlying Wikipedia text CC BY-SA 3.0) |
| `gold_wild_mr_deva.jsonl` | Dakshina v1.0 `mr/native_script_wikipedia` | CC BY-SA 4.0 (underlying Wikipedia text CC BY-SA 3.0) |
| `gold_wild_gu_gujr.jsonl` | Dakshina v1.0 `gu/native_script_wikipedia` | CC BY-SA 4.0 (underlying Wikipedia text CC BY-SA 3.0) |

- Dakshina: <https://github.com/google-research-datasets/dakshina> — Google
  Research, released under CC BY-SA 4.0.
- CMU Document Grounded Conversations (Hinglish split):
  <https://huggingface.co/datasets/festvox/cmu_hinglish_dog>.

Both licences permit redistribution **with attribution and share-alike**, which
is why these excerpts may live in the repository at all; the rest of
`python/data_wild/` (bulk raw corpora and the full samples) is gitignored. The
`text` fields of these four files, being excerpts of CC BY-SA works, remain
available under **CC BY-SA 4.0** and are not covered by the repository's MIT
licence; the `spans` annotations are ours and are MIT like the rest of the repo.

Reproduce the sampling with `python -m sankhya.wild sample` (seed 17); see
`python/data_wild/REPORT.md`.
