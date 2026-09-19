| arch | channels | seed | val_value_acc | gold_int8_hi_latn | gold_int8_hi_deva | gold_int8_mr_deva | gold_int8_combined | gold_int8_f1 | winner |
|---|---|---|---|---|---|---|---|---|---|
| v2 | 48 | 0 | 0.9086 | 0.9548 | 0.9613 | 0.9778 | 0.9632 | 0.9633 | <-- winner (shipped 0.5.0) |
| v2 | 48 | 2 | 0.9048 | 0.9447 | 0.9742 | 0.9185 | 0.9468 | — |  |

Gold = 586 examples (227 Hinglish + 186 Devanagari Hindi + 173 Devanagari Marathi), 120 negatives.
Both 0.5.0 runs trained locally on CPU (20 epochs, 200k synthetic with
`--lang hi_latn,hi_deva,mr_deva --mix 0.40,0.33,0.27 --cross 0.10`, plus
the three verified LLM corpora at 20%); no Kaggle quota used.
winning config: **v2:48**, seed 0 (seed 2's mix 0.36,0.34,0.30 scored
lower on every gold split and was not adopted).
