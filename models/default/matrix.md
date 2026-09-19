| arch | channels | seed | val_value_acc | gold_int8_hi_latn | gold_int8_hi_deva | gold_int8_mr_deva | gold_int8_gu_gujr | gold_int8_combined | gold_int8_f1 | winner |
|---|---|---|---|---|---|---|---|---|---|---|
| v2 | 48 | 0 | 0.9666 | 0.9598 | 0.9548 | 0.9704 | 0.9322 | 0.9555 | — |  |
| v2 | 48 | 2 | 0.9669 | 0.9598 | 0.9677 | 0.9852 | 0.9322 | 0.9621 | 0.9694 | <-- winner (shipped 0.6.0) |

Gold = 748 examples (227 Hinglish + 186 Devanagari Hindi + 173 Marathi +
162 Gujarati), 170 negatives.
Both 0.6.0 runs trained locally on CPU (20 epochs, 200k synthetic with
`--lang hi_latn,hi_deva,mr_deva,gu_gujr --mix 0.32,0.26,0.21,0.21 --cross 0.10`,
plus the four verified LLM corpora at 20%); no Kaggle quota used.
winning config: **v2:48**, seed 2 (seed 0's mix 0.34,0.28,0.19,0.19 scored
lower on val (0.9666) and on combined gold (0.9555) and was not adopted).
