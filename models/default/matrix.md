| arch | channels | seed | data | val_value_acc | gold_int8_hi_latn | gold_int8_hi_deva | gold_int8_f1 | winner |
|---|---|---|---|---|---|---|---|---|
| v2 | 48 | 2 | glued | 0.9243 | 0.9442 | 0.9545 | 0.9656 |  |
| v2 | 48 | 2 | no glue (SANKHYA_JOIN_WORDS_P=0) | 0.9158 | 0.9340 | 0.9545 | — |  |
| v2 | 48 | 0 | glued | 0.9242 | 0.9543 | 0.9740 | 0.9805 | <-- winner (shipped 0.3.4) |
| (0.3.3 shipped, Kaggle seed 2) | 48 | 2 | pre-glue data | 0.9190 | 0.9492 | 0.9740 | 0.9740 |  |

Gold = 410 examples (225 Hinglish + 185 Devanagari, incl. the 8 lexicon-gap cases added in 0.3.4).
All 0.3.4 runs trained locally on CPU (~75 s/epoch, 20 epochs, 200k synthetic + LLM corpora at 20%); no Kaggle quota used.
winning config: **v2:48**
