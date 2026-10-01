# Gate contamination: near-duplicates of gate prompts in the training-prompt pool

Pool: 74669 prompts (Dolci train rows of all gate benches, held-out rows excluded, + GSM8K train). 12-gram coverage, template 12-grams (df > 50) ignored (scripts/analysis/gate_contamination.py).

| gate bench | items | best coverage >= .3 | >= .6 | >= .9 | clean (< 0.3) |
|---|---:|---:|---:|---:|---:|
| dolci_dapo | 150 | 61 | 48 | 41 | 89 |
| dolci_knowledge | 100 | 0 | 0 | 0 | 100 |
| dolci_math | 300 | 128 | 104 | 82 | 172 |
| dolci_wordprob | 200 | 30 | 24 | 21 | 170 |
| dolci_yesno | 200 | 18 | 6 | 2 | 182 |
| all | 950 | 237 | 182 | 146 | 713 |

Contaminated items by source of the closest pool row: {'dolci_math': 177, 'dolci_dapo': 23, 'gsm8k_train': 16, 'dolci_yesno': 18, 'dolci_wordprob': 3}. Pool rows to exclude from training prompts: 765 ({'dolci_math': 654, 'dolci_dapo': 47, 'dolci_yesno': 44, 'gsm8k_train': 16, 'dolci_wordprob': 4}).
