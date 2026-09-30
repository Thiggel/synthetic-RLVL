# Continued-SFT arms c / l / e / le vs base (new lemma library)

| metric | base (L1 init) | l: +new families |
|---|---:|---:|
| gate greedy valid | 0.0105 | 0.0274 |
| gate greedy valid·correct | 0.0032 | 0.0095 |
| gate greedy correct | 0.2242 | 0.1632 |
| gate T=1 valid / sample | 0.0043 | 0.0149 |
| gate T=1 valid·correct / sample | 0.0022 | 0.0060 |
| gate T=1 correct / sample | 0.1864 | 0.1514 |
| gate valid@16 | 0.0432 | 0.0863 |
| gate valid·correct@16 | 0.0232 | 0.0284 |
| gate mixed@8 (valid) | 0.0256 | 0.0610 |
| gen test valid | 0.8075 | 0.8670 |
| gen test answer acc | 0.9200 | 0.9400 |
| new-family test valid | 0.0100 | 0.9520 |
| T=1 valid / sample, wordprob | 0.0056 | 0.0203 |
| T=1 valid / sample, math | 0.0013 | 0.0083 |
| T=1 valid / sample, yesno | 0.0078 | 0.0181 |
| T=1 valid / sample, dapo | 0.0000 | 0.0004 |
| T=1 valid / sample, knowledge | 0.0100 | 0.0387 |
