# Continued-SFT arms c / l / e / le vs base (new lemma library)

| metric | base (L1 init) | c: +fresh gen | l: +new families | le: +both |
|---|---:|---:|---:|---:|
| gate greedy valid | 0.0105 | 0.0147 | 0.0274 | 0.0684 |
| gate greedy valid·correct | 0.0032 | 0.0042 | 0.0095 | 0.0379 |
| gate greedy correct | 0.2242 | 0.2011 | 0.1632 | 0.1958 |
| gate T=1 valid / sample | 0.0043 | 0.0046 | 0.0149 | 0.0420 |
| gate T=1 valid·correct / sample | 0.0022 | 0.0016 | 0.0060 | 0.0266 |
| gate T=1 correct / sample | 0.1864 | 0.1641 | 0.1514 | 0.1707 |
| gate valid@16 | 0.0432 | 0.0368 | 0.0863 | 0.1642 |
| gate valid·correct@16 | 0.0232 | 0.0137 | 0.0284 | 0.0789 |
| gate mixed@8 (valid) | 0.0256 | 0.0238 | 0.0610 | 0.1197 |
| gen test valid | 0.8075 | 0.9110 | 0.8670 | 0.8230 |
| gen test answer acc | 0.9200 | 0.9580 | 0.9400 | 0.9245 |
| new-family test valid | 0.0100 | 0.0140 | 0.9520 | 0.9580 |
| T=1 valid / sample, wordprob | 0.0056 | 0.0063 | 0.0203 | 0.1269 |
| T=1 valid / sample, math | 0.0013 | 0.0019 | 0.0083 | 0.0112 |
| T=1 valid / sample, yesno | 0.0078 | 0.0050 | 0.0181 | 0.0244 |
| T=1 valid / sample, dapo | 0.0000 | 0.0000 | 0.0004 | 0.0017 |
| T=1 valid / sample, knowledge | 0.0100 | 0.0156 | 0.0387 | 0.0606 |
