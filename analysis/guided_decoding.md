# Checker-guided decoding vs unguided greedy on the Dolci gate

valid = rlvl strict; valid_s2 = Stage-2 valid with stated premise numbers (the cvf reward's validity); cvf = valid_s2 · correct; tokens = sampled tokens per item (guided: every candidate line).

## e2 SFT

| decoding | subset | n | found | valid | valid_s2 | correct | cvf | tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| unguided | all | 950 | 0.000 | 0.153 | 0.100 | 0.244 | 0.058 | 732 |
| unguided | clean | 713 | 0.000 | 0.171 | 0.101 | 0.273 | 0.065 | 684 |
| guided+repair | all | 950 | 0.355 | 0.355 | 0.355 | 0.101 | 0.101 | 4461 |
| guided+repair | clean | 713 | 0.355 | 0.355 | 0.355 | 0.108 | 0.108 | 4635 |
| guided+repair + fallback | all | 950 | 0.355 | 0.398 | 0.372 | 0.236 | 0.116 | 5016 |
| guided+repair + fallback | clean | 713 | 0.355 | 0.411 | 0.376 | 0.264 | 0.126 | 5161 |
| guided+repair ∩ agree | all | 950 | 0.157 | 0.157 | 0.157 | 0.101 | 0.081 | 4461 |
| guided+repair ∩ agree | clean | 713 | 0.163 | 0.163 | 0.163 | 0.108 | 0.087 | 4635 |
| guided | all | 950 | 0.341 | 0.341 | 0.341 | 0.102 | 0.102 | 4697 |
| guided | clean | 713 | 0.339 | 0.339 | 0.339 | 0.111 | 0.111 | 4846 |
| guided + fallback | all | 950 | 0.341 | 0.388 | 0.356 | 0.241 | 0.115 | 5252 |
| guided + fallback | clean | 713 | 0.339 | 0.401 | 0.358 | 0.272 | 0.126 | 5369 |
| guided ∩ agree | all | 950 | 0.161 | 0.161 | 0.161 | 0.102 | 0.084 | 4697 |
| guided ∩ agree | clean | 713 | 0.158 | 0.158 | 0.158 | 0.111 | 0.091 | 4846 |

Per bench (all items): valid_s2 / cvf / correct

| decoding | dolci_dapo | dolci_knowledge | dolci_math | dolci_wordprob | dolci_yesno |
|---|---|---|---|---|---|
| unguided | 0.013 / 0.000 / 0.033 | 0.060 / 0.000 / 0.010 | 0.057 / 0.007 / 0.073 | 0.295 / 0.220 / 0.440 | 0.055 / 0.045 / 0.580 |
| guided+repair | 0.240 / 0.007 / 0.007 | 0.490 / 0.000 / 0.000 | 0.300 / 0.030 / 0.030 | 0.445 / 0.180 / 0.180 | 0.365 / 0.250 / 0.250 |
| guided+repair + fallback | 0.240 / 0.007 / 0.033 | 0.490 / 0.000 / 0.000 | 0.303 / 0.033 / 0.070 | 0.515 / 0.240 / 0.400 | 0.370 / 0.255 / 0.590 |
| guided+repair ∩ agree | 0.067 / 0.007 / 0.007 | 0.130 / 0.000 / 0.000 | 0.097 / 0.020 / 0.030 | 0.250 / 0.170 / 0.180 | 0.235 / 0.180 / 0.250 |
| guided | 0.267 / 0.000 / 0.000 | 0.430 / 0.000 / 0.000 | 0.273 / 0.027 / 0.027 | 0.450 / 0.195 / 0.195 | 0.345 / 0.250 / 0.250 |
| guided + fallback | 0.267 / 0.000 / 0.027 | 0.430 / 0.000 / 0.010 | 0.277 / 0.030 / 0.063 | 0.510 / 0.245 / 0.420 | 0.350 / 0.255 / 0.605 |
| guided ∩ agree | 0.073 / 0.000 / 0.000 | 0.110 / 0.000 / 0.000 | 0.107 / 0.020 / 0.027 | 0.255 / 0.185 / 0.195 | 0.240 / 0.185 / 0.250 |

**guided+repair, found proofs** (n=337): correct 0.285 (unguided greedy on the same items 0.309); answer rests on >= 1 relation premise (`given` without numbers) 0.917; uses `know` 0.142; proof tokens 249, sampled tokens 1887; status {'budget': 315, 'found': 337, 'exhausted': 173, 'max_lines': 89, 'max_tokens': 36}

**guided, found proofs** (n=324): correct 0.299 (unguided greedy on the same items 0.309); answer rests on >= 1 relation premise (`given` without numbers) 0.892; uses `know` 0.164; proof tokens 254, sampled tokens 2060; status {'budget': 319, 'exhausted': 171, 'found': 324, 'max_tokens': 44, 'max_lines': 92}

## G14@750 (RL)

| decoding | subset | n | found | valid | valid_s2 | correct | cvf | tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| unguided | all | 950 | 0.000 | 0.179 | 0.143 | 0.277 | 0.094 | 265 |
| unguided | clean | 713 | 0.000 | 0.210 | 0.164 | 0.320 | 0.108 | 262 |
| guided+repair | all | 950 | 0.345 | 0.345 | 0.345 | 0.139 | 0.139 | 3030 |
| guided+repair | clean | 713 | 0.383 | 0.383 | 0.383 | 0.160 | 0.160 | 2989 |
| guided+repair + fallback | all | 950 | 0.345 | 0.367 | 0.348 | 0.273 | 0.142 | 3223 |
| guided+repair + fallback | clean | 713 | 0.383 | 0.411 | 0.386 | 0.314 | 0.163 | 3174 |
| guided+repair ∩ agree | all | 950 | 0.209 | 0.209 | 0.209 | 0.139 | 0.127 | 3030 |
| guided+repair ∩ agree | clean | 713 | 0.247 | 0.247 | 0.247 | 0.160 | 0.147 | 2989 |

Per bench (all items): valid_s2 / cvf / correct

| decoding | dolci_dapo | dolci_knowledge | dolci_math | dolci_wordprob | dolci_yesno |
|---|---|---|---|---|---|
| unguided | 0.000 / 0.000 / 0.067 | 0.100 / 0.000 / 0.010 | 0.013 / 0.000 / 0.107 | 0.240 / 0.190 / 0.455 | 0.370 / 0.255 / 0.645 |
| guided+repair | 0.107 / 0.000 / 0.000 | 0.620 / 0.000 / 0.000 | 0.143 / 0.010 / 0.010 | 0.425 / 0.220 / 0.220 | 0.610 / 0.425 / 0.425 |
| guided+repair + fallback | 0.107 / 0.000 / 0.060 | 0.620 / 0.000 / 0.010 | 0.143 / 0.010 / 0.103 | 0.430 / 0.225 / 0.440 | 0.620 / 0.435 / 0.650 |
| guided+repair ∩ agree | 0.013 / 0.000 / 0.000 | 0.270 / 0.000 / 0.000 | 0.020 / 0.003 / 0.010 | 0.290 / 0.210 / 0.220 | 0.530 / 0.390 / 0.425 |

**guided+repair, found proofs** (n=328): correct 0.402 (unguided greedy on the same items 0.415); answer rests on >= 1 relation premise (`given` without numbers) 0.890; uses `know` 0.079; proof tokens 210, sampled tokens 1936; status {'exhausted': 409, 'found': 328, 'budget': 213}

