# CelebA Full–minus_A: IID Benign, three views

Validation only (19,867 images), round70. Ten paired model seeds; each model uses the same checkpoint across raw/native/shared views. Mean ± sample SD (ddof=1); paired difference = minus_A − Full. ACC is percent; ΔACC is percentage points. AEOD and ASPD are lower-is-better disparities. Formal primary endpoint remains pending.

## native — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| minus_A | 10 | 87.829 ± 1.625 | 0.01205 ± 0.00918 | 0.05940 ± 0.00995 |
| minus_A minus Full | 10 | -0.430 ± 1.259 | 0.00231 ± 0.01461 | -0.00314 ± 0.01805 |

## native — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| minus_A | 9 | 88.265 ± 0.908 | 0.01077 ± 0.00874 | 0.05949 ± 0.01054 |
| minus_A minus Full | 9 | -0.087 ± 0.678 | 0.00102 ± 0.01488 | -0.00283 ± 0.01912 |

## native — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| minus_A | 6 | 88.248 ± 0.624 | 0.01098 ± 0.00794 | 0.06189 ± 0.00863 |
| minus_A minus Full | 6 | -0.310 ± 0.703 | -0.00040 ± 0.01398 | -0.00460 ± 0.02052 |

## raw — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.549 ± 1.038 | 0.04027 ± 0.01035 | 0.10308 ± 0.01007 |
| minus_A | 10 | 88.214 ± 1.695 | 0.04092 ± 0.00933 | 0.10124 ± 0.00931 |
| minus_A minus Full | 10 | -0.335 ± 1.319 | 0.00064 ± 0.01538 | -0.00184 ± 0.01319 |

## raw — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.646 ± 1.052 | 0.04004 ± 0.01095 | 0.10338 ± 0.01063 |
| minus_A | 9 | 88.676 ± 0.910 | 0.04071 ± 0.00987 | 0.10366 ± 0.00560 |
| minus_A minus Full | 9 | 0.030 ± 0.675 | 0.00067 ± 0.01631 | 0.00028 ± 0.01204 |

## raw — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.813 ± 0.633 | 0.04225 ± 0.01270 | 0.10710 ± 0.00862 |
| minus_A | 6 | 88.683 ± 0.631 | 0.04186 ± 0.00928 | 0.10352 ± 0.00430 |
| minus_A minus Full | 6 | -0.130 ± 0.778 | -0.00040 ± 0.01774 | -0.00358 ± 0.01009 |

## shared_calibration — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| minus_A | 10 | 87.829 ± 1.625 | 0.01205 ± 0.00918 | 0.05940 ± 0.00995 |
| minus_A minus Full | 10 | -0.430 ± 1.259 | 0.00231 ± 0.01461 | -0.00314 ± 0.01805 |

## shared_calibration — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| minus_A | 9 | 88.265 ± 0.908 | 0.01077 ± 0.00874 | 0.05949 ± 0.01054 |
| minus_A minus Full | 9 | -0.087 ± 0.678 | 0.00102 ± 0.01488 | -0.00283 ± 0.01912 |

## shared_calibration — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| minus_A | 6 | 88.248 ± 0.624 | 0.01098 ± 0.00794 | 0.06189 ± 0.00863 |
| minus_A minus Full | 6 | -0.310 ± 0.703 | -0.00040 ± 0.01398 | -0.00460 ± 0.02052 |

AEOD = absolute TPR gap, not full equalized odds. Raw uses margin > 0 (ties predict0); native retains the original root-only calibration recipe; shared_calibration uses the frozen common root-only rule. Native/shared metrics and counts are compared below, not treated as independent evidence of calibration gain. No threshold is refitted by this builder.
Complete-scene replay devices: Full {'cpu': 2, 'cuda:0': 8}; minus_A {'cpu': 10}. Training Torch: Full {'2.11.0+cu128': 10}; minus_A {'2.11.0+cu128': 10}. Historical/current driver provenance is retained per source record. The broader Full100 history includes98 cu128 and2 cu130 records; that fact does not turn this selected scene into a unified-device comparison.
Seed91001 participated in recipe selection; validation was exposed during development, and historical test exposure remains disclosed. The9/6 panels apply identical predefined seeds to both variants and are descriptive subsets, not untouched confirmation sets.
IID F Flip has only two accepted paired seeds and is preserved as individual/coverage data, excluded from every mean. This is one of ten minus_A scenes, not A100 or all mechanism controls. No significance, necessity, causal-isolation, final-test or whole-rebuttal-completion claim is made.
