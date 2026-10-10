# CelebA Full–minus_F: IID Benign, three views

Validation only (19,867 images), round70; IID is frozen Dirichlet alpha5000. Ten paired model seeds; each model uses the same checkpoint across raw/native/shared views. Mean ± sample SD (ddof=1); paired difference = minus_F − Full. ACC is percent; ΔACC is percentage points. AEOD and ASPD are lower-is-better disparities. Formal primary endpoint remains pending.

## native — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| minus_F | 10 | 88.472 ± 0.897 | 0.00632 ± 0.00425 | 0.06549 ± 0.01498 |
| minus_F minus Full | 10 | 0.214 ± 1.451 | -0.00341 ± 0.00804 | 0.00295 ± 0.02036 |

## native — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| minus_F | 9 | 88.643 ± 0.761 | 0.00549 ± 0.00355 | 0.07012 ± 0.00338 |
| minus_F minus Full | 9 | 0.291 ± 1.517 | -0.00426 ± 0.00805 | 0.00780 ± 0.01422 |

## native — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| minus_F | 6 | 88.486 ± 0.758 | 0.00596 ± 0.00371 | 0.07085 ± 0.00146 |
| minus_F minus Full | 6 | -0.072 ± 1.082 | -0.00542 ± 0.00917 | 0.00436 ± 0.01494 |

## raw — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.549 ± 1.038 | 0.04027 ± 0.01035 | 0.10308 ± 0.01007 |
| minus_F | 10 | 88.778 ± 0.921 | 0.04605 ± 0.00944 | 0.10965 ± 0.00771 |
| minus_F minus Full | 10 | 0.229 ± 1.374 | 0.00578 ± 0.01538 | 0.00658 ± 0.01550 |

## raw — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.646 ± 1.052 | 0.04004 ± 0.01095 | 0.10338 ± 0.01063 |
| minus_F | 9 | 88.978 ± 0.709 | 0.04401 ± 0.00729 | 0.10907 ± 0.00794 |
| minus_F minus Full | 9 | 0.333 ± 1.415 | 0.00397 ± 0.01514 | 0.00569 ± 0.01616 |

## raw — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.813 ± 0.633 | 0.04225 ± 0.01270 | 0.10710 ± 0.00862 |
| minus_F | 6 | 88.821 ± 0.706 | 0.04104 ± 0.00709 | 0.10609 ± 0.00591 |
| minus_F minus Full | 6 | 0.008 ± 1.004 | -0.00121 ± 0.01600 | -0.00101 ± 0.01015 |

## shared_calibration — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| minus_F | 10 | 88.472 ± 0.897 | 0.00632 ± 0.00425 | 0.06549 ± 0.01498 |
| minus_F minus Full | 10 | 0.214 ± 1.451 | -0.00341 ± 0.00804 | 0.00295 ± 0.02036 |

## shared_calibration — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| minus_F | 9 | 88.643 ± 0.761 | 0.00549 ± 0.00355 | 0.07012 ± 0.00338 |
| minus_F minus Full | 9 | 0.291 ± 1.517 | -0.00426 ± 0.00805 | 0.00780 ± 0.01422 |

## shared_calibration — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| minus_F | 6 | 88.486 ± 0.758 | 0.00596 ± 0.00371 | 0.07085 ± 0.00146 |
| minus_F minus Full | 6 | -0.072 ± 1.082 | -0.00542 ± 0.00917 | 0.00436 ± 0.01494 |

AEOD = absolute TPR gap, not full equalized odds. Raw uses margin > 0 (ties predict0); native retains the original root-only calibration recipe; shared_calibration uses the frozen common root-only rule. Native/shared metrics and counts are compared below, not treated as independent evidence of calibration gain. No threshold is refitted by this builder.
Complete-scene replay devices: Full {'cpu': 2, 'cuda:0': 8}; minus_F {'cpu': 10}. Training Torch: Full {'2.11.0+cu128': 10}; minus_F {'2.11.0+cu128': 10}. Historical/current driver provenance is retained per source record. The broader Full100 history includes98 cu128 and2 cu130 records; that fact does not turn this selected scene into a unified-device comparison.
Seed91001 participated in recipe selection; validation was exposed during development, and historical test exposure remains disclosed. The9/6 panels apply identical predefined seeds to both variants and are descriptive subsets, not untouched confirmation sets.
Only IID Benign seeds91001–91010 enter this one-scene table. No cross-scene aggregate is computed or published; the other nine minus_F scenes and other control coverages remain incomplete. This is not F100 or all mechanism controls. No significance, necessity, causal-isolation, final-test or whole-rebuttal-completion claim is made.
