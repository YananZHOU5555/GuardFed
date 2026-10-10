# CelebA Full–minus_F: IID Benign and F Flip, three views

Validation only (19,867 images), round70; IID is frozen Dirichlet alpha5000. Ten paired model seeds per scene; each model uses the same checkpoint across raw/native/shared views. Mean ± sample SD (ddof=1); paired difference = minus_F − Full. ACC is percent; ΔACC is percentage points. AEOD and ASPD are lower-is-better disparities. Formal primary endpoint remains pending.

## native — All 10 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| IID / Benign / minus_F | 10 | 88.472 ± 0.897 | 0.00632 ± 0.00425 | 0.06549 ± 0.01498 |
| IID / Benign / minus_F minus Full | 10 | 0.214 ± 1.451 | -0.00341 ± 0.00804 | 0.00295 ± 0.02036 |
| IID / F Flip / Full | 10 | 88.391 ± 0.626 | 0.01068 ± 0.00958 | 0.06068 ± 0.01026 |
| IID / F Flip / minus_F | 10 | 88.325 ± 1.212 | 0.01439 ± 0.00651 | 0.06600 ± 0.01819 |
| IID / F Flip / minus_F minus Full | 10 | -0.066 ± 1.239 | 0.00372 ± 0.00942 | 0.00532 ± 0.02066 |

## native — Exclude selection seed: 9 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| IID / Benign / minus_F | 9 | 88.643 ± 0.761 | 0.00549 ± 0.00355 | 0.07012 ± 0.00338 |
| IID / Benign / minus_F minus Full | 9 | 0.291 ± 1.517 | -0.00426 ± 0.00805 | 0.00780 ± 0.01422 |
| IID / F Flip / Full | 9 | 88.412 ± 0.660 | 0.01147 ± 0.00981 | 0.06043 ± 0.01085 |
| IID / F Flip / minus_F | 9 | 88.657 ± 0.642 | 0.01473 ± 0.00681 | 0.07064 ± 0.01141 |
| IID / F Flip / minus_F minus Full | 9 | 0.244 ± 0.802 | 0.00327 ± 0.00987 | 0.01021 ± 0.01452 |

## native — Seeds 91005–91010: 6 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| IID / Benign / minus_F | 6 | 88.486 ± 0.758 | 0.00596 ± 0.00371 | 0.07085 ± 0.00146 |
| IID / Benign / minus_F minus Full | 6 | -0.072 ± 1.082 | -0.00542 ± 0.00917 | 0.00436 ± 0.01494 |
| IID / F Flip / Full | 6 | 88.288 ± 0.746 | 0.01484 ± 0.01041 | 0.05808 ± 0.01290 |
| IID / F Flip / minus_F | 6 | 88.333 ± 0.500 | 0.01745 ± 0.00682 | 0.07303 ± 0.01359 |
| IID / F Flip / minus_F minus Full | 6 | 0.045 ± 0.928 | 0.00261 ± 0.01214 | 0.01496 ± 0.01590 |

## raw — All 10 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 10 | 88.549 ± 1.038 | 0.04027 ± 0.01035 | 0.10308 ± 0.01007 |
| IID / Benign / minus_F | 10 | 88.778 ± 0.921 | 0.04605 ± 0.00944 | 0.10965 ± 0.00771 |
| IID / Benign / minus_F minus Full | 10 | 0.229 ± 1.374 | 0.00578 ± 0.01538 | 0.00658 ± 0.01550 |
| IID / F Flip / Full | 10 | 88.703 ± 0.733 | 0.03684 ± 0.01343 | 0.10080 ± 0.01032 |
| IID / F Flip / minus_F | 10 | 88.695 ± 1.220 | 0.04510 ± 0.01187 | 0.10941 ± 0.01526 |
| IID / F Flip / minus_F minus Full | 10 | -0.009 ± 1.303 | 0.00826 ± 0.01806 | 0.00861 ± 0.01499 |

## raw — Exclude selection seed: 9 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 9 | 88.646 ± 1.052 | 0.04004 ± 0.01095 | 0.10338 ± 0.01063 |
| IID / Benign / minus_F | 9 | 88.978 ± 0.709 | 0.04401 ± 0.00729 | 0.10907 ± 0.00794 |
| IID / Benign / minus_F minus Full | 9 | 0.333 ± 1.415 | 0.00397 ± 0.01514 | 0.00569 ± 0.01616 |
| IID / F Flip / Full | 9 | 88.731 ± 0.772 | 0.03770 ± 0.01396 | 0.10200 ± 0.01017 |
| IID / F Flip / minus_F | 9 | 89.026 ± 0.661 | 0.04664 ± 0.01148 | 0.11306 ± 0.01058 |
| IID / F Flip / minus_F minus Full | 9 | 0.296 ± 0.932 | 0.00894 ± 0.01902 | 0.01106 ± 0.01361 |

## raw — Seeds 91005–91010: 6 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 6 | 88.813 ± 0.633 | 0.04225 ± 0.01270 | 0.10710 ± 0.00862 |
| IID / Benign / minus_F | 6 | 88.821 ± 0.706 | 0.04104 ± 0.00709 | 0.10609 ± 0.00591 |
| IID / Benign / minus_F minus Full | 6 | 0.008 ± 1.004 | -0.00121 ± 0.01600 | -0.00101 ± 0.01015 |
| IID / F Flip / Full | 6 | 88.651 ± 0.877 | 0.03646 ± 0.00936 | 0.10113 ± 0.00686 |
| IID / F Flip / minus_F | 6 | 88.728 ± 0.564 | 0.04872 ± 0.01196 | 0.11191 ± 0.01134 |
| IID / F Flip / minus_F minus Full | 6 | 0.076 ± 1.088 | 0.01226 ± 0.02037 | 0.01078 ± 0.01345 |

## shared_calibration — All 10 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| IID / Benign / minus_F | 10 | 88.472 ± 0.897 | 0.00632 ± 0.00425 | 0.06549 ± 0.01498 |
| IID / Benign / minus_F minus Full | 10 | 0.214 ± 1.451 | -0.00341 ± 0.00804 | 0.00295 ± 0.02036 |
| IID / F Flip / Full | 10 | 88.391 ± 0.626 | 0.01068 ± 0.00958 | 0.06068 ± 0.01026 |
| IID / F Flip / minus_F | 10 | 88.325 ± 1.212 | 0.01439 ± 0.00651 | 0.06600 ± 0.01819 |
| IID / F Flip / minus_F minus Full | 10 | -0.066 ± 1.239 | 0.00372 ± 0.00942 | 0.00532 ± 0.02066 |

## shared_calibration — Exclude selection seed: 9 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| IID / Benign / minus_F | 9 | 88.643 ± 0.761 | 0.00549 ± 0.00355 | 0.07012 ± 0.00338 |
| IID / Benign / minus_F minus Full | 9 | 0.291 ± 1.517 | -0.00426 ± 0.00805 | 0.00780 ± 0.01422 |
| IID / F Flip / Full | 9 | 88.412 ± 0.660 | 0.01147 ± 0.00981 | 0.06043 ± 0.01085 |
| IID / F Flip / minus_F | 9 | 88.657 ± 0.642 | 0.01473 ± 0.00681 | 0.07064 ± 0.01141 |
| IID / F Flip / minus_F minus Full | 9 | 0.244 ± 0.802 | 0.00327 ± 0.00987 | 0.01021 ± 0.01452 |

## shared_calibration — Seeds 91005–91010: 6 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| IID / Benign / minus_F | 6 | 88.486 ± 0.758 | 0.00596 ± 0.00371 | 0.07085 ± 0.00146 |
| IID / Benign / minus_F minus Full | 6 | -0.072 ± 1.082 | -0.00542 ± 0.00917 | 0.00436 ± 0.01494 |
| IID / F Flip / Full | 6 | 88.288 ± 0.746 | 0.01484 ± 0.01041 | 0.05808 ± 0.01290 |
| IID / F Flip / minus_F | 6 | 88.333 ± 0.500 | 0.01745 ± 0.00682 | 0.07303 ± 0.01359 |
| IID / F Flip / minus_F minus Full | 6 | 0.045 ± 0.928 | 0.00261 ± 0.01214 | 0.01496 ± 0.01590 |

AEOD = absolute TPR gap, not full equalized odds. Raw uses margin > 0 (ties predict0); native retains the original root-only calibration recipe; shared_calibration uses the frozen common root-only rule. Native/shared metrics and counts are compared below, not treated as independent evidence of calibration gain. No threshold is refitted by this builder.
Complete-scene replay devices: Full {'cpu': 2, 'cuda:0': 18}; minus_F {'cpu': 20}. Training Torch: Full {'2.11.0+cu128': 20}; minus_F {'2.11.0+cu128': 20}. Historical/current driver provenance is retained per source record. The broader Full100 history includes98 cu128 and2 cu130 records; that fact does not turn these selected scenes into a unified-device comparison.
Seed91001 participated in recipe selection; validation was exposed during development, and historical test exposure remains disclosed. The9/6 panels apply identical predefined seeds to both variants and are descriptive subsets, not untouched confirmation sets.
Only IID Benign and IID F Flip seeds91001–91010 enter this two-scene table. No cross-scene aggregate is computed or published; scenes are not independent seeds. The other eight minus_F scenes and other control coverages remain incomplete. This is not F100 or all mechanism controls. No significance, necessity, causal-isolation, final-test or whole-rebuttal-completion claim is made.
