# CelebA Full–minus_A: two complete IID scenes, three views

IID Benign and IID F Flip only. Validation (19,867 images), round70; ten paired model seeds per scene, with the same checkpoint across raw/native/shared views. Mean ± sample SD (ddof=1). Paired difference = minus_A − Full; ACC is percent and ΔACC is percentage points. AEOD and ASPD are lower-is-better disparities. Formal primary endpoint remains pending.

## native — All 10 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| IID / Benign / minus_A | 10 | 87.829 ± 1.625 | 0.01205 ± 0.00918 | 0.05940 ± 0.00995 |
| IID / Benign / minus_A minus Full | 10 | -0.430 ± 1.259 | 0.00231 ± 0.01461 | -0.00314 ± 0.01805 |
| IID / F Flip / Full | 10 | 88.391 ± 0.626 | 0.01068 ± 0.00958 | 0.06068 ± 0.01026 |
| IID / F Flip / minus_A | 10 | 88.392 ± 1.190 | 0.01069 ± 0.00706 | 0.05980 ± 0.01789 |
| IID / F Flip / minus_A minus Full | 10 | 0.001 ± 1.030 | 0.00001 ± 0.01228 | -0.00088 ± 0.02417 |

## native — Exclude selection seed: 9 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| IID / Benign / minus_A | 9 | 88.265 ± 0.908 | 0.01077 ± 0.00874 | 0.05949 ± 0.01054 |
| IID / Benign / minus_A minus Full | 9 | -0.087 ± 0.678 | 0.00102 ± 0.01488 | -0.00283 ± 0.01912 |
| IID / F Flip / Full | 9 | 88.412 ± 0.660 | 0.01147 ± 0.00981 | 0.06043 ± 0.01085 |
| IID / F Flip / minus_A | 9 | 88.382 ± 1.262 | 0.01122 ± 0.00728 | 0.05889 ± 0.01872 |
| IID / F Flip / minus_A minus Full | 9 | -0.030 ± 1.088 | -0.00024 ± 0.01300 | -0.00154 ± 0.02554 |

## native — Seeds 91005–91010: 6 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| IID / Benign / minus_A | 6 | 88.248 ± 0.624 | 0.01098 ± 0.00794 | 0.06189 ± 0.00863 |
| IID / Benign / minus_A minus Full | 6 | -0.310 ± 0.703 | -0.00040 ± 0.01398 | -0.00460 ± 0.02052 |
| IID / F Flip / Full | 6 | 88.288 ± 0.746 | 0.01484 ± 0.01041 | 0.05808 ± 0.01290 |
| IID / F Flip / minus_A | 6 | 88.090 ± 1.453 | 0.00958 ± 0.00537 | 0.05617 ± 0.01602 |
| IID / F Flip / minus_A minus Full | 6 | -0.198 ± 1.213 | -0.00526 ± 0.01077 | -0.00190 ± 0.02730 |

## raw — All 10 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 10 | 88.549 ± 1.038 | 0.04027 ± 0.01035 | 0.10308 ± 0.01007 |
| IID / Benign / minus_A | 10 | 88.214 ± 1.695 | 0.04092 ± 0.00933 | 0.10124 ± 0.00931 |
| IID / Benign / minus_A minus Full | 10 | -0.335 ± 1.319 | 0.00064 ± 0.01538 | -0.00184 ± 0.01319 |
| IID / F Flip / Full | 10 | 88.703 ± 0.733 | 0.03684 ± 0.01343 | 0.10080 ± 0.01032 |
| IID / F Flip / minus_A | 10 | 88.774 ± 1.169 | 0.03945 ± 0.01694 | 0.10525 ± 0.00936 |
| IID / F Flip / minus_A minus Full | 10 | 0.070 ± 1.194 | 0.00261 ± 0.01613 | 0.00445 ± 0.00985 |

## raw — Exclude selection seed: 9 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 9 | 88.646 ± 1.052 | 0.04004 ± 0.01095 | 0.10338 ± 0.01063 |
| IID / Benign / minus_A | 9 | 88.676 ± 0.910 | 0.04071 ± 0.00987 | 0.10366 ± 0.00560 |
| IID / Benign / minus_A minus Full | 9 | 0.030 ± 0.675 | 0.00067 ± 0.01631 | 0.00028 ± 0.01204 |
| IID / F Flip / Full | 9 | 88.731 ± 0.772 | 0.03770 ± 0.01396 | 0.10200 ± 0.01017 |
| IID / F Flip / minus_A | 9 | 88.763 ± 1.240 | 0.03930 ± 0.01796 | 0.10519 ± 0.00992 |
| IID / F Flip / minus_A minus Full | 9 | 0.032 ± 1.260 | 0.00161 ± 0.01677 | 0.00318 ± 0.00954 |

## raw — Seeds 91005–91010: 6 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 6 | 88.813 ± 0.633 | 0.04225 ± 0.01270 | 0.10710 ± 0.00862 |
| IID / Benign / minus_A | 6 | 88.683 ± 0.631 | 0.04186 ± 0.00928 | 0.10352 ± 0.00430 |
| IID / Benign / minus_A minus Full | 6 | -0.130 ± 0.778 | -0.00040 ± 0.01774 | -0.00358 ± 0.01009 |
| IID / F Flip / Full | 6 | 88.651 ± 0.877 | 0.03646 ± 0.00936 | 0.10113 ± 0.00686 |
| IID / F Flip / minus_A | 6 | 88.493 ± 1.430 | 0.04089 ± 0.02240 | 0.10482 ± 0.01182 |
| IID / F Flip / minus_A minus Full | 6 | -0.159 ± 1.343 | 0.00443 ± 0.01417 | 0.00369 ± 0.00911 |

## shared_calibration — All 10 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| IID / Benign / minus_A | 10 | 87.829 ± 1.625 | 0.01205 ± 0.00918 | 0.05940 ± 0.00995 |
| IID / Benign / minus_A minus Full | 10 | -0.430 ± 1.259 | 0.00231 ± 0.01461 | -0.00314 ± 0.01805 |
| IID / F Flip / Full | 10 | 88.391 ± 0.626 | 0.01068 ± 0.00958 | 0.06068 ± 0.01026 |
| IID / F Flip / minus_A | 10 | 88.392 ± 1.190 | 0.01069 ± 0.00706 | 0.05980 ± 0.01789 |
| IID / F Flip / minus_A minus Full | 10 | 0.001 ± 1.030 | 0.00001 ± 0.01228 | -0.00088 ± 0.02417 |

## shared_calibration — Exclude selection seed: 9 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| IID / Benign / minus_A | 9 | 88.265 ± 0.908 | 0.01077 ± 0.00874 | 0.05949 ± 0.01054 |
| IID / Benign / minus_A minus Full | 9 | -0.087 ± 0.678 | 0.00102 ± 0.01488 | -0.00283 ± 0.01912 |
| IID / F Flip / Full | 9 | 88.412 ± 0.660 | 0.01147 ± 0.00981 | 0.06043 ± 0.01085 |
| IID / F Flip / minus_A | 9 | 88.382 ± 1.262 | 0.01122 ± 0.00728 | 0.05889 ± 0.01872 |
| IID / F Flip / minus_A minus Full | 9 | -0.030 ± 1.088 | -0.00024 ± 0.01300 | -0.00154 ± 0.02554 |

## shared_calibration — Seeds 91005–91010: 6 seeds

| Scene / variant or paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| IID / Benign / Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| IID / Benign / minus_A | 6 | 88.248 ± 0.624 | 0.01098 ± 0.00794 | 0.06189 ± 0.00863 |
| IID / Benign / minus_A minus Full | 6 | -0.310 ± 0.703 | -0.00040 ± 0.01398 | -0.00460 ± 0.02052 |
| IID / F Flip / Full | 6 | 88.288 ± 0.746 | 0.01484 ± 0.01041 | 0.05808 ± 0.01290 |
| IID / F Flip / minus_A | 6 | 88.090 ± 1.453 | 0.00958 ± 0.00537 | 0.05617 ± 0.01602 |
| IID / F Flip / minus_A minus Full | 6 | -0.198 ± 1.213 | -0.00526 ± 0.01077 | -0.00190 ± 0.02730 |

AEOD is the absolute TPR gap, not full equalized odds. Raw uses margin > 0 (ties predict0). Native retains each original root-only calibration recipe; shared_calibration uses the frozen common root-only rule. The three views are parallel descriptions, with no endpoint selected or threshold refitted by this builder.
Actual replay devices across20 pairs: Full {'cpu': 2, 'cuda:0': 18}; minus_A {'cpu': 20}. Training Torch: Full {'2.11.0+cu128': 20}; minus_A {'2.11.0+cu128': 20}. Per-record configuration, source, checkpoint, environment and driver provenance remain in records.json. Broader Full100 history includes98 cu128 and2 cu130 records; these20 actual source records, not that broader count, define this table.
Seed91001 participated in recipe selection; validation was exposed during development, and historical test exposure remains disclosed. The9/6 panels apply identical predefined seeds to both variants and are descriptive subsets, not untouched confirmation sets. No final test was run for this table.
All negative and constant outcomes are retained. These are two of ten minus_A scenes; the other eight scenes remain outside this delivery. No scene-pooled mean is reported, and scenes are not treated as independent model seeds. This is not A100 or completion of all mechanism controls. No significance, necessity, causal-isolation or whole-rebuttal-completion claim is made.
