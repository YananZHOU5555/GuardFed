# CelebA Full–minus_C: IID Benign, F Flip and FedSA, three views

Three complete scenes, thirty matched Full–C checkpoint pairs, valid-only (19,867), round70. Mean ± sample SD (ddof=1); differences are minus_C − Full. ACC is percent; ΔACC is percentage points. ACC higher and gaps lower are better. Primary endpoint remains pending.

## native — All 10 seeds

| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID Benign | Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| IID Benign | minus_C | 10 | 88.175 ± 1.317 | 0.01265 ± 0.01127 | 0.06112 ± 0.01532 |
| IID Benign | minus_C minus Full | 10 | -0.083 ± 1.378 | 0.00292 ± 0.01315 | -0.00142 ± 0.01953 |
| IID F Flip | Full | 10 | 88.391 ± 0.626 | 0.01068 ± 0.00958 | 0.06068 ± 0.01026 |
| IID F Flip | minus_C | 10 | 88.909 ± 0.783 | 0.01207 ± 0.00962 | 0.07167 ± 0.01580 |
| IID F Flip | minus_C minus Full | 10 | 0.518 ± 0.814 | 0.00139 ± 0.01487 | 0.01098 ± 0.02176 |
| IID FedSA | Full | 10 | 88.470 ± 0.909 | 0.00607 ± 0.00416 | 0.06402 ± 0.01425 |
| IID FedSA | minus_C | 10 | 88.749 ± 0.805 | 0.00584 ± 0.00414 | 0.06500 ± 0.00669 |
| IID FedSA | minus_C minus Full | 10 | 0.279 ± 0.498 | -0.00024 ± 0.00472 | 0.00099 ± 0.01311 |

## native — Exclude selection seed: 9 seeds

| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID Benign | Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| IID Benign | minus_C | 9 | 88.512 ± 0.819 | 0.01400 ± 0.01106 | 0.06365 ± 0.01386 |
| IID Benign | minus_C minus Full | 9 | 0.161 ± 1.212 | 0.00425 ± 0.01321 | 0.00132 ± 0.01855 |
| IID F Flip | Full | 9 | 88.412 ± 0.660 | 0.01147 ± 0.00981 | 0.06043 ± 0.01085 |
| IID F Flip | minus_C | 9 | 89.029 ± 0.726 | 0.01172 ± 0.01014 | 0.07218 ± 0.01667 |
| IID F Flip | minus_C minus Full | 9 | 0.616 ± 0.797 | 0.00025 ± 0.01530 | 0.01175 ± 0.02294 |
| IID FedSA | Full | 9 | 88.622 ± 0.819 | 0.00544 ± 0.00388 | 0.06675 ± 0.01202 |
| IID FedSA | minus_C | 9 | 88.841 ± 0.797 | 0.00571 ± 0.00437 | 0.06441 ± 0.00681 |
| IID FedSA | minus_C minus Full | 9 | 0.219 ± 0.489 | 0.00026 ± 0.00472 | -0.00233 ± 0.00832 |

## native — Seeds 91005–91010: 6 seeds

| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID Benign | Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| IID Benign | minus_C | 6 | 88.219 ± 0.812 | 0.01106 ± 0.01017 | 0.07095 ± 0.00989 |
| IID Benign | minus_C minus Full | 6 | -0.339 ± 0.696 | -0.00032 ± 0.00966 | 0.00446 ± 0.02188 |
| IID F Flip | Full | 6 | 88.288 ± 0.746 | 0.01484 ± 0.01041 | 0.05808 ± 0.01290 |
| IID F Flip | minus_C | 6 | 88.953 ± 0.876 | 0.01193 ± 0.01091 | 0.07928 ± 0.01311 |
| IID F Flip | minus_C minus Full | 6 | 0.665 ± 0.997 | -0.00291 ± 0.01773 | 0.02120 ± 0.02060 |
| IID FedSA | Full | 6 | 88.569 ± 0.867 | 0.00550 ± 0.00361 | 0.06965 ± 0.01099 |
| IID FedSA | minus_C | 6 | 88.717 ± 0.911 | 0.00566 ± 0.00427 | 0.06372 ± 0.00750 |
| IID FedSA | minus_C minus Full | 6 | 0.148 ± 0.569 | 0.00016 ± 0.00415 | -0.00593 ± 0.00634 |

## raw — All 10 seeds

| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID Benign | Full | 10 | 88.549 ± 1.038 | 0.04027 ± 0.01035 | 0.10308 ± 0.01007 |
| IID Benign | minus_C | 10 | 88.535 ± 1.226 | 0.03544 ± 0.00804 | 0.09925 ± 0.01001 |
| IID Benign | minus_C minus Full | 10 | -0.014 ± 1.385 | -0.00483 ± 0.01119 | -0.00383 ± 0.01524 |
| IID F Flip | Full | 10 | 88.703 ± 0.733 | 0.03684 ± 0.01343 | 0.10080 ± 0.01032 |
| IID F Flip | minus_C | 10 | 89.233 ± 0.770 | 0.04149 ± 0.00883 | 0.10783 ± 0.00966 |
| IID F Flip | minus_C minus Full | 10 | 0.530 ± 0.875 | 0.00465 ± 0.01053 | 0.00703 ± 0.01157 |
| IID FedSA | Full | 10 | 88.832 ± 0.899 | 0.03716 ± 0.00699 | 0.10257 ± 0.01155 |
| IID FedSA | minus_C | 10 | 89.067 ± 0.783 | 0.03584 ± 0.00831 | 0.10399 ± 0.00893 |
| IID FedSA | minus_C minus Full | 10 | 0.235 ± 0.535 | -0.00132 ± 0.01034 | 0.00142 ± 0.01245 |

## raw — Exclude selection seed: 9 seeds

| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID Benign | Full | 9 | 88.646 ± 1.052 | 0.04004 ± 0.01095 | 0.10338 ± 0.01063 |
| IID Benign | minus_C | 9 | 88.827 ± 0.855 | 0.03710 ± 0.00648 | 0.10213 ± 0.00439 |
| IID Benign | minus_C minus Full | 9 | 0.181 ± 1.315 | -0.00294 ± 0.01004 | -0.00125 ± 0.01366 |
| IID F Flip | Full | 9 | 88.731 ± 0.772 | 0.03770 ± 0.01396 | 0.10200 ± 0.01017 |
| IID F Flip | minus_C | 9 | 89.348 ± 0.720 | 0.04256 ± 0.00865 | 0.11002 ± 0.00713 |
| IID F Flip | minus_C minus Full | 9 | 0.617 ± 0.880 | 0.00486 ± 0.01115 | 0.00802 ± 0.01181 |
| IID FedSA | Full | 9 | 89.001 ± 0.767 | 0.03797 ± 0.00689 | 0.10454 ± 0.01032 |
| IID FedSA | minus_C | 9 | 89.160 ± 0.770 | 0.03520 ± 0.00855 | 0.10382 ± 0.00945 |
| IID FedSA | minus_C minus Full | 9 | 0.158 ± 0.505 | -0.00277 ± 0.00983 | -0.00072 ± 0.01109 |

## raw — Seeds 91005–91010: 6 seeds

| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID Benign | Full | 6 | 88.813 ± 0.633 | 0.04225 ± 0.01270 | 0.10710 ± 0.00862 |
| IID Benign | minus_C | 6 | 88.494 ± 0.812 | 0.03809 ± 0.00776 | 0.10094 ± 0.00496 |
| IID Benign | minus_C minus Full | 6 | -0.319 ± 0.678 | -0.00416 ± 0.01235 | -0.00617 ± 0.01176 |
| IID F Flip | Full | 6 | 88.651 ± 0.877 | 0.03646 ± 0.00936 | 0.10113 ± 0.00686 |
| IID F Flip | minus_C | 6 | 89.287 ± 0.872 | 0.04425 ± 0.00831 | 0.11101 ± 0.00666 |
| IID F Flip | minus_C minus Full | 6 | 0.636 ± 1.099 | 0.00779 ± 0.00863 | 0.00988 ± 0.01213 |
| IID FedSA | Full | 6 | 88.894 ± 0.796 | 0.03453 ± 0.00524 | 0.10012 ± 0.00832 |
| IID FedSA | minus_C | 6 | 89.099 ± 0.942 | 0.03486 ± 0.00821 | 0.10253 ± 0.00705 |
| IID FedSA | minus_C minus Full | 6 | 0.206 ± 0.556 | 0.00033 ± 0.00842 | 0.00241 ± 0.01063 |

## shared_calibration — All 10 seeds

| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID Benign | Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| IID Benign | minus_C | 10 | 88.175 ± 1.317 | 0.01265 ± 0.01127 | 0.06112 ± 0.01532 |
| IID Benign | minus_C minus Full | 10 | -0.083 ± 1.378 | 0.00292 ± 0.01315 | -0.00142 ± 0.01953 |
| IID F Flip | Full | 10 | 88.391 ± 0.626 | 0.01068 ± 0.00958 | 0.06068 ± 0.01026 |
| IID F Flip | minus_C | 10 | 88.909 ± 0.783 | 0.01207 ± 0.00962 | 0.07167 ± 0.01580 |
| IID F Flip | minus_C minus Full | 10 | 0.518 ± 0.814 | 0.00139 ± 0.01487 | 0.01098 ± 0.02176 |
| IID FedSA | Full | 10 | 88.470 ± 0.909 | 0.00607 ± 0.00416 | 0.06402 ± 0.01425 |
| IID FedSA | minus_C | 10 | 88.749 ± 0.805 | 0.00584 ± 0.00414 | 0.06500 ± 0.00669 |
| IID FedSA | minus_C minus Full | 10 | 0.279 ± 0.498 | -0.00024 ± 0.00472 | 0.00099 ± 0.01311 |

## shared_calibration — Exclude selection seed: 9 seeds

| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID Benign | Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| IID Benign | minus_C | 9 | 88.512 ± 0.819 | 0.01400 ± 0.01106 | 0.06365 ± 0.01386 |
| IID Benign | minus_C minus Full | 9 | 0.161 ± 1.212 | 0.00425 ± 0.01321 | 0.00132 ± 0.01855 |
| IID F Flip | Full | 9 | 88.412 ± 0.660 | 0.01147 ± 0.00981 | 0.06043 ± 0.01085 |
| IID F Flip | minus_C | 9 | 89.029 ± 0.726 | 0.01172 ± 0.01014 | 0.07218 ± 0.01667 |
| IID F Flip | minus_C minus Full | 9 | 0.616 ± 0.797 | 0.00025 ± 0.01530 | 0.01175 ± 0.02294 |
| IID FedSA | Full | 9 | 88.622 ± 0.819 | 0.00544 ± 0.00388 | 0.06675 ± 0.01202 |
| IID FedSA | minus_C | 9 | 88.841 ± 0.797 | 0.00571 ± 0.00437 | 0.06441 ± 0.00681 |
| IID FedSA | minus_C minus Full | 9 | 0.219 ± 0.489 | 0.00026 ± 0.00472 | -0.00233 ± 0.00832 |

## shared_calibration — Seeds 91005–91010: 6 seeds

| Scene | Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID Benign | Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| IID Benign | minus_C | 6 | 88.219 ± 0.812 | 0.01106 ± 0.01017 | 0.07095 ± 0.00989 |
| IID Benign | minus_C minus Full | 6 | -0.339 ± 0.696 | -0.00032 ± 0.00966 | 0.00446 ± 0.02188 |
| IID F Flip | Full | 6 | 88.288 ± 0.746 | 0.01484 ± 0.01041 | 0.05808 ± 0.01290 |
| IID F Flip | minus_C | 6 | 88.953 ± 0.876 | 0.01193 ± 0.01091 | 0.07928 ± 0.01311 |
| IID F Flip | minus_C minus Full | 6 | 0.665 ± 0.997 | -0.00291 ± 0.01773 | 0.02120 ± 0.02060 |
| IID FedSA | Full | 6 | 88.569 ± 0.867 | 0.00550 ± 0.00361 | 0.06965 ± 0.01099 |
| IID FedSA | minus_C | 6 | 88.717 ± 0.911 | 0.00566 ± 0.00427 | 0.06372 ± 0.00750 |
| IID FedSA | minus_C minus Full | 6 | 0.148 ± 0.569 | 0.00016 ± 0.00415 | -0.00593 ± 0.00634 |

AEOD is absolute TPR gap, not full equalized odds. Native retains each original root-only calibration; raw is uncalibrated; shared uses the frozen common root-only rule. All views use each model’s same checkpoint. No model inference or threshold refit occurs in this table builder.
Mixed CPU/GPU replay, training CUDA/driver differences, seed91001 recipe selection, prior validation/test exposure and pending author endpoint remain disclosed. 9/6 panels are descriptive subsets with identical seeds for Full and C, not untouched confirmation sets.
Only C IID Benign, F Flip and FedSA are complete. Six accepted S-DFA C records are retained separately and excluded from all means; seven other C scenes and the full mechanism grid remain incomplete. No C necessity/causality, significance, final-test or whole-rebuttal completion claim.
