# CelebA Full–minus_C: IID Benign, three views

One complete scene, ten shared terminal checkpoints, valid-only (19,867), round70. Mean ± sample SD, ddof=1; paired differences are minus_C − Full. ACC is percent and its paired difference is percentage points. Primary endpoint remains pending.

## native — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| minus_C | 10 | 88.175 ± 1.317 | 0.01265 ± 0.01127 | 0.06112 ± 0.01532 |
| minus_C minus Full | 10 | -0.083 ± 1.378 | 0.00292 ± 0.01315 | -0.00142 ± 0.01953 |

## native — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| minus_C | 9 | 88.512 ± 0.819 | 0.01400 ± 0.01106 | 0.06365 ± 0.01386 |
| minus_C minus Full | 9 | 0.161 ± 1.212 | 0.00425 ± 0.01321 | 0.00132 ± 0.01855 |

## native — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| minus_C | 6 | 88.219 ± 0.812 | 0.01106 ± 0.01017 | 0.07095 ± 0.00989 |
| minus_C minus Full | 6 | -0.339 ± 0.696 | -0.00032 ± 0.00966 | 0.00446 ± 0.02188 |

## raw — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.549 ± 1.038 | 0.04027 ± 0.01035 | 0.10308 ± 0.01007 |
| minus_C | 10 | 88.535 ± 1.226 | 0.03544 ± 0.00804 | 0.09925 ± 0.01001 |
| minus_C minus Full | 10 | -0.014 ± 1.385 | -0.00483 ± 0.01119 | -0.00383 ± 0.01524 |

## raw — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.646 ± 1.052 | 0.04004 ± 0.01095 | 0.10338 ± 0.01063 |
| minus_C | 9 | 88.827 ± 0.855 | 0.03710 ± 0.00648 | 0.10213 ± 0.00439 |
| minus_C minus Full | 9 | 0.181 ± 1.315 | -0.00294 ± 0.01004 | -0.00125 ± 0.01366 |

## raw — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.813 ± 0.633 | 0.04225 ± 0.01270 | 0.10710 ± 0.00862 |
| minus_C | 6 | 88.494 ± 0.812 | 0.03809 ± 0.00776 | 0.10094 ± 0.00496 |
| minus_C minus Full | 6 | -0.319 ± 0.678 | -0.00416 ± 0.01235 | -0.00617 ± 0.01176 |

## shared_calibration — All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| minus_C | 10 | 88.175 ± 1.317 | 0.01265 ± 0.01127 | 0.06112 ± 0.01532 |
| minus_C minus Full | 10 | -0.083 ± 1.378 | 0.00292 ± 0.01315 | -0.00142 ± 0.01953 |

## shared_calibration — Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| minus_C | 9 | 88.512 ± 0.819 | 0.01400 ± 0.01106 | 0.06365 ± 0.01386 |
| minus_C minus Full | 9 | 0.161 ± 1.212 | 0.00425 ± 0.01321 | 0.00132 ± 0.01855 |

## shared_calibration — Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| minus_C | 6 | 88.219 ± 0.812 | 0.01106 ± 0.01017 | 0.07095 ± 0.00989 |
| minus_C minus Full | 6 | -0.339 ± 0.696 | -0.00032 ± 0.00966 | 0.00446 ± 0.02188 |

AEOD is absolute TPR gap, not full equalized odds. Native retains each procedure’s original root-only calibration; raw is uncalibrated; shared uses the frozen common root-only rule. All views use the same terminal checkpoint. No new calibration fit or Full inference is performed by this table builder.
IID F Flip has only two paired seeds; all four records are retained for identity/coverage but excluded from every mean. Other eight C scenes remain absent. Mixed CPU/GPU replay, historical/current CUDA/driver provenance, seed91001 selection, prior validation/test exposure and author-pending primary endpoint remain limitations. The 9/6 panels apply the same seed rule to both variants and are descriptive, not untouched confirmation sets.
No significance test, C necessity/causality claim, final test, or completed mechanism900 claim.
