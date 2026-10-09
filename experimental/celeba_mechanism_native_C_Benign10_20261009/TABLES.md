# CelebA: native Full–minus_C, IID Benign

Validation-only, round 70, valid n=19,867. Ten shared declared seeds; mean ± sample SD (ddof=1). The paired row is minus_C − Full, computed within seed before summarizing. ACC is in percent; its paired difference is in percentage points.

This fixed native112 snapshot contains Full100, minus_U100 and minus_C12. Only IID Benign is complete for C; IID F Flip has 2/10 seeds and the other eight C scenes have 0/10. No incomplete C scene contributes to this table.

## All 10 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| minus_C | 10 | 88.175 ± 1.317 | 0.01265 ± 0.01127 | 0.06112 ± 0.01532 |
| minus_C minus Full | 10 | -0.083 ± 1.378 | 0.00292 ± 0.01315 | -0.00142 ± 0.01953 |

## Exclude selection seed: 9 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| minus_C | 9 | 88.512 ± 0.819 | 0.01400 ± 0.01106 | 0.06365 ± 0.01386 |
| minus_C minus Full | 9 | 0.161 ± 1.212 | 0.00425 ± 0.01321 | 0.00132 ± 0.01855 |

## Seeds 91005–91010: 6 seeds

| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |
|---|---:|---:|---:|---:|
| Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| minus_C | 6 | 88.219 ± 0.812 | 0.01106 ± 0.01017 | 0.07095 ± 0.00989 |
| minus_C minus Full | 6 | -0.339 ± 0.696 | -0.00032 ± 0.00966 | 0.00446 ± 0.02188 |

AEOD is the absolute TPR gap, not full equalized odds; smaller AEOD/ASPD indicates less disparity. Native metrics retain each procedure’s original root-fitted calibration. This single deletion does not isolate calibration effects or establish that C is necessary or causal.

The 9-seed panel omits selection seed 91001; the 6-seed panel retains 91005–91010 for both variants. All seeds have prior validation exposure; neither panel is an untouched confirmation set. Historical test exposure and validation-based recipe selection remain limitations; no new test use occurs here.

Full reuses historical checkpoints; all ten shown Full and C records use PyTorch 2.11.0+cu128, while their historical/current driver environments differ (current C driver 595.84). The full reference cohort includes 98 cu128 and 2 cu130 records, but neither cu130 record is in this scene. This native table uses accepted original training metrics, not mixed-device three-view replay results. The formal native/shared primary endpoint remains pending author selection.

No significance test, superiority guarantee, new inference or training is performed. Source identities and per-seed paired differences are preserved in the accompanying JSON. This single scene does not complete the set of eight mechanism controls or the complete mechanism900 comparison.

Inspection SHA256: `bf9f762ef3fb3c16582fc4f9b4548bda61f792f45021c465e80349b4422b5151`. Original statistics source SHA256: `3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef`.
