# CosineFairnessHybrid: two IID scenes, three validation views

Twenty fixed round-70 checkpoints across two separately reported scenes, 19,867 validation images; raw/native/shared use the same checkpoint and seed panel. Mean ± sample SD (ddof=1). ACC (%) ↑; AEOD and ASPD ↓. AEOD is the absolute TPR gap, not full equalized odds. Formal primary endpoint remains pending.

## IID / Benign

| View | Fixed seed panel | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |
|---|---|---:|---:|---:|---:|
| native | All 10 seeds | 10 | 88.027 ± 1.386 | 0.02920 ± 0.00903 | 0.09251 ± 0.01410 |
| native | Exclude selection seed: 9 seeds | 9 | 88.285 ± 1.189 | 0.02800 ± 0.00869 | 0.09400 ± 0.01409 |
| native | Seeds 91005–91010: 6 seeds | 6 | 88.109 ± 1.360 | 0.02646 ± 0.00982 | 0.08913 ± 0.01495 |
| raw | All 10 seeds | 10 | 88.027 ± 1.386 | 0.02920 ± 0.00903 | 0.09251 ± 0.01410 |
| raw | Exclude selection seed: 9 seeds | 9 | 88.285 ± 1.189 | 0.02800 ± 0.00869 | 0.09400 ± 0.01409 |
| raw | Seeds 91005–91010: 6 seeds | 6 | 88.109 ± 1.360 | 0.02646 ± 0.00982 | 0.08913 ± 0.01495 |
| shared_calibration | All 10 seeds | 10 | 87.708 ± 1.270 | 0.02321 ± 0.01652 | 0.03383 ± 0.01892 |
| shared_calibration | Exclude selection seed: 9 seeds | 9 | 87.969 ± 1.021 | 0.02433 ± 0.01712 | 0.03268 ± 0.01969 |
| shared_calibration | Seeds 91005–91010: 6 seeds | 6 | 87.806 ± 1.154 | 0.01669 ± 0.01308 | 0.03565 ± 0.02085 |

## IID / F Flip

| View | Fixed seed panel | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |
|---|---|---:|---:|---:|---:|
| native | All 10 seeds | 10 | 89.010 ± 1.128 | 0.03192 ± 0.00767 | 0.09939 ± 0.01030 |
| native | Exclude selection seed: 9 seeds | 9 | 89.152 ± 1.098 | 0.03088 ± 0.00734 | 0.09966 ± 0.01089 |
| native | Seeds 91005–91010: 6 seeds | 6 | 88.819 ± 1.228 | 0.03151 ± 0.00738 | 0.09725 ± 0.01136 |
| raw | All 10 seeds | 10 | 89.010 ± 1.128 | 0.03192 ± 0.00767 | 0.09939 ± 0.01030 |
| raw | Exclude selection seed: 9 seeds | 9 | 89.152 ± 1.098 | 0.03088 ± 0.00734 | 0.09966 ± 0.01089 |
| raw | Seeds 91005–91010: 6 seeds | 6 | 88.819 ± 1.228 | 0.03151 ± 0.00738 | 0.09725 ± 0.01136 |
| shared_calibration | All 10 seeds | 10 | 88.600 ± 1.187 | 0.00981 ± 0.00772 | 0.05572 ± 0.01848 |
| shared_calibration | Exclude selection seed: 9 seeds | 9 | 88.752 ± 1.151 | 0.00846 ± 0.00682 | 0.05887 ± 0.01651 |
| shared_calibration | Seeds 91005–91010: 6 seeds | 6 | 88.440 ± 1.316 | 0.00850 ± 0.00836 | 0.05605 ± 0.01985 |

Benign seed91001 is the originally selected screen checkpoint; its other nine and all ten F Flip checkpoints retain their formal-coverage phase. Seed91001 was exposed during recipe selection in both conditions. The all10 panel retains the selected seed; n9 excludes only91001; n6 fixes91005–91010. Validation was exposed during development, including development after the initial official-test results were viewed. These descriptive subsets are not untouched confirmation sets; no new final-test evaluation was performed.
Raw and native retain margin > 0 (zero-margin ties predict0); shared_calibration uses the frozen common clean-training-root-only thresholds with margin ≥ group threshold. This table only reads saved accepted receipts: it performs no calibration fit, forward pass or training. Any raw/native equality is retained as measured, not treated as independent confirmation. The shared view is a diagnostic under the fixed recipe, not a newly selected endpoint.
All20 training receipts report {'2.11.0+cu128': 20}; replay receipts report {'cpu': 20} and {'2.11.0+cu128': 20}, with8 Torch threads. Original CUDA training and CPU replay remain distinct environments; no equal-device trajectories or cross-platform bitwise recalibration are claimed.
Linux whole supplied the original cached-root-refit checks; Windows supplied zero-fit saved-output audits. The original Windows8 import failure (completed0) remains preserved. For F Flip seed91005, the /server_sampling_audit/group_kl difference is −2.168404344971009e−19; the original exact diagnostic maps are retained separately. This difference does not alter the saved threshold/prediction/metric/count audit and is not relabelled as cross-platform bitwise identity.
These are IID Benign and IID F Flip only, with nineteen formal endpoints plus one selected screen checkpoint. No across-scene mean is formed. It is not full100, a Full-versus-ablation pairing, all methods, causal isolation, statistical significance or final-test completion. Other results and negative outcomes are not filtered.
