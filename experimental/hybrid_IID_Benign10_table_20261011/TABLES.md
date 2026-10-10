# CosineFairnessHybrid: IID Benign, three validation views

Ten fixed round-70 checkpoints, 19,867 validation images; raw/native/shared use the same checkpoint and seed panel. Mean ± sample SD (ddof=1). ACC (%) ↑; AEOD and ASPD ↓. AEOD is the absolute TPR gap, not full equalized odds. Formal primary endpoint remains pending.

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

Seed91001 is the originally selected screen checkpoint; seeds91002–91010 are formal-coverage checkpoints. The all10 panel retains the selected seed; n9 excludes only91001; n6 fixes91005–91010. Validation was exposed during development, including development after the initial official-test results were viewed. These descriptive subsets are not untouched confirmation sets; no new final-test evaluation was performed.
Raw and native retain margin > 0 (zero-margin ties predict0); shared_calibration uses the frozen common clean-training-root-only thresholds with margin ≥ group threshold. This table only reads saved accepted receipts: it performs no calibration fit, forward pass or training. Raw/native equality is retained as measured, not treated as independent confirmation. The shared view is a diagnostic under the fixed recipe, not a newly selected endpoint.
All10 training receipts report {'2.11.0+cu128': 10}; replay receipts report {'cpu': 10} and {'2.11.0+cu128': 10}, with8 Torch threads. Original CUDA training and CPU replay remain distinct environments; no equal-device trajectories or cross-platform bitwise recalibration are claimed.
Linux whole supplied the original cached-root-refit checks; Windows supplied zero-fit saved-output audits. The original Windows8 import failure (completed0) remains preserved. The exact seed91005 group-KL diagnostic difference is retained separately; it does not alter the saved threshold/prediction/metric/count audit.
This is one IID Benign scene only, with nine formal endpoints plus one selected screen checkpoint. It is not full100, a Full-versus-ablation pairing, all methods, causal isolation, statistical significance or final-test completion. Other results and negative outcomes are not filtered.
