# Prepared GPU repeatability gate — not started

This package is **PREPARED_PENDING_PARENT_REVIEW**. `FREEZE.json` is intentionally absent; both worker and queue refuse to train without it. It does not freeze or launch the existing 32-candidate scientific screen, final-test evaluation or a formal 100-run method row. The original CPU gate and prepared Group B files remain unchanged.

Four full-data CelebA GPU CANARY jobs cover IID/Benign and non-IID/S-DFA, seed91001, twice each. The repeated conditions swap physical GPUs0/1 to check same-seed behavior on both existing RTX5090 cards. Each task uses three rounds, localLR.001, L3, Tg1, RGB64 CNN, full train162770/root/valid19867, 20 clients and the frozen core/attack/local Adam bridge. Tg1 only exercises GMM → UCL → monitoring; these values must never enter a paper performance table.

`gpu_worker.py` imports the unchanged sealed worker's loader and full-model aggregation bridge. Each child uses one visible GPU and one Torch/OMP/MKL/OpenBLAS thread. Python/NumPy global RNG, Torch CPU/visible-GPU RNG and the states of actual NumPy `default_rng` instances are captured after every round and finally. The tracking wrapper returns the original generator unchanged and draws no numbers. Model, controller and RNG records support comparison; no exact whole-training resume is offered.

`verify_gpu.py` requires four complete three-round outputs, source/config/data identities, all artifact SHA256 hashes and all per-round controller histories. It compares same-seed GPU model tensors, all trajectory/final metrics, evaluation statistics, complete selection diagnostics/control histories and each RNG snapshot exactly. A mismatch is preserved as `REPEAT_MISMATCH`, never silently accepted. CPU→GPU model max-absolute differences, metric differences and history/selection differences are a separate report; equality across devices is not assumed. The prior CPU gate did not capture RNG, so CPU→GPU RNG equivalence is not claimed.

Every worker rehashes all21 inherited source/data files at the end and requires equality with the initial verified identities before PASS. The CPU comparison follows the frozen `LOCAL_ACCEPTANCE.json` and `BACKUP_SHA256.json` hashes through the inventory SHA to the model/result/diagnostics/state/job/provenance/acceptance members, then checks their internal artifact and freeze identities. Existing `GPU_ACCEPTANCE.json` can only be reused byte-for-byte; a differing report is rejected without overwriting the original evidence. `history/v1` preserves the first19-file prepared package before these review changes.

Before the parent freezes this **four-job scope only**:

1. Finish both CPU CANARY jobs and independent off-server acceptance; retain `LOCAL_ACCEPTANCE.json` and `BACKUP_SHA256.json` on the server alongside the CPU gate.
2. Review this source package and current GPU memory/utilization, actual CPU quota, memory/disk and ongoing formal800 progress. Existing healthy services and their eight-worker runner are unchanged.
3. Build `FREEZE.json` with `status: FROZEN`, `scope: four_3round_GPU_CANARY_only`, the exact `local_hashes` from `PREPARED_PACKAGE_SHA.json`, and `prerequisites: {cpu_offserver_verified: true, cpu_receipt_hashes: {LOCAL_ACCEPTANCE.json: SHA, BACKUP_SHA256.json: SHA}}`. `SCOPE.json` remains the prepared plan; the separate receipt records the parent's execution authorization and reviewed bytes.
4. Use a distinct normal supervisor program for `run_gpu_gate.py`; do not launch it from this preparation step. Set `autorestart=false`, `startretries=0`, `stopasgroup=true`, `killasgroup=true`. The command below is a future entrypoint, not executed here.

```text
/workspace/guardfed_envs/celeba-cu128-20261009/bin/python -u /workspace/guardfed_checks/celeba_flgmm_gpu_gate_20261009/run_gpu_gate.py --repo /workspace/GuardFed-celeba-expanded --cpu-root /workspace/guardfed_checks/celeba_flgmm_realimage_gate_20261009
```

The queue uses at most two child workers, one per GPU, preserves all output directories, and refuses automatic retry/resume. Failure stops only its own remaining canary children and writes `QUEUE_FAILURE.json`. It never touches the main formal service or opens a scientific screen. After completion, back up models/results/RNG/checksums and rerun the independent verifier off server before deciding the next screen freeze.

Official source and license are copied unchanged in `sealed_group_b`: HantaoZhu/FLGMM, commit `a064c82a4bc460e168ce09e0654924a39d258bcc`; upstream MIT license is preserved. Keep the label **FLGMM author-code aggregation adaptation**: largest-count GMM component selection, the inherited `bounds_2` behavior and zero-standard-deviation handling remain the disclosed existing adaptation. No new algorithm simplification is introduced.
