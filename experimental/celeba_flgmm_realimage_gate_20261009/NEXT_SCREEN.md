# FLGMM: real-image gate and next screen boundary

This directory owns only two **CPU real-image CANARY** runs: IID/Benign and non-IID/S-DFA, shared seed91001, full train162770 and official valid19867, 20 clients, unchanged RGB64 CNN/local Adam/root/attack bridge. They use local LR .001, L3 and Tg1 so three rounds exercise `per_round_gmm -> fit_control_limit -> monitor`. Tg1 is not one of the prepared scientific candidates. No final-test evaluation or 32-job screen is launched here.

The original `sealed_group_b/protocol.json` remains `PREPARED_NOT_FROZEN`. `SCOPE.json` and `FREEZE.json` freeze only this separately labelled two-job gate, its source and job bytes. Original worker, adapter, upstream source, MIT license and protocol bytes were copied unchanged. `source_preflight.json` is the measured server comparison against all21 inherited source/data identities. `canary.py` reuses the original worker's core-loader and full-model/delta aggregation bridge without changing the algorithm. The inherited checker has hardcoded70-round/screen requirements, so the separate checker here keeps the analogous identity, data-count, state-history, model-hash and final-metric checks while requiring CANARY labels and exactly3 rounds. No screen record is forged to pass the old checker.

The canary uses one training process with torch/OMP/MKL/OpenBLAS8 threads, no CUDA context and CPU affinity0..7 assigned by the parent. OS auxiliary thread pools may contain more than8 sleeping threads; the affinity hard-caps available logical CPUs at8. Both affinity receipts are retained. Formal800 GPU workers are read only; their real progress and resource snapshots before/after the gate are recorded separately. The initial telemetry probe incorrectly tried an old cgroup-v1 path, then used actual cgroup-v2 counters; this was a read-only diagnostic error before training, not a failed experimental result.

Run independent acceptance after off-server transfer:

```powershell
python tmp/celeba_flgmm_realimage_gate_20261009/verify_canary.py
```

It verifies all frozen input files, all declared output hashes, both real-data contracts, exact3-round stage progression, state history for20 clients, same-final-checkpoint metrics, finite model parameters and CPU-only provenance. It writes `LOCAL_ACCEPTANCE.json`. `--source-only` intentionally checks no experimental results. There is no automatic retry; any `failure.json` prevents acceptance and is retained.

## Formal32 validation-screen proposal — not executed

The existing scientific grid is **Tg{10,20} × L{2,3} × localLR{.0005,.001}**, eight candidates × IID/non-IID × Benign/S-DFA, seed91001, 70 useful rounds:32 jobs. Retain the predeclared score and four-condition mean/tie rule already in the sealed protocol, all negative results and full raw ACC/AEOD/ASPD. This is n=1 validation tuning, not multi-seed confirmation or final-test evidence.

After the parent reviews the completed gate and the runtime device boundary, create a **new** owned screen source directory from `sealed_group_b`, preserving the prepared original. In that new copy only, change protocol status to `FROZEN`, record the gate receipt and all source/data/environment identities, then run the unchanged preparer. Example, to be executed only by the parent for its approved screen scope:

```python
from pathlib import Path
import json, shutil, subprocess, sys
source = Path('/workspace/guardfed_checks/celeba_flgmm_realimage_gate_20261009/sealed_group_b')
target = Path('/workspace/guardfed_checks/celeba_flgmm_screen_reviewed_20261009')
shutil.copytree(source, target)  # fails if target exists; never overwrite a freeze
p = target / 'protocol.json'
protocol = json.loads(p.read_text())
assert protocol['status'] == 'PREPARED_NOT_FROZEN'
protocol['status'] = 'FROZEN'
p.write_text(json.dumps(protocol, indent=2) + '\n')
subprocess.run([sys.executable, str(target/'prepare_jobs.py'), '--out', str(target/'jobs')], check=True)
```

This creates32 job files and their protocol/adapter hashes; it starts no training. With the explicit image repo and one reviewed job, the existing entrypoint is:

```text
/workspace/guardfed_envs/celeba-cu128-20261009/bin/python worker.py --repo /workspace/GuardFed-celeba-expanded --job jobs/JOB.json --out NEW_RUN_DIR
```

Before dispatching a GPU screen, retain a bounded CUDA device/determinism regression gate for this adapter; the current CPU image gate does not prove GPU numerical equivalence. Do not share or modify the ongoing formal800 queue. Long-run result reuse must call the copied original `accept_result.checked_result(job_path, output)`, which requires frozen provenance, all70 rounds, valid19867, final model/diagnostic/state hashes and control-history identity. The original worker explicitly offers no exact whole-training resume; a controller JSON by itself does not include optimizer/model/RNG state.

Label the row **FLGMM author-code aggregation adaptation**. The pinned author's largest-component choice differs from the paper introduction's smaller-mean description; the inherited `bounds_2` behavior and explicit zero-standard-deviation extension remain disclosed. Neither these CANARY values nor future selected n=1 tuning values should be inserted as a ten-seed paper row.
