# One mechanism validation replay — execution attachment v2

Status: **PREPARED_NOT_APPROVED**. No replay/inference, training, new supervisor service or final-test evaluation has been started by this attachment. The original prepared 13 members and execution-v1 7 members remain sealed and unchanged.

## Scope and reviewed flow

Only resource classification and this attachment's source/output binding change. `execute_one_v2.py` reuses the sealed v1 `execute` Python code object with private globals for the v2 directory, source checks and corrected resource snapshot. Scientific imports, bridge, exact one-task approval, complete-terminal validation, saved-prediction validation and failure preservation follow that same reviewed body. The new source check validates prepared13, execution-v1 seven and the independent v2 seal.

The task remains `minus_U_IID_Benign_seed91002`, CPU112–119, eight calculation threads, one process, nice10/idleIO, validation only and native tolerance `1e-12`. Its paired Full is the existing, accepted/offserver phase3 record; no Full inference or Full weight backup is repeated. `../execution_attachments/paired_Full_reference_check.json` is a read-only reference.

## Corrected resource accounting

The classifier reads actual `/proc` Python argv and cwd, parses the main Python entry and exact bounded paths, and never uses hardcoded PIDs. Recognized computing processes are formal GPU workers (one thread), baseline validation replay workers (eight), Hybrid/FLGMM CPU `gate.py` or `canary.py` (eight), gradient `gate.py`/`shared_cache_wrapper.py run` or the trusted-cwd `-c` wrapper/main call (eight), and FLGMM `gpu_worker.py` with the actual repo/job/output interface (one, maximum two). Existing execution-v1/v2 mechanism replay workers are also counted, so an existing CPU112–119 replay prevents a second launch.

Log tee, queue coordinators, inspection/approval/summarize/verification helpers, printed wrapper text, foreign paths and incomplete GPU-worker interfaces are excluded. Duplicate process/task identities, overlapping CPU8 slots, CPU112–119 occupation, incomplete CPU8 affinity, more than two FLGMM GPU canaries and quota excess fail closed. GPU worker affinity may cover the host; it is charged its configured one calculation thread rather than being treated as an exclusive CPU8 slot.

`classification_checks.json` records 18 classification cases and seven rejection checks. `duplicate_single_replay_checks.json` adds two existing-replay rejection cases and excludes read-only inspection. These are classifier checks only, with no torch import or scientific inference.

## Measured observations, not future reservations

- At 2026-10-09 08:20:55 UTC, `real_proc_snapshot.json` identified eight formal workers, one Hybrid CPU gate and one gradient CPU gate: 24 nominal calculation threads. FLGMM's CPU gate had exited. Actual Hybrid used `gate.py`; the `canary.py` and gradient `-c` variants are explicit classification fixtures.
- At 2026-10-09 08:26:34 UTC, `real_proc_snapshot_after_gpu.json` additionally identified two actual FLGMM GPU workers: 26 nominal threads, or 34 including the proposed replay's eight. `live_classification_after_gpu.json` records the exact argv/cwd/affinity and confirms no duplicate task or overlap between CPU8 allocations. CPU quota was 122.87999 cores.
- If the independently authorized phase5 eleven-worker replay starts, those 88 additional threads produce 122 including this proposed replay, only while the observed other reservations remain unchanged. Dispatch must recompute resources; these snapshots grant no future allocation and start no work.

The seven dependency paths were read-only hashed in `dependency_paths_live.sha256`. The actual v4 evidence source is `/workspace/guardfed_checks/server_reactivation_20261009/evidence_v4.py` with the same frozen SHA `3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef`; the v2 prepared approval template corrects only that live location from the old nonexistent path.

## External approval and execution

Use the new `dispatch_receipt.PREPARED.json` as a template; it is intentionally unapproved and has no execution-seal SHA. After reviewing the independent v2 seal and current resource snapshot, the parent must produce a separate approved receipt with status `APPROVED_BOUNDED_MECHANISM_VALID_REPLAY_ONLY`, this exact v2 source SHA and v2 seal SHA. Put its full SHA in `dispatch_receipt.APPROVED.sha256`. Neither file is created by this preparation.

Read-only preparation check, after transferring the prepared tree and both attachments:

```sh
/workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B execute_one_v2.py inspect
```

The included `supervisor.conf` points to the new v2 `service.sh`, with autostart/autorestart disabled and no automatic retry. It is a concrete future execution template, not an installed service. Keep all partial/error outputs on failure. Do not start the remaining seven mechanism replays or any final-test evaluation through this single-task attachment.
