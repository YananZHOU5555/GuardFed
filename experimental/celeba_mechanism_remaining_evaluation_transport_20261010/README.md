# Remaining620 validation transport — source preparation only

This package exports one explicit, ordered ID delta from the frozen remaining620 V2 CPU evaluation queue. It does not evaluate images, dispatch tasks, select a recipe, adopt evidence, or change the queue PLAN. The original180 and Full100 are excluded. Future checkpoint SHA values in PLAN remain null; only the queue's actual immutable per-ID binding can supply a checkpoint identity.

`transport.py export` requires the actual external root approval SHA, the frozen V2 source seal, and this package's external seal. Each selected ID must have its original strict70 proof, immutable binding, delegated approval, four complete saved-output files, matching checkpoint/source/config/data/root/valid identities, unchanged original native artifacts, and a terminated evaluation child. Original native tolerance is exactly `1e-12`. Export runs on Linux CPU110 with nice10/idle I/O; it does not take the CPU112–119 CNN lock or import Torch. A separate transport lock prevents concurrent exports.

Each new ID contributes four saved-output files and five task files. The first export also carries the sealed queue/transport sources, actual root approval/preflight, and pinned original scientific/verification scripts. Later exports contain only the new ID delta and its snapshot inventory. Models, prior180 outputs and historical native archives are never repacked. Previous transport receipt SHA and the exact ID chain are mandatory after the first export. Any failed/orphan export blocks another export; there is no resume, overwrite, automatic retry or automatic loop.

The archive writer and its post-write input hash checks are extracted unchanged from the pinned original `backup_completed.py`. The original `evidence_v4.verify_archive` verifies archive/member hashes. Offserver verification reuses the original `verify_saved_increment.verify`: only its literal inventory SHA becomes the actual batch snapshot SHA, and its descriptive scope label changes. Its saved prediction, three-view metric, confusion count, CPU/runtime, unchanged-weight and native comparison checks are unchanged. It uses the original validation-label cache SHA `39e5fe16ea5f06c0a7594fd6b1c1cbc27cf1a71a261c1b4f0db7228cce380f64`; it neither refits thresholds nor performs CNN inference. Original strict server acceptance already recomputes the root-only fits.

All receipts, including the local verification proof, retain `accepted_offserver=0`. Root must independently adopt the actual archive/proof and join each checkpoint to the original accepted native archive chain. Remote closure or successful transport alone does not complete the mechanism900 cohort. Native/shared primary endpoint selection remains pending; this package makes no new performance claim.

## Root execution commands after independent review

Deploy all files listed by `FILES_SHA256.json`, plus the seal itself, to `/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010`. The V2 queue source remains `/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010`; its runtime is `attempt1`. Do not run these commands as part of source review.

```sh
taskset -c 110 nice -n 10 ionice -c 3 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B /workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010/transport.py export --source /workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010 --source-seal a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03 --transport-seal TRANSPORT_SEAL_SHA --runtime /workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010/attempt1 --review ACTUAL_ROOT_APPROVED_PATH --review-sha256 ACTUAL_ROOT_APPROVED_SHA --tag UNIQUE_TAG --ids ID1 ID2
```

The isolated interpreter path above is read from the frozen V2 supervisor template. After the first export, append `--previous ACTUAL_REMOTE_PREVIOUS_RECEIPT --previous-sha256 ACTUAL_PREVIOUS_RECEIPT_SHA`. IDs must follow the original manifest order. Root downloads only the new archive and its receipt; downloading/SSH is outside this CLI.

```powershell
python -B tmp/celeba_mechanism_remaining_evaluation_transport_20261010/transport.py verify --source tmp/celeba_mechanism_remaining_evaluation_v2_20261010 --source-seal a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03 --transport-seal TRANSPORT_SEAL_SHA --archive F:/YananResearchStorage/GuardFed/remaining620/UNIQUE_TAG/incremental_valid_three_views.tar.gz --receipt ACTUAL_LOCAL_RECEIPT --receipt-sha256 ACTUAL_RECEIPT_SHA --cache tmp/celeba_final_valid_replay_20261009/verification_inputs/original_valid_cache.npz --out F:/YananResearchStorage/GuardFed/remaining620/UNIQUE_TAG/verification
```

For later local verification, also pass the actual locally downloaded prior receipt and its SHA. The CLI checks the fresh F drive label `Yanan 2TB` and free space before archive/extraction bulk access. Small source/config/proof files may remain on E. Local extraction uses checked regular members and exclusive writes, compatible with Python3.10; no `tarfile` filter installation is needed.

## Actual checks and remaining limits

`SELF_CHECK.json` records actual pure-metadata success and 13 refusals, unchanged original writer extraction, saved-checker AST equality except the two metadata substitutions, and a fresh read-only F volume check. It imported neither NumPy nor Torch and wrote no archive/arrays/model. `metadata_only_fixtures` contains explicitly TEST_ONLY receipt bytes, not scientific evidence. The check is one-shot and preserves its files; do not overwrite it to rerun.

No remote export, SCP, local archive/member/array verification, CNN or adoption has run for this package. Actual runtime/receipt schema and dependencies still require root's independent code review and first real bounded export. Source preparation is not execution approval. Existing scientific, selection, mixed-device and historical test-exposure limitations remain unchanged.
