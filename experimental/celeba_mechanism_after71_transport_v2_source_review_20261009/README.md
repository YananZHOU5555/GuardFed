# after71 transport-v2 — PREPARED ONLY

The original WinError206 failed before SSH child creation. Its helper, source seal and ROOT_BACKUP_ATTEMPT remain unchanged. ROOT independently confirmed no remote BACKUP_LATEST or backup directory and exact11 terminal completion; actual recovery proof SHA is7c9cf002a689d8d418cfa5694e1d5751e4ffdec6d158050380cce297f7cadefe. The new helpers pin that proof, old helper and old attempt.

The sole transport change replaces the long `python -c <script>` SSH argument with fixed `python -B -` and passes the entire unchanged scientific comparison script through `input=code.encode()`. Source/identity/source-seal/complete-ID checks and the complete batch_complete equality remain. An additional actual batch_complete file SHA and empty remote backups-directory gate apply. No scientific acceptance, tolerance, inventory or output namespace changes.

ROOT invocation only, one new transport attempt:

```powershell
$closure = python -B tmp/backup_mechanism_after71_transport_v2_root_20261009.py | ConvertFrom-Json
```

This writes new ROOT_BACKUP_TRANSPORT_ATTEMPT.json and ROOT_BACKUP_TRANSPORT_COMMAND_RESULT.json. It never removes or replaces the old attempt. It uses ROOT_BACKUP_COMMAND_STDOUT.json only if still absent. The original backup_completed.py/verify_backup.py are called unchanged; original110 exact members,99 metrics,264 counts,33 rules and prior71+exact11 guards remain.

After independently reviewing the actual returned hashes/evidence, ROOT may explicitly adopt:

```powershell
python -B tmp/adopt_mechanism_after71_transport_v2_root_20261009.py --delta $closure.delta --receipt-sha256 $closure.receipt_sha256 --proof-sha256 $closure.offserver_proof_sha256 --terminal-progress-sha256 $closure.terminal_progress_sha256
```

The adoption binds the transport attempt, original failure and actual ROOT recovery proof, then runs all original archive/receipt/array-proof/source/checkpoint/prior71 guards. No automatic retry or registration follows preparation. Any failure preserves evidence and requires a new ROOT decision. This delivery performed no SSH, backup, helper main, CNN, inference, training, test, Git or canonical update. Only local source/argv checks and read-only recovery-proof validation were performed. No original sealed source was edited.
