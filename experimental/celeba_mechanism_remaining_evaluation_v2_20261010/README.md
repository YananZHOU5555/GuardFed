# Remaining620 source v2 — Supervisor compatibility only

**SOURCE_ONLY / NOT_APPROVED / NOT_DEPLOYED / CNN0.** This independent release preserves the original29-member source seal `101b0c2798456990885bd1db8305f7f47b6af49dca33c63b99edcc58e13e65f0`. Original scope, ordered620 complement, null future checkpoint SHAs, excluded U100+C80, Full100 references, per-ID original strict70 checks, immutable binding, science/thresholds/native1e-12 and fail-stop behavior remain unchanged. The scientific projection and PLAN are byte-identical to v1. The original v1 README documents those boundaries and the still-pending offserver transport adapter.

The only operational changes are in `FINAL_ENGINEERING_DIFF.patch`:

1. `training_snapshot` accepts RUNNING only with returncode0. Exact target-service EXITED is accepted with returncode0 or3 only when all800 original jobs are complete and no active jobs remain; incomplete/unknown/nonzero RUNNING states still fail. Supervisor stdout, stderr, returncode and actual counts are preserved on success and in refusal error evidence.
2. The supervisor template explicitly sets `startretries=0`, in addition to `autostart=false` and `autorestart=false`.

The source/output namespace and service name are rebound to v2; this is metadata only. No old attempt is resumed. `ROOT_APPROVAL_TEMPLATE.json` still grants no authority; root must fill actual new seal/preflight/approval SHA and a finite deadline.

```bash
PY=/workspace/guardfed_envs/celeba-cu128-20261009/bin/python
PKG=/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010
$PY -B "$PKG/evaluate_remaining.py" inspect
taskset -c112-119 nice -n10 ionice -c3 "$PY" -B -u "$PKG/evaluate_remaining.py" manage --review "$PKG/ROOT_APPROVED.json" --review-sha256 ACTUAL_EXTERNAL_ROOT_APPROVAL_SHA
```

Actual local `V2_CHECK.json` and final `V2_FINAL_CHECK.json` cover rc0 RUNNING, rc3 EXITED800, rc3 incomplete EXITED799 and rc3 UNKNOWN. The final check also requires stderr in successful and rejected evidence. All passed without Torch/CNN/SSH. The earlier v1 reports remain in the immutable v1 package; they are not represented as rerun tests. `ENGINEERING_DIFF.patch` and `V2_CHECK.json` preserve the first successful intermediate source check, before the final addition of stderr to rejected-error context; the final diff/check bind current source.

Every remote completion remains `PENDING_OFFSERVER`, `accepted_offserver=0`. The transport adapter is a separate source-only task. There is no automatic recovery/retry or final-test permission. Local large artifacts must use freshly checked F (`Yanan 2TB`); this package writes only code/config/small reports on E. No checkpoint, array or archive was produced.
