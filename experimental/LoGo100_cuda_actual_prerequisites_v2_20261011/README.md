# Minimal v2 proposal after independent review

This is a successor source proposal only. The original `LoGo100_cuda_actual_prerequisites_20261011` seal `eca43efd…e61f3`, its actual 712-member inventory `926b478e…6b2a2b`, all measured evidence and the original proposal remain unchanged. No F/SSH/transfer/install/scientific action occurred during v2 preparation.

Only two execution boundaries and the source namespace are changed:

- Sender preserves command, original exception, stdout, stderr and exit status on success, nonzero receiver exit, broken pipe, launch failure and timeout. Incomplete final draining is recorded explicitly with available output and its error; it is not labeled complete. Output files are exclusive and a prior record blocks reuse. Original failure is re-raised and no automatic retry is added. The inventory is referenced from the sealed parent rather than copied or regenerated. Receiver bytes are exactly unchanged.
- Environment creation refuses existing or dangling-symlink venv and managed-Python paths. After `uv python find`, the canonical selected executable must lie below the newly dedicated managed directory. The same Python3.10/package contract, `venv --copies`, fresh root resource gate and no scientific bootstrap remain in force.

Use `TRANSFER_AUTH.template.json` with actual new sender/receiver SHA and a future root-authorized destination; execution remains false. The CLI is unchanged: `python transfer_originals_proposal.py --root-approval ACTUAL --root-approval-sha256 ACTUAL`. Environment script is a root-reviewed proposal, not an authorization or installed environment.

`DIFF.patch` and `INVERSE.json` retain the precise old→new edits and old preimage bytes. `check_source.py` compiles code and runs six stdlib fake-process/metadata fixtures; no real subprocess, network, F volume or scientific artifact is involved. Original cache100 adoption, actual restoration, environment/pickle compatibility and fresh CPU113/SMT/resource gates remain future prerequisites.
