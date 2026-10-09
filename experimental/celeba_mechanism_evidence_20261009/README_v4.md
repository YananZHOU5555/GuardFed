# Mechanism evidence v4: diagnostic preservation correction

v1, v2 and v3 remain unchanged. The accepted first five-model v3 archive and its restore chain remain valid.

Independent review found a fail-closed diagnostic defect in v3: if a live progress/PID check raised, the exception handler repeated the check without catching that second exception. A wrong progress PID or duplicate owner could escape the per-job loop, clear the batch accepted-ID list and omit the promised immutable invalid snapshot. It could not accept an unfinished or invalid scientific result; the actual first five-model archive has no such invalid records.

v4 changes only this recheck: it catches the repeated identity exception, preserves both error messages and proceeds through the existing invalid snapshot path. Other healthy terminal records in the same batch remain available for incremental backup. The original terminal acceptance, checkpoint, partition and restore-chain guards are unchanged.

The independent reviewer reproduced the v3 failure through the actual inspection loop and verified v4 retains the invalid progress snapshot plus the other accepted IDs. Nine unchanged guard functions have identical AST across v2/v3/v4. Review: `../celeba_gradient_realimage_gate_20261009/review_mechanism_v3/review.json` (SHA `a4b2aa348f78d07080343ffe85c862c0cae264d5326bdee78e9b911cc35225a5`). The four original acceptance/backup regression groups and eight live-owner/snapshot boundaries pass with v4. These are software checks, not training results.

v4 SHA256: `3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef`. Only new inspections use v4; old evidence is not regenerated or overwritten.
