# Review and future entrypoints

Local metadata-only review, after source construction:

    python tmp/celeba_mechanism_valid_C_after1_20261009/check_prepared.py

Read FILES_SHA256.json, execution_candidate/EXECUTION_SOURCE_SHA256.json, INPUT_PINS.json, NATIVE_INPUTS.json, SOURCE_REUSE.json, MINIMAL_SOURCE_DIFF.patch and PACKAGE_RECEIPT.json. Runtime templates default to PREPARED_NOT_APPROVED and are rejected. No actual approval files are created by this task.

Future exact namespace: /workspace/guardfed_checks/celeba_mechanism_valid_C_after1_20261009. Only after independent root review and real Linux resources/dependencies checks, new external ROOT_APPROVED and EXECUTION_DRAFT may authorize:

    <existing-isolatedcu128-python> execution_candidate/install_once.py --draft-sha256 <actual-new-draft-SHA>

This is a placeholder, not an executed command or authorization. Normal service template guardfed_celeba_mechanism_valid_C_after1: autostart/autorestart false, startretries0. Prior C1 service must be EXITED/0worker. Producer-closed new IDs only can enter original backup_completed.py; verify_backup.py independently checks saved arrays/metrics/prediction rules with fixed valid cache offserver. Original model weights must not be repacked in replay backups.
