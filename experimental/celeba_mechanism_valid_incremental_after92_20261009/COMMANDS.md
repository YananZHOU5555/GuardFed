# Review / future execution entrypoints

Current state PREPARED_NOT_APPROVED. No SSH, deployment or model execution was performed here.

Local independent gate (stdlib only, outputs JSON to stdout):

    python tmp/celeba_mechanism_valid_incremental_after92_20261009/check_prepared.py

Source scope: FILES_SHA256.json (10 members). Runtime source: execution_candidate/EXECUTION_SOURCE_SHA256.json (11 members). MINIMAL_SOURCE_DIFF.patch gives the exact parent changes.

After independent root review only: deploy archive preserving this package namespace under /workspace/guardfed_checks; use ROOT_REVIEW_TEMPLATE.json and APPROVED_TEMPLATE.json to construct separate actual source-bound ROOT_APPROVED.json and EXECUTION_DRAFT.json. Templates themselves are rejected. The original installer entrypoint is:

    isolatedcu128-python execution_candidate/install_once.py --draft-sha256 ACTUAL_EXECUTION_DRAFT_SHA256

Use the actual previously installed isolatedcu128 interpreter from the parent service; this placeholder is not an executable deployment command. Installer owns normal supervisor guardfed_celeba_mechanism_valid_after92, autostart/autorestart false, startretries0. No approval/service is created by this delivery.

Backup/verification reuse execution_candidate/backup_completed.py and verify_backup.py unchanged except namespace/100-inventory/8-count. Only producer-closed new IDs may enter incremental archives; offserver verify_backup.py takes the copied backup directory and checks its saved arrays with the existing fixed valid cache. Root must retain each new receipt/proof chain. Original model weights are not part of these evaluation backups.
