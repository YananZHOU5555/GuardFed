# Source-review entrypoints

Local metadata-only review:

    python tmp/celeba_mechanism_valid_C1_gate_20261009/check_prepared.py

Review MINIMAL_SOURCE_DIFF.patch, LINEAGE.json, NATIVE_INPUTS.json, INPUT_PINS.json, FILES_SHA256.json and execution_candidate/EXECUTION_SOURCE_SHA256.json. Science seal has10 members; execution seal11. Templates remain rejected PREPARED_NOT_APPROVED; there is no ROOT_APPROVED, APPROVED, EXECUTION_DRAFT, runtime runs directory or service installation in this preparation.

Only after independent root source/runtime review and new external approval, future deployment namespace is /workspace/guardfed_checks/celeba_mechanism_valid_C1_gate_20261009. Original installer entrypoint:

    <existing-isolatedcu128-python> execution_candidate/install_once.py --draft-sha256 <actual-new-external-draft-SHA>

This is a non-executable placeholder, not execution authorization. Normal service template: guardfed_celeba_mechanism_valid_C1_gate, autostart/autorestart false/startretries0. Exact sole ID is SELECTED_1.txt. Original backup_completed.py and verify_backup.py handle only producer-closed newly accepted ID, saved-array strict comparison and offserver memberSHA; no model weights enter evaluation backups. No implicit retry.
