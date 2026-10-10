# Increment42 — actual closed C10/C80 inputs prepared, not staged

Parent commit is `b57ae3c07e1013c17820c2eb038d701a356b27c6`. No staging, commit, push, SSH, inference or statistics have run here.

The closed C10 increment is genuine: after70 adds ten non-IID FedSA checkpoints to the existing170 three-view records, totaling180. The original result archive has103 members, including102 content members. `CLOSED_C10_MAPPING_DRAFT.json` preserves the initial109-file draft. `ACTUAL_CLOSED_INPUTS.json` now binds the actual root-adopted C80 table, independent review, C10 closure and frozen entries. The table has160 records/80 pairs/eight complete scenes; complete English prose remains C60.

`publish_increment42.py` pins and reuses the sealed increment41 publisher. Only scope, parent, counts, receipt labels and the record-only `scope_guard.py` differ. Copying, force-add, `-text`, index blob verification and failure preservation still execute the original increment40 AST block unchanged. `verify_increment42.py` reuses the original committed-blob loop with explicit parent/count/receipt substitutions. The root must review and invoke staging; these scripts never commit or push.

New mappings cover the actual after70 scientific and execution source, external approval, startup, final observation, strict/backup/offserver proof, and root source/transport review. The transport report was completed after the initial deployment invocation; its source-bound supplementary verification and ordering-error journal are retained without changing their time or claiming a pre-deployment transport PASS.

Two upload-only source/deployment archives are omitted with their SHA and recovery limits recorded. Exact source and approval bytes are retained separately and in the accepted archive. `verified_extract` is a derived unpacking of that archive, not a second artifact. Old native180, FL18, Hybrid23 models/archives and prior tables/prose stay in parent41. Later C90/native190 records are outside this increment.

Actual canonical C80 ROOT is `71105f39c6345efb3706fe538a51686f80ad9e8a89f5601a04b349ebcc76e487`;31-member seal is `a456c192c01d37d4ad577e2284380d3dec1290bb09e93d6dfc5922d3cd5749cf`. The accepted snapshot is native180/views180, FL18/Hybrid23. Hybrid observed24 is not acceptance. `ROOT_ACTUAL42_FREEZE_INPUTS.json` and `frozen_inputs/` preserve the entry bytes with explicit canonical publication destinations; restore the owned snapshot paths using these mappings and SHA if needed. The main observation is03:05:38 UTC, not a claim of present server state. No C90/native190 artifact is included.

Root commands, after independent review:

```text
python tmp/publication42_preparation_20261010/publish_increment42.py --closed-inputs tmp/publication42_preparation_20261010/ACTUAL_CLOSED_INPUTS.json --closed-inputs-sha256 ea3226e795c71b0b23960aaefc7c5a70282f236d5f6a9404a6189ef17bfe4702 --output tmp/publication42_preparation_20261010/attempt --execute-stage
python tmp/publication42_preparation_20261010/verify_increment42.py --receipt <actual42receipt> --receipt-sha256 <actualSHA> --output tmp/publication42_preparation_20261010/COMMIT_VERIFICATION.json
```

After the root publishes, repeat the verifier with `--remote` and a distinct output path. Nothing here automatically publishes or updates canonical state.

Initial local mapping preparation rejected an extra upload-only deployment tar before producing a draft; `MAPPING_DRAFT_FAILURE.json` preserves the error. No scientific artifact or Git state was affected.
