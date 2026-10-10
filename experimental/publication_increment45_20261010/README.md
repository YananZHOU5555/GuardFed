# Git45 compact increment — source prepared

Exact cutoff: native 208 / replay 200; gradient 5 accepted; LoGoFair100 actual startup and first strict fit, **not** fully accepted. The LoGoFair100 summary is source-only. No test or whole-rebuttal completion claim.

`publish_increment45.py` imports the pinned Git44 tool. Its byte-copy/forced-add/-text/index/failure implementation changes only the schema and attribute comment. The planner changes role/count/scope metadata; the Git43 committed-blob verifier is imported verbatim. New Git objects and frozen/staged output stay on the F checkout; the E object alternate stays read-only. Original storage, exact safe.directory, longpaths, secret and size guards remain active.

`prepare_spec.py` reads the actual compact root proofs, verifies all members of the gradient delivery and LoGo summary seals, and prepares an exclusive JSON list. It excludes bulk/log bytes and replaces exact unchanged Git44 blobs with explicit parent-commit recovery references. The native archive, all models and arrays remain on F; this compact Git increment is not an autonomous raw backup. Historical failure/cancellation evidence is retained.

Root workflow, after checking the generated list and mutable entry hashes:

```powershell
python -B tmp/publication_increment45_20261010/prepare_spec.py --output tmp/publication_increment45_root_inputs_20261010/ACTUAL_SPEC.json
# Compute SHA256 of ACTUAL_SPEC.json; substitute it below.
python -B tmp/publication_increment45_20261010/publish_increment45.py plan --input tmp/publication_increment45_root_inputs_20261010/ACTUAL_SPEC.json --sha256 <actualSHA>
python -B tmp/publication_increment45_20261010/publish_increment45.py freeze --input tmp/publication_increment45_root_inputs_20261010/ACTUAL_SPEC.json --sha256 <actualSHA> --output-name inputs001
# Use the actual frozen-input SHA, not the source-spec SHA.
python -B tmp/publication_increment45_20261010/publish_increment45.py stage --input F:/YananResearchStorage/GuardFed/git_publication/increment45/inputs001/FROZEN_INPUTS.json --sha256 <frozenSHA> --output-name stage001
python -B tmp/publication_increment45_20261010/verify_increment45.py --receipt F:/YananResearchStorage/GuardFed/git_publication/increment45/stage001/STAGE_RECEIPT.json --receipt-sha256 <receiptSHA> --commit <actualCommit> --remote
```

Commit/push are performed separately by root. This preparation does not execute plan/freeze/stage, mutate Git, acquire remote state, or rerun scientific validation. The read-only original plan still checks parent commit blobs before any freeze; the actual stage still requires exact clean parent/branch/remote and unchanged input bytes. New observations require a newly reviewed spec; no runtime result is discovered automatically.
