# Increment36 source-only publication preparation

This package adapts the successfully used increment35 publisher and 51-line commit verifier. It has not staged, committed, pushed, used SSH, or collected scientific results. The parent is `fbe6027794ce045bdf79254b995c8d2b1de2fb56` on `codex/revision-evidence-baselines-20260928`.

Root must supply a new external closed-input JSON only after actual C6 strict/offserver/root adoption and a consistent state snapshot exist. Copy the template outside this sealed source directory, change its status to `ROOT_CLOSED_INCREMENT36_INPUTS`, fill all seven `closure_pins` and all required `extra_pins` with actual paths/SHA256, and set counts to native156 / three_view156 / FL_new11 / Hybrid19 / baseline_valid900. Root may add other actual closed source-review, transport, terminal, adoption and status pins. A running service or deployment proof cannot substitute for C6 adoption. All template hashes and counts intentionally remain null.

The new C6 scope is exactly `minus_C_non-IID_Benign_seed91001..91006`, extending the accepted150 checkpoint identities. The already adopted FL delta is two IID F Flip records, extending9 to11. Hybrid19 is unchanged: its ROOT proof must match both its fixed SHA and the increment35 published mapping. Every other file under its old execution namespace, including archives, seals and LATEST, is refused. The already published C50 table must remain `C50_table=null`. The separately adopted complete C50 author-review text is included through its actual 13-member seal, seal file and ROOT review (15 required extra pins); this is author-review text, not an applied manuscript or submission.

The source checks use an in-memory exact6 closure fixture and 26 meaningful refusals. They verify the unchanged copy/index/failure block, force-add / `-text` / renormalization / blob SHA checks, duplicate-new-archive refusal, all31 fixed source pins and unchanged Hybrid publication identity. They do not claim actual C6 closure. The historical earlier25-refusal report is retained.

Root commands, after independently reviewing this source seal and filling an external actual input file:

```powershell
python -B tmp/publication_increment36_prepared_20261010/check_prepared.py
python -B tmp/publication_increment36_prepared_20261010/publish_increment36.py --closed-inputs <actual_closed_inputs.json> --closed-inputs-sha256 <actual_sha256> --output tmp/publication_increment36_prepared_20261010/actual_stage_once --execute-stage
```

The publisher gates all closure/source/count/parent/clean-tree checks before its first write. It copies an exact allowlist, refuses tracked old archive paths, enforces unique new archive SHA and total bytes below100MB, stages with the original byte-preserving index workflow, and verifies index blobs. It neither commits nor pushes. Any failure is preserved without automatic retry. Root performs its reviewed commit/push separately, then runs the retained commit verifier:

```powershell
python -B tmp/publication_increment36_prepared_20261010/verify_increment36.py --receipt tmp/publication_increment36_prepared_20261010/actual_stage_once/publication_closed_increment36_20261010.json --receipt-sha256 <actual_receipt_sha256> --output tmp/publication_increment36_prepared_20261010/actual_commit_verification.json --remote
```

The verifier reads the actual publication worktree HEAD and checks its parent, branch, clean state and committed bytes. Source generation/checking/sealing is complete; actual closure, Git index behavior in the live publication worktree, commit and remote verification remain root execution boundaries. No new metrics, recipe choice, final test or scientific-completion claim is introduced.
