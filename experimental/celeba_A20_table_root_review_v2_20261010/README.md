# A20 independent reviewer: display-schema repair only

SOURCE_ONLY_READY_FOR_ROOT_SINGLE_ACTUAL_REVIEW_NO_ADOPTION. This directory contains a derivative of the failed original independent reviewer. No actual review, statistics recomputation, PASS certificate or root adoption has been executed here.

The original failure occurred at the display first-column assertion. The old single-scene table has labels `Full`, `minus_A`, `minus_A minus Full`; the two-scene table deliberately has `IID / Benign / Full` and analogous scene-prefixed labels. v2 strictly checks the complete new label. Its Benign regression compares columns1 onward (n and all three mean ± sample-SD cells), while also requiring the old first-column variant labels. Original24 canonical JSON/order, Benign162 statistics, all source/checkpoint/receipt checks and the original1e-12 numerical tolerance remain unchanged.

SOURCE_DIFF.patch records exactly three edits: bind the immutable failed source/failure JSON in PINS; correct the new label predicate; correct the old/new presentation comparison. Reversing those edits reproduces the original source bytes exactly. Both complete original scientific record/count and mean/sample-SD loops, as well as checked/sealed/main, remain byte-identical. SOURCE_CHECK.json contains syntax and small label-only positive/refusal fixtures, with no actual scientific review run. Original failure evidence is preserved in its original directory.

Role disclosure: the initial independent arithmetic reviewer was authored by the other agent. This v2 display-schema repair was prepared by the A20 table author at root's direction; the scientific review body was not changed or executed by this repair author. Root must review the diff and run the actual v2 entry once before claiming independent PASS.

Root execution from the repository root:

```powershell
python -B tmp/celeba_A20_table_root_review_v2_20261010/review.py --table-dir docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_IID_two_scenes20_20261010 --delivery-seal 3841a7498dad4a3dc6a0bf3b8262510082cd042671850b5f56320a079c012ef8
```

The original one-shot main writes ROOT_ARITHMETIC_REVIEW.json on complete success or REVIEW_FAILURE.json on failure into this v2 directory. It refuses overwrite and must not be automatically retried. It does not change the author table, canonical evidence, STATE or Git, and performs no inference, fitting, SSH or training.
