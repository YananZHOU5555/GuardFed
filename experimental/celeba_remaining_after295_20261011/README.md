# After295: exact five A records, source preparation only

This directory reuses the accepted after288 transport and saved-array verification pipeline. No SSH, export, array verification, fit, CNN, training, test, or root adoption has been run here. `EXECUTION_INPUTS.json` and `ROOT_ACTUAL_PINS.json` do not yet exist. Actual native300 root b52484c71ea0e602e13f38f6505bc446d2ca2a80d95a859cf0236b078e087d1d is now locally metadata-verified; execution still waits for root source review.

The sole replay scope is `minus_A_non-IID_Sp-DFA_seed91006` through `91010`, in that order. Parent replay295 remains pinned to index `7c46fcb20c15c377b1df17378f14b6282ee394bdcaacecd8d4f92be344777bef` and root adoption `70689e63467d8866caa3beb06d2be5f911defd0a4d5f7f061ab005d20bd1c1bb`. Five successful new records would produce replay300, not completion of all remaining620 or the entire rebuttal.

## Exact scope changes

- Prior replay288 becomes295; the original parent index and its ordering are retained by reference and SHA. Existing295 are excluded from transport.
- Previous transport108 becomes115, binding the actual after288 receipt `37424c0d0bdb2e4b82b0d3b61570e0debb4405677262e86a94c480b0d32ac657`; only five new saved-output records may be exported, never prior model weights.
- Native parent remains a separate identity chain: preserve all395 previous native objects and the42 ledger entries exactly. The next native root must be adopted, valid-only, contain all five selected A IDs, and account for every new native ID and its total, inspection and archive. The final code requires native exactly300 and the exact five A IDs. Native301 or any additional F is rejected. The initial broader preparation is superseded by SCOPE_TIGHTENING.patch and FINAL_SOURCE_CHECK.json. This is an explicit scope-guard change, not merely a namespace edit.
- The original `join_saved.py` per-record scientific loop is source-text and AST exact. The original transport source remains `f71e6e4152625a5a0582a61ff9b3e55e4ccee65dfd70a4e657851c0247f9c9d7`. Original saved-array formulas, count/prediction checks, same-checkpoint joins and1e-12 native tolerance are unchanged. No fresh fitting or inference is added.
- CollectorCPU111 remains a requested lane, not a claim of availability. The original live preflight must verify thread ownership, actual quota, selected-worker absence, guide/source seals and the previous receipt immediately before the one-shot export. Bulk archive/extract writes require the existing F-volume `Yanan 2TB` health/capacity guard. There is no internal-disk fallback and no automatic retry.

## Prepared entrypoints

`bind_native.py` verifies the next actual native proof and a reviewed source seal, then writes fresh execution inputs. `execute_once.py` runs the original bounded preflight, export and original offline saved-array checker once. `join_saved.py` joins only the five selected outputs to native model/config/data identities, keeping the original scientific loop. `adopt_by_root.py` is source-only for the parent: it requires an externally SHA-pinned `ROOT_ACTUAL_PINS.json`, actual transport/saved-checker evidence, exact47 archive members,45 metrics/120 counts/15 rules, and old295 preservation. No future proof values have been fabricated.

The old finalizer contains actual after288 result hashes. It is deliberately not copied as a runnable new finalizer. After the single actual execution succeeds, only fresh actual handoff/delivery hashes and root-pin input are to be filled; parent review and root adoption remain separate.

## Current local checks

`prepare_source.py` was executed once locally to derive nine original entrypoints and compile them. `check_scope.py` ran metadata fixtures only: exact5/native300; native301 with F excluded; rejection of missing A, F in replay, old object changes, old ledger changes, duplicate native IDs, and replay count expansion. `prepare_adopter_source.py` compiled the original adopter's bounded metadata adaptation without importing or executing it. Current evidence is `FINAL_SOURCE_CHECK.json`, `EXACT300_SELECTION_FIXTURES.json`, and `FINAL_SOURCE_DIFF.patch`. Earlier SOURCE_CHECK/SELECTION_FIXTURES/SOURCE_DIFF and preparation scripts record the historical, unexecuted broader source preparation; do not rerun those preparation scripts.

Once the parent supplies the actual native proof and source review, the existing CLI sequence is:

```text
python -B tmp/celeba_remaining_after295_20261011/bind_native.py --native-root <actual-root> --native-root-sha256 <actual-sha> --prepared-seal-sha256 <reviewed-source-seal>
python -B tmp/celeba_remaining_after295_20261011/execute_once.py
python -B tmp/celeba_remaining_after295_20261011/join_saved.py
```

Root alone may subsequently run the prepared adopter with `--actual-pins-sha256 <reviewed-actual-pins-sha>`. These commands are instructions for the authorized next stage, not evidence they have run.
