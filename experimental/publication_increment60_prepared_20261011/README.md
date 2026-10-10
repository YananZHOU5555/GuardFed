# Increment60 — source-only publication preparation

Publisher: `tmp/publish_guardfed_increment60_20261011.py`. Fixed parent: `27ebed940ce822f87976e0d37b90576e9a43f36a`.

No Git, F copy, SSH, runtime metadata guard, scientific checker or publisher execution occurred during preparation. Current entry updates, CPU11 work and root's final selection/review are external dependencies; no future outcome or selection SHA has been fabricated.

The three increment59 Git blocks are byte-exact: (1) Git helper plus F-label/health/capacity and clean checkout/branch/origin/remote-parent checks; (2) byte copies, `-text`, explicit staging and staged-name containment; (3) committed-blob SHA/size, clean checkout, push and remote-head verification. Existing per-file <2 MB, aggregate <100 MB and suffix limits remain. `SOURCE_CHECK.json` records compile-only checks, identical block hashes and exact inverse reconstruction of the complete increment59 source. No old A100 tables, arrays, native receipt archives or scientific verification loops are rerun.

## Root supplies two real input JSONs after closing the current entries

`--semantic-review` must resolve to a JSON under repository `tmp/`. It requires:

- `status: PASS_LIMITED_CURRENT_ENTRY_SEMANTIC_REVIEW`, `root_accepted: true`, `STATE_sha256`.
- `current_accepted`: `native_new=320`, `three_view_models=320`, `F_paired_models=20`, `F_complete_scenes=2`, `FL_native_new=67`, `FL_three_view_total=61`, `FL_complete_table_records=60`, `gradient_accepted=46`, `Hybrid_native_new=12`, and `by_variant={minus_U:100, minus_C:100, minus_A:100, minus_F:20}`.
- `latest_observation_only`: identical to STATE's current observation object, with `new_acceptance=0` and `counts_are_observation_only=true`. Observed terminal totals are not substituted for accepted counts.
- `entries`: exactly the five existing Markdown entry paths (RUNNING, REBUTTAL_COMPLETION, overview, MONITOR_HANDOFF, mechanism EXECUTION), each with actual `sha256` and `history_exact=true`. STATE is separate, not a sixth Markdown entry.
- `source_pins`: each object contains repository-relative `path` and actual `sha256`; every such file must belong to the compact selection.

`--compact-selection` must be `tmp/publication_increment60_prepared_20261011/COMPACT_SELECTION.json`. Root creates it with:

- `status: ROOT_CLOSED_INCREMENT60_COMPACT_SELECTION`, `root_accepted: true`, fixed `parent_commit`, actual `STATE_sha256`, and actual `semantic_review_sha256`.
- `files`: the complete approved set of normalized repository-relative paths, each with exactly `sha256` and integer `bytes`. No directory globbing or implied extras. Include the publisher source, STATE, five Markdown entries, the fixed F20 editorial proof, its six pinned source/documents, the parent59 publication receipt, and all semantic-review source pins. Other compact files are present only if root explicitly selects and reviews them.
- The selection does not include its own hash. Its actual SHA is a required CLI argument. The two CLI JSON inputs are the only explicit additions outside `files`; root may include the semantic review in `files` as well. Duplicate publication destinations are rejected.

Each selected SHA/size is checked before F/Git work. The unchanged copy/stage block is followed by a strict comparison of copied bytes to the selection before commit, plus a latest-STATE SHA recheck. Drift stops with reversible staged work retained, not an incorrect commit. Original one-shot behavior applies: failure is preserved and needs diagnosis, not automatic rerun. The new increment60 publication proof must not already exist.

The compact editorial contract is pinned to actual root `60bc906155a6dbb9e4c73f42d788b66579b731e8fe843b6f1c712202892409ab`: 24 comments, A100/F20, two F scenes/20 pairs, author-review-only, manuscript not applied, no final test or whole-rebuttal completion. Parent59 receipt `e2964aa9d269bb84404ef7d0b91498191a336b22cf9b5eaad01112b2685baddd` must identify the fixed parent. Scientific accepted counts remain unchanged; operational source/progress evidence is not new scientific adoption.

## Later root execution, not executed by this preparation

```text
python -B tmp/publish_guardfed_increment60_20261011.py --semantic-review <actual-root-review-relative-path> --semantic-review-sha256 <actual-sha256> --compact-selection tmp/publication_increment60_prepared_20261011/COMPACT_SELECTION.json --compact-selection-sha256 <actual-sha256>
```

There is deliberately no COMPACT_SELECTION.json or actual semantic review in this prepared packet. Root must review this source and bind real, frozen inputs before the authorized single publication command.
