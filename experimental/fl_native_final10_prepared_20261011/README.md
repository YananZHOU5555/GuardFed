# FL final native10 — source preparation only

Fixed new IDs: `FLGMM_Tg20_L2.0_lr0.001_non-IID_Sp-DFA_seed91001_fullcoverage` through seed91010, in manifest order. Actual prior accepted native86 is pinned by `PREVIOUS_LATEST.json` and `PREVIOUS_ROOT_ADOPTION.json`. Four screen references remain separate. Proposed native96 + reused4 is conditional on successful original verification and root adoption; this preparation accepts zero results. Parent reported terminal92/96; no live observation was made here.

Before execution, root must obtain one actual complete96 snapshot and fill a new `AUTHORIZED_SNAPSHOT.json` from the template (terminal_count must be96); copy the original previous F offserver proof byte-for-byte into `PREVIOUS_OFFSERVER_ACCEPTANCE.json` and verify SHA `fd2fba52cfb7d83f739993c95165481542ce6b2f52b597ca853185323328a1df`; independently review this source and create `ROOT_SOURCE_REVIEW.json` with the actual preparation-seal and authorization SHA. These actual inputs are intentionally absent. Do not execute the template or the derivation script again.

The preflight/close service gate accepts only correct service RUNNING/rc0 with the original1–2 stage workers, or EXITED/rc3 with Supervisor RPC exitstatus0, no stage worker, frozen96 unique saved70-round valid terminal identities, completed_new96/pending0/active0/failed0 and no stage failure marker. The original queue stores completed_new as an integer: the exact completed ID set is derived from the frozen manifest and saved terminal markers. Supervisor RPC uses supervisorctl's own interpreter/config; availability and correct config resolution are unmeasured and fail closed. EXITED is resource/terminal evidence, not scientific acceptance.

Root-reviewed execution commands, once actual hashes are bound:

```powershell
python -B tmp/fl_native_final10_prepared_20261011/run_once.py collect --review tmp/fl_native_final10_prepared_20261011/ROOT_SOURCE_REVIEW.json --review-sha256 ACTUAL_REVIEW_SHA256
python -B tmp/fl_native_final10_prepared_20261011/run_once.py verify --review tmp/fl_native_final10_prepared_20261011/ROOT_SOURCE_REVIEW.json --review-sha256 ACTUAL_REVIEW_SHA256
python -B tmp/fl_native_final10_prepared_20261011/run_once.py finalize --review tmp/fl_native_final10_prepared_20261011/ROOT_SOURCE_REVIEW.json --review-sha256 ACTUAL_REVIEW_SHA256
python -B tmp/fl_native_final10_prepared_20261011/root_adopt.py --handoff-sha256 ACTUAL_HANDOFF_SHA256 --delivery-seal-sha256 ACTUAL_DELIVERY_SHA256 --delivery-files ACTUAL_DELIVERY_MEMBER_COUNT
```

Each phase is called once, serially; any error stops and preserves evidence, with no automatic retry. An occupied CPU110 returns without collecting; inspect its actual OWNER/command receipts before any later authorization. No CPU104/107 use, service changes or partial recipe selection. Actual archive cardinality comes from the receipt/member inventory, never a guessed count. The adopter alone is a later root action that updates shared LATEST.

Original strict/archive tail, per-ID scientific loop, root archive/member/record loops, offserver checker and saved-tensor checker remain unchanged. Namespace/IDs/86→96 boundaries and the finite terminal gate are the differences. CPU110 all-thread conflicts, guide/source/data/package/job/checkpoint identities, one thread/nice10/idleIO/CUDA hidden and fresh F Yanan2TB/Healthy/capacity guards remain. Archive/models/fullstate/raw/log/restore go only to F, never E fallback. No old86 models are repackaged. Server cu128 and local CPU Torch verification runtimes differ; tensor verification loads saved state with zero forward/optimizer/data calls. Prediction arrays are absent and are not regenerated; no train/inference/fit/test or recipe selection is authorized by this source package.

`SOURCE_DIFF.patch` and `SOURCE_INVERSE.json` preserve the complete five-file reversible change; `SOURCE_CHECK.json` contains compile/raw/AST and pure-metadata checks only. The first preparation order assertion failed before any derived files, SSH/F/science; it and its exact source are preserved. The correction compares exact frozen96 membership while retaining the chronological prior86 prefix unchanged.
