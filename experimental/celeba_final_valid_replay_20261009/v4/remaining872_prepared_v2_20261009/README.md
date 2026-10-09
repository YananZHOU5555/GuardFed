# Existing nine-method900: remaining872 valid replay preparation v2

Status: **PREPARED_NOT_DISPATCHED**. This directory prepares execution; it has not started inference, installed a service, or produced900 results. Existing accepted28 are bound by collector SHA `11ae999294efb3ed663a8cc77d85bbd930a5fe64f3545380015255d86063852f`. The unchanged900 inventory minus those canonical scientific IDs gives exactly872, in80 chunks of at most11 (79×11+3). Equal checkpoint byte hashes across distinct scientific cells are legal.

`bounded_remaining.py` calls the sealed `replay_v4.py run` and `accept` interfaces. The scientific v2/v3/v4, core, evaluator, CNN, original checked_result, original jobs, model weights, and storage map remain unchanged. Each full chunk must pass original source/data/root/split/model/result/rawjob identity, full valid19867 and original clean-root16277, root-only threshold reconstruction, saved prediction arrays, same-checkpoint metrics, and native absolute error≤1e-12. Source/restore/provenance hashes and the exact872 complement are in `manifest.json`. Historical output identities remain historical, with runtime paths resolved only through the trusted storage map.

The parent selected11 workers from real1/2/4/8/11 representative batches. This is the highest measured throughput of different useful checkpoints, not a same-model controlled speedup or a global optimum. Every scientific worker retains8 Torch threads,1 interop thread,0 loader workers, all threads within its assigned8 CPUs, nice10 and idle I/O. Slots are indices16..103 of the original sorted allowed CPU list. The outer stdlib coordinator shares CPU16 and remains nice0. Original sealed run applies nice10 to its manager exactly once; actual CNN workers inherit10 and remain idle-I/O. No priority elevation or capability change is attempted. The initial live quota must support the conservative122-core total budget; the latest measured quota is122.87999. This does not reserve resources against uncoordinated jobs started later.

Local preparation passed22 refusals,99 direct metrics and264 confusion counts on the11 already accepted phase5 arrays, and a50-member real-evidence archiver roundtrip. These are regressions, not11 additional results. The first unsealed Windows-only structural check failed because `Path` rendered Linux remote constants with backslashes; its failure record is retained. `PurePosixPath` fixed that representation in the preserved prior preparation. `selfcheck.json` is this version’s current check. The superseded19-item preparation remains byte-for-byte preserved: it inherited outer nice10 into original run, whose own unconditional nice adjustment would have yielded19. This v2 removes the outer adjustment and requires coordinator nice0. Linux supervisor orchestration and the new chunk verifier have not yet been exercised end to end on new872 data.

## Parent review and deployment

Review `FILES_SHA256`, `manifest.json`, both implementation files, and the normal supervisor configuration. Copy exact bytes into the existing remote checks tree. Also copy the locally generated prior offserver proofs, the cumulative28 receipt, and the six prior archives to their manifest-relative remote paths if absent. Existing bytes must match; never overwrite a differing file. All original artifacts and adapter sources remain in their already verified locations. Git excludes source archives, so restore missing archives from the established published/offserver chain before reproducing the gates.

Create only `/workspace/guardfed_checks/celeba_final_valid_replay_20261009/v4/remaining872_execution_20261009/` before installing the config; supervisor needs that empty parent for its log. Do not precreate the manifest's `remaining872_attempt1` output. First run the isolated Python with `bounded_remaining.py inspect --manifest .../manifest.json --manifest-sha256 <reviewed SHA>`. Inspect performs source/data SHA checks only, no image inference. After parent authorization, install only `guardfed_celeba_valid_remaining872_20261009.conf`, use supervisor's targeted reread/update, then start that program. `autostart=false`, `autorestart=false`, and `startretries=0`. This preparation does not itself authorize dispatch.

## Incremental evidence and recovery

Every completed≤11-ID chunk produces `strict_acceptance.json`, `chunk_evidence.tar.gz`, `remote_archive_inventory.json`, and `accepted_ledger.json`. The archive contains only newly produced predictions, receipts, worker proofs, original per-chunk inventory/config records, logs, execution manifest, and sourcefreeze. No old checkpoint is repackaged. Original model/result/rawjob archive/member restoration identities remain in those inventory records and the original900 restore chain. Archive whole SHA, each member SHA/size, unchanged source files, and ledger previous-SHA links are explicit. Remote strict acceptance and an on-server archive are labeled pending offserver acceptance.

Copy each closed chunk's archive, member manifest, and ledger offserver. Independently obtain its remote archive SHA, then run locally:

```
python audit_remaining.py verify-chunk --stage <downloaded chunk directory> --archive-sha256 <independent remote SHA> --manifest manifest.json --manifest-sha256 <reviewed SHA> --chunk-index <0..79>
```

The verifier checks the exact frozen chunk, full scientific SHA set, original artifact/map identities, actual root/valid IDs, resource/thread scope, metadata prefix access, unchanged weights, raw tie0 versus calibrated>= predictions,99-or-fewer direct metrics, and24 counts per checkpoint. It then invokes the unchanged mixed-version collector on the candidate before naming the final `offserver_verification.json`. Failed candidates keep an explicit nonfinal filename.

Any identity, numerical, resource or restore error stops the outer execution without automatic retry. Original failed/partial outputs, worker logs, the original strict accepted subset, and a separately named failure archive remain. Never restart this output or assume a zero service exit means completion. Recovery requires all own workers gone, independent review of completed chunk proofs and any complete accepted subset in the failed chunk, a SHA-bound per-ID recovery registration, and a newly frozen complement manifest/attempt. This package does **not** implement an automatic partial-chunk importer: the original mixed-version collector deliberately rejects incomplete selected cohorts. A partial-chunk registration adapter must be independently reviewed before those successes can be reused; they must not be silently rerun or discarded. Complete closed chunks already use the unchanged collector.

## Complete collection and descriptive statistics

Extend `v4/execution_20261009/collection_inputs_28.json` with each actual new `['v4', '<proof relative to replay base>', '<reviewed proof SHA>']`. Each proof remains beside its corresponding archive and member manifest. Run the unchanged `collect_valid_replay.py`; it resolves original-job/inventory aliases to one canonical ID, rejects duplicates, and calls complete only when all900 frozen IDs are present with actual native and three-view acceptances. The failed v3 phase4 attempt remains preserved with0 accepted.

Only after that exact900 collection passes:

```
python audit_remaining.py summarize900 --collection-inputs <new source-version proof specification> --output <new output directory>
```

Outputs are per-scenario CSV/JSON for9 methods×2 distributions×5 attacks×raw/native/shared and10/9/6 prespecified seed cohorts, plus per-seed paired differences from Full and cross-scenario statistics averaged within seed first. ACC is percent; AEOD is the absolute TPR gap and ASPD the absolute positive-rate gap. Standard deviations use sampleSD, ddof1; n and cohort names are explicit. The summary refuses incomplete900 input instead of rendering missing cells. Original900 results are not overwritten.

Seed91001 participated in recipe selection; the9 nonselection seeds are91002..91010 and historical6 are91005..91010. Original training environments include cu128/cu130, while this new replay uses CPU isolatedcu128; do not claim all original training runtimes equal. This is validation replay of existing baseline900, excludes new800 mechanism controls, and does not freeze or execute final evaluation. No test image inference, fitting, or selection occurs. Full-file SHA operations read raw bytes that may contain test data; only train+valid metadata prefixes are materialized as semantic arrays.

The actual isolated-Linux stdlib spawn probe `preflight_spawn.json` passed: outer nice0 before/after, spawned child starts0, original `os.nice(10)` produces10, child bound16–23 with idle I/O, parent restored toCPU16. It loaded no scientific modules, images or semantic labels and changed no capability. This validates the corrected inheritance boundary, not the complete CNN orchestration.
