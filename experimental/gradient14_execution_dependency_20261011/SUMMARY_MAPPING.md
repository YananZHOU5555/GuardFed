# Actual64 → existing coverage input: source/schema preparation only

No SSH, fitting, inference, training, recipe selection, protocol freeze, gate binding, or actual64 summary was performed. Prepared does not mean dispatched. This note uses the actual adopted42 schema, not a claim that 64 now exist. Root-owned newer observations are outside this inspection.

## Existing entry to reuse

`tmp/celeba_gradient_fullcoverage_prepare_20261010/prepare.py` already implements `selected(records, protocol, manifest, score)` and `bind(...)`; reuse them unchanged. SHA `7f179f38d6c3b4dddd37522d67f2e1e0065fc421e5e3fb6ddf2e4419ad3735ec`.

`tmp/celeba_gradient_fullcoverage_gates_prepare_20261010/metadata.py` already binds the actual selected/frozen coverage into exact14. SHA `71616f5fba61d9fee704f3d57bfe220e7c4cdfbdecd95fc7abde6d6037ea50ef`.

Targeted Python-source search across the gradient screen/coverage and native acceptance namespaces found no existing strict-delta-to-64-row mapper. The screen `run_queue.py` only emits QUEUE_PROGRESS and its server-completed IDs. It does not emit the scientific summary needed by `prepare.py`. Coverage `summary.py` summarizes the eventual 200 records; it is not a screen64 mapper. Do not run it to fill this gap.

## Actual accepted source shape

Read `tmp/gradient_native_after39_20261011/ROOT_ADOPTION_REVIEW.json`, actual SHA `7eec0792793dab9777fda31ca55c779dd68827e9827186f45dbe706cb9fea1ea`: accepted_total42, 32 FedNGA + 10 Huber, screen64_complete=false. Its offserver proof holds only the new3 scientific rows; accepted_job_ids is cumulative42. The same distinction applies to earlier incremental receipts. Concatenating cumulative ID lists or treating newest records as all42 would be wrong.

The root links the handoff and offserver receipt hashes; handoff links prior root/offserver paths and hashes. The original strict record has id, rounds, metrics, checkpoint SHA, frozen job SHA, original training provenance. The offserver record additionally records data_contract, tensor layout check, constant_negative_retained. RAW_STORAGE_INDEX maps exact original artifact member paths to bytes/SHA. Native root adoption is authoritative; an unadopted server terminal or offserver proof alone is insufficient.

The eventual minimal translation is a single pass through this already adopted chain, oldest to newest, retaining all per-batch records once in frozen manifest order. Verify parent and offserver hashes, cumulative ordered prefixes, disjoint new IDs, and union exactly the original64 manifest. Read existing small result/acceptance JSON through their verified raw-member references. No checkpoint reload or repeated scientific checker is needed merely to translate already adopted evidence. A new independent framework, service or full science rerun would add no benefit.

## Exact row mapping

| Coverage summary field | Original accepted source and equality requirement |
|---|---|
| id | Original manifest entry id = frozen job id = original-strict/offserver record id |
| method, distribution, attack | Frozen job values, equal saved result values |
| candidate | Frozen job.tuning_candidate = saved result.tuning_candidate |
| seed, rounds, alpha | Frozen job.config seed/rounds/client_alpha, equal saved result seed/rounds/alpha; enforce 91001/70 and IID5000 or nonIID5 |
| evaluation_split | Saved result.data_contract.image_data_contract.evaluation_split = valid; also validate official evaluation rows19867. Not a top-level result field. |
| job_sha256 | Original manifest job_sha256 = SHA(original frozen jobs file) = strict/offserver job_sha256 = result.provenance.job_sha256 |
| source_hashes, component_hashes, local_hashes | Frozen job maps = original_training_provenance maps = saved result.provenance maps |
| metrics | Same saved result.metrics = original-strict metrics = offserver metrics; finite ACC/AEOD/ASPD in [0,1] |
| result_sha256 | Actual result.json bytes SHA = raw-member index SHA = acceptance.artifact_hashes[result.json] |
| model_sha256 | Adopted member index model.pt SHA = strict/offserver checkpoint_sha256 = acceptance.artifact_hashes[model.pt]; no need to load/rewrite weights |
| acceptance_sha256 | Actual acceptance.json bytes SHA = adopted member-index SHA; require its status PASS |
| output | Existing original run directory, absolute and usable on the host where coverage bind/eventual summary runs; never a copied/new model directory |
| strict_pass, offserver_verified | Set true only from the hash-bound original-strict and offserver records whose batch root has actually adopted them; do not infer from status strings in result alone |

Two real schema hazards were checked on the actual new3 first record: copied runs/job.json has SHA40bb8e... whereas the original frozen job/provenance SHA is 7dd61a... (different serialization). Use original jobs bytes for job_sha256, never copied runs/job.json. Also saved result has no top-level evaluation_split; its nested image contract has valid. Saved result and offserver metrics matched exactly for this one schema sample. This is not a new acceptance of all42.

The final normalized summary should retain all64 records, plus source receipt references and limitations, including constant/negative results. Ranking must call existing selected() after actual64 adoption, not duplicate the score implementation. It averages the same four IID/nonIID × Benign/S-DFA scores per candidate, separately by method, lexical candidate-ID tie break. Retain all candidates, accuracy champions and Pareto information under the original protocol; n=1, no sampleSD/significance. These reporting additions must not replace the complete64 records or selection rule.

## Missing actual pins, in dependency order

1. A final native root adopting exactly all original64 IDs, with the actual verified delta chain. Current inspected root42 cannot pass this gate. Do not synthesize a complete root from QUEUE_PROGRESS or observations.
2. The normalized complete64 summary path/SHA, derived only from those accepted records. No winner may be calculated before item1. This is the small identity translation still to implement once the final root schema/path exists; no fake placeholders should be supplied to executable bind.
3. A separate root-reviewed selection receipt with status ROOT_GRADIENT64_COMPLETE_STRICT_OFFSERVER_ADOPTED, accepted_count64, all64_offserver_verified=true, test=false, summary_sha256, screen_source_seal_sha256=11e2ae63c87e0440669a047c930f5465b13babf559cca17bd8808bf693ce7ced, and selected_candidates matching original selected(). The final incremental native adoption and this selected-summary receipt have different schemas and roles; do not simply rename one to the other.
4. Existing prepare.bind produces PREPARED_NOT_FROZEN protocols, 192 new jobs +8 reused references. Root review/freezing must separately regenerate protocol/job/manifest hashes, install exact original component bytes, and issue ROOT_GRADIENT200_FROZEN_SOURCE_ADOPTED with actual bound_inputs_sha256 and both manifest_sha256 values. Neither prepare.bind nor this note authorizes freeze.
5. Existing metadata.bind then creates PREPARED_EXACT14_NOT_DISPATCHED. Each actual gate child still requires fresh original resource approval, old64 exited, healthy main queue, free approved8 CPU IDs and scope/job hashes. All runtime gates remain unchanged.

## INPUT_TEMPLATE mapping (template itself unchanged)

- complete64_summary_path / sha256 ← actual normalized summary from item2.
- complete64_root_adoption_path / sha256 ← item3 selection receipt (not the latest partial/native root).
- selected_candidates ← item3 and existing selected() exact agreement.
- actual_new_jobs=192, actual_reused_records=8 only after actual bound manifests exist and are checked; before that they describe the planned shape only.
- future_frozen_stage_approval_sha256 ← item4 actual approval; stays null before adoption.
- actual_three_new_attack_gate ← actual14 result/root evidence only after execution and acceptance; binding is not a passed gate.
- execution_authorized remains false during mapping/preparation; test remains false. The template is descriptive; existing CLIs take explicit arguments, not this template as an execution instruction.

## Existing executable entries after the real pins exist

```text
python -B tmp/celeba_gradient_fullcoverage_prepare_20261010/prepare.py --summary <actual64_summary> --summary-sha256 <actual_sha> --root-adoption <actual_selected64_root> --root-sha256 <actual_sha> --out <new_nonexisting_prepared_dir>
```

After separate reviewed freeze and actual source approval, preserving the expected dependency layout on the execution host:

```text
python -B tmp/celeba_gradient_fullcoverage_gates_prepare_20261010/metadata.py --bound <actual_frozen_bound_dir> --stage-approval <actual_root_source_approval> --approval-sha256 <actual_sha> --out <new_nonexisting_gate_dir>
```

These are existing entry signatures, not commands run this turn. A Windows F:/ output reference cannot be silently used as a Linux run path. The selected summary and BOUND_INPUTS absolute paths must be valid on their actual consumer host; path changes alter summary/receipt hashes and require explicit rebinding, not patching sealed JSON after adoption. Root should decide the intended host before normalizing output paths. No execution wrapper was added: actual chain traversal can be a small local translation when final64 root is real, while selection, coverage creation and gate binding already exist.
