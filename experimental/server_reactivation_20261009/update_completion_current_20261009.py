"""Refresh only the current completion section from accepted receipts."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/'docs/server_deployment_20260923/training_20260923'
state=json.loads((TRAIN/'TRAINING_STATE.json').read_bytes())
main=state['celeba_mechanism_v1'];baseline=state['final_evaluator_runtime_20261009']
accepted=main['scientific_results_offserver_verified'];replayed=baseline['actual_native_valid_image_replays_accepted']
mechanism_views=main['three_view_new_models_offserver_verified']
complete_native_scenes=main['latest_interim_paper_table']['complete_paired_scenes']
published=state['latest_publication_verification']
paired_table=main.get('latest_paired_three_view_table')
paired_note=('Full paired three-view receipts remain missing and will join actual baseline replay acceptance, without new Full inference or substituting old calibration. '
    if not paired_table else
    'A subsequent root-reviewed identity join now supplies actual Full100 replay receipts and the exact60 Full counterparts for six complete scenes. '
    'The raw/native/shared tables include all three consistent10/9/6-seed panels and paired differences, with972 independently recomputed mean/sampleSD checks. '
    'For these120 records, native and shared-calibration metrics are exactly equal. DeletingU lowers native accuracy in all six scenes by0.309–1.384 percentage points and lowersASPD in all six, '
    'whileAEOD increases in five; rawAEOD decreases in five. These tradeoffs do not establish that every term is necessary. '
    'Full replay mixes5CPU/95GPU overall (5CPU/55GPU in the displayed60); controls useCPU. The disclosed training-build/driver and validation-selection history remain limitations, '
    'and neither a uniform-device final comparison nor a primary endpoint has been established. ')
if paired_table and paired_table['complete_scenes'] == 7:
    paired_note = ('A root-reviewed join supplies70 complete Full–minus_U pairs in seven scenes;71 pairs/142 individual records are retained, with the incomplete non-IID FedSA91001 pair excluded from means. '
        'Raw/native/shared10/9/6-seed tables and paired differences passed1134 independent mean/sampleSD checks. '
        'Native/shared metrics and counts coincide for all140 displayed models. DeletingU lowers native accuracy in all seven scenes by0.309–1.384pp andASPD in all seven;AEOD increases in six, while rawAEOD decreases in five and increases in two(IIDSp-DFA andnon-IID F Flip). '
        'The original six-scene records/statistics are unchanged. Full5CPU/65GPU and69cu128/1cu130 versus70CPU/cu128 controls, driver/runtime and validation-selection limits remain explicit. Neither a uniform-device final comparison nor a primary endpoint has been established. ')
if paired_table and paired_table['complete_scenes'] == 9:
    paired_note = ('The later independently accepted nine-scene join contains90 complete Full–minus_U pairs and retains92 pairs/184 individual records. '
        'The two non-IID Sp-DFA pairs are incomplete and excluded from scene means. Raw/native/shared fixed10/9/6-seed panels include729 displayed cells; '
        '1458 mean/sampleSD scalars and1656 metrics reconstructed from group counts pass independent checks. All189 prior seven-scene statistic rows and81 native92 display rows remain exact. '
        'Native and shared outcomes coincide for all184 models. In the newly complete non-IID FedSA and S-DFA scenes, deletingU lowers raw accuracy by0.492 and0.453pp, with disparity tradeoffs retained. '
        'Complete-scene Full inference is5CPU/85GPU and its training88cu128/2cu130; controls use90CPU/cu128. Mixed devices, historical environments, validation selection and prior test exposure remain limitations. '
        'This is a descriptive validation comparison, not an isolated aggregation effect, a selected primary endpoint or evidence that every scoring term is indispensable. ')
increments=main['incremental_science_backups']
latest_increment=increments[-1]
flgmm=state['flgmm_screen32_20261009']
diagnostic=state.get('native_mismatch_GPU_diagnostic_20261009')
recovery=state.get('baseline_valid_recovery_prepared_20261009')
recovery_paragraph=('The exact 900−424 complement is sealed as a prepared-only proposal: 465 unexecuted models, '
    '10 preserved CPU partial results and one separately verified GPU diagnostic. '
    'The new GPU worker and strict acceptance entry are being implemented separately; '
    'the proposal has dispatched no inference and registered no new acceptance. ' if recovery else '')
gpu_recovery = state.get('baseline_valid_GPU_recovery_20261009')
if gpu_recovery:
    recovery_paragraph = ('The independently reviewed first new GPU replay has passed the original strict checks, '
        '73 archive-member SHA checks and off-server saved-array reconstruction of nine metrics, '
        '24 confusion counts and three prediction rules. Native discrepancy is exactly zero; the old424 collector is unchanged. '
        'The cumulative425 collector explicitly retains CPU/GPU source provenance. The remaining464 are not dispatched; '
        'the10 preserved CPU partial results and one original GPU diagnostic are not registered. '
        'Mixed-device raw/shared results are not a uniform-device final comparison. ')
    if gpu_recovery.get('CPU_partial10_registered') and gpu_recovery.get('diagnostic1_registered'):
        recovery_paragraph = ('The first new GPU replay passed original strict acceptance and73 off-server archive-member checks, '
            'with exact native metrics and saved-array reconstruction. A separate explicit review then revalidated and '
            'registered the original10 CPU partial records and one saved successful GPU diagnostic without CNN inference. '
            'The new436 collector retains CPU434/GPU2 provenance, unchanged historical424/425 collectors and the invalid '
            'original CPU failure. The remaining464 are not dispatched. Mixed-device raw/shared results are not a '
            'uniform-device final comparison. ')
    if gpu_recovery.get('remaining464_dispatched'):
        recovery_paragraph = recovery_paragraph.replace('The remaining464 are not dispatched. ',
            'The exact remaining464 GPU queue is now running in43 bounded chunks with one GPU worker. '
            'CPU106 coordination, CPU105 workers and nice10 were independently observed. '
            'Worker zero exits and remote closures do not increment436 before off-server registration. ')
    if gpu_recovery.get('GPU_remaining464_offserver_accepted'):
        recovery_paragraph = (f'The original436 source chain and {gpu_recovery["GPU_remaining464_offserver_accepted"]} '
            f'new GPU records were strictly accepted and verified off server, closing the legacy chain at460/900 '
            '(CPU434/GPU26). The exact original464 scope is divided into43 bounded chunks; '
            'only independently verified chunks enter separate cumulative collectors. Original424/425/436 collectors '
            'and the invalid CPU output are unchanged. Saved arrays reconstruct all three rules, nine metrics and24 '
            'confusion counts; root refitting is checked by the bound original remote strict tool, not repeated locally. '
            'Mixed CPU/GPU results are not a uniform-device final comparison. ')
        if not gpu_recovery.get('queue_running'):
            recovery_paragraph += ('The queue stopped in chunk002 during a worker resource preflight before CNN inference, '
                'with Protected main800 health failed. The main mechanism queue is currently healthy; the instantaneous '
                'guard inputs were not preserved, so a transient handover is a hypothesis, not a proved unique cause. '
                'The failure archive is retained and the original service has not been restarted. Two completed partial '
                'records remain unregistered; any repair needs a separate reviewed version and exact missing-ID scope. ')
        if gpu_recovery.get('completed_partial2_explicitly_registered'):
            recovery_paragraph = recovery_paragraph.replace(
                'Two completed partial records remain unregistered; ',
                'The two saved partial records subsequently passed original partial strict acceptance and independent saved-array checks and were explicitly imported without new CNN inference; ')
v2 = state.get('baseline_valid_GPU_remaining440_v2_20261009')
if v2:
    recovery_paragraph += (f'The exact remaining440 complement was subsequently deployed in a new V2 namespace with unchanged scientific inference/acceptance and native1e-12. '
        f'Actual Linux processes, CPU106 coordination, CPU105 single-GPU workers, nice10/idle I/O and resource receipts were verified at {v2["measured_utc"]}; '
        f'{v2["worker_exit_complete_observed"]} worker zero exits and {v2["remote_closed_n"]} remote closures were observed in that snapshot. '
        f'Separately, {v2["new_offserver_accepted"]} V2 records have passed original strict acceptance and independent off-server checks, for the current cumulative{replayed}/900. '
        'The first supervisor attempt exited before contract/CNN because its review filename differed from the installed file; original configuration/log bytes were preserved and only the external config path was corrected before starting the fresh queue. '
        'The failed original464 queue and historical collectors remain unchanged. ')
diagnostic_paragraph=('A separately approved one-model GPU diagnostic reproduces the original three native metrics exactly. '
    'Its saved CPU/GPU arrays differ on one native/raw prediction (image172599); shared-calibration predictions match. '
    'The unchanged source/data/checkpoint and all saved metrics, counts and rules were independently verified. '
    'This is one diagnostic execution, not a cohort acceptance or restoration of the failed CPU queue. '
    'Two engineering failures are preserved; the comparison was completed offline without another CNN execution. '
    'Historical GPU per-image outputs are unavailable, so unique historical causality remains unproved. '
    if diagnostic else '')
if baseline.get('GPU_diagnostic_later_explicit_versioned_import'):
    diagnostic_paragraph = diagnostic_paragraph.replace(
        'This is one diagnostic execution, not a cohort acceptance or restoration of the failed CPU queue. ',
        'Its original diagnostic receipt remains unchanged; a later explicit saved-array import is included in the current derived collector. The failed CPU queue remains stopped. ')
screen=state.get('hybrid_screen32_20261009')
failure=state['final_evaluator_runtime_20261009'].get('failed_model_id')
failure_paragraph=("The CPU replay service has fail-stopped on FairGuard/IID/F Flip/seed91009: native metrics differ from the original record despite matching model/config/data identities and unchanged tensors. The original1e-12 tolerance is preserved and the65-member failure archive is verified off server. Chunk036's10 strict partial results were subsequently imported after explicit review into the derived436 source chain; the failed CPU result remains invalid. No original metric, model, threshold or selection rule has changed and the failed CPU service remains stopped. Previously accepted records remain valid. " if failure else "")
paragraph=(f"The original32-item Hybrid validation search has actually started under `{screen['service']}`; its independent source/startup archive and root verification bind the unchanged eight recipes, four conditions, seed91001 and70 rounds. It uses one GPU0 worker, CPU104, one compute thread and nice10. No100-job multi-seed confirmation, test or automatic retry is authorized. "
    if screen else "The unchanged original32-item Hybrid validation search is root-approved for a separate frozen execution copy; exact source/scope approval is complete, while actual dispatch and source-bound startup acceptance remain separate requirements. ")
if screen and screen.get('offserver_accepted70round_jobs'):
    paragraph += (f"The first{screen['offserver_accepted70round_jobs']}/32 terminal jobs have original server strict acceptance and linked off-server backups, including53 members in the original four-job increment and35 in the next two-job increment, with an independent CPU record-layer replay. "
        "Only three current-host runtime metadata queries are bound to the original server receipt; all scientific checks and null diagnostic policies are unchanged. Local torch2.8 CPU is disclosed separately from server torch2.11 cu128 and no local CNN inference or CUDA context was used. "
        "The still-running shared service log is saved as a prefix, not a closed per-job log; terminal artifacts were checked before and after backup. No final recipe is selected from this partial snapshot. ")
paragraph=paragraph.rstrip()
next37_status=main.get('next37_valid_replay',{})
next37_note='Dispatch and new off-server acceptance require their own actual receipts.'
if next37_status.get('startup_root_proof_sha256'):
    next37_note=(f"The exact37 scope has actually started under guardfed_celeba_mechanism_valid_next37, observed at {next37_status['measured_utc']}, "
        "after fresh Linux full source/data/terminal hash, CPU owner and quota checks. One eight-thread CPU worker is restricted to112–119, nice10 and idle I/O with CUDA hidden. "
        "Source-bound startup and actual approval receipts are saved; no new off-server scientific acceptance is inferred from this startup. The old23 replays and Full weights are excluded.")
if next37_status.get('offserver_new_accepted'):
    next37_note += (f" Separately, {next37_status['offserver_new_accepted']} new terminals have passed original strict checks, archive-member verification, independent saved-array reconstruction and root adoption; "
        f"{next37_status['offserver_remaining']} of this37-item scope remain unaccepted off server. All native discrepancies are zero. These acceptances bind later backup receipts rather than the startup snapshot.")
if next37_status.get('status') in ('COMPLETE_STRICT_OFFSERVER_NO_FULL_JOIN','COMPLETE_STRICT_OFFSERVER_PAIRED_SIX_SCENES'):
    next37_note += (f" The later terminal observation at{next37_status['latest_progress_utc']} verified service exit,37 completed IDs,zero residual workers and no failure. "
        "Both disjoint backups are now strict/off-server accepted; the exact37 scope is complete. Later native terminals remain outside this scope, and the subsequent paired Full join has its own actual root proof.")
next11_note = ('Execution has not been inferred from preparation.')
if main.get('next11_valid_replay',{}).get('execution_started'):
    next11_note = ('The separate execution passed39 refusals and actually started under guardfed_celeba_mechanism_valid_next11 after109 Linux source/data/terminal-member checks and actual quota/CPU-owner checks; one eight-thread CPU worker on112–119, nice10/idleI/O and CUDA hidden was independently observed. Old37 and next8 were not restarted.')
if main.get('next11_valid_replay',{}).get('offserver_new_accepted') == 11:
    next11_note += (' The later terminal snapshot verifies normalEXITED,11 complete,zero residual workers and no failure. All11 have original strict acceptance,120 verified archive members and independent saved-array reconstruction of99 metrics,264 confusion counts and33 rules, followed by root source/checkpoint/terminal adoption. Native discrepancy is exactly zero. This closes71 minus_U three-view models; the prior60 are unchanged. The fixed six-scene table remains separate from a future explicitly accepted seven-scene join.')
    if paired_table and paired_table['complete_scenes'] == 7:
        next11_note = next11_note.replace('a future explicitly accepted seven-scene join','the subsequent explicitly root-accepted seven-scene join')
after71_note=''
if main.get('after71_valid_replay'):
    after71=main['after71_valid_replay']
    after71_note=(f'The separate after71 exact11 replay status is {after71["status"]}, with{after71["offserver_new_accepted"]} new offserver acceptances. '
        'It excludes all previously closed71 and Full inference, covering only non-IID FedSA91002–91009 and S-DFA91001–91003. '
        'Remote completion is not acceptance. Existing seven-scene mean/SD tables remain a separate sealed evidence snapshot; these IDs cannot create a complete eighth scene because FedSA91010 is outside the frozen replay scope. '
        'The other seven variants have source-only metadata/recipe preparation, with no supplied terminal or empirical replay gate.')
after82_failure_note=''
if main.get('after82_failed_attempt'):
    failure82=main['after82_failed_attempt']
    after82_failure_note=('The exact10 after82 replay attempt stopped before runtime dependency binding and CNN because its approval guard retained the old11-ID cardinality. '
        'Actual service exit, zero workers/completions and an empty output tree are preserved with source/log/approval evidence; zero new scientific acceptances are inferred. '
        'The original82 accepted records and main800 training are unaffected. A separate engineering version requires a positive valid10 approval gate and fresh namespace; the failed service is not restarted. '
        f'Root failure review: {failure82["root_failure_review_path"]}.')
if main.get('after82_v2_valid_replay'):
    replay82=main['after82_v2_valid_replay']
    after82_failure_note+=(f' The separately reviewed V2 service has actual startup evidence: {replay82["status"]}, '
        f'{replay82["offserver_new_accepted"]} new off-server acceptances; one8-thread CPU worker on112–119, nice10/idleI/O and CUDA hidden. '
        'Positive valid10 approvals and invalid-approval refusals pass; only cardinality and namespace/pins changed, and the failed original service is retained.')
current=f'''## Current accepted increment — measured {state['last_health_check']['checked_utc']}

The main mechanism queue has{main['queue_completed_observed']} terminal jobs observed, with{accepted} independently accepted and backed up off server in{len(increments)} linked increments;100 Full controls remain explicit reuse. The latest{len(latest_increment['new_ids'])}-ID increment has{latest_increment['members_verified']} verified members, archive `{latest_increment['archive_sha256']}`. The queue continues with eight workers and no observed failures. These counts do not establish all800 controls or the whole rebuttal.

The existing nine-method terminal-model validation replay has {replayed} distinct accepted and off-server-verified models, with {900-replayed} still missing. Each closed increment has original strict acceptance and independent raw/native/shared prediction-array checks; all preserve native metrics exactly. The cumulative collector is `{baseline['accepted_collection_path']}`. {failure_paragraph}{diagnostic_paragraph}{recovery_paragraph}This is validation replay, not final-test evaluation or new model training.

The approved exact15 mechanism replay increment is complete and its service is EXITED: three disjoint backups contain149 content members plus three inventories, with135 metrics,360 confusion counts and45 prediction rules independently reconstructed. Together with the original eight, the historical increment closed23 actual minus_U terminal checkpoints. Subsequent explicitly adopted increments bring the current total to{mechanism_views} strict, off-server raw/native/shared validation replays; native discrepancy is zero. {paired_note}This does not establish the complete mechanism comparison.

The accepted native mechanism data currently support{complete_native_scenes} complete Full–minus_U scenes, with ten paired seeds per scene and consistent additional nine-seed and six-seed panels. In non-IID Benign, Full hasACC88.594±1.168%, AEOD0.00696±0.00423 andASPD0.06511±0.00902; minus_U has87.290±1.870%,0.01300±0.00732 and0.05140±0.01598. Full therefore has higher mean accuracy and lower meanAEOD but higher meanASPD in this scene. These native end-to-end results retain calibration effects and do not establish an isolated aggregation cause or that every score term is necessary. The next37 replay scope is exactly the historical accepted60 minus the already closed23; source review and36 scientific no-CNN rejection checks plus32 execution rejection checks have passed. {next37_note}

FLGMM has{flgmm['offserver_accepted70round_jobs']}/32 full70-round validation-search jobs strictly accepted and backed up off server using the frozen original acceptor. The original search continues. A partial candidate summary is not the final selection across all eight candidates; no winner has been selected and the100-job multi-seed coverage has not started.

Hybrid's CPU4 and separate CUDA4 pipeline gates are complete, strict and backed up off server. The CUDA increment has49 verified members; both same-GPU Hybrid/legacy pairs match terminal tensors, every-round metrics, attacks, diagnostic fields and RNG exactly, excluding only cumulative wall time as in the original comparator. All four CUDA terminals predict a constant negative class, withACC0.516686 and zero gaps. These negative short-run outcomes are preserved and cannot establish performance advantage, CPU/CUDA equivalence or70-round equivalence. {paragraph}

The exact11 subsequent evaluation scope is native71 minus the closed60: non-IID F Flip seeds91001–91010 and FedSA91001. Its source review preserves scientific functions and passes42 refusals. {next11_note}

{after71_note}

{after82_failure_note}

The latest previously verified Git publication is `{published['commit']}`, with{published['committed_blobs_sha256_verified']} committed blob SHA checks against the remote branch. New completion evidence is published in a separate increment. Source preparation, approval, dispatch and completed scientific results are distinct. The native three-hour chat monitor remains PAUSED; supervisor-managed training does not restore it. Final test, the remaining baselines, complete mechanism comparisons and final manuscript claims remain unfinished.

The complete900-record three-view descriptive paper tables are now independently root-reviewed:900 original receipts rejoined,8100 metrics reconstructed from saved group counts and4860 mean/sampleSD scalars checked. AllIID/non-IID/five-scene/fixed10/9/6-seed views are retained under outputs/guardfed_tables/celeba_nine_method_three_view_20261009. Old native metrics/displayed values remain exact;94 oldJSON sampleSD last-bit differences(max2.78e-17) are disclosed without changing tolerance. The24-comment complete author-review reply now uses the accepted seven-scene snapshot at {state['latest_rebuttal_draft']['entry']}; v2 corrects only the P2 six/seven-scene wording and adds the accepted900 provenance, with no new performance claim. Submission remains gated on the full cohort, and the manuscript source is not yet applied.

An independently checked accepted900 descriptive attribution now accompanies that draft:2052 scalar checks pass after averaging the ten scenarios within each seed. Native ACC/AEOD/ASPD mean advantages involve6/8,8/8,7/8 baselines; common calibration gives7/8,4/8,1/8. These are direction counts, not seed win rates or significance. GuardFed's native and shared outcomes are identical, and the baseline calibration changes the comparison. The English validation900 addendum preserves all negative differences, selection/device limits and the undecided primary endpoint; no new inference/refitting/test occurred.

The accepted nine-method three-view tables are also compiled into a nine-page A3 landscape PDF, covering all views and fixed10/9/6 panels. Independent root verification checks2430 displayed mean/SD pairs(4860 scalar strings), exact original fragment bytes and label-only compile copies, all nine page bounds and visual renders; the original67-member seal is unchanged. This is display compilation only, not new statistical or experimental evidence.

The bounded manuscript locator found historical IEEEtran source in paper.md, but its title, method and table structure differ from the submitted PDF and its three named bibliography/figure dependencies were not found in the searched paths. Nineteen input SHA pins and three source variants were independently checked. No matching submitted-version source or complete build is claimed, and the original manuscript was not edited. The submitted source-path question is pending while independent experimental work continues.

A concrete historical Fig3 correction candidate now plots original round70 ACC against AEOD and ASPD, retaining all26 settings and260 same-round records;78 mean checks and PDF/PNG review pass. It is explicitly a single-seed author-review candidate, not an adopted replacement or recovery of original plotting/execution/checkpoint identities. No FairScore, metric-wise extrema, test-based round selection or scene-as-seed SD is used. Historical test exposure, unresolved ForestDiffusion execution and PCA-label limitations remain in the caption and figure. P4 remains open.

'''
p=TRAIN/'REBUTTAL_COMPLETION_20261009.md';text=p.read_text(encoding='utf8')
start=text.index('## Current accepted increment');end=text.index('## Historical accepted increment',start)
p.write_text(text[:start]+current+text[end:],encoding='utf8')
print(json.dumps(dict(updated_current_section=True,mechanism_strict=accepted,baseline_replays=replayed,mechanism_views=mechanism_views,goal_complete=False)))
