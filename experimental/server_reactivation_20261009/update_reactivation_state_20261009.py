"""Update live project entry from measured receipts, leaving frozen protocols intact."""
from pathlib import Path
import datetime
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'docs/server_deployment_20260923/training_20260923'
CHECKS = TRAIN / 'server_reactivation_20261009'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
state_path = TRAIN / 'TRAINING_STATE.json'
state = read(state_path)
live = read(CHECKS / 'latest_preflight_live.json')
formal = (CHECKS / 'formal_launch.json').exists()
if formal:
    live = read(CHECKS / 'latest_formal_live.json')
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
phase = 'FORMAL800_RUNNING' if formal else 'CU128_REAL_IMAGE_GATES_RUNNING'
state['updated_unix'] = time.time()
state['server_reactivation_20261009'].update(status=phase, new_formal_training=len(live['active'])+live['queue_completed'] if formal else 0,
    new_formal_training_started=formal, exact_restore_acceptance='server_reactivation_20261009/restore_acceptance.json',
    cu130_gates_accepted=20, cu130_offserver_members_verified=149,
    cu128_environment='server_reactivation_20261009/cu128_environment.json',
    latest_live=live, environment_difference='New jobs use isolated torch2.11.0+cu128 and driver595.84; Full controls98cu128+2cu130 were mostly produced on driver570.211.01. Three-round checks cannot establish70-round equivalence.')
state['latest_connection_check'].update(recorded_utc=live['checked_utc'],
    current_internal_runtime=live['service'], torch='2.11.0+cu128',
    new_training_started=formal, next='Accept all20cu128 image gates, verify100Full full science-input identities and freeze before800 formal jobs' if not formal else 'Continue frozen800 missing mechanism jobs; strictly accept and incrementally back up new IDs only')
state['celeba_mechanism_v1'].update(status=phase,
    real_image_cu130_gates_accepted=20, current_runtime='torch2.11.0+cu128 driver595.84',
    queue_dispatched=formal, new_started=(len(live['active'])+live['queue_completed'] if formal else 0),
    preflight_queue_completed=live['queue_completed'] if not formal else 20,
    queue_completed_observed=live['queue_completed'] if formal else 0,
    scientific_results_strictly_accepted=state['celeba_mechanism_v1'].get('scientific_results_strictly_accepted',0))
if formal:
    state['celeba_mechanism_v1'].update(real_image_gates_passed=True,
        real_image_cu128_gates_accepted=20, cross_runtime_three_round_exact=20,
        full_reuse_strictly_accepted=100, new_total=800,
        dispatch_receipt_sha256=sha(TRAIN/'celeba_mechanism_v1/dispatch_receipt.json'),
        validation_only=True, test_started=False)
state['active_services'] = list(dict.fromkeys(state.get('active_services', []) +
    ['guardfed_celeba_mechanism_formal' if formal else 'guardfed_celeba_mechanism_preflight']))
if formal:
    state['current_stage'] = 'celeba_mechanism_v1'
state['last_health_check'] = live
gate_snapshots = list((ROOT / 'tmp/celeba_gradient_realimage_gate_20261009').glob('live_snapshot_*/remote_read_only.json'))
gate_live = None
if gate_snapshots:
    gate_path = max(gate_snapshots, key=lambda p: read(p)['at_unix'])
    gate_live = read(gate_path)
    assert gate_live['read_only']
    if 'scientific_table_records' in gate_live:
        assert gate_live['scientific_table_records'] == 0
    else:
        assert gate_live['scientific_table_records_from_canaries'] == 0
    if 'guide_sha256' in gate_live:
        assert gate_live['guide_sha256'] == '42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
    for row in gate_live['gates'].values():
        if 'canary_artifact_acceptance_count' not in row:
            row.update(canary_artifact_acceptance_count=row['canary_acceptance_files'],
                       expected_canaries=row['total_canaries'],
                       no_strict_summarize_called=row['gate_summary'] is None)
    state['baseline_gate_live_20261009'] = {
        'observed_utc': gate_live['utc'], 'snapshot_sha256': sha(gate_path),
        'snapshot_local_path': gate_path.relative_to(ROOT).as_posix(),
        'scientific_table_records': 0, 'complete_cohort_accepted': False,
        'gates': {name: {'individual_canary_artifact_acceptance_count': row['canary_artifact_acceptance_count'],
                         'expected_canaries': row['expected_canaries'],
                         'no_strict_summarize_called': row['no_strict_summarize_called']}
                  for name, row in gate_live['gates'].items()}}
state['native_monitor_20261009'] = {'id':'guardfed-training-health', 'observed_local_status':'PAUSED',
    'scheduler_status':'not_observable_with_available_tools', 'native_update_tool_available':False,
    'local_prompt_stale_endpoint':True, 'no_scheduler_changed':True,
    'handoff':'server_reactivation_20261009/MONITOR_HANDOFF.md'}
state['synthetic_figure_recovery_20261009'] = {'status':'PASS_WITH_EXPLICIT_PARTIAL_PROVENANCE',
    'original_records':260, 'pca_records':40, 'legacy_triplets_without_common_checkpoint':250,
    'exact_fig3_plot_script_recovered':False, 'executed_forest_diffusion_identity_recovered':False,
    'verification_sha256':sha(TRAIN / 'synthetic_figure_recovery_20261009/verification.json')}
state['final_evaluator_runtime_20261009'] = {'status':'PREPARED_NOT_FROZEN', 'local_acceptance_groups':11,
    'historical_valid_cached_jobs':8, 'all900_native_realimage_valid_replayed':False, 'test_started':False,
    'local_entry':'tmp/celeba_final_evaluator_20261009/README.md',
    'published_entry':'experimental/celeba_final_evaluator_20261009/README.md'}
restore_dir = TRAIN / 'validation900_restore_20261009'
if (restore_dir / 'restore_acceptance.json').exists():
    restored = read(restore_dir / 'restore_acceptance.json')
    independent = read(restore_dir / 'root_canary_independent_verification.json')
    assert restored['status'] == 'EXACT_900_ARTIFACT_STORAGE_VERIFIED'
    assert restored['models'] == 900 and restored['all_artifact_file_hashes_verified'] == 2700
    assert independent['status'] == 'PASS' and len(independent['canaries']) == 2
    state['validation900_restore_20261009'] = {
        'status': restored['status'], 'models': 900, 'artifact_hashes_verified': 2700,
        'existing_full_artifacts_reused': 300, 'new_isolated_artifacts': 2400,
        'original_output_files_modified': 0,
        'restore_acceptance_sha256': sha(restore_dir / 'restore_acceptance.json'),
        'storage_map_sha256': restored['storage_map_sha256'],
        'entry': 'validation900_restore_20261009/README.md'}
    state['final_evaluator_runtime_20261009'].update(
        actual_native_valid_image_replays_accepted=2,
        actual_canary_native_max_abs_difference=0,
        root_independent_verification='validation900_restore_20261009/root_canary_independent_verification.json',
        new_runtime_entry='tmp/celeba_final_valid_replay_20261009/README.md',
        exact_900_artifact_storage_ready=True,
        full900_replay_started=False,
        final_protocol_frozen=False)
phase1_dir = ROOT / 'tmp/celeba_final_valid_replay_20261009/v3/phase1_execution_20261009'
if (phase1_dir / 'offserver_verification.json').exists():
    replay = read(phase1_dir / 'offserver_verification.json')
    acceptance = read(phase1_dir / 'strict_acceptance.json')
    assert replay['status'] == 'PASS' and replay['archive_members_verified'] == 20
    assert sha(phase1_dir / 'strict_acceptance.json') == replay['strict_acceptance_sha256']
    assert acceptance['accepted_ids'] == ['FedAvg_IID_Benign_seed91001'] and acceptance['accepted_n'] == 1
    assert replay['native_max_abs_difference'] == 0 and not acceptance['invalid']
    state['final_evaluator_runtime_20261009'].update(
        status='VALID_REPLAY_PHASE1_ACCEPTED_PHASE2_AUTHORIZED',
        actual_native_valid_image_replays_accepted=3,
        throughput_phase1= {'workers':1, 'accepted':1,
            'batch_wall_seconds':replay['batch_wall_seconds'],
            'accepted_models_per_second':replay['batch_accepted_models_per_second'],
            'offserver_archive_sha256':replay['archive_sha256'],
            'archive_members_verified':20},
        final_protocol_frozen=False, full900_replay_started=False)
    replay_base = phase1_dir.parent
    throughput_plan = read(replay_base / 'throughput_plan.json')
    replay_ids, measured_phases = set(), []
    for planned in throughput_plan['phases']:
        phase_dir = replay_base / ('phase' + str(planned['phase']) + '_execution_20261009')
        if not (phase_dir / 'offserver_verification.json').exists():
            continue
        check = read(phase_dir / 'offserver_verification.json')
        accepted_phase = read(phase_dir / 'strict_acceptance.json')
        assert check['status'] == 'PASS' and sha(phase_dir / 'strict_acceptance.json') == check['strict_acceptance_sha256']
        ids = set(accepted_phase['accepted_ids'])
        assert ids == {m['id'] for m in planned['models']} and not ids.intersection(replay_ids)
        assert accepted_phase['workers'] == planned['workers'] and not accepted_phase['invalid']
        assert accepted_phase['max_abs_native_metric_difference'] == 0
        replay_ids.update(ids)
        measured_phases.append({'phase':planned['phase'], 'workers':planned['workers'],
            'accepted':len(ids),'batch_wall_seconds':accepted_phase['wall_seconds'],
            'models_per_second':accepted_phase['models_per_second'],
            'offserver_archive_sha256':check['archive_sha256'],
            'archive_members_verified':check['archive_members_verified']})
    replay_count = 2 + len(replay_ids)
    state['final_evaluator_runtime_20261009'].update(status='VALID_REPLAY_THROUGHPUT_STAGES_IN_PROGRESS',
        actual_native_valid_image_replays_accepted=replay_count,
        measured_throughput_phases=measured_phases, final_protocol_frozen=False,
        full900_replay_started=False)
phase4_failure_path = phase1_dir.parent / 'phase4_execution_20261009/failure_offserver_verification.json'
phase4_failure = None
if phase4_failure_path.exists():
    phase4_failure = read(phase4_failure_path)
    assert phase4_failure['status'] == 'FAILURE_EVIDENCE_VERIFIED_OFFSERVER_NOT_ACCEPTED'
    assert phase4_failure['accepted_n'] == 0 and phase4_failure['all_member_sha_verified']
    state['final_evaluator_runtime_20261009'].update(
        status='V3_PHASE4_REJECTED_COMPATIBILITY_REPAIR_IN_PREPARATION',
        preserved_phase4_failure_proof=phase4_failure_path.relative_to(ROOT).as_posix(),
        preserved_phase4_failure_proof_sha256=sha(phase4_failure_path),
        phase4_scientific_replays_accepted=0, phase5_started=False,
        failure_scope='FedAA historical rawjob lacks output; seven peer workers interrupted; no training changes',
        failure_archive_sha256=phase4_failure['archive_sha256'],
        failure_archive_members_offserver_verified=phase4_failure['member_n'])
v4 = ROOT / 'tmp/celeba_final_valid_replay_20261009/v4'
v4_semantic_path = v4 / 'semantic900_offserver_verification.json'
v4_semantic = None
if v4_semantic_path.exists():
    v4_semantic = read(v4_semantic_path)
    assert v4_semantic['status'] == 'REAL900_SEMANTIC_RECEIPT_OFFSERVER_IDENTITY_VERIFIED'
    assert v4_semantic['accepted_original_semantic_records'] == 900 and v4_semantic['invalid_n'] == 0
    assert sha(v4 / 'semantic900_inspection.json') == v4_semantic['semantic_receipt_sha256']
    assert sha(v4 / 'replay_v4.py') == v4_semantic['source_sha256']
    assert v4_semantic['new_image_inference'] == 0
    state['final_evaluator_runtime_20261009'].update(
        status='V4_COMPATIBILITY_SEMANTICS_PASS_PHASE4_AUTHORIZED',
        v4_semantic_receipt_sha256=v4_semantic['semantic_receipt_sha256'],
        semantic_originals_accepted=900, semantic_acceptance_is_image_replay=False,
        v4_semantic_offserver_proof=v4_semantic_path.relative_to(ROOT).as_posix(),
        v4_semantic_offserver_proof_sha256=sha(v4_semantic_path))
    for phase_number in (4, 5):
        phase_dir = v4 / f'phase{phase_number}_attempt1_execution_20261009'
        proof_path = phase_dir / 'offserver_verification.json'
        if not proof_path.exists():
            continue
        check = read(proof_path)
        accepted_phase = read(phase_dir / 'strict_acceptance.json')
        contract = read(phase_dir / 'execution_contract.json')
        planned = next(p for p in throughput_plan['phases'] if p['phase'] == phase_number)
        canonical_ids = {m['id'] for m in planned['models']}
        assert check['status'] == 'PASS' and sha(phase_dir / 'strict_acceptance.json') == check['strict_acceptance_sha256']
        assert accepted_phase['accepted_ids'] == contract['selected_ids']
        assert accepted_phase['accepted_n'] == len(canonical_ids) == contract['workers']
        assert contract['sealed_plan_sha256'] == sha(phase1_dir.parent / 'throughput_plan.json')
        assert contract['v4_source_sha256'] == v4_semantic['source_sha256']
        assert not canonical_ids.intersection(replay_ids) and not accepted_phase['invalid']
        assert accepted_phase['max_abs_native_metric_difference'] == 0
        replay_ids.update(canonical_ids)
        measured_phases.append({'phase':phase_number, 'workers':contract['workers'],
            'accepted':len(canonical_ids),'batch_wall_seconds':accepted_phase['wall_seconds'],
            'models_per_second':accepted_phase['models_per_second'],
            'source_version':'v4', 'offserver_archive_sha256':check['archive_sha256'],
            'archive_members_verified':check['archive_members_verified']})
    replay_count = 2 + len(replay_ids)
    state['final_evaluator_runtime_20261009'].update(
        status='VALID_REPLAY_THROUGHPUT_STAGES_IN_PROGRESS',
        actual_native_valid_image_replays_accepted=replay_count,
        measured_throughput_phases=measured_phases,
        v4_phases_accepted=[p['phase'] for p in measured_phases if p.get('source_version') == 'v4'])
    collected_paths = list((v4 / 'execution_20261009').glob('cumulative_*_accepted.json'))
    if collected_paths:
        collected_path = max(collected_paths, key=lambda p: read(p)['accepted_n'])
        collected = read(collected_path)
        assert collected['expected_n'] == 900 and collected['accepted_n'] >= replay_count
        assert collected['accepted_n'] == len(collected['accepted']) == len(set(collected['accepted_ids']))
        # Distinct scientific cells may legitimately save identical weight bytes.
        assert 0 < collected['distinct_checkpoint_sha256_n'] <= collected['accepted_n']
        assert sha(v4 / 'execution_20261009/collect_valid_replay.py') == collected['collector_source_sha256']
        assert not collected['test_inference_performed'] and not collected['final_dispatch_created']
        assert all(r['native_max_abs_difference'] == 0 and r['actual_three_views_verified'] for r in collected['accepted'])
        replay_count = collected['accepted_n']
        state['final_evaluator_runtime_20261009'].update(
            actual_native_valid_image_replays_accepted=replay_count,
            cumulative_unique_checkpoint_acceptance=collected_path.relative_to(ROOT).as_posix(),
            cumulative_unique_checkpoint_acceptance_sha256=sha(collected_path),
            distinct_checkpoint_sha256_count=collected['distinct_checkpoint_sha256_n'],
            all900_native_realimage_valid_replayed=collected['all900_native_valid_replayed'])
flgmm_cpu = ROOT / 'tmp/celeba_flgmm_realimage_gate_20261009'
flgmm_cpu_proof_path = flgmm_cpu / 'OFFSERVER_VERIFICATION.json'
flgmm_cpu_proof = None
if flgmm_cpu_proof_path.exists():
    flgmm_cpu_proof = read(flgmm_cpu_proof_path)
    flgmm_cpu_accepted = read(flgmm_cpu / 'LOCAL_ACCEPTANCE.json')
    assert flgmm_cpu_proof['status'] == 'ARCHIVE_AND_ALL_MEMBERS_PASS'
    assert flgmm_cpu_proof['members_verified'] == 54 and flgmm_cpu_accepted['actual_results_checked'] == 2
    assert sha(flgmm_cpu / 'LOCAL_ACCEPTANCE.json') == flgmm_cpu_proof['local_acceptance_sha256']
    assert flgmm_cpu_accepted['status'] == 'PASS' and not flgmm_cpu_accepted['formal_table_eligible']
    state['flgmm_cpu_canary_20261009'] = {
        'status': 'TWO_REAL_IMAGE_CPU_CANARIES_ACCEPTED_AND_OFFSERVER_VERIFIED',
        'accepted_canaries': 2, 'scientific_table_records': 0,
        'archive_sha256': flgmm_cpu_proof['archive_sha256'], 'members_verified': 54,
        'offserver_proof_sha256': sha(flgmm_cpu_proof_path),
        'gpu_equivalence_claim': False, 'formal_screen_started': False,
        'metrics': [r['metrics'] for r in flgmm_cpu_accepted['results']],
        'negative_constant_predictions_retained': True}
flgmm_gpu = ROOT / 'tmp/celeba_flgmm_gpu_gate_20261009'
flgmm_gpu_proof_path = flgmm_gpu / 'OFFSERVER_VERIFICATION.json'
flgmm_gpu_proof = None
if flgmm_gpu_proof_path.exists():
    flgmm_gpu_proof = read(flgmm_gpu_proof_path)
    flgmm_gpu_result = read(flgmm_gpu / 'GPU_ACCEPTANCE.json')
    flgmm_gpu_inventory = read(flgmm_gpu / 'GPU_BACKUP_MEMBERS.json')
    assert flgmm_gpu_proof['status'] == 'OFFSERVER_BACKUP_AND_NEGATIVE_GATE_REPRODUCTION_PASS'
    assert flgmm_gpu_proof['members_verified'] == 121 and flgmm_gpu_proof['individual_results_rechecked'] == 4
    assert sha(flgmm_gpu / 'GPU_BACKUP_MEMBERS.json') == flgmm_gpu_proof['inventory_sha256']
    assert sha(flgmm_gpu / 'GPU_ACCEPTANCE.json') == flgmm_gpu_inventory['GPU_ACCEPTANCE.json']['sha256']
    assert flgmm_gpu_result['status'] == 'REPEAT_MISMATCH'
    state['flgmm_gpu_canary_20261009'] = {
        'status': 'REPEAT_MISMATCH_PRESERVED_OFFSERVER', 'individual_results_rechecked': 4,
        'archive_sha256': flgmm_gpu_proof['archive_sha256'], 'members_verified': 121,
        'offserver_proof_sha256': sha(flgmm_gpu_proof_path),
        'scientific_table_records': 0, 'formal_screen_started': False,
        'training_model_metric_control_torch_rng_exact': True,
        'failure_scope': 'default_rng JSON includes entropy-seeded SciPy import-only documentation examples',
        'new_recording_boundary_fix': 'PREPARATION_ONLY_NOT_EXECUTED',
        'negative_report_unchanged': flgmm_gpu_proof['original_mismatch_report_unchanged']}
flgmm_v3 = ROOT / 'tmp/celeba_flgmm_gpu_gate_v3_20261009'
flgmm_v3_live_path = flgmm_v3 / 'dispatch/launch_observation.json'
flgmm_v3_live = None
if flgmm_v3_live_path.exists():
    flgmm_v3_live = read(flgmm_v3_live_path)
    assert sha(flgmm_v3 / 'FREEZE.json') == flgmm_v3_live['freeze_sha256']
    assert sha(flgmm_v3 / 'PREPARED_PACKAGE_SHA.json') == flgmm_v3_live['prepared_sha256']
    assert len(flgmm_v3_live['jobs']) == 4 and not flgmm_v3_live['queue_failed']
    state['flgmm_gpu_canary_v3_20261009'] = {
        'status': 'FOUR_CANARY_SCOPE_FROZEN_AND_LAUNCH_OBSERVED',
        'observed_utc': flgmm_v3_live['observed_utc'],
        'service_status_at_observation': flgmm_v3_live['live_service_status'],
        'launch_pid': flgmm_v3_live['launch_pid'],
        'freeze_sha256': flgmm_v3_live['freeze_sha256'],
        'prepared_source_sha256': flgmm_v3_live['prepared_sha256'],
        'individual_canaries_accepted_at_observation': sum(r['accepted'] for r in flgmm_v3_live['jobs']),
        'jobs_at_observation': flgmm_v3_live['jobs'],
        'live_receipt_sha256': sha(flgmm_v3_live_path),
        'strict_four_cohort_accepted': False, 'scientific_table_records': 0,
        'formal_screen_started': False, 'original_negative_gate_preserved': True}
    state['flgmm_gpu_canary_20261009']['new_recording_boundary_fix'] = 'V3_FOUR_GPU_CANARIES_FROZEN_LAUNCH_OBSERVED'
science_backup = CHECKS / 'mechanism_science_backups_20261009'
first_verification = science_backup / 'incremental_new5_offserver_verification.json'
if first_verification.exists():
    proof = read(first_verification)
    receipt_path = science_backup / 'incremental_new5_v3_20261009T073000Z.tar.gz.receipt.json'
    receipt = read(receipt_path)
    inspection = read(science_backup / 'mechanism_inspection_new_v3_20261009T073000Z/inspection.json')
    assert proof['pass'] and proof['different_host_observed'] and proof['members_verified'] == 70
    assert proof['archive_sha256'] == receipt['archive_sha256']
    assert set(proof['accepted_new_ids']) == set(receipt['accepted_new_ids']) == set(inspection['accepted_new_ids'])
    assert inspection['new_count'] == 5 and inspection['reused_count'] == 100 and not inspection['invalid']
    state['celeba_mechanism_v1'].update(scientific_results_strictly_accepted=5,
        scientific_results_offserver_verified=5,
        science_acceptance_inspection='server_reactivation_20261009/mechanism_science_backups_20261009/mechanism_inspection_new_v3_20261009T073000Z/inspection.json',
        latest_science_backup_sha256=proof['archive_sha256'],
        latest_science_backup_members_verified=70,
        backup_tool_version='evidence_v3.py; sealed v2 unchanged',
        active_partial_outputs_are_pending=True,
        mechanism_raw_native_shared_evaluation='PENDING')
    # Count each model once from the actually verified off-server chain.
    ledger = read(science_backup / 'verified_ledger.json')
    verified_proofs = {read(p)['archive_sha256']: read(p)
                       for p in science_backup.glob('*offserver_verification.json')}
    backed_up, previous, backup_entries = set(), None, []
    for entry in ledger['entries']:
        local_receipt = science_backup / Path(entry['receipt']).name
        assert sha(local_receipt) == entry['receipt_sha256']
        record = read(local_receipt)
        proof = verified_proofs[record['archive_sha256']]
        assert proof['pass'] and proof['different_host_observed']
        assert record['previous_receipt_sha256'] == previous
        assert record['manifest_sha256'] == ledger['manifest_sha256']
        ids = set(record['accepted_new_ids'])
        assert ids == set(proof['accepted_new_ids']) and not ids.intersection(backed_up)
        backed_up.update(ids)
        previous = entry['receipt_sha256']
        backup_entries.append({'archive_sha256':record['archive_sha256'],
            'new_ids':sorted(ids),'members_verified':proof['members_verified']})
    inspections = [(p, read(p)) for p in science_backup.glob('mechanism_inspection_*/inspection.json')]
    accepted_path, accepted = max(((p,d) for p,d in inspections if d['status'] != 'INVALID'), key=lambda item:item[1]['new_count'])
    assert backed_up <= set(accepted['accepted_new_ids'])
    state['celeba_mechanism_v1'].update(scientific_results_strictly_accepted=accepted['new_count'],
        scientific_results_offserver_verified=len(backed_up),
        incremental_science_backups=backup_entries,
        latest_science_backup_sha256=backup_entries[-1]['archive_sha256'],
        latest_science_backup_members_verified=backup_entries[-1]['members_verified'],
        science_acceptance_inspection=accepted_path.relative_to(TRAIN).as_posix(),
        backup_tool_version='evidence_v4.py; sealed v1/v2/v3 and original five-model archive unchanged')
flgmm_v3_final_path = flgmm_v3 / 'FINAL_STATUS.json'
if flgmm_v3_final_path.exists():
    terminal = read(flgmm_v3_final_path)
    root_proof_path = CHECKS / 'FLGMM_V3_ROOT_VERIFICATION.json'
    root_proof = read(root_proof_path)
    assert terminal['status'] == 'GATE_COMPLETE_ACCEPTED_BACKED_UP'
    assert terminal['individual_canaries_accepted'] == 4 and terminal['failed'] == 0
    assert root_proof['archive_sha256'] == terminal['archive_sha256']
    assert sha(flgmm_v3/'GPU_ACCEPTANCE.json') == terminal['gpu_acceptance_sha256']
    assert sha(flgmm_v3/'OFFSERVER_VERIFICATION.json') == terminal['offserver_verification_sha256']
    state['flgmm_gpu_canary_v3_20261009'].update(status=terminal['status'],
        terminal_observed_utc=terminal['observed_utc'], service_state='EXITED',
        strict_four_cohort_accepted=True, canaries_accepted_and_backed_up=4,
        cross_gpu_repeat_conditions_passed=2, archive_sha256=terminal['archive_sha256'],
        terminal_receipt_sha256=sha(flgmm_v3_final_path), root_verification_sha256=sha(root_proof_path),
        cpu_gpu_equivalence_claim=False, formal_screen_started=False)
    state['flgmm_gpu_canary_20261009']['new_recording_boundary_fix'] = 'V3_FOUR_GPU_CANARIES_ACCEPTED_OFFSERVER_VERIFIED'
single_dir = ROOT / 'tmp/celeba_mechanism_valid_replay_20261009/recovery_attempt03_20261009T085000Z'
single_root_proof = CHECKS / 'MECHANISM_VALID_ONE_ROOT_VERIFICATION.json'
if single_root_proof.exists():
    proof = read(single_root_proof)
    assert sha(single_dir/'FILES_SHA256.json') == proof['delivery_seal_sha256']
    assert len(proof['accepted_new_ids']) == 1 and proof['archive_members_verified'] == 14
    state['celeba_mechanism_v1'].update(mechanism_raw_native_shared_evaluation='ONE_STRICT_VALID_REPLAY_OFFSERVER_VERIFIED',
        three_view_new_models_accepted=1, three_view_new_models_offserver_verified=1,
        three_view_accepted_ids=proof['accepted_new_ids'], three_view_native_max_abs_difference=0,
        three_view_root_proof_sha256=sha(single_root_proof),
        three_view_scope_limit='One minus_U/IID/Benign checkpoint; no Full reinference, no test, no component necessity inference',
        original_two_engineering_failures_preserved=True)
gradient_dir = ROOT / 'tmp/celeba_gradient_realimage_gate_20261009/completed_four_backup_20261009'
gradient_root_proof = CHECKS / 'GRADIENT_FOUR_ROOT_VERIFICATION.json'
if gradient_root_proof.exists():
    proof = read(gradient_root_proof)
    assert sha(gradient_dir/'strict_delivery.json') == proof['delivery_sha256']
    assert sha(gradient_dir/'offserver_verification.json') == proof['offserver_proof_sha256']
    state['gradient_canaries_20261009'] = dict(status='FOUR_EXPLORATORY_CANARIES_STRICT_ACCEPTED_OFFSERVER_VERIFIED',
        accepted=4, rounds=12, same_point_gradient_checks=240,
        archive_sha256=proof['archive_sha256'], archive_members_verified=86,
        root_verification_sha256=sha(gradient_root_proof), service_state='EXITED',
        scientific_table_records=0, formal_protocol_status='PREPARED_NOT_FROZEN',
        formal_decisions_unresolved=5, negative_constant_predictions_retained=True)
hybrid_dir = ROOT / 'tmp/celeba_hybrid_realimage_gate_20261009'
hybrid_diagnosis = hybrid_dir / 'terminal_failure_diagnosis_20261009.json'
if hybrid_diagnosis.exists():
    diagnosis = read(hybrid_diagnosis)
    off = read(hybrid_dir/'terminal_failure_backup_20261009/offserver_verification.json')
    assert off['pass'] and off['different_host_observed']
    state['hybrid_canaries_20261009'] = dict(status=diagnosis['status'],
        diagnosis_sha256=sha(hybrid_diagnosis), strict_individual_canaries=2, expected=4,
        complete_cohort_accepted=False, scientific_table_records=0, service_state='EXITED',
        failed_id=diagnosis['failed_id'], failed_stage=diagnosis['failed_stage'],
        archive_sha256=off['archive_sha256'], offserver_members_verified=off['members_verified'],
        original_failure_preserved=True, precise_writer_fix='PREPARATION_ONLY_NOT_EXECUTED')
reply = ROOT / 'docs/server_deployment_20260923/revision_20260923/rebuttal_20261009'
state['rebuttal_draft_20261009'].update(sha256=sha(reply/'rebuttal_20261009.md'),verification_sha256=sha(reply/'verification.json'))
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
running = TRAIN / 'RUNNING.md'
old = running.read_text(encoding='utf-8')
boundary = '# HISTORICAL: Nine-method coverage COMPLETE'
assert boundary in old
history = old[old.index(boundary):]
top = f'''# CURRENT: GuardFed mechanism {'formal800' if formal else 'cu128 preflight'} — measured {live['checked_utc']}

用户重新提供89.22.197.55:60350并明确授权停止sglang，现使用实例52183675开展缺失返修实验。sglang已停止；两张5090实际CUDA张量检查通过。先读/etc/vast-agents-guide.md（SHA42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa）。旧完成GuardFed队列仍禁止重启；213.224.31.105:26712内部当前状态未知，不自动切回。

源码五文件、全部RGB64缓存与官方标签/划分身份已核，100已验收Full对照及缺失历史依赖准确恢复（307成员验收，303新建+4原有相同）。20项cu130真实图像三轮门检全部严格接受，两套Full与原worker同horizon模型张量/全部指标/诊断精确一致；149备份成员及archive SHA在本机通过。原cu130门检完整保留在preflight_history/cu130_20261009，仅原job字节复制回空输出路径。

当前使用独立/workspace/guardfed_envs/celeba-cu128-20261009/bin/python，torch2.11.0+cu128、driver595.84，未改原环境/驱动/方法/seed/指标。实测服务{live['service']}；完成{live['queue_completed']}，活动{len(live['active'])}，等待{live['pending']}，失败{len(live['failed'])}；资源与轮次见server_reactivation_20261009/latest_{live['phase']}_live.json。{'800新机制正式队列已启动，70round valid-only，8并发；100Full显式复用；检查dispatch receipt与full identity acceptance。' if formal else '目前仅20项cu128三轮门检，未启动800项科学训练/最终test。所有20项严格接受、实际runtime逐条核验、100Full全科学输入/数据契约验收后才冻结启动800新任务。'}

正式机制设计800新+100复用，IID/non-IID×5场景×10共享seed，按原协议固定recipe。Full98cu128+2cu130且多数原产出driver570.211.01，环境混合须披露；短程门检不能证明70round等价，提供排除两条cu130的配对敏感性口径。PROTOCOL.md为不可改动的准备快照，当前执行事实优先读EXECUTION.md、dispatch receipt、本入口及状态JSON。

原三小时任务guardfed-training-health本地TOML为PAUSED且提示仍指旧阶段；当前没有原生automation_update工具，未修改调度器、未建立替代监督机制。服务器supervisor只管理已启动队列，不等于三小时聊天巡检已恢复。可审阅提示与限制见server_reactivation_20261009/MONITOR_HANDOFF.md。

2454个历史新增完整训练及九方法900验证记录/备份保持原值。返修回复已逐字核24块原意见、40本地引用、209项SHA声明；新增260条生成器/PCA数值追溯与20份CelebA联合分组重建已核，投稿Fig3原脚本/FD执行身份仍缺。其余8基线与正式最终评价仍未完成；准备代码/门检不当作论文科学结果。主入口REBUTTAL_COMPLETION_20261009.md。

'''
if (restore_dir / 'restore_acceptance.json').exists():
    top += '''九方法900终轮模型/result/raw-job已全部精确接入当前服务器：100Full复用现存路径，其他800恢复至独立artifact_store，共2700文件逐SHA核验，原历史output修改0。两条完整valid19867/root16277原图CPU重放已接受，native三指标误差0，raw/native/shared三个视图的18指标与48混淆计数经主代理独立复核；52封存文件及27归档成员离机通过。900全批尚未启动，不称最终评价完成；详见validation900_restore_20261009/README.md。各基线真实图像门检的完整接受及备份状态分开记录，首轮证据不等于完整门检PASS。

'''
if first_verification.exists():
    top += f'''机制新结果已有{accepted['new_count']}项通过独立70轮严格验收，{len(backed_up)}项离机备份，100Full身份复核保持有效；{len(backup_entries)}份增量各SHA/member通过本机验收，原Full权重不重复打包。实时queue完成数与该已验收/备份分母分开。v2首备份因活动日志增长而在preflight拒绝，原检查保留；独立v3处理正常活动目录，新v4修正异常重检的诊断保全路径，经独立审查/回归通过。训练及封存v1/v2/v3不变，首5项归档保持有效。当前不是800或整个返修完成。

CPU端另有Fed-NGA/Huber四条真实图像三轮探索门检已启动，CPU104–111、8线程、独立supervisor、无自动重试；仅首client梯度已实测（7351样本、optimizer0步、同点/符号oracle一致），尚无完整四项PASS。原加载器会物化全split标签元数据，包括test尾部；仅训练/验证像素参与运算，不称untouched test。执行附件见tmp/celeba_gradient_realimage_gate_20261009/EXECUTION_HANDOFF.md。九方法验证CPU重放正在1/2/4/8/11互斥有用任务测吞吐，未启动900全批。

'''
if (phase1_dir / 'offserver_verification.json').exists():
    continuation = ('v3的8并发阶段因FedAA旧rawjob结构不兼容而拒收，0/8完成、7个同批worker中断；40个失败证据成员已离机验收。正在准备独立v4兼容修复，11并发和900全批未启动，原9条结果有效' if phase4_failure else '其余授权阶段依次严格接受/离机后自动推进，未启动900全批')
    if v4_semantic:
        continuation = 'v3的8并发失败证据及原9条有效重放保留；独立v4兼容修复已核900条历史语义身份/2700文件SHA并离机接受，覆盖FedAA/LASA原记录差异。仅原8/11并发有用任务阶段获授权，语义接受不计新图像推理，900全批与test未启动'
    top += f'''九方法验证重放已有{len(replay_ids)}项吞吐阶段新任务严格接受并离机SHA/member验收，加之前2条共{replay_count}个实际重放；native误差0，三视图指标/混淆计数独立重算一致。已完成1/2/4/8/11计划中的前{len(measured_phases)}阶段，只报实测吞吐，不称已知最优或受控提速。{continuation}；阶段明细见tmp/celeba_final_valid_replay_20261009/v3/。该CPU重放只读train-root/valid语义标签，完整文件SHA读取包含test所在字节；它不调用会物化全split标签的原完整loader，不能与梯度gate的元数据边界混淆。

'''
if gate_live:
    counts = [f"{name}：{row['canary_artifact_acceptance_count']}/{row['expected_canaries']}条单项canary产物PASS"
              for name, row in gate_live['gates'].items()]
    top += f"新增基线门检实测（{gate_live['utc']}）：{'；'.join(counts)}。各完整cohort尚未严格汇总/离机验收，科学表记录仍为0；不将三轮canary计入正式70轮结果。原始只读快照及SHA见TRAINING_STATE.json的baseline_gate_live_20261009，保留原startup封存证明。\n\n"
if flgmm_cpu_proof:
    top += 'FLGMM两条完整真实图像CPU三轮canary均已严格接受并离机验收54成员，CPU任务已退出。两条ACC均0.516686、AEOD/ASPD为0的恒定预测负结果保留；Tg1为管线覆盖，不计正式论文结果，不推断CPU/GPU等价。GPU四项跨卡重复门检另行接受；32项搜索尚未启动。凭据见TRAINING_STATE.json的flgmm_cpu_canary_20261009。\n\n'
if flgmm_gpu_proof:
    top += 'FLGMM原v2四项GPU三轮canary均完成，各自身份通过，跨卡训练张量、指标、controller及Torch RNG逐位一致；整体门检按原冻结规则保留REPEAT_MISMATCH。实际差异限于被快照混入的SciPy导入期文档示例default_rng熵状态，原失败报告与121成员已离机核验。记录范围缺陷由独立v3修复；原v2门检不追改，32项搜索未启动；三轮canary不计正式性能结果。\n\n'
if flgmm_v3_live:
    top += f"独立v3修正已通过同core导入回归及42成员离机核验后，明确冻结仅4条GPU三轮canary。实测{flgmm_v3_live['observed_utc']}：{flgmm_v3_live['live_service_status']}，{sum(r['accepted'] for r in flgmm_v3_live['jobs'])}/4单项完成，尚不称完整跨卡cohort通过；旧v2失败证据原样保留，32项搜索仍未启动。源码/冻结与该实测凭据见TRAINING_STATE.json的flgmm_gpu_canary_v3_20261009。\n\n"
if flgmm_v3_final_path.exists():
    top += '更新：FLGMM v3四项GPU门检已全部严格接受、两组跨卡重复精确，139内容成员+清单及四份旧模型引用离机核验，服务正常EXITED；此前3/4启动快照只作历史记录。32项新搜索包已准备，尚未启动，不计正式性能样本。\n\n'
if gradient_root_proof.exists():
    top += '更新：Fed-NGA/Huber四项三轮真实图像门检全部严格接受，240条同点client gradient与攻击符号oracle通过；86成员离机及主代理核验，服务EXITED。四项恒定预测负结果保留；五项正式协议选择仍未冻结，科学表记录0。此前2/4快照及“尚无完整四项PASS”为历史观察。\n\n'
if hybrid_diagnosis.exists():
    top += '更新：Hybrid第三条non-IID/S-DFA在result严格JSON记录层失败，已定位敏感组零方差使原攻击审计相关系数未定义；两条IID已接受，整体2/4未通过。46成员失败证据离机保留，第三条模型有限且三轮aggregate精确仍不能替代完整接受。仅准备限定字段null与独立原因记录的writer修复；未知NaN和主指标/训练状态仍必须拒绝，未重启队列。\n\n'
if single_root_proof.exists():
    top += '机制统一评价更新：一条minus_U/IID/Benign/seed91002终轮checkpoint已完成raw/native/shared三视图重放，native误差0，9指标/24混淆计数/3规则独立重算一致，14成员离机及主代理核验。它与70轮科学训练验收分母分开；Full仅引用已有严格结果，未重复推理/备份，未运行test。两项工程失败及第三次受限恢复链完整保留。\n\n'
running.write_text(top+history,encoding='utf-8')
execution = TRAIN / 'celeba_mechanism_v1/EXECUTION.md'
text = execution.read_text(encoding='utf-8')
marker = '# HISTORICAL PREPARATION SNAPSHOT — no execution at time of preparation\n\n'
if marker in text:
    text = text.split(marker,1)[1]
execution.write_text(top.replace('# CURRENT:', '# CURRENT EXECUTION:')+marker+text,encoding='utf-8')
print(json.dumps({'status':phase,'measured_utc':live['checked_utc'],'completed':live['queue_completed'],'active':len(live['active']),'formal':formal}))
