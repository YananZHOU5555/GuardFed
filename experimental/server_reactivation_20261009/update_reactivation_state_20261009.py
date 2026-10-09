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
        new_completed=accepted['new_count'], scientific_results_completed=accepted['new_count'],
        scientific_results_offserver_verified=len(backed_up),
        incremental_science_backups=backup_entries,
        latest_science_backup_sha256=backup_entries[-1]['archive_sha256'],
        latest_science_backup_members_verified=backup_entries[-1]['members_verified'],
        science_acceptance_inspection=accepted_path.relative_to(TRAIN).as_posix(),
        backup_tool_version='evidence_v4.py; sealed v1/v2/v3 and original five-model archive unchanged')
    state['mechanism_evidence_tooling_20261009'].update(
        status='SEALED_VALIDATORS_AND_INCREMENTAL_TERMINAL_ACCEPTANCE',
        live_new_strictly_accepted=accepted['new_count'], live_reused_strictly_accepted=100,
        live_inspection_sha256=sha(accepted_path), offserver_new_count=len(backed_up))
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
bounded_startup_root = CHECKS / 'BOUNDED_STARTUPS4_ROOT_VERIFICATION.json'
bounded_started = bounded_startup_root.exists()
if bounded_started:
    proof = read(bounded_startup_root)
    hybrid_start = hybrid_dir / 'startup_delivery_20261009'
    observed = read(hybrid_start / 'startup_receipt.json')
    assert sha(hybrid_start / 'FILES_SHA256.json') == proof['hybrid_delivery_seal_sha256']
    assert sha(hybrid_start / 'startup_receipt.json') == proof['hybrid_startup_receipt_sha256']
    state['hybrid_canaries_20261009'].update(
        status='TWO_NONIID_CANARIES_BOUNDED_WRITER_RECOVERY_RUNNING',
        original_stage_status=diagnosis['status'], precise_writer_fix='APPROVED_EXECUTED_ORIGINAL_NINE_SOURCE_SEAL_UNCHANGED',
        service_state='RUNNING_AT_STARTUP_OBSERVATION',
        service_at_observation=proof['hybrid_service_at_observation'],
        observed_utc=proof['hybrid_observed_utc'],
        first_real_round=observed['first_round']['round'],
        first_round_metrics=observed['first_round']['metrics'],
        recovery_new_canaries_accepted=0, reused_IID_canaries=2,
        engineering_failures_preserved=2, repeated_IID_inference_or_training=0,
        startup_delivery_members_verified=20, startup_root_proof_sha256=sha(bounded_startup_root))
    seven_start = ROOT / 'tmp/celeba_mechanism_valid_replay_20261009/remaining_seven_startup_delivery_20261009'
    observed = read(seven_start / 'startup_receipt.json')
    assert sha(seven_start / 'FILES_SHA256.json') == proof['seven_delivery_seal_sha256']
    assert sha(seven_start / 'startup_receipt.json') == proof['seven_startup_receipt_sha256']
    state['celeba_mechanism_v1']['remaining_seven_valid_replay'] = dict(
        status='EXACT_SEVEN_ORIGINAL_TERMINALS_RUNNING_AT_OBSERVATION',
        service_at_observation=proof['seven_service_at_observation'],
        observed_utc=proof['seven_observed_utc'], new_original_terminals=7,
        already_accepted_seed91002_excluded=True, new_training=0, new_Full_inference=0,
        new_test_inference=0, offserver_accepted_in_startup_receipt=0,
        source_seal_sha256=observed['source_seal_sha256'], approval_sha256=observed['approval_sha256'],
        startup_delivery_members_verified=20, startup_root_proof_sha256=sha(bounded_startup_root))
    state['active_services'] = list(dict.fromkeys(state['active_services'] + [
        'guardfed_celeba_hybrid_writer_repair_v2', 'guardfed_celeba_mechanism_valid_remaining7']))
seven_root = CHECKS / 'MECHANISM_VALID_SEVEN_ROOT_VERIFICATION.json'
hybrid_complete_root = CHECKS / 'HYBRID_REPAIRED_TWO_ROOT_VERIFICATION.json'
if hybrid_complete_root.exists():
    proof = read(hybrid_complete_root)
    folder = ROOT / 'tmp/celeba_hybrid_realimage_gate_20261009/repaired_two_completed_backup_20261009'
    assert sha(folder / 'FILES_SHA256.json') == proof['source_seal_sha256']
    assert sha(folder / 'offserver_verification.json') == proof['offserver_proof_sha256']
    assert proof['aggregate_four_canaries_accepted'] and proof['scientific_table_records'] == 0
    state['hybrid_canaries_20261009'].update(
        status='FOUR_CPU_CANARIES_STRICT_OFFSERVER_ACCEPTED_TWO_NEW_TWO_REUSED',
        service_terminal_state='EXITED', service_state='EXITED', complete_canaries=4,
        complete_cohort_accepted=True, recovery_new_canaries_accepted=2,
        new_canaries=2, reused_canaries=2,
        root_verification_sha256=sha(hybrid_complete_root),
        archive_sha256=proof['archive_sha256'], archive_members_verified=56,
        CPU_gate_not_CUDA_or70_round_equivalence=True, scientific_table_records=0)
    state['active_services'] = [s for s in state['active_services'] if s != 'guardfed_celeba_hybrid_writer_repair_v2']
if seven_root.exists():
    proof = read(seven_root)
    folder = ROOT / 'tmp/celeba_mechanism_valid_replay_20261009/remaining_seven_completed_backup_20261009'
    assert sha(folder / 'FILES_SHA256.json') == proof['delivery_seal_sha256']
    assert sha(folder / 'offserver_verification.json') == proof['offserver_verification_sha256']
    assert proof['cumulative_new_mechanism_three_views'] == 8 and proof['native_max_abs_difference'] == 0
    single_ids = read(single_root_proof)['accepted_new_ids']
    assert not set(single_ids).intersection(proof['accepted_new_ids'])
    combined_ids = sorted(single_ids + proof['accepted_new_ids'])
    assert len(combined_ids) == 8
    state['celeba_mechanism_v1'].update(
        mechanism_raw_native_shared_evaluation='EIGHT_STRICT_VALID_REPLAYS_OFFSERVER_VERIFIED',
        three_view_new_models_accepted=8, three_view_new_models_offserver_verified=8,
        three_view_accepted_ids=combined_ids, three_view_scope_limit='Eight actual minus_U/IID/Benign terminals; no Full reinference or test')
    state['celeba_mechanism_v1']['remaining_seven_valid_replay'].update(
        status='SEVEN_COMPLETE_STRICT_OFFSERVER_AND_ROOT_INDEPENDENTLY_ACCEPTED',
        service_terminal_state='EXITED', accepted_new_ids=proof['accepted_new_ids'],
        archive_sha256=proof['archive_sha256'], archive_members_verified=74,
        independent_metrics=63, independent_confusion_counts=168, independent_prediction_rules=21,
        root_verification_sha256=sha(seven_root), paired_Full_six_three_views_missing=True)
    state['active_services'] = [s for s in state['active_services'] if s != 'guardfed_celeba_mechanism_valid_remaining7']
fifteen_root = CHECKS / 'MECHANISM_FIFTEEN_STARTUP_ROOT_VERIFICATION.json'
if fifteen_root.exists():
    proof = read(fifteen_root)
    folder = ROOT / 'tmp/celeba_mechanism_valid_incremental_v2_execution_20261009/startup_delivery'
    assert sha(folder/'startup_delivery.tar.gz') == proof['archive_sha256']
    observed = proof['raw_live_evidence']
    workers = [p for p in observed['processes'] if p['role'] == 'worker']
    assert len(workers) == 1 and workers[0]['cpus'] == list(range(112,120))
    assert workers[0]['nice'] == 10 and workers[0]['cuda_visible_devices'] == ''
    assert workers[0]['user_seconds'] + workers[0]['system_seconds'] > 0
    assert workers[0]['thread_environment']['OMP_NUM_THREADS'] == workers[0]['thread_environment']['MKL_NUM_THREADS'] == '8'
    assert observed['batch_failure'] is None and not observed['new_training'] and not observed['new_Full_inference'] and not observed['new_test_inference']
    state['celeba_mechanism_v1']['incremental_fifteen_valid_replay'] = dict(
        status='EXACT15_RUNNING_AT_STARTUP_OBSERVATION',service_at_observation=observed['service'],
        observed_utc=observed['utc'],selected_original_terminals=15,earlier_eight_excluded=True,
        new_training=0,new_Full_inference=0,new_test_inference=0,
        remote_strict_complete_at_observation=len(observed['completed']),
        startup_not_offserver_scientific_acceptance=True,
        source_seal_sha256=proof['execution_source_seal_sha256'],
        startup_archive_sha256=proof['archive_sha256'],startup_members_verified=44,
        root_startup_verification_sha256=sha(fifteen_root),cpu_threads=8,cpus=list(range(112,120)),
        entry='tmp/celeba_mechanism_valid_incremental_v2_execution_20261009')
    state['active_services'] = list(dict.fromkeys(state['active_services']+['guardfed_celeba_mechanism_valid_incremental15']))
    previous_ids = state['celeba_mechanism_v1']['three_view_accepted_ids']
    incremental_proofs = list(CHECKS.glob('MECHANISM_VALID_INCREMENTAL_*_ROOT_VERIFICATION.json'))
    newly_accepted_ids = []
    for p in incremental_proofs:
        checked = read(p)
        batch = ROOT/'tmp/celeba_mechanism_valid_incremental_v2_execution_20261009/backups'/checked['batch']
        assert sha(batch/'incremental_valid_three_views.tar.gz') == checked['archive_sha256']
        assert sha(batch/'OFFSERVER_VERIFICATION.json') == checked['offserver_verification_sha256']
        assert not set(checked['accepted_new_ids']).intersection(previous_ids+newly_accepted_ids)
        assert checked['native_max_abs_difference'] == checked['Full_reinference'] == checked['original_models_repacked'] == 0
        newly_accepted_ids.extend(checked['accepted_new_ids'])
    if newly_accepted_ids:
        combined = sorted(previous_ids+newly_accepted_ids)
        state['celeba_mechanism_v1'].update(three_view_new_models_accepted=len(combined),
            three_view_new_models_offserver_verified=len(combined),three_view_accepted_ids=combined,
            mechanism_raw_native_shared_evaluation='INCREMENTAL_STRICT_VALID_REPLAYS_OFFSERVER_VERIFIED',
            three_view_scope_limit='Actual accepted minus_U terminals only; original eight plus explicit new deltas; no Full reinference or test')
        state['celeba_mechanism_v1']['incremental_fifteen_valid_replay'].update(
            new_offserver_root_accepted=len(newly_accepted_ids),new_offserver_root_accepted_ids=newly_accepted_ids,
            original_eight_reference_only=True,root_incremental_proof_paths=[p.relative_to(TRAIN).as_posix() for p in incremental_proofs])
hybrid_cuda_start_root = CHECKS / 'HYBRID_CUDA_STARTUP_ROOT_VERIFICATION.json'
if hybrid_cuda_start_root.exists():
    proof = read(hybrid_cuda_start_root)
    folder = ROOT/'tmp/celeba_hybrid_cuda_execution_20261009/execution_attachments/startup_backup'
    assert sha(folder/'FILES_SHA256.json') == proof['startup_delivery_seal_sha256']
    assert sha(folder/'offserver_verification.json') == proof['offserver_proof_sha256']
    state['hybrid_cuda_gate_20261009'] = dict(status='FOUR_CUDA_CANARIES_RUNNING_AT_STARTUP_OBSERVATION',
        observed_service=proof['observed_service'],observed_first_round=1,gate_total=4,
        source_startup_members_verified=81,source_seal_sha256=proof['execution_source_seal_sha256'],
        root_verification_sha256=sha(hybrid_cuda_start_root),max_processes=1,cpu_threads=1,
        allowed_cpus=[104],physical_gpu=0,nominal_compute_threads_including_gate=107,
        quota_cores=proof['quota_cores'],startup_not_complete_gate=True,
        formal_screen32_started=False,test_started=False,automatic_retry=False)
    state['active_services'] = list(dict.fromkeys(state['active_services']+['guardfed_celeba_hybrid_cuda_four']))
flscreen_root = CHECKS / 'FLGMM32_STARTUP_ROOT_VERIFICATION.json'
flscreen_started = flscreen_root.exists()
if flscreen_started:
    proof = read(flscreen_root)
    folder = ROOT / 'tmp/celeba_flgmm_screen_20261009_v2_dispatch'
    observed = read(folder/'FIRST_PROGRESS_V2.json')
    assert sha(folder/'FIRST_PROGRESS_V2.json') == proof['first_progress_sha256']
    state['flgmm_screen32_20261009'] = dict(status='FROZEN_RUNNING_VALID_ONLY',
        total=32, rounds=70, seed=91001, accepted70round_jobs=0,
        observed_utc=observed['utc'], service_at_observation=observed['service'],
        package_sha256=proof['package_sha256'], source_archive_members_offserver_verified=67,
        startup_receipts_offserver_verified=4, startup_root_proof_sha256=sha(flscreen_root),
        active_at_observation=[dict(id=r['id'],round=r['progress']['round'],gpu=r['gpu']) for r in observed['active']],
        formal100_started=False,test_started=False,automatic_retry=False,
        entry='tmp/celeba_flgmm_screen_20261009_v2_dispatch/BACKUP_HANDOFF.md')
    state['flgmm_gpu_canary_v3_20261009']['formal_screen_started'] = True
    state['active_services'] = list(dict.fromkeys(state['active_services']+['guardfed_celeba_flgmm_screen']))
    fl_growth = folder / 'observations/growth_20261009T095851Z.json'
    if fl_growth.exists():
        assert sha(fl_growth) == '83286cee25ad20f95a3b6f541c3464620657db052e5f65ab22d3638b7aa75549'
        state['flgmm_screen32_20261009'].update(
            subsequent_growth_observation=fl_growth.relative_to(ROOT).as_posix(),
            subsequent_growth_observation_sha256=sha(fl_growth),
            growth_from_rounds=[2,1], growth_to_rounds=[52,49], observed_complete70=0,
            subsequent_failure_count=0, rounds_are_not_completed_results=True)
fl_two_root = CHECKS / 'FLGMM_FIRST_TWO_ROOT_VERIFICATION.json'
if fl_two_root.exists():
    proof = read(fl_two_root)
    folder = ROOT / 'tmp/celeba_flgmm_screen_20261009_v2_dispatch'
    assert sha(folder / 'backups/first_two_20261009/OFFSERVER_ACCEPTANCE.json') == proof['offserver_acceptance_sha256']
    assert sha(folder / 'BACKUP_CHAIN_first_two_20261009.json') == '61d40a7e91ae4d05f4bafaa3bfcffaca0292a68f58e3fe9d491bd41aa3588b11'
    state['flgmm_screen32_20261009'].update(
        accepted70round_jobs=2, offserver_accepted70round_jobs=2, not_yet_accepted=30,
        first_partial_archive_sha256=proof['archive_sha256'], first_partial_archive_members_verified=22,
        first_partial_records=proof['records'], first_partial_root_proof_sha256=sha(fl_two_root),
        candidate_selection_performed=False, scientific_fullcoverage_complete=False)
fl_delta_root = CHECKS / 'FLGMM_DELTA_FOUR_20261009T1133Z_ROOT_VERIFICATION.json'
if fl_delta_root.exists():
    proof = read(fl_delta_root)
    folder = ROOT / 'tmp/celeba_flgmm_screen_20261009_v2_dispatch'
    chain_path = folder / 'BACKUP_CHAIN_increment_20261009T1133Z.json'
    chain = read(chain_path)
    assert sha(chain_path) == 'e2e6b6f98abc5d66bc493c2b4e09dc6d9470fca46116778d01e08c69328528a4'
    assert proof['accepted'] == chain['accepted'] == 6 and proof['accepted_new'] == 4
    assert proof['archive_sha256'] == chain['new_batch']['archive_sha256']
    assert proof['offserver_acceptance_sha256'] == chain['new_batch']['offserver_acceptance_sha256']
    state['flgmm_screen32_20261009'].update(
        accepted70round_jobs=6, offserver_accepted70round_jobs=6, not_yet_accepted=26,
        latest_delta_root_verification=fl_delta_root.relative_to(TRAIN).as_posix(),
        latest_delta_root_proof_sha256=sha(fl_delta_root), latest_backup_chain_sha256=sha(chain_path),
        latest_delta_archive_sha256=proof['archive_sha256'], latest_delta_members_verified=45,
        latest_delta_records=proof['records'], candidate_selection_performed=False)
remaining_root = CHECKS / 'REMAINING872_STARTUP_ROOT_VERIFICATION.json'
remaining_started = remaining_root.exists()
if remaining_started:
    proof = read(remaining_root)
    folder = ROOT / 'tmp/celeba_final_valid_replay_20261009/v4/remaining872_execution_20261009'
    observed = read(folder/'live_start_sample.json')
    assert sha(folder/'live_start_sample.json') == proof['live_sample_sha256']
    state['final_evaluator_runtime_20261009'].update(status='REMAINING872_VALID_REPLAY_RUNNING',
        full900_valid_replay_dispatched=True, reused_actual_acceptances=28, remaining_actual_dispatch=872,
        new_bulk_actual_acceptances_offserver_verified=0, scientific_workers=11,threads_per_worker=8,
        observed_worker_nice=10,observed_outer_nice=0,source_archive_members_offserver_verified=38,
        startup_root_proof_sha256=sha(remaining_root), service_at_observation=observed['service'],
        observed_effective_global_cpu_cores=observed['global_effective_cpu_cores'],
        observed_cpu_quota_cores=observed['quota_cores'],test_started=False)
    state['active_services'] = list(dict.fromkeys(state['active_services']+['guardfed_celeba_valid_remaining872_20261009']))
    new_collections = list(folder.glob('cumulative_*_accepted.json'))
    if new_collections:
        collection_path = max(new_collections,key=lambda p:read(p)['accepted_n'])
        collection = read(collection_path)
        n = collection['accepted_n']
        assert n == len(set(collection['accepted_ids'])) == len(collection['accepted']) and 28 <= n <= 900
        assert collection['collector_source_sha256'] == '19066f63c341b9ee23b7c6f491802cfdde0c1c0c833c2fe64724a16de9bb2234'
        assert collection['prepared_remaining_manifest_sha256'] == 'ad6eebf517f534fb8489acb241c51a9ec5328bb285406e55275f7dd9c0c3ed43'
        assert len(collection['missing_ids']) == 900-n and not collection['test_inference_performed']
        assert set(collected['accepted_ids']) <= set(collection['accepted_ids'])
        assert sha(folder/f'collection_inputs_{n}.json') == collection['collection_inputs_sha256']
        assert all(r['native_max_abs_difference'] == 0 and r['actual_three_views_verified'] for r in collection['accepted'])
        state['final_evaluator_runtime_20261009'].update(actual_native_valid_image_replays_accepted=n,
            actual_valid_three_view_replays_accepted=n,new_bulk_actual_acceptances_offserver_verified=n-28,
            accepted_collection_path=collection_path.relative_to(ROOT).as_posix(),accepted_collection_sha256=sha(collection_path),
            cumulative_unique_checkpoint_acceptance=collection_path.relative_to(ROOT).as_posix(),
            cumulative_unique_checkpoint_acceptance_sha256=sha(collection_path),
            distinct_checkpoint_sha256_count=collection['distinct_checkpoint_sha256_n'],
            all900_native_realimage_valid_replayed=collection['all900_native_valid_replayed'],
            all900_three_view_valid_replayed=collection['all900_three_views_valid_replayed'],missing900_models=900-n)
    diagnostic_path = folder / 'bounded_queue_cpu_diagnostic_10s.json'
    if diagnostic_path.exists():
        assert sha(diagnostic_path) == 'd0bc67386f6268e549a19dbb1c66a8db54688b3df2d71a21987d7d676a7a39e3'
        diagnostic = read(diagnostic_path)
        assert diagnostic['status'] == 'STEADY10S_PROC_DIAGNOSTIC_COMPLETE'
        interpretation = folder / 'bounded_queue_cpu_diagnostic_interpretation.json'
        assert sha(interpretation) == 'ef299d39c1823085433eb47f9bcdab4a847a10d561c17b683b161fa8c56d7ef6'
        state['final_evaluator_runtime_20261009']['cpu_diagnostic'] = dict(
            measurement_sha256=sha(diagnostic_path), interpretation_sha256=sha(interpretation),
            sample_seconds=diagnostic['sample_seconds'],
            global_effective_cores=diagnostic['global_effective_cpu_cores'],
            quota_cores=diagnostic['quota_cores'],
            actual_cnn_effective_cores=diagnostic['cnn_effective_cpu_cores'],
            eleven_eight_thread_workers_are_not_88_actual_cores=True,
            low_cpu_target_diagnosis_performed=True,
            supported_window_observations='No worker disk-read bytes, major page faults or cgroup throttling in this sample',
            unresolved='Cannot identify CNN operator, allocator or memory bandwidth cause from endpoint samples',
            execution_or_scientific_changes=0, artificial_load=0)
reply = ROOT / 'docs/server_deployment_20260923/revision_20260923/rebuttal_20261009'
state['rebuttal_draft_20261009'].update(sha256=sha(reply/'rebuttal_20261009.md'),verification_sha256=sha(reply/'verification.json'))
publication_proofs = list(TRAIN.glob('publication_*verified_20261009.json'))
if publication_proofs:
    publication_proof = max(publication_proofs, key=lambda p: read(p).get('verified_utc', ''))
    verified = read(publication_proof)
    assert verified['status'] == 'COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS'
    state['latest_publication_verification'] = dict(
        commit=verified['commit'], branch=verified['branch'], verified_utc=verified['verified_utc'],
        committed_blobs_sha256_verified=verified['committed_blobs_sha256_verified'],
        proof_path=publication_proof.relative_to(TRAIN).as_posix(), proof_sha256=sha(publication_proof),
        earlier_publication_snapshot_is_historical=True)
fifteen_final_path = ROOT/'tmp/celeba_mechanism_valid_incremental_v2_execution_20261009/FINAL_DELIVERY.json'
fifteen_complete = fifteen_final_path.exists() and len(newly_accepted_ids) == 15
if fifteen_complete:
    final = read(fifteen_final_path)
    assert sha(fifteen_final_path) == '8098ae66620adc85c3168aa16df78880f3462691708f5e91655b7e216d32cf16'
    assert final['accepted_new'] == 15 and final['mechanism_replayed_from_this_snapshot_total'] == len(combined) == 23
    assert final['remaining_workers'] == final['failures'] == final['new_training'] == final['new_Full_inference'] == final['new_test_inference'] == 0
    assert 'EXITED' in final['service'] and set(final['selected_ids']) == set(newly_accepted_ids)
    state['celeba_mechanism_v1']['incremental_fifteen_valid_replay'].update(
        status=final['status'],terminal_service=final['service'],terminal_workers=0,failures=0,
        final_delivery_sha256=sha(fifteen_final_path),final_delivery_verified_utc=final['verified_utc'])
    state['active_services'] = [s for s in state['active_services'] if s != 'guardfed_celeba_mechanism_valid_incremental15']
hybrid_cuda_complete_root = CHECKS/'HYBRID_CUDA_FOUR_ROOT_VERIFICATION.json'
if hybrid_cuda_complete_root.exists():
    proof = read(hybrid_cuda_complete_root)
    assert proof['status'] == 'ROOT_ARCHIVE_AND_PAIRED_CUDA_TENSORS_PASS' and proof['actual_cuda_canaries_accepted'] == 4
    state['hybrid_cuda_gate_20261009'].update(status='FOUR_CUDA_CANARIES_STRICT_ACCEPTED_OFFSERVER',
        terminal_service='EXITED',actual_cuda_canaries_accepted=4,incremental_members_verified=49,
        incremental_archive_sha256=proof['archive_sha256'],root_complete_verification_sha256=sha(hybrid_cuda_complete_root),
        paired_model_tensors_exact=True,constant_negative_prediction_preserved=True,startup_not_complete_gate=False,
        scientific_table_records=0,CPU_CUDA_or_70_round_equivalence_claim=False)
    state['active_services'] = [s for s in state['active_services'] if s != 'guardfed_celeba_hybrid_cuda_four']
baseline_failure_root = CHECKS/'BASELINE_VALID_CHUNK036_FAILURE_ROOT_VERIFICATION.json'
if baseline_failure_root.exists():
    proof = read(baseline_failure_root)
    assert proof['status'] == 'ROOT_NATIVE_METRIC_MISMATCH_FAILURE_PRESERVED_OFFSERVER'
    assert proof['failed_model_id'] == 'FairGuard_IID_F-Flip_seed91009' and proof['native_tolerance_unchanged'] == 1e-12
    state['final_evaluator_runtime_20261009'].update(status='FAILSTOP_NATIVE_METRIC_MISMATCH_PRESERVED',
        actual_service_terminal=proof['terminal_service'],failed_model_id=proof['failed_model_id'],
        failed_chunk='chunk_036',failed_chunk_partial_strict_not_counted=10,
        failure_root_proof_sha256=sha(baseline_failure_root),failure_archive_sha256=proof['archive_sha256'],
        native_mismatch=proof['metric_differences'],native_tolerance_unchanged=1e-12,
        numerical_cause_not_established=True,automatic_retry=False,original_source_or_metrics_changed=False)
    state['active_services'] = [s for s in state['active_services'] if s != 'guardfed_celeba_valid_remaining872_20261009']
hybrid_screen_root = CHECKS/'HYBRID_SCREEN32_STARTUP_ROOT_VERIFICATION.json'
if hybrid_screen_root.exists():
    proof = read(hybrid_screen_root)
    assert proof['status']=='ROOT_HYBRID32_INCREMENTAL_SOURCE_AND_REAL_GPU_STARTUP_PASS'
    assert proof['source_members_restorable']==69 and proof['new_members_verified']==73 and proof['scientific_accepted_records']==0
    state['hybrid_screen32_20261009']=dict(status='RUNNING_AT_SOURCE_BOUND_FIRST_ROUND_OBSERVATION',
        service=proof['service'],observed_service=proof['observed_service'],observed_unix=proof['observed_unix'],
        source_seal_sha256=proof['source_seal_sha256'],scope_sha256=proof['scope_sha256'],
        execution_approval_sha256=proof['execution_approval_sha256'],startup_archive_sha256=proof['archive_sha256'],
        root_startup_verification_sha256=sha(hybrid_screen_root),startup_new_members_verified=73,
        source_members_restorable=69,source_reused_members=20,first_round=1,snapshot_round=proof['snapshot_round'],
        cpu_threads=1,cpus=[104],nice=10,physical_gpu=0,source_and_data_before_after_verified=True,
        original_grid_unchanged=True,new_jobs=32,rounds=70,seed_n=1,seed=91001,validation_only=True,
        scientific_70round_results_strict_accepted=0,formal100_started=False,test_started=False,automatic_retry=False,
        entry='tmp/celeba_hybrid_screen_execution_20261009/execution_dispatch_v1/startup_incremental_backup/BACKUP_HANDOFF.md')
    state['hybrid_cuda_gate_20261009']['formal_screen32_started']=True
    state['active_services']=list(dict.fromkeys(state['active_services']+[proof['service']]))
gpu_diagnostic_root = CHECKS / 'NATIVE_GPU_DIAGNOSTIC_ROOT_VERIFICATION.json'
if gpu_diagnostic_root.exists():
    proof = read(gpu_diagnostic_root)
    folder = ROOT / 'tmp/celeba_native_mismatch_diagnostic_execution_20261009'
    delivery = read(folder / 'FINAL_DELIVERY.json')
    assert sha(folder / 'FINAL_DELIVERY.json') == proof['delivery_sha256']
    assert proof['status'] == 'ROOT_SAVED_CPU_GPU_ARRAYS_AND_ARCHIVES_PASS_DIAGNOSTIC_ONLY'
    assert delivery['scientific_GPU_executions'] == 1 and proof['accepted_cohort_unchanged'] == 424
    state['native_mismatch_GPU_diagnostic_20261009'] = dict(
        status='COMPLETE_DIAGNOSTIC_ONLY_NOT_COHORT_ACCEPTED', id=delivery['id'],
        native_GPU_max_abs_difference=0, original_tolerance=1e-12,
        prediction_flip_image_ids=proof['prediction_flip_image_ids'],
        original_CPU_failure_preserved=True, scientific_GPU_executions=1,
        scientific_seconds=delivery['scientific_seconds'],
        historical_GPU_array_available=False, unique_historical_cause_established=False,
        source_and_weights_unchanged=True, original872_restarted=False,
        delivery_sha256=sha(folder / 'FINAL_DELIVERY.json'), root_proof_sha256=sha(gpu_diagnostic_root),
        archive_members_including_inventories=proof['archive_members_verified'],
        preserved_engineering_failures=delivery['preserved_failures'],
        entry='tmp/celeba_native_mismatch_diagnostic_execution_20261009/FINAL_DELIVERY.json')
    state['final_evaluator_runtime_20261009']['GPU_diagnostic_completed_not_added_to_cohort'] = True
recovery_folder = ROOT / 'tmp/celeba_valid_recovery_prepared_20261009'
if (recovery_folder / 'PACKAGE_SHA256.json').exists():
    assert sha(recovery_folder / 'PACKAGE_SHA256.json') == '586d4443ae6f074e686f872400dd90b91356f0e504d29faa8c8cb0ff3ea8e26a'
    recovery = read(recovery_folder / 'manifest.json')
    assert sha(recovery_folder / 'manifest.json') == '125ab344736a0bf306be99fefc71ca701daf7fc0c337f76b3da7dd62e24c5058'
    assert len(recovery['records']) == 476 and all(x is None for x in recovery['root_review_decisions'].values())
    state['baseline_valid_recovery_prepared_20261009'] = dict(
        status='EXACT476_PREPARED_ONLY_GPU_ACCEPTOR_IMPLEMENTATION_IN_PROGRESS',
        remaining=476, unexecuted=465, CPU_partial_awaiting_review=10, GPU_diagnostic_awaiting_review=1,
        accepted_cohort_count_unchanged=424, deployed=False, dispatched=False, new_acceptances=0,
        old872_restarted=False, tolerance=1e-12,
        manifest_sha256=sha(recovery_folder / 'manifest.json'), package_sha256=sha(recovery_folder / 'PACKAGE_SHA256.json'),
        entry='tmp/celeba_valid_recovery_prepared_20261009/README.md')
gpu_execution = ROOT / 'tmp/celeba_valid_gpu_recovery_execution_20261009'
gpu_collector_path = gpu_execution / 'cumulative_425_accepted.json'
gpu_recovery_paragraph = 'exact900−424=476恢复提案已封存：465未执行、10 CPU partial和1 GPU诊断。独立GPU工具已准备，执行以外部root审阅和实际凭据为准。'
if gpu_collector_path.exists():
    collector = read(gpu_collector_path)
    verified_path = gpu_execution / 'first1_backup/ROOT_OFFSERVER_VERIFICATION.json'
    verified = read(verified_path)
    assert collector['accepted_n'] == len(set(collector['accepted_ids'])) == 425
    assert collector['new_GPU_verification_sha256'] == sha(verified_path)
    assert verified['status'] == 'ROOT_FIRST_GPU_VALID_REPLAY_OFFSERVER_AND_SAVED_ARRAY_PASS'
    assert verified['native_max_abs_difference'] == 0 and verified['old424_unchanged']
    state['final_evaluator_runtime_20261009'].update(
        status='GPU_RECOVERY_FIRST1_ACCEPTED_AND_OFFSERVER_VERIFIED',
        original_CPU_service_failstop_preserved=True,
        actual_native_valid_image_replays_accepted=425,
        actual_native_valid_image_replays_remaining=475,
        accepted_collection_path=gpu_collector_path.relative_to(ROOT).as_posix(),
        accepted_collection_sha256=sha(gpu_collector_path),
        cumulative_unique_checkpoint_acceptance=gpu_collector_path.relative_to(ROOT).as_posix(),
        cumulative_unique_checkpoint_acceptance_sha256=sha(gpu_collector_path),
        explicitly_mixed_CPU_GPU_provenance=True, uniform_device_comparison=False)
    state['baseline_valid_GPU_recovery_20261009'] = dict(
        status='FIRST1_STRICT_OFFSERVER_REGISTERED_REMAINING464_NOT_DISPATCHED',
        accepted_new_ids=verified['accepted_new_ids'], accepted_new_n=1,
        source_package_sha256=verified['source_package_sha256'],
        review_sha256=verified['review_sha256'], archive_sha256=verified['archive_sha256'],
        archive_members_verified=verified['archive_members_verified'],
        offserver_verification_path=verified_path.relative_to(ROOT).as_posix(),
        offserver_verification_sha256=sha(verified_path),
        CPU_partial10_registered=False, diagnostic1_registered=False,
        remaining464_dispatched=False, native_tolerance=1e-12,
        original424_unchanged=True, old872_restarted=False, test=False, training=False)
    state['baseline_valid_recovery_prepared_20261009'].update(
        status='FROZEN476_PROPOSAL_FIRST_GPU_RECORD_ACCEPTED', initial_missing=476,
        current_missing=475, unexecuted=464, deployed=True, dispatched=True, new_acceptances=1)
    gpu_recovery_paragraph = ('476恢复提案的首条未执行记录已在新GPU工具中完成run、原strict、本机离机与保存数组独立核验，'
        '原三指标差值0；73成员SHA、9指标、24混淆计数及3预测规则通过。累计425/900，旧424未改。'
        '其余464未派发，10 CPU partial和1原诊断仍未登记；原CPU数值失败保留。')
import_collector_path = gpu_execution / 'cumulative_436_accepted.json'
if import_collector_path.exists():
    collector = read(import_collector_path)
    checked_path = gpu_execution / 'preserved11_import/ROOT_OFFSERVER_IMPORT_VERIFICATION.json'
    checked = read(checked_path)
    assert collector['accepted_n'] == len(set(collector['accepted_ids'])) == 436
    assert collector['preserved11_import_verification_sha256'] == sha(checked_path)
    assert checked['status'] == 'ROOT_PRESERVED_IMPORT11_STRICT_AND_OFFSERVER_REPORTS_PASS'
    assert checked['new_CNN_inference'] == 0 and checked['original_CPU_failure_still_invalid']
    for name, row in checked['members'].items():
        assert sha(checked_path.parent / name) == row['sha256']
    state['final_evaluator_runtime_20261009'].update(
        actual_native_valid_image_replays_accepted=436, actual_native_valid_image_replays_remaining=464,
        accepted_collection_path=import_collector_path.relative_to(ROOT).as_posix(),
        accepted_collection_sha256=sha(import_collector_path),
        cumulative_unique_checkpoint_acceptance=import_collector_path.relative_to(ROOT).as_posix(),
        cumulative_unique_checkpoint_acceptance_sha256=sha(import_collector_path),
        GPU_diagnostic_completed_not_added_to_cohort=False,
        GPU_diagnostic_later_explicit_versioned_import=True)
    state['baseline_valid_GPU_recovery_20261009'].update(
        status='436_ACCEPTED_INCLUDING_EXPLICIT_PRESERVED11_IMPORT_REMAINING464_NOT_DISPATCHED',
        CPU_partial10_registered=True, diagnostic1_registered=True,
        preserved11_new_CNN_inference=0, preserved11_offserver_proof_sha256=sha(checked_path),
        current_collector_sha256=sha(import_collector_path))
    state['baseline_valid_recovery_prepared_20261009'].update(current_missing=464)
    gpu_recovery_paragraph = ('独立GPU工具首1已strict及离机接受，原三指标差值0、73成员SHA通过。'
        '随后另经显式审阅重验并登记原CPU partial10和已保存GPU诊断1，新增CNN推理0；'
        '累计436/900（CPU434、GPU2），旧424/425账本及原CPU数值失败不改。余464未派发。')
queue_execution = ROOT / 'tmp/celeba_valid_gpu_remaining464_execution_20261009'
if (queue_execution / 'ROOT_LAUNCH.json').exists():
    launch = read(queue_execution / 'ROOT_LAUNCH.json')
    observed_path = max(queue_execution.glob('live_*.json'))
    observed = read(observed_path)
    assert launch['status'] == 'TARGETED464_SERVICE_START_OBSERVED'
    assert launch['queue_package_sha256'] == observed['source_package_sha256'] == 'fa5626ad0ab8be12ac501aea531d7b8ad2f2c05b1a18937dd86fb3708acc8d6b'
    queue_running = 'RUNNING' in observed['service'] and not observed['worker_failed'] and not observed['queue_failure']
    state['baseline_valid_GPU_recovery_20261009'].update(
        status='436_ACCEPTED_GPU_QUEUE_RUNNING' if queue_running else '436_ACCEPTED_GPU_QUEUE_STOPPED_PRESERVED', remaining464_dispatched=True,
        service='guardfed_celeba_valid_gpu_remaining464_20261009',
        queue_source_package_sha256=observed['source_package_sha256'],
        queue_review_sha256=launch['review_sha256'],
        queue_startup_path=(queue_execution / 'ROOT_LAUNCH.json').relative_to(ROOT).as_posix(),
        queue_live=observed, queue_live_sha256=sha(observed_path),
        coordinator_CPU=106, worker_CPU=105, threads=1, nice=10,
        queued_new_replays=464, remote_closed_not_offserver=observed['remote_closed_n'],
        queue_running=queue_running, queue_failure=observed['queue_failure'])
    state['final_evaluator_runtime_20261009']['status'] = '436_ACCEPTED_GPU_VALID_REPLAY_RUNNING' if queue_running else '436_ACCEPTED_GPU_VALID_REPLAY_STOPPED_PRESERVED'
    if queue_running:
        state['active_services'] = list(dict.fromkeys(state['active_services'] + ['guardfed_celeba_valid_gpu_remaining464_20261009']))
    else:
        state['active_services'] = [s for s in state['active_services'] if s != 'guardfed_celeba_valid_gpu_remaining464_20261009']
    gpu_recovery_paragraph = ('独立GPU工具首1及另行审阅的原CPU partial10/成功GPU诊断1已strict、离机和登记，'
        '累计436/900（CPU434、GPU2）；导入11新增CNN推理0。原424/425账本和CPU失效现场不改。'
        '余464已按原顺序、43批次、每批至多11条、单GPU/单线程启动；实际CPU106协调、CPU105 worker/nice10已核。'
        f"最近仅观测{observed['worker_complete_exit_only']}条worker退出成功、{observed['remote_closed_n']}条远端闭合，未离机登记前不增加436。")
evidence_dir = ROOT / 'tmp/celeba_valid_gpu_remaining464_evidence_20261009'
closed_collectors = list(evidence_dir.glob('chunk_*/cumulative_*_accepted.json'))
if closed_collectors:
    import importlib.util
    spec = importlib.util.spec_from_file_location('closed_gpu_evidence', evidence_dir / 'evidence.py')
    evidence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evidence)
    assert sha(evidence_dir / 'PACKAGE_SHA256.json') == '52f46820fd3d01ea532af1ec70f7a8c731116662541c85307549730ac20f609b'
    for name, row in read(evidence_dir / 'PACKAGE_SHA256.json')['members'].items():
        assert sha(evidence_dir / name) == row['sha256']
    current_path = max(closed_collectors, key=lambda p: read(p)['accepted_n'])
    current = read(current_path)
    context = evidence.context()
    evidence.prior_chain(current_path, sha(current_path), context, current['last_chunk_index'] + 1)
    proof_path = Path(current['new_proof_path'])
    proof = read(proof_path)
    assert sha(proof_path) == current['new_proof_sha256']
    assert proof['native_max_abs_difference'] <= 1e-12 and not proof['final_test']
    assert sha(proof_path.parent / 'chunk_evidence.tar.gz') == proof['archive_sha256']
    n = current['accepted_n']
    state['final_evaluator_runtime_20261009'].update(
        status=f'{n}_ACCEPTED_GPU_VALID_REPLAY_RUNNING' if queue_running else f'{n}_ACCEPTED_GPU_VALID_REPLAY_STOPPED_PRESERVED',
        actual_native_valid_image_replays_accepted=n, actual_native_valid_image_replays_remaining=900-n,
        accepted_collection_path=current_path.relative_to(ROOT).as_posix(),
        accepted_collection_sha256=sha(current_path),
        cumulative_unique_checkpoint_acceptance=current_path.relative_to(ROOT).as_posix(),
        cumulative_unique_checkpoint_acceptance_sha256=sha(current_path))
    state['baseline_valid_GPU_recovery_20261009'].update(
        status=f'{n}_ACCEPTED_GPU_QUEUE_RUNNING' if queue_running else f'{n}_ACCEPTED_GPU_QUEUE_STOPPED_PRESERVED',
        current_collector_sha256=sha(current_path), GPU_remaining464_offserver_accepted=n-436,
        current_CPU_provenance_n=434, current_GPU_provenance_n=n-434,
        current_chunk_proof_path=proof_path.relative_to(ROOT).as_posix(), current_chunk_proof_sha256=sha(proof_path),
        remote_closed_not_offserver=max(0, observed['remote_closed_n']-(n-436)))
    state['baseline_valid_recovery_prepared_20261009']['current_missing'] = 900-n
    gpu_recovery_paragraph = (f'旧436来源链及新GPU队列的前{current["last_chunk_index"]+1}批已严格验收、离机登记，'
        f'累计{n}/900（CPU434、GPU{n-434}）；import11新增CNN推理0，旧424/425/436账本及原CPU失效现场不改。'
        '原464范围按43批次、每批至多11条、单GPU/单线程派发；实际CPU106协调、CPU105 worker/nice10已核。'
        '已登记与远端闭合分开统计，保存预测按原规则重建9指标、24混淆计数；root重拟合核验引用原远端strict，未声称本机重新拟合。')
    if not queue_running:
        failure_dir = queue_execution / 'failure_chunk002'
        failure_proof = failure_dir / 'ROOT_OFFSERVER_FAILURE_VERIFICATION.json'
        state['baseline_valid_GPU_recovery_20261009'].update(
            stopped_chunk=observed['queue_failure']['failed_chunk'], original_service_not_restarted=True,
            failure_diagnosis_path=(queue_execution / 'ROOT_FAILURE_DIAGNOSIS_20261009.json').relative_to(ROOT).as_posix(),
            completed_partial_not_registered=observed['worker_complete_exit_only']-observed['remote_closed_n'])
        if failure_proof.exists():
            checked = read(failure_proof)
            assert checked['status'] == 'ROOT_FAILURE_CHUNK002_ARCHIVE_MEMBER_OFFSERVER_PASS_NOT_ACCEPTED'
            assert sha(failure_dir / 'failure_chunk_evidence.tar.gz') == checked['archive_sha256']
            state['baseline_valid_GPU_recovery_20261009'].update(
                failure_offserver_proof_sha256=sha(failure_proof), failure_archive_sha256=checked['archive_sha256'],
                failure_before_CNN=True, instantaneous_main_queue_snapshot_missing=True,
                unique_turnover_cause_proved=False)
        gpu_recovery_paragraph += ('第三批在worker资源预检、CNN之前触发Protected main800 health failed而自动停下。'
            '当前主训练正常，但失败瞬间的三个健康条件未保存原始截图，不能唯一归因为任务交接。'
            '原服务未重启，两条成功partial未登记；修复需独立版本和不重复已完成项的补集。')
partial_path = queue_execution / 'partial2_explicit_import/cumulative_460_accepted.json'
if partial_path.exists():
    partial = read(partial_path)
    checked_path = Path(partial['new_proof_path'])
    checked = read(checked_path)
    assert sha(checked_path) == partial['new_proof_sha256']
    assert sha(Path(partial['previous_collector_path'])) == partial['previous_collector_sha256'] == sha(current_path)
    assert partial['accepted_ids'] == current['accepted_ids'] + checked['accepted_new_ids']
    assert partial['accepted_n'] == len(set(partial['accepted_ids'])) == 460
    assert checked['status'] == 'ROOT_EXPLICIT_GPU_PARTIAL2_ORIGINAL_STRICT_AND_SAVED_ARRAY_PASS'
    assert checked['new_CNN_inference'] == 0 and checked['missing9_not_accepted']
    assert sha(checked_path.parent / 'original_partial_strict_acceptance.json') == checked['partial_strict_sha256']
    assert checked['saved_metrics_verified'] == 18 and checked['saved_confusion_counts_verified'] == 48
    state['final_evaluator_runtime_20261009'].update(
        status='460_ACCEPTED_GPU_VALID_REPLAY_STOPPED_PRESERVED',
        actual_native_valid_image_replays_accepted=460, actual_native_valid_image_replays_remaining=440,
        accepted_collection_path=partial_path.relative_to(ROOT).as_posix(), accepted_collection_sha256=sha(partial_path),
        cumulative_unique_checkpoint_acceptance=partial_path.relative_to(ROOT).as_posix(),
        cumulative_unique_checkpoint_acceptance_sha256=sha(partial_path))
    state['baseline_valid_GPU_recovery_20261009'].update(
        status='460_ACCEPTED_GPU_QUEUE_STOPPED_PRESERVED', current_collector_sha256=sha(partial_path),
        GPU_remaining464_offserver_accepted=24, current_GPU_provenance_n=26,
        completed_partial_not_registered=0, completed_partial2_explicitly_registered=True,
        partial2_no_CNN_import_proof_path=checked_path.relative_to(ROOT).as_posix(),
        partial2_no_CNN_import_proof_sha256=sha(checked_path))
    state['baseline_valid_recovery_prepared_20261009']['current_missing'] = 440
    gpu_recovery_paragraph = gpu_recovery_paragraph.replace('累计458/900（CPU434、GPU24）', '累计460/900（CPU434、GPU26）').replace(
        '两条成功partial未登记；修复需独立版本和不重复已完成项的补集。',
        '两条成功partial随后通过原strict部分验收与保存数组复核显式登记，新增CNN推理0；仍缺440条，修复需独立版本和不重复已完成项的补集。')
guard_review = CHECKS / 'GPU_RESOURCE_GUARD_V2_ROOT_REVIEW.json'
if guard_review.exists():
    reviewed_guard = read(guard_review)
    assert reviewed_guard['status'] == 'ROOT_RESOURCE_GUARD_V2_DIFF_AND_NO_CNN_REVIEW_PASS_PREPARED'
    guard_dir = ROOT / 'tmp/celeba_valid_gpu_resource_gate_fix_20261009'
    assert sha(guard_dir / 'release/PACKAGE_SHA256.json') == reviewed_guard['runtime_seal_sha256']
    state['baseline_valid_GPU_resource_guard_v2_20261009'] = dict(
        status='SOURCE_REVIEWED_NO_CNN_GUARDS_PASS_NOT_DEPLOYED',
        runtime_seal_sha256=reviewed_guard['runtime_seal_sha256'],
        root_review_sha256=sha(guard_review), remaining=440, accepted_source=460,
        main_active_guard_range=[1,8], scientific_concurrency_unchanged=8, native_tolerance=1e-12,
        new_CNN_inference=0, deployed=False, Linux_runtime_verified=False,
        entry='tmp/celeba_valid_gpu_resource_gate_fix_20261009/README.md')
    gpu_recovery_paragraph += ('独立V2工程修复已通过root源码差异审阅、60个健康组合、11个资源拒收边界及两份实际快照schema核验，'
        '仅资源预检与输入保全变化，原科学body/strict及24个成员字节不变；尚未上机，新440补集需独立命名空间。')
v2_execution = ROOT / 'tmp/celeba_valid_gpu_remaining440_resource_gate_v2_execution_20261009'
v2_observations = list(v2_execution.glob('live_*.ROOT.json'))
if v2_observations:
    v2_checked_path = max(v2_observations)
    v2_checked = read(v2_checked_path)
    v2_live_path = v2_checked_path.with_name(v2_checked_path.name.replace('.ROOT.json', '.json'))
    v2_live = read(v2_live_path)
    assert v2_checked['status'] == 'ROOT_LINUX_SPAWN_AND_V2_RESOURCE_RECEIPTS_PASS'
    assert sha(v2_live_path) == v2_checked['sha256']
    assert v2_live['runtime_package_sha256'] == reviewed_guard['runtime_seal_sha256']
    assert v2_live['source_package_sha256'] == '355e22697f213b4b4b9f2509cf5e12000009be0d6e836a5c5cceb1f913906a93'
    fixed_launch_path = v2_execution / 'ROOT_PRECONTRACT_CONFIG_FIX_AND_LAUNCH.json'
    fixed_launch = read(fixed_launch_path)
    assert fixed_launch['before']['output_absent'] and not fixed_launch['before']['CNN_started']
    assert all(row['returncode'] == 0 for row in fixed_launch['commands'])
    v2_running = 'RUNNING' in v2_live['service'] and not v2_live['queue_failure']
    v2_service = 'guardfed_celeba_valid_gpu_remaining440_resource_gate_v2_20261009'
    state['baseline_valid_GPU_remaining440_v2_20261009'] = dict(
        status='RUNNING_ACTUAL_LINUX_SPAWN_VERIFIED' if v2_running else 'STOPPED_READ_LATEST_EVIDENCE',
        service=v2_service, queue_running=v2_running, queue_size=440,
        output_parent='/workspace/guardfed_checks/celeba_valid_gpu_recovery_execution_20261009/remaining440_resource_gate_v2_attempt1',
        queue_package_sha256=v2_live['source_package_sha256'], runtime_package_sha256=v2_live['runtime_package_sha256'],
        root_review_sha256=v2_live['review_sha256'], actual_config_sha256=v2_live['config_sha256'],
        launch_proof_path=fixed_launch_path.relative_to(ROOT).as_posix(), launch_proof_sha256=sha(fixed_launch_path),
        measured_utc=v2_live['checked_utc'], live_path=v2_live_path.relative_to(ROOT).as_posix(), live_sha256=sha(v2_live_path),
        spawn_verification_path=v2_checked_path.relative_to(ROOT).as_posix(), spawn_verification_sha256=sha(v2_checked_path),
        worker_exit_complete_observed=v2_live['worker_complete_exit_only'], remote_closed_n=v2_live['remote_closed_n'],
        new_offserver_accepted=0, accepted_prior=460, accepted_prior_sha256=sha(partial_path),
        exact_inventory900_minus460=True, original464_not_restarted=True, scientific_changes=False,
        precontract_config_mismatch_preserved=True, precontract_failed_CNN_started=False,
        CPU_coordinator=106, CPU_worker=105, nice=10, max_GPU_workers=1, native_tolerance=1e-12)
    state['baseline_valid_GPU_resource_guard_v2_20261009'].update(
        status='DEPLOYED_LINUX_SPAWN_AND_RESOURCE_RECEIPTS_VERIFIED', deployed=True, Linux_runtime_verified=True,
        new_CNN_inference=None, worker_exit_complete_observed=v2_live['worker_complete_exit_only'], new_offserver_accepted=0)
    state['final_evaluator_runtime_20261009']['status'] = '460_ACCEPTED_REMAINING440_GPU_V2_RUNNING' if v2_running else '460_ACCEPTED_READ_V2_STOP_EVIDENCE'
    if v2_running:
        state['active_services'] = list(dict.fromkeys(state['active_services'] + [v2_service]))
    gpu_recovery_paragraph = gpu_recovery_paragraph.replace(
        '尚未上机，新440补集需独立命名空间。',
        f'新440精确补集已在独立目录启动，{v2_live["checked_utc"]}实际CPU106协调、CPU105单GPU/单线程worker、nice10/idle及资源凭据通过；观测{v2_live["worker_complete_exit_only"]}条推理正常退出、远端闭合{v2_live["remote_closed_n"]}条，新增离机接受0，累计仍为460。初次supervisor审批文件名不一致导致contract前退出、未创建输出或推理；原日志与配置保存后仅修正包外路径，现用配置SHA71826d102a628eb9fa0869dfa9f360a71527f2c595d8467d7464f6b720f429f5。')
v2_evidence_dir = ROOT / 'tmp/celeba_valid_gpu_remaining440_v2_evidence_20261009'
v2_collectors = list(v2_evidence_dir.glob('chunk_*/cumulative_*_accepted.json'))
if v2_collectors:
    assert sha(v2_evidence_dir / 'PACKAGE_SHA256.json') == 'e389767b0d7846509226f2669b34e5a87ad4e407983d85cbcdde1c3edf62a5a8'
    v2_spec = importlib.util.spec_from_file_location('closed_gpu440_v2_evidence', v2_evidence_dir / 'evidence.py')
    v2_tool = importlib.util.module_from_spec(v2_spec); v2_spec.loader.exec_module(v2_tool)
    v2_current_path = max(v2_collectors, key=lambda p: read(p)['accepted_n'])
    v2_current = read(v2_current_path); v2_context = v2_tool.context()
    v2_tool.prior_chain(v2_current_path, sha(v2_current_path), v2_context, v2_current['last_chunk_index']+1)
    v2_proof_path = Path(v2_current['new_proof_path']); v2_proof = read(v2_proof_path)
    assert sha(v2_proof_path)==v2_current['new_proof_sha256'] and v2_proof['native_max_abs_difference']<=1e-12
    assert sha(v2_proof_path.parent/'chunk_evidence.tar.gz')==v2_proof['archive_sha256']
    assert v2_proof['new_CNN_inference']==0 and not v2_proof['final_test']
    v2_n = v2_current['accepted_n']
    state['final_evaluator_runtime_20261009'].update(
        status=('900_VALID_THREE_VIEWS_STRICT_OFFSERVER_COMPLETE' if v2_n == 900 else f'{v2_n}_ACCEPTED_READ_LATEST_V2_RUNTIME'),
        actual_native_valid_image_replays_accepted=v2_n,actual_native_valid_image_replays_remaining=900-v2_n,
        accepted_collection_path=v2_current_path.relative_to(ROOT).as_posix(), accepted_collection_sha256=sha(v2_current_path),
        cumulative_unique_checkpoint_acceptance=v2_current_path.relative_to(ROOT).as_posix(),
        cumulative_unique_checkpoint_acceptance_sha256=sha(v2_current_path))
    state['baseline_valid_GPU_remaining440_v2_20261009'].update(
        new_offserver_accepted=v2_n-460,current_accepted=v2_n,remaining_unaccepted=900-v2_n,
        collector_path=v2_current_path.relative_to(ROOT).as_posix(),collector_sha256=sha(v2_current_path),
        latest_offserver_proof_sha256=sha(v2_proof_path),CPU_provenance_n=434,GPU_provenance_n=v2_n-434)
    state['baseline_valid_GPU_resource_guard_v2_20261009']['new_offserver_accepted']=v2_n-460
    state['baseline_valid_recovery_prepared_20261009']['current_missing']=900-v2_n
    if v2_n == 900:
        assert v2_live['remote_closed_n'] == v2_live['worker_complete_exit_only'] == 440
        assert 'EXITED' in v2_live['service'] and not v2_live['queue_failure']
        state['baseline_valid_GPU_remaining440_v2_20261009']['status'] = 'COMPLETE_STRICT_OFFSERVER_440_NEW_PLUS460_PRIOR'
        state['active_services'] = [s for s in state['active_services'] if s != v2_service]
    gpu_recovery_paragraph += (f'V2队列随后已有{v2_n-460}条通过原严格验收、全部archive/member SHA和独立保存数组复核，'
        f'累计{v2_n}/900（CPU434/GPU{v2_n-434}），仍缺{900-v2_n}；原460账本不改，离机验收新增CNN推理0。')
population_proof = CHECKS / 'LOGOFAIR_POPULATION_PROPOSAL_ROOT_VERIFICATION.json'
fl_observation = ROOT / 'tmp/celeba_flgmm_screen_20261009_v2_dispatch/observations/bounded_review_20261009T130206Z/STATUS.json'
if fl_observation.exists():
    observed_fl = read(fl_observation)
    assert sha(fl_observation) == '4d56e805779a707b26d4aa40877576dcb5784026eabb802d3c10b35e5ae6bb75'
    assert observed_fl['accepted_after'] == state['flgmm_screen32_20261009']['offserver_accepted70round_jobs'] == 6
    assert observed_fl['new_strict_offserver_accepted'] == 0 and not observed_fl['failure_paths']
    state['flgmm_screen32_20261009']['latest_readonly_terminal_observation'] = dict(
        checked_utc=observed_fl['snapshot_utc'], observed_complete=11,
        active=observed_fl['active'], pending=19, failures=0,
        new_terminal_candidates_not_strict_or_offserver=5,
        entry=fl_observation.relative_to(ROOT).as_posix(), sha256=sha(fl_observation))
fl_delta = ROOT / 'tmp/celeba_flgmm_screen_20261009_v2_dispatch/accepted_delta_after6_20261009'
if (fl_delta/'ROOT_ADOPTION_REVIEW.json').exists():
    adopted = read(fl_delta/'ROOT_ADOPTION_REVIEW.json'); link = read(fl_delta/'ROOT_READY_CHAIN_LINK.json')
    assert adopted['status']=='ROOT_FLGMM_NEW7_LINK_ARCHIVE_MEMBER_AND_ORIGINAL_ACCEPTOR_REVIEW_PASS'
    assert adopted['reviewed_link_sha256']==sha(fl_delta/'ROOT_READY_CHAIN_LINK.json')
    assert adopted['offserver_proof_sha256']==sha(fl_delta/'OFFSERVER_ACCEPTANCE.json')
    assert adopted['archive_sha256']==sha(fl_delta/'accepted_delta_after6.tar.gz') and adopted['accepted_total']==13
    fl_chain = fl_delta.parent/'BACKUP_CHAIN_increment_after6_20261009.json'
    assert read(fl_chain)['root_adoption_sha256']==sha(fl_delta/'ROOT_ADOPTION_REVIEW.json')
    state['flgmm_screen32_20261009'].update(
        accepted70round_jobs=13,offserver_accepted70round_jobs=13,not_yet_accepted=19,
        accepted_ids=link['accepted_job_ids'],latest_chain_path=fl_chain.relative_to(ROOT).as_posix(),latest_chain_sha256=sha(fl_chain),
        latest_root_adoption_sha256=sha(fl_delta/'ROOT_ADOPTION_REVIEW.json'),latest_archive_sha256=adopted['archive_sha256'],
        latest_readonly_terminal_observation=dict(checked_utc=link['snapshot_utc'],observed_complete=13,active=link['snapshot_active'],pending=17,failures=0))
next37 = ROOT/'tmp/celeba_mechanism_valid_incremental_next37_20261009'
if (next37/'root_source_review/ROOT_REVIEW.json').exists():
    reviewed37 = read(next37/'root_source_review/ROOT_REVIEW.json')
    assert reviewed37['status']=='ROOT_NEXT37_SOURCE_SCOPE_AND_NO_CNN_REJECTIONS_PASS_NOT_RUNTIME_APPROVAL'
    assert reviewed37['source_seal_sha256']==sha(next37/'FILES_SHA256.json')
    assert reviewed37['selected_new_three_view']==37 and reviewed37['closed_three_view']==23
    state['celeba_mechanism_v1']['next37_valid_replay'] = dict(
        status='SOURCE_REVIEWED_RUNTIME_NOT_STARTED',selected_terminal_models=37,previous23_excluded=True,
        actual_native_accepted_snapshot=60,source_seal_sha256=reviewed37['source_seal_sha256'],
        root_source_review_sha256=sha(next37/'root_source_review/ROOT_REVIEW.json'),
        inventory_sha256=reviewed37['inventory_sha256'],bridge_sha256=reviewed37['bridge_sha256'],
        source_scientific_functions_unchanged=True,native_tolerance=1e-12,new_training=0,new_Full_inference=0,
        final_test_dispatch=False,offserver_new_accepted=0,
        entry=next37.relative_to(ROOT).as_posix())
    next37_execution = next37/'execution_candidate'
    if (next37_execution/'ROOT_STARTUP_OBSERVATION.json').exists():
        started37 = read(next37_execution/'ROOT_STARTUP_OBSERVATION.json')
        deployed37 = read(next37_execution/'deployment_receipt.json')
        assert started37['status']=='ROOT_NEXT37_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS'
        assert started37['deployment_receipt_sha256']==sha(next37_execution/'deployment_receipt.json')
        assert started37['execution_seal_sha256']==sha(next37_execution/'EXECUTION_SOURCE_SHA256.json')
        assert deployed37['remote_installation']['returncode']==0 and not started37['batch_failure']
        assert started37['original23_not_rerun'] and started37['scientific_offserver_new_accepted']==0
        state['celeba_mechanism_v1']['next37_valid_replay'].update(
            status='RUNNING_AT_SOURCE_BOUND_LINUX_STARTUP_OBSERVATION' if 'RUNNING' in started37['service'] else 'REMOTE_COMPLETE_OFFSERVER_PENDING',
            service='guardfed_celeba_mechanism_valid_next37',observed_service=started37['service'],measured_utc=started37['utc'],
            actual_processes=len(started37['processes']),CPU_threads=8,CPUs=list(range(112,120)),nice=10,io_priority='idle',
            startup_root_proof_sha256=sha(next37_execution/'ROOT_STARTUP_OBSERVATION.json'),
            execution_seal_sha256=started37['execution_seal_sha256'],root_execution_approval_sha256=deployed37['root_approval_sha256'],
            actual_approval_sha256=started37['files']['APPROVED.json']['sha256'],
            deployment_receipt_sha256=sha(next37_execution/'deployment_receipt.json'),
            remote_terminal_candidates=len(started37['completed']),offserver_new_accepted=0)
        if 'RUNNING' in started37['service']:
            state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_mechanism_valid_next37']))
    progress37_paths = [p for p in next37_execution.glob('ROOT_PROGRESS_*.json') if not p.name.endswith('.RAW.json')]
    if progress37_paths:
        progress37_path = max(progress37_paths, key=lambda p: read(p)['utc'])
        progress37 = read(progress37_path)
        assert progress37['status'] == 'ROOT_NEXT37_REAL_LINUX_PROGRESS_AND_ALLOCATION_PASS'
        assert progress37['source_startup_proof_sha256'] == sha(next37_execution/'ROOT_STARTUP_OBSERVATION.json')
        assert progress37['execution_seal_sha256'] == sha(next37_execution/'EXECUTION_SOURCE_SHA256.json')
        assert progress37['offserver_acceptance_not_measured'] and not progress37['batch_failure']
        state['celeba_mechanism_v1']['next37_valid_replay'].update(
            latest_progress_utc=progress37['utc'],latest_progress_sha256=sha(progress37_path),
            latest_progress_path=progress37_path.relative_to(ROOT).as_posix(),
            observed_service=progress37['service'],actual_processes=len(progress37['processes']),
            remote_terminal_candidates=len(progress37['completed']))
        if progress37['batch_complete']:
            assert len(progress37['completed']) == 37 and not progress37['processes']
            assert {r['id'] for r in progress37['completed']} == set(read(next37/'SCOPE.json')['selected_ids'])
            state['celeba_mechanism_v1']['next37_valid_replay']['status'] = 'REMOTE_COMPLETE_OFFSERVER_PENDING'
            state['active_services'] = [s for s in state['active_services'] if s != 'guardfed_celeba_mechanism_valid_next37']
    prior23 = state['celeba_mechanism_v1']['three_view_accepted_ids']
    assert len(prior23) == len(set(prior23)) == 23
    allowed37 = {row['id'] for row in read(next37/'inventory_actual60_Full100refs.json')['records']} - set(prior23)
    assert len(allowed37) == 37
    adopted37_ids, adopted37_backups = [], []
    for adopted_path in sorted((next37_execution/'backups').glob('*/ROOT_ADOPTION_REVIEW.json')):
        adopted37 = read(adopted_path); delta37 = adopted_path.parent
        proof37_path = delta37/'OFFSERVER_VERIFICATION.json'
        receipt37_path = delta37/'backup_receipt.json'
        proof37, receipt37 = read(proof37_path), read(receipt37_path)
        assert adopted37['status'] == 'ROOT_NEXT37_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
        assert adopted37['execution_seal_sha256'] == sha(next37_execution/'EXECUTION_SOURCE_SHA256.json')
        assert adopted37['science_seal_sha256'] == sha(next37/'FILES_SHA256.json')
        assert adopted37['offserver_verification_sha256'] == sha(proof37_path)
        assert adopted37['backup_receipt_sha256'] == sha(receipt37_path)
        assert adopted37['archive_sha256'] == receipt37['archive_sha256'] == proof37['archive_sha256'] == sha(delta37/'incremental_valid_three_views.tar.gz')
        assert proof37['status'] == 'INCREMENTAL_INDEPENDENT_SAVED_ARRAYS_THREE_VIEWS_PASS'
        ids37 = adopted37['accepted_new_ids']
        assert ids37 == proof37['accepted_new_ids'] == receipt37['accepted_new_ids']
        assert len(ids37) == adopted37['accepted_new'] == len(set(ids37))
        assert set(ids37) <= allowed37 and not set(ids37).intersection(adopted37_ids)
        assert adopted37['prior_three_view_models'] == 23 + len(adopted37_ids)
        assert adopted37['cumulative_three_view_models'] == 23 + len(adopted37_ids) + len(ids37)
        assert adopted37['all_native_differences_zero'] and adopted37['original23_unchanged']
        assert adopted37['new_training'] == adopted37['new_Full_inference'] == 0 and not adopted37['test_inference']
        adopted37_ids.extend(ids37)
        adopted37_backups.append(dict(new_ids=ids37,checked_utc=adopted37['checked_utc'],
            root_proof_path=adopted_path.relative_to(ROOT).as_posix(),root_proof_sha256=sha(adopted_path),
            archive_sha256=adopted37['archive_sha256'],receipt_sha256=sha(receipt37_path),offserver_proof_sha256=sha(proof37_path)))
    if adopted37_ids:
        total37 = prior23 + adopted37_ids
        state['celeba_mechanism_v1'].update(three_view_new_models_accepted=len(total37),
            three_view_new_models_offserver_verified=len(total37),three_view_accepted_ids=total37,
            three_view_scope_limit='Original23 plus explicitly adopted actual terminals from the exact37 frozen scope; no Full reinference or test')
        state['celeba_mechanism_v1']['next37_valid_replay'].update(offserver_new_accepted=len(adopted37_ids),
            offserver_remaining=37-len(adopted37_ids),accepted_ids=adopted37_ids,incremental_backups=adopted37_backups,
            latest_acceptance_utc=adopted37_backups[-1]['checked_utc'],startup_snapshot_is_historical=True)
        if len(adopted37_ids) == 37:
            assert state['celeba_mechanism_v1']['next37_valid_replay']['status'] == 'REMOTE_COMPLETE_OFFSERVER_PENDING'
            state['celeba_mechanism_v1']['next37_valid_replay']['status'] = 'COMPLETE_STRICT_OFFSERVER_NO_FULL_JOIN'
hybrid_delta = ROOT/'tmp/celeba_hybrid_screen_execution_20261009/accepted_delta_first_20261009'
hybrid_chain_path = hybrid_delta.parent/'BACKUP_CHAIN_first4_20261009.json'
if hybrid_chain_path.exists():
    hybrid_chain = read(hybrid_chain_path); hybrid_proof = read(hybrid_delta/'ROOT_RECORD_REVIEW.json')
    hybrid_ready = read(hybrid_delta/'ROOT_READY_DELIVERY.json')
    assert hybrid_chain['status']=='PARTIAL_STRICT_OFFSERVER_ROOT_RECORD_REVIEW_ADOPTED'
    assert hybrid_chain['root_record_review_sha256']==sha(hybrid_delta/'ROOT_RECORD_REVIEW.json')
    assert hybrid_chain['archive_sha256']==sha(hybrid_delta/'hybrid_first_delta.tar.gz')
    assert hybrid_proof['accepted_new']==hybrid_chain['accepted_total']==4
    assert hybrid_proof['scientific_checks_and_null_policy_unchanged'] and hybrid_proof['local_CNN_inference']==0
    state['hybrid_screen32_20261009'].update(
        scientific_70round_results_strict_accepted=4,offserver_accepted70round_jobs=4,not_yet_accepted=28,
        accepted_ids=hybrid_chain['accepted_job_ids'],latest_chain_path=hybrid_chain_path.relative_to(ROOT).as_posix(),
        latest_chain_sha256=sha(hybrid_chain_path),latest_root_review_sha256=sha(hybrid_delta/'ROOT_RECORD_REVIEW.json'),
        local_runtime_not_claimed_equal=True,source_scientific_checks_unchanged=True,
        latest_terminal_observation=dict(checked_utc=hybrid_ready['snapshot_utc'],observed_complete=4,
            active=hybrid_ready['active'],pending=hybrid_ready['pending'],failures=len(hybrid_ready['failures'])),
        selected_recipe=None,formal100_started=False,test_started=False)
if population_proof.exists():
    proposal = ROOT / 'tmp/celeba_logofair_population_proposal_20261009'
    checked = read(population_proof)
    assert checked['status'] == 'ROOT_POPULATION_PROPOSAL_IDENTITY_SUPPORT_PASS_NOT_APPROVED'
    assert sha(proposal / 'FILES_SHA256.json') == checked['seal_sha256']
    assert sha(proposal / 'mapping.npz') == checked['mapping_sha256']
    state['logofair_population_proposal_20261009'] = dict(
        status='PREPARED_NOT_APPROVED_USER_POPULATION_CHOICE_PENDING',
        conditions=4, cohorts=20, semantics='virtual image-ID hash cohorts, not true training clients',
        root=16277, valid=19867, minimum_root_label_sensitive_cell=116,
        root_proof_sha256=sha(population_proof), source_seal_sha256=checked['seal_sha256'],
        mapping_sha256=checked['mapping_sha256'], metadata_sha256=checked['metadata_sha256'],
        valid_labels_or_scores_decoded=False, CNN_or_Beta_or_performance=False,
        execution_started=False, formal_approval=False,
        entry='tmp/celeba_logofair_population_proposal_20261009/REPORT.md')
interim_path = max((TRAIN / 'celeba_mechanism_v1').glob('interim_tables_*/tables.json'))
native92_dir = TRAIN / 'celeba_mechanism_v1/native_interim92_20261009'
if (native92_dir / 'ROOT_REVIEW.json').exists():
    native92 = read(native92_dir / 'ROOT_REVIEW.json')
    assert native92['status'] == 'ROOT_NATIVE92_NINE_SCENE_GRID_AND_INDEPENDENT_STATISTICS_PASS'
    assert native92['source_seal_sha256'] == sha(native92_dir / 'FILES_SHA256.json')
    assert native92['accepted_native92'] == 92 and native92['complete_scenes'] == 9 and native92['scalar_checks'] == 486
    for name, pin in read(native92_dir / 'FILES_SHA256.json')['files'].items():
        assert sha(native92_dir / name) == pin['sha256']
    if len(read(native92_dir / 'tables.json')['accepted_new_ids']) > len(read(interim_path)['accepted_new_ids']):
        interim_path = native92_dir / 'tables.json'
    state['celeba_mechanism_v1']['native92_table_root_review'] = dict(
        entry=(native92_dir / 'TABLES.md').relative_to(TRAIN).as_posix(),
        root_proof_sha256=sha(native92_dir / 'ROOT_REVIEW.json'), complete_native_scenes=9,
        scalar_checks=486, three_view_nine_scene_claim=False, new_inference=0, test=False)
native100_dir = TRAIN/'celeba_mechanism_v1/native_interim100_20261009'
if (native100_dir/'ROOT_REVIEW.json').exists():
    native100 = read(native100_dir/'ROOT_REVIEW.json')
    assert sha(native100_dir/'ROOT_REVIEW.json') == '22972d2bb879680d334324c6796cdd7c044c85af98153a3262f1b8c9a494fa4a'
    assert native100['source_seal_sha256'] == sha(native100_dir/'FILES_SHA256.json')
    assert native100['accepted_native100'] == 100 and native100['complete_scenes'] == 10 and native100['scalar_checks'] == 540
    for name,pin in read(native100_dir/'FILES_SHA256.json')['files'].items():
        assert sha(native100_dir/name) == pin['sha256']
    interim_path = native100_dir/'rendered/tables.json'
    state['celeba_mechanism_v1']['native100_table_root_review'] = dict(
        entry=(native100_dir/'rendered/TABLES.md').relative_to(TRAIN).as_posix(),
        root_proof_sha256=sha(native100_dir/'ROOT_REVIEW.json'),complete_native_scenes=10,
        Full100_paired=True,minus_U_complete=100,minus_C_partial=4,scalar_checks=540,
        accepted_native_controls_at_table_snapshot=104,three_view100_claim=False,new_inference=0,test=False)
interim = read(interim_path)
assert interim['status'] == 'INTERIM_COMPLETE_SCENES_ONLY_NO_NEW_INFERENCE'
assert interim['new_training'] == interim['new_inference'] == 0 and not interim['test_used']
state['celeba_mechanism_v1']['latest_interim_paper_table'] = dict(
    table_path=interim_path.with_name('TABLES.md').relative_to(TRAIN).as_posix(),
    table_sha256=sha(interim_path.with_name('TABLES.md')), statistics_sha256=sha(interim_path),
    complete_paired_scenes=interim['complete_paired_scenes'],
    identity='native Full versus minus_U only; other components incomplete',
    mean_sampleSD=True, additional_9_and_6_seed_panels=True, whole_comparison_complete=False)
paired_root_path = TRAIN/'celeba_mechanism_v1/three_view_interim_20261009T145900Z/ROOT_REVIEW.json'
paired_note = '完整Full配对三视图仍需实际来源连接，不从Full引用数量推断完成。'
if paired_root_path.exists():
    paired_root = read(paired_root_path); paired_dir = paired_root_path.parent
    assert paired_root['status'] == 'ROOT_SIX_SCENE_THREE_VIEW_PAIRED_SAVED_RECEIPTS_AND_STATISTICS_PASS'
    assert paired_root['paired_checkpoints'] == 60 and paired_root['new_inference'] == 0 and not paired_root['final_test']
    for name, expected in paired_root['artifact_sha256'].items():
        assert sha(paired_dir/name) == expected
    state['celeba_mechanism_v1']['latest_paired_three_view_table'] = dict(
        status='SIX_SCENES_STRICT_OFFSERVER_FULL_JOIN_AND_ROOT_STATISTICS_ACCEPTED',
        complete_scenes=6, paired_checkpoints=60, Full_identity_available=100,
        Full_replay_devices={'cpu':5,'gpu':95}, paired_Full_replay_devices={'cpu':5,'gpu':55},
        minus_U_replay_device='cpu', native_shared_metrics_exact=True,
        table_path=(paired_dir/'TABLES.md').relative_to(TRAIN).as_posix(),
        root_proof_path=paired_root_path.relative_to(ROOT).as_posix(), root_proof_sha256=sha(paired_root_path),
        main_endpoint_selected=False, uniform_device_comparison=False, final_test=False, whole_mechanism_complete=False)
    state['celeba_mechanism_v1']['next37_valid_replay'].update(
        status='COMPLETE_STRICT_OFFSERVER_PAIRED_SIX_SCENES', paired_table_root_proof_sha256=sha(paired_root_path))
    paired_note = ('六个完整场景、60对checkpoint已连接实际Full三视图并独立核验：'
        'raw/native/shared并列表、10/9/6同种子面板及配对差值均保留。native与shared在这120记录的三指标完全相同；'
        '去U后native准确率六场景均降低0.309–1.384个百分点，ASPD均更低，AEOD五场景更高；'
        'raw下AEOD五场景更低。结果表明准确率与差距取舍，不能宣称U在所有指标上不可或缺。'
        '表入口celeba_mechanism_v1/three_view_interim_20261009T145900Z/TABLES.md；混合设备/训练环境与验证集选择历史已披露，主评价口径仍待决定。')
next11 = ROOT/'tmp/celeba_mechanism_valid_incremental_next11_20261009'
if (next11/'root_source_review/ROOT_REVIEW.json').exists():
    reviewed11 = read(next11/'root_source_review/ROOT_REVIEW.json')
    assert reviewed11['status'] == 'ROOT_NEXT11_SOURCE_SCOPE_AND_NO_CNN_REJECTIONS_PASS_NOT_RUNTIME_APPROVAL'
    assert reviewed11['source_seal_sha256'] == sha(next11/'FILES_SHA256.json')
    assert reviewed11['actual_native_accepted']==71 and reviewed11['closed_three_view']==60 and reviewed11['selected_new_three_view']==11
    state['celeba_mechanism_v1']['next11_valid_replay'] = dict(status='SOURCE_REVIEWED_NOT_DISPATCHED',
        actual_native_snapshot=71, excluded_closed=60, selected_new=11, pending_native=729,
        source_seal_sha256=reviewed11['source_seal_sha256'], root_source_proof_sha256=sha(next11/'root_source_review/ROOT_REVIEW.json'),
        scientific_functions_unchanged=True, scientific_no_CNN_refusals=42, execution_started=False,
        entry=next11.relative_to(ROOT).as_posix())
    started11_path = next11/'execution_candidate/ROOT_STARTUP_OBSERVATION.json'
    if started11_path.exists():
        started11 = read(started11_path); execution11 = started11_path.parent
        deployment11 = read(execution11/'deployment_receipt.json')
        assert started11['status'] == 'ROOT_NEXT11_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS'
        assert started11['deployment_receipt_sha256'] == sha(execution11/'deployment_receipt.json')
        assert started11['execution_seal_sha256'] == sha(execution11/'EXECUTION_SOURCE_SHA256.json')
        assert deployment11['remote_installation']['returncode']==0 and started11['original60_not_rerun']
        assert started11['scientific_offserver_new_accepted']==0 and not started11['test_inference']
        state['celeba_mechanism_v1']['next11_valid_replay'].update(
            status='RUNNING_STARTUP_ACCEPTED_NO_NEW_OFFSERVER_RESULTS',execution_started=True,
            service='guardfed_celeba_mechanism_valid_next11',observed_service=started11['service'],measured_utc=started11['utc'],
            startup_root_proof_sha256=sha(started11_path),execution_seal_sha256=started11['execution_seal_sha256'],
            deployment_receipt_sha256=sha(execution11/'deployment_receipt.json'),actual_processes=len(started11['processes']),
            source_data_terminal_members_checked=109,allowed_cpus=list(range(112,120)),compute_threads=8,
            new_training=0,new_Full_inference=0,offserver_new_accepted=0)
        if 'RUNNING' in started11['service']:
            state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_mechanism_valid_next11']))
        progress11_paths = [p for p in execution11.glob('ROOT_PROGRESS_*.json') if not p.name.endswith('.RAW.json')]
        if progress11_paths:
            progress11_path = max(progress11_paths, key=lambda p: read(p)['utc'])
            progress11 = read(progress11_path)
            assert progress11['status'] == 'ROOT_NEXT11_REAL_LINUX_PROGRESS_AND_ALLOCATION_PASS'
            assert progress11['source_startup_proof_sha256'] == sha(started11_path)
            assert progress11['execution_seal_sha256'] == sha(execution11/'EXECUTION_SOURCE_SHA256.json')
            assert not progress11['batch_failure'] and progress11['original60_not_rerun']
            state['celeba_mechanism_v1']['next11_valid_replay'].update(
                latest_progress_utc=progress11['utc'],latest_progress_path=progress11_path.relative_to(ROOT).as_posix(),
                latest_progress_sha256=sha(progress11_path),observed_service=progress11['service'],
                actual_processes=len(progress11['processes']),remote_terminal_candidates=len(progress11['completed']))
            if progress11['batch_complete']:
                assert not progress11['processes'] and len(progress11['completed']) == 11
                assert {r['id'] for r in progress11['completed']} == set(read(next11/'SCOPE.json')['selected_ids'])
                state['celeba_mechanism_v1']['next11_valid_replay']['status'] = 'REMOTE_COMPLETE_OFFSERVER_PENDING'
                state['active_services'] = [s for s in state['active_services'] if s != 'guardfed_celeba_mechanism_valid_next11']
        adopted11_paths = list((execution11/'backups').glob('*/ROOT_ADOPTION_REVIEW.json'))
        if adopted11_paths:
            assert len(adopted11_paths) == 1
            adopted11_path = adopted11_paths[0]; delta11 = adopted11_path.parent
            adopted11 = read(adopted11_path); proof11_path = delta11/'OFFSERVER_VERIFICATION.json'
            receipt11_path = delta11/'backup_receipt.json'; proof11 = read(proof11_path); receipt11 = read(receipt11_path)
            assert adopted11['status'] == 'ROOT_NEXT11_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
            assert adopted11['science_seal_sha256'] == sha(next11/'FILES_SHA256.json')
            assert adopted11['execution_seal_sha256'] == sha(execution11/'EXECUTION_SOURCE_SHA256.json')
            assert adopted11['offserver_verification_sha256'] == sha(proof11_path)
            assert adopted11['backup_receipt_sha256'] == sha(receipt11_path)
            assert adopted11['archive_sha256'] == proof11['archive_sha256'] == receipt11['archive_sha256'] == sha(delta11/'incremental_valid_three_views.tar.gz')
            assert adopted11['remote_terminal_proof_sha256'] == sha(progress11_path)
            ids11 = adopted11['accepted_new_ids']
            assert ids11 == proof11['accepted_new_ids'] == receipt11['accepted_new_ids'] == read(next11/'SCOPE.json')['selected_ids']
            prior60 = state['celeba_mechanism_v1']['three_view_accepted_ids']
            assert len(prior60) == len(set(prior60)) == adopted11['prior_three_view_models'] == 60
            assert len(ids11) == adopted11['accepted_new'] == 11 and not set(ids11).intersection(prior60)
            assert adopted11['cumulative_three_view_models'] == 71 and adopted11['all_native_differences_zero']
            assert adopted11['original60_unchanged'] and adopted11['new_training'] == adopted11['new_Full_inference'] == 0 and not adopted11['test_inference']
            state['celeba_mechanism_v1'].update(three_view_new_models_accepted=71,three_view_new_models_offserver_verified=71,
                three_view_accepted_ids=prior60+ids11,three_view_scope_limit='Original60 plus explicitly root-adopted next11 terminals; no Full reinference or test')
            state['celeba_mechanism_v1']['next11_valid_replay'].update(status='COMPLETE_STRICT_OFFSERVER_NO_FULL_JOIN',
                offserver_new_accepted=11,offserver_remaining=0,accepted_ids=ids11,
                root_adoption_path=adopted11_path.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(adopted11_path),
                archive_sha256=adopted11['archive_sha256'],receipt_sha256=sha(receipt11_path),offserver_proof_sha256=sha(proof11_path),
                latest_acceptance_utc=adopted11['checked_utc'],startup_snapshot_is_historical=True)
paired71_path = TRAIN/'celeba_mechanism_v1/three_view_interim71_20261009/ROOT_REVIEW.json'
if paired71_path.exists():
    paired71 = read(paired71_path); paired71_dir = paired71_path.parent
    assert paired71['status'] == 'ROOT_SEVEN_SCENE_THREE_VIEW_PAIRED_SAVED_RECEIPTS_AND_STATISTICS_PASS'
    assert paired71['complete_paired_scenes'] == 7 and paired71['paired_checkpoints'] == 70 and paired71['preserved_pairs'] == 71
    assert paired71['table_json_sha256'] == sha(paired71_dir/'tables.json')
    assert paired71['records_sha256'] == sha(paired71_dir/'records.json')
    assert paired71['source_inputs_sha256'] == sha(paired71_dir/'INPUTS_SHA256.json')
    assert paired71['agent_numeric_checks_sha256'] == sha(paired71_dir/'NUMERIC_CHECKS.json')
    assert paired71['new_inference'] == 0 and not paired71['final_test']
    state['celeba_mechanism_v1']['latest_paired_three_view_table'] = dict(
        status='SEVEN_SCENES_STRICT_OFFSERVER_FULL_JOIN_AND_ROOT_STATISTICS_ACCEPTED',
        complete_scenes=7,paired_checkpoints=70,preserved_pairs=71,Full_identity_available=100,
        Full_replay_devices={'cpu':5,'gpu':95},paired_Full_replay_devices={'cpu':5,'gpu':65},
        minus_U_replay_device='cpu',native_shared_metrics_exact=True,root_mean_sampleSD_checks=1134,
        table_path=(paired71_dir/'TABLES.md').relative_to(TRAIN).as_posix(),
        root_proof_path=paired71_path.relative_to(ROOT).as_posix(),root_proof_sha256=sha(paired71_path),
        main_endpoint_selected=False,uniform_device_comparison=False,final_test=False,whole_mechanism_complete=False)
    state['celeba_mechanism_v1']['next11_valid_replay'].update(
        status='COMPLETE_STRICT_OFFSERVER_PAIRED_SEVEN_SCENES',paired_table_root_proof_sha256=sha(paired71_path))
    paired_note = ('七个完整场景、70对checkpoint已连接实际Full三视图并独立核验：'
        'raw/native/shared并列表、10/9/6同种子面板及配对差值均保留，1134均值/SD独立复算通过。'
        '71对个体记录全部保留；non-IID FedSA seed91001仅一对，不入均值表。'
        'native/shared在140展示记录三指标和混淆计数完全相同；去U后native准确率七场景均降低0.309–1.384个百分点，'
        'ASPD均更低，AEOD六场景更高；raw AEOD五场景更低、两场景更高（IID Sp-DFA及non-IID F Flip）。这是取舍而非所有指标上的必要性。'
        '表入口celeba_mechanism_v1/three_view_interim71_20261009/TABLES.md；历史六场景数值完全保留，'
        'Full5CPU/65GPU、69cu128/1cu130与control70CPU/cu128、driver及valid选择限制仍披露，主口径待决定。')
for base_name,state_key,minimum in (
        ('celeba_flgmm_screen_20261009_v2_dispatch','flgmm_screen32_20261009',13),
        ('celeba_hybrid_screen_execution_20261009','hybrid_screen32_20261009',4)):
    base=ROOT/'tmp'/base_name;latest_path=base/'LATEST_BACKUP.json';latest=read(latest_path)
    if latest['accepted']>minimum:
        chain_path=base/latest['chain_file'];chain=read(chain_path)
        assert sha(chain_path)==latest['chain_sha256']
        proof_path=ROOT/chain['root_adoption_path'];proof=read(proof_path)
        assert proof['status']=='ROOT_LOCKED_AUXILIARY_DELTA_ARCHIVE_MEMBER_AND_ORIGINAL_RECORD_ACCEPTANCE_PASS'
        assert sha(proof_path)==chain['root_adoption_sha256'] and proof['accepted_total']==chain['accepted_total']==latest['accepted']
        assert len(set(chain['accepted_job_ids']))==chain['accepted_total'] and not proof['selection_performed'] and not proof['final_test']
        assert sha(base/chain['delta_dir']/chain['archive'])==chain['archive_sha256']==proof['archive_sha256']
        state[state_key].update(offserver_accepted70round_jobs=latest['accepted'],accepted70round_jobs=latest['accepted'],
            not_yet_accepted=32-latest['accepted'],accepted_ids=chain['accepted_job_ids'],
            latest_chain_path=chain_path.relative_to(ROOT).as_posix(),latest_chain_sha256=sha(chain_path),
            latest_root_review_path=proof_path.relative_to(ROOT).as_posix(),latest_root_review_sha256=sha(proof_path),
            latest_new_accepted=proof['accepted_new'],latest_archive_members=proof['members_verified'],
            latest_archive_sha256=proof['archive_sha256'],selected_recipe=None,formal100_started=False,final_test=False)
        if state_key=='flgmm_screen32_20261009' and latest['accepted']==32:
            terminal_path=base/chain['delta_dir']/'AUTHORIZED_SNAPSHOT.json';terminal=read(terminal_path)
            assert sha(terminal_path)==proof['authorized_snapshot_sha256']
            assert terminal['queue']['completed']==32 and not terminal['queue']['active'] and terminal['queue']['pending']==0
            assert 'EXITED' in terminal['service']['stdout'] and not terminal['failure_paths']
            state[state_key].update(status='SCREEN32_COMPLETE_STRICT_OFFSERVER_SUMMARY_PENDING',
                service='guardfed_celeba_flgmm_screen',terminal_service=terminal['service']['stdout'],terminal_workers=0,
                latest_readonly_terminal_observation=dict(checked_utc=terminal['utc'],observed_complete=32,active=0,pending=0,
                    failures=0,source_bound=True,observed_terminal_ids=chain['accepted_job_ids'],active_rounds=[],
                    snapshot_path=terminal_path.relative_to(ROOT).as_posix(),snapshot_sha256=sha(terminal_path)))
            state['active_services']=[name for name in state['active_services'] if name!=state[state_key]['service']]
baseline_table_root = ROOT/'outputs/guardfed_tables/celeba_nine_method_three_view_20261009/ROOT_REVIEW.json'
if baseline_table_root.exists():
    table_proof = read(baseline_table_root)
    assert table_proof['status']=='ROOT_NINE_METHOD900_THREE_VIEW_RECEIPTS_COUNTS_AND_TABLE_STATISTICS_PASS'
    assert table_proof['unique_records']==900 and table_proof['saved_source_receipts_rejoined']==900
    assert not table_proof['final_test'] and table_proof['new_inference']==0
    state['final_evaluator_runtime_20261009']['latest_three_view_tables']=dict(
        status='COMPLETE900_VALIDATION_DESCRIPTIVE_TABLES_ROOT_ACCEPTED',
        entry=baseline_table_root.with_name('README.md').relative_to(ROOT).as_posix(),
        root_proof_path=baseline_table_root.relative_to(ROOT).as_posix(),root_proof_sha256=sha(baseline_table_root),
        views=['raw','native','shared_calibration'],fixed_seed_counts=[10,9,6],main_endpoint_pending=True,
        original_native_metrics_exact=2700,independent_statistics_scalars=4860,
        original_snapshot_SD_lastbit_differences=94,original_snapshot_SD_max_difference=2.7755575615628914e-17,
        final_test=False,full17_complete=False)
integrated_root = ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated71_20261009/ROOT_REVIEW.json'
if integrated_root.exists():
    draft_proof=read(integrated_root)
    assert draft_proof['status']=='ROOT_SOURCE_BOUND_SEVEN_SCENE_COMPLETE_DRAFT_REVIEW_PASS_PENDING_FULL_COHORT'
    assert draft_proof['original_comments_preserved']==24 and not draft_proof['submission_ready']
    state['latest_rebuttal_draft']=dict(entry=integrated_root.with_name('rebuttal_integrated_20261009.md').relative_to(ROOT).as_posix(),
        root_proof_sha256=sha(integrated_root),original_comments_preserved=24,complete_paired_scenes=7,
        manuscript_source_applied=False,submission_ready=False,raw_AEOD_decrease=5,raw_AEOD_increase=2)
integrated_v2=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated71_v2_20261009/ROOT_REVIEW.json'
if integrated_v2.exists():
    v2_proof=read(integrated_v2)
    assert v2_proof['status']=='ROOT_MINIMAL_V2_SEVEN_SCENE_PROSE_CORRECTION_AND_ACCEPTED900_SOURCE_PASS'
    assert v2_proof['original_comments_preserved']==24 and v2_proof['all_allowed_substitutions_reversed_to_exact_originals']
    assert not v2_proof['new_performance_claims'] and not v2_proof['submission_ready']
    state['latest_rebuttal_draft'].update(entry=integrated_v2.with_name('rebuttal_integrated_20261009.md').relative_to(ROOT).as_posix(),
        root_proof_sha256=sha(integrated_v2),source_seal_sha256=sha(integrated_v2.with_name('FILES_SHA256.json')),
        P2_six_to_seven_factual_correction=True,accepted900_three_view_source_extension_added=True,prior_sealed_draft_unchanged=True)
after71=ROOT/'tmp/celeba_mechanism_valid_incremental_after71_20261009'
after71_execution=after71/'execution_candidate'
if (after71_execution/'ROOT_STARTUP_OBSERVATION.json').exists():
    startup71=read(after71_execution/'ROOT_STARTUP_OBSERVATION.json')
    assert startup71['status']=='ROOT_AFTER71_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS'
    assert startup71['deployment_receipt_sha256']==sha(after71_execution/'deployment_receipt.json')
    assert startup71['execution_seal_sha256']==sha(after71_execution/'EXECUTION_SOURCE_SHA256.json')
    assert startup71['original71_not_rerun'] and startup71['scientific_offserver_new_accepted']==0 and not startup71['test_inference']
    scope71=read(after71/'SCOPE.json')
    state['celeba_mechanism_v1']['after71_valid_replay']=dict(status='ACTUAL_STARTUP_PASS_OFFSERVER_PENDING',
        service='guardfed_celeba_mechanism_valid_after71',observed_service=startup71['service'],measured_utc=startup71['utc'],
        actual_native_snapshot=82,excluded_closed=71,selected_new=11,selected_ids=scope71['selected_ids'],
        startup_root_proof_path=(after71_execution/'ROOT_STARTUP_OBSERVATION.json').relative_to(ROOT).as_posix(),
        startup_root_proof_sha256=sha(after71_execution/'ROOT_STARTUP_OBSERVATION.json'),
        execution_seal_sha256=startup71['execution_seal_sha256'],CPU_affinity=list(range(112,120)),compute_threads=8,
        new_Full_inference=0,new_training=0,test_inference=False,offserver_new_accepted=0,original71_not_rerun=True)
    state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_mechanism_valid_after71']))
    transport_review=after71_execution/'ROOT_TRANSPORT_RECOVERY_REVIEW.json'
    if transport_review.exists():
        transport71=read(transport_review)
        assert transport71['status']=='ROOT_WINDOWS_ARGV206_PRE_SSH_FAILURE_AND_EMPTY_REMOTE_BACKUP_CONFIRMED'
        assert not transport71['remote_readonly']['backup_latest_exists'] and not transport71['remote_readonly']['backup_directories']
        state['celeba_mechanism_v1']['after71_valid_replay']['preserved_backup_transport_failure']=dict(
            kind='Windows CreateProcess WinError206 before SSH child creation; no scientific failure',
            old_attempt_sha256=transport71['old_attempt_sha256'],recovery_review_path=transport_review.relative_to(ROOT).as_posix(),
            recovery_review_sha256=sha(transport_review),original_evaluation_outputs_unchanged=True,automatic_retry=False)
    progress71=[p for p in after71_execution.glob('ROOT_PROGRESS_*.json') if not p.name.endswith('.RAW.json')]
    if progress71:
        progress71_path=max(progress71);p71=read(progress71_path)
        assert p71['status']=='ROOT_AFTER71_REAL_LINUX_PROGRESS_AND_ALLOCATION_PASS' and not p71['batch_failure']
        state['celeba_mechanism_v1']['after71_valid_replay'].update(latest_measured_utc=p71['utc'],
            latest_progress_path=progress71_path.relative_to(ROOT).as_posix(),latest_progress_sha256=sha(progress71_path),
            observed_service=p71['service'],remote_terminal_candidates=len(p71['completed']))
        if p71['batch_complete']:
            assert 'EXITED' in p71['service'] and not p71['processes']
            assert {r['id'] for r in p71['completed']}==set(scope71['selected_ids'])
            state['celeba_mechanism_v1']['after71_valid_replay']['status']='REMOTE_COMPLETE_OFFSERVER_PENDING'
            state['active_services']=[s for s in state['active_services'] if s!='guardfed_celeba_mechanism_valid_after71']
    closure71=list((after71_execution/'backups').glob('*/ROOT_ADOPTION_REVIEW.json'))
    if closure71:
        assert len(closure71)==1
        adopted71_path=closure71[0];adopted71=read(adopted71_path);delta71=adopted71_path.parent
        assert adopted71['status']=='ROOT_AFTER71_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
        assert adopted71['prior_three_view_models']==71 and adopted71['accepted_new']==11 and adopted71['cumulative_three_view_models']==82
        assert adopted71['science_seal_sha256']==sha(after71/'FILES_SHA256.json') and adopted71['execution_seal_sha256']==sha(after71_execution/'EXECUTION_SOURCE_SHA256.json')
        assert adopted71['offserver_verification_sha256']==sha(delta71/'OFFSERVER_VERIFICATION.json')
        assert adopted71['backup_receipt_sha256']==sha(delta71/'backup_receipt.json')
        assert adopted71['archive_sha256']==sha(delta71/'incremental_valid_three_views.tar.gz')
        assert adopted71['accepted_new_ids']==scope71['selected_ids'] and adopted71['all_native_differences_zero']
        assert adopted71['original71_unchanged'] and adopted71['new_Full_inference']==0 and not adopted71['test_inference']
        prior71=state['celeba_mechanism_v1']['three_view_accepted_ids']
        assert len(prior71)==len(set(prior71))==71 and set(prior71)==set(scope71['excluded_prior_ids'])
        assert not set(prior71)&set(adopted71['accepted_new_ids'])
        state['celeba_mechanism_v1'].update(three_view_new_models_accepted=82,three_view_new_models_offserver_verified=82,
            three_view_accepted_ids=prior71+adopted71['accepted_new_ids'],
            three_view_scope_limit='Prior71 plus exact after71 eleven root-adopted terminals; Full only joined from existing900, no new Full inference/test')
        state['celeba_mechanism_v1']['after71_valid_replay'].update(status='COMPLETE_STRICT_OFFSERVER_NO_NEW_FULL_INFERENCE',
            offserver_new_accepted=11,archive_members_verified=adopted71['archive_members_verified'],
            root_adoption_path=adopted71_path.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(adopted71_path),
            archive_sha256=adopted71['archive_sha256'])
        state['active_services']=[s for s in state['active_services'] if s!='guardfed_celeba_mechanism_valid_after71']
after82_failure_path=ROOT/'tmp/celeba_mechanism_valid_incremental_after82_20261009/execution_candidate/ROOT_FAILURE_REVIEW.json'
if after82_failure_path.exists():
    failed82=read(after82_failure_path)
    assert failed82['status']=='ROOT_AFTER82_FIRST_WORKER_APPROVAL_CARDINALITY_FAILURE_PRESERVED'
    assert failed82['completed']==failed82['accepted_new']==0 and failed82['output_files']==[]
    assert failed82['failure_before_runtime_dependency_binding_and_CNN'] and failed82['original_service_must_not_restart']
    failed82_ex=after82_failure_path.parent
    assert failed82['failure_raw_sha256']==sha(failed82_ex/'ROOT_FAILURE_DIAGNOSIS_1.RAW.json')
    assert failed82['startup_raw_sha256']==sha(failed82_ex/'ROOT_STARTUP_OBSERVATION.RAW.json')
    state['celeba_mechanism_v1']['after82_failed_attempt']=dict(failed82,
        root_failure_review_path=after82_failure_path.relative_to(ROOT).as_posix(),root_failure_review_sha256=sha(after82_failure_path))
    state['active_services']=[s for s in state['active_services'] if s!='guardfed_celeba_mechanism_valid_after82']
after82_v2=ROOT/'tmp/celeba_mechanism_valid_incremental_after82_v2_20261009'
after82_v2_ex=after82_v2/'execution_candidate'
if (after82_v2_ex/'ROOT_STARTUP_OBSERVATION.json').exists():
    startup82=read(after82_v2_ex/'ROOT_STARTUP_OBSERVATION.json')
    assert startup82['status']=='ROOT_AFTER82_V2_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS'
    assert startup82['deployment_receipt_sha256']==sha(after82_v2_ex/'deployment_receipt.json')
    assert startup82['execution_seal_sha256']==sha(after82_v2_ex/'EXECUTION_SOURCE_SHA256.json')
    assert startup82['original82_not_rerun'] and startup82['scientific_offserver_new_accepted']==0 and not startup82['test_inference']
    scope82=read(after82_v2/'SCOPE.json')
    assert len(scope82['selected_ids'])==10 and len(scope82['excluded_prior_ids'])==82
    state['celeba_mechanism_v1']['after82_v2_valid_replay']=dict(status='ACTUAL_STARTUP_PASS_OFFSERVER_PENDING',
        service='guardfed_celeba_mechanism_valid_after82_v2',observed_service=startup82['service'],measured_utc=startup82['utc'],
        actual_native_snapshot=92,excluded_closed=82,selected_new=10,selected_ids=scope82['selected_ids'],
        startup_root_proof_path=(after82_v2_ex/'ROOT_STARTUP_OBSERVATION.json').relative_to(ROOT).as_posix(),
        startup_root_proof_sha256=sha(after82_v2_ex/'ROOT_STARTUP_OBSERVATION.json'),
        execution_seal_sha256=startup82['execution_seal_sha256'],CPU_affinity=list(range(112,120)),compute_threads=8,
        new_Full_inference=0,new_training=0,test_inference=False,offserver_new_accepted=0,original82_not_rerun=True)
    state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_mechanism_valid_after82_v2']))
    progress82=[p for p in after82_v2_ex.glob('ROOT_PROGRESS_*.json') if not p.name.endswith('.RAW.json')]
    if progress82:
        latest82=max(progress82);observed82=read(latest82)
        assert observed82['status']=='ROOT_AFTER82_V2_REAL_LINUX_PROGRESS_AND_ALLOCATION_PASS' and not observed82['batch_failure']
        state['celeba_mechanism_v1']['after82_v2_valid_replay'].update(latest_measured_utc=observed82['utc'],
            latest_progress_path=latest82.relative_to(ROOT).as_posix(),latest_progress_sha256=sha(latest82),
            observed_service=observed82['service'],remote_terminal_candidates=len(observed82['completed']))
        if observed82['batch_complete']:
            assert 'EXITED' in observed82['service'] and not observed82['processes']
            assert {r['id'] for r in observed82['completed']}==set(scope82['selected_ids'])
            state['celeba_mechanism_v1']['after82_v2_valid_replay']['status']='REMOTE_COMPLETE_OFFSERVER_PENDING'
            state['active_services']=[s for s in state['active_services'] if s!='guardfed_celeba_mechanism_valid_after82_v2']
    closures82=list((after82_v2_ex/'backups').glob('*/ROOT_ADOPTION_REVIEW.json'))
    if closures82:
        assert len(closures82)==1
        adoption82_path=closures82[0];adoption82=read(adoption82_path);delta82=adoption82_path.parent
        assert adoption82['status']=='ROOT_AFTER82_V2_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
        assert (adoption82['prior_three_view_models'],adoption82['accepted_new'],adoption82['cumulative_three_view_models'])==(82,10,92)
        assert adoption82['science_seal_sha256']==sha(after82_v2/'FILES_SHA256.json')
        assert adoption82['execution_seal_sha256']==sha(after82_v2_ex/'EXECUTION_SOURCE_SHA256.json')
        for key,name in [('archive_sha256','incremental_valid_three_views.tar.gz'),('offserver_verification_sha256','OFFSERVER_VERIFICATION.json'),('backup_receipt_sha256','backup_receipt.json')]:
            assert adoption82[key]==sha(delta82/name)
        assert adoption82['accepted_new_ids']==scope82['selected_ids'] and adoption82['all_native_differences_zero']
        assert adoption82['original82_unchanged'] and adoption82['new_Full_inference']==0 and not adoption82['test_inference']
        prior82=state['celeba_mechanism_v1']['three_view_accepted_ids']
        assert len(prior82)==len(set(prior82))==82 and set(prior82)==set(scope82['excluded_prior_ids'])
        assert not set(prior82)&set(adoption82['accepted_new_ids'])
        state['celeba_mechanism_v1'].update(three_view_new_models_accepted=92,three_view_new_models_offserver_verified=92,
            three_view_accepted_ids=prior82+adoption82['accepted_new_ids'],
            three_view_scope_limit='Prior82 plus exact10 from independently repaired after82_v2; original failed after82 retained, Full existing900 only, no test')
        state['celeba_mechanism_v1']['after82_v2_valid_replay'].update(status='COMPLETE_STRICT_OFFSERVER_NO_NEW_FULL_INFERENCE',
            offserver_new_accepted=10,archive_members_verified=adoption82['archive_members_verified'],
            root_adoption_path=adoption82_path.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(adoption82_path),archive_sha256=adoption82['archive_sha256'])
        state['active_services']=[s for s in state['active_services'] if s!='guardfed_celeba_mechanism_valid_after82_v2']
variant_source=ROOT/'tmp/celeba_mechanism_remaining_variants_source_plan_20261009/ROOT_SOURCE_REVIEW.json'
if variant_source.exists():
    variant_proof=read(variant_source)
    assert variant_proof['status']=='ROOT_SOURCE_ONLY_REMAINING_VARIANT_MAPPING_AND_RECIPE_REVIEW_PASS_NOT_DISPATCH'
    state['celeba_mechanism_v1']['remaining_variant_replay_source_plan']=dict(status='PREPARED_NOT_DISPATCHED',
        entry=variant_source.with_name('PLAN.md').relative_to(ROOT).as_posix(),root_source_review_sha256=sha(variant_source),
        paired_planned_recipes_verified=800,actual_remaining_variant_terminals_supplied_at_source_preparation=0,new_inference=0)
auxiliary_paths = [p for p in CHECKS.glob('auxiliary_screens_*.json') if not p.name.endswith('.RAW.json')]
if auxiliary_paths:
    auxiliary_path = max(auxiliary_paths, key=lambda p: read(p)['utc'])
    auxiliary = read(auxiliary_path)
    assert auxiliary['read_only'] and auxiliary['new_acceptances'] == 0
    for row in auxiliary['screens']:
        key = {'flgmm':'flgmm_screen32_20261009','hybrid':'hybrid_screen32_20261009'}[row['name']]
        assert not row['failures']
        expected_seal = ('aec95ceb5e8c9b7aa9cec89e9d70d2700c2f242c29e918792088648d6269bad4' if row['name']=='flgmm'
            else '2c496ae11369465d27ed223f8552321ec8cd424e5fd87e9f2d34e5ea8531e06f')
        assert row['source_seal_sha256'] == expected_seal
        terminal_n = len(row['terminal_candidates']);active_n = len(row['active_or_partial'])
        state[key]['latest_readonly_terminal_observation'] = dict(
            checked_utc=auxiliary['utc'],observed_complete=terminal_n,active=active_n,pending=32-terminal_n-active_n,
            failures=0,source_bound=True,new_acceptances=0,
            observed_terminal_ids=[r['id'] for r in row['terminal_candidates']],
            active_rounds=[r['progress'].get('round') if r['progress'] else None for r in row['active_or_partial']],
            entry=auxiliary_path.relative_to(ROOT).as_posix(),sha256=sha(auxiliary_path))
attribution_dir = ROOT / 'outputs/guardfed_tables/celeba_nine_method_view_attribution_20261009'
attribution_proof = attribution_dir / 'ROOT_REVIEW.json'
if attribution_proof.exists():
    proof = read(attribution_proof)
    assert proof['status'] == 'ROOT_ACCEPTED900_DESCRIPTIVE_ATTRIBUTION_INPUT_GRID_AND_FSUM_STATISTICS_PASS'
    assert proof['source_seal_sha256'] == sha(attribution_dir / 'FILES_SHA256.json')
    assert proof['source_records_sha256'] == '983bca43dff7e94dd79a312f273c65ed3ea1d146fff3ebfbf3bd2a854e124529'
    assert proof['scalar_checks'] == 2052 and proof['max_abs_difference'] < 1e-12
    addendum = ROOT / 'docs/server_deployment_20260923/revision_20260923/rebuttal_validation900_addendum_20261009.md'
    state['nine_method_view_attribution_20261009'] = dict(
        status=proof['status'], entry=(attribution_dir / 'REPORT.md').relative_to(ROOT).as_posix(),
        root_proof_sha256=sha(attribution_proof), source_seal_sha256=proof['source_seal_sha256'],
        scalar_checks=proof['scalar_checks'], positive_mean_advantage_baseline_counts=proof['positive_mean_advantage_baseline_counts'],
        within_seed_scenarios=10, independent_seeds=10, new_inference=0, test=False, primary_endpoint_selected=False,
        reply_addendum=addendum.relative_to(ROOT).as_posix(), reply_addendum_sha256=sha(addendum))
    state['latest_rebuttal_draft']['validation900_interpretation_addendum'] = addendum.relative_to(ROOT).as_posix()
locator_dir = TRAIN / 'manuscript_source_locator_20261009'
if (locator_dir / 'ROOT_REVIEW.json').exists():
    locator = read(locator_dir / 'ROOT_REVIEW.json')
    assert locator['status'] == 'ROOT_HISTORICAL_MANUSCRIPT_SOURCE_AND_SUBMITTED_VERSION_MISMATCH_PINS_PASS'
    assert locator['report_sha256'] == sha(locator_dir / 'REPORT.md') and locator['evidence_sha256'] == sha(locator_dir / 'EVIDENCE.json')
    state['manuscript_source_locator_20261009'] = dict(
        entry=(locator_dir / 'REPORT.md').relative_to(ROOT).as_posix(), root_proof_sha256=sha(locator_dir / 'ROOT_REVIEW.json'),
        historical_source='paper.md', historical_source_found=True, submitted_matching_source_found=False,
        historical_project_complete=False, manuscript_edited=False, source_path_question_pending=True)
figure_dir = ROOT / 'outputs/guardfed_figures/synthetic_terminal_candidate_20261009'
if (figure_dir / 'ROOT_REVIEW.json').exists():
    figure = read(figure_dir / 'ROOT_REVIEW.json')
    assert figure['status'] == 'ROOT_HISTORICAL_TERMINAL_FIGURE_CANDIDATE_SOURCE_GRID_MEANS_AND_VISUAL_PASS_NOT_ADOPTED'
    assert figure['source_seal_sha256'] == sha(figure_dir / 'FILES_SHA256.json')
    assert figure['original_same_round_triplets_checked'] == 260 and figure['mean_scalars_checked'] == 78
    assert not figure['author_adoption'] and not figure['original_figure_replaced']
    state['synthetic_terminal_figure_candidate_20261009'] = dict(
        status=figure['status'], entry=(figure_dir / 'README.md').relative_to(ROOT).as_posix(),
        pdf=(figure_dir / 'fig3_terminal_candidate.pdf').relative_to(ROOT).as_posix(),
        pdf_sha256=sha(figure_dir / 'fig3_terminal_candidate.pdf'), root_proof_sha256=sha(figure_dir / 'ROOT_REVIEW.json'),
        original_same_round_triplets=260, points=26, independent_seed_n=1, historical_test_visible=True,
        checkpoint_binary_identity_available=False, author_adoption=False, original_figure_replaced=False)
display_dir = ROOT / 'outputs/guardfed_tables/celeba_nine_method_three_view_pdf_20261009'
if (display_dir / 'ROOT_REVIEW.json').exists():
    display = read(display_dir / 'ROOT_REVIEW.json')
    assert display['status'] == 'ROOT_NINE_PANEL_DISPLAY_PDF_EXACT_VALUES_SOURCE_PINS_AND_ALL_PAGE_VISUAL_PASS'
    assert display['source_seal_sha256'] == sha(display_dir / 'FILES_SHA256.json')
    assert display['pdf_sha256'] == sha(display_dir / 'celeba_nine_method_three_view.pdf')
    assert display['exact_mean_sd_pairs'] == 2430 and len(display['pages']) == 9
    state['nine_method_three_view_pdf_20261009'] = dict(
        status=display['status'], entry=(display_dir / 'README.md').relative_to(ROOT).as_posix(),
        pdf=(display_dir / 'celeba_nine_method_three_view.pdf').relative_to(ROOT).as_posix(),
        pdf_sha256=display['pdf_sha256'], root_proof_sha256=sha(display_dir / 'ROOT_REVIEW.json'),
        pages=9, exact_mean_sd_pairs=2430, statistics_recomputed=False, new_inference=0, test=False)
paired92_dir = TRAIN/'celeba_mechanism_v1/three_view_interim92_20261009'
if (paired92_dir/'ROOT_REVIEW.json').exists():
    paired92 = read(paired92_dir/'ROOT_REVIEW.json')
    assert sha(paired92_dir/'ROOT_REVIEW.json') == '7f432387cdc8afc97cc1972d528b3b141ec6316eddec523d6134ba0f20563546'
    assert paired92['source_seal_sha256'] == sha(paired92_dir/'FILES_SHA256.json')
    assert paired92['accepted_three_view92'] == state['celeba_mechanism_v1']['three_view_new_models_offserver_verified'] == 92
    assert paired92['complete_scenes'] == 9 and paired92['independent_mean_sd_scalars'] == 1458
    assert not paired92['test'] and paired92['new_inference'] == 0
    state['celeba_mechanism_v1']['latest_paired_three_view_table'] = dict(
        status=paired92['status'], complete_scenes=9, paired_checkpoints=90, preserved_pairs=92,
        incomplete_pairs=2, Full_identity_available=100, Full_replay_devices={'cpu':5,'gpu':95},
        paired_Full_replay_devices={'cpu':5,'gpu':85}, minus_U_replay_device='cpu',
        native_shared_metrics_exact=True, root_mean_sampleSD_checks=1458, saved_count_metrics_reconstructed=1656,
        table_path=(paired92_dir/'snapshot92/TABLES.md').relative_to(TRAIN).as_posix(),
        root_proof_path=(paired92_dir/'ROOT_REVIEW.json').relative_to(ROOT).as_posix(),
        root_proof_sha256=sha(paired92_dir/'ROOT_REVIEW.json'), main_endpoint_selected=False,
        uniform_device_comparison=False, final_test=False, whole_mechanism_complete=False)
    paired_note = ('九个完整场景、90对Full–minus_U checkpoint的raw/native/shared三视图表已独立验收：'
        '1458均值/样本SD标量和1656组计数指标复算通过，729展示单元及原七场景189统计行保留。'
        '共92对184记录，non-IID Sp-DFA仅2seed，保留个体但不入场景均值。'
        '新non-IID FedSA/S-DFA中删除U的raw准确率分别下降0.492/0.453个百分点；公平性方向存在取舍，不能证明各评分项不可或缺。'
        '完整场景Full为5CPU/85GPU、88cu128/2cu130，control为90CPU/cu128，native/shared在全部184记录相同；'
        '混合设备、历史环境、验证集选择和test暴露仍披露，不是最终test或独立聚合因果证据。'
        f'入口{state["celeba_mechanism_v1"]["latest_paired_three_view_table"]["table_path"]}。')

after92 = ROOT/'tmp/celeba_mechanism_valid_incremental_after92_20261009'
after92_ex = after92/'execution_candidate'
if (after92_ex/'ROOT_STARTUP_OBSERVATION.json').exists():
    start92 = read(after92_ex/'ROOT_STARTUP_OBSERVATION.json')
    deployed92 = read(after92_ex/'deployment_receipt.json'); scope92 = read(after92/'SCOPE.json')
    assert start92['status'] == 'ROOT_AFTER92_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS'
    assert start92['deployment_receipt_sha256'] == sha(after92_ex/'deployment_receipt.json')
    assert start92['execution_seal_sha256'] == sha(after92_ex/'EXECUTION_SOURCE_SHA256.json')
    assert start92['original92_not_rerun'] and not start92['batch_failure']
    assert deployed92['remote_installation']['returncode'] == 0
    assert scope92['selected_ids'] == [f'minus_U_non-IID_Sp-DFA_seed{s}' for s in range(91003,91011)]
    assert set(scope92['excluded_prior_ids']) == set(state['celeba_mechanism_v1']['three_view_accepted_ids'])
    state['celeba_mechanism_v1']['after92_valid_replay'] = dict(
        status='RUNNING_AT_REAL_SOURCE_BOUND_LINUX_STARTUP_OBSERVATION',service='guardfed_celeba_mechanism_valid_after92',
        measured_utc=start92['utc'],observed_service=start92['service'],selected_new=8,selected_ids=scope92['selected_ids'],
        excluded_prior=92,actual_inventory_minus_U100=True,other_variant_C4_excluded=True,
        startup_root_proof_path=(after92_ex/'ROOT_STARTUP_OBSERVATION.json').relative_to(ROOT).as_posix(),
        startup_root_proof_sha256=sha(after92_ex/'ROOT_STARTUP_OBSERVATION.json'),
        execution_seal_sha256=start92['execution_seal_sha256'],deployment_receipt_sha256=sha(after92_ex/'deployment_receipt.json'),
        CPU_affinity=list(range(112,120)),compute_threads=8,nice=10,IO='idle',CUDA_visible='',
        offserver_new_accepted=0,new_Full_inference=0,new_training=0,test_inference=False)
    state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_mechanism_valid_after92']))
    progress92 = [p for p in after92_ex.glob('ROOT_PROGRESS_*.json') if not p.name.endswith('.RAW.json')]
    if progress92:
        latest92 = max(progress92); observed92 = read(latest92)
        assert observed92['status'] == 'ROOT_AFTER92_REAL_LINUX_PROGRESS_AND_ALLOCATION_PASS' and not observed92['batch_failure']
        state['celeba_mechanism_v1']['after92_valid_replay'].update(
            latest_measured_utc=observed92['utc'], latest_progress_path=latest92.relative_to(ROOT).as_posix(),
            latest_progress_sha256=sha(latest92), observed_service=observed92['service'],
            remote_terminal_candidates=len(observed92['completed']))
        if observed92['batch_complete']:
            assert 'EXITED' in observed92['service'] and not observed92['processes']
            assert {r['id'] for r in observed92['completed']} == set(scope92['selected_ids'])
            state['celeba_mechanism_v1']['after92_valid_replay']['status'] = 'REMOTE_COMPLETE_OFFSERVER_PENDING'
            state['active_services'] = [s for s in state['active_services'] if s != 'guardfed_celeba_mechanism_valid_after92']
    closures92 = list((after92_ex/'backups').glob('*/ROOT_ADOPTION_REVIEW.json'))
    if closures92:
        assert len(closures92) == 1
        adoption92_path = closures92[0]; adoption92 = read(adoption92_path); delta92 = adoption92_path.parent
        assert adoption92['status'] == 'ROOT_AFTER92_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
        assert (adoption92['prior_three_view_models'], adoption92['accepted_new'], adoption92['cumulative_three_view_models']) == (92,8,100)
        assert adoption92['science_seal_sha256'] == sha(after92/'FILES_SHA256.json')
        assert adoption92['execution_seal_sha256'] == sha(after92_ex/'EXECUTION_SOURCE_SHA256.json')
        for key,name in [('archive_sha256','incremental_valid_three_views.tar.gz'), ('offserver_verification_sha256','OFFSERVER_VERIFICATION.json'), ('backup_receipt_sha256','backup_receipt.json')]:
            assert adoption92[key] == sha(delta92/name)
        assert adoption92['accepted_new_ids'] == scope92['selected_ids'] and adoption92['all_native_differences_zero']
        assert adoption92['original92_unchanged'] and adoption92['new_Full_inference'] == 0 and not adoption92['test_inference']
        prior92 = state['celeba_mechanism_v1']['three_view_accepted_ids']
        assert len(prior92) == len(set(prior92)) == 92 and set(prior92) == set(scope92['excluded_prior_ids'])
        assert not set(prior92) & set(adoption92['accepted_new_ids'])
        state['celeba_mechanism_v1'].update(three_view_new_models_accepted=100, three_view_new_models_offserver_verified=100,
            three_view_accepted_ids=prior92+adoption92['accepted_new_ids'],
            three_view_scope_limit='Complete minus_U100: prior92 plus exact8 accepted after92; Full existing900 only, C4 excluded, no test')
        state['celeba_mechanism_v1']['after92_valid_replay'].update(status='COMPLETE_STRICT_OFFSERVER_NO_NEW_FULL_INFERENCE',
            offserver_new_accepted=8, archive_members_verified=adoption92['archive_members_verified'],
            root_adoption_path=adoption92_path.relative_to(ROOT).as_posix(), root_adoption_sha256=sha(adoption92_path),
            archive_sha256=adoption92['archive_sha256'])
        state['active_services'] = [s for s in state['active_services'] if s != 'guardfed_celeba_mechanism_valid_after92']

paired100_dir = TRAIN/'celeba_mechanism_v1/three_view_interim100_20261009'
if (paired100_dir/'ROOT_REVIEW.json').exists():
    paired100 = read(paired100_dir/'ROOT_REVIEW.json')
    assert sha(paired100_dir/'ROOT_REVIEW.json') == '20d1031448701938063a398fc5416e404b1e8f0549807c85dab0c0204618a2a0'
    assert paired100['source_seal_sha256'] == sha(paired100_dir/'FINAL_FILES_SHA256.json') == '766c08939e7f1d9fa6ab46e233ba7d2859bb0d3b9a0f418d73f5942b8eec0e88'
    assert paired100['accepted_three_view100'] == state['celeba_mechanism_v1']['three_view_new_models_offserver_verified'] == 100
    assert paired100['complete_scenes'] == 10 and paired100['independent_mean_sd_scalars'] == 1620
    assert paired100['independently_reconstructed_metrics'] == 1800 and paired100['display_cells_verified'] == 810
    assert not paired100['test'] and paired100['new_inference'] == 0
    for name,row in read(paired100_dir/'FINAL_FILES_SHA256.json')['files'].items():
        assert sha(paired100_dir/name) == row['sha256']
    state['celeba_mechanism_v1']['latest_paired_three_view_table'] = dict(
        status=paired100['status'],complete_scenes=10,paired_checkpoints=100,preserved_pairs=100,incomplete_pairs=0,
        Full_identity_available=100,Full_replay_devices={'cpu':5,'gpu':95},paired_Full_replay_devices={'cpu':5,'gpu':95},
        minus_U_replay_device='cpu',native_shared_metrics_exact=True,root_mean_sampleSD_checks=1620,
        saved_count_metrics_reconstructed=1800,display_cells_verified=810,
        table_path=(paired100_dir/'snapshot100/TABLES.md').relative_to(TRAIN).as_posix(),
        root_proof_path=(paired100_dir/'ROOT_REVIEW.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(paired100_dir/'ROOT_REVIEW.json'),
        main_endpoint_selected=False,uniform_device_comparison=False,final_test=False,whole_mechanism_complete=False)
    paired_note = ('十个完整场景、100对Full–minus_U checkpoint的raw/native/shared三视图表已独立验收：'
        '1620均值/样本SD标量、1800组计数指标和810展示单元复算通过；旧九场景243统计行及184记录精确保持。'
        'IID/non-IID×五场景×十seed已齐，另外统一保留9/6seed面板，native/shared在200记录相同。'
        '新增non-IID Sp-DFA删除U的raw准确率下降0.151个百分点、两个差距均值上升；native准确率下降0.121个百分点但ASPD下降，保留取舍。'
        'Full5CPU/95GPU及98cu128/2cu130，control100CPU/cu128；混合设备、历史环境、验证选择和test暴露仍披露。'
        '此表只闭合minus_U，不是其余七variant完成或最终test，也不能证明每项不可或缺。'
        f'入口{state["celeba_mechanism_v1"]["latest_paired_three_view_table"]["table_path"]}。')

C1_base = ROOT/'tmp/celeba_mechanism_valid_C1_gate_20261009'
C1_ex = C1_base/'execution_candidate'
if (C1_ex/'ROOT_STARTUP_OBSERVATION.json').exists():
    C1_start = read(C1_ex/'ROOT_STARTUP_OBSERVATION.json')
    C1_scope = read(C1_base/'SCOPE.json')
    C1_deployment = read(C1_ex/'deployment_receipt.json')
    assert C1_start['deployment_receipt_sha256'] == sha(C1_ex/'deployment_receipt.json')
    assert C1_start['execution_seal_sha256'] == sha(C1_ex/'EXECUTION_SOURCE_SHA256.json')
    assert C1_scope['selected_ids'] == ['minus_C_IID_Benign_seed91001']
    assert set(C1_scope['excluded_prior_ids']) == set(state['celeba_mechanism_v1']['three_view_accepted_ids'])
    C1_state = dict(status='REAL_SINGLE_C_GATE_STARTED_OFFSERVER_PENDING',service='guardfed_celeba_mechanism_valid_C1_gate',
        selected_ids=C1_scope['selected_ids'],excluded_U100=True,other_C3_excluded=True,new_Full_inference=0,test=False,
        startup_root_proof_sha256=sha(C1_ex/'ROOT_STARTUP_OBSERVATION.json'),
        deployment_receipt_sha256=sha(C1_ex/'deployment_receipt.json'),offserver_new_accepted=0)
    state['active_services']=list(dict.fromkeys(state['active_services']+[C1_state['service']]))
    C1_progress = sorted(p for p in C1_ex.glob('ROOT_PROGRESS_*.json') if not p.name.endswith('.RAW.json'))
    if C1_progress:
        C1_latest=read(C1_progress[-1])
        C1_state.update(latest_observation_sha256=sha(C1_progress[-1]),actual_service=C1_latest['service'],
            actual_processes=len(C1_latest['processes']),observed_complete=len(C1_latest['completed']))
        if C1_latest['batch_complete'] and not C1_latest['processes']:
            state['active_services']=[s for s in state['active_services'] if s != C1_state['service']]
    C1_roots=list((C1_ex/'backups').glob('*/ROOT_ADOPTION_REVIEW.json'))
    if C1_roots:
        assert len(C1_roots)==1
        C1_path=C1_roots[0]; C1_review=read(C1_path)
        assert sha(C1_path)=='d045665b066dafc25f9970adfdffef9c9a8a388575ec87b9b54d5dcabfa65cab'
        assert (C1_review['prior_three_view_models'],C1_review['accepted_new'],C1_review['cumulative_three_view_models'])==(100,1,101)
        assert C1_review['accepted_new_ids']==C1_scope['selected_ids'] and C1_review['all_native_differences_zero']
        assert C1_review['science_seal_sha256']==sha(C1_base/'FILES_SHA256.json')
        assert C1_review['execution_seal_sha256']==sha(C1_ex/'EXECUTION_SOURCE_SHA256.json')
        for key,name in [('archive_sha256','incremental_valid_three_views.tar.gz'),('offserver_verification_sha256','OFFSERVER_VERIFICATION.json'),('backup_receipt_sha256','backup_receipt.json')]:
            assert C1_review[key]==sha(C1_path.parent/name)
        prior_U100=state['celeba_mechanism_v1']['three_view_accepted_ids']
        assert len(prior_U100)==100 and all(i.startswith('minus_U_') for i in prior_U100)
        state['celeba_mechanism_v1'].update(three_view_new_models_accepted=101,three_view_new_models_offserver_verified=101,
            three_view_accepted_ids=prior_U100+C1_review['accepted_new_ids'],three_view_counts_by_variant={'minus_U':100,'minus_C':1},
            three_view_scope_limit='Complete minus_U100 plus one accepted minus_C interface gate; C1 is not a full C scene, Full only existing900, no test')
        C1_state.update(status='SINGLE_C_GATE_COMPLETE_STRICT_OFFSERVER',offserver_new_accepted=1,
            root_adoption_path=C1_path.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(C1_path),
            archive_members_verified=C1_review['archive_members_verified'],native_max_abs_difference=0)
    state['celeba_mechanism_v1']['C1_valid_gate']=C1_state

C_after1_base=ROOT/'tmp/celeba_mechanism_valid_C_after1_20261009'
C_after1_ex=C_after1_base/'execution_candidate'
if (C_after1_ex/'ROOT_STARTUP_OBSERVATION.json').exists():
    C_start=read(C_after1_ex/'ROOT_STARTUP_OBSERVATION.json');C_scope=read(C_after1_base/'SCOPE.json')
    assert C_start['deployment_receipt_sha256']==sha(C_after1_ex/'deployment_receipt.json')
    assert C_start['execution_seal_sha256']==sha(C_after1_ex/'EXECUTION_SOURCE_SHA256.json')
    assert len(C_scope['selected_ids'])==11 and set(C_scope['excluded_prior_ids'])==set(state['celeba_mechanism_v1']['three_view_accepted_ids'])
    C_state=dict(status='C_AFTER1_STARTED_OFFSERVER_PENDING',service='guardfed_celeba_mechanism_valid_C_after1',
        selected_ids=C_scope['selected_ids'],offserver_new_accepted=0,prior_models=101,
        startup_root_proof_sha256=sha(C_after1_ex/'ROOT_STARTUP_OBSERVATION.json'),
        original101_not_rerun=True,new_Full_inference=0,test=False)
    state['active_services']=list(dict.fromkeys(state['active_services']+[C_state['service']]))
    C_progress=sorted(p for p in C_after1_ex.glob('ROOT_PROGRESS_*.json') if not p.name.endswith('.RAW.json'))
    if C_progress:
        C_live=read(C_progress[-1])
        C_state.update(latest_observation_sha256=sha(C_progress[-1]),actual_service=C_live['service'],
            observed_complete=len(C_live['completed']),actual_processes=len(C_live['processes']))
        if C_live['batch_complete'] and not C_live['processes']:
            state['active_services']=[s for s in state['active_services'] if s!=C_state['service']]
    C_roots=list((C_after1_ex/'backups').glob('*/ROOT_ADOPTION_REVIEW.json'))
    if C_roots:
        assert len(C_roots)==1
        C_root=C_roots[0];C_review=read(C_root)
        assert (C_review['prior_three_view_models'],C_review['accepted_new'],C_review['cumulative_three_view_models'])==(101,11,112)
        assert C_review['accepted_new_ids']==C_scope['selected_ids'] and C_review['all_native_differences_zero']
        assert C_review['science_seal_sha256']==sha(C_after1_base/'FILES_SHA256.json')
        assert C_review['execution_seal_sha256']==sha(C_after1_ex/'EXECUTION_SOURCE_SHA256.json')
        assert C_review['prior101_root_adoption_sha256']==sha(C1_roots[0])
        for key,name in [('offserver_verification_sha256','OFFSERVER_VERIFICATION.json'),('backup_receipt_sha256','backup_receipt.json')]:
            assert C_review[key]==sha(C_root.parent/name)
        before_ids=state['celeba_mechanism_v1']['three_view_accepted_ids']
        assert len(before_ids)==101 and not set(before_ids)&set(C_review['accepted_new_ids'])
        state['celeba_mechanism_v1'].update(three_view_new_models_offserver_verified=112,
            three_view_accepted_ids=before_ids+C_review['accepted_new_ids'],three_view_counts_by_variant={'minus_U':100,'minus_C':12},
            three_view_scope_limit='Complete U100 plus C12 accepted; C IID Benign10 is complete but its three-view paired table requires separate acceptance; Full reused, no test')
        C_state.update(status='C_AFTER1_COMPLETE_STRICT_OFFSERVER',offserver_new_accepted=11,
            root_adoption_path=C_root.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(C_root),
            archive_members_verified=C_review['archive_members_verified'],native_max_abs_difference=0)
    state['celeba_mechanism_v1']['C_after1_valid_replay']=C_state

C_native_dir=TRAIN/'celeba_mechanism_v1/native_C_Benign10_20261009'
if (C_native_dir/'ROOT_VERIFICATION.json').exists():
    C_native_proof=read(C_native_dir/'ROOT_VERIFICATION.json')
    assert sha(C_native_dir/'ROOT_VERIFICATION.json')=='77d046d695d9a36988b7ecbae0a37256389eb92413a70d74d0a838805a6d8872'
    assert C_native_proof['source_seal_sha256']==sha(C_native_dir/'FILES_SHA256.json')
    for row in read(C_native_dir/'FILES_SHA256.json')['members']:
        assert sha(C_native_dir/row['path'])==row['sha256']
    state['celeba_mechanism_v1']['C_native_single_scene_table']=dict(status=C_native_proof['status'],
        table_path=(C_native_dir/'TABLES.md').relative_to(ROOT).as_posix(),
        root_proof_sha256=sha(C_native_dir/'ROOT_VERIFICATION.json'),complete_scenes=1,
        variant='minus_C',paired_seeds=10,panels=[10,9,6],mean_SD_scalars=54,other_C_scenes_complete=False,
        interpretation='Deletion has ACC/AEOD/ASPD tradeoffs; panel directions change; no necessity/causal/significance claim')

C_table_dir=TRAIN/'celeba_mechanism_v1/three_view_C_Benign10_20261009'
if (C_table_dir/'ROOT_VERIFICATION.json').exists():
    C_table_proof=read(C_table_dir/'ROOT_VERIFICATION.json')
    assert sha(C_table_dir/'ROOT_VERIFICATION.json')=='0d047461186184718f58c5423c546137930b287d3bd7cff8e06eacd6e28ad73f'
    assert C_table_proof['C11_root_adoption_sha256']=='8d064687ad7841e1050120a12ada9e10458fea5bb6a4d9da5aeed77584437fc5'
    assert C_table_proof['source_seal_sha256']==sha(C_table_dir/'FINAL_FILES_SHA256.json')
    for item in read(C_table_dir/'FINAL_FILES_SHA256.json')['members']:
        assert sha(C_table_dir/item['path'])==item['sha256']
    state['celeba_mechanism_v1']['C_three_view_single_scene_table']=dict(status=C_table_proof['status'],
        table_path=C_table_proof['table_path'],root_proof_sha256=sha(C_table_dir/'ROOT_VERIFICATION.json'),
        source_seal_sha256=C_table_proof['source_seal_sha256'],variant='minus_C',complete_scenes=1,
        paired_seeds=10,panels=[10,9,6],mean_SD_scalars=162,display_cells=81,
        other_C_scenes_complete=False,partial_F_Flip_pairs_excluded=2,native_shared_records_exact=24,
        interpretation='Raw deletion reduces both disparity means; calibrated tradeoffs and 9/6 changes retained; no necessity or causal claim')
    state['celeba_mechanism_v1']['three_view_scope_limit']='Complete U100 ten scenes and C12 accepted; C IID Benign10 paired three-view table separately adopted, other C scenes incomplete, Full reused, no test'

rebuttal100_dir=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated100_20261009'
if (rebuttal100_dir/'ROOT_REVIEW.json').exists():
    rebuttal100=read(rebuttal100_dir/'ROOT_REVIEW.json')
    assert sha(rebuttal100_dir/'ROOT_REVIEW.json')=='6dfb210c5d53c2badbe4fb53220008e784430fa33f9e68f0ebb81968ce18c391'
    assert rebuttal100['source_seal_sha256']==sha(rebuttal100_dir/'FINAL_FILES_SHA256.json')
    for name,pin in read(rebuttal100_dir/'FINAL_FILES_SHA256.json')['files'].items():
        assert sha(rebuttal100_dir/name)==pin['sha256'] and (rebuttal100_dir/name).stat().st_size==pin['bytes']
    state['latest_rebuttal_draft']=dict(status=rebuttal100['status'],entry=rebuttal100['entry'],
        manuscript_candidate=rebuttal100['manuscript_candidate'],comments_verbatim=24,complete_U_scenes=10,
        root_proof_sha256=sha(rebuttal100_dir/'ROOT_REVIEW.json'),source_seal_sha256=rebuttal100['source_seal_sha256'],
        numeric_pointer_checks=37,submission_gate=rebuttal100['submission_gate'],submitted_manuscript_edited=False,
        other_seven_controls_pending=True,remaining_eight_methods_pending=True,final_test_pending=True)

# Bind newer terminal observations to each adopted delta's sealed snapshot.
for state_key, base_name, expected_terminal, expected_active in (
        ('flgmm_screen32_20261009','celeba_flgmm_screen_20261009_v2_dispatch',26,2),
        ('hybrid_screen32_20261009','celeba_hybrid_screen_execution_20261009',10,1)):
    base = ROOT/'tmp'/base_name
    latest = read(base/'LATEST_BACKUP.json'); chain = read(base/latest['chain_file'])
    if latest['accepted'] != expected_terminal:
        continue
    delta = base/chain['delta_dir']; link = read(delta/'ROOT_READY_CHAIN_LINK.json')
    snapshot_path = delta/'AUTHORIZED_SNAPSHOT.json'; snapshot = read(snapshot_path)
    assert sha(snapshot_path) == link['authorized_snapshot_sha256']
    prior_observation = state[state_key].get('latest_readonly_terminal_observation', {})
    if prior_observation.get('checked_utc', '') >= snapshot['utc']:
        continue
    if state_key.startswith('flgmm'):
        terminals = [row['id'] for row in snapshot['rows'] if row['result_exists'] and row['progress']['round'] == 70]
        active = [row for row in snapshot['rows'] if row['active']]
        active_rounds = [row['progress']['round'] for row in active]
    else:
        terminals = [row['id'] for row in snapshot['rows'] if row['terminal']]
        active = [row for row in snapshot['rows'] if not row['terminal'] and row.get('round') is not None]
        active_rounds = [row['round'] for row in active]
    assert len(terminals) == expected_terminal and len(active) == expected_active
    assert set(chain['accepted_job_ids']) == set(terminals)
    state[state_key]['latest_readonly_terminal_observation'] = dict(
        checked_utc=snapshot['utc'], observed_complete=len(terminals), active=len(active),
        pending=32-len(terminals)-len(active), failures=0, source_bound=True, new_acceptances=0,
        observed_terminal_ids=terminals, active_rounds=active_rounds,
        snapshot_path=snapshot_path.relative_to(ROOT).as_posix(), snapshot_sha256=sha(snapshot_path))

spec_dir=ROOT/'tmp/guardfed_remaining_three_baseline_spec_decision_20261009'
if (spec_dir/'FILES_SHA256.json').exists():
    assert sha(spec_dir/'FILES_SHA256.json')=='0e955b461a0543c615fb5307f693901a310c0479e24a433e4c6856cc8cdf427f'
    for name,pin in read(spec_dir/'FILES_SHA256.json')['files'].items():
        assert sha(spec_dir/name)==pin['sha256'] and (spec_dir/name).stat().st_size==pin['bytes']
    decision=read(spec_dir/'DECISION.json')
    assert decision['scientific_acceptance_added']==0 and not decision['new_clue_found']
    state['remaining_three_baseline_spec_decision_20261009']=dict(status=decision['status'],
        entry=(spec_dir/'REPORT.md').relative_to(ROOT).as_posix(),
        source_seal_sha256=sha(spec_dir/'FILES_SHA256.json'),identities_matched=3,implementations_unlocked=0,
        FedWA='Complete held extended abstract lacks executable DRL specification',
        SmartFL_FedDNA='Complete method text not yet obtained; no claim that the papers omit formulas',
        requested_input='Source-matched full methods/supplements or code/defaults; executable mathematics is sufficient',
        other_experiments_continue=True)

fl_summary_path=ROOT/'tmp/celeba_flgmm_final6_closure_20261009/ROOT_SUMMARY_ADOPTION.json'
if fl_summary_path.exists():
    fl_summary=read(fl_summary_path)
    assert sha(fl_summary_path)=='e602761016e199da157862da3f24c9f9d0f191cfde10fad49074540c672b4a7f'
    assert fl_summary['status']=='ROOT_FROZEN_VALIDATION32_RECIPE_SUMMARY_ADOPTED'
    assert fl_summary['accepted_records']==32 and not fl_summary['formal100_binding_or_execution'] and not fl_summary['final_test']
    assert sha(ROOT/fl_summary['summary_path'])==fl_summary['summary_sha256']
    assert sha(ROOT/fl_summary['independent_review_path'])==fl_summary['independent_review_sha256']
    state['flgmm_screen32_20261009'].update(status='SCREEN32_COMPLETE_STRICT_OFFSERVER_RECIPE_SUMMARY_ADOPTED',
        selected_recipe=fl_summary['selected_recipe'],recipe_summary_root_path=fl_summary_path.relative_to(ROOT).as_posix(),
        recipe_summary_root_sha256=sha(fl_summary_path),summary_path=fl_summary['summary_path'],summary_sha256=fl_summary['summary_sha256'],
        selected_four_condition_mean=fl_summary['selected_four_condition_mean'],accuracy_champion=fl_summary['accuracy_champion'],
        pareto_candidates=len(fl_summary['three_metric_pareto']),score_gap_to_second=fl_summary['score_gap_to_second'],
        seed_n=1,sample_SD_reported=False,significance_claimed=False,formal100_started=False,final_test=False)

state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
running = TRAIN / 'RUNNING.md'
old = running.read_text(encoding='utf-8')
boundary = '# HISTORICAL: Nine-method coverage COMPLETE'
assert boundary in old
history = old[old.index(boundary):]
main = state['celeba_mechanism_v1']
baseline = state['final_evaluator_runtime_20261009']
flgmm = state.get('flgmm_screen32_20261009', {})
hybrid = state.get('hybrid_screen32_20261009', {})
published = state['latest_publication_verification']
next11_note = ('后续准确11份模型评价已经源审：non-IID F Flip十seed及FedSA seed91001，排除已闭合60；42拒收检查和原科学函数复用通过。仅为准备，历史next8包不派发。')
if main.get('next11_valid_replay',{}).get('execution_started'):
    next11_note = ('准确11份模型评价已实际启动guardfed_celeba_mechanism_valid_next11：non-IID F Flip十seed及FedSA seed91001，排除已闭合60；'
        'Linux实测核109项源码/数据/终轮身份、实际配额和CPU112–119无占用，单进程8线程/nice10/idleIO/CUDA隐藏已独立观察。'
        '执行封条9f1252dd7c11abfe7cee297b2028ca9c58f179d964efb39ee6008ea860508b15；新科学离机接受仍0，不能以启动推断完成。'
        '主机制8并发保持，旧37与next8不重启，失败即保留停止，不自动重试。')
if main.get('next11_valid_replay',{}).get('offserver_new_accepted') == 11:
    next11_note = ('准确11份机制三视图评价已严格验收并离机，累计71；non-IID F Flip十seed及FedSA seed91001，排除已闭合60。'
        '实际服务正常EXITED、11完成、0残留worker、无失败；120 archive成员、99指标、264混淆计数和33预测规则通过，native偏差全0。'
        'archive SHA cbbaab743385b5c3354a4d93538a66188d0fed5c96df6245e547c484c17e0e87；'
        'backups/incremental_20261009T152532Z/ROOT_ADOPTION_REVIEW.json单独核源码/数据/终轮及全部成员。'
        '历史六场景表保持不变；新七场景三视图表仍须实际Full配对和独立统计验收，不从71条数量推断表完成。')
    if main.get('latest_paired_three_view_table',{}).get('complete_scenes') == 7:
        next11_note = next11_note.replace('新七场景三视图表仍须实际Full配对和独立统计验收，不从71条数量推断表完成。',
            '新七场景三视图表已另经实际Full配对与1134独立统计核验；FedSA单seed仍不入均值，不推断全机制完成。')
after71_note=''
if main.get('after71_valid_replay'):
    replay71=main['after71_valid_replay']
    after71_note=('新一批准确11份已训练模型的三视图valid评价已实际启动guardfed_celeba_mechanism_valid_after71，'
        '仅覆盖native82减已闭合71：non-IID FedSA seed91002–91009和S-DFA seed91001–91003。'
        'CPU112–119、单进程8线程/nice10/idleIO/CUDA隐藏已实测；Full仅引用，已接受71不重跑。'
        f'当前状态{replay71["status"]}，该批新增离机接受{replay71["offserver_new_accepted"]}；启动与远端完成不代替科学验收。'
        f'实际启动凭据{replay71["startup_root_proof_path"]}。'
        '该历史11项范围只含minus_U；其余variant的后续实际终轮以当前native账本为准，不能自动扩大本11评价范围。')
    if replay71['offserver_new_accepted']==11:
        after71_note=('准确11份after71三视图valid评价已严格验收、离机备份及root登记，累计82份；仅覆盖此前native82减已闭合71，'
            'non-IID FedSA seed91002–91009及S-DFA seed91001–91003。实际服务正常EXITED、11完整、无残留/失败，'
            f'{replay71["archive_members_verified"]}个archive成员、99指标/264计数/33规则均验证，native偏差全0。'
            'Full仅引用900已接受的原三视图身份，原71记录不变。现有七场景均值表保持原封存；新FedSA仍仅九seed，不能称八个完整场景。'
            f'新批验收入口{replay71["root_adoption_path"]}。该历史评价范围不含其余variant；后续minus_C等native终轮单独验收，不从源准备推断评价完成。')
after82_note=''
if main.get('after82_failed_attempt'):
    after82_note=('新增准确10份after82模型评价在首worker审批检查停止：固定10项范围仍遇旧11项基数断言。'
        '实际EXITED、0worker、0完成、输出目录无结果文件，失败发生于运行时依赖绑定和CNN之前；新增离机接受0。'
        '原失败source/log/审批/启动检查完整保留，原82评价及主800训练不变；旧after82禁止重启。'
        f'证据{main["after82_failed_attempt"]["root_failure_review_path"]}。独立工程修复须新namespace和有效10审批正向门检，准备不计启动或接受。')
if main.get('after82_v2_valid_replay'):
    replay82=main['after82_v2_valid_replay']
    after82_note+=(' 修复版仅改审批数量11→10和独立namespace/pin；有效10审批及拒收门检通过，原科学计算保持。'
        f'实际新服务guardfed_celeba_mechanism_valid_after82_v2，状态{replay82["status"]}，新增离机接受{replay82["offserver_new_accepted"]}。'
        'Full与已接受82不重推，固定CPU112–119/8线程/nice10/idleIO/CUDA隐藏，后续观测和验收按实际凭据。')
C_after1_note=''
C_replay_note=f"minus_C单项门检{main.get('C1_valid_gate',{}).get('offserver_new_accepted',0)}，尚无完整C场景"
if main.get('C_after1_valid_replay'):
    C_now=main['C_after1_valid_replay']
    C_replay_note=f"minus_C累计{main['three_view_counts_by_variant']['minus_C']}已离机，C表须单独核验"
    C_after1_note=(f"准确11份C补集三视图已实际启动；状态{C_now['status']}，新增离机接受{C_now['offserver_new_accepted']}。"
        '只含IID Benign seed91002–91010及F Flip seed91001/2，原U100+C1不重推，Full只引用。'
        'CPU112–119单8线程/nice10/idleIO/CUDA隐藏与来源身份已经实测；冻结原1e-12及指标/校准规则。'
        '即使C IID Benign十seed评价齐备，其论文表仍须另经配对统计验收，其他C场景仍不完整。')
    if C_now['offserver_new_accepted']==11:
        C_after1_note=(f"准确11份C补集三视图已正常退出、0残留worker并严格验收离机，累计C12、U100。"
            f"{C_now['archive_members_verified']}个归档成员、99指标/264计数/33规则通过，native偏差全0；原101份及Full不重推。"
            f"凭据{C_now['root_adoption_path']}。C IID Benign十seed评价齐备，三视图论文表仍须另经配对统计验收；F Flip仅2seed，其他C场景未齐。")
if main.get('C_three_view_single_scene_table'):
    C_table=main['C_three_view_single_scene_table']
    C_replay_note='minus_C累计12已离机，IID Benign十seed三视图表已独立验收'
    C_after1_note=C_after1_note.replace('三视图论文表仍须另经配对统计验收','该单场景三视图表亦已独立验收')
    C_after1_note+=(f" C IID Benign三视图表现已核验162统计标量、81展示单元和216计数指标，入口{C_table['table_path']}。"
        'raw删除C后ACC−0.014pp、AEOD−0.00483、ASPD−0.00383；native/shared为−0.083pp、+0.00292、−0.00142。'
        '保留10/9/6面板及Full2CPU/8GPU对C10CPU、环境/选择历史；不作C必要性、因果或显著性主张。')

after92_note=''
if main.get('after92_valid_replay'):
    replay92=main['after92_valid_replay']
    after92_note=(f'最后准确8项minus_U三视图评价已实际启动，状态{replay92["status"]}，新增离机接受仍{replay92["offserver_new_accepted"]}。'
        '仅non-IID Sp-DFA seed91003–91010，原92及Full不重推、C4排除；CPU112–119单8线程/nice10/idleIO/CUDA隐藏，源码/数据/终轮及实时资源已核。'
        f'实际启动凭据{replay92["startup_root_proof_path"]}；不得提前称100三视图或十场景三视图完成。')
    if replay92['offserver_new_accepted'] == 8:
        after92_note = ('最后准确8项minus_U三视图评价已正常EXITED、0残留/失败，全部严格验收并离机，累计100份minus_U。'
            f'{replay92["archive_members_verified"]}个归档成员、72指标/192计数/24规则验证通过，native偏差全0；原92及Full不重推、C4排除。'
            f'独立验收凭据{replay92["root_adoption_path"]}；十场景三视图论文表须另有配对统计验收，不从评价数量推断表完成。')
top = f'''# CURRENT: GuardFed返修实验 — 实测 {live['checked_utc']}

当前服务器：ssh -p60350 root@89.22.197.55，实例52183675；repo /workspace/GuardFed-celeba-expanded。用户明确授权停止sglang，模型/文件保留。213.224.31.105:26712当前内部状态未知，不自动切换。先遵守/etc/vast-agents-guide.md，既有SHA为42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa。

## 当前执行与验收

| 阶段 | 实际状态与分母 | 接续入口 |
|---|---|---|
| CelebA机制消融 | 当前观测完成{live['queue_completed']}、活动{len(live['active'])}、等待{live['pending']}、失败{len(live['failed'])}；已独立严格验收并离机{main['scientific_results_offserver_verified']}/800新增，另100 Full显式复用 | server_reactivation_20261009/latest_formal_live.json；celeba_mechanism_v1/EXECUTION.md及dispatch receipt |
| FLGMM验证搜索 | {flgmm.get('offserver_accepted70round_jobs', 0)}/32已严格验收并离机；最新来源绑定终轮/活动读STATE对应快照，不把未验收完成项计作接受 | tmp/celeba_flgmm_screen_20261009_v2_dispatch/LATEST_BACKUP.json及accepted_delta_after6_20261009/ROOT_ADOPTION_REVIEW.json |
| 组合基线验证搜索 | {hybrid.get('offserver_accepted70round_jobs', 0)}/32项已严格验收、离机并通过本机来源绑定的记录复核；尚未完整选recipe | tmp/celeba_hybrid_screen_execution_20261009/LATEST_BACKUP.json |
| 九方法旧checkpoint三视图评价 | {baseline['actual_native_valid_image_replays_accepted']}/900已严格验收并离机；原CPU872服务因native偏差failstop EXITED，不重启 | {baseline['accepted_collection_path']} |
| 机制三视图评价 | 累计{main['three_view_new_models_offserver_verified']}份：minus_U完整100，{C_replay_note}；U论文表{main.get('latest_paired_three_view_table',{}).get('complete_scenes',0)}完整场景 | {main.get('latest_paired_three_view_table',{}).get('table_path','需独立配对')}；其他七variant未完成 |

{C_after1_note}

删除C的IID Benign native十seed论文表已独立核验54统计标量/27展示单元/10对checkpoint，保留9/6seed面板；入口celeba_mechanism_v1/native_C_Benign10_20261009/TABLES.md。十seed配对删除差ACC−0.083个百分点、AEOD+0.00292、ASPD−0.00142，9/6面板方向有变化，不作必要性/因果/显著性结论。F Flip仅2seed、其他C场景未齐，不纳入本表均值。

主机制服务guardfed_celeba_mechanism_formal，固定70round/valid-only/8并发，IID(alpha5000)/non-IID(alpha5)×5场景×10共享seed；100 Full身份已复核，旧权重不重训/重复打包。FLGMM原搜索服务已正常EXITED、0worker，32/32严格离机，冻结规则选Tg20/L2/lr0.001，32评分与32候选均值标量经独立及root复核；前两分差0.00004978，仅n=1验证搜索，不作SD或显著性结论。组合基线服务guardfed_celeba_hybrid_screen32仍运行，GPU0/CPU104单线程，未选完整recipe。两套搜索均固定8候选×四条件、seed91001，100项多seed确认尚未启动，不运行test。

主机制最近实测CPU {live['cpu_used_cores_2sec']:.2f}/{live['cpu_quota_cores']:.2f}核，RAM {live['memory_used_bytes']/1e9:.2f}GB，磁盘余{live['disk_free_bytes']/1e12:.3f}TB；GPU/温度/RecoveryAction与近期错误读同一实时JSON。只在真实轮次/日志、进程身份和资源证据支持时判断健康，低瞬时占用不重启。服务标签与完成文件不代替验收。

## 当前恢复与研究选择

原CPU失败为FairGuard/IID/F Flip/seed91009：native超原1e-12，65成员失败现场完整保留，原chunk036的10份strict partial当时未登记，后来通过显式审阅导入派生436账本。独立单模型GPU诊断已复现原三指标，差值全0；当前CPU/GPU native/raw只有image172599一处翻转，共享校准预测无翻转。三份归档与保存数组已独立验收；缺历史GPU逐图数组，不声称唯一历史根因，原424账本保持不变。凭据NATIVE_GPU_DIAGNOSTIC_ROOT_VERIFICATION.json。

{gpu_recovery_paragraph} native1e-12、原model/source/data/map/root/valid、同checkpoint全部视图与失败保留规则不变；混合CPU/GPU来源不能冒充统一设备的最终公平比较。入口tmp/celeba_valid_gpu_recovery_implementation_20261009/README.md。监控不自动启动准备包。

LoGoFair虚拟人口映射提案已独立核验：四条件共用固定image-ID哈希20组，root/valid顺序相同，80个root(label,Male)格最小116人；未解码valid标签/score、未拟合或评价。虚拟群体不是原训练client，人口定义仍待用户裁定；原32草案的mapping SHA仍null。入口tmp/celeba_logofair_population_proposal_20261009/REPORT.md。

## 已完成证据与剩余交付

2454项历史新增训练、九方法900原始验证结果及旧备份保持原值；当前九方法三视图评价已验收{baseline['actual_native_valid_image_replays_accepted']}/900。重放为混合CPU/GPU来源的验证集评价，最终test仍未完成。旧TableII的480个原值已追溯实际重复数，缺乏依据的SD不补造。完整入口REBUTTAL_COMPLETION_20261009.md；英文rebuttal已对齐24块原意见、40处本地引用及209项SHA声明，尚不能把待补实验写成完成。

原20项cu130与另20项cu128真实图像三轮门检已严格接受、离机，两套Full与原worker短程张量/指标/诊断精确。Fed-NGA/Huber四条真实图像三轮门检含240梯度/攻击oracle已接受；FLGMM的CPU2/GPU4及组合基线的CPU4/CUDA4门检已接受。所有恒定负类和工程/数值失败保留。三轮门检不证明70轮跨环境等价或科学性能优势；门检服务已EXITED，不重启。

完整17行比较仍缺8方法的完整多seed结果：LoGoFair、Fed-NGA、FedWA、Huber、FLGMM、SmartFL、FedDNA及组合控制。梯度方法正式协议、LoGoFair人口和最终评价主终点/测试边界仍待裁定；FedWA/SmartFL/FedDNA忠实规格仍缺，不能用简化旧分支冒充。主机制800、完整机制三视图、冻结最终评价、正文及最终回复仍未完成。Fig3原脚本/ForestDiffusion执行身份仍缺；已核数值与缺失来源明确区分。

已接受native场景的10/9/6种子中期论文表：{state['celeba_mechanism_v1']['latest_interim_paper_table']['table_path']}。仅展示{interim['complete_paired_scenes']}个齐备的Full–minus_U配对场景，保留所有指标及取舍，不补造未完成场景，不以Full最佳seed对比消融均值。在IID Sp-DFA场景，Full准确率较高、去U的两个公平性差距更低，不能声称每项不可或缺。AEOD为绝对TPR差，不是完整equalized odds；Full98cu128+2cu130、多数旧driver570.211.01和当前driver595.84差异、seed91001选择历史均披露。native含各方法原校准，不能据此单独证明聚合机制。native100十场景表已单独核验540统计标量，旧九场景54展示行不变；所有十场景删除U的ACC/ASPD均更低，AEOD八场景更高、两场景更低。non-IID FedSA/S-DFA删除U后ACC分别下降0.601/0.453个百分点，公平性方向存在取舍。九场景三视图已由另一份root凭据独立闭合，数量与来源见下方；不从native表推断评价完成。

{paired_note}

九方法三视图论文表已另行完成并通过root实际900份原receipt连接、8100个组计数指标重建及4860个均值/样本SD核验：outputs/guardfed_tables/celeba_nine_method_three_view_20261009/README.md。完整IID/non-IID、五场景、10/9/6共享种子和三视图均平行保留；旧native三指标及展示表值精确一致，94个旧JSON的SD最后bit差异最大2.78e-17单独保留。英文24意见完整草稿入口{state['latest_rebuttal_draft']['entry']}；最新版本纳入U100十场景及900校准归因，24原意见逐字、37数值pointer与37链接复核，保留COMPAS反例/真实n/环境/选择史及全部pending。原七场景封存稿保留。仍为作者审阅稿，正文源文件未应用，最终评价与其余方法/机制未写成完成。

{next11_note}

{after71_note}

{after82_note}

{after92_note}

九方法校准解释已独立复算2052标量：先每seed平均十场景，再跨seed统计；原native下GuardFed的ACC/AEOD/ASPD均值分别优于6/8、8/8、7/8基线，shared下为7/8、4/8、1/8。这是均值方向，不是显著性或seed胜率。原生公平性优势不能全部归因于聚合。全部正负差、10/9/6面板保留，入口outputs/guardfed_tables/celeba_nine_method_view_attribution_20261009/REPORT.md；英文解释补稿rebuttal_validation900_addendum_20261009.md不替换原封存24意见稿，不改变主终点。

九方法三视图论文表已编译为9页A3横向PDF：outputs/guardfed_tables/celeba_nine_method_three_view_pdf_20261009/celeba_nine_method_three_view.pdf。raw/native/shared各3页10/9/6种子，2430个均值±SD单元及4860个展示数字与原TeX精确一致，九页视觉/页界检查通过。仅label唯一化及wrapper排版，原67成员封条保持；本次无统计重算、推理或test。

提交版源项目尚缺：paper.md虽是IEEEtran LaTeX，但属于不同标题、方法和表结构的历史稿；已核19输入SHA及三工作副本，未找到所查paper.bib/archi.png/inv.pdf，也未认证为提交PDF同版。正文未修改、完整构建未声称。有限检索记录manuscript_source_locator_20261009/REPORT.md；已询问提交版路径，其他训练继续。

Fig3终轮修正候选已交付outputs/guardfed_figures/synthetic_terminal_candidate_20261009/fig3_terminal_candidate.pdf：260条第70轮完整三指标、26设置及78均值独立核验，全部13设置/数据集保留；只一个历史seed，不算场景SD，不用旧逐列最优或争议FairScore。图中明确历史test可见、ForestDiffusion执行身份缺失/PCA标签局限及checkpoint二进制SHA未恢复。候选未采用，不替换原图，不宣称P4关闭。

## Git与巡检

最近已验证推送：{published['commit']}，分支codex/revision-evidence-baselines-20260928，{published['committed_blobs_sha256_verified']}份committed blob逐SHA及远端分支核验；后续本机变化未自动算作已推送。记录{published['proof_path']}。

三小时聊天任务guardfed-training-health仍PAUSED；本会话没有原生automation_update工具，未编辑调度器或建立替代cron/Windows任务。supervisor持续运行训练不等于聊天巡检恢复。待原生接口可用时按server_reactivation_20261009/MONITOR_HANDOFF.md恢复同一任务；不从历史计划自动派发新队列。

下方为历史记录；当前事实以本入口、TRAINING_STATE.json及对应实际凭据为准。源/数据/冻结配置保持一致、无重复worker且已验收项严格跳过时才有限恢复外部中断；数值/逻辑错误保留证据，不循环重试、不改统计/seed/并发或driver/实例。

'''
running.write_text(top+history,encoding='utf-8')
execution = TRAIN / 'celeba_mechanism_v1/EXECUTION.md'
text = execution.read_text(encoding='utf-8')
marker = '# HISTORICAL PREPARATION SNAPSHOT — no execution at time of preparation\n\n'
if marker in text:
    text = text.split(marker,1)[1]
execution.write_text(top.replace('# CURRENT:', '# CURRENT EXECUTION:')+marker+text,encoding='utf-8')
print(json.dumps({'status':phase,'measured_utc':live['checked_utc'],'completed':live['queue_completed'],'active':len(live['active']),'formal':formal}))
