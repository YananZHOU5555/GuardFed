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
interim = read(interim_path)
assert interim['status'] == 'INTERIM_COMPLETE_SCENES_ONLY_NO_NEW_INFERENCE'
assert interim['new_training'] == interim['new_inference'] == 0 and not interim['test_used']
state['celeba_mechanism_v1']['latest_interim_paper_table'] = dict(
    table_path=interim_path.with_name('TABLES.md').relative_to(TRAIN).as_posix(),
    table_sha256=sha(interim_path.with_name('TABLES.md')), statistics_sha256=sha(interim_path),
    complete_paired_scenes=interim['complete_paired_scenes'],
    identity='native Full versus minus_U only; other components incomplete',
    mean_sampleSD=True, additional_9_and_6_seed_panels=True, whole_comparison_complete=False)
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
running = TRAIN / 'RUNNING.md'
old = running.read_text(encoding='utf-8')
boundary = '# HISTORICAL: Nine-method coverage COMPLETE'
assert boundary in old
history = old[old.index(boundary):]
main = state['celeba_mechanism_v1']
baseline = state['final_evaluator_runtime_20261009']
flgmm = state.get('flgmm_screen32_20261009', {})
published = state['latest_publication_verification']
top = f'''# CURRENT: GuardFed返修实验 — 实测 {live['checked_utc']}

当前服务器：ssh -p60350 root@89.22.197.55，实例52183675；repo /workspace/GuardFed-celeba-expanded。用户明确授权停止sglang，模型/文件保留。213.224.31.105:26712当前内部状态未知，不自动切换。先遵守/etc/vast-agents-guide.md，既有SHA为42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa。

## 当前执行与验收

| 阶段 | 实际状态与分母 | 接续入口 |
|---|---|---|
| CelebA机制消融 | 当前观测完成{live['queue_completed']}、活动{len(live['active'])}、等待{live['pending']}、失败{len(live['failed'])}；已独立严格验收并离机{main['scientific_results_offserver_verified']}/800新增，另100 Full显式复用 | server_reactivation_20261009/latest_formal_live.json；celeba_mechanism_v1/EXECUTION.md及dispatch receipt |
| FLGMM验证搜索 | {flgmm.get('offserver_accepted70round_jobs', 0)}/32已严格验收并离机；13:03实测11终轮、2活动、0失败，新增5项待验收，不能计为接受 | tmp/celeba_flgmm_screen_20261009_v2_dispatch/LATEST_BACKUP.json及observations/bounded_review_20261009T130206Z/STATUS.json |
| 组合基线验证搜索 | 原32项队列已启动；最新离机快照尚无完整70轮接受，不把首轮或文件存在计作完成 | tmp/celeba_hybrid_screen_execution_20261009/results_incremental_20261009T1133Z/STATUS.json |
| 九方法旧checkpoint三视图评价 | {baseline['actual_native_valid_image_replays_accepted']}/900已严格验收并离机；原CPU872服务因native偏差failstop EXITED，不重启 | {baseline['accepted_collection_path']} |
| 机制三视图评价 | 23份minus_U已严格验收并离机，另行统计；不是800份均已完成三视图 | server_reactivation_20261009/MECHANISM_VALID_INCREMENTAL_20261009T104749Z_ROOT_VERIFICATION.json |

主机制服务guardfed_celeba_mechanism_formal，固定70round/valid-only/8并发，IID(alpha5000)/non-IID(alpha5)×5场景×10共享seed；100 Full身份已复核，旧权重不重训/重复打包。FLGMM服务guardfed_celeba_flgmm_screen，两张GPU各1任务；组合基线服务guardfed_celeba_hybrid_screen32，GPU0/CPU104单线程。两套32搜索均固定8候选×四条件、seed91001，尚未完整选recipe或启动100项多seed确认，不运行test。

主机制最近实测CPU {live['cpu_used_cores_2sec']:.2f}/{live['cpu_quota_cores']:.2f}核，RAM {live['memory_used_bytes']/1e9:.2f}GB，磁盘余{live['disk_free_bytes']/1e12:.3f}TB；GPU/温度/RecoveryAction与近期错误读同一实时JSON。只在真实轮次/日志、进程身份和资源证据支持时判断健康，低瞬时占用不重启。服务标签与完成文件不代替验收。

## 当前恢复与研究选择

原CPU失败为FairGuard/IID/F Flip/seed91009：native超原1e-12，65成员失败现场完整保留，原chunk036的10份strict partial当时未登记，后来通过显式审阅导入派生436账本。独立单模型GPU诊断已复现原三指标，差值全0；当前CPU/GPU native/raw只有image172599一处翻转，共享校准预测无翻转。三份归档与保存数组已独立验收；缺历史GPU逐图数组，不声称唯一历史根因，原424账本保持不变。凭据NATIVE_GPU_DIAGNOSTIC_ROOT_VERIFICATION.json。

{gpu_recovery_paragraph} native1e-12、原model/source/data/map/root/valid、同checkpoint全部视图与失败保留规则不变；混合CPU/GPU来源不能冒充统一设备的最终公平比较。入口tmp/celeba_valid_gpu_recovery_implementation_20261009/README.md。监控不自动启动准备包。

LoGoFair虚拟人口映射提案已独立核验：四条件共用固定image-ID哈希20组，root/valid顺序相同，80个root(label,Male)格最小116人；未解码valid标签/score、未拟合或评价。虚拟群体不是原训练client，人口定义仍待用户裁定；原32草案的mapping SHA仍null。入口tmp/celeba_logofair_population_proposal_20261009/REPORT.md。

## 已完成证据与剩余交付

2454项历史新增训练、九方法900原始验证结果及旧备份保持原值；900原始训练结果完整，不等于900统一评价重放完整或最终test已完成。旧TableII的480个原值已追溯实际重复数，缺乏依据的SD不补造。完整入口REBUTTAL_COMPLETION_20261009.md；英文rebuttal已对齐24块原意见、40处本地引用及209项SHA声明，尚不能把待补实验写成完成。

原20项cu130与另20项cu128真实图像三轮门检已严格接受、离机，两套Full与原worker短程张量/指标/诊断精确。Fed-NGA/Huber四条真实图像三轮门检含240梯度/攻击oracle已接受；FLGMM的CPU2/GPU4及组合基线的CPU4/CUDA4门检已接受。所有恒定负类和工程/数值失败保留。三轮门检不证明70轮跨环境等价或科学性能优势；门检服务已EXITED，不重启。

完整17行比较仍缺8方法的完整多seed结果：LoGoFair、Fed-NGA、FedWA、Huber、FLGMM、SmartFL、FedDNA及组合控制。梯度方法正式协议、LoGoFair人口和最终评价主终点/测试边界仍待裁定；FedWA/SmartFL/FedDNA忠实规格仍缺，不能用简化旧分支冒充。主机制800、完整机制三视图、冻结最终评价、正文及最终回复仍未完成。Fig3原脚本/ForestDiffusion执行身份仍缺；已核数值与缺失来源明确区分。

已接受场景的10/9/6种子中期论文表：{state['celeba_mechanism_v1']['latest_interim_paper_table']['table_path']}。仅展示{interim['complete_paired_scenes']}个齐备的Full–minus_U配对场景，保留所有指标及取舍，不补造未完成场景，不以Full最佳seed对比消融均值。新增Sp-DFA场景Full准确率较高、去U的两个公平性差距更低，不能声称每项不可或缺。AEOD为绝对TPR差，不是完整equalized odds；Full98cu128+2cu130、多数旧driver570.211.01和当前driver595.84差异、seed91001选择历史均披露。native含各方法原校准，不能据此单独证明聚合机制。

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
