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
    proof_paths = list(science_backup.glob('*offserver_verification.json'))
    proof_paths += list(science_backup.glob('root_delta_*/OFFSERVER_VERIFICATION.json'))
    verified_proofs = {read(p)['archive_sha256']: read(p) for p in proof_paths}
    backed_up, previous, backup_entries = set(), None, []
    for entry in ledger['entries']:
        local_receipt = science_backup / Path(entry['receipt']).name
        if not local_receipt.exists():
            # New batches keep their actual original files together; do not
            # duplicate model archives merely to fit the historical flat layout.
            batch_name = Path(entry['archive']).name.removesuffix('.tar.gz')
            local_receipt = science_backup / batch_name / Path(entry['receipt']).name
            assert local_receipt.resolve().is_relative_to(science_backup.resolve())
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
    inspections += [(p, read(p)) for p in science_backup.glob('root_delta_*/inspection/inspection.json')]
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
publication_proofs = list(TRAIN.glob('publication_*verified_*.json'))
if publication_proofs:
    publication_proof = max(publication_proofs, key=lambda p: read(p).get('verified_utc', ''))
    verified = read(publication_proof)
    assert verified['status'] == 'COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS'
    state['latest_publication_verification'] = dict(
        commit=verified['commit'], branch=verified['branch'], verified_utc=verified['verified_utc'],
        committed_blobs_sha256_verified=verified['committed_blobs_sha256_verified'],
        proof_path=publication_proof.relative_to(TRAIN).as_posix(), proof_sha256=sha(publication_proof),
        acceptance_cutoff=verified.get('acceptance_cutoff',{}),
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
        ('hybrid_screen32_20261009','celeba_hybrid_screen_execution_20261009',10,1),
        ('hybrid_screen32_20261009','celeba_hybrid_screen_execution_20261009',14,1),
        ('hybrid_screen32_20261009','celeba_hybrid_screen_execution_20261009',27,1)):
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

fl_bound_path=ROOT/'tmp/celeba_flgmm_fullcoverage_binding_20261009/ROOT_BOUND_ADOPTION.json'
if fl_bound_path.exists():
    fl_bound=read(fl_bound_path)
    assert sha(fl_bound_path)=='fcecfc0a3582695edfd54c70db38e7dafdd5bf46dcdff8212b9dfc03fc7506fc'
    assert fl_bound['status']=='ROOT_ACTUAL_BOUND96_PLUS4_METADATA_MEMBER_AND_SCOPE_PASS'
    assert (fl_bound['planned_new'],fl_bound['reused'],fl_bound['planned_total'])==(96,4,100)
    assert not fl_bound['formal100_started'] and not fl_bound['final_test']
    state['flgmm_fullcoverage_v2_20261009']=dict(status='BOUND_METADATA_ACCEPTED_NOT_STARTED',
        server='root@89.22.197.55:60350',stage='/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/stage',
        root_bound_path=fl_bound_path.relative_to(ROOT).as_posix(),root_bound_sha256=sha(fl_bound_path),
        package_sha256=fl_bound['package_sha256'],manifest_sha256=fl_bound['manifest_sha256'],
        selected_recipe=fl_bound['selected_recipe'],planned_new=96,reused=4,planned_total=100,
        prepared_new_canaries=5,original_references=2,canaries_started=0,new_accepted=0,
        actual_metadata_archive_members=144,source_v2_minimal_engineering_repairs=True,
        local_extract_failure_preserved=True,local_manual_safe_extraction_pass=True,repeated_remote_binding=0,
        checkpoint_repacked=0,formal100_started=False,final_test=False)

fl_start_path=ROOT/'tmp/celeba_flgmm_fullcoverage_binding_20261009/ROOT_CANARY_STARTUP.json'
if fl_start_path.exists():
    fl_start=read(fl_start_path)
    assert sha(fl_start_path)=='22486ecfd6e4012fcb30a27c120574dee918c62069dec55c6ad225e24d68d68b'
    for key,pin in [('start_receipt_path','start_receipt_sha256'),('resource_path','resource_sha256'),('snapshot_path','snapshot_sha256')]:
        assert sha(ROOT/fl_start[key])==fl_start[pin]
    assert fl_start['main_growth_verified'] and not fl_start['formal100_started'] and not fl_start['final_test']
    coverage=state['flgmm_fullcoverage_v2_20261009']
    coverage.pop('canaries_started',None)
    coverage.update(status='SEVEN_CANARY_QUEUE_ACTUALLY_STARTED_NOT_ACCEPTED',canary_runner_started=True,
        canary_scope=7,canaries_accepted=0,startup_root_path=fl_start_path.relative_to(ROOT).as_posix(),
        startup_root_sha256=sha(fl_start_path),startup_observed_utc=fl_start['observed_utc'])
    observations=list((ROOT/'tmp/celeba_flgmm_fullcoverage_root_operations_20261009').glob('observation_*/SNAPSHOT.json'))
    if observations:
        observed_path=max(observations,key=lambda p:read(p)['utc']);observed=read(observed_path)
        coverage.update(latest_readonly_observation_path=observed_path.relative_to(ROOT).as_posix(),
            latest_readonly_observation_sha256=sha(observed_path),latest_observed_utc=observed['utc'],
            service_observed=observed['canary_service'],failure_paths_observed=observed['failure_paths'],
            source_members_match_observed=observed['source_members_match'])
    if 'RUNNING' in coverage['service_observed']['stdout']:
        state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_flgmm_fullcoverage_canary']))

fl_closure_path=ROOT/'tmp/celeba_flgmm_fullcoverage_binding_20261009/ROOT_SEVEN_CANARY_CLOSURE.json'
if fl_closure_path.exists():
    fl_closure=read(fl_closure_path)
    assert sha(fl_closure_path)=='e17c50890129b512dbcf1f426fcf6392e47b1cb9db95a038a8a61b515ca8b958'
    assert sha(ROOT/fl_closure['offserver_path'])==fl_closure['offserver_sha256']
    assert (fl_closure['accepted_new_canaries'],fl_closure['same_horizon_pairs'],fl_closure['total_canary_runs'])==(5,2,7)
    state['flgmm_fullcoverage_v2_20261009'].update(status='SEVEN_CANARIES_COMPLETE_STRICT_OFFSERVER_ROOT_ADOPTED',
        canaries_accepted=5,same_horizon_reference_pairs_verified=2,total_canary_runs_verified=7,canary_archive_members_verified=315,
        closure_root_path=fl_closure_path.relative_to(ROOT).as_posix(),closure_root_sha256=sha(fl_closure_path),
        gate_sha256=fl_closure['gate_sha256'],offserver_sha256=fl_closure['offserver_sha256'],canary_archive_sha256=fl_closure['archive_sha256'])
    state['active_services']=[s for s in state['active_services'] if s!='guardfed_celeba_flgmm_fullcoverage_canary']

fl_coverage_start_path=ROOT/'tmp/celeba_flgmm_fullcoverage_binding_20261009/ROOT_COVERAGE_STARTUP.json'
if fl_coverage_start_path.exists():
    full_start=read(fl_coverage_start_path)
    assert sha(fl_coverage_start_path)=='abfa572c21cd5724379ff007a71c1a030f69c773dfc3021358cee4fe08845d1d'
    for key,pin in [('startup_path','startup_sha256'),('resource_path','resource_sha256'),('runtime_path','runtime_sha256')]:
        assert sha(ROOT/full_start[key])==full_start[pin]
    assert full_start['formal100_started'] and not full_start['final_test'] and full_start['active_rounds']==[1,1]
    state['flgmm_fullcoverage_v2_20261009'].update(status=full_start['status'],formal100_started=True,
        coverage_startup_root_path=fl_coverage_start_path.relative_to(ROOT).as_posix(),coverage_startup_root_sha256=sha(fl_coverage_start_path),
        coverage_observed_utc=full_start['observed_utc'],active=2,pending=94,active_rounds=full_start['active_rounds'],
        new_completed_observed=0,new_accepted=0,worker_pids=full_start['worker_pids'],canary_authorization_preserved=True)
    latest_full_obs=read(ROOT/state['flgmm_fullcoverage_v2_20261009']['latest_readonly_observation_path'])
    if latest_full_obs.get('queue') is not None:
        q=latest_full_obs['queue'];assert q['failed'] is False
        progress_rows={r['id']:r for r in latest_full_obs['rows'] if r['kind']=='new'}
        state['flgmm_fullcoverage_v2_20261009'].update(coverage_observed_utc=latest_full_obs['utc'],
            active=len(q['active']),pending=q['pending'],new_completed_observed=q['completed_new'],
            active_rounds=[(progress_rows[r['id']].get('progress') or {}).get('round') for r in q['active']],
            worker_pids=[r['pid'] for r in q['active']],coverage_service_observed=latest_full_obs['coverage_service'])
    state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_flgmm_fullcoverage']))

C8_dir=ROOT/'tmp/celeba_mechanism_valid_C_after12_20261009/execution_candidate'
if (C8_dir/'ROOT_STARTUP_OBSERVATION.json').exists():
    C8_start=read(C8_dir/'ROOT_STARTUP_OBSERVATION.json')
    assert sha(C8_dir/'ROOT_STARTUP_OBSERVATION.json')=='a812bff3084254968d2fdafddd2dac99ebbf0d28eab942effb751b7f3de03757'
    assert C8_start['deployment_receipt_sha256']==sha(C8_dir/'deployment_receipt.json')
    assert C8_start['scientific_offserver_new_accepted']==0 and not C8_start['test_inference']
    state['celeba_mechanism_v1']['C_after12_valid_replay']=dict(status='ACTUAL_STARTUP_VERIFIED_ACCEPTANCE_PENDING',
        selected_count=8,prior_three_view_models=112,offserver_new_accepted=0,execution_started=True,
        startup_root_proof_path=(C8_dir/'ROOT_STARTUP_OBSERVATION.json').relative_to(ROOT).as_posix(),
        startup_root_proof_sha256=sha(C8_dir/'ROOT_STARTUP_OBSERVATION.json'),
        actual_worker_pids=[p['pid'] for p in C8_start['processes'] if 'worker' in p['argv']],
        CPU_affinity=list(range(112,120)),compute_threads=8,nice=10,IO='idle',CUDA_visible='',new_training=0,
        new_Full_inference=0,final_test=False)
    state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_mechanism_valid_C_after12']))
    C8_roots=list((C8_dir/'backups').glob('incremental_*/ROOT_ADOPTION_REVIEW.json'))
    if C8_roots:
        assert len(C8_roots)==1
        C8_root_path=C8_roots[0];C8_root=read(C8_root_path);C8_delta=C8_root_path.parent
        assert sha(C8_root_path)=='817d5f8ebebb566ee4b851fd600edcddaf07a410d29e618d1e5a821c5748b775'
        assert (C8_root['prior_three_view_models'],C8_root['accepted_new'],C8_root['cumulative_three_view_models'])==(112,8,120)
        for key,name in [('archive_sha256','incremental_valid_three_views.tar.gz'),('offserver_verification_sha256','OFFSERVER_VERIFICATION.json'),('backup_receipt_sha256','backup_receipt.json')]:
            assert sha(C8_delta/name)==C8_root[key]
        old112=state['celeba_mechanism_v1']['three_view_accepted_ids'];assert len(old112)==112
        assert not set(old112)&set(C8_root['accepted_new_ids']) and C8_root['original112_unchanged'] and C8_root['all_native_differences_zero']
        state['celeba_mechanism_v1'].update(three_view_new_models_accepted=120,three_view_new_models_offserver_verified=120,
            three_view_accepted_ids=old112+C8_root['accepted_new_ids'],three_view_counts_by_variant={'minus_U':100,'minus_C':20},
            three_view_scope_limit='U100 ten scenes and C20 IID Benign/F Flip complete terminal replays; C F Flip paired table remains a separate review; Full reused, no test')
        state['celeba_mechanism_v1']['C_after12_valid_replay'].update(status='EXACT8_COMPLETE_STRICT_OFFSERVER_ROOT_ADOPTED',
            offserver_new_accepted=8,root_adoption_path=C8_root_path.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(C8_root_path),
            archive_sha256=C8_root['archive_sha256'],archive_members_verified=89,independent_metric_checks=72,
            confusion_count_checks=192,prediction_rule_checks=24,native_max_abs_difference=0,actual_worker_pids=[])
        state['active_services']=[s for s in state['active_services'] if s!='guardfed_celeba_mechanism_valid_C_after12']

C20_table_dir=TRAIN/'celeba_mechanism_v1/three_view_C_two_scenes_20261009'
if (C20_table_dir/'ROOT_VERIFICATION.json').exists():
    C20_table=read(C20_table_dir/'ROOT_VERIFICATION.json')
    assert sha(C20_table_dir/'ROOT_VERIFICATION.json')=='f29be3341d12558099c92a59a06cdf7de7f45b5c8d3194e126ae21899a12b875'
    assert (C20_table['unique_records'],C20_table['paired_models'],C20_table['mean_SD_scalars_recomputed'])==(40,20,324)
    assert sha(C20_table_dir/'snapshot/TABLES.md')==C20_table['display_sha256']
    state['celeba_mechanism_v1']['C_three_view_two_scene_table']=dict(status=C20_table['status'],
        root_proof_path=(C20_table_dir/'ROOT_VERIFICATION.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(C20_table_dir/'ROOT_VERIFICATION.json'),
        table_path=C20_table['canonical_table'],paired_models=20,unique_records=40,complete_scenes=2,mean_SD_scalars=324,display_cells=162,
        count_metrics=360,replay_devices=C20_table['replay_devices'],other_C_scenes_complete=False,final_test=False)

rebuttal_C20_dir=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C20_20261009'
if (rebuttal_C20_dir/'ROOT_REVIEW.json').exists():
    writing=read(rebuttal_C20_dir/'ROOT_REVIEW.json')
    assert writing['status']=='ROOT_C20_COMPLETE_AUTHOR_REVIEW_TEXT_DELTA_AND_SOURCE_POINTERS_PASS'
    assert sha(rebuttal_C20_dir/'FILES_SHA256.json')==writing['source_seal_sha256']
    for name,pin in read(rebuttal_C20_dir/'FILES_SHA256.json')['files'].items():
        assert sha(rebuttal_C20_dir/name)==pin['sha256'] and (rebuttal_C20_dir/name).stat().st_size==pin['bytes']
    assert sha(rebuttal_C20_dir/'rebuttal_integrated_20261009.md')==writing['rebuttal_sha256']
    assert sha(rebuttal_C20_dir/'manuscript_insertions_integrated_20261009.md')==writing['insertions_sha256']
    state['latest_rebuttal_draft'].update(status=writing['status'],
        entry=(rebuttal_C20_dir/'rebuttal_integrated_20261009.md').relative_to(ROOT).as_posix(),
        manuscript_candidate=(rebuttal_C20_dir/'manuscript_insertions_integrated_20261009.md').relative_to(ROOT).as_posix(),
        root_proof_path=(rebuttal_C20_dir/'ROOT_REVIEW.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(rebuttal_C20_dir/'ROOT_REVIEW.json'),
        source_seal_sha256=writing['source_seal_sha256'],complete_C_scenes=2,new_C_scalar_pointer_checks=22,
        new_C_scope_environment_checks=12,links_checked=41,changed_passages=10)
C5_dir=ROOT/'tmp/celeba_mechanism_valid_C_after20_20261009/execution_candidate'
if (C5_dir/'ROOT_STARTUP_OBSERVATION.json').exists():
    C5_start=read(C5_dir/'ROOT_STARTUP_OBSERVATION.json')
    assert C5_start['status']=='ROOT_C_AFTER20_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS'
    assert C5_start['execution_seal_sha256']=='d359805ca590958da37a02735fdd2daf1efe615332d91c37118835b97e24b260'
    assert C5_start['deployment_receipt_sha256']==sha(C5_dir/'deployment_receipt.json')
    assert C5_start['original120_not_rerun'] and not C5_start['test_inference']
    C5_stage=dict(status='ACTUAL_STARTUP_VERIFIED_ACCEPTANCE_PENDING',selected_count=5,prior_three_view_models=120,
        offserver_new_accepted=0,execution_started=True,startup_root_proof_path=(C5_dir/'ROOT_STARTUP_OBSERVATION.json').relative_to(ROOT).as_posix(),
        startup_root_proof_sha256=sha(C5_dir/'ROOT_STARTUP_OBSERVATION.json'),CPU_affinity=list(range(112,120)),
        compute_threads=8,nice=10,IO='idle',CUDA_visible='',new_training=0,new_Full_inference=0,final_test=False)
    state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_mechanism_valid_C_after20']))
    C5_roots=list((C5_dir/'backups').glob('incremental_*/ROOT_ADOPTION_REVIEW.json'))
    if C5_roots:
        assert len(C5_roots)==1
        C5_path=C5_roots[0];C5=read(C5_path);C5_delta=C5_path.parent
        assert C5['status']=='ROOT_C_AFTER20_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
        assert (C5['prior_three_view_models'],C5['accepted_new'],C5['cumulative_three_view_models'])==(120,5,125)
        assert C5['original120_unchanged'] and C5['all_native_differences_zero'] and C5['new_Full_inference']==0 and not C5['test_inference']
        for key,name in [('archive_sha256','incremental_valid_three_views.tar.gz'),('offserver_verification_sha256','OFFSERVER_VERIFICATION.json'),('backup_receipt_sha256','backup_receipt.json')]:
            assert sha(C5_delta/name)==C5[key]
        prior_ids=state['celeba_mechanism_v1']['three_view_accepted_ids']
        assert len(prior_ids)==120 and not set(prior_ids)&set(C5['accepted_new_ids'])
        state['celeba_mechanism_v1'].update(three_view_new_models_accepted=125,three_view_new_models_offserver_verified=125,
            three_view_accepted_ids=prior_ids+C5['accepted_new_ids'],three_view_counts_by_variant={'minus_U':100,'minus_C':25},
            three_view_scope_limit='U100 complete; C20 IID Benign/F Flip complete plus five IID FedSA pairs; FedSA incomplete, no final test.')
        C5_stage.update(status='EXACT5_COMPLETE_STRICT_OFFSERVER_ROOT_ADOPTED',offserver_new_accepted=5,
            root_adoption_path=C5_path.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(C5_path),
            archive_members_verified=C5['archive_members_verified'],all_native_differences_zero=True,service_terminal='EXITED')
        state['active_services']=[s for s in state['active_services'] if s!='guardfed_celeba_mechanism_valid_C_after20']
    state['celeba_mechanism_v1']['C_after20_valid_replay']=C5_stage
C3_dir=ROOT/'tmp/celeba_mechanism_valid_C_after25_20261009/execution_candidate'
if (C3_dir/'ROOT_STARTUP_OBSERVATION.json').exists():
    C3_start=read(C3_dir/'ROOT_STARTUP_OBSERVATION.json')
    assert sha(C3_dir/'ROOT_STARTUP_OBSERVATION.json')=='b25f5ac600b7ddda42674042dfd241fbc97ec67465fdfd5615c2061da2797487'
    assert C3_start['status']=='ROOT_C_AFTER25_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS'
    assert C3_start['execution_seal_sha256']=='adbad62e9a7fa3936790255e40bd29262441798d17e104a8cda26c1d9b722560'
    assert C3_start['deployment_receipt_sha256']==sha(C3_dir/'deployment_receipt.json')
    assert C3_start['original125_not_rerun'] and not C3_start['test_inference']
    C3_stage=dict(status='ACTUAL_STARTUP_VERIFIED_ACCEPTANCE_PENDING',selected_count=3,prior_three_view_models=125,
        offserver_new_accepted=0,execution_started=True,startup_root_proof_path=(C3_dir/'ROOT_STARTUP_OBSERVATION.json').relative_to(ROOT).as_posix(),
        startup_root_proof_sha256=sha(C3_dir/'ROOT_STARTUP_OBSERVATION.json'),CPU_affinity=list(range(112,120)),
        compute_threads=8,nice=10,IO='idle',CUDA_visible='',new_training=0,new_Full_inference=0,final_test=False)
    state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_mechanism_valid_C_after25']))
    C3_roots=list((C3_dir/'backups').glob('incremental_*/ROOT_ADOPTION_REVIEW.json'))
    if C3_roots:
        assert len(C3_roots)==1
        C3_path=C3_roots[0];C3=read(C3_path);C3_delta=C3_path.parent
        assert C3['status']=='ROOT_C_AFTER25_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
        assert (C3['prior_three_view_models'],C3['accepted_new'],C3['cumulative_three_view_models'])==(125,3,128)
        assert C3['original125_unchanged'] and C3['all_native_differences_zero'] and C3['new_Full_inference']==0 and not C3['test_inference']
        assert C3['prior125_root_adoption_sha256']=='ba3edd176219106c781e8b437017ff7444cee8b3e47073b25932928682661ef3'
        assert C3['archive_members_verified']==54
        C3_arrays=read(C3_delta/'OFFSERVER_VERIFICATION.json')
        assert (C3_arrays['independent_metric_checks'],C3_arrays['independent_confusion_count_checks'],C3_arrays['prediction_rule_checks'])==(27,72,9)
        for key,name in [('archive_sha256','incremental_valid_three_views.tar.gz'),('offserver_verification_sha256','OFFSERVER_VERIFICATION.json'),('backup_receipt_sha256','backup_receipt.json')]:
            assert sha(C3_delta/name)==C3[key]
        prior_ids=state['celeba_mechanism_v1']['three_view_accepted_ids']
        assert len(prior_ids)==125 and not set(prior_ids)&set(C3['accepted_new_ids'])
        state['celeba_mechanism_v1'].update(three_view_new_models_accepted=128,three_view_new_models_offserver_verified=128,
            three_view_accepted_ids=prior_ids+C3['accepted_new_ids'],three_view_counts_by_variant={'minus_U':100,'minus_C':28},
            three_view_scope_limit='U100 complete; C20 IID Benign/F Flip complete plus eight IID FedSA pairs; FedSA incomplete, no final test.')
        C3_stage.update(status='EXACT3_COMPLETE_STRICT_OFFSERVER_ROOT_ADOPTED',offserver_new_accepted=3,
            root_adoption_path=C3_path.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(C3_path),
            archive_members_verified=54,all_native_differences_zero=True,service_terminal='EXITED')
        state['active_services']=[s for s in state['active_services'] if s!='guardfed_celeba_mechanism_valid_C_after25']
    state['celeba_mechanism_v1']['C_after25_valid_replay']=C3_stage
C28_dir=ROOT/'tmp/celeba_mechanism_valid_C_after28_20261009/execution_candidate'
if (C28_dir/'ROOT_STARTUP_OBSERVATION.json').exists():
    C28_start=read(C28_dir/'ROOT_STARTUP_OBSERVATION.json')
    assert sha(C28_dir/'ROOT_STARTUP_OBSERVATION.json')=='7a1b78af706f3b3a8659340bfdba8574cd3efa0fba6694299dacab52a6a57e67'
    assert C28_start['status']=='ROOT_C_AFTER28_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS'
    assert C28_start['execution_seal_sha256']=='c823dce69ef511ff15faf1b964805184a6a8c10f271214de1cc6706446a49ba2'
    assert C28_start['deployment_receipt_sha256']==sha(C28_dir/'deployment_receipt.json')
    assert C28_start['original128_not_rerun'] and not C28_start['test_inference']
    C28_stage=dict(status='ACTUAL_STARTUP_VERIFIED_ACCEPTANCE_PENDING',selected_count=8,prior_three_view_models=128,
        offserver_new_accepted=0,execution_started=True,startup_root_proof_path=(C28_dir/'ROOT_STARTUP_OBSERVATION.json').relative_to(ROOT).as_posix(),
        startup_root_proof_sha256=sha(C28_dir/'ROOT_STARTUP_OBSERVATION.json'),CPU_affinity=list(range(112,120)),
        compute_threads=8,nice=10,IO='idle',CUDA_visible='',new_training=0,new_Full_inference=0,final_test=False)
    state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_mechanism_valid_C_after28']))
    C28_roots=list((C28_dir/'backups').glob('incremental_*/ROOT_ADOPTION_REVIEW.json'))
    if C28_roots:
        assert len(C28_roots)==1
        C28_path=C28_roots[0];C28=read(C28_path);C28_delta=C28_path.parent
        assert sha(C28_path)=='cd197e57b42dda87bc239736e401cb5accd029313546533f6a48caf27950e1e0'
        assert C28['status']=='ROOT_C_AFTER28_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
        assert (C28['prior_three_view_models'],C28['accepted_new'],C28['cumulative_three_view_models'])==(128,8,136)
        assert C28['original128_unchanged'] and C28['all_native_differences_zero'] and C28['new_Full_inference']==0 and not C28['test_inference']
        assert C28['prior128_root_adoption_sha256']=='0d1661bdc21025b957fa4e6ac39d4c8924cb50ab1b91920268119941312a615a'
        assert C28['startup_observation_sha256']==sha(C28_dir/'ROOT_STARTUP_OBSERVATION.json')
        assert C28['archive_members_verified']==89
        C28_arrays=read(C28_delta/'OFFSERVER_VERIFICATION.json')
        assert (C28_arrays['independent_metric_checks'],C28_arrays['independent_confusion_count_checks'],C28_arrays['prediction_rule_checks'])==(72,192,24)
        for key,name in [('archive_sha256','incremental_valid_three_views.tar.gz'),('offserver_verification_sha256','OFFSERVER_VERIFICATION.json'),('backup_receipt_sha256','backup_receipt.json')]:
            assert sha(C28_delta/name)==C28[key]
        prior_ids=state['celeba_mechanism_v1']['three_view_accepted_ids']
        assert len(prior_ids)==128 and not set(prior_ids)&set(C28['accepted_new_ids'])
        state['celeba_mechanism_v1'].update(three_view_new_models_accepted=136,three_view_new_models_offserver_verified=136,
            three_view_accepted_ids=prior_ids+C28['accepted_new_ids'],three_view_counts_by_variant={'minus_U':100,'minus_C':36},
            three_view_scope_limit='U100 complete; C30 IID Benign/F Flip/FedSA terminal replays complete, six IID S-DFA records partial; paired table review separate, no final test.')
        C28_stage.update(status='EXACT8_COMPLETE_STRICT_OFFSERVER_ROOT_ADOPTED',offserver_new_accepted=8,
            root_adoption_path=C28_path.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(C28_path),
            archive_members_verified=89,all_native_differences_zero=True,service_terminal='EXITED')
        state['active_services']=[s for s in state['active_services'] if s!='guardfed_celeba_mechanism_valid_C_after28']
    state['celeba_mechanism_v1']['C_after28_valid_replay']=C28_stage
C36_dir=ROOT/'tmp/celeba_mechanism_valid_C_after36_20261010/execution_candidate'
if (C36_dir/'ROOT_STARTUP_OBSERVATION.json').exists():
    C36_start=read(C36_dir/'ROOT_STARTUP_OBSERVATION.json')
    assert sha(C36_dir/'ROOT_STARTUP_OBSERVATION.json')=='8803f150d8d81ea23d27f5888a9d3735873c68c910fb2d79c50299a2714a73b4'
    assert C36_start['status']=='ROOT_C_AFTER36_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS'
    assert C36_start['execution_seal_sha256']=='b1e6467eac5b7218cda6af189c2ae2b655fb80780d48db05b45463aa9bdb578f'
    assert C36_start['deployment_receipt_sha256']==sha(C36_dir/'deployment_receipt.json')
    assert C36_start['original136_not_rerun'] and not C36_start['test_inference']
    C36_stage=dict(status='ACTUAL_STARTUP_VERIFIED_ACCEPTANCE_PENDING',selected_count=4,prior_three_view_models=136,
        offserver_new_accepted=0,execution_started=True,startup_root_proof_path=(C36_dir/'ROOT_STARTUP_OBSERVATION.json').relative_to(ROOT).as_posix(),
        startup_root_proof_sha256=sha(C36_dir/'ROOT_STARTUP_OBSERVATION.json'),CPU_affinity=list(range(112,120)),
        compute_threads=8,nice=10,IO='idle',CUDA_visible='',new_training=0,new_Full_inference=0,final_test=False)
    state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_mechanism_valid_C_after36']))
    C36_roots=list((C36_dir/'backups').glob('incremental_*/ROOT_ADOPTION_REVIEW.json'))
    if C36_roots:
        assert len(C36_roots)==1
        C36_path=C36_roots[0];C36=read(C36_path);C36_delta=C36_path.parent
        assert C36['status']=='ROOT_C_AFTER36_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
        assert (C36['prior_three_view_models'],C36['accepted_new'],C36['cumulative_three_view_models'])==(136,4,140)
        assert C36['original136_unchanged'] and C36['all_native_differences_zero'] and C36['new_Full_inference']==0 and not C36['test_inference']
        assert C36['prior136_root_adoption_sha256']=='cd197e57b42dda87bc239736e401cb5accd029313546533f6a48caf27950e1e0'
        assert C36['startup_observation_sha256']==sha(C36_dir/'ROOT_STARTUP_OBSERVATION.json')
        C36_arrays=read(C36_delta/'OFFSERVER_VERIFICATION.json')
        assert (C36_arrays['independent_metric_checks'],C36_arrays['independent_confusion_count_checks'],C36_arrays['prediction_rule_checks'])==(36,96,12)
        for key,name in [('archive_sha256','incremental_valid_three_views.tar.gz'),('offserver_verification_sha256','OFFSERVER_VERIFICATION.json'),('backup_receipt_sha256','backup_receipt.json')]:
            assert sha(C36_delta/name)==C36[key]
        prior_ids=state['celeba_mechanism_v1']['three_view_accepted_ids']
        assert len(prior_ids)==136 and not set(prior_ids)&set(C36['accepted_new_ids'])
        state['celeba_mechanism_v1'].update(three_view_new_models_accepted=140,three_view_new_models_offserver_verified=140,
            three_view_accepted_ids=prior_ids+C36['accepted_new_ids'],three_view_counts_by_variant={'minus_U':100,'minus_C':40},
            three_view_scope_limit='U100 complete; C40 IID Benign/F Flip/FedSA/S-DFA replays complete; paired table review separate, no final test.')
        C36_stage.update(status='EXACT4_COMPLETE_STRICT_OFFSERVER_ROOT_ADOPTED',offserver_new_accepted=4,
            root_adoption_path=C36_path.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(C36_path),
            archive_members_verified=C36['archive_members_verified'],all_native_differences_zero=True,service_terminal='EXITED')
        state['active_services']=[s for s in state['active_services'] if s!='guardfed_celeba_mechanism_valid_C_after36']
    state['celeba_mechanism_v1']['C_after36_valid_replay']=C36_stage
C47_dir=ROOT/'tmp/celeba_mechanism_valid_C_after40_20261010/execution_candidate'
if (C47_dir/'ROOT_STARTUP_OBSERVATION.json').exists():
    C47_start=read(C47_dir/'ROOT_STARTUP_OBSERVATION.json')
    assert C47_start['status']=='ROOT_C_AFTER40_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS'
    assert C47_start['execution_seal_sha256']=='1ac84b16f00c30ed3effd230f539824bb426540e66ec4981a53fdee676c50ec8'
    assert C47_start['deployment_receipt_sha256']==sha(C47_dir/'deployment_receipt.json')
    assert C47_start['original140_not_rerun'] and not C47_start['test_inference']
    C47_stage=dict(status='ACTUAL_STARTUP_VERIFIED_ACCEPTANCE_PENDING',selected_count=7,prior_three_view_models=140,
        offserver_new_accepted=0,execution_started=True,startup_root_proof_path=(C47_dir/'ROOT_STARTUP_OBSERVATION.json').relative_to(ROOT).as_posix(),
        startup_root_proof_sha256=sha(C47_dir/'ROOT_STARTUP_OBSERVATION.json'),CPU_affinity=list(range(112,120)),
        compute_threads=8,nice=10,IO='idle',CUDA_visible='',new_training=0,new_Full_inference=0,final_test=False)
    state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_mechanism_valid_C_after40']))
    C47_roots=list((C47_dir/'backups').glob('incremental_*/ROOT_ADOPTION_REVIEW.json'))
    if C47_roots:
        assert len(C47_roots)==1
        C47_path=C47_roots[0];C47=read(C47_path);C47_delta=C47_path.parent
        expected47=[f'minus_C_IID_Sp-DFA_seed{s}' for s in range(91001,91008)]
        assert C47['status']=='ROOT_C_AFTER40_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
        assert (C47['prior_three_view_models'],C47['accepted_new'],C47['cumulative_three_view_models'])==(140,7,147)
        assert C47['accepted_new_ids']==expected47
        assert C47['original140_unchanged'] and C47['all_native_differences_zero'] and C47['new_Full_inference']==0 and not C47['test_inference']
        assert C47['prior140_root_adoption_sha256']=='eb1f6c3120febb138e32af25484ac790cf962cd15d5ea9921140119bc5a3317a'
        assert C47['startup_observation_sha256']==sha(C47_dir/'ROOT_STARTUP_OBSERVATION.json')
        C47_arrays=read(C47_delta/'OFFSERVER_VERIFICATION.json')
        assert (C47_arrays['independent_metric_checks'],C47_arrays['independent_confusion_count_checks'],C47_arrays['prediction_rule_checks'])==(63,168,21)
        for key,name in [('archive_sha256','incremental_valid_three_views.tar.gz'),('offserver_verification_sha256','OFFSERVER_VERIFICATION.json'),('backup_receipt_sha256','backup_receipt.json')]:
            assert sha(C47_delta/name)==C47[key]
        prior_ids=state['celeba_mechanism_v1']['three_view_accepted_ids']
        assert len(prior_ids)==140 and not set(prior_ids)&set(expected47)
        state['celeba_mechanism_v1'].update(three_view_new_models_accepted=147,three_view_new_models_offserver_verified=147,
            three_view_accepted_ids=prior_ids+expected47,three_view_counts_by_variant={'minus_U':100,'minus_C':47},
            three_view_scope_limit='U100 complete; C40 four IID scenes complete plus seven IID Sp-DFA pairs; Sp-DFA incomplete and excluded from complete-scene means; no final test.')
        C47_stage.update(status='EXACT7_COMPLETE_STRICT_OFFSERVER_ROOT_ADOPTED',offserver_new_accepted=7,
            root_adoption_path=C47_path.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(C47_path),
            archive_members_verified=C47['archive_members_verified'],all_native_differences_zero=True,service_terminal='EXITED')
        state['active_services']=[s for s in state['active_services'] if s!='guardfed_celeba_mechanism_valid_C_after40']
    state['celeba_mechanism_v1']['C_after40_valid_replay']=C47_stage
C50_replay_dir=ROOT/'tmp/celeba_mechanism_valid_C_after47_20261010/execution_candidate'
if (C50_replay_dir/'ROOT_STARTUP_OBSERVATION.json').exists():
    C50_start=read(C50_replay_dir/'ROOT_STARTUP_OBSERVATION.json')
    assert C50_start['status']=='ROOT_C_AFTER47_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS'
    assert C50_start['execution_seal_sha256']=='b647c5a6759e709ab4c4fdd9d7ba74361901f98973d67f69176569b1a14c8ef5'
    assert C50_start['deployment_receipt_sha256']==sha(C50_replay_dir/'deployment_receipt.json')
    assert C50_start['original147_not_rerun'] and not C50_start['test_inference']
    C50_stage=dict(status='ACTUAL_STARTUP_VERIFIED_ACCEPTANCE_PENDING',selected_count=3,prior_three_view_models=147,
        offserver_new_accepted=0,execution_started=True,startup_root_proof_path=(C50_replay_dir/'ROOT_STARTUP_OBSERVATION.json').relative_to(ROOT).as_posix(),
        startup_root_proof_sha256=sha(C50_replay_dir/'ROOT_STARTUP_OBSERVATION.json'),CPU_affinity=list(range(112,120)),
        compute_threads=8,nice=10,IO='idle',CUDA_visible='',new_training=0,new_Full_inference=0,final_test=False)
    state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_mechanism_valid_C_after47']))
    C50_roots=list((C50_replay_dir/'backups').glob('incremental_*/ROOT_ADOPTION_REVIEW.json'))
    if C50_roots:
        assert len(C50_roots)==1
        C50_path=C50_roots[0];C50_proof=read(C50_path);C50_delta=C50_path.parent
        expected50=[f'minus_C_IID_Sp-DFA_seed{s}' for s in (91008,91009,91010)]
        assert C50_proof['status']=='ROOT_C_AFTER47_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
        assert (C50_proof['prior_three_view_models'],C50_proof['accepted_new'],C50_proof['cumulative_three_view_models'])==(147,3,150)
        assert C50_proof['accepted_new_ids']==expected50
        assert C50_proof['original147_unchanged'] and C50_proof['all_native_differences_zero'] and C50_proof['new_Full_inference']==0 and not C50_proof['test_inference']
        assert C50_proof['prior147_root_adoption_sha256']=='64732337f35f81bed49e40c60fa1d5229fb6a7557c45272505fee488c5ad20ea'
        assert C50_proof['startup_observation_sha256']==sha(C50_replay_dir/'ROOT_STARTUP_OBSERVATION.json')
        C50_arrays=read(C50_delta/'OFFSERVER_VERIFICATION.json')
        assert (C50_arrays['independent_metric_checks'],C50_arrays['independent_confusion_count_checks'],C50_arrays['prediction_rule_checks'])==(27,72,9)
        for key,name in [('archive_sha256','incremental_valid_three_views.tar.gz'),('offserver_verification_sha256','OFFSERVER_VERIFICATION.json'),('backup_receipt_sha256','backup_receipt.json')]:
            assert sha(C50_delta/name)==C50_proof[key]
        prior_ids=state['celeba_mechanism_v1']['three_view_accepted_ids']
        assert len(prior_ids)==147 and not set(prior_ids)&set(expected50)
        state['celeba_mechanism_v1'].update(three_view_new_models_accepted=150,three_view_new_models_offserver_verified=150,
            three_view_accepted_ids=prior_ids+expected50,three_view_counts_by_variant={'minus_U':100,'minus_C':50})
        C50_stage.update(status='EXACT3_COMPLETE_STRICT_OFFSERVER_ROOT_ADOPTED',offserver_new_accepted=3,
            root_adoption_path=C50_path.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(C50_path),
            archive_members_verified=C50_proof['archive_members_verified'],all_native_differences_zero=True,service_terminal='EXITED')
        state['active_services']=[s for s in state['active_services'] if s!='guardfed_celeba_mechanism_valid_C_after47']
    state['celeba_mechanism_v1']['C_after47_valid_replay']=C50_stage
C56_replay_dir=ROOT/'tmp/celeba_mechanism_valid_C_after50_20261010/execution_candidate'
if (C56_replay_dir/'ROOT_STARTUP_OBSERVATION.json').exists():
    C56_start=read(C56_replay_dir/'ROOT_STARTUP_OBSERVATION.json')
    assert C56_start['status']=='ROOT_C_AFTER50_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS'
    assert C56_start['execution_seal_sha256']=='073bfde67b2286f6e29fe5f6e2c1f7580f74465b4b0a4e42c583ab0332aa6595'
    assert C56_start['deployment_receipt_sha256']==sha(C56_replay_dir/'deployment_receipt.json')
    assert C56_start['original150_not_rerun'] and not C56_start['test_inference']
    C56_stage=dict(status='ACTUAL_STARTUP_VERIFIED_ACCEPTANCE_PENDING',selected_count=6,prior_three_view_models=150,
        offserver_new_accepted=0,execution_started=True,startup_root_proof_path=(C56_replay_dir/'ROOT_STARTUP_OBSERVATION.json').relative_to(ROOT).as_posix(),
        startup_root_proof_sha256=sha(C56_replay_dir/'ROOT_STARTUP_OBSERVATION.json'),CPU_affinity=list(range(112,120)),
        compute_threads=8,nice=10,IO='idle',CUDA_visible='',new_training=0,new_Full_inference=0,final_test=False)
    state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_mechanism_valid_C_after50']))
    C56_roots=list((C56_replay_dir/'backups').glob('incremental_*/ROOT_ADOPTION_REVIEW.json'))
    if C56_roots:
        assert len(C56_roots)==1
        C56_path=C56_roots[0];C56_proof=read(C56_path);C56_delta=C56_path.parent
        expected56=[f'minus_C_non-IID_Benign_seed{s}' for s in range(91001,91007)]
        assert C56_proof['status']=='ROOT_C_AFTER50_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
        assert (C56_proof['prior_three_view_models'],C56_proof['accepted_new'],C56_proof['cumulative_three_view_models'])==(150,6,156)
        assert C56_proof['accepted_new_ids']==expected56
        assert C56_proof['original150_unchanged'] and C56_proof['all_native_differences_zero'] and C56_proof['new_Full_inference']==0 and not C56_proof['test_inference']
        assert C56_proof['prior150_root_adoption_sha256']==state['celeba_mechanism_v1']['C_after47_valid_replay']['root_adoption_sha256']
        assert C56_proof['startup_observation_sha256']==sha(C56_replay_dir/'ROOT_STARTUP_OBSERVATION.json')
        C56_arrays=read(C56_delta/'OFFSERVER_VERIFICATION.json')
        assert (C56_arrays['independent_metric_checks'],C56_arrays['independent_confusion_count_checks'],C56_arrays['prediction_rule_checks'])==(54,144,18)
        for key,name in [('archive_sha256','incremental_valid_three_views.tar.gz'),('offserver_verification_sha256','OFFSERVER_VERIFICATION.json'),('backup_receipt_sha256','backup_receipt.json')]:
            assert sha(C56_delta/name)==C56_proof[key]
        prior_ids=state['celeba_mechanism_v1']['three_view_accepted_ids']
        assert len(prior_ids)==150 and not set(prior_ids)&set(expected56)
        state['celeba_mechanism_v1'].update(three_view_new_models_accepted=156,three_view_new_models_offserver_verified=156,
            three_view_accepted_ids=prior_ids+expected56,three_view_counts_by_variant={'minus_U':100,'minus_C':56})
        C56_stage.update(status='EXACT6_COMPLETE_STRICT_OFFSERVER_ROOT_ADOPTED',offserver_new_accepted=6,
            root_adoption_path=C56_path.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(C56_path),
            archive_members_verified=C56_proof['archive_members_verified'],all_native_differences_zero=True,service_terminal='EXITED')
        state['active_services']=[s for s in state['active_services'] if s!='guardfed_celeba_mechanism_valid_C_after50']
    state['celeba_mechanism_v1']['C_after50_valid_replay']=C56_stage
C60_replay_dir=ROOT/'tmp/celeba_mechanism_valid_C_after56_20261010/execution_candidate'
if (C60_replay_dir/'ROOT_STARTUP_OBSERVATION.json').exists():
    C60_start=read(C60_replay_dir/'ROOT_STARTUP_OBSERVATION.json')
    assert C60_start['status']=='ROOT_C_AFTER56_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS'
    assert C60_start['execution_seal_sha256']=='d1679c0bbd53bc66e4ea7ae792000d398efcafe5192bc5164ffd80e7a2eeb236'
    assert C60_start['deployment_receipt_sha256']==sha(C60_replay_dir/'deployment_receipt.json')
    assert C60_start['original156_not_rerun'] and not C60_start['test_inference']
    C60_stage=dict(status='ACTUAL_STARTUP_VERIFIED_ACCEPTANCE_PENDING',selected_count=4,prior_three_view_models=156,
        offserver_new_accepted=0,execution_started=True,startup_root_proof_path=(C60_replay_dir/'ROOT_STARTUP_OBSERVATION.json').relative_to(ROOT).as_posix(),
        startup_root_proof_sha256=sha(C60_replay_dir/'ROOT_STARTUP_OBSERVATION.json'),CPU_affinity=list(range(112,120)),
        compute_threads=8,nice=10,IO='idle',CUDA_visible='',new_training=0,new_Full_inference=0,final_test=False)
    state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_mechanism_valid_C_after56']))
    C60_roots=list((C60_replay_dir/'backups').glob('incremental_*/ROOT_ADOPTION_REVIEW.json'))
    if C60_roots:
        assert len(C60_roots)==1
        C60_path=C60_roots[0];C60_proof=read(C60_path);C60_delta=C60_path.parent
        assert sha(C60_path)=='21b7f5beac762bf808415685817a4027e9da66640a7ef0e4c7d4ac74eeb1f05e'
        assert (C60_proof['archive_members_verified'],C60_proof['content_members_verified'])==(61,60)
        expected56=[f'minus_C_non-IID_Benign_seed{s}' for s in range(91007,91011)]
        assert C60_proof['status']=='ROOT_C_AFTER56_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
        assert (C60_proof['prior_three_view_models'],C60_proof['accepted_new'],C60_proof['cumulative_three_view_models'])==(156,4,160)
        assert C60_proof['accepted_new_ids']==expected56
        assert C60_proof['original156_unchanged'] and C60_proof['all_native_differences_zero'] and C60_proof['new_Full_inference']==0 and not C60_proof['test_inference']
        assert C60_proof['prior156_root_adoption_sha256']==state['celeba_mechanism_v1']['C_after50_valid_replay']['root_adoption_sha256']
        assert C60_proof['startup_observation_sha256']==sha(C60_replay_dir/'ROOT_STARTUP_OBSERVATION.json')
        C60_arrays=read(C60_delta/'OFFSERVER_VERIFICATION.json')
        assert (C60_arrays['independent_metric_checks'],C60_arrays['independent_confusion_count_checks'],C60_arrays['prediction_rule_checks'])==(36,96,12)
        for key,name in [('archive_sha256','incremental_valid_three_views.tar.gz'),('offserver_verification_sha256','OFFSERVER_VERIFICATION.json'),('backup_receipt_sha256','backup_receipt.json')]:
            assert sha(C60_delta/name)==C60_proof[key]
        prior_ids=state['celeba_mechanism_v1']['three_view_accepted_ids']
        assert len(prior_ids)==156 and not set(prior_ids)&set(expected56)
        state['celeba_mechanism_v1'].update(three_view_new_models_accepted=160,three_view_new_models_offserver_verified=160,
            three_view_accepted_ids=prior_ids+expected56,three_view_counts_by_variant={'minus_U':100,'minus_C':60})
        C60_stage.update(status='EXACT4_COMPLETE_STRICT_OFFSERVER_ROOT_ADOPTED',offserver_new_accepted=4,
            root_adoption_path=C60_path.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(C60_path),
            archive_members_verified=C60_proof['archive_members_verified'],all_native_differences_zero=True,service_terminal='EXITED')
        state['active_services']=[s for s in state['active_services'] if s!='guardfed_celeba_mechanism_valid_C_after56']
    state['celeba_mechanism_v1']['C_after56_valid_replay']=C60_stage
C70_replay_dir=ROOT/'tmp/celeba_mechanism_valid_C_after60_20261010/execution_candidate'
if (C70_replay_dir/'ROOT_STARTUP_OBSERVATION.json').exists():
    C70_start=read(C70_replay_dir/'ROOT_STARTUP_OBSERVATION.json')
    assert sha(C70_replay_dir/'ROOT_STARTUP_OBSERVATION.json')=='dedd8aa9c83901a3415c509b1c6994e4495af1c7e810e0feff674c6f7e295e0e'
    assert C70_start['status']=='ROOT_C_AFTER60_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS'
    assert C70_start['execution_seal_sha256']=='12c1b612d9c9697f6a537344ebace5aaa7c1bf5d7bbe760a3866168adca89896'
    assert C70_start['deployment_receipt_sha256']==sha(C70_replay_dir/'deployment_receipt.json')
    assert C70_start['original160_not_rerun'] and not C70_start['test_inference']
    state['celeba_mechanism_v1']['C_after60_valid_replay']=dict(status='ACTUAL_STARTUP_VERIFIED_ACCEPTANCE_PENDING',
        selected_count=10,prior_three_view_models=160,offserver_new_accepted=0,execution_started=True,
        startup_root_proof_path=(C70_replay_dir/'ROOT_STARTUP_OBSERVATION.json').relative_to(ROOT).as_posix(),
        startup_root_proof_sha256=sha(C70_replay_dir/'ROOT_STARTUP_OBSERVATION.json'),
        CPU_affinity=list(range(112,120)),compute_threads=8,nice=10,IO='idle',CUDA_visible='',
        new_training=0,new_Full_inference=0,final_test=False)
    state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_mechanism_valid_C_after60']))
    C70_roots=list((C70_replay_dir/'backups').glob('incremental_*/ROOT_ADOPTION_REVIEW.json'))
    if C70_roots:
        assert len(C70_roots)==1
        C70_path=C70_roots[0];C70_proof=read(C70_path);C70_delta=C70_path.parent
        assert sha(C70_path)=='7e687f5a050aa0496b9a8c2b3606bd8787c138de6679e436dee3eb5b60df2dc8'
        expected60=[f'minus_C_non-IID_F Flip_seed{s}' for s in range(91001,91011)]
        assert C70_proof['status']=='ROOT_C_AFTER60_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
        assert (C70_proof['archive_members_verified'],C70_proof['content_members_verified'])==(103,102)
        assert (C70_proof['prior_three_view_models'],C70_proof['accepted_new'],C70_proof['cumulative_three_view_models'])==(160,10,170)
        assert C70_proof['accepted_new_ids']==expected60
        assert C70_proof['original160_unchanged'] and C70_proof['all_native_differences_zero'] and C70_proof['new_Full_inference']==0 and not C70_proof['test_inference']
        assert C70_proof['prior160_root_adoption_sha256']==state['celeba_mechanism_v1']['C_after56_valid_replay']['root_adoption_sha256']
        assert C70_proof['startup_observation_sha256']==sha(C70_replay_dir/'ROOT_STARTUP_OBSERVATION.json')
        C70_arrays=read(C70_delta/'OFFSERVER_VERIFICATION.json')
        assert (C70_arrays['independent_metric_checks'],C70_arrays['independent_confusion_count_checks'],C70_arrays['prediction_rule_checks'])==(90,240,30)
        for key,name in [('archive_sha256','incremental_valid_three_views.tar.gz'),('offserver_verification_sha256','OFFSERVER_VERIFICATION.json'),('backup_receipt_sha256','backup_receipt.json')]:
            assert sha(C70_delta/name)==C70_proof[key]
        prior_ids=state['celeba_mechanism_v1']['three_view_accepted_ids']
        assert len(prior_ids)==160 and not set(prior_ids)&set(expected60)
        state['celeba_mechanism_v1'].update(three_view_new_models_accepted=170,three_view_new_models_offserver_verified=170,
            three_view_accepted_ids=prior_ids+expected60,three_view_counts_by_variant={'minus_U':100,'minus_C':70})
        state['celeba_mechanism_v1']['C_after60_valid_replay'].update(status='EXACT10_COMPLETE_STRICT_OFFSERVER_ROOT_ADOPTED',offserver_new_accepted=10,
            root_adoption_path=C70_path.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(C70_path),
            archive_members_verified=103,all_native_differences_zero=True,service_terminal='EXITED')
        state['active_services']=[s for s in state['active_services'] if s!='guardfed_celeba_mechanism_valid_C_after60']
C80_replay_dir=ROOT/'tmp/celeba_mechanism_valid_C_after70_20261010/execution_candidate'
if (C80_replay_dir/'ROOT_STARTUP_OBSERVATION.json').exists():
    C80_start_path=C80_replay_dir/'ROOT_STARTUP_OBSERVATION.json'
    C80_start=read(C80_start_path)
    assert sha(C80_start_path)=='008106d198dccefe2b084e22cc35111c303601e027c9974b75da18971f4b64de'
    assert C80_start['status']=='ROOT_C_AFTER70_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS'
    assert C80_start['deployment_receipt_sha256']==sha(C80_replay_dir/'deployment_receipt.json')=='aed3cd4f02440026878728821066eb4fe298a0a48c10a39012acf7af6aab3f5e'
    assert C80_start['execution_seal_sha256']=='c93a5c605f7553bb336ba566a18bd87050f3d1e32d2b6320ee9ae15a51681224'
    assert C80_start['original170_not_rerun'] and not C80_start['test_inference']
    assert C80_start['new_training']==C80_start['new_Full_inference']==C80_start['scientific_offserver_new_accepted']==0
    assert C80_start['batch_failure'] is None
    assert all(p['cpus']==list(range(112,120)) and p['nice']==10 and p['io']=='idle'
        and p['environment']==dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1') for p in C80_start['processes'])
    C80_transport_path=ROOT/'tmp/celeba_mechanism_C_after70_root_review_20261010/ROOT_TRANSPORT_REVIEW.json'
    assert sha(C80_transport_path)=='98da473513e7ee2162803ad67a60ce263eda4a9c8745ea96f22f75cff9b44872'
    C80_error_path=C80_transport_path.with_name('TRANSPORT_REVIEW_INITIAL_READER_ERRORS.json')
    assert sha(C80_error_path)=='3bd2349e272afa6d2ab8bc8e3596720410a0ceb770dbf007b3ba93ea0b9565e4'
    C80_stage=dict(status='ACTUAL_STARTUP_VERIFIED_ACCEPTANCE_PENDING',selected_count=10,
        selected_ids=[f'minus_C_non-IID_FedSA_seed{s}' for s in range(91001,91011)],
        prior_three_view_models=170,offserver_new_accepted=0,execution_started=True,
        service='guardfed_celeba_mechanism_valid_C_after70',
        startup_root_proof_path=C80_start_path.relative_to(ROOT).as_posix(),startup_root_proof_sha256=sha(C80_start_path),
        deployment_receipt_sha256=C80_start['deployment_receipt_sha256'],execution_seal_sha256=C80_start['execution_seal_sha256'],
        source_review_sha256='bc73777fdba77ac5fba53203bf20993fef3087edc261a6c80f5e1d83f21f43e5',
        CPU_affinity=list(range(112,120)),compute_threads=8,nice=10,IO='idle',CUDA_visible='',
        new_training=0,new_Full_inference=0,final_test=False,
        transport_report_after_deployment_invocation=True,transport_review_path=C80_transport_path.relative_to(ROOT).as_posix(),
        transport_review_sha256=sha(C80_transport_path),preserved_local_reader_errors_path=C80_error_path.relative_to(ROOT).as_posix())
    C80_progress_paths=sorted(C80_replay_dir.glob('ROOT_PROGRESS_*.json'))
    C80_progress_paths=[p for p in C80_progress_paths if not p.name.endswith('.RAW.json')]
    if C80_progress_paths:
        C80_progress_path=C80_progress_paths[-1];C80_progress=read(C80_progress_path)
        assert C80_progress['source_startup_proof_sha256']==sha(C80_start_path)
        assert C80_progress['deployment_receipt_sha256']==C80_start['deployment_receipt_sha256']
        assert C80_progress['execution_seal_sha256']==C80_start['execution_seal_sha256']
        assert set(r['id'] for r in C80_progress['completed'])<=set(C80_stage['selected_ids'])
        C80_stage.update(observed_utc=C80_progress['utc'],observed_remote_completed=len(C80_progress['completed']),
            observed_service=C80_progress['service'],observed_process_count=len(C80_progress['processes']),
            progress_path=C80_progress_path.relative_to(ROOT).as_posix(),progress_sha256=sha(C80_progress_path))
    state['celeba_mechanism_v1']['C_after70_valid_replay']=C80_stage
    state['active_services']=list(dict.fromkeys(state['active_services']+[C80_stage['service']]))
    C80_roots=list((C80_replay_dir/'backups').glob('incremental_*/ROOT_ADOPTION_REVIEW.json'))
    if C80_roots:
        assert len(C80_roots)==1
        C80_path=C80_roots[0];C80_proof=read(C80_path);C80_delta=C80_path.parent
        assert sha(C80_path)=='3fc1e49e927a971a577d648dd9a7ff44ec7ac81552ea250026349d4f2e06d615'
        assert C80_proof['status']=='ROOT_C_AFTER70_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
        assert (C80_proof['archive_members_verified'],C80_proof['content_members_verified'])==(103,102)
        assert (C80_proof['prior_three_view_models'],C80_proof['accepted_new'],C80_proof['cumulative_three_view_models'])==(170,10,180)
        assert C80_proof['accepted_new_ids']==C80_stage['selected_ids']
        assert C80_proof['original170_unchanged'] and C80_proof['all_native_differences_zero'] and C80_proof['new_Full_inference']==0 and not C80_proof['test_inference']
        assert C80_proof['prior170_root_adoption_sha256']==state['celeba_mechanism_v1']['C_after60_valid_replay']['root_adoption_sha256']
        assert C80_proof['startup_observation_sha256']==sha(C80_start_path)
        C80_arrays=read(C80_delta/'OFFSERVER_VERIFICATION.json')
        assert (C80_arrays['independent_metric_checks'],C80_arrays['independent_confusion_count_checks'],C80_arrays['prediction_rule_checks'])==(90,240,30)
        for key,name in [('archive_sha256','incremental_valid_three_views.tar.gz'),('offserver_verification_sha256','OFFSERVER_VERIFICATION.json'),('backup_receipt_sha256','backup_receipt.json')]:
            assert sha(C80_delta/name)==C80_proof[key]
        prior_ids=state['celeba_mechanism_v1']['three_view_accepted_ids']
        assert len(prior_ids)==170 and not set(prior_ids)&set(C80_stage['selected_ids'])
        state['celeba_mechanism_v1'].update(three_view_new_models_accepted=180,three_view_new_models_offserver_verified=180,
            three_view_accepted_ids=prior_ids+C80_stage['selected_ids'],three_view_counts_by_variant={'minus_U':100,'minus_C':80})
        C80_stage.update(status='EXACT10_COMPLETE_STRICT_OFFSERVER_ROOT_ADOPTED',offserver_new_accepted=10,
            root_adoption_path=C80_path.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(C80_path),
            archive_members_verified=103,all_native_differences_zero=True,service_terminal='EXITED')
        state['active_services']=[s for s in state['active_services'] if s!=C80_stage['service']]
C30_dir=TRAIN/'celeba_mechanism_v1/three_view_C_three_scenes_20261009'
if (C30_dir/'ROOT_VERIFICATION.json').exists():
    C30=read(C30_dir/'ROOT_VERIFICATION.json')
    assert sha(C30_dir/'ROOT_VERIFICATION.json')=='2cce0519555efbff75f559875c4c1afe63e07d242cb8e3ebe6af644efc95961d'
    assert (C30['unique_records'],C30['paired_models'],C30['complete_scenes'],C30['mean_SD_scalars_recomputed'],C30['display_cells'],C30['count_metrics_recomputed'])==(60,30,3,486,243,540)
    assert C30['C8_root_adoption_sha256']==state['celeba_mechanism_v1']['C_after28_valid_replay']['root_adoption_sha256']
    assert sha(C30_dir/'snapshot/TABLES.md')==C30['display_sha256'] and sha(C30_dir/'snapshot/tables.json')==C30['tables_sha256']
    assert sha(ROOT/C30['independent_review_path'])==C30['independent_review_sha256']
    assert C30['original_two_scene40_records_exact'] and C30['original_two_scene324_statistics_exact'] and C30['original_two_scene162_cells_preserved']
    assert C30['excluded_partial_C_records']==6 and not C30['test'] and not C30['whole_rebuttal_complete']
    state['celeba_mechanism_v1']['C_three_view_three_scene_table']=dict(status=C30['status'],
        root_proof_path=(C30_dir/'ROOT_VERIFICATION.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(C30_dir/'ROOT_VERIFICATION.json'),
        table_path=C30['canonical_table'],paired_models=30,unique_records=60,complete_scenes=3,mean_SD_scalars=486,display_cells=243,
        count_metrics=540,excluded_partial_C_records=6,replay_devices=C30['replay_devices'],other_C_scenes_complete=False,final_test=False)
C40_dir=TRAIN/'celeba_mechanism_v1/three_view_C_four_scenes_20261010'
if (C40_dir/'ROOT_VERIFICATION.json').exists():
    C40=read(C40_dir/'ROOT_VERIFICATION.json')
    assert C40['status']=='ROOT_C40_FOUR_SCENE_THREE_VIEW_TABLES_ADOPTED'
    assert (C40['unique_records'],C40['paired_models'],C40['complete_scenes'],C40['mean_SD_scalars_recomputed'],C40['display_cells'],C40['count_metrics_recomputed'])==(80,40,4,648,324,720)
    assert C40['C4_root_adoption_sha256']==state['celeba_mechanism_v1']['C_after36_valid_replay']['root_adoption_sha256']=='eb1f6c3120febb138e32af25484ac790cf962cd15d5ea9921140119bc5a3317a'
    assert sha(C40_dir/'ACTUAL_FILES_SHA256.json')==C40['actual_delivery_seal_sha256']
    for name,pin in read(C40_dir/'ACTUAL_FILES_SHA256.json')['files'].items():
        assert sha(C40_dir/name)==pin['sha256'] and (C40_dir/name).stat().st_size==pin['bytes']
    assert sha(ROOT/C40['independent_review_path'])==C40['independent_review_sha256']
    assert C40['original_three_scene60_records_exact'] and C40['original_three_scene486_statistics_exact'] and C40['original_three_scene243_cells_preserved']
    assert C40['prior_S_DFA_six_original_record_bytes_preserved'] and not C40['test'] and not C40['whole_rebuttal_complete']
    state['celeba_mechanism_v1']['C_three_view_four_scene_table']=dict(status=C40['status'],
        root_proof_path=(C40_dir/'ROOT_VERIFICATION.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(C40_dir/'ROOT_VERIFICATION.json'),
        table_path=C40['canonical_table'],paired_models=40,unique_records=80,complete_scenes=4,mean_SD_scalars=648,display_cells=324,
        count_metrics=720,replay_devices=C40['replay_devices'],other_C_scenes_complete=False,final_test=False)
    state['celeba_mechanism_v1']['three_view_scope_limit']='U100 ten scenes plus C40 four IID scenes independently adopted; remaining six C scenes and six other variants incomplete; Full reused; no final test.'
    if state['celeba_mechanism_v1'].get('C_after40_valid_replay',{}).get('offserver_new_accepted')==7:
        state['celeba_mechanism_v1']['three_view_scope_limit']+=' Seven additional IID Sp-DFA pairs accepted separately; incomplete 7/10 and excluded from complete-scene means.'
C50_table_dir=TRAIN/'celeba_mechanism_v1/three_view_C_five_scenes_20261010'
if (C50_table_dir/'ROOT_VERIFICATION.json').exists():
    C50_table=read(C50_table_dir/'ROOT_VERIFICATION.json')
    assert C50_table['status']=='ROOT_C50_FIVE_SCENE_THREE_VIEW_TABLES_ADOPTED'
    assert (C50_table['unique_records'],C50_table['paired_models'],C50_table['complete_scenes'],C50_table['mean_SD_scalars_recomputed'],C50_table['display_cells'],C50_table['count_metrics_recomputed'],C50_table['cross_scene_mean_SD_scalars_recomputed'])==(100,50,5,810,405,900,162)
    assert C50_table['C3_root_adoption_sha256']==state['celeba_mechanism_v1']['C_after47_valid_replay']['root_adoption_sha256']
    assert sha(C50_table_dir/'ACTUAL_FILES_SHA256.json')==C50_table['actual_delivery_seal_sha256']
    for name,pin in read(C50_table_dir/'ACTUAL_FILES_SHA256.json')['files'].items():
        assert sha(C50_table_dir/name)==pin['sha256'] and (C50_table_dir/name).stat().st_size==pin['bytes']
    assert sha(ROOT/C50_table['independent_review_path'])==C50_table['independent_review_sha256']
    assert C50_table['original_four_scene80_records_exact'] and C50_table['original_four_scene648_statistics_exact'] and C50_table['original_four_scene324_cells_preserved']
    assert not C50_table['test'] and not C50_table['whole_rebuttal_complete']
    state['celeba_mechanism_v1']['C_three_view_five_scene_table']=dict(status=C50_table['status'],
        root_proof_path=(C50_table_dir/'ROOT_VERIFICATION.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(C50_table_dir/'ROOT_VERIFICATION.json'),
        table_path=C50_table['canonical_table'],paired_models=50,unique_records=100,complete_scenes=5,mean_SD_scalars=810,display_cells=405,
        count_metrics=900,cross_scene_mean_SD_scalars=162,replay_devices=C50_table['replay_devices'],other_C_scenes_complete=False,final_test=False)
    state['celeba_mechanism_v1']['three_view_scope_limit']='U100 ten scenes plus C50 five IID scenes independently adopted; five non-IID C scenes and six other variants incomplete; Full reused; no final test.'
if state['celeba_mechanism_v1'].get('C_after56_valid_replay',{}).get('offserver_new_accepted')==4:
    state['celeba_mechanism_v1']['three_view_scope_limit']='U100 and C60 replays adopted; C five-IID-scene table retained, non-IID Benign ten pairs await separate table adoption; four other non-IID C scenes and six other variants incomplete; no final test.'
C60_table_dir=TRAIN/'celeba_mechanism_v1/three_view_C_six_scenes_20261010'
if (C60_table_dir/'ROOT_VERIFICATION.json').exists():
    C60_table=read(C60_table_dir/'ROOT_VERIFICATION.json')
    assert sha(C60_table_dir/'ROOT_VERIFICATION.json')=='f2465c5e81291deca92535adcd0b0a29c49b3f76c2330927aa3db50925a9c742'
    assert C60_table['status']=='ROOT_C60_SIX_SCENE_THREE_VIEW_TABLES_ADOPTED'
    assert tuple(C60_table[k] for k in ('unique_records','paired_models','complete_scenes','mean_SD_scalars_recomputed','display_cells','count_metrics_recomputed','confusion_count_checks','cross_scene_mean_SD_scalars_recomputed'))==(120,60,6,972,486,1080,2880,162)
    assert C60_table['C4_root_adoption_sha256']==state['celeba_mechanism_v1']['C_after56_valid_replay']['root_adoption_sha256']
    assert sha(C60_table_dir/'ACTUAL_FILES_SHA256.json')==C60_table['actual_delivery_seal_sha256']
    for name,pin in read(C60_table_dir/'ACTUAL_FILES_SHA256.json')['files'].items():
        assert sha(C60_table_dir/name)==pin['sha256'] and (C60_table_dir/name).stat().st_size==pin['bytes']
    assert sha(ROOT/C60_table['independent_review_path'])==C60_table['independent_review_sha256']
    assert all(C60_table[k] for k in ('original_five_scene100_records_exact','original_five_scene810_statistics_exact','original_five_scene405_cells_preserved','original_IID_aggregate162_scalars_bytes_exact','all_negative_results_retained'))
    assert C60_table['IID_complete_scenes']==5 and C60_table['nonIID_complete_scenes']==['Benign'] and not C60_table['full_nonIID_coverage']
    assert C60_table['seed_panels']==[10,9,6] and C60_table['aggregate_scope']=='Original five IID scenes only; no imbalanced six-scene mean'
    assert C60_table['new_CNN']==C60_table['new_training']==C60_table['new_Full_inference']==0
    assert not C60_table['test'] and not C60_table['whole_rebuttal_complete'] and not C60_table['other_C_scenes_complete'] and C60_table['primary_endpoint']=='PENDING_AUTHOR'
    state['celeba_mechanism_v1']['C_three_view_six_scene_table']=dict(status=C60_table['status'],
        root_proof_path=(C60_table_dir/'ROOT_VERIFICATION.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(C60_table_dir/'ROOT_VERIFICATION.json'),
        table_path=C60_table['canonical_table'],paired_models=60,unique_records=120,complete_scenes=6,mean_SD_scalars=972,display_cells=486,
        count_metrics=1080,confusion_count_checks=2880,cross_scene_mean_SD_scalars=162,aggregate_scope=C60_table['aggregate_scope'],
        replay_devices=C60_table['replay_devices'],IID_complete_scenes=5,nonIID_complete_scenes=['Benign'],other_C_scenes_complete=False,final_test=False,incorporated_into_full_rebuttal=False)
    state['celeba_mechanism_v1']['three_view_scope_limit']='U100 ten scenes plus C60 five IID scenes and non-IID Benign independently adopted; four other non-IID C scenes and six other variants incomplete; IID-only seed-first aggregate retained; C60 table separate from C50 author-review prose; no final test.'
C30_reply_dir=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_C30_addendum_20261009'
if (C30_reply_dir/'ROOT_REVIEW.json').exists():
    C30_reply=read(C30_reply_dir/'ROOT_REVIEW.json')
    assert sha(C30_reply_dir/'ROOT_REVIEW.json')=='e5ae4ea68ce7b41d0f98b918af8873b167d6d5cff758f0a03221f6356911ff7e'
    assert C30_reply['root_table_sha256']==state['celeba_mechanism_v1']['C_three_view_three_scene_table']['root_proof_sha256']
    assert sha(C30_reply_dir/'FILES_SHA256.json')==C30_reply['source_seal_sha256']
    for name,pin in read(C30_reply_dir/'FILES_SHA256.json')['files'].items():assert sha(C30_reply_dir/name)==pin['sha256']
    assert C30_reply['author_review_only'] and not C30_reply['manuscript_applied'] and C30_reply['old_24_comment_draft_unchanged']
    state['latest_rebuttal_addendum']=dict(status=C30_reply['status'],
        entry=(C30_reply_dir/'C30_REVIEWER_ADDENDUM.md').relative_to(ROOT).as_posix(),
        manuscript_candidate=(C30_reply_dir/'C30_MANUSCRIPT_INSERTIONS.md').relative_to(ROOT).as_posix(),
        root_proof_path=(C30_reply_dir/'ROOT_REVIEW.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(C30_reply_dir/'ROOT_REVIEW.json'),
        complete_C_scenes=3,scalar_pointer_checks=98,display_cells_checked=49,scope_fact_checks=25,links_checked=9,
        author_review_only=True,whole_rebuttal_complete=False,manuscript_applied=False)
C40_reply_dir=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_C40_addendum_20261010'
if (C40_reply_dir/'ROOT_REVIEW.json').exists():
    C40_reply=read(C40_reply_dir/'ROOT_REVIEW.json')
    assert C40_reply['status']=='ROOT_C40_AUTHOR_REVIEW_ADDENDUM_QUOTED_VALUES_SCOPE_AND_SOURCE_PINS_PASS'
    assert C40_reply['root_table_sha256']==state['celeba_mechanism_v1']['C_three_view_four_scene_table']['root_proof_sha256']=='eb08dbc328339e8293133bbd35f60e627ca70b7436ec3b20871bc149e6047172'
    assert sha(C40_reply_dir/'FILES_SHA256.json')==C40_reply['source_seal_sha256']
    for name,pin in read(C40_reply_dir/'FILES_SHA256.json')['files'].items():assert sha(C40_reply_dir/name)==pin['sha256']
    assert C40_reply['author_review_only'] and not C40_reply['manuscript_applied'] and C40_reply['old_24_comment_draft_unchanged']
    assert C40_reply['complete_C_scenes']==4 and not C40_reply['final_test'] and not C40_reply['whole_rebuttal_complete']
    state['latest_rebuttal_addendum']=dict(status=C40_reply['status'],
        entry=(C40_reply_dir/'C40_REVIEWER_ADDENDUM.md').relative_to(ROOT).as_posix(),
        manuscript_candidate=(C40_reply_dir/'C40_MANUSCRIPT_INSERTIONS.md').relative_to(ROOT).as_posix(),
        root_proof_path=(C40_reply_dir/'ROOT_REVIEW.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(C40_reply_dir/'ROOT_REVIEW.json'),
        complete_C_scenes=4,scalar_pointer_checks=C40_reply['scalar_pointer_checks'],display_cells_checked=C40_reply['display_cells_checked'],
        scope_fact_checks=C40_reply['scope_fact_checks'],links_checked=C40_reply['links_checked'],
        author_review_only=True,whole_rebuttal_complete=False,manuscript_applied=False)
C50_reply_dir=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_C50_update_20261010'
if (C50_reply_dir/'ROOT_REVIEW.json').exists():
    C50_reply=read(C50_reply_dir/'ROOT_REVIEW.json')
    assert sha(C50_reply_dir/'ROOT_REVIEW.json')=='cb57e51f7ba426583f7975a78078c79b59149e475e63901ba4079c218e1354ff'
    assert C50_reply['status']=='ROOT_C50_AUTHOR_REVIEW_UPDATE_QUOTED_VALUES_SCOPE_AND_SOURCE_PINS_PASS'
    assert C50_reply['root_table_sha256']==state['celeba_mechanism_v1']['C_three_view_five_scene_table']['root_proof_sha256']
    assert sha(C50_reply_dir/'FILES_SHA256.json')==C50_reply['source_seal_sha256']
    for name,pin in read(C50_reply_dir/'FILES_SHA256.json')['files'].items():assert sha(C50_reply_dir/name)==pin['sha256']
    assert C50_reply['author_review_only'] and not C50_reply['manuscript_applied'] and C50_reply['old_24_comment_draft_unchanged']
    assert C50_reply['complete_C_scenes']==5 and not C50_reply['final_test'] and not C50_reply['whole_rebuttal_complete']
    state['latest_rebuttal_addendum']=dict(status=C50_reply['status'],
        entry=(C50_reply_dir/'C50_REVIEWER_ADDENDUM.md').relative_to(ROOT).as_posix(),
        manuscript_candidate=(C50_reply_dir/'C50_MANUSCRIPT_INSERTIONS.md').relative_to(ROOT).as_posix(),
        root_proof_path=(C50_reply_dir/'ROOT_REVIEW.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(C50_reply_dir/'ROOT_REVIEW.json'),
        complete_C_scenes=5,scalar_pointer_checks=C50_reply['scalar_pointer_checks'],display_cells_checked=C50_reply['display_cells_checked'],
        scope_fact_checks=C50_reply['scope_fact_checks'],links_checked=C50_reply['links_checked'],
        author_review_only=True,whole_rebuttal_complete=False,manuscript_applied=False)
rebuttal_C50_dir=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C50_20261010'
if (rebuttal_C50_dir/'ROOT_REVIEW.json').exists():
    writing=read(rebuttal_C50_dir/'ROOT_REVIEW.json')
    assert sha(rebuttal_C50_dir/'ROOT_REVIEW.json')=='64b083fb61cb8bff17304031af01a0d692d03fdc16e956de0dcd154f5a524691'
    assert writing['status']=='ROOT_C50_COMPLETE_AUTHOR_REVIEW_TEXT_REVERSIBLE_DELTA_AND_SOURCE_POINTERS_PASS'
    assert sha(rebuttal_C50_dir/'FILES_SHA256.json')==writing['source_seal_sha256']
    for name,pin in read(rebuttal_C50_dir/'FILES_SHA256.json')['files'].items():
        assert sha(rebuttal_C50_dir/name)==pin['sha256'] and (rebuttal_C50_dir/name).stat().st_size==pin['bytes']
    assert sha(rebuttal_C50_dir/'rebuttal_integrated_20261009.md')==writing['rebuttal_sha256']
    assert sha(rebuttal_C50_dir/'manuscript_insertions_integrated_20261009.md')==writing['insertions_sha256']
    assert writing['root_C50_table_sha256']==state['celeba_mechanism_v1']['C_three_view_five_scene_table']['root_proof_sha256']
    assert writing['root_C50_short_update_sha256']==state['latest_rebuttal_addendum']['root_proof_sha256']
    assert (writing['original_comments_verbatim'],writing['complete_C_scenes'],writing['prior_documents_reverse_diff_exact'])==(24,5,2)
    assert writing['author_review_only'] and not writing['manuscript_applied'] and not writing['final_test'] and not writing['whole_rebuttal_complete']
    state['latest_rebuttal_draft'].update(status=writing['status'],
        entry=(rebuttal_C50_dir/'rebuttal_integrated_20261009.md').relative_to(ROOT).as_posix(),
        manuscript_candidate=(rebuttal_C50_dir/'manuscript_insertions_integrated_20261009.md').relative_to(ROOT).as_posix(),
        root_proof_path=(rebuttal_C50_dir/'ROOT_REVIEW.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(rebuttal_C50_dir/'ROOT_REVIEW.json'),
        source_seal_sha256=writing['source_seal_sha256'],complete_C_scenes=5,new_C_scalar_pointer_checks=writing['C_scalar_pointer_checks'],
        new_C_scope_environment_checks=writing['C_scope_fact_pointer_checks'],links_checked=writing['links_checked'],changed_passages=writing['changed_spans'],
        author_review_only=True,manuscript_applied=False,whole_rebuttal_complete=False)
rebuttal_C60_dir=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C60_20261010'
if (rebuttal_C60_dir/'ROOT_REVIEW.json').exists():
    writing=read(rebuttal_C60_dir/'ROOT_REVIEW.json')
    assert sha(rebuttal_C60_dir/'ROOT_REVIEW.json')=='b2e3958ed4000fe0365929c94408c411cd47c70e637c5151bb40603ad097098a'
    assert writing['status']=='ROOT_C60_COMPLETE_AUTHOR_REVIEW_TEXT_REVERSIBLE_DELTA_AND_SOURCE_POINTERS_PASS'
    assert sha(rebuttal_C60_dir/'FILES_SHA256.json')==writing['source_seal_sha256']=='e8e67d7e8b63e50b14b44a57c09fcad749824f7647f8916f1bd20e9ea893f3ac'
    C60_files=read(rebuttal_C60_dir/'FILES_SHA256.json')['files'];assert len(C60_files)==11
    for name,pin in C60_files.items():
        assert sha(rebuttal_C60_dir/name)==pin['sha256'] and (rebuttal_C60_dir/name).stat().st_size==pin['bytes']
    assert sha(rebuttal_C60_dir/'rebuttal_integrated_20261009.md')==writing['rebuttal_sha256']=='ee946d604ff56ed047071a53da12ff9ad9a23c32f9f7b3a3c5a83feeaabdc200'
    assert sha(rebuttal_C60_dir/'manuscript_insertions_integrated_20261009.md')==writing['insertions_sha256']=='200ff6fa300f78bdaf9ae177c601b76343349973e758b9a1ce9fcbb9db20aa48'
    assert writing['root_C60_table_sha256']==state['celeba_mechanism_v1']['C_three_view_six_scene_table']['root_proof_sha256']=='f2465c5e81291deca92535adcd0b0a29c49b3f76c2330927aa3db50925a9c742'
    assert writing['prior_C50_reply_root_sha256']==state['latest_rebuttal_draft']['root_proof_sha256']=='64b083fb61cb8bff17304031af01a0d692d03fdc16e956de0dcd154f5a524691'
    assert tuple(writing[k] for k in ('original_comments_verbatim','complete_C_scenes','prior_documents_reverse_diff_exact','forward_diff_exact','changed_spans','C_scalar_pointer_checks','new_numeric_cells','direction_checks','links_checked'))==(24,6,2,2,10,36,18,27,50)
    assert writing['remaining_nonIID_C_scenes']==4 and writing['remaining_other_image_controls']==6
    assert writing['original_numeric_displays_preserved'] and writing['all_negative_results_retained'] and writing['author_review_only']
    assert not writing['manuscript_applied'] and not writing['final_test'] and not writing['whole_rebuttal_complete'] and writing['primary_endpoint']=='PENDING_AUTHOR'
    assert writing['new_inference']==writing['new_training']==0
    state['prior_rebuttal_C50_draft']=dict(state['latest_rebuttal_draft'])
    state['latest_rebuttal_draft'].update(status=writing['status'],
        entry=(rebuttal_C60_dir/'rebuttal_integrated_20261009.md').relative_to(ROOT).as_posix(),
        manuscript_candidate=(rebuttal_C60_dir/'manuscript_insertions_integrated_20261009.md').relative_to(ROOT).as_posix(),
        root_proof_path=(rebuttal_C60_dir/'ROOT_REVIEW.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(rebuttal_C60_dir/'ROOT_REVIEW.json'),
        source_seal_sha256=writing['source_seal_sha256'],complete_C_scenes=6,new_C_scalar_pointer_checks=36,new_C_numeric_cells=18,
        original_comments_verbatim=24,direction_checks=27,links_checked=50,changed_passages=10,
        author_review_only=True,manuscript_applied=False,final_test=False,whole_rebuttal_complete=False)
    state['latest_rebuttal_draft'].pop('new_C_scope_environment_checks',None)
    state['celeba_mechanism_v1']['C_three_view_six_scene_table']['incorporated_into_full_rebuttal']=True
    state['celeba_mechanism_v1']['three_view_scope_limit']=state['celeba_mechanism_v1']['three_view_scope_limit'].replace('C60 table separate from C50 author-review prose','C60 evidence incorporated into the full author-review response; submitted manuscript not applied')
    semantic_path=rebuttal_C60_dir/'semantic_review/ROOT_SEMANTIC_REVIEW.json'
    if semantic_path.exists():
        assert sha(semantic_path)=='db60632721bdf3ba8da18589ab095566d45f6fbedb43c6a026867ea451b40104'
        semantic=read(semantic_path)
        assert semantic['status']=='INDEPENDENT_C60_AFFECTED_PROSE_SEMANTIC_PASS_NO_CANONICAL_EDIT' and not semantic['findings'] and not semantic['required_corrections']
        state['latest_rebuttal_draft']['C60_affected_prose_semantic_review']=dict(proof_path=semantic_path.relative_to(ROOT).as_posix(),proof_sha256=sha(semantic_path),topics_checked=8,scope='Ten introduced or affected C60 spans and related reviewer blocks only; no numerical re-audit or submission-readiness claim')
C70_table_dir=TRAIN/'celeba_mechanism_v1/three_view_C_seven_scenes_20261010'
if (C70_table_dir/'ROOT_VERIFICATION.json').exists():
    C70=read(C70_table_dir/'ROOT_VERIFICATION.json')
    assert sha(C70_table_dir/'ROOT_VERIFICATION.json')=='abab0188adfa2d857316b23a282fd104c50dd282ca00ffed628b78b2accaea70'
    assert C70['status']=='ROOT_C70_SEVEN_SCENE_THREE_VIEW_TABLES_ADOPTED'
    assert sha(C70_table_dir/'ACTUAL_FILES_SHA256.json')=='c06dd986ecbb62f6a7d3cbf0ac307900404de4204eb5ae1511cf5deb881690dc'
    C70_files=read(C70_table_dir/'ACTUAL_FILES_SHA256.json')['files'];assert len(C70_files)==35
    for name,pin in C70_files.items():
        assert sha(C70_table_dir/name)==pin['sha256'] and (C70_table_dir/name).stat().st_size==pin['bytes']
    assert C70['C10_root_adoption_sha256']==state['celeba_mechanism_v1']['C_after60_valid_replay']['root_adoption_sha256']
    assert (C70['unique_records'],C70['paired_models'],C70['complete_scenes'],C70['mean_SD_scalars_recomputed'],C70['display_cells'],C70['count_metrics_recomputed'],C70['confusion_count_checks'])==(140,70,7,1134,567,1260,3360)
    assert all(C70[k] for k in ('old120_records_preserved','old972_scalars_preserved','old486_cells_preserved','old162_IID_aggregate_bytes_exact','all_negative_results_retained'))
    assert C70['new_CNN']==C70['new_training']==C70['new_Full_inference']==0 and not C70['test'] and not C70['whole_rebuttal_complete']
    state['celeba_mechanism_v1']['C_three_view_seven_scene_table']=dict(status=C70['status'],
        root_proof_path=(C70_table_dir/'ROOT_VERIFICATION.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(C70_table_dir/'ROOT_VERIFICATION.json'),
        actual_seal_sha256=sha(C70_table_dir/'ACTUAL_FILES_SHA256.json'),table_path=C70['canonical_table'],paired_models=70,unique_records=140,complete_scenes=7,
        mean_SD_scalars=1134,display_cells=567,count_metrics=1260,confusion_count_checks=3360,cross_scene_mean_SD_scalars=162,aggregate_scope=C70['aggregate_scope'],
        replay_devices=C70['replay_devices'],IID_complete_scenes=5,nonIID_complete_scenes=['Benign','F Flip'],other_C_scenes_complete=False,final_test=False,incorporated_into_full_rebuttal=False)
    state['celeba_mechanism_v1']['three_view_scope_limit']='U100 ten scenes and C70 seven scenes independently adopted; three non-IID C scenes and six variants remain incomplete. Original five-IID-scene aggregate only. Full author-review response incorporates C60; C70 separate table not yet incorporated; submitted manuscript not applied; no final test.'
C80_table_dir=TRAIN/'celeba_mechanism_v1/three_view_C_eight_scenes_20261010'
if (C80_table_dir/'ROOT_VERIFICATION.json').exists():
    C80=read(C80_table_dir/'ROOT_VERIFICATION.json')
    assert sha(C80_table_dir/'ROOT_VERIFICATION.json')=='71105f39c6345efb3706fe538a51686f80ad9e8a89f5601a04b349ebcc76e487'
    assert C80['status']=='ROOT_C80_EIGHT_SCENE_THREE_VIEW_TABLES_ADOPTED'
    assert sha(C80_table_dir/'ACTUAL_FILES_SHA256.json')=='a456c192c01d37d4ad577e2284380d3dec1290bb09e93d6dfc5922d3cd5749cf'
    C80_files=read(C80_table_dir/'ACTUAL_FILES_SHA256.json')['files'];assert len(C80_files)==31
    for name,pin in C80_files.items():
        assert sha(C80_table_dir/name)==pin['sha256'] and (C80_table_dir/name).stat().st_size==pin['bytes']
    assert C80['C10_root_adoption_sha256']==state['celeba_mechanism_v1']['C_after70_valid_replay']['root_adoption_sha256']
    assert (C80['unique_records'],C80['paired_models'],C80['complete_scenes'],C80['mean_SD_scalars_recomputed'],C80['display_cells'],C80['count_metrics_recomputed'],C80['confusion_count_checks'],C80['paired_seed_metric_checks'])==(160,80,8,1296,648,1440,3840,720)
    assert all(C80[k] for k in ('old140_records_preserved','old1134_scalars_preserved','old567_cells_preserved','old162_IID_aggregate_bytes_exact','all_negative_results_retained'))
    assert C80['new_CNN']==C80['new_training']==C80['new_Full_inference']==0 and not C80['test'] and not C80['whole_rebuttal_complete']
    state['celeba_mechanism_v1']['C_three_view_eight_scene_table']=dict(status=C80['status'],
        root_proof_path=(C80_table_dir/'ROOT_VERIFICATION.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(C80_table_dir/'ROOT_VERIFICATION.json'),
        actual_seal_sha256=sha(C80_table_dir/'ACTUAL_FILES_SHA256.json'),table_path=C80['canonical_table'],paired_models=80,unique_records=160,complete_scenes=8,
        mean_SD_scalars=1296,display_cells=648,count_metrics=1440,confusion_count_checks=3840,cross_scene_mean_SD_scalars=162,aggregate_scope=C80['aggregate_scope'],
        replay_devices=C80['replay_devices'],IID_complete_scenes=5,nonIID_complete_scenes=['Benign','F Flip','FedSA'],other_C_scenes_complete=False,final_test=False,incorporated_into_full_rebuttal=False)
    state['celeba_mechanism_v1']['three_view_scope_limit']='U100 ten scenes and C80 eight scenes independently adopted; two non-IID C scenes and six variants remain incomplete. Original five-IID-scene aggregate only. Full author-review response incorporates C60; C70/C80 separate tables not yet incorporated; submitted manuscript not applied; no final test.'
FL96_base=ROOT/'tmp/celeba_flgmm_fullcoverage_incremental_20261009'
if (FL96_base/'LATEST_BACKUP.json').exists():
    FL96_latest=read(FL96_base/'LATEST_BACKUP.json')
    FL96_first_path=FL96_base/'attempt_20261009T210943751719Z/ROOT_ADOPTION_REVIEW.json'
    FL96_root_path=FL96_first_path;FL96_root=read(FL96_root_path)
    assert sha(FL96_root_path)=='9663fe584e27eadcd6b81260e676e4a37a47331d6e31e6098c3cb2c407ba8a65'
    assert FL96_root['status']=='ROOT_FL96_FIRST_DELTA_ARCHIVE_SOURCE_CHECKPOINT_AND_ORIGINAL_STRICT_BINDING_PASS'
    assert FL96_root['accepted_new']==FL96_root['accepted_total']==1
    assert FL96_root['source_package_sha256']==state['flgmm_fullcoverage_v2_20261009']['package_sha256']
    FL96_batch=FL96_root_path.parent/'batch'
    assert sha(FL96_batch/'accepted_delta.tar.gz')==FL96_root['archive_sha256']
    assert sha(FL96_batch/'OFFSERVER_ACCEPTANCE.json')==FL96_root['offserver_acceptance_sha256']
    state['flgmm_fullcoverage_v2_20261009'].update(new_accepted=1,accepted_new_ids=FL96_root['accepted_new_ids'],
        coverage_verified_cells_including_reuse=5,latest_backup=FL96_latest,archive_members_verified=17,
        first_full70_root_adoption_path=FL96_root_path.relative_to(ROOT).as_posix(),first_full70_root_adoption_sha256=sha(FL96_root_path))
    FL96_current_path=ROOT/FL96_latest['root_adoption_path'];FL96_current=read(FL96_current_path)
    assert sha(FL96_current_path)==FL96_latest['root_adoption_sha256']
    if FL96_latest['accepted_total'] in (28,32,38,44,54,57):
        current_count=FL96_latest['accepted_total']
        assert sha(FL96_current_path) == {28:'e51007c549f8cc95970ce06cb61b4a477c11eff5d00e05a9aeac9ffb30dd449c',32:'ae1e65bf764f3ed3e6657e95fa3798617788c8659ac30eb542035a0031663cf3',38:'5e95872f3216a9842e95c534ba87629b2211c298d72a95d810bdaf9428021051',44:'f61a7fa480a62a9394d67a64533568b43e6491ee01d8e1f02005f35a42b6a510',54:'ead149673512a8c900d570d1f6d7c57e88c0608eb1dc0fb3a5c477d542e5a6a6',57:'66f8b578393b76fb8402a2f3c3d29ecbaf3f31aea4459371876c0f9f07e363ee'}[current_count]
        assert FL96_current['status'] == 'ROOT_FL96_LINKED_DELTA_ARCHIVE_SOURCE_CHECKPOINT_AND_ORIGINAL_STRICT_BINDING_PASS'
        previous = read(ROOT/FL96_current['previous_root_adoption_path'])
        assert sha(ROOT/FL96_current['previous_root_adoption_path']) == FL96_current['previous_root_adoption_sha256'] == {28:'31a1e0d16f14a1855acad3654d656ccb84377b7e6bd8fc3d9c10bf560658c8bc',32:'e51007c549f8cc95970ce06cb61b4a477c11eff5d00e05a9aeac9ffb30dd449c',38:'ae1e65bf764f3ed3e6657e95fa3798617788c8659ac30eb542035a0031663cf3',44:'5e95872f3216a9842e95c534ba87629b2211c298d72a95d810bdaf9428021051',54:'f61a7fa480a62a9394d67a64533568b43e6491ee01d8e1f02005f35a42b6a510',57:'ead149673512a8c900d570d1f6d7c57e88c0608eb1dc0fb3a5c477d542e5a6a6'}[current_count]
        assert (FL96_current['accepted_before'],FL96_current['accepted_new'],FL96_current['accepted_total']) == {28:(22,6,28),32:(28,4,32),38:(32,6,38),44:(38,6,44),54:(44,10,54),57:(54,3,57)}[current_count]
        batch = Path(FL96_current['archive_local_path']).parent
        off = read(batch/'OFFSERVER_ACCEPTANCE.json')
        assert sha(batch/'OFFSERVER_ACCEPTANCE.json') == FL96_current['offserver_acceptance_sha256'] == FL96_latest['next_collector_previous_sha256']
        assert sha(batch/'accepted_delta.tar.gz') == FL96_current['archive_sha256']
        assert off['accepted_job_ids'] == FL96_current['accepted_job_ids'] == previous['accepted_job_ids']+FL96_current['accepted_new_ids']
        assert len(set(off['accepted_job_ids'])) == current_count and FL96_current['source_package_sha256'] == FL96_root['source_package_sha256']
        state['flgmm_fullcoverage_v2_20261009'].update(new_accepted=current_count,accepted_new_ids=off['accepted_job_ids'],
            coverage_verified_cells_including_reuse=current_count+4,latest_backup=FL96_latest,archive_members_verified=FL96_current['archive_members_verified'],
            latest_full70_root_adoption_path=FL96_current_path.relative_to(ROOT).as_posix(),latest_full70_root_adoption_sha256=sha(FL96_current_path))
    elif FL96_latest['accepted_total'] in (2,3,5,7,9,11,12,14,16,18,22):
        assert sha(FL96_current_path)=={2:'d3accbbaeaa6ff34e526c9c9a6daac46c4dad9eb1a69014328295140fb2f20cb',3:'c9aacd305eedf737f313ddcef9ab0b2c2c7ededc9b5aea2230f9205ee45be638',5:'51ae9a0798d763b8bac6ef92022ae028f60bdc059ac9d0f0f7d114b597450288',7:'4403439d39196206e169f14428d68b59b779e1cdd4a5a9fb7d0dd0a3b13dabcf',9:'ecaaa936289589c2b8b28ff42fa81e7e4eb09206fce49a187781be9ed4143577',11:'6feb41c9f2f06980d29865ca03d59e5d2cffeeb2a6f0d6f0209f065cf80caf80',12:'67c355f4cd4fae1d2f015b327fee4b662499987ef9f0307043a1c22ca303a5b0',14:'801ec992899529dc7b38e68acfde3b101d2b90a7966def64f0bbf901cf88793b',16:'9d587b8261c2d1340339e1905db133bc113aeff7b7150db95cbe260103c77617',18:'99de692986c2623b6a351801195094ac35a82e173d4f1c502b9471455c56e3d7',22:'31a1e0d16f14a1855acad3654d656ccb84377b7e6bd8fc3d9c10bf560658c8bc'}[FL96_latest['accepted_total']]
        assert FL96_current['status']=='ROOT_FL96_LINKED_DELTA_ARCHIVE_SOURCE_CHECKPOINT_AND_ORIGINAL_STRICT_BINDING_PASS'
        FL96_previous_path=ROOT/FL96_current['previous_root_adoption_path'];FL96_previous=read(FL96_previous_path)
        assert sha(FL96_previous_path)==FL96_current['previous_root_adoption_sha256']=={2:sha(FL96_first_path),3:'d3accbbaeaa6ff34e526c9c9a6daac46c4dad9eb1a69014328295140fb2f20cb',5:'c9aacd305eedf737f313ddcef9ab0b2c2c7ededc9b5aea2230f9205ee45be638',7:'51ae9a0798d763b8bac6ef92022ae028f60bdc059ac9d0f0f7d114b597450288',9:'4403439d39196206e169f14428d68b59b779e1cdd4a5a9fb7d0dd0a3b13dabcf',11:'ecaaa936289589c2b8b28ff42fa81e7e4eb09206fce49a187781be9ed4143577',12:'6feb41c9f2f06980d29865ca03d59e5d2cffeeb2a6f0d6f0209f065cf80caf80',14:'67c355f4cd4fae1d2f015b327fee4b662499987ef9f0307043a1c22ca303a5b0',16:'801ec992899529dc7b38e68acfde3b101d2b90a7966def64f0bbf901cf88793b',18:'9d587b8261c2d1340339e1905db133bc113aeff7b7150db95cbe260103c77617',22:'99de692986c2623b6a351801195094ac35a82e173d4f1c502b9471455c56e3d7'}[FL96_latest['accepted_total']]
        assert (FL96_current['accepted_before'],FL96_current['accepted_new'],FL96_current['accepted_total'])==(FL96_previous['accepted_total'],{2:1,3:1,5:2,7:2,9:2,11:2,12:1,14:2,16:2,18:2,22:4}[FL96_latest['accepted_total']],FL96_latest['accepted_total'])
        assert FL96_current['accepted_total']==FL96_current['accepted_before']+FL96_current['accepted_new']
        FL96_current_batch=Path(FL96_current['archive_local_path']).parent if FL96_current.get('archive_local_path') else FL96_current_path.parent/'batch';FL96_off=read(FL96_current_batch/'OFFSERVER_ACCEPTANCE.json')
        assert sha(FL96_current_batch/'OFFSERVER_ACCEPTANCE.json')==FL96_current['offserver_acceptance_sha256']==FL96_latest['next_collector_previous_sha256']
        assert sha(FL96_current_batch/'accepted_delta.tar.gz')==FL96_current['archive_sha256']
        assert FL96_off['accepted_job_ids']==FL96_current['accepted_job_ids']==FL96_previous.get('accepted_job_ids',FL96_previous['accepted_new_ids'])+FL96_current['accepted_new_ids']
        assert len(set(FL96_off['accepted_job_ids']))==FL96_latest['accepted_total'] and FL96_current['source_package_sha256']==FL96_root['source_package_sha256']
        state['flgmm_fullcoverage_v2_20261009'].update(new_accepted=FL96_latest['accepted_total'],accepted_new_ids=FL96_off['accepted_job_ids'],
            coverage_verified_cells_including_reuse=FL96_latest['accepted_total']+4,latest_backup=FL96_latest,archive_members_verified=FL96_current['archive_members_verified'],
            latest_full70_root_adoption_path=FL96_current_path.relative_to(ROOT).as_posix(),latest_full70_root_adoption_sha256=sha(FL96_current_path))
    else:
        assert FL96_latest['accepted_total']==1 and FL96_current_path==FL96_first_path
reply_progress_note=('最新版本纳入U100十场景及900校准归因，24原意见逐字、37数值pointer与37链接复核' if not state['latest_rebuttal_draft'].get('complete_C_scenes') else
    '最新版本纳入U100十场景、C的IID Benign/F Flip两场景及900校准归因，24原意见逐字保持，新增22数值pointer、12范围/环境事实和41链接复核')
if state['latest_rebuttal_draft'].get('complete_C_scenes')==5:
    reply_progress_note='最新完整稿纳入U100十场景、C的IID五场景及900校准归因；24原意见逐字保持，11处可逆修改、38个C数值pointer、19个范围/环境事实、24项方向与44链接通过核验；两份旧完整文档可逐字恢复，全部反例及待完成项保留'
if state['latest_rebuttal_draft'].get('complete_C_scenes')==6:
    reply_progress_note='最新完整C60作者审阅稿纳入五IID及non-IID Benign，U100/900旧证据保持；24原意见逐字、10处可逆修改、36数值pointer/18展示值/27方向/50链接通过，原C50两全文可逐字恢复；其他四non-IID C场景及六变体待完成，正文未应用、最终test未运行'
if state['celeba_mechanism_v1'].get('C_three_view_eight_scene_table'):
    reply_progress_note+='；后续C70/C80表已独立采用，尚未合入该封存全文；当前仅non-IID S-DFA/Sp-DFA两个C场景及六变体仍缺，不把旧稿范围当实时进度'
elif state['celeba_mechanism_v1'].get('C_three_view_seven_scene_table'):
    reply_progress_note+='；后续C70七场景表已独立采用，新增non-IID F Flip尚未合入该封存全文；当前仅三non-IID C场景及六变体仍缺，不把旧稿范围当实时进度'
if state.get('latest_rebuttal_addendum') and state['latest_rebuttal_addendum']['complete_C_scenes']>state['latest_rebuttal_draft'].get('complete_C_scenes',0):
    addendum=state['latest_rebuttal_addendum']
    reply_progress_note+=(f"；C{10*addendum['complete_C_scenes']}英文独立补稿已另核{addendum['scalar_pointer_checks']}数值pointer/{addendum['display_cells_checked']}展示值/{addendum['scope_fact_checks']}范围事实/{addendum['links_checked']}链接，入口"+addendum['entry']+'；原24意见完整稿保持封存')
storage_policy=TRAIN/'LOCAL_STORAGE_20261010.json'
if storage_policy.exists():
    state['local_storage_policy_20261010']=read(storage_policy)
C80_addendum=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_C80_addendum_20261010'
if (C80_addendum/'ROOT_REVIEW.json').exists():
    C80_root=read(C80_addendum/'ROOT_REVIEW.json')
    assert C80_root['status']=='ROOT_C80_ADDENDUM_SOURCE_NUMERICS_AND_SCOPE_REVIEW_PASS'
    assert sha(C80_addendum/'ADDENDUM.md')==C80_root['addendum_sha256']=='d9285007dc737f8397a1d605efc78e8bd6042c5466353b10cb5a5b3216fdb503'
    assert sha(C80_addendum/'FILES_SHA256.json')==C80_root['seal_sha256']=='7cb403a74d49766438373faa2cf1bf989c54cc8c2c5486322dd496bd59217d73'
    state['latest_rebuttal_addendum']=dict(status=C80_root['status'],entry=(C80_addendum/'ADDENDUM.md').relative_to(ROOT).as_posix(),root_proof_path=(C80_addendum/'ROOT_REVIEW.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(C80_addendum/'ROOT_REVIEW.json'),complete_C_scenes=8,scalar_pointer_checks=15,display_cells_checked=15,scope_fact_checks=14,links_checked=4,author_review_only=True,whole_rebuttal_complete=False,manuscript_applied=False)
    reply_progress_note+='；C80独立英文四段增补已核15数值pointer/4链接并保存，24意见完整C60稿不变，尚未最终整合或提交'
author_decisions = TRAIN/'AUTHOR_DECISIONS_20261010.json'
if author_decisions.exists():
    assert sha(author_decisions)=='aee89e8210b5aa83d8ee814d5afde4bb655f3ba559bb53ae6ebcd1ca0d46851d'
    state['author_decisions_20261010']=dict(record=read(author_decisions),path=author_decisions.relative_to(ROOT).as_posix(),sha256=sha(author_decisions))
adaptation_reply=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_adaptations_20261010'
if (adaptation_reply/'ROOT_REVIEW.json').exists():
    assert sha(adaptation_reply/'ROOT_REVIEW.json')=='bcd0c0e51a99572eb10c9fef0a8305f067cdaae78359349e4afcc9f8baaf83f4'
    proof=read(adaptation_reply/'ROOT_REVIEW.json')
    assert proof['status']=='ROOT_TEXT_REVIEW_AND_ORIGINAL_READONLY_PATCH_CHECK_PASS_NOT_APPLIED'
    assert proof['author_decisions_sha256']==sha(author_decisions)
    assert sha(adaptation_reply/'FILES_SHA256.json')==proof['delivery_sha256']=='f87eb44eb85ac2acc333641e2764d72055191793cc96edd329683a3333fcc44d'
    assert proof['original_comments_preserved']==24 and proof['reversible_spans']==2
    assert proof['new_scientific_acceptance']==0 and proof['manuscript_applied'] is False
    state['author_adaptation_reply_patch_20261010']=dict(status=proof['status'],root_review_path=(adaptation_reply/'ROOT_REVIEW.json').relative_to(ROOT).as_posix(),
        root_review_sha256=sha(adaptation_reply/'ROOT_REVIEW.json'),reply_patch=(adaptation_reply/'REBUTTAL_PATCH.md').relative_to(ROOT).as_posix(),
        manuscript_patch=(adaptation_reply/'MANUSCRIPT_INSERTION_PATCH.md').relative_to(ROOT).as_posix(),comments_preserved=24,changed_spans=2,
        original_checks_rerun_exit=0,manuscript_applied=False,new_scientific_acceptance=0,test=False)
logofair_gate=ROOT/'tmp/celeba_logofair_real_gate_20261010/ROOT_REVIEW.json'
if logofair_gate.exists():
    assert sha(logofair_gate)=='729030cfb8cc76b8f553f35fb859117f5a4f0655d4cd780879714d74b883b5ed'
    state['logofair_real_score_gate_20261010']=dict(root_review=read(logofair_gate),root_review_path=logofair_gate.relative_to(ROOT).as_posix(),root_review_sha256=sha(logofair_gate),scientific_results=0)
gradient_failure=ROOT/'tmp/celeba_gradient_screen64_root_operations_20261010/LATEST_OBSERVATION.json'
if gradient_failure.exists():
    actual=read(gradient_failure)
    state['gradient64_first_preflight_20261010']=dict(observation_path=gradient_failure.relative_to(ROOT).as_posix(),observation_sha256=sha(gradient_failure),service=actual['service'],actual_resource_pass=actual['resource_proof'] is not None,actual_new_training=len(actual['rows']),scientific_offserver_accepted=0,failure_preserved=True)
gradient_startup=ROOT/'tmp/celeba_gradient_screen64_v2_root_operations_20261010/ROOT_STARTUP_REVIEW.json'
if gradient_startup.exists():
    verified=read(gradient_startup)
    assert verified['status']=='ROOT_ACTUAL_GRADIENT64_SINGLE_GPU_WORKER_AND_ROUND_PROGRESS_VERIFIED'
    assert verified['actual_workers']==1 and verified['actual_CPU']==[105] and verified['physical_GPU']==1
    state['gradient64_validation_search_20261010']=dict(status=verified['status'],root_startup_path=gradient_startup.relative_to(ROOT).as_posix(),root_startup_sha256=sha(gradient_startup),root_startup=verified,source_seal_sha256='11e2ae63c87e0440669a047c930f5465b13babf559cca17bd8808bf693ce7ced',jobs=64,actual_training_started=True,offserver_accepted=0,test=False)
    observation=gradient_startup.parent/'LATEST_ATTEMPT2_OBSERVATION.json'
    if observation.exists():
        measured=read(observation)
        assert not measured['failure_paths']
        rows=[dict(id=r['id'],round=r['progress']['round'] if r['progress'] else None,result_exists=r['result'],server_acceptance_exists=r['accepted']) for r in measured['rows']]
        immutable=[p for p in observation.parent.glob('ATTEMPT2_OBSERVATION_*.json') if sha(p)==sha(observation)]
        assert len(immutable)==1
        state['gradient64_validation_search_20261010']['latest_measured_observation']=dict(path=immutable[0].relative_to(ROOT).as_posix(),sha256=sha(observation),at_unix=measured['at_unix'],rows=rows,terminal70_observed=sum(r['round']==70 for r in rows),offserver_accepted=0)
logofair_execution=ROOT/'tmp/celeba_logofair_screen32_root_execution_20261010'
if (logofair_execution/'queue_LIVE_HANDLE.json').exists():
    handle=read(logofair_execution/'queue_LIVE_HANDLE.json')
    actual_out=Path('F:/YananResearchStorage/GuardFed/logofair_screen32_20261010/attempt001')
    progress=read(actual_out/'PROGRESS.json') if (actual_out/'PROGRESS.json').exists() else None
    failure=read(actual_out/'QUEUE_FAILURE.json') if (actual_out/'QUEUE_FAILURE.json').exists() else None
    state['logofair32_validation_search_20261010']=dict(status='EXPLICIT_FAILURE_PRESERVED' if failure else 'LOCAL_POSTPROCESSING_STARTED_OBSERVE_ORIGINAL_STRICT_PROGRESS',handle=handle,handle_sha256=sha(logofair_execution/'queue_LIVE_HANDLE.json'),source_seal_sha256='accd5cb8582a344f870188f1e70661b6dc9dc948cc88c9e6f6651e451607bc49',jobs=32,original_strict_closed=len(progress['records']) if progress else 0,root_adopted=0,new_CNN_calls=0,new_training=0,test=False,output=str(actual_out),failure=failure)
logofair_adoption=ROOT/'tmp/celeba_logofair32_root_adoption_20261010/ROOT_ADOPTION.json'
if logofair_adoption.exists():
    assert sha(logofair_adoption)=='145f30628270b457d84d47aeab60825c9d4fdb23a0ae27fcd44a35a19e62ae90'
    accepted=read(logofair_adoption)
    assert accepted['status']=='ROOT_LOGOFAIR_SCREEN32_ADOPTED' and accepted['accepted_count']==32 and not accepted['test_evaluated']
    assert sha(ROOT/accepted['independent_acceptance_path'])==accepted['independent_acceptance_sha256']
    assert sha(Path(accepted['summary_path']))==accepted['summary_sha256']
    state['logofair32_validation_search_20261010'].update(status=accepted['status'],root_adopted=32,
        root_adoption_path=logofair_adoption.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(logofair_adoption),
        selected_recipe=accepted['selected_recipe'],constant_prediction_ids=accepted['constant_prediction_ids'],
        seed_n=1,fit_seed=1719,fullcoverage100_started=False)
logofair100_startup=ROOT/'tmp/celeba_logofair100_root_operations_20261010/ROOT_STARTUP_REVIEW.json'
if logofair100_startup.exists():
    verified=read(logofair100_startup)
    assert sha(logofair100_startup)=='1ee8a6b002320e6dae552f816772f298d1ecf15e085a90fbecec775dbece8694'
    assert verified['status']=='ROOT_LOGOFAIR100_FIXED_RECIPE_ACTUAL_STARTUP_AND_FIRST_STRICT_FIT_PASS'
    assert verified['new_fits_planned']==96 and verified['reused']==4 and verified['first_rounds']==30
    assert verified['model_seeds']==list(range(91001,91011)) and verified['fit_seed']==1719
    assert verified['root_adopted']==verified['offserver_accepted']==verified['CNN_training']==verified['CNN_inference']==0
    binding=ROOT/'tmp/celeba_logofair100_root_operations_20261010/bind_attempt002/BIND_RESULT.json'
    assert sha(binding)==verified['bind_result_sha256']=='527440e974ce83510310c7692f088cb87e6fd5b07b36f8c4ffa2280f4e558962'
    cancellation=ROOT/'tmp/celeba_logofair100_reader_v2_20261010/CANCELLATION.json'
    assert sha(cancellation)==read(binding)['cancellation_sha256']=='287191c7d11bcf069d46fdf8eaac2cdc9e282c444cef33ce83c6a6cad7cb7ee0'
    assert read(cancellation)['original_process_stopped'] and read(cancellation)['stage_absent_after_stop']
    progress_path=Path(verified['output'])/'PROGRESS.json'
    progress=read(progress_path)
    failure_path=Path(verified['output'])/'QUEUE_FAILURE.json'
    state['logofair100_fullcoverage_20261010']=dict(status=verified['status'],
        root_startup_path=logofair100_startup.relative_to(ROOT).as_posix(),root_startup_sha256=sha(logofair100_startup),
        startup_checked_utc=verified['checked_utc'],coordinator_pid=verified['coordinator_pid'],
        selected_recipe=verified['selected_recipe'],new_fits_planned=96,reused=4,model_seeds=verified['model_seeds'],fit_seed=1719,
        output=verified['output'],stage=verified['stage'],binding_sha256=sha(binding),
        metadata_reader_cancellation_sha256=sha(cancellation),original_readonly_attempt_preserved=True,
        local_strict_closed_observed=len(progress['records']),progress_checked_utc=now,progress_sha256=sha(progress_path),
        failure=read(failure_path) if failure_path.exists() else None,
        root_adopted=0,offserver_accepted=0,new_CNN=0,final_test=False,automatic_retry=False,
        limitation='Fixed virtual20 cohorts, original DP/root-only adaptation; ten model seeds with shared post-fit seed1719; validation selection history retained')
    state['logofair32_validation_search_20261010']['fullcoverage100_started']=True
logofair100_adoption=TRAIN/'celeba_logofair100_accepted_20261010/ROOT_ADOPTION.json'
if logofair100_adoption.exists():
    proof=read(logofair100_adoption)
    assert sha(logofair100_adoption)=='1529a852b3bd02561d274fdea832186706bb09725c8d02594d09114f44f977c2'
    assert proof['status']=='ROOT_LOGOFAIR_FIXED_RECIPE100_STRICT_SAVED_PREDICTION_AND_TABLES_ADOPTED'
    assert (proof['accepted_count'],proof['root_adopted'],proof['new_accepted'],proof['reused'])==(100,100,96,4)
    assert proof['fit_seed']==1719 and proof['virtual_cohorts']==20 and not proof['final_test']
    for name,digest in proof['files_sha256'].items():assert sha(logofair100_adoption.parent/name)==digest
    assert sha(ROOT/proof['independent_review_path'])==proof['independent_review_sha256']
    state['logofair100_fullcoverage_20261010'].update(status=proof['status'],root_adopted=100,offserver_accepted=100,
        new_strict_offserver_accepted=96,reused_accepted=4,complete_scenes=10,coverage_complete=True,
        root_adoption_path=logofair100_adoption.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(logofair100_adoption),
        canonical_table=proof['canonical_table'],records_path=proof['records_path'],records_sha256=proof['records_sha256'],
        artifact_hashes_recomputed=1027,saved_prediction_items=1986700,saved_metric_checks=300,
        mean_SD_scalars_recomputed=198,display_cells=99,seed_panels=[10,9,6],
        constant_prediction_ids=proof['constant_prediction_ids'],new_CNN=0,final_test=False)
native1000=ROOT/'outputs/guardfed_tables/celeba_ten_method_native_20261010/ROOT_REVIEW.json'
if native1000.exists():
    proof=read(native1000)
    assert sha(native1000)=='a6009db154213944482ed194e5df81c0a6ce353dfc6821b984b222a18f5eeeec'
    assert proof['status']=='ROOT_TEN_METHOD_NATIVE1000_DESCRIPTIVE_TABLE_ADOPTED'
    assert (proof['records'],proof['methods'],proof['rendered_scene_cells'],proof['seed_first_aggregate_scalars'])==(1000,10,900,540)
    assert not proof['final_test'] and not proof['full17_complete']
    for name,digest in proof['files_sha256'].items():assert sha(native1000.parent/name)==digest
    state['celeba_native_ten_method_table_20261010']=dict(status=proof['status'],root_proof_path=native1000.relative_to(ROOT).as_posix(),
        root_proof_sha256=sha(native1000),table_path=proof['canonical_table'],methods=10,records=1000,
        distributions=['IID','non-IID'],scenes_per_distribution=5,seed_panels=[10,9,6],view='native',
        remaining_methods=['Fed-NGA','FedWA','Huber','FLGMM','SmartFL','FedDNA','CosineFairness'],
        original900_three_view_scope_unchanged=True,new_CNN=0,new_fit=0,final_test=False,full17_complete=False,primary_endpoint_selected=False)
remaining620_startup=ROOT/'tmp/celeba_mechanism_remaining620_root_operations_20261010/ROOT_STARTUP_REVIEW_V2.json'
if remaining620_startup.exists():
    verified=read(remaining620_startup)
    assert verified['status']=='ROOT_ACTUAL_REMAINING620_CPU_QUEUE_AND_FIRST_REMOTE_STRICT_CLOSURE_VERIFIED'
    assert sha(ROOT/verified['observation_path'])==verified['observation_sha256']
    assert verified['source_seal_sha256']=='a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03'
    state['mechanism_remaining620_valid_20261010']=dict(status=verified['status'],root_startup_path=remaining620_startup.relative_to(ROOT).as_posix(),root_startup_sha256=sha(remaining620_startup),actual_service=verified['actual_service'],observed_utc=verified['observed_utc'],root_approval_sha256=verified['root_approval_sha256'],remote_strict_closed_at_startup=verified['remote_strict_closed'],new_offserver_accepted=0,selected_count=620,original180_excluded=True,Full_inference=0,new_training=0,test=False,automatic_retry=False,source_seal_sha256=verified['source_seal_sha256'],first_wrapper_failure_preserved=True)
    transport_failure=remaining620_startup.parent/'FIRST_TRANSPORT_COMMAND.json'
    if transport_failure.exists() and read(transport_failure)['returncode']:
        state['mechanism_remaining620_valid_20261010']['first_transport_failure']=dict(path=transport_failure.relative_to(ROOT).as_posix(),sha256=sha(transport_failure),cause='Pinned archive-verifier remote path missing; same hash tool exists at canonical reactivation path. Finite recovery separately reviewed, not blind export retry.',scientific_outputs_unchanged=True)
    observation=remaining620_startup.parent/'LATEST_OBSERVATION.json'
    if observation.exists():
        latest=read(observation)
        immutable=[p for p in observation.parent.glob('OBSERVATION_*.json') if sha(p)==sha(observation)]
        assert len(immutable)==1 and not latest['failures']
        state['mechanism_remaining620_valid_20261010']['latest_measured_observation']=dict(path=immutable[0].relative_to(ROOT).as_posix(),sha256=sha(observation),utc=latest['utc'],remote_strict_closed=latest['remote_closed_n'],remote_closed_ids=latest['remote_closed_ids'],active_evaluation_workers=sum('worker' in p['argv'] for p in latest['processes']),service=latest['service'])
    state['active_services']=list(dict.fromkeys(state['active_services']+[verified['actual_service']]))
root_first=ROOT/'tmp/root_adopt_first_closed_20261010'
if (root_first/'GRADIENT1_ROOT_ADOPTION.json').exists():
    proof=read(root_first/'GRADIENT1_ROOT_ADOPTION.json')
    assert sha(root_first/'GRADIENT1_ROOT_ADOPTION.json')=='ca4a38076e5080d94069cdabd50a032d84a96a339914761ec568db44a43db978'
    assert proof['accepted_count']==1 and not proof['method_champion_claim'] and proof['constant_negative_retained']
    state['gradient64_validation_search_20261010'].update(offserver_accepted=1,root_adoption_path=(root_first/'GRADIENT1_ROOT_ADOPTION.json').relative_to(ROOT).as_posix(),root_adoption_sha256=sha(root_first/'GRADIENT1_ROOT_ADOPTION.json'),accepted_ids=proof['accepted_ids'],first_negative_metrics=proof['metrics'])
    state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_gradient_screen64_v2a']))
gradient4_root=ROOT/'tmp/celeba_gradient64_delta_after1_20261010/ROOT_ADOPTION_REVIEW.json'
if gradient4_root.exists():
    proof=read(gradient4_root)
    assert sha(gradient4_root)=='e040d742c082dbc65dbb8b7cc36055cf869b5e75950dd927d24a362471b0ff95'
    assert proof['status']=='ROOT_GRADIENT64_EXACT4_ORIGINAL_STRICT_OFFSERVER_ADOPTED'
    assert (proof['accepted_before'],proof['accepted_new'],proof['accepted_total'])==(1,4,5)
    assert proof['archive_members_root_verified']==132 and proof['raw_files_root_verified']==135
    assert proof['previous_root_sha256']==state['gradient64_validation_search_20261010']['root_adoption_sha256']
    auth=read(gradient4_root.parent/'AUTHORIZED_SNAPSHOT.json')
    assert proof['accepted_new_ids']==auth['authorized_ids'] and proof['all_negative_results_retained']
    state['gradient64_validation_search_20261010'].update(offserver_accepted=5,
        root_adoption_path=gradient4_root.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(gradient4_root),
        accepted_ids=proof['accepted_ids'],all_negative_results_retained=True,
        delta_acceptance_snapshot=dict(path=(gradient4_root.parent/'SNAPSHOT.json').relative_to(ROOT).as_posix(),
            sha256=sha(gradient4_root.parent/'SNAPSHOT.json'),at_unix=auth['snapshot_unix'],
            terminal70_observed=auth['terminal_count'],offserver_accepted=5,
            scope='Same frozen five-terminal snapshot; exact four delta excludes original accepted one'))
    if auth['snapshot_unix']>state['gradient64_validation_search_20261010'].get('latest_measured_observation',{}).get('at_unix',0):
        state['gradient64_validation_search_20261010']['latest_measured_observation']=state['gradient64_validation_search_20261010']['delta_acceptance_snapshot']
gradient5_root=ROOT/'tmp/celeba_gradient64_delta_after5_20261010/ROOT_ADOPTION_REVIEW.json'
if gradient5_root.exists():
    proof=read(gradient5_root)
    assert sha(gradient5_root)=='99a07fefad41e0f441288f86c5335e175eb5992983fec047ab3a1c9f1eacea57'
    assert proof['status']=='ROOT_GRADIENT64_EXACT5_ORIGINAL_STRICT_OFFSERVER_ADOPTED'
    assert (proof['accepted_before'],proof['accepted_new'],proof['accepted_total'])==(5,5,10)
    assert proof['archive_members_root_verified']==139 and proof['raw_files_root_verified']==142
    prior=state['gradient64_validation_search_20261010']
    assert proof['previous_root_sha256']==prior['root_adoption_sha256']
    assert proof['accepted_ids']==prior['accepted_ids']+proof['accepted_new_ids']
    assert len(set(proof['accepted_ids']))==10 and len(proof['constant_negative_ids'])==3
    auth=read(gradient5_root.parent/'AUTHORIZED_SNAPSHOT.json')
    assert proof['accepted_new_ids']==auth['authorized_ids'] and proof['all_negative_results_retained']
    prior.update(offserver_accepted=10,root_adoption_path=gradient5_root.relative_to(ROOT).as_posix(),
        root_adoption_sha256=sha(gradient5_root),accepted_ids=proof['accepted_ids'],
        delta_acceptance_snapshot=dict(path=(gradient5_root.parent/'SNAPSHOT.json').relative_to(ROOT).as_posix(),
            sha256=sha(gradient5_root.parent/'SNAPSHOT.json'),at_unix=auth['snapshot_unix'],
            terminal70_observed=auth['terminal_count'],offserver_accepted=10,
            scope='One frozen ten-terminal snapshot; exact five delta excludes the previous accepted five'))
    if auth['snapshot_unix']>prior.get('latest_measured_observation',{}).get('at_unix',0):
        prior['latest_measured_observation']=prior['delta_acceptance_snapshot']
if (root_first/'MECHANISM1_ROOT_ADOPTION.json').exists():
    p=root_first/'MECHANISM1_ROOT_ADOPTION.json';proof=read(p)
    assert sha(p)=='ef9b6cc30821b3c376a7560e4317b790524365849c31609ab557a26185f7815b'
    index=read(ROOT/proof['records_index_path'])
    assert sha(ROOT/proof['records_index_path'])==proof['records_index_sha256']
    assert index['prior180_ids']==state['celeba_mechanism_v1']['three_view_accepted_ids']
    state['mechanism_remaining620_valid_20261010'].update(new_offserver_accepted=1,root_adoption_path=p.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(p),accepted_ids=proof['accepted_new_ids'],first_transport_recovered_original_archive_unchanged=True)
    state['celeba_mechanism_v1'].update(three_view_new_models_accepted=181,three_view_new_models_offserver_verified=181,three_view_accepted_ids=index['all_ids'],three_view_counts_by_variant={'minus_U':100,'minus_C':81})
root_C19=ROOT/'tmp/celeba_mechanism_remaining620_C100_root_adoption_20261010/ROOT_ADOPTION.json'
if root_C19.exists():
    proof=read(root_C19)
    assert sha(root_C19)=='2ac1d2f200d9671de5271f24b0cbb3a0772afb88c16ae6ccf71805ed1588ee46'
    assert proof['status']=='ROOT_C19_SAVED_ARRAYS_AND_NATIVE200_RESTORE_CHAIN_ADOPTED'
    assert proof['cumulative_accepted']==200 and proof['new_accepted']==19 and proof['native_max_abs_difference']==0 and not proof['test']
    index=read(ROOT/proof['records_index_path']);assert sha(ROOT/proof['records_index_path'])==proof['records_index_sha256']
    assert index['all_ids'][:181]==state['celeba_mechanism_v1']['three_view_accepted_ids']
    assert sha(ROOT/proof['native200_inspection_path'])==proof['native200_inspection_sha256']
    state['mechanism_remaining620_valid_20261010'].update(new_offserver_accepted=20,root_adoption_path=root_C19.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(root_C19),accepted_ids=[index['first_record']['id']]+proof['accepted_new_ids'],C100_replay_complete=True,C100_table_adopted=False)
    state['celeba_mechanism_v1'].update(three_view_new_models_accepted=200,three_view_new_models_offserver_verified=200,
        three_view_accepted_ids=index['all_ids'],three_view_counts_by_variant={'minus_U':100,'minus_C':100},
        three_view_root_proof_sha256=sha(root_C19),three_view_native_max_abs_difference=0,
        three_view_scope_limit='U100 and C100 replays strictly adopted; C80 table remains the latest accepted C table until separate C100 statistical review. Six other variants remain unfinished; final test and submitted manuscript are not complete.')
C100_dir=TRAIN/'celeba_mechanism_v1/three_view_C_full100_20261010'
if (C100_dir/'ROOT_VERIFICATION.json').exists():
    C100=read(C100_dir/'ROOT_VERIFICATION.json')
    assert sha(C100_dir/'ROOT_VERIFICATION.json')=='0bed6d372c63a60a978f8faa1353ec3257dcba66cc236b01906abd7097a0d0e7'
    assert C100['status']=='ROOT_C100_TEN_SCENE_THREE_VIEW_TABLES_ADOPTED'
    assert tuple(C100[k] for k in ('unique_records','paired_models','complete_scenes','mean_SD_scalars_recomputed','display_cells'))==(200,100,10,1620,810)
    assert C100['source_acceptance_sha256']==sha(root_C19) and not C100['test']
    state['celeba_mechanism_v1'].update(C_three_view_full100_table=dict(status=C100['status'],table_path=C100['canonical_table'],root_proof_path=(C100_dir/'ROOT_VERIFICATION.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(C100_dir/'ROOT_VERIFICATION.json'),complete_scenes=10,paired_models=100,seed_panels=[10,9,6],final_test=False,incorporated_into_full_rebuttal=False),
        three_view_scope_limit='U100 and C100 complete ten-scene three-view tables independently adopted; six other variants remain unfinished. All10/9/6 panels, paired effects, source/environment and selection/test history retained. No necessity, causal, significance or universal-win claim. Submitted manuscript and frozen final evaluation remain incomplete.')
    state['mechanism_remaining620_valid_20261010']['C100_table_adopted']=True
    reply_progress_note+='；C100十场景三视图表已独立采用，六个其他变体未完成，完整英文稿待本次整合'
reply_C100=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C100_20261010'
if (reply_C100/'ROOT_REVIEW.json').exists():
    proof=read(reply_C100/'ROOT_REVIEW.json')
    assert sha(reply_C100/'ROOT_REVIEW.json')=='84e73cfb22780deb1a0fdefb328237c0c6a08596bc593ea1da09cb1cd14dbfed'
    assert proof['status']=='ROOT_COMPLETE_C100_REBUTTAL_AND_INSERTION_CANDIDATES_ADOPTED_FOR_AUTHOR_REVIEW'
    assert proof['root_table_sha256']==sha(C100_dir/'ROOT_VERIFICATION.json') and proof['complete_C_scenes']==10
    for name,want in proof['documents_sha256'].items():assert sha(reply_C100/name)==want
    state['latest_rebuttal_draft'].update(status=proof['status'],entry=(reply_C100/'rebuttal_integrated_20261009.md').relative_to(ROOT).as_posix(),manuscript_candidate=(reply_C100/'manuscript_insertions_integrated_20261009.md').relative_to(ROOT).as_posix(),root_proof_path=(reply_C100/'ROOT_REVIEW.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(reply_C100/'ROOT_REVIEW.json'),source_seal_sha256=proof['source_seal_sha256'],complete_C_scenes=10,new_C_scalar_pointer_checks=90,new_C_numeric_cells=45,links_checked=60,changed_passages=14,direction_checks=54,other_seven_controls_pending=False,other_six_controls_pending=True,whole_rebuttal_complete=False,manuscript_applied=False)
    state['celeba_mechanism_v1']['C_three_view_full100_table']['incorporated_into_full_rebuttal']=True
    reply_progress_note='最新完整C100作者审阅稿已纳入U/C各十场景及900校准归因；24原意见逐字、14处可逆修改、90数值pointer/45单元/54方向/60链接经root实际复核通过，旧C60两全文及原数值保持。Huber/LoGoFair适配已明确；六个其他变体、P1–P6、正文和最终评价仍未完成'
root_A12=ROOT/'tmp/celeba_mechanism_remaining620_A12_root_adoption_20261010/ROOT_ADOPTION.json'
if root_A12.exists():
    proof=read(root_A12)
    assert sha(root_A12)=='1221482d564a2c735b0de0680fe8a42512c9fe5774a9d157bd3bfda5dd9c858b'
    assert proof['status']=='ROOT_A12_SAVED_ARRAYS_AND_NATIVE212_RESTORE_CHAIN_ADOPTED'
    assert (proof['new_accepted'],proof['prior_accepted'],proof['cumulative_accepted'])==(12,200,212)
    assert proof['native_max_abs_difference']==0 and not proof['test']
    index=read(ROOT/proof['records_index_path'])
    assert sha(ROOT/proof['records_index_path'])==proof['records_index_sha256']=='9a90f3d74a27d9ca4225b850797faec3c3aac2b6e2f83f87d3afe49c21912496'
    assert index['all_ids'][:200]==state['celeba_mechanism_v1']['three_view_accepted_ids']
    remaining=state['mechanism_remaining620_valid_20261010']
    assert remaining['new_offserver_accepted']==20
    remaining.update(new_offserver_accepted=32,root_adoption_path=root_A12.relative_to(ROOT).as_posix(),
        root_adoption_sha256=sha(root_A12),accepted_ids=remaining['accepted_ids']+proof['accepted_new_ids'],
        A_complete_scenes=proof['complete_A_scenes'],A_partial_scenes=proof['partial_A_scenes'],A_table_adopted=False)
    state['celeba_mechanism_v1'].update(three_view_new_models_accepted=212,three_view_new_models_offserver_verified=212,
        three_view_accepted_ids=index['all_ids'],three_view_counts_by_variant={'minus_U':100,'minus_C':100,'minus_A':12},
        three_view_root_proof_sha256=sha(root_A12),three_view_scope_limit='U100 and C100 ten-scene tables retained; twelve minus_A replays strictly adopted with native212 restore chain. A IID Benign has ten shared seeds, F Flip only two; A table requires separate review. Other controls, frozen final evaluation and submitted manuscript remain incomplete.')
A10_dir=TRAIN/'celeba_mechanism_v1/three_view_A_Benign10_20261010'
if (A10_dir/'ROOT_VERIFICATION.json').exists():
    proof=read(A10_dir/'ROOT_VERIFICATION.json')
    assert sha(A10_dir/'ROOT_VERIFICATION.json')=='c3d65134f0e6eeda36fd37c64b6ac0df3794828af054d920d46775670a54df9d'
    assert proof['status']=='ROOT_A12_SINGLE_COMPLETE_SCENE_THREE_VIEW_TABLE_ADOPTED'
    assert (proof['paired_models'],proof['complete_scenes'],proof['preserved_records'])==(10,1,24)
    assert proof['source_acceptance_sha256']==sha(root_A12) and not proof['test']
    for name,pin in proof['files_sha256'].items():assert sha(A10_dir/name)==pin
    state['celeba_mechanism_v1'].update(A_three_view_single_scene_table=dict(status=proof['status'],table_path=proof['canonical_table'],root_proof_path=(A10_dir/'ROOT_VERIFICATION.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(A10_dir/'ROOT_VERIFICATION.json'),complete_scenes=1,paired_models=10,seed_panels=[10,9,6],final_test=False,incorporated_into_full_rebuttal=False),
        three_view_scope_limit='U100 and C100 ten-scene tables and A IID Benign ten-seed table independently adopted. A F Flip has only two seeds, excluded from scene means. All10/9/6 panels, source/environment and negative effects retained; other controls, final evaluation and submitted manuscript unfinished.')
    state['mechanism_remaining620_valid_20261010']['A_table_adopted']=True
root_A20=ROOT/'tmp/celeba_mechanism_remaining620_A20_root_adoption_20261010/ROOT_ADOPTION.json'
if root_A20.exists():
    proof=read(root_A20)
    assert sha(root_A20)=='e088871fbd98cbc9415cc79a44626667a532389ce0b6dddbaaf2ab25f72a4979'
    assert proof['status']=='ROOT_A20_SAVED_ARRAYS_AND_NATIVE220_RESTORE_CHAIN_ADOPTED'
    assert (proof['prior_accepted'],proof['new_accepted'],proof['cumulative_accepted'])==(212,8,220)
    assert (proof['archive_members'],proof['independent_metrics'],proof['independent_counts'],proof['prediction_rules'])==(74,72,192,24)
    assert proof['native_members_rehashed']==16 and proof['native_max_abs_difference']==0 and not proof['test']
    index_path=ROOT/proof['records_index_path'];index=read(index_path)
    assert sha(index_path)==proof['records_index_sha256']=='af414a0ac6705230c53d324cd1f51e7a76b893dabd7416bb6914a6690b6fdddc'
    assert index['all_ids'][:212]==state['celeba_mechanism_v1']['three_view_accepted_ids']
    assert proof['accepted_new_ids']==[f'minus_A_IID_F Flip_seed{s}' for s in range(91003,91011)]
    remaining=state['mechanism_remaining620_valid_20261010']
    assert remaining['new_offserver_accepted']==32
    remaining.update(new_offserver_accepted=40,root_adoption_path=root_A20.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(root_A20),
        accepted_ids=remaining['accepted_ids']+proof['accepted_new_ids'],A_complete_scenes=proof['complete_A_scenes'],A_partial_scenes=[],A_two_scene_table_adopted=False)
    state['celeba_mechanism_v1'].update(three_view_new_models_accepted=220,three_view_new_models_offserver_verified=220,
        three_view_accepted_ids=index['all_ids'],three_view_counts_by_variant={'minus_U':100,'minus_C':100,'minus_A':20},three_view_root_proof_sha256=sha(root_A20),
        three_view_scope_limit='U100/C100 complete tables and the original A Benign10 table retained. A IID Benign/F Flip now each have ten strictly adopted paired checkpoints; their new two-scene table awaits separate arithmetic adoption. Other eight A scenes, other controls, final evaluation and submitted manuscript remain incomplete.')

A20_table=TRAIN/'celeba_mechanism_v1/three_view_A_IID_two_scenes20_20261010'
if (A20_table/'ROOT_VERIFICATION.json').exists():
    proof=read(A20_table/'ROOT_VERIFICATION.json')
    assert sha(A20_table/'ROOT_VERIFICATION.json')=='dbc2f8fe481c2a049f4e556a500c29d3122fc8392d69e4ab2bc131c0e3f4b3c0'
    assert proof['status']=='ROOT_A20_TWO_COMPLETE_IID_SCENE_THREE_VIEW_TABLE_ADOPTED' and proof['root_adoption']
    assert (proof['paired_models'],proof['complete_scenes'],proof['preserved_records'])==(20,2,40)
    assert (proof['mean_SD_scalars_recomputed'],proof['display_cells'],proof['metrics_from_group_counts'])==(324,162,360)
    assert proof['source_acceptance_sha256']==sha(root_A20) and not proof['test']
    for name,digest in proof['files_sha256'].items():assert sha(A20_table/name)==digest
    assert sha(ROOT/proof['independent_review_path'])==proof['independent_review_sha256']
    assert sha(ROOT/proof['preserved_first_review_failure_path'])==proof['preserved_first_review_failure_sha256']
    state['celeba_mechanism_v1'].update(A_three_view_two_scene_table=dict(status=proof['status'],table_path=proof['canonical_table'],root_proof_path=(A20_table/'ROOT_VERIFICATION.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(A20_table/'ROOT_VERIFICATION.json'),complete_scenes=2,paired_models=20,seed_panels=[10,9,6],final_test=False,incorporated_into_full_rebuttal=False),
        three_view_scope_limit='U100/C100 ten-scene tables and A IID Benign/F Flip two-scene table independently adopted. All10/9/6 panels, old24 records and Benign162 statistics/81 cells retained. Eight other A scenes, other controls, final evaluation and submitted manuscript remain incomplete; no necessity or significance claim.')
    state['mechanism_remaining620_valid_20261010']['A_two_scene_table_adopted']=True

root_A28=ROOT/'tmp/celeba_mechanism_remaining620_A28_root_adoption_20261010/ROOT_ADOPTION.json'
if root_A28.exists():
    proof=read(root_A28)
    assert sha(root_A28)=='f6761dd1c2b844724aee91d235d71f6474511ac8aec63f3dc4fe0308272e0966'
    assert proof['status']=='ROOT_A28_SAVED_ARRAYS_AND_NATIVE228_RESTORE_CHAIN_ADOPTED'
    assert (proof['prior_accepted'],proof['new_accepted'],proof['cumulative_accepted'])==(220,8,228)
    assert (proof['archive_members'],proof['independent_metrics'],proof['independent_counts'],proof['prediction_rules'])==(74,72,192,24)
    assert proof['native_members_rehashed']==16 and proof['native_max_abs_difference']==0 and not proof['test']
    index_path=ROOT/proof['records_index_path'];index=read(index_path)
    assert sha(index_path)==proof['records_index_sha256']=='765ea715defea1e54aebbee0115f38c926ee022f47f520d151d1417c6c8592a9'
    assert index['all_ids'][:220]==state['celeba_mechanism_v1']['three_view_accepted_ids']
    assert proof['accepted_new_ids']==[f'minus_A_IID_FedSA_seed{s}' for s in range(91001,91009)]
    assert sha(ROOT/proof['independent_review_path'])==proof['independent_review_sha256']
    remaining=state['mechanism_remaining620_valid_20261010'];assert remaining['new_offserver_accepted']==40
    remaining.update(new_offserver_accepted=48,root_adoption_path=root_A28.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(root_A28),
        accepted_ids=remaining['accepted_ids']+proof['accepted_new_ids'],A_partial_scenes=proof['partial_A_scenes'])
    state['celeba_mechanism_v1'].update(three_view_new_models_accepted=228,three_view_new_models_offserver_verified=228,
        three_view_accepted_ids=index['all_ids'],three_view_counts_by_variant={'minus_U':100,'minus_C':100,'minus_A':28},three_view_root_proof_sha256=sha(root_A28),
        three_view_scope_limit='U100/C100 ten-scene and A20 two-IID-scene tables remain adopted. Eight IID FedSA checkpoints are additionally root-adopted with exact native228 recovery identity, but do not create a ten-seed scene mean. A100, remaining controls, final evaluation and submitted manuscript remain incomplete.')

root_A36=ROOT/'tmp/celeba_mechanism_remaining620_A36_root_adoption_20261010/ROOT_ADOPTION.json'
if root_A36.exists():
    proof=read(root_A36)
    assert sha(root_A36)=='1513441c17a3d6439df4a2944c6fcf6a0e6bb6aa2280d2176d7bc1527a78c36b'
    assert proof['status']=='ROOT_A36_SAVED_ARRAYS_AND_NATIVE236_RESTORE_CHAIN_ADOPTED'
    assert (proof['prior_accepted'],proof['new_accepted'],proof['cumulative_accepted'])==(228,8,236)
    assert (proof['archive_members'],proof['independent_metrics'],proof['independent_counts'],proof['prediction_rules'])==(74,72,192,24)
    assert proof['native_members_rehashed']==16 and proof['native_max_abs_difference']==0 and not proof['test']
    index_path=ROOT/proof['records_index_path'];index=read(index_path)
    assert sha(index_path)==proof['records_index_sha256']=='f378bab97b5a2fba436370f4508f122198559d05920dac2ac9075d83ef7f6e59'
    assert index['all_ids'][:228]==state['celeba_mechanism_v1']['three_view_accepted_ids']
    expected=[f'minus_A_IID_FedSA_seed{s}' for s in (91009,91010)]+[f'minus_A_IID_S-DFA_seed{s}' for s in range(91001,91007)]
    assert proof['accepted_new_ids']==expected
    assert sha(ROOT/proof['independent_review_path'])==proof['independent_review_sha256']
    remaining=state['mechanism_remaining620_valid_20261010'];assert remaining['new_offserver_accepted']==48
    remaining.update(new_offserver_accepted=56,root_adoption_path=root_A36.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(root_A36),
        accepted_ids=remaining['accepted_ids']+expected,A_partial_scenes=proof['partial_A_scenes'],A_complete_data_scenes=proof['complete_A_scenes'])
    state['celeba_mechanism_v1'].update(three_view_new_models_accepted=236,three_view_new_models_offserver_verified=236,
        three_view_accepted_ids=index['all_ids'],three_view_counts_by_variant={'minus_U':100,'minus_C':100,'minus_A':36},three_view_root_proof_sha256=sha(root_A36),
        three_view_scope_limit='U100/C100 ten-scene and A20 two-IID-scene tables remain adopted. A IID FedSA ten paired checkpoints are now complete at record level, while S-DFA is6/10 and excluded from complete-scene means. No additional A scene table has been adopted; A100, remaining controls, final evaluation and submitted manuscript remain incomplete.')

root_A40=ROOT/'tmp/celeba_mechanism_remaining620_A40_root_adoption_20261010/ROOT_ADOPTION.json'
if root_A40.exists():
    proof=read(root_A40)
    assert sha(root_A40)=='8e58b5b568ccd1ccf9a9e6a58978a1882533f83a5034d8d086e96924e40e778e'
    assert proof['status']=='ROOT_A40_SAVED_ARRAYS_AND_NATIVE243_RESTORE_CHAIN_ADOPTED'
    assert (proof['prior_accepted'],proof['new_accepted'],proof['cumulative_accepted'])==(236,4,240)
    assert (proof['archive_members'],proof['independent_metrics'],proof['independent_counts'],proof['prediction_rules'])==(38,36,96,12)
    assert proof['native_members_rehashed']==8 and proof['native_max_abs_difference']==0 and proof['original236_unchanged'] and not proof['test']
    index_path=ROOT/proof['records_index_path'];index=read(index_path)
    assert sha(index_path)==proof['records_index_sha256']=='aa0df30db62549f1be808f188fa9a0f9479b9daa812406cd98ff438b2068f8cc'
    assert index['all_ids'][:236]==state['celeba_mechanism_v1']['three_view_accepted_ids']
    expected=[f'minus_A_IID_S-DFA_seed{s}' for s in range(91007,91011)]
    assert proof['accepted_new_ids']==expected
    assert sha(ROOT/proof['independent_review_path'])==proof['independent_review_sha256']
    remaining=state['mechanism_remaining620_valid_20261010'];assert remaining['new_offserver_accepted']==56
    remaining.update(new_offserver_accepted=60,root_adoption_path=root_A40.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(root_A40),
        accepted_ids=remaining['accepted_ids']+expected,A_partial_scenes=[],A_complete_data_scenes=proof['complete_A_scenes'])
    state['celeba_mechanism_v1'].update(three_view_new_models_accepted=240,three_view_new_models_offserver_verified=240,
        three_view_accepted_ids=index['all_ids'],three_view_counts_by_variant={'minus_U':100,'minus_C':100,'minus_A':40},three_view_root_proof_sha256=sha(root_A40),
        three_view_scope_limit='U100/C100 ten-scene and A20 two-IID-scene tables remain adopted. A IID Benign/F Flip/FedSA/S-DFA each have ten adopted paired checkpoints; A40 table requires separate arithmetic adoption. Six other A scenes, remaining controls, final evaluation and submitted manuscript remain incomplete.')

reply_A20_LoGo100=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A20_LoGo100_20261010'
if (reply_A20_LoGo100/'ROOT_REVIEW.json').exists():
    proof=read(reply_A20_LoGo100/'ROOT_REVIEW.json')
    assert proof['status']=='ROOT_COMPLETE_A20_LOGO100_NATIVE1000_REBUTTAL_AND_INSERTION_CANDIDATES_ADOPTED_FOR_AUTHOR_REVIEW'
    assert proof['source_seal_sha256']==sha(reply_A20_LoGo100/'FILES_SHA256.json')=='713cfb3d32c05ea0a61d419f344cda823b5fc36c33e80233961cf86366a207d8'
    assert (proof['original_comments'],proof['A_complete_scenes'],proof['LoGoFair_native_records'],proof['native_comparison_records'])==(24,2,100,1000)
    assert proof['actual_root_command_exit']==0 and not proof['final_test'] and not proof['manuscript_applied']
    assert sha(ROOT/proof['independent_review_path'])==proof['independent_review_sha256']
    for name,want in proof['documents_sha256'].items():assert sha(reply_A20_LoGo100/name)==want
    state['latest_rebuttal_draft'].update(status=proof['status'],entry=(reply_A20_LoGo100/'rebuttal_integrated_20261009.md').relative_to(ROOT).as_posix(),
        manuscript_candidate=(reply_A20_LoGo100/'manuscript_insertions_integrated_20261009.md').relative_to(ROOT).as_posix(),
        root_proof_path=(reply_A20_LoGo100/'ROOT_REVIEW.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(reply_A20_LoGo100/'ROOT_REVIEW.json'),
        source_seal_sha256=proof['source_seal_sha256'],complete_C_scenes=10,A_complete_scenes=2,LoGoFair_native_records=100,native_comparison_records=1000,
        scalar_pointer_checks=46,numeric_cells=23,fact_pointers=59,links_checked=82,changed_passages=25,A_direction_checks=54,
        independent_review_path=proof['independent_review_path'],independent_review_sha256=proof['independent_review_sha256'],
        whole_rebuttal_complete=False,manuscript_applied=False,author_review_only=True)
    state['celeba_mechanism_v1']['A_three_view_two_scene_table']['incorporated_into_full_rebuttal']=True
    state['logofair100_fullcoverage_20261010']['incorporated_into_full_rebuttal']=True
    state['celeba_native_ten_method_table_20261010']['incorporated_into_full_rebuttal']=True
    reply_progress_note='最新完整作者审阅稿纳入U/C各十场景、A两IID场景、LoGoFair100及十方法native千格表；24原意见逐字、25可逆修改、23新增均值SD单元/59来源事实/54方向/6取舍均值/82链接通过root及独立语义审阅。旧C100两全文和旧数值保持；七方法完整覆盖、六机制变体、P1–P6、最终评价和正文仍未完成'

gradient_review=ROOT/'tmp/celeba_gradient_fullcoverage_prepare_review_20261010/REVIEW.json'
if gradient_review.exists():
    proof=read(gradient_review)
    assert sha(gradient_review)=='792da228c102a7044628aa30f2fc6310bb486441c822d321ab62a2ee78843d09'
    assert proof['source_preparation_adoptable'] and not proof['actual_dispatch_authorized_by_this_review']
    state['gradient200_fullcoverage_source_preparation_20261010']=dict(status=proof['status'],review_path=gradient_review.relative_to(ROOT).as_posix(),review_sha256=sha(gradient_review),source_seal_sha256=proof['candidate_seal_sha256'],new_jobs_planned=192,reused_jobs_planned=8,actual_jobs=0,selected_recipes=0,actual_dispatch=False,test=False,limitations='Requires complete64 strict/offserver/root acceptance, frozen source and real-image new-attack gates before execution; preparation is not a result.')
aux_dir=ROOT/'tmp/celeba_aux_live_followup_20261010T064710Z'
if (aux_dir/'FILES_SHA256.json').exists():
    assert sha(aux_dir/'FILES_SHA256.json')=='8f72ea527d8f7dd93c18dab3cb01e0aea71dac54d5fde55dab417f0c4b8b1c7c'
    for name,pin in read(aux_dir/'FILES_SHA256.json')['files'].items():
        assert sha(aux_dir/name)==pin['sha256'] and (aux_dir/name).stat().st_size==pin['bytes']
    measured=read(aux_dir/'FINDINGS.json')
    assert measured['source_and_failure_clean'] and not measured['collect_performed']
    state['auxiliary_readonly_observation_20261010']=dict(path=(aux_dir/'FINDINGS.json').relative_to(ROOT).as_posix(),sha256=sha(aux_dir/'FINDINGS.json'),utc=measured['utc'],FL_observed_terminal=measured['FL_observed_terminal'],FL_accepted=measured['FL_accepted'],Hybrid_observed_terminal=measured['Hybrid_observed_terminal'],Hybrid_accepted=measured['Hybrid_accepted'],observation_is_atomic=False,acceptance_unchanged=True)
gates_dir=ROOT/'tmp/celeba_gradient_fullcoverage_gates_review_20261010'
if (gates_dir/'REVIEW.json').exists():
    assert sha(gates_dir/'FILES_SHA256.json')=='76c3d10fdb67d45a2d06d96dfbdf6f241faf6f8c271aa5e4222012da0f2f11a4'
    for name,pin in read(gates_dir/'FILES_SHA256.json')['files'].items():
        assert sha(gates_dir/name)==pin['sha256'] and (gates_dir/name).stat().st_size==pin['bytes']
    proof=read(gates_dir/'REVIEW.json')
    assert sha(gates_dir/'REVIEW.json')=='47291c77f49777d64a1949ce09fef0e57def822ddecf741be883860f4a8abb68'
    assert proof['source_adoptable'] and not proof['actual_execution_authorized']
    state['gradient200_new_attack_gates_source_20261010']=dict(status=proof['status'],review_path=(gates_dir/'REVIEW.json').relative_to(ROOT).as_posix(),review_sha256=sha(gates_dir/'REVIEW.json'),source_seal_sha256=proof['source_seal_sha256'],planned_gate_jobs=14,rounds=3,actual_jobs=0,actual_image_gates_passed=0,dispatch=False,test=False)
scope_dir=ROOT/'tmp/celeba_added_baseline_three_view_scope_20261010'
if (scope_dir/'REVIEW.json').exists():
    assert sha(scope_dir/'FILES_SHA256.json')=='1cc32cee3f41abcc644510251a79f5751a60fdf8185533c9f0352ba89c150564'
    for name,pin in read(scope_dir/'FILES_SHA256.json')['files'].items():
        assert sha(scope_dir/name)==pin['sha256'] and (scope_dir/name).stat().st_size==pin['bytes']
    assert sha(scope_dir/'REVIEW.json')=='acdb0a0646d47b8773ce8ce9ec672eee157cb5f0b9ccccfd4ed05eb5ec0ceac1'
    state['added_baseline_three_view_scope_20261010']=dict(status='SOURCE_COMPATIBILITY_REVIEW_ONLY',report_path=(scope_dir/'REPORT.md').relative_to(ROOT).as_posix(),review_sha256=sha(scope_dir/'REVIEW.json'),new_evaluations=0,new_fits=0,dispatch=False,test=False,limitation='Four CNN methods need private identity bridges to their original strict checkers; LoGoFair native must preserve fitted DP state and virtual mapping. Its cache valid_native_prediction is FedAvg raw, not LoGo native. Backbone raw/shared diagnostics must be explicitly labelled; final primary endpoint remains undecided.')
hybrid32_root=ROOT/'tmp/celeba_hybrid_screen_execution_20261009/accepted_delta_after27_20261010/ROOT32_SUMMARY_ADOPTION.json'
if hybrid32_root.exists():
    proof=read(hybrid32_root)
    assert sha(hybrid32_root)=='6fcbdc41c7af01815e15995e0cb3688672404bf3b96844dc8d9bffa21028587f'
    assert proof['status']=='ROOT_HYBRID32_SUMMARY_ADOPTED' and proof['accepted_total']==32 and proof['all32_offserver_verified']
    summary_path=ROOT/proof['summary_path'];summary=read(summary_path)
    assert sha(summary_path)==proof['summary_sha256']=='46b5f8fdca9536166ed868e50d4c7bc2578f8a1100ffea97878cf044c95748ae'
    assert summary['selected_recipe']==proof['selected_recipe']=='CosineFairness_lam20.0_tau0.1_lr0.001'
    assert sha(ROOT/proof['independent_review_path'])==proof['independent_review_sha256']
    assert not proof['final_test'] and not proof['formal100_started'] and not proof['training_authorized_by_this_record']
    terminal_path=hybrid32_root.parent/'AUTHORIZED_SNAPSHOT.json';terminal=read(terminal_path)
    assert len(terminal['rows'])==32 and all(r['terminal'] and r['round']==r['result_rounds']==70 and not r['failures'] for r in terminal['rows'])
    assert 'EXITED' in terminal['service']['stdout'] and not terminal['screen_failure']
    state['hybrid_screen32_20261009'].update(status=proof['status'],selected_recipe=proof['selected_recipe'],selected_candidate=proof['selected_candidate'],
        summary_path=proof['summary_path'],summary_sha256=proof['summary_sha256'],summary_root_path=hybrid32_root.relative_to(ROOT).as_posix(),summary_root_sha256=sha(hybrid32_root),
        offserver_accepted70round_jobs=32,accepted70round_jobs=32,not_yet_accepted=0,terminal_workers=0,
        accuracy_champion=proof['accuracy_champion'],three_metric_Pareto=proof['three_metric_Pareto'],n_seeds=1,sample_SD=False,significance=False,
        latest_readonly_terminal_observation=dict(checked_utc=terminal['utc'],observed_complete=32,active=0,pending=0,failures=0,source_bound=True,
            active_rounds=[],snapshot_path=terminal_path.relative_to(ROOT).as_posix(),snapshot_sha256=sha(terminal_path)))
    state['active_services']=[name for name in state['active_services'] if name!='guardfed_celeba_hybrid_screen32']
aux_v2=ROOT/'tmp/celeba_aux_live_followup_20261010_afterHybrid32/attempt_v2'
if (aux_v2/'FILES_SHA256.json').exists():
    assert sha(aux_v2/'FILES_SHA256.json')=='5c20e5c77fa7dd50f30bb4d5c464e413ac8cb1568dd51cc40394233f66d69fa0'
    for name,pin in read(aux_v2/'FILES_SHA256.json')['files'].items():assert sha(aux_v2/name)==pin['sha256']
    measured=read(aux_v2/'FINDINGS.json');assert measured['new_accepted']==0 and not measured['remote_writes']
    state['auxiliary_readonly_observation_20261010']=dict(path=(aux_v2/'FINDINGS.json').relative_to(ROOT).as_posix(),sha256=sha(aux_v2/'FINDINGS.json'),
        utc=measured['actual_snapshot_utc'],FL_observed_terminal=38,FL_accepted=32,Hybrid_observed_terminal=32,Hybrid_accepted=32,
        gradient_observed_terminal=16,gradient_actual_workers=1,gradient_active_round=None,observation_is_atomic=False,acceptance_unchanged=True)
    state['flgmm_fullcoverage_v2_20261009']['latest_readonly_terminal_observation']=dict(checked_utc=measured['actual_snapshot_utc'],observed_complete=38,
        active=2,pending=56,failures=0,active_rounds=[10,10],source_bound=True,snapshot_sha256=measured['snapshot_sha256'])
    state['gradient64_validation_search_20261010']['latest_measured_observation']=dict(checked_utc=measured['actual_snapshot_utc'],terminal70_observed=16,
        actual_active_workers=1,active_rounds=[None],round_capture_complete=False,failures=0,snapshot_sha256=measured['snapshot_sha256'])
gradient8_root=ROOT/'tmp/celeba_gradient64_delta_after10_20261010/ROOT_ADOPTION_REVIEW.json'
if gradient8_root.exists():
    proof=read(gradient8_root);prior=state['gradient64_validation_search_20261010']
    assert sha(gradient8_root)=='83d22e5833fecddb398625893c457d2dfadb3b745bc4c5613d59cf87b6c02d39'
    assert proof['status']=='ROOT_GRADIENT64_EXACT8_ORIGINAL_STRICT_OFFSERVER_ADOPTED' and (proof['accepted_before'],proof['accepted_new'],proof['accepted_total'])==(10,8,18)
    assert proof['previous_root_sha256']==prior['root_adoption_sha256']=='99a07fefad41e0f441288f86c5335e175eb5992983fec047ab3a1c9f1eacea57'
    assert proof['accepted_ids'][:10]==prior['accepted_ids'] and len(set(proof['accepted_ids']))==18 and not proof['constant_negative_ids']
    assert sha(gradient8_root.parent/'OFFSERVER_ACCEPTANCE.json')==proof['offserver_sha256'] and sha(gradient8_root.parent/'ROOT_READY_HANDOFF.json')==proof['handoff_sha256']
    auth=read(gradient8_root.parent/'AUTHORIZED_SNAPSHOT.json');assert proof['accepted_new_ids']==auth['authorized_ids'] and proof['CPU110_released'] and proof['new_CNN']==proof['new_training']==0 and not proof['final_test']
    prior.update(offserver_accepted=18,root_adoption_path=gradient8_root.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(gradient8_root),accepted_ids=proof['accepted_ids'],
        delta_acceptance_snapshot=dict(path=(gradient8_root.parent/'SNAPSHOT.json').relative_to(ROOT).as_posix(),sha256=sha(gradient8_root.parent/'SNAPSHOT.json'),
            root_accepted_utc=proof['utc'],terminal70_observed=18,offserver_accepted=18,scope='Exact8 original strict/saved-state delta; prior10 prefix preserved, negative outcomes retained, no recipe selected.'))
aux_after7=ROOT/'tmp/celeba_aux_live_followup_afterHybrid7_20261010'
if (aux_after7/'FILES_SHA256.json').exists():
    assert sha(aux_after7/'FILES_SHA256.json')=='012b366ca0a3f433262420ef7122880c8477bc6c00bd86ca155be5f083b0ed3e'
    for name,pin in read(aux_after7/'FILES_SHA256.json')['files'].items():assert sha(aux_after7/name)==pin['sha256']
    measured=read(aux_after7/'FINDINGS.json');queues=measured['queues']
    assert measured['new_accepted']==0 and measured['no_collection_or_training'] and measured['all_source_data_identity_match'] and not measured['current_failures']
    assert sha(aux_after7/'SNAPSHOT.json')==measured['snapshot_sha256']
    state['auxiliary_readonly_observation_after_Hybrid7_20261010']=dict(path=(aux_after7/'FINDINGS.json').relative_to(ROOT).as_posix(),sha256=sha(aux_after7/'FINDINGS.json'),
        utc=measured['utc'],FL_observed_terminal=40,FL_root_accepted_at_input=38,gradient_observed_terminal=18,gradient_root_accepted_at_input=10,
        remaining620_remote_closed=56,remaining620_root_accepted_at_input=48,acceptance_unchanged_by_observation=True)
    state['flgmm_fullcoverage_v2_20261009']['latest_readonly_terminal_observation']=dict(checked_utc=measured['utc'],observed_complete=40,
        active=len(queues['FLGMM']['active']),pending=54,failures=0,active_rounds=[r['round'] for r in queues['FLGMM']['active']],source_bound=True,snapshot_sha256=measured['snapshot_sha256'])
    state['gradient64_validation_search_20261010']['latest_measured_observation']=dict(checked_utc=measured['utc'],terminal70_observed=18,
        actual_active_workers=len(queues['gradient64']['active']),active_rounds=[r['round'] for r in queues['gradient64']['active']],round_capture_complete=True,failures=0,snapshot_sha256=measured['snapshot_sha256'])
    state['mechanism_remaining620_valid_20261010']['latest_measured_observation']=dict(utc=measured['utc'],remote_strict_closed=56,
        progress_status=queues['remaining620']['progress_status'],waiting_id=queues['remaining620']['waiting_id'],snapshot_sha256=measured['snapshot_sha256'],source_bound=True)
hybrid_bound=ROOT/'tmp/celeba_hybrid_fullcoverage_root_binding_v3_20261010'
if (hybrid_bound/'ROOT_CANARY_STARTUP.json').exists():
    proof=read(hybrid_bound/'ROOT_CANARY_STARTUP.json');bound=read(hybrid_bound/'ROOT_BOUND_ADOPTION.json')
    assert sha(hybrid_bound/'ROOT_CANARY_STARTUP.json')=='d90c19c8aedf12a2ac58e37d653b638fcdae8e9b0044d3fcf6487af172a4bbd1'
    assert bound['status']=='ROOT_HYBRID100_BOUND_METADATA_ADOPTED' and (bound['new'],bound['reused'],bound['canaries'])==(96,4,7)
    assert sha(hybrid_bound/'BOUND_TRANSFER_VERIFICATION.json')==bound['bound_offserver_sha256']
    assert sha(ROOT/proof['observation_path'])==proof['observation_sha256']
    state['hybrid100_fullcoverage_20261010']=dict(status=proof['status'],planned_new=96,reused=4,planned_total=100,new_accepted=0,
        package_sha256=bound['package_sha256'],implementation_source_seal_sha256=bound['implementation_source_seal_sha256'],
        bound_root_path=(hybrid_bound/'ROOT_BOUND_ADOPTION.json').relative_to(ROOT).as_posix(),bound_root_sha256=sha(hybrid_bound/'ROOT_BOUND_ADOPTION.json'),
        canary_start_path=(hybrid_bound/'ROOT_CANARY_STARTUP.json').relative_to(ROOT).as_posix(),canary_start_sha256=sha(hybrid_bound/'ROOT_CANARY_STARTUP.json'),
        observed_utc=proof['observed_utc'],actual_worker_id=proof['actual_worker_id'],actual_round_at_startup=proof['actual_round'],
        canary_service='guardfed_celeba_hybrid_fullcoverage_canary',actual_service=None,canary_scope=7,canaries_offserver_adopted=0,
        CPU104_GPU0_single_worker=True,formal100_started=False,final_test=False,prior_metadata_failure_preserved=True,
        limits='Original32 selection unchanged; sum compatibility repaired without tolerance change. Actual gate startup is not7 closure or70round performance.')
    state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_hybrid_fullcoverage_canary']))
if (hybrid_bound/'ROOT_SEVEN_CANARY_CLOSURE.json').exists():
    closure=read(hybrid_bound/'ROOT_SEVEN_CANARY_CLOSURE.json')
    assert sha(hybrid_bound/'ROOT_SEVEN_CANARY_CLOSURE.json')=='eda03511ec037be8d06401886355b3c49929de3523bed272ca8df94e5799b05a'
    assert closure['status']=='ROOT_SEVEN_HYBRID_CANARIES_OFFSERVER_ADOPTED' and (closure['accepted_new_canaries'],closure['same_horizon_pairs'],closure['total_canary_runs'],closure['rounds'],closure['formal_table_samples'])==(7,2,7,3,0)
    assert sha(Path(closure['offserver_path']))==closure['offserver_sha256'] and sha(Path(closure['gate_path']))==closure['gate_sha256']
    state['hybrid100_fullcoverage_20261010'].update(status=closure['status'],canaries_offserver_adopted=7,canary_service_exited=True,
        canary_closure_path=(hybrid_bound/'ROOT_SEVEN_CANARY_CLOSURE.json').relative_to(ROOT).as_posix(),canary_closure_sha256=sha(hybrid_bound/'ROOT_SEVEN_CANARY_CLOSURE.json'),
        canary_offserver_sha256=closure['offserver_sha256'],canary_archive_members=closure['archive_members_verified'],canary_closure_utc=closure['checked_utc'],
        limits=closure['limitations'])
    state['active_services']=[s for s in state['active_services'] if s!='guardfed_celeba_hybrid_fullcoverage_canary']
if (hybrid_bound/'ROOT_COVERAGE_STARTUP.json').exists():
    proof=read(hybrid_bound/'ROOT_COVERAGE_STARTUP.json')
    assert sha(hybrid_bound/'ROOT_COVERAGE_STARTUP.json')=='136205c2b101c05f3fd0bbfa1fe16df4a364a5b4f64a140aa51ee754ceb50045'
    assert proof['status']=='ROOT_ACTUAL_HYBRID96_VALID_COVERAGE_STARTUP_AND_ROUNDS_VERIFIED' and proof['formal100_started'] and proof['new_accepted']==0
    assert sha(ROOT/proof['observation_path'])==proof['observation_sha256'] and sha(ROOT/proof['start_receipt_path'])==proof['start_receipt_sha256']
    state['hybrid100_fullcoverage_20261010'].update(status=proof['status'],formal100_started=True,actual_service='guardfed_celeba_hybrid_fullcoverage',
        observed_utc=proof['observed_utc'],actual_worker_id=proof['actual_worker_id'],actual_worker_pid=proof['actual_worker_pid'],actual_round_at_startup=proof['actual_round'],
        coverage_start_path=(hybrid_bound/'ROOT_COVERAGE_STARTUP.json').relative_to(ROOT).as_posix(),coverage_start_sha256=sha(hybrid_bound/'ROOT_COVERAGE_STARTUP.json'),
        physical_gpu_uuid=proof['physical_gpu_uuid'],formal_authorization_sha256=proof['authorization_sha256'],canary_previous_authorization_preserved=True,limits=proof['limits'])
    state['active_services']=list(dict.fromkeys(state['active_services']+['guardfed_celeba_hybrid_fullcoverage']))
hybrid_first1=ROOT/'tmp/celeba_hybrid_first1_root_adoption_20261010/ROOT_ADOPTION.json'
if hybrid_first1.exists():
    proof=read(hybrid_first1)
    assert sha(hybrid_first1)=='04d62c367609d1c6d079ff538f45a575bff368944d4833f47fd1cf28159c5fa4'
    assert proof['status']=='ROOT_HYBRID_FIRST1_ORIGINAL_STRICT_OFFSERVER_RESTORE_CHAIN_ADOPTED'
    assert (proof['prior_accepted'],proof['new_accepted'],proof['cumulative_accepted'],proof['planned_new'],proof['reused_separate'],proof['archive_members'],proof['saved_tensor_count'])==(0,1,1,96,4,188,8)
    assert proof['rounds']==70 and proof['n_eval']==19867 and proof['all_metrics_same_checkpoint'] and not proof['final_test']
    assert sha(ROOT/proof['independent_review_path'])==proof['independent_review_sha256']
    assert sha(ROOT/proof['offserver_path'])==proof['offserver_sha256']
    state['hybrid100_fullcoverage_20261010'].update(status='FORMAL96_RUNNING_FIRST1_STRICT_OFFSERVER_ROOT_ADOPTED',new_accepted=1,
        accepted_ids=proof['accepted_new_ids'],first_delta_root_path=hybrid_first1.relative_to(ROOT).as_posix(),first_delta_root_sha256=sha(hybrid_first1),
        first_delta_archive_sha256=proof['archive_sha256'],first_delta_offserver_sha256=proof['offserver_sha256'],first_delta_utc=proof['utc'],
        limits='One of96 new70-round validation records accepted from original strict and188-member offserver restore chain; four reuse separate, seven gates zero formal samples. No scenario mean or runtime equivalence claim. Remaining95, final evaluation and manuscript incomplete.')
hybrid9=ROOT/'tmp/celeba_hybrid_native9_root_adoption_20261011/ROOT_ADOPTION.json'
if hybrid9.exists():
    proof=read(hybrid9)
    assert sha(hybrid9)=='1434b40de5116bf3d53bc5a6ae2bd3b90f4e54222f23ad099ff72b1fd3ef1775'
    assert proof['status']=='ROOT_HYBRID_EXACT8_ORIGINAL_STRICT_OFFSERVER_CHAIN_ADOPTED'
    assert (proof['prior_accepted'],proof['new_accepted'],proof['cumulative_accepted'],proof['archive_members'],proof['saved_tensor_count'])==(1,8,9,272,64)
    assert proof['previous_root_sha256']==sha(hybrid_first1) and proof['accepted_ids'][:1]==state['hybrid100_fullcoverage_20261010']['accepted_ids']
    assert len(set(proof['accepted_ids']))==9 and proof['archive_all_members_rehashed'] and proof['all_metrics_same_terminal_checkpoint']
    assert sha(Path(proof['offserver_path']))==proof['offserver_sha256'] and not proof['final_test']
    state['hybrid100_fullcoverage_20261010'].update(status='FORMAL96_RUNNING_NATIVE9_STRICT_OFFSERVER_ROOT_ADOPTED',new_accepted=9,
        accepted_ids=proof['accepted_ids'],latest_delta_root_path=hybrid9.relative_to(ROOT).as_posix(),latest_delta_root_sha256=sha(hybrid9),
        latest_delta_archive_sha256=proof['archive_sha256'],latest_delta_offserver_sha256=proof['offserver_sha256'],latest_delta_utc=proof['utc'],
        limits='Nine of96 new70-round validation records accepted; exact8 increment passes original strict and272-member offserver chain. Four reuse and seven short gates separate. IID Benign ten-seed table requires separate statistical adoption; other scenarios and final test incomplete.')
hybrid_scene=ROOT/'outputs/guardfed_tables/celeba_hybrid_IID_Benign10_20261011/ROOT_VERIFICATION.json'
if hybrid_scene.exists():
    proof=read(hybrid_scene)
    assert sha(hybrid_scene)=='6de83f1f9b643980ec6cefaf21fd3d1c12af864b6abe24bea127b827144e5d6f'
    assert proof['root_adopted'] and proof['native_root_sha256']==sha(hybrid9)
    assert (proof['complete_scenes'],proof['individual_records'],proof['mean_SD_scalars_recomputed'],proof['display_cells_checked'])==(1,10,18,9)
    assert proof['seed_panels']==[10,9,6] and proof['ddof']==1 and not proof['final_test'] and not proof['whole100_complete']
    for name,digest in proof['files_sha256'].items():assert sha(hybrid_scene.parent/name)==digest
    state['hybrid100_fullcoverage_20261010']['native_IID_Benign_table']=dict(root_proof_path=hybrid_scene.relative_to(ROOT).as_posix(),root_proof_sha256=sha(hybrid_scene),table_path=proof['table_path'],complete_scenes=1,individual_records=10,seed_panels=[10,9,6],same_terminal_checkpoint=True,whole100_complete=False,final_test=False)
A40_table=TRAIN/'celeba_mechanism_v1/three_view_A_IID_four_scenes40_20261010'
if (A40_table/'ROOT_VERIFICATION.json').exists():
    proof=read(A40_table/'ROOT_VERIFICATION.json')
    assert sha(A40_table/'ROOT_VERIFICATION.json')=='edf4a520c3b797e2376005ef132bc21021ab5dbd0962b4105299d67309f8ebf9'
    assert proof['status']=='ROOT_A40_FOUR_COMPLETE_IID_SCENE_THREE_VIEW_TABLE_ADOPTED' and proof['root_adoption']
    assert (proof['paired_models'],proof['complete_scenes'],proof['preserved_records'])==(40,4,80)
    assert (proof['mean_SD_scalars_recomputed'],proof['display_cells'],proof['metrics_from_group_counts'],proof['base_integer_confusion_counts_checked'])==(648,324,720,1920)
    assert proof['source_acceptance_sha256']==sha(root_A40) and not proof['test']
    for name,digest in proof['files_sha256'].items():assert sha(A40_table/name)==digest
    assert sha(ROOT/proof['independent_review_path'])==proof['independent_review_sha256']
    state['celeba_mechanism_v1'].update(A_three_view_four_scene_table=dict(status=proof['status'],table_path=proof['canonical_table'],root_proof_path=(A40_table/'ROOT_VERIFICATION.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(A40_table/'ROOT_VERIFICATION.json'),complete_scenes=4,paired_models=40,seed_panels=[10,9,6],final_test=False,incorporated_into_full_rebuttal=False),
        three_view_scope_limit='U100/C100 ten-scene and A40 four-IID-scene tables independently adopted. All10/9/6 panels, old40 JSON object bytes/order,324 scalars/162 cells and unfavorable effects retained. Six other A scenes, other controls, final evaluation and submitted manuscript remain incomplete; no necessity, causal or significance claim.')
    state['mechanism_remaining620_valid_20261010']['A_four_scene_table_adopted']=True
cpu_diagnosis=CHECKS/'CPU_THROUGHPUT_DIAGNOSIS_20261010T1040.json'
if cpu_diagnosis.exists():
    assert sha(cpu_diagnosis)=='1477bae7433ffd36e33ef18f6eb54120936ade75478fa98cf071d588fbb8bcf1'
    state['latest_CPU_throughput_diagnosis_20261010']=dict(path=cpu_diagnosis.relative_to(ROOT).as_posix(),sha256=sha(cpu_diagnosis),
        interpretation='Measured27.16min mean13.63/122.88cores, GPU100% at both endpoints and progress in all five queues support GPU bottleneck; no continuous-saturation or optimal-concurrency claim. Frozen8/FP32 retained; useful acceptance/table work uses spare CPU.')
reply_A40_LoGo100=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A40_LoGo100_20261010'
if (reply_A40_LoGo100/'ROOT_REVIEW.json').exists():
    proof=read(reply_A40_LoGo100/'ROOT_REVIEW.json')
    assert sha(reply_A40_LoGo100/'ROOT_REVIEW.json')=='f506eba0e28bdd69a95f8dc90490be08ec0278fac889409e2e7b6cdd609da1cd'
    assert proof['status']=='ROOT_COMPLETE_A40_LOGO100_NATIVE1000_REBUTTAL_AND_INSERTION_CANDIDATES_ADOPTED_FOR_AUTHOR_REVIEW'
    assert proof['source_seal_sha256']==sha(reply_A40_LoGo100/'FILES_SHA256.json')=='93fe49079c9a439448da236c0e5c231b0b3c338ae375a06273cdf4eb03a3c71b'
    assert (proof['original_comments'],proof['A_complete_scenes'],proof['A_paired_models'],proof['LoGoFair_native_records'],proof['native_comparison_records'])==(24,4,40,100,1000)
    assert (proof['scalar_pointer_checks'],proof['mean_SD_cells'],proof['fact_pointers'],proof['A_direction_checks'],proof['links_checked'])==(118,59,69,108,88)
    assert proof['actual_root_command_exit']==0 and proof['author_review_only'] and not proof['final_test'] and not proof['manuscript_applied']
    assert sha(ROOT/proof['actual_root_command_path'])==proof['actual_root_command_sha256']
    assert sha(ROOT/proof['independent_review_path'])==proof['independent_review_sha256']
    for name,want in proof['documents_sha256'].items():assert sha(reply_A40_LoGo100/name)==want
    state['latest_rebuttal_draft'].update(status=proof['status'],entry=(reply_A40_LoGo100/'rebuttal_integrated_20261009.md').relative_to(ROOT).as_posix(),
        manuscript_candidate=(reply_A40_LoGo100/'manuscript_insertions_integrated_20261009.md').relative_to(ROOT).as_posix(),
        root_proof_path=(reply_A40_LoGo100/'ROOT_REVIEW.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(reply_A40_LoGo100/'ROOT_REVIEW.json'),
        source_seal_sha256=proof['source_seal_sha256'],complete_C_scenes=10,A_complete_scenes=4,A_paired_models=40,LoGoFair_native_records=100,native_comparison_records=1000,
        scalar_pointer_checks=118,numeric_cells=59,fact_pointers=69,links_checked=88,changed_passages=8,A_direction_checks=108,
        independent_review_path=proof['independent_review_path'],independent_review_sha256=proof['independent_review_sha256'],
        whole_rebuttal_complete=False,manuscript_applied=False,author_review_only=True)
    state['celeba_mechanism_v1']['A_three_view_four_scene_table']['incorporated_into_full_rebuttal']=True
    reply_progress_note='最新完整作者审阅稿已纳入U/C各十场景、A四IID场景、LoGoFair100及十方法native千格表；24原意见逐字、两旧全文可逆恢复、59均值SD单元/118数值pointer/69事实/108方向/88链接经root及独立语义复核通过。A40全部指标取舍和10/9/6面板保持；七方法完整覆盖、六机制变体、P1–P6、最终评价及正文仍未完成'
gradient23=ROOT/'tmp/celeba_gradient64_delta_after18_20261010/ROOT_ADOPTION_REVIEW.json'
if gradient23.exists():
    proof=read(gradient23); prior=state['gradient64_validation_search_20261010']
    assert sha(gradient23)=='fe35ea95d58697d05c53635a38771c6180ccde23821b37972a330ff7020333d4'
    assert (proof['accepted_before'],proof['accepted_new'],proof['accepted_total'])==(18,5,23)
    assert proof['previous_root_sha256']==prior['root_adoption_sha256'] and proof['accepted_ids'][:18]==prior['accepted_ids']
    assert len(set(proof['accepted_ids']))==23 and proof['all_negative_results_retained'] and not proof['final_test']
    assert sha(gradient23.parent/'OFFSERVER_ACCEPTANCE.json')==proof['offserver_sha256']
    prior.update(offserver_accepted=23,root_adoption_path=gradient23.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(gradient23),accepted_ids=proof['accepted_ids'],latest_delta_constant_negative_ids=proof['constant_negative_ids'])
gradient32=ROOT/'tmp/celeba_gradient64_delta_after23_closed32_20261011/ROOT_ADOPTION_REVIEW.json'
if gradient32.exists():
    proof=read(gradient32);prior=state['gradient64_validation_search_20261010']
    assert sha(gradient32)=='185f0fa292be27b6921776818d13871cb8a5b787a765867b802362991d1f9a81'
    assert (proof['accepted_before'],proof['accepted_new'],proof['accepted_total'])==(23,9,32)
    assert proof['previous_root_sha256']==prior['root_adoption_sha256'] and proof['accepted_ids'][:23]==prior['accepted_ids']
    assert len(set(proof['accepted_ids']))==32 and all(x.startswith('FedNGA_') for x in proof['accepted_ids'])
    assert proof['all_negative_results_retained'] and not proof['final_test'] and not proof['screen64_complete']
    assert sha(gradient32.parent/'OFFSERVER_ACCEPTANCE.json')==proof['offserver_sha256']
    assert proof['archive_members_root_verified']==167 and proof['raw_files_root_verified']==170 and len(proof['constant_negative_ids'])==3
    prior.update(offserver_accepted=32,root_adoption_path=gradient32.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(gradient32),accepted_ids=proof['accepted_ids'],latest_delta_constant_negative_ids=proof['constant_negative_ids'],FedNGA_screen32_accepted=True,Huber_screen_accepted=0)
gradient39=ROOT/'tmp/gradient_native_after32_20261011/ROOT_ADOPTION_REVIEW.json'
if gradient39.exists():
    proof=read(gradient39);prior=state['gradient64_validation_search_20261010']
    assert sha(gradient39)=='850b91b68e01f98f53a82f26897a43a486857b69fd4335c2e8e8e79fba9e1e3b'
    assert proof['status']=='ROOT_GRADIENT64_EXACT7_ORIGINAL_STRICT_OFFSERVER_ADOPTED'
    assert (proof['accepted_before'],proof['accepted_new'],proof['accepted_total'])==(32,7,39)
    assert proof['previous_root_sha256']==prior['root_adoption_sha256'] and proof['accepted_ids'][:32]==prior['accepted_ids']
    assert len(set(proof['accepted_ids']))==39 and all(x.startswith('Huber_') for x in proof['accepted_new_ids'])
    assert proof['all_negative_results_retained'] and not proof['final_test'] and not proof['screen64_complete']
    assert sha(gradient39.parent/'OFFSERVER_ACCEPTANCE.json')==proof['offserver_sha256']
    assert sha(gradient39.parent/'ROOT_READY_HANDOFF.json')==proof['handoff_sha256']
    assert (proof['archive_members_root_verified'],proof['raw_files_root_verified'],len(proof['constant_negative_ids']))==(153,156,7)
    prior.update(offserver_accepted=39,root_adoption_path=gradient39.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(gradient39),accepted_ids=proof['accepted_ids'],latest_delta_constant_negative_ids=proof['constant_negative_ids'],FedNGA_screen32_accepted=True,Huber_screen_accepted=7,recipe_selected=False)
replay251=ROOT/'tmp/celeba_mechanism_remaining620_after240_root_adoption_20261010/ROOT_ADOPTION.json'
if replay251.exists():
    proof=read(replay251); remaining=state['mechanism_remaining620_valid_20261010']; main=state['celeba_mechanism_v1']
    assert sha(replay251)=='edd16e71d6fef9f6e2fc7fd15a7b824d2dfb2a7f809290990878b23477a1e838'
    assert (proof['prior_accepted'],proof['new_accepted'],proof['cumulative_accepted'],proof['remaining620_new_accepted'])==(240,11,251,71)
    index=read(ROOT/proof['records_index_path']); assert sha(ROOT/proof['records_index_path'])==proof['records_index_sha256']
    assert index['all_ids'][:240]==main['three_view_accepted_ids'] and len(set(index['all_ids']))==251
    assert (proof['independent_metrics'],proof['independent_counts'],proof['prediction_rules'],proof['native_max_abs_difference'])==(99,264,33,0)
    remaining.update(new_offserver_accepted=71,root_adoption_path=replay251.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(replay251),accepted_ids=remaining['accepted_ids']+proof['accepted_new_ids'],A_complete_scenes=proof['complete_A_scenes'],A_partial_scenes=proof['partial_A_scenes'])
    main.update(three_view_new_models_accepted=251,three_view_new_models_offserver_verified=251,three_view_accepted_ids=index['all_ids'],three_view_counts_by_variant={'minus_U':100,'minus_C':100,'minus_A':51},three_view_root_proof_sha256=sha(replay251),
        latest_three_view_index=proof['records_index_path'],latest_three_view_index_sha256=proof['records_index_sha256'],
        three_view_scope_limit='251 original strict/offserver/native-identity replays adopted: U100/C100/A51. All five A IID scenarios have ten seeds; non-IID Benign has one only. Existing A40 four-IID-scene statistics remain the reviewed table; new A scenario statistics await aggregation. Five non-IID A scenarios, other variants, final evaluation and submitted manuscript remain unfinished.')
bridge_review=ROOT/'tmp/celeba_added_cnn_three_view_bridge_20261010/ROOT_SOURCE_REVIEW.json'
if bridge_review.exists():
    proof=read(bridge_review)
    assert sha(bridge_review)=='be5e6c949254960f55b765ec976f1e681a26450416f156ca237965dedc87fe69'
    assert proof['source_adopted'] and proof['actual_tests_replayed_by_root']==64 and proof['new_three_view_scientific_results']==0
    state['added_CNN_three_view_bridge_20261010']=dict(proof,root_review_path=bridge_review.relative_to(ROOT).as_posix(),root_review_sha256=sha(bridge_review))
native_pdf=ROOT/'outputs/guardfed_tables/celeba_ten_method_native_pdf_20261010/ROOT_REVIEW.json'
if native_pdf.exists():
    proof=read(native_pdf)
    assert sha(native_pdf)=='f187b7a3efc721218fa393048ecabf2ea6a7fbbd4bea0ef263e57ad972ee7611'
    assert sha(ROOT/proof['pdf_path'])==proof['pdf_sha256'] and proof['pages']==3 and proof['displayed_mean_SD_pairs']==900
    assert proof['source_bytes_unchanged'] and not proof['final_test'] and not proof['full17_complete']
    state['celeba_native_ten_method_PDF_20261010']=dict(proof,root_review_path=native_pdf.relative_to(ROOT).as_posix(),root_review_sha256=sha(native_pdf))
five_snapshot=CHECKS/'root_five_queue_20261010T111336Z.raw.json'
if five_snapshot.exists():
    obs=read(five_snapshot)
    assert sha(five_snapshot)=='a07c978fe3055f832f76f2ec323c894a18f5cb7be306c63332d3846cec2f1611'
    assert obs['status']=='SINGLE_FIVE_STAGE_READONLY_HEALTH_NOT_ACCEPTANCE' and not obs['source_data_binding']['changed_paths']
    state['latest_five_queue_readonly_observation']=dict(path=five_snapshot.relative_to(ROOT).as_posix(),sha256=sha(five_snapshot),utc=obs['utc'],counts_are_observation_only=True,
        FLGMM_terminal=len(obs['FLGMM']['terminal_ids']),gradient_terminal=len(obs['gradient64']['terminal_ids']),remaining620_closed=obs['remaining620']['remote_closed_n'],Hybrid_terminal=len(obs['Hybrid96']['terminal_records']),new_acceptance=0)
    for key, stage in [('flgmm_fullcoverage_v2_20261009','FLGMM'),('gradient64_validation_search_20261010','gradient64')]:
        o=obs[stage]; state[key]['latest_measured_observation']=dict(checked_utc=obs['utc'],terminal70_observed=len(o['terminal_ids']),active_rounds=[r['round'] for r in o['active']],snapshot_path=five_snapshot.relative_to(ROOT).as_posix(),snapshot_sha256=sha(five_snapshot),acceptance_unchanged_by_observation=True)
    state['mechanism_remaining620_valid_20261010']['latest_measured_observation']=dict(utc=obs['utc'],remote_strict_closed=obs['remaining620']['remote_closed_n'],snapshot_sha256=sha(five_snapshot),source_bound=True)
    state['hybrid100_fullcoverage_20261010']['latest_measured_observation']=dict(utc=obs['utc'],terminal70_observed=len(obs['Hybrid96']['terminal_records']),snapshot_sha256=sha(five_snapshot),acceptance_unchanged_by_observation=True)
baseline_snapshot=ROOT/'tmp/celeba_flgmm_fullcoverage_delta_after38_20261010/SNAPSHOT.json'
if baseline_snapshot.exists():
    obs=read(baseline_snapshot); assert sha(baseline_snapshot)=='a5f70d5d5ab72cd33d353cc0892e0f9bfab24d86d26dff9276dd311e3945b64c'
    for key,stage in [('flgmm_fullcoverage_v2_20261009','FLGMM'),('gradient64_validation_search_20261010','gradient64')]:
        o=obs[stage]; assert not o['failure_paths'] and not o['source']['changed_members']
        state[key]['latest_measured_observation']=dict(checked_utc=obs['utc'],terminal70_observed=len(o['terminal_ids']),active_rounds=[r['round'] for r in o['active']],snapshot_path=baseline_snapshot.relative_to(ROOT).as_posix(),snapshot_sha256=sha(baseline_snapshot),acceptance_unchanged_by_observation=True)
if replay251.exists():
    p=ROOT/'tmp/celeba_remaining620_after240_transport_20261010/PREFLIGHT.stdout.json'; h=read(replay251)
    assert sha(p)=='59a67254689dd562af7e4953fc3822ac7e3efd78967dc46bcfd3448c2423c6db'
    o=read(p); assert not o['failure_paths'] and o['actual_remote_closed']>=71
    state['mechanism_remaining620_valid_20261010']['latest_measured_observation']=dict(utc=o['utc'],remote_strict_closed=o['actual_remote_closed'],snapshot_path=p.relative_to(ROOT).as_posix(),snapshot_sha256=sha(p),source_bound=True,acceptance_unchanged_by_observation=True)
A50_table=TRAIN/'celeba_mechanism_v1/three_view_A_IID_five_scenes50_20261010'
if (A50_table/'ROOT_VERIFICATION.json').exists():
    proof=read(A50_table/'ROOT_VERIFICATION.json')
    assert sha(A50_table/'ROOT_VERIFICATION.json')=='2ab91a854691624c79124a5eccbb43efe30541474cb7ef542b76db300554f7c4'
    assert proof['root_adoption'] and (proof['paired_models'],proof['complete_scenes'],proof['preserved_records'])==(50,5,100)
    assert (proof['mean_SD_scalars_recomputed'],proof['display_cells'],proof['metrics_from_group_counts'],proof['base_integer_confusion_counts_checked'])==(972,486,900,2400)
    assert proof['source_acceptance_sha256']==sha(replay251) and proof['actual_root_command_exit']==0 and not proof['test']
    for name,digest in proof['files_sha256'].items():assert sha(A50_table/name)==digest
    assert sha(ROOT/proof['independent_review_path'])==proof['independent_review_sha256']
    state['celeba_mechanism_v1'].update(A_three_view_five_scene_table=dict(status=proof['status'],table_path=proof['canonical_table'],root_proof_path=(A50_table/'ROOT_VERIFICATION.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(A50_table/'ROOT_VERIFICATION.json'),complete_scenes=5,paired_models=50,seed_panels=[10,9,6],mean_SD_scalars=972,display_cells=486,final_test=False,incorporated_into_full_rebuttal=False),
        three_view_scope_limit='U100/C100 cover all ten scenes; A50 five-IID-scene table is independently adopted, with one non-IID singleton retained only in the251 index. OldA40 exact; all three views,10/9/6 panels, seed-first IID aggregate and tradeoffs retained. Five non-IID A scenes, remaining controls, final evaluation and submitted manuscript incomplete.')
    state['mechanism_remaining620_valid_20261010']['A_five_scene_table_adopted']=True
replay260=ROOT/'tmp/celeba_mechanism_remaining620_after251_root_adoption_20261010/ROOT_ADOPTION.json'
if replay260.exists():
    proof=read(replay260);remaining=state['mechanism_remaining620_valid_20261010'];main=state['celeba_mechanism_v1']
    assert sha(replay260)=='e0a7ebef21f5cf9dad24fbb92ee4a874efc10fbb7f2ed7f507d9b4dfaef85200'
    assert (proof['prior_accepted'],proof['new_accepted'],proof['cumulative_accepted'],proof['remaining620_new_accepted'])==(251,9,260,80)
    index=read(ROOT/proof['records_index_path']);assert sha(ROOT/proof['records_index_path'])==proof['records_index_sha256']
    assert index['all_ids'][:251]==main['three_view_accepted_ids'] and len(set(index['all_ids']))==260
    assert (proof['independent_metrics'],proof['independent_counts'],proof['prediction_rules'],proof['native_max_abs_difference'])==(81,216,27,0)
    remaining.update(new_offserver_accepted=80,root_adoption_path=replay260.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(replay260),accepted_ids=remaining['accepted_ids']+proof['accepted_new_ids'],A_complete_scenes=proof['complete_A_scenes'],A_partial_scenes=[])
    main.update(three_view_new_models_accepted=260,three_view_new_models_offserver_verified=260,three_view_accepted_ids=index['all_ids'],three_view_counts_by_variant={'minus_U':100,'minus_C':100,'minus_A':60},three_view_root_proof_sha256=sha(replay260),latest_three_view_index=proof['records_index_path'],latest_three_view_index_sha256=proof['records_index_sha256'],three_view_scope_limit='260 original strict/offserver/native-identity replays adopted: U100/C100/A60. Five IID A scenes plus non-IID Benign each have ten seeds; four other non-IID A scenes and other variants remain incomplete. A60 new table awaits separate adoption. Native264 is a separate cutoff; no final test or submitted-manuscript completion.')
A60_table=TRAIN/'celeba_mechanism_v1/three_view_A_six_scenes60_20261011'
if (A60_table/'ROOT_VERIFICATION.json').exists():
    proof=read(A60_table/'ROOT_VERIFICATION.json')
    assert sha(A60_table/'ROOT_VERIFICATION.json')=='71eda2082961dff461b9507b534168757be7d579dee856f202383e5607bda204'
    assert proof['root_adoption'] and (proof['paired_models'],proof['complete_scenes'],proof['preserved_records'])==(60,6,120)
    assert (proof['mean_SD_scalars_recomputed'],proof['display_cells'],proof['metrics_from_group_counts'],proof['base_integer_confusion_counts_checked'])==(1134,567,1080,2880)
    assert proof['source_acceptance_sha256']==sha(replay260) and proof['actual_root_command_exit']==0 and not proof['test']
    for name,digest in proof['files_sha256'].items():assert sha(A60_table/name)==digest
    main.update(A_three_view_six_scene_table=dict(status=proof['status'],table_path=proof['canonical_table'],root_proof_path=(A60_table/'ROOT_VERIFICATION.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(A60_table/'ROOT_VERIFICATION.json'),complete_scenes=6,complete_IID_scenes=5,complete_nonIID_scenes=['Benign'],paired_models=60,seed_panels=[10,9,6],mean_SD_scalars=1134,display_cells=567,final_test=False,incorporated_into_full_rebuttal=False),three_view_scope_limit='U100/C100 cover all ten scenes; A60 table covers five IID scenes plus non-IID Benign, each ten paired seeds. Original A50 records/statistics and IID-only seed-first aggregate bytes are preserved. Four non-IID A scenes, other controls, remaining baseline coverage, final evaluation and submitted manuscript remain incomplete.')
    remaining['A_six_scene_table_adopted']=True
replay280=ROOT/'tmp/celeba_mechanism_remaining620_after260_root_adoption_20261011/ROOT_ADOPTION.json'
if replay280.exists():
    proof=read(replay280);remaining=state['mechanism_remaining620_valid_20261010'];main=state['celeba_mechanism_v1']
    assert sha(replay280)=='3da6be148e4ba649a35f2b6b130022051019d7446f519472efdf3bd4a851473a'
    assert proof['status']=='ROOT_AFTER260_EXACT20_SAVED_ARRAYS_NATIVE280_ADOPTED'
    assert (proof['prior_accepted'],proof['new_accepted'],proof['cumulative_accepted'],proof['remaining620_new_accepted'])==(260,20,280,100)
    index=read(ROOT/proof['records_index_path']);assert sha(ROOT/proof['records_index_path'])==proof['records_index_sha256']=='ed61c7424a510a70a1f695579341a41cea125b3024067cad874f66af4e1e53b2'
    assert len(index['all_ids'])==len(set(index['all_ids']))==280 and index['all_ids'][:260]==main['three_view_accepted_ids']
    assert index['new_ids']==index['all_ids'][260:]==proof['accepted_new_ids'] and len(proof['accepted_new_ids'])==20
    assert remaining['new_offserver_accepted']==80 and len(remaining['accepted_ids'])==80
    assert (proof['independent_metrics'],proof['independent_counts'],proof['prediction_rules'],proof['native_max_abs_difference'])==(180,480,60,0)
    assert proof['original260_unchanged'] and not proof['new_scene_table_created'] and not proof['test']
    remaining.update(new_offserver_accepted=100,root_adoption_path=replay280.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(replay280),accepted_ids=remaining['accepted_ids']+proof['accepted_new_ids'],latest_accepted_new_ids=proof['accepted_new_ids'],A_complete_scenes=proof['complete_A_scenes'],A_partial_scenes=[],A_eight_scene_table_adopted=False)
    assert len(remaining['accepted_ids'])==len(set(remaining['accepted_ids']))==100
    main.update(three_view_new_models_accepted=280,three_view_new_models_offserver_verified=280,three_view_accepted_ids=index['all_ids'],three_view_counts_by_variant={'minus_U':100,'minus_C':100,'minus_A':80},three_view_root_proof_sha256=sha(replay280),latest_three_view_index=proof['records_index_path'],latest_three_view_index_sha256=proof['records_index_sha256'],three_view_scope_limit='280 original strict/offserver/native-identity replays adopted: U100/C100/A80. A80 contains five IID scenes plus non-IID Benign/F Flip/FedSA, each ten paired seeds; its eight-scene table awaits separate adoption. Non-IID A S-DFA/Sp-DFA and other control variants remain incomplete. The adopted A60 table and complete author-review reply retain their six-scene scope; no final test or submitted-manuscript completion.')

A80_table=TRAIN/'celeba_mechanism_v1/three_view_A_eight_scenes80_20261011'
if (A80_table/'ROOT_VERIFICATION.json').exists():
    proof=read(A80_table/'ROOT_VERIFICATION.json')
    assert sha(A80_table/'ROOT_VERIFICATION.json')=='d2245e4d9de68b415931dccc7d1f70c39a871c226c59d9973675dfc1d3cc1bc7'
    assert proof['root_adoption'] and (proof['paired_models'],proof['complete_scenes'],proof['preserved_records'])==(80,8,160)
    assert proof['complete_nonIID_scenes']==['Benign','F Flip','FedSA']
    assert (proof['mean_SD_scalars_recomputed'],proof['display_cells'],proof['metrics_from_group_counts'],proof['base_integer_confusion_counts_checked'])==(1458,729,1440,3840)
    assert proof['source_acceptance_sha256']==sha(replay280) and proof['actual_root_command_exit']==0 and not proof['test']
    assert proof['old120_record_bytes_order_exact'] and proof['old972_scalars_exact'] and proof['old486_cells_exact'] and proof['IID_seed_first_JSON_bytes_exact']
    for name,digest in proof['files_sha256'].items():assert sha(A80_table/name)==digest
    main.update(A_three_view_eight_scene_table=dict(status=proof['status'],table_path=proof['canonical_table'],root_proof_path=(A80_table/'ROOT_VERIFICATION.json').relative_to(ROOT).as_posix(),root_proof_sha256=sha(A80_table/'ROOT_VERIFICATION.json'),complete_scenes=8,complete_IID_scenes=5,complete_nonIID_scenes=['Benign','F Flip','FedSA'],paired_models=80,seed_panels=[10,9,6],mean_SD_scalars=1458,display_cells=729,final_test=False,incorporated_into_full_rebuttal=False),three_view_scope_limit='U100/C100 cover all ten scenes; A80 table covers five IID plus non-IID Benign/F Flip/FedSA, each ten paired seeds. Old120 records/972 statistics/486 cells and IID aggregate bytes are preserved. Raw FedSA deletion improves all three ten-seed means; native/shared tradeoffs and all negative results remain. Non-IID A S-DFA/Sp-DFA, other controls, remaining baseline coverage, final evaluation and submitted manuscript are incomplete.')
    remaining['A_eight_scene_table_adopted']=True
replay288=ROOT/'tmp/celeba_mechanism_remaining620_after280_root_adoption_20261011/ROOT_ADOPTION.json'
if replay288.exists():
    proof=read(replay288);remaining=state['mechanism_remaining620_valid_20261010'];main=state['celeba_mechanism_v1']
    assert sha(replay288)=='7f53a03f055e86fbe2e4fd8a9b92edffa819ec261978813bcb714130b144a5e7'
    assert proof['status']=='ROOT_AFTER280_EXACT8_SAVED_ARRAYS_NATIVE288_ADOPTED'
    assert (proof['prior_accepted'],proof['new_accepted'],proof['cumulative_accepted'],proof['remaining620_new_accepted'])==(280,8,288,108)
    index=read(ROOT/proof['records_index_path']);assert sha(ROOT/proof['records_index_path'])==proof['records_index_sha256']=='15676070f43acab92b7d72aea2235432c459a95a3c6d33c08368f23ad19bb48d'
    assert index['all_ids'][:280]==main['three_view_accepted_ids'] and index['all_ids'][280:]==proof['accepted_new_ids'] and len(set(index['all_ids']))==288
    assert (proof['independent_metrics'],proof['independent_counts'],proof['prediction_rules'],proof['native_members_rehashed'])==(72,192,24,16)
    assert proof['original280_unchanged'] and proof['native_max_abs_difference']==0 and not proof['new_scene_table_created'] and not proof['test']
    assert remaining['new_offserver_accepted']==100
    remaining.update(new_offserver_accepted=108,root_adoption_path=replay288.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(replay288),accepted_ids=remaining['accepted_ids']+proof['accepted_new_ids'],latest_accepted_new_ids=proof['accepted_new_ids'],A_complete_scenes=proof['complete_A_scenes'],A_partial_scenes=proof['partial_A_scenes'])
    assert len(remaining['accepted_ids'])==len(set(remaining['accepted_ids']))==108
    main.update(three_view_new_models_accepted=288,three_view_new_models_offserver_verified=288,three_view_accepted_ids=index['all_ids'],three_view_counts_by_variant={'minus_U':100,'minus_C':100,'minus_A':88},three_view_root_proof_sha256=sha(replay288),latest_three_view_index=proof['records_index_path'],latest_three_view_index_sha256=proof['records_index_sha256'],three_view_scope_limit='288 original strict/offserver/native-identity replays adopted: U100/C100/A88. A80 eight-scene table remains the complete-scene table; eight additional non-IID S-DFA seeds are coverage only, excluded from complete-scene means. Old280 remain exact; other controls, final evaluation and submitted manuscript incomplete.')
growth=CHECKS/'ROOT_FIVE_QUEUE_GROWTH_20261010T1606.json'
if growth.exists():
    proof=read(growth)
    assert sha(growth)=='c91d71a96ee01496a49954c4a9cf44075aa174cba3ee62f29af25018fe729de0'
    assert proof['status']=='ROOT_FIVE_LIVE_QUEUES_IDENTITY_AND_PROGRESS_VERIFIED' and not proof['source_changes'] and not proof['failures']
    pin=proof['snapshots'][-1];five_snapshot=ROOT/pin['path'];assert sha(five_snapshot)==pin['sha256']
    obs=read(five_snapshot)
    state['latest_five_queue_readonly_observation']=dict(path=pin['path'],sha256=pin['sha256'],utc=obs['utc'],growth_proof_path=growth.relative_to(ROOT).as_posix(),growth_proof_sha256=sha(growth),counts_are_observation_only=True,FLGMM_terminal=len(obs['FLGMM']['terminal_ids']),gradient_terminal=len(obs['gradient64']['terminal_ids']),remaining620_closed=obs['remaining620']['remote_closed_n'],Hybrid_terminal=len(obs['Hybrid96']['terminal_records']),new_acceptance=0)
    for key,stage in [('flgmm_fullcoverage_v2_20261009','FLGMM'),('gradient64_validation_search_20261010','gradient64')]:
        o=obs[stage];state[key]['latest_measured_observation']=dict(checked_utc=obs['utc'],terminal70_observed=len(o['terminal_ids']),active_rounds=[r['round'] for r in o['active']],snapshot_path=pin['path'],snapshot_sha256=pin['sha256'],acceptance_unchanged_by_observation=True)
    state['mechanism_remaining620_valid_20261010']['latest_measured_observation']=dict(utc=obs['utc'],remote_strict_closed=obs['remaining620']['remote_closed_n'],snapshot_path=pin['path'],snapshot_sha256=pin['sha256'],source_bound=True,acceptance_unchanged_by_observation=True)
    state['hybrid100_fullcoverage_20261010']['latest_measured_observation']=dict(utc=obs['utc'],terminal70_observed=len(obs['Hybrid96']['terminal_records']),snapshot_path=pin['path'],snapshot_sha256=pin['sha256'],acceptance_unchanged_by_observation=True)
reader_reply=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_reader_20261010/ROOT_REVIEW.json'
if reader_reply.exists():
    proof=read(reader_reply)
    assert sha(reader_reply)=='0b46e6ba9a8f30f0b2523cbe7c6947ef47a402e0af55a3ce578e18036c94a511'
    assert (proof['original_comments'],proof['number_strings_preserved'],proof['links_preserved'],proof['whole_tables_preserved'])==(24,2292,88,11)
    assert proof['quantitative_sentences_changed']==0 and proof['root_editorial_review'] and proof['author_review_only'] and not proof['manuscript_applied'] and not proof['final_test']
    for name,digest in proof['documents_sha256'].items():assert sha(reader_reply.parent/name)==digest
    assert sha(ROOT/proof['actual_root_check_path'])==proof['actual_root_check_sha256']
    state['latest_rebuttal_draft'].update(status=proof['status'],entry=proof['entry'],manuscript_candidate=proof['manuscript_candidate'],root_proof_path=reader_reply.relative_to(ROOT).as_posix(),root_proof_sha256=sha(reader_reply),editorial_source_seal_sha256=proof['source_seal_sha256'],editorial_reversible_edits=27,quantitative_sentences_changed=0,response_preface_words_before=1876,response_preface_words_after=242,A50_incorporated=False)
A50_reader=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A50_reader_20261010/ROOT_REVIEW.json'
if A50_reader.exists():
    proof=read(A50_reader)
    assert sha(A50_reader)=='4018404805e67b526b2f8b6f748c35fbc329c46a6326a84af17e9753fa1a6788'
    assert (proof['original_comments'],proof['number_strings_preserved'],proof['links_preserved'],proof['whole_scientific_tables_preserved'])==(24,2292,88,11)
    assert proof['A50_incorporated'] and proof['A_paired_models']==50 and proof['A_complete_scenes']==5
    assert proof['paired_mean_SD_values_bound_to_JSON']==12 and proof['reversible_edits']==16 and proof['author_review_only']
    assert proof['A50_table_root_sha256']==sha(A50_table/'ROOT_VERIFICATION.json') and not proof['final_test'] and not proof['manuscript_applied']
    for name,digest in proof['documents_sha256'].items():assert sha(A50_reader.parent/name)==digest
    assert sha(ROOT/proof['actual_root_check_path'])==proof['actual_root_check_sha256']
    state['latest_rebuttal_draft'].update(status=proof['status'],entry=proof['entry'],manuscript_candidate=proof['manuscript_candidate'],root_proof_path=A50_reader.relative_to(ROOT).as_posix(),root_proof_sha256=sha(A50_reader),A50_incorporated=True,A_complete_scenes=5,A_paired_models=50,A50_reversible_edits=16,A50_mean_SD_pointer_pairs=12,source_seal_sha256=proof['source_seal_sha256'],author_review_only=True,whole_rebuttal_complete=False,manuscript_applied=False)
    state['celeba_mechanism_v1']['A_three_view_five_scene_table']['incorporated_into_full_rebuttal']=True
A60_reader=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A60_reader_20261011/ROOT_REVIEW.json'
if A60_reader.exists():
    proof=read(A60_reader)
    assert sha(A60_reader)=='101ba9240e0be4ea471ae12b6d438ff30336d2ae8d5411cdd68c41e689435afb'
    assert (proof['original_comments'],proof['number_strings_preserved'],proof['links_preserved'])==(24,2457,98)
    assert proof['A60_incorporated'] and (proof['A_paired_models'],proof['A_complete_scenes'],proof['A_nonIID_complete_scenes'])==(60,6,1)
    assert proof['paired_mean_SD_values_bound_to_JSON']==6 and proof['reversible_edits']==21 and proof['author_review_only']
    assert proof['A60_table_root_sha256']==sha(A60_table/'ROOT_VERIFICATION.json') and not proof['final_test'] and not proof['manuscript_applied']
    for name,digest in proof['documents_sha256'].items():assert sha(A60_reader.parent/name)==digest
    assert sha(ROOT/proof['actual_root_check_path'])==proof['actual_root_check_sha256']
    state['latest_rebuttal_draft'].update(status=proof['status'],entry=proof['entry'],manuscript_candidate=proof['manuscript_candidate'],root_proof_path=A60_reader.relative_to(ROOT).as_posix(),root_proof_sha256=sha(A60_reader),A60_incorporated=True,A_complete_scenes=6,A_paired_models=60,A_nonIID_complete_scenes=1,A60_reversible_edits=21,A60_mean_SD_pointer_pairs=6,source_seal_sha256=proof['source_seal_sha256'],author_review_only=True,whole_rebuttal_complete=False,manuscript_applied=False)
    main['A_three_view_six_scene_table']['incorporated_into_full_rebuttal']=True
A80_reader=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A80_reader_20261011/ROOT_REVIEW.json'
if A80_reader.exists():
    proof=read(A80_reader)
    assert sha(A80_reader)=='1cf5c5b2122202800927e696bf8757546610e9fe977709c8019d5e1048208466'
    assert (proof['original_comments'],proof['number_strings_preserved'],proof['links_preserved'])==(24,2609,108)
    assert proof['A80_incorporated'] and (proof['A_paired_models'],proof['A_complete_scenes'],proof['A_nonIID_complete_scenes'])==(80,8,3)
    assert (proof['paired_mean_SD_values_bound_to_JSON'],proof['reversible_edits'],proof['fixed_direction_panels'],proof['interpretation_sign_bindings'])==(12,17,18,6)
    assert proof['author_review_only'] and proof['A80_table_root_sha256']==sha(A80_table/'ROOT_VERIFICATION.json')
    assert not proof['final_test'] and not proof['manuscript_applied'] and not proof['whole_rebuttal_complete']
    for name,digest in proof['documents_sha256'].items():assert sha(A80_reader.parent/name)==digest
    assert sha(ROOT/proof['actual_root_check_path'])==proof['actual_root_check_sha256']
    assert sha(ROOT/proof['root_command_path'])==proof['root_command_sha256']
    state['latest_rebuttal_draft'].update(status=proof['status'],entry=proof['entry'],manuscript_candidate=proof['manuscript_candidate'],root_proof_path=A80_reader.relative_to(ROOT).as_posix(),root_proof_sha256=sha(A80_reader),A80_incorporated=True,A_complete_scenes=8,A_paired_models=80,A_nonIID_complete_scenes=3,A80_reversible_edits=17,A80_mean_SD_pointer_pairs=12,source_seal_sha256=proof['source_seal_sha256'],author_review_only=True,whole_rebuttal_complete=False,manuscript_applied=False)
    main['A_three_view_eight_scene_table']['incorporated_into_full_rebuttal']=True
clear_reader=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_clear_20261011'
if (clear_reader/'EDITORIAL_CHECK.json').exists():
    check=read(clear_reader/'EDITORIAL_CHECK.json')
    assert check['status']=='CLEAR_AUTHOR_REVIEW_DRAFT_EDITORIAL_CHECK_PASS'
    assert check['original_comments']==check['response_sections']==24 and check['quotation_lines_exact_and_ordered']
    assert check['source_sha256']==sha(A80_reader.parent/'rebuttal_integrated_20261011.md')
    assert check['draft_sha256']==sha(clear_reader/'rebuttal_clear_20261011.md') and check['R3_independent_review']
    assert not check['final_test'] and not check['submitted_manuscript_applied'] and not check['whole_rebuttal_complete']
    state['latest_rebuttal_draft'].update(clear_reader_entry=(clear_reader/'rebuttal_clear_20261011.md').relative_to(ROOT).as_posix(),clear_reader_sha256=check['draft_sha256'],clear_reader_check_path=(clear_reader/'EDITORIAL_CHECK.json').relative_to(ROOT).as_posix(),clear_reader_check_sha256=sha(clear_reader/'EDITORIAL_CHECK.json'),clear_reader_words=check['draft_words'],detailed_evidence_entry=state['latest_rebuttal_draft']['entry'])
exact3_dir=ROOT/'tmp/celeba_added_cnn_exact3_root_execution_20261010'
if (exact3_dir/'START_RECEIPT.json').exists():
    started=read(exact3_dir/'START_RECEIPT.json');review=read(exact3_dir/'ROOT_SOURCE_REVIEW.json')
    assert started['status']=='ROOT_EXACT3_SUPERVISOR_STARTED_NOT_SCIENTIFIC_ACCEPTANCE'
    assert sha(exact3_dir/'ROOT_SOURCE_REVIEW.json')=='e81d178f0390ca9c5be0939a0c62ff9da762b2b3dc59f51b9b8d82b26f878323' and review['source_adoptable']
    assert started['authorization_sha256']==sha(exact3_dir/'AUTHORIZATION.json') and not started['autorestart'] and started['startretries']==0
    state['added_CNN_exact3_valid_interface_20261010']=dict(status=started['status'],start_utc=started['utc'],service=started['program'],start_receipt_path=(exact3_dir/'START_RECEIPT.json').relative_to(ROOT).as_posix(),start_receipt_sha256=sha(exact3_dir/'START_RECEIPT.json'),source_review_path=(exact3_dir/'ROOT_SOURCE_REVIEW.json').relative_to(ROOT).as_posix(),source_review_sha256=sha(exact3_dir/'ROOT_SOURCE_REVIEW.json'),package_sha256=started['package_sha256'],linux_preflight_sha256=started['linux_preflight_sha256'],authorization_sha256=started['authorization_sha256'],exact_ids=review['exact_ids'],cpu_affinity=list(range(120,128)),threads=8,max_processes=1,device='cpu',test=False,training=False,root_scientific_acceptances=0,scope='Exactly three representative accepted checkpoints; source/interface stage, not full100 or final endpoint; actual runtime receipts and offserver checks still required')
exact3_adoption=exact3_dir/'ROOT_SCIENTIFIC_ADOPTION.json'
if exact3_adoption.exists():
    proof=read(exact3_adoption)
    assert sha(exact3_adoption)=='631d7ee2acf1cbe3453d849523e262f29ff5b354b5ddb56c60e5779e94364456'
    assert proof['root_adoption'] and proof['interface_records_accepted']==3 and proof['mechanism_three_view_cutoff_unchanged']==251
    assert proof['Linux_whole_original_saved_check_pass'] and proof['Windows_original_array_refit_block_pass'] and not proof['Windows_whole_saved_check_pass']
    assert proof['cross_platform_audit_failure_preserved'] and proof['native_max_abs_difference']==0 and not proof['final_test']
    for name,digest in proof['proof_files_sha256'].items():assert sha(exact3_dir/name)==digest
    state['added_CNN_exact3_valid_interface_20261010'].update(status=proof['status'],root_scientific_acceptances=3,root_proof_path=exact3_adoption.relative_to(ROOT).as_posix(),root_proof_sha256=sha(exact3_adoption),Linux_whole_original_saved_check_pass=True,Windows_whole_saved_check_pass=False,Windows_original_array_refit_block_pass=True,cross_platform_audit_failure_preserved=True,native_max_abs_difference=0.0,metric_values_checked=27,integer_base_counts_checked=72,prediction_rules_checked=9,scope='Exactly three representative checkpoints: Linux whole original check and independent F saved-array/root-refit block pass; original Windows whole audit equality failure remains preserved. No full100, full17 or final endpoint/test claim.')
fl47=ROOT/'tmp/celeba_flgmm_closed47_root_execution_20261011'
if (fl47/'START_RECEIPT.json').exists():
    started=read(fl47/'START_RECEIPT.json');review=read(fl47/'ROOT_SOURCE_REVIEW.json');auth=read(fl47/'AUTHORIZATION.json')
    assert started['status']=='ROOT_FLGMM_FINITE47_SUPERVISOR_STARTED_NOT_SCIENTIFIC_ACCEPTANCE'
    assert sha(fl47/'ROOT_SOURCE_REVIEW.json')=='37ddad29641f4d9a30987717fc8cf56153eb2a1f6c0ac178d2e36fde6c2bc84d' and review['source_adoptable']
    assert started['authorization_sha256']==sha(fl47/'AUTHORIZATION.json') and not started['autorestart'] and started['startretries']==0
    assert len(auth['exact_ids'])==len(set(auth['exact_ids']))==47 and not auth['test'] and not auth['training']
    observation_path=fl47/'LIVE_OBSERVATION.json'
    if (fl47/'LATEST_LIVE_OBSERVATION.json').exists():
        pin=read(fl47/'LATEST_LIVE_OBSERVATION.json');observation_path=ROOT/pin['path'];assert sha(observation_path)==pin['sha256'] and observation_path.resolve().is_relative_to(fl47.resolve())
    obs=read(observation_path);receipts=[k for k in obs['json'] if k.endswith('/receipt.json')]
    failures=[k for k in obs['json'] if 'failure' in k.lower()]
    state['FLGMM_closed47_valid_three_view_20261011']=dict(status=started['status'],service=started['program'],start_utc=started['utc'],start_receipt_path=(fl47/'START_RECEIPT.json').relative_to(ROOT).as_posix(),start_receipt_sha256=sha(fl47/'START_RECEIPT.json'),source_review_sha256=sha(fl47/'ROOT_SOURCE_REVIEW.json'),package_sha256=started['package_sha256'],exact_ids=auth['exact_ids'],device='cpu',cpu_affinity=list(range(120,128)),threads=8,max_processes=1,training=False,test=False,source_native_training_records=44,separate_screen_reuse=4,prior_FL_interface_skipped=1,root_new_three_view_acceptances=0,live_observation_path=observation_path.relative_to(ROOT).as_posix(),live_observation_sha256=sha(observation_path),observed_utc=obs['utc'],receipts_observed=len(receipts),failures_observed=failures,complete_queue_receipt_observed='GATE_RESULT.json' in obs['members'],gate_result_member_observed=obs['members'].get('GATE_RESULT.json'),failure_stop_no_auto_retry=True,whole_original_saved_check_and_offserver_acceptance_pending=True,scope='Exactly47 already-accepted FLGMM terminal70 checkpoints; CPU valid19867 replay/root-only calibration only, not training or final test. Observed receipts do not increase scientific acceptance.')
fl47_hold=fl47/'ROOT_VALIDATION_HOLD.json'
if fl47_hold.exists():
    proof=read(fl47_hold)
    assert sha(fl47_hold)=='a87f1dac4b42d6ec1754d88b772c44a5c0d7c95ca3bb3d684cc7a97a34dfa6f1'
    assert sha(ROOT/proof['failure_path'])==proof['failure_sha256']
    assert proof['new_three_view_records_adopted']==0 and proof['Linux_whole_original_saved_check_pass'] and not proof['Windows_original_array_block_pass']
    state['FLGMM_closed47_valid_three_view_20261011'].update(status=proof['status'],validation_hold=True,validation_hold_path=fl47_hold.relative_to(ROOT).as_posix(),validation_hold_sha256=sha(fl47_hold),failed_id=proof['failed_id'],prior_loop_completions_not_adopted=4,Linux_whole_original_saved_check_pass=True,F_archive_and98_members_verified=True,Windows_original_array_block_pass=False,root_cause_unmeasured=True,failed_verifier_retry_authorized=False)
    diagnostic=fl47/'saved_acceptance_actual001/ROOT_SINGLE_RECORD_DIAGNOSTIC_REVIEW.json'
    if diagnostic.exists():
        d=read(diagnostic)
        assert sha(diagnostic)=='5644fc89cd3ac4b3005612ef3c96d1a8b83459b556aa891c616e49b33fe5537d'
        assert d['scientific_acceptances']==d['FLGMM47_new_three_view_adopted']==0 and d['platform_cause_not_established']
        for rel,h in d['files'].items():assert sha(ROOT/rel)==h
        state['FLGMM_closed47_valid_three_view_20261011']['single_failed_record_diagnostic']=dict(root_review_path=diagnostic.relative_to(ROOT).as_posix(),root_review_sha256=sha(diagnostic),id=d['id'],adaptive_lambda_difference=d['adaptive_lambda_difference'],effective_thresholds_exact=d['effective_thresholds_exact'],predictions_metrics_counts_exact=True,root_receipt_exact=True,first_comparator_failure_preserved=True,platform_cause_not_established=True,scientific_acceptances=0)
        trace=diagnostic.parent/'operation_trace001/ROOT_OPERATION_TRACE_REVIEW.json'
        if trace.exists():
            t=read(trace)
            assert sha(trace)=='4cb8a6dea7df06e02fbf8d45acf8ad1db509c227743a2aafa9f1b3d698e680f6'
            assert t['first_different_operation']=='log1p' and t['fit_calls']==t['scientific_acceptances']==0
            for rel,h in t['files'].items():assert sha(trace.parent/rel)==h
            state['FLGMM_closed47_valid_three_view_20261011']['single_failed_record_diagnostic'].update(operation_trace_path=trace.relative_to(ROOT).as_posix(),operation_trace_sha256=sha(trace),first_different_operation='math.log1p',same_operands_and_exp=True,individual_OS_libm_Python_causal_factor_not_isolated=True)
# Complementary adoption supersedes the current hold, not the preserved failed refit.
fl47_adoption=fl47/'ROOT_SCIENTIFIC_ADOPTION.json'
if fl47_adoption.exists():
    a47=read(fl47_adoption)
    assert sha(fl47_adoption)=='22fc113add73814ae40accf563ce5f63bd63d50bcf53b7262bdb2b996b016dfe'
    assert a47['status']=='ROOT_FLGMM_CLOSED47_COMPLEMENTARY_EVIDENCE_ADOPTED_WINDOWS_REFIT_FAILED_PRESERVED' and a47['root_adoption']
    e=state['FLGMM_closed47_valid_three_view_20261011']
    assert e['exact_ids']==a47['exact_ids'] and len(set(a47['exact_ids']))==len(a47['records'])==47
    assert [r['id'] for r in a47['records']]==a47['exact_ids']
    assert a47['new_three_view_records_accepted']==47 and a47['FLGMM_total_three_view_records']==48
    assert len(a47['prior_interface_explicitly_reused'])==1 and a47['native_training_acceptance_unchanged_by_this_action']==44 and a47['original_screen_reuse_separate']==4
    assert a47['mechanism_three_view_cutoff_unchanged']==260 and main['three_view_new_models_offserver_verified']>=260
    assert a47['Linux_whole_original_saved_check_pass'] and a47['Linux_root_fit_verified'] and a47['Linux_original_root_refit_records']==47
    assert a47['Windows_saved_outputs_audit_pass'] and a47['Windows_saved_outputs_audit_fit_calls']==0
    assert not any(a47[k] for k in ('Windows_exact_refit_pass','Windows_whole_saved_check_pass','Windows_original_array_refit_block_pass','previous_Windows_exact_refit_condition_satisfied','cross_platform_bitwise_recalibration_claimed','test','full100_complete','full17_complete'))
    assert a47['complementary_evidence_role_explicitly_acknowledged'] and a47['new_fit']==a47['new_CNN']==a47['new_training']==0
    assert (a47['metric_values_checked'],a47['integer_base_counts_checked'],a47['prediction_rules_checked'],a47['root_audit_difference_records'])==(423,1128,141,4)
    assert a47['native_max_abs_difference']==0 and a47['original_tolerance_unchanged']==1e-12 and a47['archive_members']==98
    assert sha(fl47_hold)=='a87f1dac4b42d6ec1754d88b772c44a5c0d7c95ca3bb3d684cc7a97a34dfa6f1'
    for rel,pin in a47['preserved_evidence_pins'].items():
        p47=ROOT/rel
        assert p47.stat().st_size==pin['bytes'] and sha(p47)==pin['sha256']
    for name,h in a47['proof_files_sha256'].items():
        assert sha(fl47/'saved_acceptance_actual001'/name)==h
    e.update(status=a47['status'],complementary_adopted=True,root_proof_path=fl47_adoption.relative_to(ROOT).as_posix(),root_proof_sha256=sha(fl47_adoption),root_adoption_utc=a47['utc'],root_new_three_view_acceptances=47,FLGMM_total_three_view_records=48,prior_FL_interface_accepted_separately=1,prior_interface_explicitly_reused=a47['prior_interface_explicitly_reused'],mechanism_three_view_cutoff_unchanged=260,validation_hold=False,original_validation_hold_preserved=True,original_validation_hold_status=read(fl47_hold)['status'],whole_original_saved_check_and_offserver_acceptance_pending=False,Linux_root_fit_verified=True,Linux_original_root_refit_records=47,Windows_saved_outputs_audit_pass=True,Windows_saved_outputs_audit_fit_calls=0,Windows_exact_refit_pass=False,Windows_whole_saved_check_pass=False,Windows_original_array_refit_block_pass=False,Windows_original_refit_failure_sha256=a47['Windows_original_refit_failure_sha256'],Windows_original_refit_completed_before_failure=4,previous_Windows_exact_refit_condition_satisfied=False,cross_platform_bitwise_recalibration_claimed=False,root_audit_difference_records=4,metric_values_checked=423,integer_base_counts_checked=1128,prediction_rules_checked=141,native_max_abs_difference=0,original_tolerance_unchanged=1e-12,proof_files_sha256=a47['proof_files_sha256'],scope=a47['scope'])
split_metadata=TRAIN/'final_split_metadata_20261011/ROOT_METADATA_VERIFICATION.json'
if split_metadata.exists():
    proof=read(split_metadata)
    assert sha(split_metadata)=='b234a0689f80179998ea80e8c520ca7b8c0bcb9fa17004b0107477bae8f00c84'
    assert proof['status']=='ROOT_CANDIDATE_OFFICIAL_SPLIT_IDS_VERIFIED_METADATA_ONLY'
    assert sha(split_metadata.parent/'STDOUT.json')==proof['observation_sha256']
    assert sha(split_metadata.parent/'ACTUAL_COMMAND.json')==proof['actual_command_sha256']
    assert sha(TRAIN/'final_evaluation_prepared_20261009/protocol.json')==proof['prepared_protocol_sha256']
    assert proof['candidate_target']['n']==19962 and proof['no_label_array_decoding']
    assert proof['no_pixels_models_fit_inference_or_metrics'] and proof['prepared_protocol_unchanged']
    assert not any(proof[k] for k in ('protocol_frozen','primary_endpoint_selected','final_evaluation_started'))
    state['final_split_metadata_20261011']=dict(proof,root_proof_path=split_metadata.relative_to(ROOT).as_posix(),root_proof_sha256=sha(split_metadata))
fl61=ROOT/'tmp/fl_three_view_after48_20261011/ROOT_SCIENTIFIC_ADOPTION.json'
if fl61.exists():
    proof=read(fl61)
    assert sha(fl61)=='d6c7bdadb05ffcf8786221ed15a16125cd0fe84d1745fc15f9b7e6cc0a2f68d6'
    assert proof['root_adoption'] and proof['status']=='ROOT_FLGMM_AFTER48_EXACT13_COMPLEMENTARY_EVIDENCE_ADOPTED_PRIOR_WINDOWS47_FAIL_PRESERVED'
    assert (proof['new_three_view_records_accepted'],proof['FLGMM_total_three_view_records'],proof['native_training_records'],proof['original_screen_reuse_separate'])==(13,61,57,4)
    assert proof['prior_root_sha256']==state['FLGMM_closed47_valid_three_view_20261011']['root_proof_sha256']
    assert proof['Linux_whole_original_saved_check_pass'] and proof['Linux_original_root_refit_records_new']==13 and proof['Windows_saved_outputs_audit_pass'] and proof['Windows_saved_outputs_audit_fit_calls']==0
    assert not any(proof[k] for k in ('Windows_new13_refit_executed','Windows_original47_exact_refit_pass','Windows_whole_saved_check_pass','cross_platform_bitwise_recalibration_claimed','mechanism_scope_modified','test','full100_complete'))
    for pin in proof['proof_files'].values():assert sha(ROOT/pin['path'])==pin['sha256']
    assert len(proof['new_records'])==13 and len({r['id'] for r in proof['records']+proof['prior_interface_explicitly_reused']})==61
    state['FLGMM_after48_valid_three_view_20261011']=dict(status=proof['status'],root_proof_path=fl61.relative_to(ROOT).as_posix(),root_proof_sha256=sha(fl61),new_three_view_records_accepted=13,FLGMM_total_three_view_records=61,native_training_records=57,original_screen_reuse_separate=4,Linux_original_root_refit_records_new=13,Windows_saved_outputs_audit_fit_calls=0,metric_values_checked_new=117,integer_base_counts_checked_new=312,prediction_rules_checked_new=39,archive_members=30,archive_sha256=proof['archive_sha256'],new_exact_ids=[r['id'] for r in proof['new_records']],new_root_audit_difference_records=sum(bool(r['preserved_root_audit_differences']) for r in proof['new_records']),original_Windows47_failure_preserved=True,Windows_new13_refit_executed=False,Windows_whole_saved_check_pass=False,full100_complete=False,final_test=False)
fl_six=ROOT/'outputs/guardfed_tables/celeba_flgmm_six_scenes60_20261011/ROOT_VERIFICATION.json'
if fl_six.exists():
    proof=read(fl_six)
    assert sha(fl_six)=='a7ed22f1f6366135b361433581ad59a76aa52271ba0516cd0bb11efc18e9eef0'
    assert proof['root_adoption'] and proof['source_root61_sha256']==sha(fl61)
    assert (proof['records_preserved'],proof['complete_scene_records'],proof['retained_partial_records'],proof['scenes'])==(61,60,1,6)
    assert (proof['metric_count'],proof['base_integer_count'],proof['statistical_scalars'],proof['Markdown_cells'],proof['CSV_cells'])==(549,1464,324,162,162)
    assert proof['caption_only_body_equality'] and proof['prior_Windows47_exact_refit_failure_preserved'] and not proof['final_test']
    for name,digest in proof['adopted_outputs'].items():assert sha(fl_six.parent/name)==digest
    state['FLGMM_after48_valid_three_view_20261011']['six_scene_table']=dict(root_proof_path=fl_six.relative_to(ROOT).as_posix(),root_proof_sha256=sha(fl_six),table_path=(fl_six.parent/'TABLES.md').relative_to(ROOT).as_posix(),complete_scene_records=60,retained_partial_records=1,scenes=6,views=['raw','native','shared_calibration'],fixed_seed_panels=[10,9,6],sample_SD_ddof=1,validation_only=True,full100_complete=False,final_test=False)
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
if main.get('C_after12_valid_replay',{}).get('offserver_new_accepted')==8:
    C_replay_note='minus_C累计20已离机：IID Benign与F Flip各10seed；后者完整配对表待另核'
if main.get('C_three_view_two_scene_table'):
    C_replay_note='minus_C累计20已离机，IID Benign/F Flip两完整场景三视图表已独立采用，324统计/162单元/360计数指标'
    C_after1_note=C_after1_note.replace('F Flip仅2seed，其他C场景未齐。',
        '该历史11项增量只含F Flip两seed；后续8项已补齐F Flip十seed并完成两场景表验收，其余八个C场景仍未齐。')

if main.get('C_after20_valid_replay',{}).get('offserver_new_accepted')==5:
    C_replay_note+='；另5项IID FedSA终轮三视图已严格离机，累计C25，FedSA仅5/10不入完整场景均值'
if main.get('C_after25_valid_replay',{}).get('offserver_new_accepted')==3:
    C_replay_note='minus_C累计28已严格离机；IID Benign/F Flip两完整场景三视图表已独立核验324统计/162单元/360计数指标；IID FedSA仅8/10、不入完整场景均值'
if main.get('C_after28_valid_replay',{}).get('offserver_new_accepted')==8:
    C_replay_note='minus_C累计36已严格离机；IID Benign/F Flip/FedSA各10seed评价齐备，S-DFA仅6/10不入完整场景均值；三场景配对表须另核'
if main.get('C_three_view_three_scene_table'):
    C_replay_note='minus_C累计36已严格离机；IID Benign/F Flip/FedSA三场景表已独立核验486统计/243单元/540计数指标；旧两场景精确保持，S-DFA仅6/10单列排除，其他七C场景未齐'
if main.get('C_after36_valid_replay',{}).get('offserver_new_accepted')==4:
    C_replay_note='minus_C累计40已严格离机；IID Benign/F Flip/FedSA/S-DFA各10seed齐备，四场景配对表须单独核验；旧136与Full不重推'
if main.get('C_three_view_four_scene_table'):
    C_replay_note='minus_C累计40已严格离机；IID Benign/F Flip/FedSA/S-DFA四场景表已独立核验648统计/324单元/720计数指标；旧三场景精确保持，其他六C场景未齐'
if main.get('C_after40_valid_replay',{}).get('offserver_new_accepted')==7:
    C_replay_note='minus_C累计47已严格离机；四完整IID场景表保持648统计/324单元/720计数指标；新增IID Sp-DFA仅7/10，63指标/168计数/21规则核验通过，不进入完整场景均值；其他六C场景未齐'
if main.get('C_after47_valid_replay',{}).get('offserver_new_accepted')==3:
    C_replay_note='minus_C累计50已严格离机；IID五场景各10seed齐备；最新三项27指标/72计数/9规则通过、native偏差0；完整五场景表以独立采用凭据为准，五个non-IID C场景与其余六变体仍待完成'
if main.get('C_three_view_five_scene_table'):
    C_replay_note='minus_C累计50已严格离机；五完整IID场景论文表已独立采用，810均值/样本SD标量、405展示单元、900计数指标及162个先seed内平均五场景的汇总标量通过；旧80记录/648统计/324展示值不变；五个non-IID C场景与其余六变体仍待完成'
if main.get('C_after50_valid_replay'):
    C56=main['C_after50_valid_replay']
    if C56['offserver_new_accepted']==6:
        C_replay_note+='；另6项non-IID Benign checkpoint三视图已严格离机，75归档成员/54指标/144计数/18规则通过、native偏差0，累计C56；该场景仅6/10，不进入完整场景均值'
    else:
        C_replay_note+='；另6项non-IID Benign checkpoint三视图已实际启动，严格离机接受仍0，该场景不进入完整场景均值'
if main.get('C_after56_valid_replay',{}).get('offserver_new_accepted')==4:
    C_replay_note='minus_C累计60已严格离机；新增准确4项non-IID Benign通过61归档成员/36指标/96计数/12规则，native偏差0，原156与Full不重推；该场景10/10已齐，六场景表须独立采用；其他四个non-IID C场景及六变体未完成'
if main.get('C_three_view_six_scene_table'):
    C_replay_note='minus_C六场景60对/120记录表已独立采用：五IID及non-IID Benign，972统计/486单元/1080计数指标/2880计数通过；旧100记录/810统计/405展示和162个IID seed-first标量保持，未计算不平衡六场景总均值；10/9/6面板、负结果及混合设备/环境/选择史保留；C60表独立于C50全文作者审阅稿，其他四个non-IID C场景及六变体未完成'
if main.get('C_three_view_six_scene_table',{}).get('incorporated_into_full_rebuttal'):
    C_replay_note=C_replay_note.replace('C60表独立于C50全文作者审阅稿','C60证据已纳入完整英文作者审阅稿，原C50稿保持、提交版正文未应用')
if main.get('C_after60_valid_replay',{}).get('execution_started'):
    if main['C_after60_valid_replay']['offserver_new_accepted']==10:
        C_replay_note+='；新增十项non-IID F Flip三视图已正常EXITED、严格验收并离机，103归档成员/90指标/240计数/30规则通过、native偏差0，累计C70；旧160与Full不重推，第七场景表须单独统计验收'
    else:
        C_replay_note+='；新增十项non-IID F Flip三视图已实测启动，CPU112–119单进程8线程，旧160与Full复用；该批严格离机接受仍0，不计入已完成论文表'
if main.get('C_three_view_seven_scene_table'):
    C_replay_note='minus_C七场景70对/140记录表已独立采用：五IID及non-IID Benign/F Flip，各10seed；1134统计/567单元/1260计数指标/3360计数与630配对指标通过；旧120记录/972统计/486单元及162个IID seed-first标量保持，不计算不平衡七场景总均值。新增F Flip删除C的native/shared差为ACC+0.943pp、AEOD−0.00308、ASPD+0.01177；全部10/9/6面板及负结果保留。Full5CPU/65GPU、C70CPU与训练环境/选择史披露；其他三non-IID C场景及六变体未完成。最新完整英文稿仍封存C60，尚未合入第七场景，正文未应用、未运行test'
if main.get('C_after70_valid_replay',{}).get('execution_started'):
    C80_stage=main['C_after70_valid_replay']
    if C80_stage['offserver_new_accepted']==10:
        C_replay_note+=('；新增准确十项non-IID FedSA三视图已正常EXITED、0残留worker、严格离机并根验收，103成员/90指标/240计数/30规则通过，native偏差0，累计U100+C80=180。旧170/Full不重推，C第八场景表尚待独立统计采用')
    else:
        C_replay_note+=(f'；另准确十项non-IID FedSA三视图已实际启动，最新远端完成{C80_stage.get("observed_remote_completed",1)}/10，'
            f'该批严格离机接受{C80_stage["offserver_new_accepted"]}，累计采用仍{main["three_view_new_models_offserver_verified"]}；'
            'CPU112–119单进程8线程，旧170及Full不重推')
    C_replay_note+='。部署前独立科学源码审阅已通过；本地运输封条读取器错误及报告晚于部署调用的顺序问题完整留档，补核17文件及运行资源通过'
if main.get('C_three_view_eight_scene_table'):
    C_replay_note=('minus_C八场景80对/160记录表已独立采用：五IID及non-IID Benign/F Flip/FedSA，各10seed；1296统计/648单元/1440计数指标/3840计数及720配对指标通过。'
        '旧140记录/1134统计/567单元及162个IID seed-first标量保持，不计算不平衡八场景总均值。新增FedSA删除C的native/shared差为ACC+0.410pp、AEOD+0.00190、ASPD−0.0000093，9/6seed中ASPD方向变化；10/9/6面板及负结果保留。'
        'Full5CPU/75GPU对C80CPU，训练环境/选择史披露；另两non-IID C场景及六变体未完成。完整英文稿仍封存C60，C70/C80尚未合入、正文未应用、未test。'
        '新10项评价103归档成员严格离机，native偏差0，原170/Full不重推；运输补核晚于部署的本地顺序问题及首辅助审查断言错误保留，科学输出未改')
C_latest_table=main.get('C_three_view_five_scene_table',main.get('C_three_view_four_scene_table',main.get('C_three_view_three_scene_table',main.get('C_three_view_two_scene_table',{})))).get('table_path','待独立核验')
if state.get('mechanism_remaining620_valid_20261010',{}).get('C100_replay_complete'):
    C_replay_note=C_replay_note.replace('另两non-IID C场景及六变体未完成','C余下两场景评价已严格离机，C100完整统计表待独立采用；六变体未完成')
C_other_scenes=5 if main.get('C_three_view_five_scene_table') else (6 if main.get('C_three_view_four_scene_table') else (7 if main.get('C_three_view_three_scene_table') else 8))
if main.get('C_three_view_six_scene_table'):
    C_latest_table=main['C_three_view_six_scene_table']['table_path'];C_other_scenes=4
if main.get('C_three_view_seven_scene_table'):
    C_latest_table=main['C_three_view_seven_scene_table']['table_path'];C_other_scenes=3
if main.get('C_three_view_eight_scene_table'):
    C_latest_table=main['C_three_view_eight_scene_table']['table_path'];C_other_scenes=2
if main.get('C_three_view_full100_table'):
    C_latest_table=main['C_three_view_full100_table']['table_path'];C_other_scenes=0
    C_replay_note=('minus_C完整十场景100对/200记录表已独立采用：IID/non-IID各五场景×十共享seed；1620统计/810单元/1800指标/4800计数/900配对指标通过，旧160/1296/648及162 IID汇总保持。另non-IID和平衡十场景seed-first各162标量通过，n取seed数。'
        '全部10/9/6面板与负结果保留；non-IID S-DFA删除C在n10三指标均值更好，Sp-DFA为准确率–ASPD取舍。Full5CPU95GPU/98cu1282cu130对C100CPU/cu128及选择史披露，不作必要性/因果/显著性主张。六个其他变体、正文和最终评价未完成；完整英文稿已纳入C100')

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
| FLGMM完整覆盖 | 96新+4复用；状态{state.get('flgmm_fullcoverage_v2_20261009',{}).get('status','未启动')}，短程5新+2参考已核；新增70轮离机接受{state.get('flgmm_fullcoverage_v2_20261009',{}).get('new_accepted',0)}/96 | tmp/celeba_flgmm_fullcoverage_binding_20261009/ROOT_SEVEN_CANARY_CLOSURE.json及ROOT_COVERAGE_STARTUP.json |
| 组合基线验证搜索 | {hybrid.get('offserver_accepted70round_jobs', 0)}/32项已严格验收、离机并通过本机来源绑定的记录复核；尚未完整选recipe | tmp/celeba_hybrid_screen_execution_20261009/LATEST_BACKUP.json |
| 九方法旧checkpoint三视图评价 | {baseline['actual_native_valid_image_replays_accepted']}/900已严格验收并离机；原CPU872服务因native偏差failstop EXITED，不重启 | {baseline['accepted_collection_path']} |
| 机制三视图评价 | 累计{main['three_view_new_models_offserver_verified']}份：minus_U100、minus_C100各十完整场景；minus_A{main.get('three_view_counts_by_variant',{}).get('minus_A',0)}，已采用{main.get('A_three_view_four_scene_table',main.get('A_three_view_two_scene_table',{})).get('complete_scenes',0)}个完整IID场景。raw/native/shared及10/9/6seed面板保留，其他控制未齐 | U表：{main.get('latest_paired_three_view_table',{}).get('table_path','需独立配对')}；C表：{C_latest_table}；A表：{main.get('A_three_view_four_scene_table',main.get('A_three_view_two_scene_table',{})).get('table_path','需独立采用')} |

{C_after1_note}

历史8项C/IID/F Flip seed91003–91010终轮checkpoint评价状态{main.get('C_after12_valid_replay',{}).get('status','未启动')}，新增离机接受{main.get('C_after12_valid_replay',{}).get('offserver_new_accepted',0)}。只重建valid三视图，CPU112–119/8线程/nice10/idleIO/CUDA隐藏；旧112与Full不重推。最新C表入口{C_latest_table}，其余{C_other_scenes}个C场景仍待完成。

删除C的IID Benign native十seed论文表已独立核验54统计标量/27展示单元/10对checkpoint，保留9/6seed面板；入口celeba_mechanism_v1/native_C_Benign10_20261009/TABLES.md。十seed配对删除差ACC−0.083个百分点、AEOD+0.00292、ASPD−0.00142，9/6面板方向有变化，不作必要性/因果/显著性结论。该封存单场景快照中的F Flip只有两seed、不纳入其均值；最新完整场景评价与表以本页当前表记录为准。

主机制服务guardfed_celeba_mechanism_formal，固定70round/valid-only/8并发，IID(alpha5000)/non-IID(alpha5)×5场景×10共享seed；100 Full身份已复核，旧权重不重训/重复打包。FLGMM原搜索服务已正常EXITED、0worker，32/32严格离机，冻结规则选Tg20/L2/lr0.001，32评分与32候选均值标量经独立及root复核；前两分差0.00004978，仅n=1搜索不作SD或显著性结论。其后5新+2参考短程已严格离机，FLGMM完整覆盖启动状态{state.get('flgmm_fullcoverage_v2_20261009',{}).get('formal100_started',False)}，新96项仍待逐项严格验收，不把短程计入论文样本。组合基线服务guardfed_celeba_hybrid_screen32仍运行，GPU0/CPU104单线程，未选完整recipe，组合100项尚未启动。全程不运行test。

主机制最近实测CPU {live['cpu_used_cores_2sec']:.2f}/{live['cpu_quota_cores']:.2f}核，RAM {live['memory_used_bytes']/1e9:.2f}GB，磁盘余{live['disk_free_bytes']/1e12:.3f}TB；GPU/温度/RecoveryAction与近期错误读同一实时JSON。只在真实轮次/日志、进程身份和资源证据支持时判断健康，低瞬时占用不重启。服务标签与完成文件不代替验收。

## 当前恢复与研究选择

原CPU失败为FairGuard/IID/F Flip/seed91009：native超原1e-12，65成员失败现场完整保留，原chunk036的10份strict partial当时未登记，后来通过显式审阅导入派生436账本。独立单模型GPU诊断已复现原三指标，差值全0；当前CPU/GPU native/raw只有image172599一处翻转，共享校准预测无翻转。三份归档与保存数组已独立验收；缺历史GPU逐图数组，不声称唯一历史根因，原424账本保持不变。凭据NATIVE_GPU_DIAGNOSTIC_ROOT_VERIFICATION.json。

{gpu_recovery_paragraph} native1e-12、原model/source/data/map/root/valid、同checkpoint全部视图与失败保留规则不变；混合CPU/GPU来源不能冒充统一设备的最终公平比较。入口tmp/celeba_valid_gpu_recovery_implementation_20261009/README.md。监控不自动启动准备包。

LoGoFair原映射提案保留：四条件共用固定image-ID哈希20组，80个root(label,Male)格最小116人。用户2026-10-10委托主代理决定，现已采用该虚拟cohort，明确不能称真实训练client公平性。实际真实分数3post-round接口门检通过40个Beta拟合、19,867验证样本及序列化重载预测exact，但恒定负预测保留，只作流程证据、科学结果0；原32草案不改，新搜索包独立冻结，尚不能称32项完成。门检及root审查入口tmp/celeba_logofair_real_gate_20261010/ROOT_REVIEW.json。

## 已完成证据与剩余交付

2454项历史新增训练、九方法900原始验证结果及旧备份保持原值；当前九方法三视图评价已验收{baseline['actual_native_valid_image_replays_accepted']}/900。重放为混合CPU/GPU来源的验证集评价，最终test仍未完成。旧TableII的480个原值已追溯实际重复数，缺乏依据的SD不补造。完整入口REBUTTAL_COMPLETION_20261009.md；英文rebuttal已对齐24块原意见、40处本地引用及209项SHA声明，尚不能把待补实验写成完成。

原20项cu130与另20项cu128真实图像三轮门检已严格接受、离机，两套Full与原worker短程张量/指标/诊断精确。Fed-NGA/Huber四条真实图像三轮门检含240梯度/攻击oracle已接受；FLGMM的CPU2/GPU4及组合基线的CPU4/CUDA4门检已接受。所有恒定负类和工程/数值失败保留。三轮门检不证明70轮跨环境等价或科学性能优势；门检服务已EXITED，不重启。

完整17行比较仍缺8方法的完整多seed结果：LoGoFair、Fed-NGA、FedWA、Huber、FLGMM、SmartFL、FedDNA及组合控制。用户2026-10-10已接受Huber的R^p恒等投影CNN适配，仅报告经验结果、不继承原理论保证；LoGoFair人口亦已决定，梯度五个常规字段按原算法落实，不再把H/L写成待决。64搜索源包已冻结、独立审查并实际启动；此前宽CPU调度mask及日志tee误判均为训练前工程失败，已保留并修复，最新实测须读STATE新增凭据。最终评价主终点/测试边界仍待冻结；FedWA/SmartFL/FedDNA忠实规格仍缺，不能用简化旧分支冒充。主机制800、完整机制三视图、正文及最终回复仍未完成。Fig3原脚本/ForestDiffusion执行身份仍缺；已核数值与缺失来源明确区分。

已接受native场景的10/9/6种子中期论文表：{state['celeba_mechanism_v1']['latest_interim_paper_table']['table_path']}。仅展示{interim['complete_paired_scenes']}个齐备的Full–minus_U配对场景，保留所有指标及取舍，不补造未完成场景，不以Full最佳seed对比消融均值。在IID Sp-DFA场景，Full准确率较高、去U的两个公平性差距更低，不能声称每项不可或缺。AEOD为绝对TPR差，不是完整equalized odds；Full98cu128+2cu130、多数旧driver570.211.01和当前driver595.84差异、seed91001选择历史均披露。native含各方法原校准，不能据此单独证明聚合机制。native100十场景表已单独核验540统计标量，旧九场景54展示行不变；所有十场景删除U的ACC/ASPD均更低，AEOD八场景更高、两场景更低。non-IID FedSA/S-DFA删除U后ACC分别下降0.601/0.453个百分点，公平性方向存在取舍。九场景三视图已由另一份root凭据独立闭合，数量与来源见下方；不从native表推断评价完成。

{paired_note}

九方法三视图论文表已另行完成并通过root实际900份原receipt连接、8100个组计数指标重建及4860个均值/样本SD核验：outputs/guardfed_tables/celeba_nine_method_three_view_20261009/README.md。完整IID/non-IID、五场景、10/9/6共享种子和三视图均平行保留；旧native三指标及展示表值精确一致，94个旧JSON的SD最后bit差异最大2.78e-17单独保留。英文24意见完整草稿入口{state['latest_rebuttal_draft']['entry']}；{reply_progress_note}，保留COMPAS反例/真实n/环境/选择史及全部pending。旧封存稿保留。仍为作者审阅稿，正文源文件未应用，最终评价与其余方法/机制未写成完成。

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
if main.get('C_after20_valid_replay'):
    C5=main['C_after20_valid_replay']
    top+=(f"\n历史5项C/IID FedSA增量（seed91001/03/05/06/08）终轮valid三视图评价：{C5['status']}，新增严格离机接受{C5['offserver_new_accepted']}。"
        '固定CPU112–119/8线程、nice10/idleIO/CUDA隐藏；原120与Full不重推。来源/实际预检与后续验收入口tmp/celeba_mechanism_valid_C_after20_20261009/execution_candidate。'
        '该批次闭合时FedSA仅5/10；最新覆盖以本页当前表与STATE为准，原科学计算及1e-12保持。\n\n')
if main.get('C_after25_valid_replay'):
    C3=main['C_after25_valid_replay']
    top+=(f"\n历史3项C/IID FedSA增量（seed91002/04/07）终轮valid三视图评价：{C3['status']}，新增严格离机接受{C3['offserver_new_accepted']}。"
        '固定CPU112–119/8线程、nice10/idleIO/CUDA隐藏；原125与Full不重推。来源入口tmp/celeba_mechanism_valid_C_after25_20261009/execution_candidate。'
        '该批次闭合时FedSA为8/10；最新覆盖以本页当前表与STATE为准，原科学计算及1e-12保持。\n\n')
if state.get('gradient64_validation_search_20261010'):
    entry=state['gradient64_validation_search_20261010']
    observed=entry.get('latest_measured_observation',{})
    top+='\n新增梯度搜索已实际启动：Fed-NGA32+Huber32，70round/valid-only/n=1；新服务guardfed_celeba_gradient_screen64_v2a，1 worker/CPU105/GPU1/nice10/idle，启动时round '+str(entry['root_startup']['observed_round'])+'及physical GPU UUID已核。最新实测终轮70任务'+str(observed.get('terminal70_observed',0))+'/64，终轮严格离机根接受'+str(entry['offserver_accepted'])+'，二者分开记录。首Fed-NGA候选constant-negative（ACC0.516686、AEOD/ASPD0）保留，不作冠军判断。前两次root预检把宽调度mask、日志tee误判资源/重复worker的工程错误均在训练前退出并保留；科学source/64jobs原字节不变，既有800队列未改。实际凭据入口'+entry['root_startup_path']+'。\n'
if state.get('logofair32_validation_search_20261010'):
    entry=state['logofair32_validation_search_20261010']
    top+='\nLoGoFair32原30post-round搜索在F运行，4个原接受模型/cache已实际提取并核SHA；已原strict闭合'+str(entry['original_strict_closed'])+'/32，root adopted'+str(entry['root_adopted'])+'；不重训CNN、不评价test、不选未齐recipe，保留恒定预测与全部候选。输出'+entry['output']+'。\n'
    if entry['root_adopted']==32:
        top+='完整32原strict及635744项保存预测复核通过，冻结规则选LoGoFair-DP_07（global/local delta0.06、post_lr0.005、30轮）；准确率冠军不同、四项Pareto和8项恒定预测保留。单模型seed91001/fit_seed1719，四条件不当独立seed；虚拟20cohort非真实client公平性。\n'
if state.get('logofair100_fullcoverage_20261010'):
    l=state['logofair100_fullcoverage_20261010']
    if l.get('root_adopted')==100:
        top+='\nLoGoFair固定配置100覆盖已完整root采用：96新30轮后处理+4显式复用，IID/non-IID×五场景×十模型seed、fit_seed1719固定。原strict与1,986,700保存预测/300指标复算误差0，独立1027哈希/198统计/99展示通过；1项恒定预测及10/9/6面板完整保留。虚拟20cohort非真实client公平性，不重训CNN或test。bulk留F，四compact报告及恢复索引在本机；入口'+l['canonical_table']+'。\n'
    else:
        top+=f"\nLoGoFair固定配置100覆盖已实际启动：96新增+4复用，本机原strict观测{l['local_strict_closed_observed']}/96；完整独立验收尚未完成。入口{l['root_startup_path']}。\n"
if state.get('celeba_native_ten_method_table_20261010'):
    top=top.replace('完整17行比较仍缺8方法的完整多seed结果：LoGoFair、Fed-NGA、FedWA、Huber、FLGMM、SmartFL、FedDNA及组合控制。','完整17行比较现有10方法native千格表已独立采用，仍缺7方法完整多seed：Fed-NGA、FedWA、Huber、FLGMM、SmartFL、FedDNA及组合控制。')
    top+='十方法native论文表已独立采用：1000格、IID/non-IID各五场景，10/9/6seed面板；原九方法810统计对象精确保持，1800场景统计/900展示/540先seed内汇总统计/270汇总展示通过。LoGo为原fit_DP native，不能称1000三视图或final；入口'+state['celeba_native_ten_method_table_20261010']['table_path']+'。\n'
if state.get('mechanism_remaining620_valid_20261010'):
    entry=state['mechanism_remaining620_valid_20261010']
    top+='\n剩余620机制终轮valid三视图评价已实际启动'+entry['actual_service']+'，排除原180与Full重复推理；CPU112–119单进程8计算线程/nice10/idleIO/CUDA隐藏，首条原strict闭合native差0且绑定原已接受checkpoint。最新remote闭合'+str(entry.get('latest_measured_observation',{}).get('remote_strict_closed',entry['remote_strict_closed_at_startup']))+'，新离机根接受'+str(entry['new_offserver_accepted'])+'；不把服务运行或服务器闭合计入论文表。首次taskset包装语法错误发生在Python执行前，失败字节保留。首条运输因原验证工具路径缺失停止后，已有归档未重建，原49成员与9指标/24计数/3规则离机核验并与原native188 checkpoint恢复链精确join；单次有限恢复凭据独立保存，不盲重试、不重复推理。入口'+entry['root_startup_path']+'。\n'
    if entry.get('C100_table_adopted'):
        top+='C100完整IID/non-IID×五场景×十seed三视图表已独立采用；1620统计/810单元/1800指标/4800计数及全部配对、seed-first汇总通过，旧C80保持。入口celeba_mechanism_v1/three_view_C_full100_20261010/snapshot/TABLES.md；其他六变体及最终评价/正文未完成。\n'
    elif entry.get('C100_replay_complete'):
        top+='最后19份C三视图已严格离机及root采用：173归档成员/171指标/456计数/57规则、native偏差0；20份与native200逐记录相同，40个原model/result归档成员重核。旧180及first181不改，累计U100+C100=200；C100十场景统计表须独立采用，其他六变体仍未完成。\n'
    else:top+='C81仅比C80表多1条不完整场景记录，不新计均值。\n'
if main.get('three_view_counts_by_variant',{}).get('minus_A')==12:
    top+='\n新增12份minus_A终轮三视图已由原strict、离机110成员/108指标/288计数/36预测规则和native212恢复链核验，累计U100+C100+A12=212；24个原model/result成员重核、native偏差0。A IID Benign有10个共享seed，F Flip仅2个且不入均值；A场景表须单独采用，尚未完成A全部100格。入口tmp/celeba_mechanism_remaining620_A12_root_adoption_20261010/ROOT_ADOPTION.json。旧200与Full不重推/重包。\n'
if main.get('A_three_view_single_scene_table'):
    top=top.replace('A场景表须单独采用，尚未完成A全部100格','A IID Benign十seed三视图表已独立采用，尚未完成A全部100格')
    top+='A单场景已核162统计/81展示单元/216计数指标，入口'+main['A_three_view_single_scene_table']['table_path']+'；native/shared删除差ACC−0.430个百分点、AEOD+0.002313、ASPD−0.003142。9/6面板方向变化与Full2CPU8GPU对A10CPU保留，不作必要性/因果/显著性主张。\n'
if main.get('three_view_counts_by_variant',{}).get('minus_A')==20:
    top+='A IID F Flip新增八份已严格离机并与native218/220恢复链对齐：74成员/72指标/192计数/24预测规则及16原model/result成员核验通过，native差0。累计三视图U100+C100+A20=220；Benign和F Flip各10seed齐全，新增两场景表仍须独立统计采用，原Benign表保持。其余八个A场景及其他机制仍在训练。\n'
if main.get('A_three_view_two_scene_table'):
    top+='A两完整IID场景表已独立采用：Benign/F Flip各十共享seed，324统计/162展示/360计数指标/960基础计数通过；旧24记录和Benign162统计/81展示保持。F Flip native/shared删除A差ACC约+0.001pp、AEOD约+0.00001、ASPD约−0.00088，9/6面板方向变化完整保留；不作每场景不可或缺主张。入口'+main['A_three_view_two_scene_table']['table_path']+'。其他八A场景及六变体余项未齐。\n'
if main.get('three_view_counts_by_variant',{}).get('minus_A')==28:
    top+='A新增IID FedSA准确8项已严格离机、独立连接native228并根采用；74成员/72指标/192计数/24规则和16原model/result成员通过，native差0，累计U100+C100+A28=228。FedSA仅8/10，不生成均值或新场景表；原220及A两场景表保持。入口tmp/celeba_mechanism_remaining620_A28_root_adoption_20261010/ROOT_ADOPTION.json。\n'
if state['celeba_mechanism_v1'].get('three_view_counts_by_variant',{}).get('minus_A')==36:
    top+='最新A36八项已由原strict/离机74成员和native236恢复链核验并根采用，累计U100+C100+A36=236，原228不变；IID FedSA十seed记录齐备，S-DFA仅6/10不入完整场景均值。已审A20两场景表保持，新增场景统计尚未采用。入口tmp/celeba_mechanism_remaining620_A36_root_adoption_20261010/ROOT_ADOPTION.json。\n'
if state.get('author_adaptation_reply_patch_20261010'):
    top+='两项作者适配决定的英文回复/正文局部补丁已根核：24审稿原话保持、两个替换段可逆、原数字未变；正文未应用，入口'+state['author_adaptation_reply_patch_20261010']['reply_patch']+'。\n'
if state.get('gradient200_fullcoverage_source_preparation_20261010'):
    top+='Fed-NGA/Huber完整200格的192新+8复用源准备已独立审查通过，仅为source-only：未选recipe、未生成实际jobs、未启动。须等待全部64搜索严格离机、冻结实际配置及新增攻击真实图像门检；不从准备文件推断实验完成。\n'
if state.get('gradient200_new_attack_gates_source_20261010'):
    top+='新增攻击14项共同三轮门检源码已独立审查通过，实际图像门检仍0；仅source-only，不授权派发192。原正式70轮验收未放宽。\n'
if state.get('added_baseline_three_view_scope_20261010'):
    top+='新增五方法三视图接线仅完成源码范围审查：四CNN标签须接各自原strict，LoGo原生必须保留DP后处理与虚拟映射。其cache的valid_native_prediction是FedAvg原始预测，不得冒充LoGo native；backbone raw/shared仅可标为诊断。没有新增评价/拟合，最终主终点未定。\n'
if hybrid.get('status')=='ROOT_HYBRID32_SUMMARY_ADOPTED':
    top=top.replace('项已严格验收、离机并通过本机来源绑定的记录复核；尚未完整选recipe','项已严格验收、离机并经独立复核；冻结规则选λ20/τ0.1/lr0.001，n=1')
    top=top.replace('组合基线服务guardfed_celeba_hybrid_screen32仍运行，GPU0/CPU104单线程，未选完整recipe，组合100项尚未启动。','组合基线32项验证搜索已正常EXITED、0worker，完整严格离机并独立复核；冻结规则选λ20/τ0.1/lr0.001，同时为准确率冠军及唯一三指标Pareto候选。n=1，不报跨seed SD/显著性；100格尚未启动，接线需修复记录状态接口并完成七个真实短程门检。')
if state.get('hybrid100_fullcoverage_20261010'):
    top=top.replace('100格尚未启动，接线需修复记录状态接口并完成七个真实短程门检。','100格已生成96新+4显式复用清单；七个真实三轮门检已启动，实测首任务round2，正式70轮队列尚未启动。')
    top+='\n组合100接线已在独立v3目录实际绑定147个metadata成员，原Python3.10汇总在服务器3.12用显式顺序求和精确重放，未放宽容差/修改选择。原v2绑定失败发生在任务生成前，证据保留。七门检CPU104/GPU0单worker，启动前主队列真实增长/GPU RecoveryNone/CPU配额通过；门检科学70轮样本0。实际入口tmp/celeba_hybrid_fullcoverage_root_binding_v3_20261010/ROOT_CANARY_STARTUP.json。\n'
    if state['hybrid100_fullcoverage_20261010']['canaries_offserver_adopted']==7:
        top=top.replace('七个真实三轮门检已启动，实测首任务round2，正式70轮队列尚未启动。','七个真实三轮门检已全部严格验收及离机，254归档成员、两组同轮数新旧对照通过；正式70轮队列尚未启动。')
        top+='组合七门队列已正常EXITED，原strict/保存终轮张量/RNG摘要及全部成员哈希通过root验收；三轮不能证明70轮等价或每轮权重完整，论文样本0。入口'+state['hybrid100_fullcoverage_20261010']['canary_closure_path']+'。\n'
    if state['hybrid100_fullcoverage_20261010']['formal100_started']:
        top=top.replace('正式70轮队列尚未启动。','96新增70轮valid队列已实际启动，另4旧匹配结果显式复用；首worker实测round1，新增70轮接受仍0。')
        top+='组合完整覆盖CPU104/GPU0单worker/一计算线程，首worker PID85931的真实GPU UUID、源码/数据/配置身份及第一轮已核。旧canary授权原字节保留，新授权精确绑定七门离机root凭据；无自动重试、不运行test。入口'+state['hybrid100_fullcoverage_20261010']['coverage_start_path']+'。\n'
if main.get('A_three_view_four_scene_table'):
    top+='\n最新A40四完整IID场景表已独立采用：Benign/F Flip/FedSA/S-DFA各十共享seed，648均值SD/324展示/720计数指标/1920计数通过，旧A20的40对象字节/顺序、324统计/162展示精确保持。累计三视图240；S-DFA删除A的native/shared差ACC−0.361pp、AEOD+0.00542、ASPD−0.00804，FedSA为+0.134pp/−0.00065/+0.00141，各指标取舍和9/6方向变化均保留。Full3CPU37GPU/A40CPU、各40cu128；六其他A场景、剩余控制和最终评价/正文未齐，不作必要性/因果/显著性主张。入口'+main['A_three_view_four_scene_table']['table_path']+'。\n'
top+='\n本机存储：2026-10-10用户指定F:/YananResearchStorage/GuardFed/；大文件写入前实核F卷标Yanan 2TB与容量，无内置盘回退。代码/配置/索引/精简报告保留E，服务器大文件优先原地保留；已存在E证据未删除或宣称全量迁移。记录LOCAL_STORAGE_20261010.json。\n\n'
if state.get('hybrid100_fullcoverage_20261010',{}).get('formal100_started'):
    top=top.replace('组合基线服务guardfed_celeba_hybrid_screen32仍运行，GPU0/CPU104单线程，未选完整recipe，组合100项尚未启动。',
        '组合32搜索已正常完成并严格离机，冻结规则选λ20/τ0.1/lr0.001；七个真实三轮门检通过后，guardfed_celeba_hybrid_fullcoverage已实际启动96新+4复用70轮valid覆盖，CPU104/GPU0单线程，新增70轮接受0。')
if state.get('hybrid100_fullcoverage_20261010',{}).get('new_accepted',0):
    top=top.replace('新增70轮接受仍0','启动时新增70轮接受0，最新严格离机根接受1/96').replace('新增70轮接受0。','启动时新增70轮接受0，最新严格离机根接受1/96。')
    top+='Hybrid首项70轮IID Benign seed91002已由原strict、188归档成员和8张量摘要/源码数据配置身份独审及root采用，新增1/96；4旧复用另计，7三轮门检不算正式样本。不生成单seed场景SD，不称100覆盖完成。入口'+state['hybrid100_fullcoverage_20261010']['first_delta_root_path']+'。\n'
if state.get('celeba_native_ten_method_PDF_20261010'):
    top+='\n十方法native验证集表PDF已交付：三页分别为10/9/6共享seed，IID/non-IID各五场景，900组均值与sampleSD/1800数值逐项匹配源表，三页视觉检查通过。入口'+state['celeba_native_ten_method_PDF_20261010']['pdf_path']+'；仍缺余七方法完整覆盖及冻结最终评价。\n'
if replay251.exists():
    top=top.replace('累计三视图240；','A40表采用时累计三视图240；')
    top+='最新11条保存数组已独立核验101归档成员、99指标/264计数/33规则及22原native成员，累计三视图251=U100+C100+A51；native差0、原240保持。A的五IID场景各十seed齐备，non-IID Benign仅1/10；既有A40四场景表保持，新统计尚未汇总，不用部分seed填完整表。入口'+replay251.relative_to(ROOT).as_posix()+'。\n'
if state.get('added_CNN_three_view_bridge_20261010'):
    top+='四新增CNN的三视图身份桥源码及64拒收/边界检查经root复跑，17原评价函数保持；当前仅注册旧FL6/Hybrid1/NGA8精确接受chunk，无Huber70轮proof则拒收。没有新增科学评价/fit/训练或test，不把准备当完成。\n'
# Keep the accumulated cohort timeline as a compact historical artifact; the
# entry itself should state only the current accepted boundary and next work.
detail=CHECKS/'entry_history'/('detailed_current_'+hashlib.sha256(top.encode('utf8')).hexdigest()[:16]+'.md')
detail.parent.mkdir(parents=True,exist_ok=True)
if detail.exists():assert detail.read_bytes()==top.encode('utf8')
else:detail.write_bytes(top.encode('utf8'))
A_current=main.get('A_three_view_eight_scene_table',main.get('A_three_view_six_scene_table',main.get('A_three_view_five_scene_table',main.get('A_three_view_four_scene_table',{}))))
five=state.get('latest_five_queue_readonly_observation',{})
fl=state['flgmm_fullcoverage_v2_20261009'];gradient=state['gradient64_validation_search_20261010'];hy=state['hybrid100_fullcoverage_20261010']
hy_scene_note=('组合基线IID Benign十seed native验证表已独立及root采用，固定10/9/6面板、18个均值/样本SD标量和9个展示格核验；来源为9新增+1screen复用。仅此场景齐备，其他九场景和三视图未齐。入口'+hy['native_IID_Benign_table']['table_path']+'。' if hy.get('native_IID_Benign_table') else '')
hub_negative_note=('Huber首批7个终轮候选均恒负预测，ACC=0.5166859616449389、AEOD=ASPD=0，作为退化负结果保留，不据零差距称有效或选冠军。' if gradient.get('offserver_accepted')==39 else '')
fl47_note=''
if state.get('FLGMM_closed47_valid_three_view_20261011'):
    e=state['FLGMM_closed47_valid_three_view_20261011']
    fl47_note=f"FLGMM有限47项三视图评价已在CPU120–127/8线程单进程启动：{e['observed_utc']}观测回放结果{e['receipts_observed']}/47、失败{len(e['failures_observed'])}，新增科学采用仍0，须完成原whole saved检查及离机验收。准确范围44新训练+4旧复用−1已采用FL接口；不重训、不读test，GPU冻结队列不变。票据：{e['start_receipt_path']}。"
    if e.get('validation_hold'):
        fl47_note=f"FLGMM有限47项回放已正常结束；Linux完整原检查及F盘98成员哈希通过。本机原数组块在第5项{e['failed_id']}的Root-only threshold fit changed处停止，原失败保留，47项新增采用为0；前4循环通过不单独计采用。不重试失败命令或改容差；不影响原native验收或冻结训练。凭据：{e['validation_hold_path']}。"
        if e.get('single_failed_record_diagnostic'):
            d=e['single_failed_record_diagnostic']
            fl47_note+=f" 单记录诊断实测：shared校准诊断系数server_adaptive_lambda相差−1.1102230246251565e−16及其派生fit SHA不同，实际阈值/三视图预测/全部指标与计数/root receipt完全相同。初次比较器键类型错误另行保留；尚未证明平台原因，仍不采用47。诊断：{d['root_review_path']}。"
            if d.get('operation_trace_path'):
                fl47_note+=f" 两环境相同输入的逐运算实测在math.log1p首次分歧并各自复现保存系数；尚未单独隔离OS/libm/Python因素。此比较无拟合/推理/新采用，见{d['operation_trace_path']}。"
    if e.get('complementary_adopted'):
        fl47_note='FLGMM有限47条valid终轮三视图已按互补证据root采用：新增47＋此前单列1＝48，来源为44条新训练native验收＋4条screen复用；该FL批验收时机制三视图为260；当前机制接受数另列，不合并计数。Linux完整原检查承担47条root-only重拟合验收，F盘98成员运输核验通过；Windows保存输出审计47条通过、0次拟合，独立复核423指标/1128基础计数/141规则，native差0。Windows原始重拟合及whole仍FAIL，原hold/失败/单记录诊断与逐运算证据保留，不能称双平台重拟合逐位一致；4条root审计group_kl差−2.168404344971009e−19保留。未改容差、未新增训练/CNN/test，不代表完整100格或17方法完成。入口'+e['root_proof_path']+'；历史hold：'+e['validation_hold_path']+'。'
if state.get('FLGMM_after48_valid_three_view_20261011'):
    e=state['FLGMM_after48_valid_three_view_20261011']
    fl47_note='FLGMM三视图累计61个valid终轮checkpoint：此前48原记录保持，新增13完成Linux原whole检查/root-only拟合、F盘30成员SHA和Windows零拟合保存输出审计（117指标/312计数/39规则，native差0）。新增一条root审计group_kl差−2.168404344971009e−19保留；原Windows47重拟合和whole仍FAIL，新13未在Windows重拟合，不称跨平台逐位等价。来源57新增native训练＋4screen复用；与机制288分列，未新增训练或test。FL100及17方法未齐。入口'+e['root_proof_path']+'。'
    if e.get('six_scene_table'):
        t=e['six_scene_table']
        fl47_note+=' 六完整场景（五IID＋non-IID Benign）各十seed的三视图表已独立及root采用，固定10/9/6面板，324统计标量/162展示格、549计数派生指标核验；61条全保留，non-IID S-DFA的单条screen不进完整场景统计。IID alpha5000/non-IID alpha5，验证集选择史和校准取舍披露。表格：'+t['table_path']+'。'
top=f'''# CURRENT: GuardFed返修实验

服务器：ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded。用户已授权停止sglang，文件保留；不自动切回213实例。先遵守/etc/vast-agents-guide.md。

## 当前队列

以下“已验收”均经过原strict、离机SHA和root核验；实时完成文件与验收数量分列。主机制实测时间{live['checked_utc']}；五队列合并观测{five.get('utc','见STATE')}。

| 阶段 | 已验收 | 实测活动/剩余边界 |
|---|---:|---|
| 机制训练 | {main['scientific_results_strictly_accepted']}/800新增，100 Full另复用 | 观测完成{live['queue_completed']}、活动{len(live['active'])}、等待{live['pending']}、失败{len(live['failed'])}；固定70轮/8并发 |
| 机制三视图 | {main['three_view_new_models_offserver_verified']}终轮checkpoint | U100/C100各十场景；A{main['three_view_counts_by_variant'].get('minus_A',0)}的完整场景以接受索引及已采用表为准；远端闭合不等于离机验收 |
| FLGMM完整覆盖 | {fl['new_accepted']}/96新增，4复用另计 | 观测终轮{five.get('FLGMM_terminal','见STATE')}，2个worker有轮次增长 |
| Fed-NGA/Huber搜索 | {gradient['offserver_accepted']}/64 | 观测终轮{five.get('gradient_terminal','见STATE')}，单worker推进；所有候选/恒定预测保留，未选recipe |
| 组合基线完整覆盖 | {hy['new_accepted']}/96新增，4复用另计 | 观测终轮{five.get('Hybrid_terminal','见STATE')}，单worker推进；7个三轮门检不计正式样本 |

五队列已按真实worker身份、轮次增长、来源和错误检查核验；详见{five.get('growth_proof_path','STATE')}。最近主资源采样为CPU{live['cpu_used_cores_2sec']:.2f}/{live['cpu_quota_cores']:.2f}核、RAM{live['memory_used_bytes']/1e9:.2f}GB、磁盘余{live['disk_free_bytes']/1e12:.3f}TB；GPU瞬时利用率/显存/温度和Recovery原值见server_reactivation_20261009/latest_formal_live.json。这是该采样时刻的值，不代表连续占用；不因瞬时低利用率重启健康任务。冻结8并发/FP32/参数/seed不变。

{fl47_note}

{hy_scene_note}

{hub_negative_note}

## 已交付与尚缺

- 十方法native验证表已接受1000格，IID/non-IID各五场景，10/9/6种子；三页PDF：{state['celeba_native_ten_method_PDF_20261010']['pdf_path']}。仍缺其余7方法完整覆盖，不能称17方法完成。
- 九方法三视图900记录与2052项校准归因已接受；native和共享校准的优势方向不同，准确率代价及负结果保留。这仍是验证集证据。
- U/C各100对三视图表已接受。A最新{A_current.get('paired_models',0)}对、{A_current.get('complete_scenes',0)}个完整场景：{A_current.get('table_path','见STATE')}。{'A80范围为五IID及non-IID Benign/F Flip/FedSA；五IID seed-first聚合原字节保持，non-IID单列；S-DFA/Sp-DFA尚未齐。' if main.get('A_three_view_eight_scene_table') else 'A60范围为五IID及non-IID Benign；五IID seed-first聚合保持原字节，non-IID单列，其他四non-IID场景未齐。'}删除项存在指标取舍，不声称每项不可或缺。
- 24条原意见完整英文作者审阅稿，优先阅读清晰版：{state['latest_rebuttal_draft'].get('clear_reader_entry',state['latest_rebuttal_draft']['entry'])}。详细证据版：{state['latest_rebuttal_draft']['entry']}。{'A80八场景及预测规则依赖解释已纳入；' if state['latest_rebuttal_draft'].get('A80_incorporated') else '已有配对消融及seed-first取舍已纳入；'}原意见/旧数字/整表保留在详细版；清晰版保留关键结论、反例与全部未完成边界。正文插入稿尚未应用到提交版源项目。
- 旧TableII的480个原值已追溯真实重复数；缺依据的SD不补造。Fig3终轮候选已核260记录/78均值，但执行来源缺口仍保留。

Huber采用作者接受的恒等投影，明确CNN项目适配且不继承原理论；LoGoFair采用固定图像ID的20虚拟cohort，不能称真实client公平性。LoGoFair100已接受，cache原始预测不冒充其DP后处理native。新增CNN三视图身份桥源码/64门检已通过。FLGMM、组合及一个NGA搜索checkpoint的三条真实图像接口已核验采用：Linux完整原检查、F盘原保存数组/校准重拟合块通过，27指标/72基础计数/9规则和原native差0；Windows完整检查的FL审计group_kl约2.2e-19差异保留，不改容差，不称Windows全文检查通过。仅3代表接口，不是三方法100格齐备或最终test。入口tmp/celeba_added_cnn_exact3_root_execution_20261010/ROOT_SCIENTIFIC_ADOPTION.json。

下一步继续冻结队列，按有价值批次增量验收；补齐七方法、其余机制控制及新增CNN三视图。FedWA/SmartFL/FedDNA忠实规格、最终主终点/测试边界、提交版LaTeX源及Fig3来源仍待解决；已提出的作者问题不重复询问。历史test暴露、seed91001选择史、混合推理设备与cu128/cu130/driver差异保持披露，不把短程门检写成70轮等价。

官方最终评价候选分区已作独立ID元数据核验：partition2共19962张、顺序SHA见final_split_metadata_20261011/ROOT_METADATA_VERIFICATION.json。仅解码image_id/split并核整体文件身份；未解码标签数组、读取像素或模型、拟合或test推理。旧准备协议原字节保留；该事实不选择主终点、不冻结最终协议，历史test暴露仍披露。

## 保存、发布与巡检

本机大文件只写F:/YananResearchStorage/GuardFed，写前核F为Yanan 2TB且容量足；服务器大文件优先原地保留。E只保留代码、索引、配置与精简报告，不删除科学原始证据。

最近验证推送{published['commit']}（分支codex/revision-evidence-baselines-20260928）；发布截止以{published['proof_path']}为准，后续本机新增不自动算已推送。

三小时聊天任务guardfed-training-health仍PAUSED；本会话无automation_update接口，未建立替代cron/Windows任务。supervisor运行训练不等于聊天定时巡检恢复。后续接续读TRAINING_STATE.json和server_reactivation_20261009/MONITOR_HANDOFF.md。

工程失败、恢复和旧批次完整时间线：{detail.relative_to(ROOT).as_posix()}。下方历史原字节保留；当前事实以本段、STATE和实测凭据为准。

'''
running.write_text(top+history,encoding='utf-8')
execution = TRAIN / 'celeba_mechanism_v1/EXECUTION.md'
text = execution.read_text(encoding='utf-8')
marker = '# HISTORICAL PREPARATION SNAPSHOT — no execution at time of preparation\n\n'
if marker in text:
    text = text.split(marker,1)[1]
execution.write_text(top.replace('# CURRENT:', '# CURRENT EXECUTION:')+marker+text,encoding='utf-8')
print(json.dumps({'status':phase,'measured_utc':live['checked_utc'],'completed':live['queue_completed'],'active':len(live['active']),'formal':formal}))
