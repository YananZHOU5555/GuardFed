"""One native C scene; reuse accepted evidence_v4 statistics without inference."""
import hashlib
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METRICS = ['accuracy_pct', 'aeod', 'aspd']
SEEDS = list(range(91001, 91011))
SCENES = {(d, a) for d in ['IID', 'non-IID'] for a in ['Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA']}
PANELS = [('All 10 seeds', SEEDS), ('Exclude selection seed: 9 seeds', SEEDS[1:]), ('Seeds 91005–91010: 6 seeds', SEEDS[4:])]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_bytes())
def save(name, value):
    with (HERE / name).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def validate_scope(inspection):
    assert inspection['source_script_sha256'] == '3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
    assert inspection['manifest_sha256'] == '498ed0e033ef6eb5532286820ec987af3b44ca6251d8a84c42ce9c3e5ff5ab2d'
    assert not inspection['invalid'] and inspection['new_count'] == 112 and inspection['reused_count'] == 100
    rows = inspection['records']
    assert len(rows) == len({r['id'] for r in rows}) == len({(r['variant'], r['distribution'], r['attack'], r['seed']) for r in rows}) == 212
    assert set(inspection['accepted_new_ids']) == {r['id'] for r in rows if r['variant'] != 'Full'}
    assert set(inspection['accepted_reused_ids']) == {r['id'] for r in rows if r['variant'] == 'Full'}
    for v in ['Full', 'minus_U']:
        assert {(r['distribution'], r['attack'], r['seed']) for r in rows if r['variant'] == v} == {(d, a, s) for d, a in SCENES for s in SEEDS}
    controls = [r for r in rows if r['variant'] not in ['Full', 'minus_U']]
    assert len(controls) == 12 and {r['variant'] for r in controls} == {'minus_C'}
    assert {(r['distribution'], r['attack'], r['seed']) for r in controls} == {('IID', 'Benign', s) for s in SEEDS} | {('IID', 'F Flip', s) for s in SEEDS[:2]}
    assert all(len(r['checkpoint_sha256']) == 64 and r['files'] and r['prediction_support']['prediction_count'] == 19867 for r in rows)
    selected = [r for r in rows if r['variant'] in ['Full', 'minus_C'] and (r['distribution'], r['attack']) == ('IID', 'Benign')]
    assert len(selected) == 20
    return rows, selected


def main():
    basis = read(HERE / 'INPUTS.json')
    for name, pin in basis['files'].items():
        assert sha(ROOT / name) == pin['sha256'] and (ROOT / name).stat().st_size == pin['bytes'], name
    inspection = read(ROOT / basis['inspection']); rows, selected = validate_scope(inspection)
    prior = read(ROOT / basis['prior_inspection']); indexed = {r['id']: r for r in rows}
    assert len(prior['records']) == 204 and all(indexed[r['id']] == r for r in prior['records'])
    proof = read(ROOT / basis['root_proof']); receipt = read(ROOT / basis['receipt']); offserver = read(ROOT / basis['offserver'])
    assert proof['status'] == 'ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS' and proof['total_new_strict_and_offserver'] == 112
    assert proof['inspection_sha256'] == sha(ROOT / basis['inspection']) and proof['ledger_sha256'] == sha(ROOT / basis['ledger'])
    assert proof['receipt_sha256'] == sha(ROOT / basis['receipt']) and proof['offserver_proof_sha256'] == sha(ROOT / basis['offserver'])
    assert offserver['pass'] and offserver['different_host_observed'] and proof['archive_sha256'] == offserver['archive_sha256'] == receipt['archive_sha256']
    previous_receipt = read(ROOT / basis['prior_receipt'])
    assert receipt['previous_receipt_sha256'] == sha(ROOT / basis['prior_receipt'])
    ledger = read(ROOT / basis['ledger']); prior_ledger = read(ROOT / basis['prior_ledger'])
    assert ledger['entries'][:-1] == prior_ledger['entries'] and ledger['entries'][-1]['receipt_sha256'] == sha(ROOT / basis['receipt'])
    manifest = read(ROOT / basis['manifest']); jobs = {j['id']: j for j in manifest['jobs']}
    full = {r['id']: r for r in read(ROOT / basis['full_inventory'])['records']}
    bindings = []
    for archive_name in basis['archives']:
        with tarfile.open(ROOT / archive_name) as archive:
            inventory = json.load(archive.extractfile('backup_inventory.json'))
            for row in sorted(selected, key=lambda r: (r['seed'], r['variant'])):
                if row['variant'] != 'minus_C' or row['id'] not in inventory['accepted_new_ids']: continue
                ident = row['id']; prefix = 'runs/' + ident + '/'; loaded = {}; members = {}
                for key, member in [('job', 'jobs/' + ident + '.json'), ('config', prefix + 'config.json'), ('result', prefix + 'result.json'), ('acceptance', prefix + 'mechanism_acceptance.json')]:
                    data = archive.extractfile(member).read(); pin = inventory['members'][member]
                    assert len(data) == pin['bytes'] and hashlib.sha256(data).hexdigest() == pin['sha256']
                    loaded[key] = json.loads(data); members[key] = dict(member=member, **pin)
                job, config, result, acceptance = (loaded[k] for k in ['job', 'config', 'result', 'acceptance'])
                full_row = next(r for r in selected if r['variant'] == 'Full' and r['seed'] == row['seed']); reference = full[full_row['id']]
                assert job['id'] == ident and job['variant'] == 'minus_C' and job['distribution'] == 'IID' and job['attack'] == 'Benign'
                assert jobs[ident]['job_sha256'] == members['job']['sha256'] == row['files'][row['job']]
                assert job['output'] == row['output'] == jobs[ident]['output'] and job['source_hashes'] == manifest['source_hashes']
                assert all(job['source_hashes'][k] == v for k, v in reference['source_hashes'].items())
                assert job['adapter_hashes'] == manifest['adapter_hashes'] == acceptance['adapter_hashes']
                assert job['config'] == config == result['config'] == result['revision_job']['config']
                assert config['ablation_component'] == 'C' and config['seed'] == result['seed'] == row['seed']
                assert config['rounds'] == result['rounds'] == acceptance['rounds'] == reference['terminal_round'] == 70
                assert [r['round'] for r in result['round_summaries']] == list(range(1, 71))
                assert config['celeba_evaluation_split'] == reference['original_split'] == 'valid' and reference['original_n_eval'] == 19867
                assert config['ad2_calibration_enabled'] and reference['config']['ad2_calibration_enabled']
                config_differences = {k: [reference['config'][k], config[k]] for k in config if reference['config'][k] != config[k]}
                assert set(config_differences) == {'ablation_component', 'experiment_suite', 'experiment_tag'}
                contract = result['data_contract']; image = contract['image_data_contract']
                assert image == reference['data_contract'] and contract['train_rows'] == 162770 and contract['root_clean_rows'] == 16277
                assert image['actual_evaluation_rows'] == 19867 and image['train_eval_disjoint'] and image['root_client_disjoint']
                assert row['group_denominator_support'] == full_row['group_denominator_support']
                assert full_row['checkpoint_sha256'] == reference['checkpoint']['sha256'] and full_row['torch_version'] == reference['training_torch']
                assert full_row['files'][full_row['output'] + '/result.json'] == reference['result']['sha256']
                for metric, rawkey in [('accuracy_pct', 'accuracy'), ('aeod', 'aeod'), ('aspd', 'aspd')]:
                    factor = 100 if metric == 'accuracy_pct' else 1
                    assert row[metric] == result['metrics'][rawkey] * factor and full_row[metric] == reference['prior_validation_metrics'][rawkey] * factor
                assert acceptance['pass'] and acceptance['candidate_calls_verified'] == acceptance['expected_candidate_calls'] == 700
                assert acceptance['checkpoint_sha256'] == row['checkpoint_sha256'] == inventory['members'][prefix + 'model.pt']['sha256']
                assert acceptance['job_sha256'] == members['job']['sha256'] and acceptance['result_sha256'] == members['result']['sha256'] == row['files'][row['output'] + '/result.json']
                assert acceptance['candidate_audit_sha256'] == inventory['members'][prefix + 'candidate_mask_audit.json']['sha256']
                bindings.append(dict(seed=row['seed'], id=ident, Full_id=full_row['id'], archive=archive_name, members=members,
                    checkpoint_sha256=row['checkpoint_sha256'], Full_checkpoint=reference['checkpoint'], Full_result=reference['result'], Full_raw_job=reference['raw_job'],
                    config=config, Full_config_canonical_sha256=reference['config_canonical_sha256'], config_differences=config_differences,
                    source_hashes=job['source_hashes'], adapter_hashes=job['adapter_hashes'], data_contract=contract, acceptance=acceptance,
                    native_prediction_rule=reference['native_prediction_rule'], Full_training_torch=reference['training_torch'], C_training_torch=row['torch_version']))
    assert len(bindings) == 10 and {b['seed'] for b in bindings} == set(SEEDS)
    spec = importlib.util.spec_from_file_location('accepted_evidence_v4', ROOT / basis['evidence'])
    evidence = importlib.util.module_from_spec(spec); spec.loader.exec_module(evidence)
    summary = evidence.summarize(selected); complete = [r for r in summary['paired_per_scene'] if r['complete']]
    assert len(complete) == 1 and complete[0]['variant'] == 'minus_C'
    panels = []; text = ['# CelebA: native Full–minus_C, IID Benign', '',
        'Validation-only, round 70, valid n=19,867. Ten shared declared seeds; mean ± sample SD (ddof=1). The paired row is minus_C − Full, computed within seed before summarizing. ACC is in percent; its paired difference is in percentage points.', '',
        'This fixed native112 snapshot contains Full100, minus_U100 and minus_C12. Only IID Benign is complete for C; IID F Flip has 2/10 seeds and the other eight C scenes have 0/10. No incomplete C scene contributes to this table.', '']
    for label, seeds in PANELS:
        text += ['## ' + label, '', '| Variant / paired difference | n | ACC (%) / ΔACC (pp) | AEOD | ASPD |', '|---|---:|---:|---:|---:|']
        panel = dict(label=label, seeds=seeds, rows=[])
        by_variant = {}
        for variant in ['Full', 'minus_C']:
            records = [r for r in selected if r['variant'] == variant and r['seed'] in seeds]
            by_variant[variant] = {r['seed']: r for r in records}
            panel['rows'].append(dict(distribution='IID', attack='Benign', variant=variant, **evidence.statistic(records, len(seeds))))
        differences = [dict(seed=s, **{m: by_variant['minus_C'][s][m] - by_variant['Full'][s][m] for m in METRICS}) for s in seeds]
        panel['rows'].append(dict(distribution='IID', attack='Benign', variant='minus_C minus Full', **evidence.statistic(differences, len(seeds))))
        for row in panel['rows']:
            values = [f"{row[m]['mean']:.3f} ± {row[m]['sample_sd_ddof1']:.3f}" if m == 'accuracy_pct' else f"{row[m]['mean']:.5f} ± {row[m]['sample_sd_ddof1']:.5f}" for m in METRICS]
            text.append('| ' + row['variant'] + ' | ' + str(len(seeds)) + ' | ' + ' | '.join(values) + ' |')
        panels.append(panel); text.append('')
    text += ['AEOD is the absolute TPR gap, not full equalized odds; smaller AEOD/ASPD indicates less disparity. Native metrics retain each procedure’s original root-fitted calibration. This single deletion does not isolate calibration effects or establish that C is necessary or causal.', '',
        'The 9-seed panel omits selection seed 91001; the 6-seed panel retains 91005–91010 for both variants. All seeds have prior validation exposure; neither panel is an untouched confirmation set. Historical test exposure and validation-based recipe selection remain limitations; no new test use occurs here.', '',
        'Full reuses historical checkpoints; all ten shown Full and C records use PyTorch 2.11.0+cu128, while their historical/current driver environments differ (current C driver 595.84). The full reference cohort includes 98 cu128 and 2 cu130 records, but neither cu130 record is in this scene. This native table uses accepted original training metrics, not mixed-device three-view replay results. The formal native/shared primary endpoint remains pending author selection.', '',
        'No significance test, superiority guarantee, new inference or training is performed. Source identities and per-seed paired differences are preserved in the accompanying JSON. This single scene does not complete the set of eight mechanism controls or the complete mechanism900 comparison.', '',
        'Inspection SHA256: `' + sha(ROOT / basis['inspection']) + '`. Original statistics source SHA256: `' + sha(ROOT / basis['evidence']) + '`.', '']
    with (HERE / 'TABLES.md').open('x', encoding='utf-8', newline='\n') as f: f.write('\n'.join(text))
    save('tables.json', dict(status='NATIVE_C_IID_BENIGN_COMPLETE_SINGLE_SCENE', inspection_sha256=sha(ROOT / basis['inspection']), evidence_source_sha256=sha(ROOT / basis['evidence']), panels=panels, complete_paired_scenes=1, accepted_new_total=112, new_inference=0, new_training=0, test=False, primary_endpoint='PENDING_AUTHOR'))
    save('records.json', dict(inspection_sha256=sha(ROOT / basis['inspection']), records=selected))
    save('paired_differences.json', dict(direction='minus_C minus Full', records=summary['paired_per_seed'], checkpoint_pairs=[dict(seed=b['seed'], id=b['id'], Full_id=b['Full_id'], checkpoint_sha256=b['checkpoint_sha256'], Full_checkpoint_sha256=b['Full_checkpoint']['sha256']) for b in sorted(bindings, key=lambda b: b['seed'])]))
    save('identity_bindings.json', dict(status='ACCEPTED_ORIGINAL_NATIVE_IDENTITIES_BOUND_NO_MODEL_READ', bindings=sorted(bindings, key=lambda b: b['seed']), root_proof_sha256=sha(ROOT / basis['root_proof']), independent_root_review_sha256=sha(ROOT / basis['independent_root_review'])))
    save('coverage.json', dict(original_record_count=212, original_records_source=basis['inspection'], accepted_new112_ids=inspection['accepted_new_ids'], Full100_ids=inspection['accepted_reused_ids'], minus_U_count=100, minus_C_count=12,
        minus_C_scenes=[dict(distribution=d, attack=a, seeds=sorted(r['seed'] for r in rows if r['variant']=='minus_C' and (r['distribution'],r['attack'])==(d,a)), expected_n=10) for d,a in sorted(SCENES)], selected_record_count=20, paired_seeds=SEEDS, panel_paired_counts=[10,9,6], full_cohort_torch_counts=inspection['torch_counts_by_role'], dispatch_environment=inspection['dispatch_environment'], other_C_scenes_complete=False, whole_mechanism900_complete=False, three_views_claimed=False))
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    print(json.dumps(dict(status='BUILT_NATIVE_C_BENIGN10', records=20, paired=10, panels=3, display_cells=27, mean_sd_scalars=54, CNN=0)))


if __name__ == '__main__': main()
