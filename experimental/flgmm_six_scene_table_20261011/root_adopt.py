"""Root adoption of the fixed six-scene table; compact receipts only, no refits."""
from pathlib import Path
from datetime import datetime, timezone
import copy, csv, hashlib, json, math, subprocess

R = Path(__file__).resolve().parents[2]
H = Path(__file__).resolve().parent
O = R / 'outputs/guardfed_tables/celeba_flgmm_six_scenes60_20261011'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
canonical = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def main():
    target = O / 'ROOT_VERIFICATION.json'
    require(not target.exists(), 'Adoption is once only; preserve earlier proof')
    volume = json.loads(subprocess.check_output(['powershell', '-NoProfile', '-Command', 'Get-Volume -DriveLetter F | Select-Object FileSystemLabel,HealthStatus,SizeRemaining | ConvertTo-Json -Compress']))
    require(volume['FileSystemLabel'] == 'Yanan 2TB' and volume['HealthStatus'] == 'Healthy', 'F evidence volume unavailable')
    seal = H / 'FILES_SHA256.json'
    require(sha(seal) == '6855caf56a7337a34c7e20640783b1005e65ad3b25721b76062ec9a327556f9c', 'Handoff seal changed')
    members = read(seal)['members']
    require(len(members) == 20, 'Unexpected sealed denominator')
    for pin in members:
        p = (R / pin['path']).resolve()
        require(p.is_relative_to(R) and p.stat().st_size == pin['bytes'] and sha(p) == pin['sha256'], 'Sealed member changed: ' + str(p))
    binding = read(O / 'SOURCE_BINDINGS.json')
    require(len(binding['sources']) == 73, 'Unexpected source denominator')
    for path, pin in binding['sources'].items():
        p = Path(path)
        require(sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes'], 'Source bytes changed')
    root_path = Path(binding['root61']['path'])
    require(sha(root_path) == binding['root61']['sha256'] == 'd6c7bdadb05ffcf8786221ed15a16125cd0fe84d1745fc15f9b7e6cc0a2f68d6', 'Actual root61 changed')
    adopted = read(root_path)
    require(adopted['root_adoption'] is True and adopted['FLGMM_total_three_view_records'] == 61, 'No actual61 adoption')
    data = read(O / 'records61.json')
    rows = data['records']
    require([x['root_adoption_record'] for x in rows] == adopted['records'] + adopted['prior_interface_explicitly_reused'], 'Adopted objects/order changed')
    require(len(rows) == len({x['id'] for x in rows}) == 61, 'Record uniqueness changed')
    scenes = [('IID', a) for a in ['Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA']] + [('non-IID', 'Benign')]
    partial = ['FLGMM_Tg20_L2.0_lr0.001_non-IID_S-DFA_seed91001_screen']
    require([x['id'] for x in rows if (x['distribution'], x['attack']) not in scenes] == data['retained_partial_ids'] == partial, 'Partial record boundary changed')
    require(sum(x['included_complete_scene'] for x in rows) == 60, 'Complete-scene denominator changed')
    views = ['raw', 'native', 'shared_calibration']
    metrics = ['accuracy', 'aeod', 'aspd']
    manifests = {}
    metric_count, integer_count, maximum_metric_delta = 0, 0, 0.0
    for row in rows:
        p = Path(row['receipt_path'])
        require(p.resolve().is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()), 'Receipt outside F evidence volume')
        receipt = read(p)
        require(sha(p) == row['receipt_sha256'] and receipt['views'] == row['views'] and receipt['fits'] == row['fits'], 'Accepted receipt content changed')
        require(receipt['checkpoint_sha256'] == row['checkpoint_sha256'] == row['root_adoption_record']['checkpoint_sha256'], 'Checkpoint join changed')
        mp = Path(row['source_manifest_path'])
        require(sha(mp) == row['source_manifest_sha256'], 'Manifest changed')
        if str(mp) not in manifests:
            manifests[str(mp)] = {x['id']: x for x in read(mp)['records']}
        source = manifests[str(mp)][row['id']]
        identity = copy.deepcopy(source['identity'])
        identity.update(distribution=source['distribution'], attack=source['attack'], seed=source['seed'], result=identity['original_artifact_pins']['result'], raw_job=identity['original_artifact_pins']['job'], config_canonical_sha256=canonical(identity['config']), training_torch=source['original_training_torch'])
        require(canonical(identity) == row['external_identity_record_sha256'] == receipt['external_identity_record_sha256'], 'Frozen source identity changed')
        cfg = identity['config']
        require(cfg['rounds'] == row['terminal_round'] == 70 and cfg['celeba_evaluation_split'] == row['evaluation_split'] == 'valid', 'Nonterminal/test record')
        require(cfg['client_alpha'] == {'IID': 5000.0, 'non-IID': 5.0}[row['distribution']] and cfg['learning_rate'] == .001 and cfg['celeba_train_limit'] == cfg['celeba_eval_limit'] == 0, 'Frozen partition or recipe changed')
        require(receipt['weights_before'] == receipt['weights_after'] and row['valid_n'] == receipt['valid_n'] == 19867, 'Weight/sample denominator changed')
        require(row['valid_image_ids_sha256'] == receipt['valid_image_ids_sha256'] == '64a15cf28caf1d177ac3dcf96a4408bc21091796974b243923ca947a37b554bf', 'Validation IDs changed')
        require(row['views']['native'] == row['views']['raw'] and row['same_checkpoint_all_views'], 'Native/raw equivalence changed')
        for view in views:
            value = row['views'][view]
            counts = value['group_confusion_counts']
            require(set(counts) == {'0', '1'}, 'Protected groups changed')
            for g in counts.values():
                require(all(type(g[k]) is int and g[k] >= 0 for k in ['tp', 'fp', 'tn', 'fn']), 'Invalid confusion counts')
                require(g['n'] == g['tp'] + g['fp'] + g['tn'] + g['fn'] and g['positives'] == g['tp'] + g['fn'] > 0 and g['negatives'] == g['tn'] + g['fp'] > 0, 'Group marginals changed')
                integer_count += 4
            a, b = counts['0'], counts['1']
            n = a['n'] + b['n']
            require(n == value['prediction_count'] == 19867, 'Evaluation denominator changed')
            calculated = [(a['tp'] + a['tn'] + b['tp'] + b['tn']) / n, abs(a['tp'] / a['positives'] - b['tp'] / b['positives']), abs((a['tp'] + a['fp']) / a['n'] - (b['tp'] + b['fp']) / b['n'])]
            for name, v in zip(metrics, calculated):
                delta = abs(v - value[name])
                require(math.isfinite(value[name]) and delta <= 1e-15, 'Metric/count inconsistency')
                maximum_metric_delta = max(maximum_metric_delta, delta)
                metric_count += 1
            fit = row['fits'][view]
            require(canonical({k: v for k, v in fit.items() if k != 'fit_sha256'}) == fit['fit_sha256'], 'Saved fit pin changed')
    table = read(O / 'tables.json')
    fixed = {'ten': list(range(91001, 91011)), 'nonselection_nine': list(range(91002, 91011)), 'matching_six': list(range(91005, 91011))}
    require(table['scenes'] == [list(x) for x in scenes] and table['views'] == views and table['fixed_seed_panels'] == fixed, 'Table scope changed')
    require([(p['view'], p['panel']) for p in table['panels']] == [(v, p) for v in views for p in fixed], 'Panel order changed')
    display_rows, csv_rows = [], []
    scalar_count, maximum_stats_delta = 0, 0.0
    for panel in table['panels']:
        seeds = fixed[panel['panel']]
        require(panel['seeds'] == seeds and panel['n'] == len(seeds), 'Panel seed identity changed')
        require([(x['distribution'], x['attack']) for x in panel['scenes']] == scenes, 'Scene order changed')
        cells = {k: [] for k in metrics}
        for cell in panel['scenes']:
            chosen = sorted([r for r in rows if (r['distribution'], r['attack']) == (cell['distribution'], cell['attack']) and r['seed'] in seeds], key=lambda r: r['seed'])
            require([x['seed'] for x in chosen] == seeds and [x['id'] for x in chosen] == cell['ids'] and cell['n'] == len(seeds), 'Cell seed membership changed')
            for metric in metrics:
                vals = [r['views'][panel['view']][metric] for r in chosen]
                mean = math.fsum(vals) / len(vals)
                sd = math.sqrt(math.fsum((x - mean) ** 2 for x in vals) / (len(vals) - 1))
                actual = cell['metrics'][metric]
                for k, v in [('mean', mean), ('sample_sd', sd)]:
                    delta = abs(actual[k] - v)
                    require(delta <= 1e-14, 'Root independent sample statistic differs')
                    maximum_stats_delta = max(maximum_stats_delta, delta)
                    scalar_count += 1
                scale, precision = (100, 2) if metric == 'accuracy' else (1, 4)
                rendered = f'{mean * scale:.{precision}f} ± {sd * scale:.{precision}f}'
                require(rendered == actual['display'], 'Displayed rounding changed')
                cells[metric].append(rendered)
                csv_rows.append([panel['view'], panel['panel'], str(len(seeds)), cell['distribution'], cell['attack'], metric, str(actual['mean']), str(actual['sample_sd']), rendered])
        for k, label in zip(metrics, ['ACC (%) ↑', 'AEOD ↓', 'ASPD ↓']):
            display_rows.append('| ' + label + ' | ' + ' | '.join(cells[k]) + ' |')
    final_md = (O / 'TABLES.md').read_bytes()
    actual_rows = [line for line in final_md.decode().splitlines() if line.startswith(('| ACC (%) ↑ |', '| AEOD ↓ |', '| ASPD ↓ |'))]
    require(actual_rows == display_rows, 'Markdown cells changed')
    with (O / 'cells.csv').open(encoding='utf8', newline='') as f:
        require(list(csv.reader(f))[1:] == csv_rows, 'CSV cells changed')
    old_md = (H / 'TABLES_before_caption.md').read_bytes()
    edit = read(H / 'CAPTION_EDIT.json')
    require(sha(H / 'TABLES_before_caption.md') == edit['original_table_sha256'] and sha(O / 'TABLES.md') == edit['final_table_sha256'], 'Caption source/final pin changed')
    require(final_md.split(b'## raw', 1)[1] == old_md.split(b'## raw', 1)[1], 'Caption edit altered scientific body')
    require(metric_count == 549 and integer_count == 1464 and scalar_count == 324 and len(csv_rows) == 162, 'Root check denominator changed')
    proof = dict(status='ROOT_FLGMM_SIX_COMPLETE_SCENES60_TABLE_ADOPTED', verified_utc=datetime.now(timezone.utc).isoformat(), root_adoption=True, source_root61_sha256=sha(root_path), sealed_members_verified=20, source_identity_pins_verified=73, records_preserved=61, complete_scene_records=60, retained_partial_records=1, scenes=6, views=views, fixed_seed_panels=fixed, metric_count=metric_count, base_integer_count=integer_count, statistical_scalars=scalar_count, Markdown_cells=162, CSV_cells=162, max_count_metric_difference=maximum_metric_delta, max_independent_fsum_statistic_difference=maximum_stats_delta, caption_only_body_equality=True, original_verification_sha256=sha(O / 'VERIFICATION.json'), seal_sha256=sha(seal), source_sha256=sha(__file__), adopted_outputs={n: sha(O / n) for n in ['records61.json', 'tables.json', 'TABLES.md', 'cells.csv', 'SOURCE_BINDINGS.json', 'CAPTION.md']}, arrays_read=False, new_fit=0, new_inference=0, new_training=0, final_test=False, whole_rebuttal_complete=False, prior_Windows47_exact_refit_failure_preserved=True, universal_bitwise_refit_claim=False)
    with target.open('x', encoding='utf8') as f:
        json.dump(proof, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(status=proof['status'], proof_sha256=sha(target), records=61, complete_scene_records=60, metrics=metric_count, scalars=scalar_count, cells=162)))

if __name__ == '__main__':
    main()
