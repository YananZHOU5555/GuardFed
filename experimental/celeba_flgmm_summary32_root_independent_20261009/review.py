"""Bounded, read-only SUMMARY32 audit; never imports a collector or runs summary.main."""
from pathlib import Path
import ast, copy, datetime, hashlib, json, math, statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'tmp/celeba_flgmm_final6_closure_20261009'
FINAL = BASE / 'actual_20261009T194424Z'
SUMMARY = BASE / 'summary32_final/SUMMARY32.json'
WRAPPER = ROOT / 'tmp/run_flgmm_summary32_legacy2_root_20261009.py'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
pins = {}
def pin(p, expected=None):
    digest = sha(p)
    if expected is not None:
        assert digest == expected, (str(p), digest, expected)
    pins[p.relative_to(ROOT).as_posix()] = digest
    return digest

pin(SUMMARY, 'd8f32441402f0ce7870bacc34e94de2a334133b2881d0a59ccf5123ef0781291')
pin(FINAL / 'ROOT_ADOPTION_REVIEW.json', '66097564f346b0dd5a0194ea7bd8c1413d51936819db086283f1d9a99a85738e')
pin(WRAPPER, '303ee5a49e7f1ca3ee846b43e3e9f4bc70584a764e5c9bd459b06e730edc7985')
pin(BASE / 'FILES_SHA256.json', '1d7cc11f95727a57478dd8575f65170024b36df98187030ac2de05e1e08cf6b9')
seal = read(BASE / 'FILES_SHA256.json')
for name, item in seal['files'].items():
    pin(BASE / name, item['sha256'])
    assert (BASE / name).stat().st_size == item['bytes']
protocol = read(BASE / 'protocol.json')
manifest = read(BASE / 'manifest.json')
index = read(BASE / 'PRIOR26_RECORD_SOURCES.json')
previous = read(BASE / 'PREVIOUS_CHAIN.json')
summary = read(SUMMARY)
proof = read(FINAL / 'ROOT_ADOPTION_REVIEW.json')
assert proof['status'] == 'ROOT_LOCKED_AUXILIARY_DELTA_ARCHIVE_MEMBER_AND_ORIGINAL_RECORD_ACCEPTANCE_PASS'
assert (proof['accepted_before'], proof['accepted_new'], proof['accepted_total']) == (26, 6, 32)
assert proof['previous_chain_sha256'] == index['parent_chain_sha256'] == sha(BASE / 'PREVIOUS_CHAIN.json')
assert proof['prepared_seal_sha256'] == sha(BASE / 'FILES_SHA256.json')
assert proof['new_inference'] == 0 and proof['selection_performed'] is False
assert proof['scientific_changes'] is False and proof['final_test'] is False and proof['formal100_started'] is False
pin(ROOT / index['parent_chain_path'], index['parent_chain_sha256'])
pin(ROOT / previous['root_adoption_path'], previous['root_adoption_sha256'])
pin(FINAL / 'PARTIAL_ACCEPTANCE.json', proof['strict_receipt_sha256'])
pin(FINAL / 'OFFSERVER_ACCEPTANCE.json', proof['offserver_proof_sha256'])
pin(FINAL / 'accepted_final6_delta.tar.gz', proof['archive_sha256'])
pin(FINAL / 'BACKUP_SHA256.json')
failure_path = BASE / 'ROOT_SUMMARY_ATTEMPT1_SCHEMA_FAILURE.json'
pin(failure_path, '86092308fff51c5d32f65389f0f790fe2df36d1b2b3aa61fa338b29f83d87d61')
assert read(failure_path)['output_directory_created'] is False

records = []
batches = []
legacy_server = legacy_offserver = None
pair_fields = ('rounds', 'seed', 'distribution', 'attack', 'metrics', 'checkpoint_sha256')
for entry in index['sources'] + [{'server_path': (FINAL/'PARTIAL_ACCEPTANCE.json').relative_to(ROOT).as_posix(), 'offserver_path': (FINAL/'OFFSERVER_ACCEPTANCE.json').relative_to(ROOT).as_posix(), 'pins': {}}]:
    for name, digest in entry['pins'].items():
        pin(ROOT / name, digest)
    server, off = read(ROOT / entry['server_path']), read(ROOT / entry['offserver_path'])
    assert off['status'] == 'PARTIAL_ACCEPTED_OFFSERVER_VERIFIED'
    assert off['original_checked_result_replayed_locally'] is True
    assert server['before_source_data'] == server['after_source_data'] == protocol['source_hashes']
    assert len(protocol['source_hashes']) == 21
    a, b = {r['id']: r for r in server['records']}, {r['id']: r for r in off['records']}
    assert len(a) == len(b) == len(server['records']) == len(off['records']) and set(a) == set(b)
    for key, row in a.items():
        assert all(row[field] == b[key][field] for field in pair_fields)
    if len(batches) == 0:
        assert sha(ROOT / entry['offserver_path']) == 'a53c87cbe62cdcc206dddb18cec0f62426292c05adb28a4a00de03fa64fe3750'
        assert len(a) == 2 and off['accepted'] == 2 and 'different_host_observed' not in off
        legacy_server, legacy_offserver = server, off
    else:
        assert off['different_host_observed'] is True
    assert server['final_test'] is False and off['final_test'] is False
    records.extend(server['records'])
    batches.append({'server_path': entry['server_path'], 'offserver_path': entry['offserver_path'], 'records': len(a), 'source_data_pins': 21})

assert [b['records'] for b in batches] == [2, 4, 7, 6, 7, 6]
assert len(records) == len({r['id'] for r in records}) == 32
assert set(r['id'] for r in records[:26]) == set(previous['accepted_job_ids'])
assert set(r['id'] for r in records[26:]) == set(read(BASE / 'EXACT_DELTA.json')['selected_ids'])
final_receipt = read(FINAL / 'BACKUP_SHA256.json')
assert final_receipt['accepted_new'] == 6 and final_receipt['accepted_total'] == 32
assert final_receipt['archive_sha256'] == proof['archive_sha256']
assert read(FINAL/'OFFSERVER_ACCEPTANCE.json')['server_backup_receipt_sha256'] == sha(FINAL/'BACKUP_SHA256.json')

# Execute only two extracted pure record-reader definitions, never either module/main.
def function_node(path, name):
    return next(n for n in ast.parse(path.read_text('utf8')).body if isinstance(n, ast.FunctionDef) and n.name == name)
original_env = {}
exec(compile(ast.Module(body=[function_node(BASE/'summary32.py', 'same_records')], type_ignores=[]), '<original-reader-only>', 'exec'), original_env)
wrapper_env = {'legacy': legacy_offserver, 'BASE': BASE, 'json': json, 'original': original_env['same_records']}
exec(compile(ast.Module(body=[function_node(WRAPPER, 'same_records')], type_ignores=[]), '<wrapper-reader-only>', 'exec'), wrapper_env)
reader = wrapper_env['same_records']
assert reader(legacy_server, legacy_offserver) == legacy_server['records']
assert reader(read(FINAL/'PARTIAL_ACCEPTANCE.json'), read(FINAL/'OFFSERVER_ACCEPTANCE.json')) == records[26:]
def refuses(a, b):
    try:
        reader(a, b)
    except (AssertionError, KeyError):
        return True
    raise AssertionError('reader accepted malformed schema/identity')
nonlegacy_missing = copy.deepcopy(legacy_offserver); nonlegacy_missing['extra_unpinned_field'] = True
assert refuses(legacy_server, nonlegacy_missing)
current_missing = copy.deepcopy(read(FINAL/'OFFSERVER_ACCEPTANCE.json')); current_missing.pop('different_host_observed')
assert refuses(read(FINAL/'PARTIAL_ACCEPTANCE.json'), current_missing)
bad_source = copy.deepcopy(legacy_server); bad_source['after_source_data']['src/data_loader.py'] = '0'*64
assert refuses(bad_source, legacy_offserver)
bad_checkpoint = copy.deepcopy(legacy_server); bad_checkpoint['records'][0]['checkpoint_sha256'] = '0'*64
assert refuses(bad_checkpoint, legacy_offserver)
bad_length = copy.deepcopy(legacy_server); bad_length['records'].pop()
assert refuses(bad_length, legacy_offserver)

# Independent arithmetic from original accepted metrics; no original score/summary import.
def frozen_score(metrics):
    gap = max(metrics['aeod'], metrics['aspd'])
    return metrics['accuracy'] - .35 * (.45 * metrics['aeod'] + .45 * metrics['aspd'] + .10 * gap) - .10 * max(0, gap - .06)
expected = {r['id']: r for r in manifest['jobs']}
assert len(expected) == 32 and set(expected) == {r['id'] for r in records}
output_records = {r['id']: r for r in summary['records']}
assert len(output_records) == len(summary['records']) == 32 and set(output_records) == set(expected)
errors = []
condition_scores = {}
for row in records:
    job = expected[row['id']]
    assert (row['candidate'], row['distribution'], row['attack'], row['seed'], row['rounds']) == (job['tuning_candidate'], job['distribution'], job['attack'], 91001, 70)
    assert row['job_sha256'] == job['job_sha256'] and row['source_hashes'] == protocol['source_hashes']
    assert all(output_records[row['id']][k] == v for k, v in row.items())
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in row['metrics'].values())
    value = frozen_score(row['metrics']); condition_scores[row['id']] = value
    errors.append(abs(value-output_records[row['id']]['score']))
    for k in ('accuracy','aeod','aspd'):
        assert output_records[row['id']][k] == row['metrics'][k]
assert max(errors) == 0
candidates = []
conditions = {(d, a) for d in protocol['distributions'] for a in protocol['attacks']}
for candidate in sorted(c['id'] for c in protocol['candidates']):
    group = [r for r in records if r['candidate'] == candidate]
    assert len(group) == 4 and {(r['distribution'], r['attack']) for r in group} == conditions
    result = {'candidate': candidate, 'n_seeds': 1}
    for key in ('accuracy','aeod','aspd'):
        result[key] = statistics.mean(r['metrics'][key] for r in group)
    result['score'] = statistics.mean(condition_scores[r['id']] for r in group)
    candidates.append(result)
assert summary['candidates'] == candidates
selected = sorted(candidates, key=lambda r: (-r['score'], r['candidate']))[0]
champion = sorted(candidates, key=lambda r: (-r['accuracy'], r['candidate']))[0]
def dominates(a, b):
    directions = [a['accuracy'] >= b['accuracy'], a['aeod'] <= b['aeod'], a['aspd'] <= b['aspd']]
    strict = [a['accuracy'] > b['accuracy'], a['aeod'] < b['aeod'], a['aspd'] < b['aspd']]
    return all(directions) and any(strict)
pareto = [c for c in candidates if not any(dominates(other,c) for other in candidates)]
assert summary['selected_per_method'] == {manifest['method']: selected}
assert summary['accuracy_champion'] == champion and summary['three_metric_pareto'] == pareto
assert summary['selected_candidate_recipe'] == next(c for c in protocol['candidates'] if c['id'] == selected['candidate'])
assert summary['status'] == 'ROOT_ACCEPTED32_RECORD_ONLY_RECIPE_SUMMARY' and summary['accepted'] == 32
assert summary['seed_n'] == 1
for name in ('sample_SD_reported','significance_claimed','final_test','formal100_started','recipe_adopted'):
    assert summary[name] is False
assert summary['all_negative_results_retained'] is True
assert summary['source_bindings']['prior26_sources'] == index['sources']
for key, value in [('final_root_proof_sha256', sha(FINAL/'ROOT_ADOPTION_REVIEW.json')), ('final_archive_sha256', proof['archive_sha256']), ('final_strict_sha256', proof['strict_receipt_sha256']), ('final_offserver_sha256', proof['offserver_proof_sha256'])]:
    assert summary['source_bindings'][key] == value
source = (BASE/'summary32.py').read_text('utf8')
assert "selected=min(candidates,key=lambda row:(-row['score'],row['candidate']))" in source
assert "champion=min(candidates,key=lambda row:(-row['accuracy'],row['candidate']))" in source
ranking = sorted(candidates, key=lambda r: (-r['score'],r['candidate']))
near = {'first': ranking[0]['candidate'], 'second': ranking[1]['candidate'], 'score_gap_fraction': ranking[0]['score']-ranking[1]['score'], 'score_gap_times100': 100*(ranking[0]['score']-ranking[1]['score']), 'equal_at_score_times100_two_decimals': f"{100*ranking[0]['score']:.2f}" == f"{100*ranking[1]['score']:.2f}", 'exact_score_tie': ranking[0]['score'] == ranking[1]['score'], 'epsilon_used_for_selection': False}
assert near['exact_score_tie'] is False
review = {'status':'PASS', 'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'scope':'Independent record-only SUMMARY32 and exact-legacy2 reader audit; no strict acceptor, inference, training or protocol adoption.', 'summary_sha256':sha(SUMMARY), 'final_root_sha256':sha(FINAL/'ROOT_ADOPTION_REVIEW.json'), 'wrapper_sha256':sha(WRAPPER), 'prepared_members_verified':len(seal['files']), 'batches':batches, 'accepted_records':32, 'original26_records_exact':True, 'final6_ids_exact':True, 'unique_ids':32, 'candidates':8, 'conditions_per_candidate':4, 'checkpoint_identities_paired':32, 'source_data_pins_per_record':21, 'source_hash_entries_compared':32*21, 'original_record_fields_preserved':True, 'condition_scores_recomputed':32, 'candidate_mean_scalars_recomputed':32, 'maximum_abs_numeric_difference':0.0, 'selection_exact_tie_rule_verified':'(-score,candidate); ACC uses (-accuracy,candidate); no epsilon', 'actual_exact_score_ties':len(candidates)-len({c['score'] for c in candidates}), 'pareto_ordered_comparisons':64, 'pareto_candidates':len(pareto), 'frozen_score_leader':selected, 'accuracy_champion':champion, 'three_metric_pareto':pareto, 'near_tie':near, 'legacy_reader':{'exact_legacy2_positive_records':2,'current6_positive_records':6,'legacy_before_after_source_pins':21,'nonlegacy_missing_host_refused':True,'current_missing_host_refused':True,'legacy_source_mutation_refused':True,'legacy_checkpoint_mutation_refused':True,'legacy_length_mutation_refused':True,'original_summary_main_executed':False,'original_scientific_functions_modified':False,'missing_host_field_fabricated':False,'limitation':'Legacy proof lacks different_host_observed; this audit verifies its exact pinned accepted schema and six record fields, and does not retrospectively measure a host flag.'}, 'first_schema_failure_preserved':{'path':failure_path.relative_to(ROOT).as_posix(),'sha256':sha(failure_path)}, 'seed_n':1, 'sample_SD_reported':False, 'significance_claimed':False, 'recipe_adopted':False, 'formal100_started':False, 'final_test':False, 'new_inference':0, 'all_candidates_retained':True, 'limits':protocol['limits']+['Valid-only exposed recipe search; four conditions are not four seeds. Score is a frozen exploratory selection rule, not paper performance or a chosen author endpoint. This review binds accepted archive hashes and existing ROOT/offserver proofs without rerunning strict/tensor acceptance.'], 'input_pins':pins}
assert not (OUT/'ROOT_INDEPENDENT_REVIEW.json').exists()
(OUT/'ROOT_INDEPENDENT_REVIEW.json').write_text(json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
lines = ['# FLGMM 32-record exploratory recipe screen — independent verification', '', 'PASS: 32 unique original strict records, eight candidates, and four conditions per candidate (IID/non-IID × Benign/S-DFA), seed 91001, 70 rounds. All eight candidates and negative results are retained. Values below are arithmetic means across conditions; n=1 seed, no sample SD or significance.', '', '| Tg | L | LR | ACC (%) ↑ | AEOD (%) ↓ | ASPD (%) ↓ | Frozen score | Scope |', '|---:|---:|---:|---:|---:|---:|---:|:---|']
for row in candidates:
    recipe = next(c for c in protocol['candidates'] if c['id'] == row['candidate'])
    tags = []
    if row['candidate'] == selected['candidate']: tags.append('score leader')
    if row['candidate'] == champion['candidate']: tags.append('ACC champion')
    if row in pareto: tags.append('Pareto')
    lines.append(f"| {recipe['adapter']['warmup_rounds']} | {recipe['adapter']['control_width']:.1f} | {recipe['learning_rate']:g} | {100*row['accuracy']:.4f} | {100*row['aeod']:.4f} | {100*row['aspd']:.4f} | {row['score']:.8f} | {', '.join(tags) or 'dominated'} |")
lines += ['', f"Frozen-score leader: `{selected['candidate']}`; ACC champion: `{champion['candidate']}`. The three-metric Pareto set contains {len(pareto)} candidates. Exact score ties: 0.", '', f"The score leader exceeds `{ranking[1]['candidate']}` by {near['score_gap_fraction']:.12g} score units ({near['score_gap_times100']:.12g} after multiplying score by 100). Both round to {100*ranking[0]['score']:.2f} on that latter scale at two decimals; this is a rounded near-tie, not an exact tie. Selection used unrounded scores and no epsilon; exact ties would use candidate lexical order.", '', 'The frozen score is calculated separately for each condition, then averaged over the four conditions: `ACC - .35*(.45*AEOD + .45*ASPD + .10*max(AEOD,ASPD)) - .10*max(0,max(AEOD,ASPD)-.06)`. Metrics in this expression are fractions. A score calculated from averaged metrics is not substituted.', '', 'This exposed valid-only search does not establish multi-seed paper performance, select an author endpoint, or adopt a formal-100 protocol/recipe. Author-code clustering differs from the paper introduction; upstream bounds behavior and the declared zero-standard-deviation extension remain disclosed in the frozen protocol. No new training, inference or final-test evaluation occurred in this audit.', '', 'The first SUMMARY attempt failed on a missing host-observation field in the exact legacy-two proof. Its failure record remains unchanged. The separate reader passed the exact two-record/21-source check and rejected nonlegacy missing-host proofs and source/checkpoint/length mutations. No absent host flag was fabricated.', '', f"SUMMARY32 SHA256: `{sha(SUMMARY)}`.", f"Final6 ROOT SHA256: `{sha(FINAL/'ROOT_ADOPTION_REVIEW.json')}`.", f"Reader SHA256: `{sha(WRAPPER)}`.", '', 'All 32 condition scores and 32 candidate mean scalars match exactly. Original record identity, metrics, source/data pins and checkpoint hashes are preserved. The independent audit reads existing ROOT/offserver/archive hashes; it does not rerun strict acceptance or inspect tensors anew.', '']
(OUT/'EXPLORATORY_TABLE.md').write_text('\n'.join(lines),encoding='utf8',newline='\n')
files = {p.name:{'sha256':sha(p),'size':p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'FILES_SHA256.json'}
(OUT/'FILES_SHA256.json').write_text(json.dumps({'status':'SEALED_INDEPENDENT_REVIEW_ONLY','files':files},indent=2)+'\n',encoding='utf8',newline='\n')
print(json.dumps({'status':'PASS','records':32,'candidates':8,'condition_scores':32,'mean_scalars':32,'max_diff':0.0,'pareto':len(pareto),'near_tie':near,'review_sha256':sha(OUT/'ROOT_INDEPENDENT_REVIEW.json'),'table_sha256':sha(OUT/'EXPLORATORY_TABLE.md'),'seal_sha256':sha(OUT/'FILES_SHA256.json')},ensure_ascii=False))
