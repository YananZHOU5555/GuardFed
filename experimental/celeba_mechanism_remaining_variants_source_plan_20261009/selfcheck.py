"""Local source/config checks only: no scientific module imports or inference."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE = ROOT / 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1'
BRIDGE = ROOT / 'tmp/celeba_mechanism_valid_incremental_after71_20261009/bridge.py'

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(name, value):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    target = HERE / name
    if target.exists():
        assert target.read_text(encoding='utf-8') == data, 'Refuse changed existing output: ' + name
    else:
        target.write_text(data, encoding='utf-8')

pins = read(HERE / 'INPUTS.json')['inputs']
for name, pin in pins.items():
    p = ROOT / name
    assert sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes'], name
original = BRIDGE.read_text(encoding='utf-8')
helper = (HERE / 'variant_metadata_draft.py').read_text(encoding='utf-8')
# Apply only the semantic boundary proposal in memory; exact old82/selected11
# cohort and approval constants intentionally remain, so this is NOT a dispatch.
old = "require(all(r['variant'] == 'minus_U' and r['distribution'] in DISTRIBUTIONS for r in records), 'Only actual minus_U cohort is covered; no future variant semantics')"
new = "require(all(r['variant'] in VARIANT_COMPONENTS and r['distribution'] in DISTRIBUTIONS for r in records), 'Unknown mechanism variant/distribution')"
assert original.count(old) == 1
candidate = original.replace(old, new)
needle = "        cfg, contract, control = r['config'], r['data_contract'], paired[cell(r)]"
assert candidate.count(needle) == 1
candidate = candidate.replace(needle, needle + "\n        validate_variant_metadata(r)")
oldmask = "cfg['ablation_component'] == r['variant'][-1]"
assert candidate.count(oldmask) == 1
candidate = candidate.replace(oldmask, "cfg['ablation_component'] == VARIANT_COMPONENTS[r['variant']]")
candidate = candidate.replace('def validate_inventory(inventory, baseline):', helper + '\n\ndef validate_inventory(inventory, baseline):', 1)
diff = ''.join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True), fromfile='sealed_after71/bridge.py', tofile='SOURCE_ONLY_DRAFT/new_bridge.py'))
diff_path = HERE / 'BRIDGE_SEMANTICS_DRAFT.diff'
if diff_path.exists():
    assert diff_path.read_text(encoding='utf-8') == diff
else:
    diff_path.write_text(diff, encoding='utf-8')
functions = lambda s: {n.name: n for n in ast.parse(s).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
before, after = functions(original), functions(candidate)
unchanged = [n for n in before if n != 'validate_inventory']
assert all(ast.dump(before[n], include_attributes=False) == ast.dump(after[n], include_attributes=False) for n in unchanged)
assert all(ast.get_source_segment(original,before[n]) == ast.get_source_segment(candidate,after[n]) for n in unchanged)
# Both bridge sources have stdlib imports only. Never call bind_runtime.
ns = {'__file__': str(BRIDGE)}
exec(compile(candidate, 'SOURCE_ONLY_DRAFT', 'exec'), ns)
baseline = read(ROOT / 'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json')
inventory = read(BRIDGE.with_name('inventory_actual82_Full100refs.json'))
assert len(ns['validate_inventory'](inventory, baseline)) == 82
manifest = read(STAGE / 'manifest.json')
assert sha(STAGE / 'manifest.json') == ns['MANIFEST_SHA']
full = {(r['distribution'], r['attack'], r['seed']): r for r in baseline['records'] if r['method'] == 'GuardFed-AD2+'}
components = ns['VARIANT_COMPONENTS']
summary = {v: {'ablation_component': c, 'planned_jobs_byte_verified': 0, 'actual_config_difference_fields_vs_paired_Full': set(), 'example_job': None,
               'expected_candidate_audit_entries_at70': 70 if v == 'fixed_balanced' else 700} for v,c in components.items()}
rejections = []
for entry in manifest['jobs']:
    path = STAGE / 'jobs' / Path(entry['job']).name
    assert sha(path) == entry['job_sha256']
    job = read(path); cfg = job['config']; v = job['variant']
    assert (job['id'], v, job['output']) == (entry['id'], entry['variant'], entry['output'])
    assert job['source_hashes'] == manifest['source_hashes'] and job['adapter_hashes'] == manifest['adapter_hashes']
    assert job['protocol_sha256'] == manifest['protocol_sha256'] and job['tuning_candidate'] == 'lr0.0005_drop0.005'
    record = dict(job, seed=cfg['seed'])
    ns['validate_variant_metadata'](record)
    control = full[job['distribution'], job['attack'], cfg['seed']]
    assert {k:v for k,v in cfg.items() if k not in ns['IGNORE_RECIPE']} == {k:v for k,v in control['config'].items() if k not in ns['IGNORE_RECIPE']}
    assert cfg['client_alpha'] == ns['DISTRIBUTIONS'][job['distribution']] and cfg['rounds'] == 70 and cfg['celeba_evaluation_split'] == 'valid'
    item = summary[v]; item['planned_jobs_byte_verified'] += 1
    item['actual_config_difference_fields_vs_paired_Full'].update(k for k in set(cfg)|set(control['config']) if cfg.get(k) != control['config'].get(k))
    if item['example_job'] is None:
        item['example_job'] = {'id': job['id'], 'path': str(path.relative_to(ROOT)).replace('\\','/'), 'sha256':sha(path)}
        for field, value in [('ablation_component','INVALID'),('experiment_suite','other'),('experiment_tag','other'),('full_round_diagnostics',False)]:
            bad=copy.deepcopy(record); bad['config'][field]=value
            try: ns['validate_variant_metadata'](bad)
            except ValueError: rejections.append(v+':'+field)
            else: raise AssertionError('Mutation accepted')
        for field,value in [('id','other'),('variant','Full')]:
            bad=copy.deepcopy(record);bad[field]=value
            try: ns['validate_variant_metadata'](bad)
            except ValueError: rejections.append(v+':'+field)
            else: raise AssertionError('Mutation accepted')
for item in summary.values():
    assert item['planned_jobs_byte_verified'] == 100
    item['actual_config_difference_fields_vs_paired_Full'] = sorted(item['actual_config_difference_fields_vs_paired_Full'])
# Existing acceptance boundary still refuses fabricated pending or changed scientific inputs.
for label, edit in [
    ('unknown_variant',lambda r:r.update(variant='no_candidate')),
    ('recipe_drift',lambda r:r['config'].update(learning_rate=.01)),
    ('checkpoint_mix',lambda r:r['checkpoint'].update(sha256='0'*64)),
    ('changed_data',lambda r:r['data_contract'].update(root_image_ids_sha256='0'*64)),
    ('future_unaccepted_id',lambda r:r.update(id='minus_C_IID_Benign_seed91001')),
]:
    bad=copy.deepcopy(inventory);edit(bad['records'][0])
    try: ns['validate_inventory'](bad,baseline)
    except (ValueError,KeyError): rejections.append('actual82:'+label)
    else: raise AssertionError('Boundary mutation accepted')
save('SOURCE_REUSE_PROOF.json', {'status':'SOURCE_ONLY_AST_AND_BYTES_PASS','original_bridge_sha256':sha(BRIDGE),'unchanged_top_level_functions':unchanged,'unchanged_top_level_function_count':len(unchanged),'only_changed_existing_function':'validate_inventory','added_metadata_helper':'validate_variant_metadata','bind_runtime_source_exact':True,'strict_acceptor_source_unchanged':True,'scientific_modules_imported':False})
save('VARIANT_CONFIG_MAPPING.json', {'status':'PREPARED_NOT_APPROVED','manifest_sha256':sha(STAGE/'manifest.json'),'variants':summary,'paired_Full_config_comparison_exclusions':sorted(ns['IGNORE_RECIPE']),'no_new_terminal_records_created':True})
save('CHECKS.json', {'status':'SOURCE_METADATA_CHECKS_PASS_NOT_IMAGE_GATE','planned_job_bytes_verified':800,'original_actual82_metadata_regression':True,'negative_checks_passed':len(rejections),'negative_cases':rejections,'inference_performed':False,'new_variant_replay_validated':False,'dispatch_authorized':False})
print(json.dumps({'status':'SOURCE_ONLY_PASS','jobs':800,'negative_checks':len(rejections),'unchanged_functions':len(unchanged)}))
