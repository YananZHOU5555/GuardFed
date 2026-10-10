"""Independent source/metadata check only. No table build or new arithmetic."""
import ast, copy, hashlib, importlib.util, json, re, subprocess, sys
from pathlib import Path
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
R = H.parents[1]
S = R / 'tmp/celeba_mechanism_C_six_scenes_prepared_20261010'
O = R / 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_five_scenes_20261010'
EXPECTED_SEAL = '4a1a2f9d71bd432e58229f98cf7f98b1fdba9e8b60dad4c9338c8801b240067c'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_bytes())
def save(name, value):
    (H/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def fn(source, name):
    return ast.get_source_segment(source, next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name))
def spans(raw):
    pos = re.search(r'"records"\s*:\s*\[', raw).end()
    dec = json.JSONDecoder(); result = []
    while True:
        while raw[pos].isspace() or raw[pos] == ',': pos += 1
        if raw[pos] == ']': return result
        _, end = dec.raw_decode(raw, pos); result.append(raw[pos:end]); pos = end

assert sha(S/'FILES_SHA256.json') == EXPECTED_SEAL
seal = read(S/'FILES_SHA256.json')
assert len(seal['members']) == 13
for row in seal['members']:
    p = S/row['path']; assert p.stat().st_size == row['size'] and sha(p) == row['sha256'], row['path']
inputs = read(S/'INPUTS.json')
assert len(inputs['files']) == 42
for name, pin in inputs['files'].items():
    p = R/name; assert p.stat().st_size == pin['bytes'] and sha(p) == pin['sha256'], name
assert all(v is None for k,v in inputs.items() if k.startswith('future_C4_'))
assert all(read(S/'C4_BINDING_TEMPLATE.json')[k] is None for k in ['science_sha256','execution_sha256','inventory_sha256','adoption','adoption_sha256'])
assert not (S/'snapshot').exists()

spec = importlib.util.spec_from_file_location('reviewed_prepared_C60', S/'build.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
prior = read(b.C6/'inventory_actual156_Full100refs.json')
future = copy.deepcopy(prior)
future['selected_replay_ids'] = list(b.EXPECTED)
future['excluded_prior_replay_ids'] = [r['id'] for r in prior['records']]
for rid in b.EXPECTED:
    row = copy.deepcopy(next(r for r in prior['records'] if r['id']==b.TEN[0]))
    row.update(id=rid, seed=int(rid[-5:])); future['records'].append(row)
assert len(b.scope(prior, future)) == 60
proof = dict(status='ROOT_C_AFTER56_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS', prior_three_view_models=156, accepted_new=4, cumulative_three_view_models=160, accepted_new_ids=list(b.EXPECTED), original156_unchanged=True, source_scope_complete=True, negative_results_preserved=True, science_seal_sha256='fixture_science', execution_seal_sha256='fixture_execution', prior156_root_adoption_sha256=inputs['prior_C6_adoption_sha256'], all_native_differences_zero=True, server_strict_bound_in_saved_receipts=True, new_training=0, new_Full_inference=0, test_inference=False)
def proof_gate(p): b.adoption_metadata(p, 'fixture_science', 'fixture_execution', inputs['prior_C6_adoption_sha256'])
proof_gate(proof)
refused=[]
def reject(label, call):
    try: call()
    except (ValueError, KeyError, TypeError, FileNotFoundError) as e:
        refused.append(dict(case=label, exception=type(e).__name__, message=str(e))); return
    raise AssertionError('Unexpected acceptance: '+label)
x=copy.deepcopy(future);x['selected_replay_ids'].reverse();reject('selected_exact4_order_drift',lambda:b.scope(prior,x))
x=copy.deepcopy(future);x['full_references'].reverse();reject('Full100_reference_order_drift',lambda:b.scope(prior,x))
x=copy.deepcopy(future);x['records'][-1]['config']['ablation_component']='U';reject('wrong_C_config',lambda:b.scope(prior,x))
for k,v in [('prior_three_view_models',155),('cumulative_three_view_models',159),('new_training',1)]:
    x=copy.deepcopy(proof);x[k]=v;reject('adoption_'+k,lambda x=x:proof_gate(x))
binding=read(S/'C4_BINDING_TEMPLATE.json');binding['adoption']='tmp/celeba_mechanism_valid_C_after56_20261010/execution_candidate/backups/incremental_FIXTURE_NO_FILE/ROOT_ADOPTION_REVIEW.json'
x=copy.deepcopy(binding);x['stage']='tmp/other_future_stage';reject('wrong_future_namespace',lambda:b.verify_future_binding(x))
reject('future_science_SHA_null',lambda:b.verify_future_binding(binding))
assert len(refused)==8
save('INDEPENDENT_METADATA_FIXTURES.json',dict(status='PASS_SYNTHETIC_METADATA_ONLY', positive_scope_and_adoption_fixtures=2, refusals=refused, actual_future_artifacts_created=False, actual_future_adoption_read=False, statistics_computed=0))

run=subprocess.run([sys.executable,'-B',str(S/'check_prepared.py')],capture_output=True)
(H/'ORIGINAL_28_REFUSALS.stdout.txt').write_bytes(run.stdout)
(H/'ORIGINAL_28_REFUSALS.stderr.txt').write_bytes(run.stderr)
assert run.returncode==0, run.stderr.decode('utf-8','replace')
original_checks=json.loads(run.stdout)
assert original_checks['refusal_count']==28 and original_checks['input_pins']==42 and original_checks['new_statistics']==0 and original_checks['CNN']==0

oldraw=(O/'snapshot/records.json').read_text('utf-8'); records=read(O/'snapshot/records.json')['records']
serialized=json.dumps(dict(records=records),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
assert len(records)==len(spans(oldraw))==100 and spans(oldraw)==spans(serialized)
oldpanel=(O/'panels.py').read_text('utf-8');newpanel=(S/'panels.py').read_text('utf-8')
assert newpanel==oldpanel.replace("('IID','Sp-DFA')]","('IID','Sp-DFA'),('non-IID','Benign')]").replace('Only exact C IID Benign10, F Flip10, FedSA10 S-DFA10 and Sp-DFA10 scenes are publishable','Only exact five IID scenes plus non-IID Benign10 are publishable')
assert fn(oldpanel,'aggregate_panels')==fn(newpanel,'aggregate_panels')
oldnum=(O/'verify_numeric.py').read_text('utf-8');newnum=(S/'verify_numeric.py').read_text('utf-8')
arith=lambda s:s.split('    errors=[]',1)[1].split('    assert len(errors)',1)[0]
counts=lambda s:s.split('    metric_checks=0;count_checks=0',1)[1].split('    assert metric_checks',1)[0]
assert arith(oldnum).replace('len(rows)==15','len(rows)==18')==arith(newnum)
assert counts(oldnum)==counts(newnum)
assert fn(oldnum,'verify_aggregate')==fn(newnum,'verify_aggregate')
build=(S/'build.py').read_text('utf-8');tree=ast.parse(build)
assert "aggregate=(OLD/'snapshot/cross_scene_seed_first.json').read_bytes()" in build
assert "(args.output/'cross_scene_seed_first.json').write_bytes(aggregate)" in build
assert 'pure.aggregate_panels' not in build
assert "need(filtered==oldpanels,'Old810 statistics changed')" in build
assert "need(oldlines==newlines and len(oldlines)*3==405,'Old405 cells/order changed')" in build
assert 'original.summarize(flat)' in newpanel and 'original.statistic(selected,len(seeds))' in newpanel
assert "accepted.source_functions(basis)" in build and "scientific['receipt_identity'],scientific['normalized'],scientific['canonical']" in build
assert "full_builder.full_record" in build and "<=1e-12" in build
assert not any('aggregate_panels' in ast.unparse(n) for n in ast.walk(tree) if isinstance(n,ast.Call))
for row in seal['members']:
    p=S/row['path'];assert p.stat().st_size==row['size'] and sha(p)==row['sha256']
assert 'torch' not in sys.modules and 'numpy' not in sys.modules and not (S/'snapshot').exists()
save('CHECK_RESULTS.json',dict(status='SOURCE_AND_METADATA_PASS_NO_C60_RESULTS', sealed_source_members=13, input_pins_verified=42, old_C50_raw_record_spans_and_order_exact=100, original_refusals_reproduced=28, independent_positive_metadata_fixtures=2, additional_independent_refusals=8, arithmetic_body_unchanged=True, confusion_count_body_exact=True, seed_panels_10_9_6_unchanged=True, strict_join_and_original_statistic_reused_unchanged=True, old_IID_aggregate_source_function_exact=True, old162_IID_aggregate_copy_only=True, old_IID_aggregate_sha256=sha(O/'snapshot/cross_scene_seed_first.json'), old810_and405_preservation_runtime_gates_present=True, external_actual_C4_SHA_and_adoption_required=True, actual_future_adoption_read=False, new_statistics=0, CNN=0, source_seal_unchanged_after_review=True))
print(json.dumps(read(H/'CHECK_RESULTS.json'),ensure_ascii=False))
