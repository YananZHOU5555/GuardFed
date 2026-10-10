"""Bounded C70 source review; metadata fixtures never establish adoption."""
import ast, copy, datetime, hashlib, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent; R = H.parents[1]
B = R/'tmp/celeba_mechanism_C70_table_preparation_20261010'
O = R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_six_scenes_20261010'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())

def need(ok, message):
    if not ok: raise ValueError(message)

def function(path, name):
    text = Path(path).read_text('utf8')
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(text, node)

def raw_records(path):
    text = Path(path).read_text('utf8'); pos = text.index('[', text.index('"records"')) + 1
    decoder = json.JSONDecoder(); out = []
    while True:
        while text[pos].isspace() or text[pos] == ',': pos += 1
        if text[pos] == ']': return out
        value, end = decoder.raw_decode(text, pos); out.append((value['id'], text[pos:end])); pos = end

def refuse(label, fn):
    try: fn()
    except (ValueError, AssertionError, TypeError, KeyError): return label
    raise AssertionError('Boundary accepted: ' + label)

def main():
    assert not sys.flags.optimize
    assert sha(B/'HANDOFF.json') == 'cb97fd1e629a78e551cef6b4ee4c3b70421a546b5d621e8646487fbaf59497bb'
    assert sha(B/'FILES_SHA256.json') == '758d49ac894a423f151f64cb0619f630bb6fdb697d9e8d16949b163ba09f336e'
    rows = read(B/'FILES_SHA256.json')['members']; assert len(rows) == 12
    for row in rows:
        p = B/row['path']; assert p.resolve().is_relative_to(B.resolve())
        assert sha(p) == row['sha256'] and p.stat().st_size == row['size']
        if p.suffix == '.py': ast.parse(p.read_text('utf8'))
    inputs = read(B/'INPUTS.json'); assert len(inputs['files']) == 67
    for name, pin in inputs['files'].items():
        p = R/name; assert p.resolve().is_relative_to(R.resolve())
        assert sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes'], name
    assert sha(O/'ROOT_VERIFICATION.json') == inputs['old_C60_root_sha256'] == 'f2465c5e81291deca92535adcd0b0a29c49b3f76c2330927aa3db50925a9c742'
    assert sha(R/'tmp/celeba_mechanism_C60_root_arithmetic_review_20261010/ROOT_ARITHMETIC_REVIEW.json') == '7273c2f3340a1c62c3bb9874117b5bff0bbe447751e0621d2c737de8ff9189ec'
    reuse = read(B/'SOURCE_REUSE.json')
    for name, key in [('build.py','original_C60_builder_sha256'),('panels.py','original_C60_panels_sha256'),('verify_numeric.py','original_C60_numeric_sha256')]:
        assert sha(O/name) == reuse[key]
    actual = function(B/'verify_numeric.py','verify')
    for before, after in reversed(reuse['numeric_guard_changes']): actual = actual.replace(after,before)
    assert actual == function(O/'verify_numeric.py','verify')
    assert function(B/'verify_numeric.py','verify_aggregate') == function(O/'verify_numeric.py','verify_aggregate')
    assert function(B/'panels.py','aggregate_panels') == function(O/'panels.py','aggregate_panels')
    panel_text = (B/'panels.py').read_text('utf8').replace(",('non-IID','F Flip')",'').replace('Only five IID plus non-IID Benign/F Flip10 scenes are publishable','Only exact five IID scenes plus non-IID Benign10 are publishable')
    assert panel_text == (O/'panels.py').read_text('utf8')
    stage = R/'tmp/celeba_mechanism_valid_C_after60_20261010'
    prior_path = R/'tmp/celeba_mechanism_valid_C_after56_20261010/inventory_actual160_Full100refs.json'
    current_path = stage/'inventory_actual170_Full100refs.json'
    prior, current = read(prior_path), read(current_path)
    expected = [f'minus_C_non-IID_F Flip_seed{s}' for s in range(91001,91011)]
    scenes = [('IID',a) for a in ['Benign','F Flip','FedSA','S-DFA','Sp-DFA']]+[('non-IID','Benign'),('non-IID','F Flip')]
    assert [x for x in raw_records(current_path) if x[0] not in expected] == raw_records(prior_path)
    assert current['full_references'] == prior['full_references'] and len(current['full_references']) == 100
    # Load only three metadata guards; do not import the builder or scientific libraries.
    namespace = dict(need=need, R=R, H=B, C70=stage, EXPECTED=expected, SCENES=scenes, sha=sha, read=read)
    for name in ['scope','adoption_metadata','verify_future_binding']:
        exec(compile(function(B/'build.py',name),str(B/'build.py')+':'+name,'exec'),namespace)
    controls = namespace['scope'](prior,current); assert len(controls) == 70
    template = read(B/'C10_BINDING_TEMPLATE.json')
    assert template['adoption'] is None and template['adoption_sha256'] is None
    refusals = [refuse('null future adoption',lambda: namespace['verify_future_binding'](template))]
    fixture = dict(status='ROOT_C_AFTER60_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS',prior_three_view_models=160,accepted_new=10,cumulative_three_view_models=170,accepted_new_ids=expected,original160_unchanged=True,source_scope_complete=True,negative_results_preserved=True,science_seal_sha256=template['science_sha256'],execution_seal_sha256=template['execution_sha256'],prior160_root_adoption_sha256=inputs['prior160_adoption_sha256'],all_native_differences_zero=True,server_strict_bound_in_saved_receipts=True,new_training=0,new_Full_inference=0,test_inference=False)
    call = lambda proof: namespace['adoption_metadata'](proof,template['science_sha256'],template['execution_sha256'],inputs['prior160_adoption_sha256'])
    call(fixture)
    for key, value in [('execution_seal_sha256','0'*64),('cumulative_three_view_models',169),('server_strict_bound_in_saved_receipts',False),('source_scope_complete',False)]:
        bad = copy.deepcopy(fixture); bad[key] = value
        refusals.append(refuse(key,lambda bad=bad: call(bad)))
    for label, mutate in [('Full reference drift',lambda d:d['full_references'][0].update(checkpoint_sha256='0'*64)),('terminal round69',lambda d:d['records'][-1].update(terminal_round=69)),('nonIID alpha5000',lambda d:d['records'][-1].update(actual_alpha=5000)),('duplicate selected seed',lambda d:d.update(selected_replay_ids=[expected[0]]*10))]:
        bad = copy.deepcopy(current); mutate(bad)
        refusals.append(refuse(label,lambda bad=bad: namespace['scope'](prior,bad)))
    old = read(O/'snapshot/records.json')['records']; assert len(old) == len({r['id'] for r in old}) == 120
    text = (B/'build.py').read_text('utf8')
    for fragment in ['join(adoption.parent',"records=old+full+added",'record_spans(raw)[:120]',"'Old972 statistics changed'","'Old486 display cells/order changed'","(OLD/'snapshot/cross_scene_seed_first.json').read_bytes()",'original.summarize']:
        assert fragment in text + (B/'panels.py').read_text('utf8'), fragment
    assert 'other image controls remain incomplete' in text and 'prior valid/officialtest exposure' in text
    assert 'actual_C10_binding' in text and 'new_inference=0' in text
    out = dict(status='INDEPENDENT_C70_SOURCE_REVIEW_PASS_WAIT_ACTUAL_REPLAY_ADOPTION_AND_TABLE',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_handoff_sha256=sha(B/'HANDOFF.json'),source_seal_sha256=sha(B/'FILES_SHA256.json'),source_members_verified=12,input_pins_verified=67,old_C60_root_sha256=sha(O/'ROOT_VERIFICATION.json'),old120_source_records_available=True,old160_inventory_record_JSON_bytes_and_order_exact=True,Full100_references_exact=True,numeric_original_body_exact_after_scope_guard_normalization=True,verify_aggregate_byte_exact=True,panels_only_scene_guard_and_message_changed=True,old120_972scalar_486cell_162IIDaggregate_preservation_guards_present=True,actual_native_inventory_scope=170,metadata_scope_positive_exact10=expected,additional_metadata_refusals=refusals,synthetic_schema_fixture_is_not_actual_adoption=True,future_adoption_template_is_null=True,actual_new_three_view_accepted_by_this_review=0,actual_C70_statistics_recomputed_by_this_review=0,source_ready_for_root_build_only_after_actual_C10_adoption=True,reviewer_authored_C10_evaluator_source=True,reviewer_authored_C70_table_builder=False,reviewer_authored_C70_numeric_verifier=False,source_review_is_not_replay_adoption_or_table_acceptance=True,source_review_code_sha256=sha(Path(__file__)),SSH=False,CNN=0,test=False,STATE_changed=False,Git_changed=False)
    with (H/'SOURCE_REVIEW.json').open('x',encoding='utf8',newline='\n') as stream:
        json.dump(out,stream,ensure_ascii=False,indent=2); stream.write('\n')
    print(json.dumps(dict(status=out['status'],sha256=sha(H/'SOURCE_REVIEW.json'),extra_refusals=len(refusals))))

if __name__ == '__main__': main()
