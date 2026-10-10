"""Bounded read-only source/identity review. Metadata fixtures only; no fit/arrays."""
from pathlib import Path
import ast, copy, hashlib, importlib.util, json, math, sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
TARGET=ROOT/'tmp/celeba_logofair_fullcoverage_20261010'
SCREEN=ROOT/'tmp/celeba_logofair_screen32_20261010'
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
seal=TARGET/'FILES_SHA256.json'
assert digest(seal)=='89f3e4b4fd02352fa3c0f5080ce00ab6854eaabb4e3803a2635a1d3ba51196c3'
members=read(seal)['files'];assert len(members)==14
assert all(digest(TARGET/n)==v for n,v in members.items())
for p in TARGET.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
for p,wanted in read(TARGET/'INPUT_PINS.json').items():assert digest(p)==wanted
assert digest(SCREEN/'FILES_SHA256.json')=='accd5cb8582a344f870188f1e70661b6dc9dc948cc88c9e6f6651e451607bc49'
assert all(digest(SCREEN/n)==v for n,v in read(SCREEN/'FILES_SHA256.json')['files'].items())

spec=importlib.util.spec_from_file_location('reviewed_logofair100_metadata',TARGET/'metadata.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.verify_sources()
old=(SCREEN/'snapshot/logofair_bridge_20261010/bridge.py').read_text(encoding='utf8');new=m.bridge_source()
assert new.replace('job["seed"] not in protocol["seeds"]','job["seed"] != 91001')==old
func=lambda s:{n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
a,b=func(old),func(new);exact=[n for n in a if a[n]==b[n]]
assert len(exact)==15 and set(a)-set(exact)=={'validate_job'}
namespace={'LOCAL_FILES':{'bridge.py','prepare_reuse.py','protocol.json','reuse_manifest.json'}}
node=next(n for n in ast.parse(new).body if isinstance(n,ast.FunctionDef) and n.name=='validate_job')
exec(compile(ast.Module(body=[node],type_ignores=[]),'identity_fixture','exec'),namespace)
manifest=read(SCREEN/'jobs/manifest.json')['jobs']
job=read(SCREEN/'jobs'/manifest[0]['job']);protocol=read(SCREEN/'snapshot/logofair_bridge_20261010/protocol.json');protocol['seeds']=list(range(91001,91011))
for seed in protocol['seeds']:namespace['validate_job'](dict(job,seed=seed),protocol)
refusals=[]
for label,bad in [('seed91011',dict(job,seed=91011)),('test',dict(job,evaluation_split='test')),('changed_fit_seed',dict(job,fit_seed=91001))]:
    try:namespace['validate_job'](bad,protocol)
    except ValueError:refusals.append(label)
    else:raise AssertionError(label)

refs=read(TARGET/'CACHE_IDENTITIES100.json')['references']
source={e['id']:e for e in read(ROOT/'tmp/celeba_baselines/logofair_bridge_20261009/reuse_manifest.json')['entries']}
old900={r['id']:r for r in read(ROOT/'outputs/guardfed_tables/celeba_nine_method_three_view_20261009/records_three_views_900.json')['records']}
grid={(d,a,s) for d in ('IID','non-IID') for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA') for s in range(91001,91011)}
assert len(refs)==len({r['id'] for r in refs})==100 and {(r['distribution'],r['attack'],r['seed']) for r in refs}==grid
for r in refs:
    original=old900[r['id']];inv=original['original_inventory_record'];entry=source[r['id']]
    assert (r['distribution'],r['attack'],r['seed'],r['alpha'])==(original['distribution'],original['attack'],original['seed'],inv['actual_alpha'])
    assert r['checkpoint']==inv['checkpoint'] and r['result']==inv['result'] and r['source_job']==inv['raw_job']
    assert entry['source_job']['config']==inv['config'] and entry['source_job']['source_hashes']==inv['source_hashes']
    assert r['cache']['sha256']==entry['accepted_record']['cache_sha256']
    assert r['checkpoint']['sha256']==entry['accepted_record']['checkpoint_sha256']==original['checkpoint_sha256']
    assert r['root_image_ids_sha256']==original['root_reconstruction']['root_image_ids_sha256']
    assert r['valid_image_ids_sha256']==inv['data_contract']['evaluation_image_ids_sha256']
    assert r['accepted_ID_arrays']['archive']==original['source_binding']['archive']
    assert r['accepted_ID_arrays']['member']==original['source_binding']['array_member']
    assert r['accepted_ID_arrays']['sha256']==original['prediction_arrays_sha256']
    assert r['accepted_ID_arrays']['receipt_sha256']==original['receipt_sha256']
    assert inv['terminal_round']==70 and inv['original_split']=='valid' and inv['original_n_eval']==19867
reused=[r for r in refs if r['seed']==91001 and r['attack'] in ('Benign','S-DFA')]
assert len(reused)==4 and len(refs)-len(reused)==96
roots={s:{r['root_image_ids_sha256'] for r in refs if r['seed']==s} for s in range(91001,91011)}
assert all(len(v)==1 for v in roots.values()) and len({next(iter(v)) for v in roots.values()})==10
assert len({r['valid_image_ids_sha256'] for r in refs})==1
assert all(r['current_mapping_sha256'] is None for r in refs if r['seed']!=91001)
template=read(TARGET/'BINDING_TEMPLATE.json');handoff=read(TARGET/'HANDOFF.json')
assert all(v is None for k,v in template.items() if k not in ('status','execution_authorized')) and template['execution_authorized'] is False
assert all(handoff[k] is None for k in ('selected_recipe','summary32_sha256','independent32_acceptance_sha256','root_execution_approval','new_mapping_hashes_seeds91002_to91010'))

# Two synthetic summaries: exact tie and heterogeneous per-condition penalties.
reuse32={r['id']:r for r in read(SCREEN/'snapshot/logofair_bridge_20261010/reuse_manifest.json')['entries']}
fixtures=[]
for heterogeneous in (False,True):
    rows=[]
    for i,e in enumerate(manifest):
        j=read(SCREEN/'jobs'/e['job']);src=reuse32[j['baseline_id']]['source_job'];c=i//4;t=i%4
        metrics=dict(accuracy=.5,aeod=.1,aspd=.1) if not heterogeneous else dict(accuracy=.6+c*.001,aeod=.01*(t+1),aspd=.03*(4-t))
        rows.append(dict(id=j['id'],candidate=j['candidate'],distribution=src['distribution'],attack=src['attack'],seed=91001,fit_seed=1719,metrics=metrics,checkpoint_sha256='0'*64))
    scores={}
    for candidate in sorted({r['candidate'] for r in rows}):
        values=[]
        for r in rows:
            if r['candidate']!=candidate:continue
            x=r['metrics'];gap=max(x['aeod'],x['aspd'])
            values.append(x['accuracy']-.35*(.45*x['aeod']+.45*x['aspd']+.10*gap)-.10*max(0,gap-.06))
        scores[candidate]=math.fsum(values)/4
    expected=min(scores,key=lambda c:(-scores[c],c))
    summary=m.summarize(rows);summary['strict_index_sha256']='1'*64
    adoption=dict(status='ROOT_LOGOFAIR_SCREEN32_ADOPTED',accepted_count=32,summary_sha256='0'*64,strict_index_sha256='1'*64,source_seal_sha256=m.SCREEN_SEAL,test_evaluated=False,independent_acceptance_sha256='2'*64)
    assert m.selection(summary,adoption,'0'*64,'1'*64)['id']==expected
    for label,ss,aa in [('partial31',dict(summary,records=rows[:-1]),adoption),('no_root_adoption',summary,dict(adoption,status='PREPARED_NOT_APPROVED')),('changed_summary_pin',summary,dict(adoption,summary_sha256='3'*64))]:
        try:m.selection(ss,aa,'0'*64,'1'*64)
        except ValueError:refusals.append(label)
        else:raise AssertionError(label)
    fixtures.append(dict(synthetic_only=True,heterogeneous=heterogeneous,independent_score_and_tie_match=True))
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
assert all(digest(TARGET/n)==v for n,v in members.items()) and digest(seal)=='89f3e4b4fd02352fa3c0f5080ce00ab6854eaabb4e3803a2635a1d3ba51196c3'
report=dict(status='PASS_SOURCE_ONLY_CONDITIONAL_ON_ACTUAL_COMPLETE32_BINDING_AND_ROOT_RESOURCE_APPROVAL',
    source_adoptable=True,actual_dispatch_authorized_by_this_review=False,source_seal_sha256=digest(seal),sealed_members=14,
    input_pins_verified=len(read(TARGET/'INPUT_PINS.json')),original32_seal_members_verified=57,
    science_functions_byte_exact=exact,unique_original_grid100=True,new_fits96=True,explicit_reused4=True,
    accepted900_identity_joins=100,root_ID_populations=10,valid_ID_orders=1,new_mapping_SHA_not_fabricated=True,
    future_recipe_summary_approval_null=True,metadata_fixture_checks=fixtures,refusals=refusals,
    no_Torch_NumPy_fit_CNN_or_arrays_loaded=True,no_SSH_Git_STATE_or_source_changes=True,findings=[],
    resource_and_failstop_source_review=dict(fresh_child_one_at_a_time=True,CUDA_hidden=True,thread_environment_one=True,
        original_bridge_torch_threads_one=True,F_guard_inherited_from_pinned32_stage_inputs=True,
        volume_label_health_capacity_and_no_internal_fallback=True,source_and_job_SHA_before_each_fit=True,
        original_bridge_checked_output_called_before_child_success=True,exclusive_started_marker=True,
        partial_attempt_preserved_no_retry=True,offserver_and_root_counts_zero_until_independent_adoption=True),
    execution_dependencies=['Actual complete32 original strict index, original summary and independent/root adoption SHA; no partial selection',
        'Exact existing100 checkpoint/result/cache bytes staged on guarded F; no new CNN cache',
        'Nine actual ID-only mappings prepared with original fixed rule, all10 mapping metadata/root approval; seed91001 bytes unchanged',
        'Root-reviewed bound96+4 stage and explicit execution approval, actual host/resource/runtime check'],
    boundaries=['No100 fit, population arrays, fitted state, new-seed support, Beta convergence or performance checked here.',
        'Reviewed root approvals are external authority inputs; the adapter checks their bytes and declared identities, and does not manufacture independent32/offserver evidence.',
        'Resource code limits workers/threads/CUDA; actual CPU placement, priority and host availability remain root launch checks.',
        'Constant-negative and all unfavorable results retained; virtual20 cohorts are not true client fairness. Fixed fit_seed1719 and validation selection/test exposure limitations retained.',
        'Reviewer prepared original32 packaging/runner, did not author this100 package; bridge/identity/mathematical checks above independently read actual files.'])
with (HERE/'REVIEW.json').open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(status=report['status'],report_sha256=digest(HERE/'REVIEW.json'),members=14,source_seal_sha256=digest(seal))))
