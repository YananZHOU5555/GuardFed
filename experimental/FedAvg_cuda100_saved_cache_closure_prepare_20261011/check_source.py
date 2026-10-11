"""Pure metadata fixtures; imports only stdlib verifier, no numpy/arrays/science."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json
H=Path(__file__).resolve().parent;R=H.parents[1];P=R/'tmp/old800_cuda3_saved_audit_prepare_20261011/mixed_producer_v2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
for name in ['audit_saved.py','transport.py']:
 inv=json.loads((H/(name+'.inverse.json')).read_bytes());s=(H/name).read_text('utf8');restored=s
 for z in reversed(inv['spans']):
  assert restored.count(z['after'])==1;restored=restored.replace(z['after'],z['before'],1)
 assert restored.encode()==(P/name).read_bytes();compile(s,name,'exec');checks.append(name+' full raw inverse exact')
orig=(P/'audit_saved.py').read_text();new=(H/'audit_saved.py').read_text()
def science_loop(s):
 return next(ast.get_source_segment(s,n) for n in ast.walk(ast.parse(s)) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='identity')
assert science_loop(orig)==science_loop(new);checks.append('entire original per-record saved audit loop raw exact')
t=ast.parse((H/'transport.py').read_text());embedded=next(n.value.value for n in t.body if isinstance(n,ast.Assign) and any(isinstance(k,ast.Name) and k.id=='code' for k in n.targets));compile(embedded,'embedded Linux transport','exec');checks.append('Linux transport payload compiles')
for p in H.glob('*.py'):compile(p.read_text('utf-8-sig'),str(p),'exec')
refs=json.loads((H/'EXACT100_REFS.json').read_bytes());logo=json.loads((R/'tmp/LoGo100_uniform_cuda_cache_adapter_prepare_20261011/EXACT100.json').read_bytes());by={r['id']:r for r in refs['records']}
assert len(by)==100 and all(by[r['baseline_id']]['canonical_sha256']==r['route']['identity_canonical_sha256'] for r in logo['records']);checks.append('100 original FedAvg identities equal LoGo requested identities; explicit order projection')
# Import only the new stdlib helper; no load_cache invocation.
spec=importlib.util.spec_from_file_location('_test_cache_verifier',H/'cache_verifier.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
D={};hashes={}
def reg(path,value,digest=None):
 D[path]=value;hashes[path]=digest or m.canonical(value);return dict(path=path,sha256=hashes[path])
def mocked_pin(pin):
 m.need(isinstance(pin,dict) and set(pin)=={'path','sha256'} and pin['path'] in D and hashes[pin['path']]==pin['sha256'],'Mock pinned identity changed');return pin['path']
m.pinned=mocked_pin;m.sha=lambda path:hashes[str(path)];m.read=lambda path:D[str(path)]
ids=refs['ordered_ids'];rrs=[];rootrows={};arows=[];members={};selectionrows=[];fake_refs=copy.deepcopy(refs)
cohortsha='a'*64
for i,identity in enumerate(ids):
 producer=refs['record_producers'][identity];rec=dict(id=identity,method='FedAvg',terminal_round=70,original_split='valid',original_n_eval=19867,checkpoint={'sha256':str(i)+'c'},result={'sha256':str(i)+'r'},raw_job={'sha256':str(i)+'j'})
 fake_refs['records'][i]['canonical_sha256']=m.canonical(rec);selectionrows.append(rec);comp={'accepted':True,'max_abs_difference':0};views={'native':{'accuracy':0.5}}
 result=reg('/result/'+identity,{},rec['result']['sha256']);arr=reg('/array/'+identity,{})
 receipt=reg('/receipt/'+identity,dict(id=identity,model_inventory_record_sha256=m.canonical(rec),status='NATIVE_VALID_REPLAY_PASS',scope='FORMAL_ORIGINAL800_CUDA_PER_RECORD_V1',runtime={'device':'cuda:0'},checkpoint_sha256=rec['checkpoint']['sha256'],original_result_sha256=rec['result']['sha256'],original_job_sha256=rec['raw_job']['sha256'],prediction_arrays_sha256=arr['sha256'],valid_n=19867,valid_image_ids_sha256=m.VALID_SHA,views=views,native_comparison=comp))
 binding=reg('/binding/'+identity,dict(id=identity,scope='UNIFORM_CUDA_ORIGINAL800_FORMAL_V1',receipt_sha256=receipt['sha256'],source_before={},source_after={},artifact_before={},artifact_after={}))
 saved=reg('/saved/'+identity,dict(id=identity,status='PER_RECORD_ORIGINAL800_CUDA_SAVED_WHOLE_ACCEPTED',binding_sha256=binding['sha256'],views=views,native_comparison=comp))
 pins=dict(original_result=result,receipt=receipt,predictions=arr,formal_binding=binding,saved_acceptance=saved)
 for role,name in [('receipt','receipt.json'),('predictions','validation_predictions.npz'),('formal_binding','formal_binding.json'),('saved_acceptance','formal_saved_acceptance.json')]:members[producer+'/runs/'+identity+'/'+name]={'sha256':pins[role]['sha256']}
 members['original_results/'+identity+'.json']={'sha256':result['sha256']}
 rootrows[identity]=dict(producer=producer,receipt_sha256=receipt['sha256'],arrays_sha256=arr['sha256'],acceptance=saved,views=views)
 arows.append(dict(id=identity,receipt_sha256=receipt['sha256'],arrays_sha256=arr['sha256'],views=views,native_comparison=comp))
 rrs.append(dict(id=identity,inventory_record=rec,members=pins,metadata={'path':'/metadata','sha256':m.META_SHA}))
reg(str(H/'EXACT100_REFS.json'),fake_refs,m.REFS_SHA);meta=reg('/metadata',{},m.META_SHA)
sel=reg('/selection',dict(exact_ids=ids,record_producers=refs['record_producers'],original_inventory_sha256=refs['inventory']['sha256'],records=selectionrows));members['base/COHORT_SELECTION.json']={'sha256':sel['sha256']};members['source/metadata.npz']={'sha256':m.META_SHA}
tp=reg('/transport',dict(status='PASS_ARCHIVE_AND_ALL_MEMBER_SHA',all_members_verified=True,archive_sha256='a'*64,members=members))
au=reg('/audit',dict(status='FEDAVG100_SAVED_OUTPUTS_AUDIT_NO_REFIT_PASS_NOT_ROOT_ADOPTED',exact_ids=ids,fit_calls=0,new_CNN=0,new_training=0,test=False,records=arows,metric_values_checked=900,integer_base_counts_checked=2400,prediction_rules_checked=300,transport_sha256=tp['sha256'],archive_sha256='a'*64))
ro=reg('/root',dict(status=m.ROOT_STATUS,ids=ids,accepted=100,metric_values_checked=900,integer_base_counts_checked=2400,prediction_rules_checked=300,archive_all_members_rehashed=True,new_training=False,test_evaluation=False,records=rootrows,audit=au,transport=tp,archive={'sha256':'a'*64}))
ix=reg('/index',dict(status='ROOT_ADOPTED_FEDAVG100_CACHE_INDEX',root=ro,records=rrs,audit=au,transport=tp,selection=sel,metadata=meta))
assert [r['id'] for r in m.verify_root_index(ro,ix,list(reversed(ids)))['records']]==list(reversed(ids));checks.append('synthetic complete100 and explicit reverse projection pass')
for label,mutate in [('missing one',lambda: D['/index']['records'].pop()),('duplicate request',None),('changed original model',lambda:D['/index']['records'][0]['inventory_record'].update(terminal_round=69)),('wrong root count',lambda:D['/root'].update(accepted=99)),('missing role',lambda:D['/index']['records'][0]['members'].pop('original_result')),('native rejected',lambda:D['/receipt/'+ids[0]]['native_comparison'].update(accepted=False)),('transport unverified',lambda:D['/transport'].update(all_members_verified=False)),('future root pin',None)]:
 old=copy.deepcopy(D)
 if mutate:mutate()
 try:m.verify_root_index({'path':None,'sha256':None} if label=='future root pin' else ro,ix,ids[:-1]+[ids[0]] if label=='duplicate request' else ids)
 except (ValueError,KeyError):checks.append('reject '+label)
 else:raise AssertionError('tampering accepted '+label)
 D.clear();D.update(old)
report=dict(status='SOURCE_AND_PURE_METADATA_CHECKS_PASS_NOT_SCIENTIFIC_ACCEPTANCE',checks=checks,science_loop_raw_sha256=hashlib.sha256(science_loop(new).encode()).hexdigest(),scientific_imports=0,arrays_read=0,CNN=0,fit=0,SSH=0,F_access=0,actual100_complete=False,root_adopted=False)
(H/'SOURCE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
