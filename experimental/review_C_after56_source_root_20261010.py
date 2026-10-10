"""Independent offline check of actual exact-four source and saved native identities."""
from pathlib import Path,PurePosixPath
import ast,copy,datetime,hashlib,importlib.util,json,subprocess,sys,tarfile
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1]
B=R/'tmp/celeba_mechanism_valid_C_after56_20261010'
P=R/'tmp/celeba_mechanism_valid_C_after50_20261010'
H=R/'tmp/celeba_mechanism_C_after56_source_review_20261010'
IDS=[f'minus_C_non-IID_Benign_seed{s}' for s in range(91007,91011)]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def funcs(p):
 s=p.read_text('utf8');return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
def load(n,p):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);sys.modules[n]=m;spec.loader.exec_module(m);return m
def pins(root,rows):
 if isinstance(rows,dict):rows=[dict(v,path=k) for k,v in rows.items()]
 for row in rows:
  p=root/row['path'];assert p.is_file() and sha(p)==row['sha256'] and p.stat().st_size==row.get('size',row.get('bytes')),str(p)
 return len(rows)
assert not sys.flags.optimize
H.mkdir(exist_ok=False)
pin={'package':'6fd1881566baf4b7de107752f07354bf8f591f117681e385a12fa1e92307235a','science':'2095ab384a7844355fc92453dbfa5d2922f88f378838e7b7eda2524578f3bbd6','execution':'d1679c0bbd53bc66e4ea7ae792000d398efcafe5192bc5164ffd80e7a2eeb236','inventory':'302dd45e9f05c646671d31d26775607af7a4fe70876fa1e60643939f972742f4'}
paths={'package':B/'PACKAGE_SHA256.json','science':B/'FILES_SHA256.json','execution':B/'execution_candidate/EXECUTION_SOURCE_SHA256.json','inventory':B/'inventory_actual160_Full100refs.json'}
assert all(sha(paths[k])==v for k,v in pin.items())
counts={k:pins(paths[k].parent,read(paths[k]).get('members',read(paths[k]).get('files'))) for k in ['package','science','execution']}
counts['inputs']=pins(R,read(B/'INPUT_PINS.json'))
receipt=read(B/'PACKAGE_RECEIPT.json');archive=B/receipt['source_archive'];assert sha(archive)==receipt['source_archive_sha256']
with tarfile.open(archive) as bundle:
 assert len(bundle.getnames())==len(set(bundle.getnames()))==receipt['archive_members']==29
 assert set(bundle.getnames())==set(receipt['members'])
 for item in bundle:
  rel=PurePosixPath(item.name);assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts
  raw=bundle.extractfile(item).read();row=receipt['members'][item.name]
  assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'] and raw==(B/item.name).read_bytes()
old=read(P/'inventory_actual156_Full100refs.json');current=read(paths['inventory'])
before={r['id']:r for r in old['records']};after={r['id']:r for r in current['records']}
assert len(before)==156 and len(after)==len(current['records'])==160
assert set(after)-set(before)==set(IDS) and current['selected_replay_ids']==IDS
assert all(after[k]==v for k,v in before.items()) and set(current['excluded_prior_replay_ids'])==set(before)
old_review=load('original_record_bytes_review',R/'tmp/celeba_mechanism_C_after50_source_review_20261010/review_sealed.py')
assert [x for x in old_review.record_bytes(paths['inventory']) if x[0] in before]==old_review.record_bytes(P/'inventory_actual156_Full100refs.json')
assert current['full_references']==old['full_references'] and len(current['full_references'])==100 and current['native_tolerance']==1e-12
functions=['require','digest','canonical','read','save_new','load','cell','full_reference','validate_variant_metadata','bind_runtime','reference_baseline_full']
assert all(funcs(P/'bridge.py')[n]==funcs(B/'bridge.py')[n] for n in functions)
assert (P/'execution_candidate/resource_extra.py').read_bytes()==(B/'execution_candidate/resource_extra.py').read_bytes()
for n in ['budget_snapshot','assert_cpu_available','fresh_cpu_scan']:
 assert funcs(P/'execution_candidate/install_once.py')[n]==funcs(B/'execution_candidate/install_once.py')[n]
assert funcs(P/'execution_candidate/batch.py')['runtime_policy']==funcs(B/'execution_candidate/batch.py')['runtime_policy']
run=subprocess.run([sys.executable,'-B',str(B/'check_prepared.py')],capture_output=True,timeout=180)
(H/'METADATA_STDOUT.json').write_bytes(run.stdout);(H/'METADATA_STDERR.log').write_bytes(run.stderr)
assert run.returncode==0,run.stderr.decode(errors='replace')
metadata=json.loads(run.stdout);assert metadata==read(B/'SELF_CHECK.json')
assert metadata['worker_original_pre_bind_path_reached']==IDS and metadata['per_child_exact4_science_approval_positive']==4 and metadata['refusal_count']==69
assert not metadata['CNN'] and not metadata['torch_imported'] and not metadata['numpy_imported']
ni=read(B/'NATIVE_INPUTS.json');native=R/ni['files']['root_review']['path']
assert sha(native)==ni['files']['root_review']['sha256']=='369c5d80169f5188604f46565116a184de3b1c84eaf53ac4bc3794279a0e6ee4'
assert read(native)['native_accepted']==160 and read(native)['added_ids']==IDS
baseline=read(R/'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json')
bridge=load('original_C_after56_bridge_metadata',B/'bridge.py')
full={bridge.cell(r):r for r in baseline['records'] if r['method']=='GuardFed-AD2+'}
manifest={r['id']:r for r in read(R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/manifest.json')['jobs']}
inspection={r['id']:r for r in read(R/ni['files']['inspection']['path'])['records']}
constructor_path=R/'tmp/celeba_mechanism_valid_incremental_next11_20261009/prepare.py'
nodes=[n for n in ast.walk(ast.parse(constructor_path.read_text('utf8'))) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='record' for t in n.targets)]
assert len(nodes)==1
constructor=compile(ast.Module(body=nodes,type_ignores=[]),'original_record_constructor','exec')
native_archive=R/ni['files']['archive']['path']
with tarfile.open(native_archive) as bundle:
 members=json.load(bundle.extractfile('backup_inventory.json'))['members']
 for model_id in IDS:
  result=json.load(bundle.extractfile('runs/'+model_id+'/result.json'));job=json.load(bundle.extractfile('jobs/'+model_id+'.json'))
  refs={k:{'archive':native_archive.relative_to(R).as_posix(),'archive_sha256':sha(native_archive),'member':n,**members[n]} for k,n in [('checkpoint','runs/'+model_id+'/model.pt'),('result','runs/'+model_id+'/result.json'),('raw_job','jobs/'+model_id+'.json')]}
  ns=dict(model_id=model_id,result=result,job=job,row=inspection[model_id],entry=manifest[model_id],control=full[job['distribution'],job['attack'],job['config']['seed']],refs=refs,b=bridge,copy=copy)
  exec(constructor,ns);assert ns['record']==after[model_id]
assert all(inspection[r['id']]==r['accepted_v4_row'] for r in current['records'])
bindings=read(B/'FULL100_ACTUAL_SOURCE_BINDINGS.json');assert bindings['records']==read(P/'FULL100_ACTUAL_SOURCE_BINDINGS.json')['records']
assert sha(R/bindings['source_records_path'])==bindings['source_records_sha256']
actual={r['id']:r for r in read(R/bindings['source_records_path'])['records']}
for row in bindings['records']:
 ref=row['full_reference'];r=actual[ref['id']]
 assert r['checkpoint_sha256']==ref['checkpoint_sha256'] and r['receipt_sha256']==row['receipt_sha256'] and r['source_binding']==row['source_binding']
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
report=dict(status='PASS_SOURCE_READY_FOR_ROOT_LINUX_PREFLIGHT_AND_EXACT4_APPROVAL',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_adoptable=True,actual_dispatch_authorized_by_this_review=False,
 science_seal_sha256=pin['science'],execution_seal_sha256=pin['execution'],package_sha256=pin['package'],inventory_sha256=pin['inventory'],package_receipt_sha256=sha(B/'PACKAGE_RECEIPT.json'),source_handoff_sha256=sha(B/'HANDOFF.json'),
 package_members_verified=counts['package'],science_members_verified=counts['science'],execution_members_verified=counts['execution'],source_tar_members_verified=29,input_pins_verified=counts['inputs'],
 native_accepted_snapshot=160,excluded_prior_three_view_ids=156,old156_records_exact=True,old156_inventory_record_bytes_exact=True,Full100_references_exact=True,Full100_actual900_source_records_exact=True,
 new_three_view_accepted=0,exact_selected_ids=IDS,actual_worker_pre_science_bind_ids=IDS,positive_approval_exact4=True,metadata_refusals=69,actual_metadata_check=metadata,
 unchanged_bridge_scientific_functions=functions,resource_module_byte_exact_to_parent=True,budget_plus3_and_all_thread_cpu_owner_functions_byte_exact=True,prior_service_EXITED_guard=True,
 frozen_tolerance=1e-12,views=['native','raw','shared_calibration'],CPU_budget=list(range(112,120)),torch_threads=8,max_processes=1,nice=10,idle_IO=True,CUDA_hidden=True,
 actual_Linux_resource_availability_checked=False,new_external_root_approval_required=True,independent_native_review_sha256=sha(native),prior156_root_adoption_sha256='a7231a724ea3a4d3443bbe196137b0c72db025cbc98d3ac66842334ab0fe9080',
 findings=[],new_training=0,new_CNN=0,test=False,whole_rebuttal_complete=False)
with (H/'ROOT_INDEPENDENT_REVIEW.json').open('x',encoding='utf8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'status':report['status'],'sha256':sha(H/'ROOT_INDEPENDENT_REVIEW.json'),'counts':counts}))
