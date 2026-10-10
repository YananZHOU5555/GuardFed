"""Metadata-only checks; mocked approval/Linux I/O exists only in this process."""
import ast,copy,hashlib,importlib.util,io,json,sys,types
from pathlib import Path
from unittest.mock import patch
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;E=H/'execution_candidate';ROOT=H.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
b=load('test_bridge',H/'bridge.py');q=load('test_batch',E/'batch.py')
inv=read(H/'inventory_actual156_Full100refs.json');base=read(ROOT/'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json')
assert len(b.validate_inventory(inv,base))==156
scope=q.identities();assert scope['actual_native_accepted_records']==156 and scope['pending_native_count_without_model']==644 and len(scope['excluded_prior_ids'])==150;ids=q.SELECTED;ss=sha(H/'SCOPE.json');es=sha(E/'EXECUTION_SOURCE_SHA256.json')
checks=[]
def reject(name,fn):
 try:fn()
 except (ValueError,KeyError,AssertionError):checks.append(name);return
 raise AssertionError('Not rejected: '+name)
def mutate_inv(name,change):
 bad=copy.deepcopy(inv);change(bad);reject(name,lambda:b.validate_inventory(bad,base))
for n in (0,1,3,5,7,11):
 mutate_inv('inventory_selected_count_'+str(n),lambda v,n=n:v.update(selected_replay_ids=(ids+['minus_C_IID_Benign_seed91002']*11)[:n]))
for name,change in [
 ('duplicate_record',lambda v:v['records'].__setitem__(-1,v['records'][0])),
 ('prior150_in_scope',lambda v:v['selected_replay_ids'].__setitem__(0,v['excluded_prior_replay_ids'][0])),
 ('Full_in_scope',lambda v:v['selected_replay_ids'].__setitem__(0,v['full_references'][0]['id'])),
 ('other_variant',lambda v:v['records'][0].update(variant='minus_A')),
 ('wrong_mask',lambda v:v['records'][0]['config'].update(ablation_component='U')),
 ('wrong_source',lambda v:v['records'][0]['source_hashes'].update({'scripts/reproduce_paper_tables.py':'0'*64})),
 ('wrong_checkpoint',lambda v:v['records'][0]['checkpoint'].update(sha256='0'*64)),
 ('wrong_split',lambda v:v['records'][0].update(original_split='test')),
 ('short_rounds',lambda v:v['records'][0].update(terminal_round=69)),
 ('subset',lambda v:v['records'][0].update(original_n_eval=100)),
 ('wrong_seed',lambda v:v['records'][0].update(seed=1)),
 ('wrong_alpha',lambda v:v['records'][0].update(actual_alpha=5)),
 ('tolerance',lambda v:v.update(native_tolerance=1e-6)),
 ('wrong_views',lambda v:v.update(views=['native'])),
 ('Full_reference',lambda v:v['full_references'][0].update(checkpoint_sha256='0'*64)),
 ('wrong_boundary',lambda v:v.update(excluded_prior_replay_ids=[]))]:mutate_inv(name,change)
# Exact bound C identity and explicit semantic mapping from frozen source proposal.
assert ids==[f'minus_C_non-IID_Benign_seed{s}' for s in range(91001,91007)]
c=next(r for r in inv['records'] if r['id']==ids[0]);b.validate_variant_metadata(c)
assert c['config']['ablation_component']=='C'
for field,value in [('ablation_component','U'),('ablation_component','none'),('experiment_suite','other'),('experiment_tag','minus_U_IID_Benign_seed91001'),('full_round_diagnostics',False)]:
 bad=copy.deepcopy(c);bad['config'][field]=value
 reject('C1_semantic_'+field+'_'+str(value),lambda bad=bad:b.validate_variant_metadata(bad))
for variant in ['Full','minus_U','no_hard_screen','fixed_balanced','no_candidate']:
 bad=copy.deepcopy(c);bad['variant']=variant
 reject('C1_disguised_'+variant,lambda bad=bad:b.validate_variant_metadata(bad))
for foreign in ['minus_C_IID_Benign_seed91001','minus_U_IID_Benign_seed91001','minus_C_IID_S-DFA_seed91006']:
 mutate_inv('single_gate_foreign_'+foreign,lambda v,foreign=foreign:v.update(selected_replay_ids=[foreign]))
structural=[]  # Unchanged planned structural variants are outside this bounded C6 check.
# Real original batch approval guards; synthetic authority is process memory only.
root=read(E/'ROOT_REVIEW_TEMPLATE.json');root.update(status='ROOT_REVIEW_PASS_BOUNDED_C_AFTER50_VALID_REPLAY',execution_authorized_within_existing_user_request=True,execution_seal_sha256=es)
rsha=hashlib.sha256(json.dumps(root,sort_keys=True).encode()).hexdigest()
ap=read(E/'APPROVED_TEMPLATE.json');ap.update(status='APPROVED_C_AFTER50_MECHANISM_VALID_REPLAY_ONLY',root_approval_sha256=rsha,execution_seal_sha256=es)
orig_read=q.read;orig_digest=q.digest
with patch.object(q,'read',lambda p:root if Path(p).name=='ROOT_APPROVED.json' else orig_read(p)),patch.object(q,'digest',lambda p:rsha if Path(p).name=='ROOT_APPROVED.json' else orig_digest(p)):
 q.check_approval(ap,scope,ss,es)
 reject('execution_template_unapproved',lambda:q.check_approval(read(E/'APPROVED_TEMPLATE.json'),scope,ss,es))
 for key,value in [('selected_ids',ids[:-1]),('selected_ids',ids+[ids[0]]),('compute_threads',7),('max_processes',2),('allowed_cpus',[105]),('target_split','test'),('native_tolerance',1e-6),('automatic_retry_authorized',True),('new_full_inference',1),('inventory_sha256','0'*64),('bridge_sha256','0'*64),('execution_seal_sha256','0'*64),('root_approval_sha256','0'*64),('outputs',{}),('dependency_paths',{})]:
  bad=copy.deepcopy(ap);bad[key]=value;reject('batch_'+key+'_'+str(len(checks)),lambda bad=bad:q.check_approval(bad,scope,ss,es))
 for key,value in [('status','PREPARED_NOT_APPROVED'),('selected_ids',ids[:-1]),('closed150_must_not_replay',[]),('source_only_root_review_sha256','0'*64)]:
  saved=copy.deepcopy(root);root[key]=value;reject('root_'+key,lambda:q.check_approval(ap,scope,ss,es));root.clear();root.update(saved)
child=dict(status='APPROVED_BOUNDED_MECHANISM_VALID_REPLAY_ONLY',scope=b.SCOPE,inventory_sha256=q.INVENTORY_SHA,bridge_sha256=q.BRIDGE_SHA,selected_ids=ids,selected_id=ids[0],device='cpu',compute_threads=8,max_processes=1,allowed_cpus=q.CPUS,target_split='valid',native_tolerance=1e-12,final_test_dispatch=False)
for i in ids:b.require_approval(child,q.INVENTORY_SHA,i,q.BRIDGE_SHA)
for key,value in [('selected_ids',ids[:-1]),('selected_ids',ids+[ids[0]]),('device','cuda'),('compute_threads',1),('max_processes',2),('allowed_cpus',[105]),('target_split','test'),('native_tolerance',1e-6),('status','PREPARED_NOT_APPROVED')]:
 bad=copy.deepcopy(child);bad[key]=value;reject('bridge_'+key,lambda bad=bad:b.require_approval(bad,q.INVENTORY_SHA,ids[0],q.BRIDGE_SHA))
for key,value in [('outputs',{}),('dependency_paths',{}),('native_tolerance',1e-3)]:
 bind=read(E/'RUNTIME_BINDINGS.json');bind[key]=value;reject('runtime_binding_'+key,lambda bind=bind:q.bind_scope(read(H/'SCOPE.json'),bind))
# Execute unchanged worker from approved() through the real bind_runtime call site.
# Remote file I/O, Linux lock and runtime_policy are simulated, scientific binding is not run.
remote=q.REMOTE;real_stat=Path.stat;real_exists=Path.exists;real_open=Path.open;real_resolve=Path.resolve
memory={remote/'ROOT_APPROVED.json':root,remote/'APPROVED.json':ap}
psha=hashlib.sha256(json.dumps(ap,sort_keys=True).encode()).hexdigest();hashes={remote/'ROOT_APPROVED.json':rsha,remote/'APPROVED.json':psha}
for i in ids:
 c=dict(child,selected_id=i,parent_batch_approval_sha256=psha,output=ap['outputs'][i],dependency_paths=ap['dependency_paths'])
 p=remote/'approvals'/(i+'.json');memory[p]=c;hashes[p]=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
def mapped(p):
 p=Path(p)
 try:return E/p.relative_to(remote)
 except ValueError:return p
def mr(p):return copy.deepcopy(memory[Path(p)]) if Path(p) in memory else orig_read(mapped(p))
def md(p):return hashes[Path(p)] if Path(p) in hashes else orig_digest(mapped(p))
def st(p,*a,**k):return real_stat(E/'batch.py' if p in memory else mapped(p),*a,**k)
def op(p,*a,**k):return io.StringIO() if p==remote/'compute_worker.lock' else real_open(mapped(p),*a,**k)
class BeforeScientificBind(Exception):pass
reached=[]
def boundary(inventory_path,inventory_sha,deps,repo,identity,approval_path,approval_sha):
 assert Path(inventory_path)==H/'inventory_actual156_Full100refs.json' and inventory_sha==q.INVENTORY_SHA
 assert approval_sha==md(approval_path) and deps==q.DEPENDENCIES
 b.require_approval(mr(approval_path),inventory_sha,identity,q.BRIDGE_SHA)
 reached.append(identity);raise BeforeScientificBind()
original_load=q.load
def bridge_load(*args):
 module=original_load(*args);module.bind_runtime=boundary;return module
fakefcntl=types.SimpleNamespace(flock=lambda *a:None,LOCK_EX=1,LOCK_NB=2)
with patch.object(q,'HERE',remote),patch.object(q,'read',mr),patch.object(q,'digest',md),patch.object(q,'runtime_policy',lambda:None),patch.object(q,'load',bridge_load),patch.object(Path,'stat',st),patch.object(Path,'open',op),patch.object(Path,'resolve',lambda p,*a,**k:p if str(p).startswith(str(remote)) else real_resolve(p,*a,**k)),patch.dict(sys.modules,{'fcntl':fakefcntl}):
 for i in ids:
  try:q.worker(remote/'APPROVED.json',psha,i)
  except BeforeScientificBind:pass
  else:raise AssertionError('Unexpected scientific continuation')
assert reached==ids and 'torch' not in sys.modules and 'numpy' not in sys.modules
# Original fresh-output guard: no writes, no scientific execution.
probe=Path('/bounded_metadata_only_new_output')
with patch.object(Path,'exists',lambda p:False):q.fresh_output(probe)
for badpath in [probe,probe.with_name(probe.name+'.bridge_failure.json')]:
 with patch.object(Path,'exists',lambda p,badpath=badpath:p==badpath):reject('preserved_partial_or_failure_'+str(badpath),lambda:q.fresh_output(probe))
# Source identity: ten science functions and complete resource module exact.
old=H.with_name('celeba_mechanism_valid_C_after47_20261010')
def funcs(p):
 s=p.read_text();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
a,z=funcs(old/'bridge.py'),funcs(H/'bridge.py');names=read(H/'SOURCE_REUSE.json')['exact_bridge_functions'];assert all(a[n]==z[n] for n in names)
assert sha(E/'resource_extra.py')==sha(old/'execution_candidate/resource_extra.py')
ins=(E/'install_once.py').read_text();assert 'len(chosen)==6' in ins;assert "service = 'guardfed_celeba_mechanism_valid_C_after47'" in ins and '3859d49fb57c3ecc4b23244012431224d255b02590dab221b2d6464e28aa7dd8' in ins
report={'status':'METADATA_GATE_PASS_NOT_LINUX_RUNTIME_OR_EXECUTION_APPROVAL','science_seal':sha(H/'FILES_SHA256.json'),'execution_seal':es,'positive_inventory_records':156,'real_batch_identities_pass':True,'batch_approval_positive_exact6':True,'worker_original_pre_bind_path_reached':reached,'per_child_exact6_science_approval_positive':6,'refusals':checks,'refusal_count':len(checks),'unchanged_scientific_functions':names,'resource_module_byte_exact':True,'mock_boundaries':['external approval exists only in memory','remote file mapping','Linux runtime_policy and fcntl','bind_runtime replaced with terminal sentinel before any scientific import'],'CNN':False,'torch_imported':False,'numpy_imported':False,'actual_execution_authorized':False,'C_component':'C','structural_source_metadata_only':structural}
print(json.dumps(report,indent=2))
