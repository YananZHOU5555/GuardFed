"""Stdlib-only source/evidence-shape check; no collector, SSH, models or statistics."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json
H=Path(__file__).resolve().parent;R=H.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def raw_node(raw,n):return b''.join(raw.splitlines(keepends=True)[n.lineno-1:n.end_lineno])
def main():
 p=read(H/'SOURCE_PARENT_PINS.json');scope=read(H/'FROZEN_SCOPE.json')
 for prefix in ['parent_wrapper','parent_adopter','original_collector','original_science']:assert sha(R/p[prefix+'_path'])==p[prefix+'_sha256']
 for n in ['run_once.py','prepare_source.py','adopt_root.py','check_source.py']:compile((H/n).read_bytes(),str(H/n),'exec')
 spec=importlib.util.spec_from_file_location('source_derivation_only',H/'run_once.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 effective,s=m.derive();compile(effective,'<derived original collector plus exact10 metadata guard>','exec')
 original=(R/p['original_collector_path']).read_text('utf8')
 inverse=effective.replace(m.GUARD,'').replace("bindings=dict(EXACT_IDS=SCOPE['candidate_ids'],PRIOR_INSPECTION=SCOPE['parent_remote_inspection_path'],PRIOR_INSPECTION_SHA=SCOPE['parent_inspection_sha256'],TOOL=TOOL,HASH=HASH,",'bindings=dict(TOOL=TOOL,HASH=HASH,')
 for a,b in [('native_delta_after350_nonIID_Benign10_20261011','native_delta_after288_20261011'),('celeba_native_after350_nonIID_Benign10_prepared_20261011','celeba_native_after288_20261011')]:inverse=inverse.replace(a,b)
 assert inverse==original and ast.dump(ast.parse(inverse),include_attributes=False)==ast.dump(ast.parse(original),include_attributes=False)
 # Entire original evidence implementation is referenced by the same SHA; no science copy exists here.
 tool=(R/p['original_science_path']).read_text('utf8');tree=ast.parse(tool)
 functions={n.name:hashlib.sha256(ast.get_source_segment(tool,n).encode()).hexdigest() for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['inspect','backup','verify_archive']}
 assert len(functions)==3
 raw=(H/'adopt_root.py').read_bytes();old=(R/p['parent_adopter_path']).read_bytes()
 def find_loop(b,target,condition=lambda n:True):
  nodes=[n for n in ast.walk(ast.parse(b)) if isinstance(n,ast.For) and ast.unparse(n.target)==target and condition(n)];assert len(nodes)==1;return nodes[0]
 loops={}
 for target,condition in [('m',lambda n:True),('row',lambda n:True),('key',lambda n:'frozen_source_hashes' in ast.unparse(n.iter))]:
  a=find_loop(raw,target,condition);b=find_loop(old,target,condition)
  assert raw_node(raw,a)==raw_node(old,b) and ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
  loops[target]=hashlib.sha256(raw_node(raw,a)).hexdigest()
 assert "TOTAL==360" in raw.decode() and 'assert NEW_IDS==REQUIRED_SCENE and len(NEW_IDS)==10' in raw.decode()
 assert 'off[\'members_verified\'] == MEMBERS' in raw.decode()
 assert sha(H/'PARENT_LEDGER.json')==scope['parent_ledger_sha256']==p['parent_ledger_sha256']
 assert len(read(H/'PARENT_LEDGER.json')['entries'])==49
 oldreport=read(R/scope['parent_inspection_path']);assert len(oldreport['records'])==450 and oldreport['new_count']==350
 assert scope['candidate_ids']==[f'minus_F_non-IID_Benign_seed{s}' for s in range(91001,91011)]
 assert not set(scope['candidate_ids'])&set(oldreport['accepted_new_ids'])
 # Evaluate only the actual pure metadata assignment/assertions; never Path/SHA/remote operations.
 guard=ast.parse('if True:\n'+m.GUARD).body[0].body[0].body
 selected=[n for n in guard if (isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='scope') or (isinstance(n,ast.Assert) and 'sha(' not in ast.unparse(n.test))]
 assert len(selected)==6
 code=compile(ast.Module(body=selected,type_ignores=[]),'<exact actual metadata guard assertions>','exec')
 ids=scope['candidate_ids'];prior={'new_count':350,'accepted_new_ids':oldreport['accepted_new_ids']}
 base={'invalid':[],'new_count':360,'accepted_new_ids':prior['accepted_new_ids']+ids}
 def fixture(report,prior_value=prior,selected_ids=ids):
  ns={'inspected':report,'prior_inspected':prior_value,'EXACT_IDS':selected_ids};exec(code,ns);return ns['scope']
 assert fixture(base)['actual_delta']==ids
 tests=[]
 for label,mut in [('extra361',lambda d:d.update(new_count=361,accepted_new_ids=d['accepted_new_ids']+['minus_F_non-IID_F Flip_seed91001'])),('missing_seed10',lambda d:d.update(new_count=359,accepted_new_ids=d['accepted_new_ids'][:-1])),('duplicate',lambda d:d.update(accepted_new_ids=d['accepted_new_ids']+[ids[0]])),('old_ID_missing',lambda d:d['accepted_new_ids'].remove(prior['accepted_new_ids'][0])),('invalid_record',lambda d:d.update(invalid=['fixture'])),('false_count',lambda d:d.update(new_count=352))]:
  d=copy.deepcopy(base);mut(d)
  try:fixture(d)
  except AssertionError:tests.append(label)
  else:raise AssertionError('Metadata guard accepted '+label)
 assert all(read(H/n)['snapshot_path'] is None for n in ['READINESS.template.json'])
 assert read(H/'ACTUAL_INPUTS.template.json')['archive_members'] is None
 assert not any((H/n).exists() for n in ['ACTUAL_INPUTS.json','COLLECTOR_SOURCE_REVIEW.json','AUTHORIZED_SNAPSHOT.json'])
 return dict(status='SOURCE_ONLY_EXACT10_METADATA_GUARD_PASS',source_only=True,execution_performed=False,SSH=0,science=0,models_arrays_read=0,parent_native=350,target_native=360,Full_reused=100,target_records=460,saved_views_change=0,
  original_collector_inverse_text_and_AST_exact=True,effective_collector_sha256=hashlib.sha256(effective.encode()).hexdigest(),original_science_sha256=p['original_science_sha256'],original_scientific_function_source_sha256=functions,root_adopter_raw_and_AST_preserved_loops=loops,
  original_source_resource_strict_archive_transport_offserver_unmodified=True,metadata_guard_added_before_backup=True,extra_terminal_action='REFUSE_PRESERVE_INSPECTION_NO_BACKUP_NO_RETRY',positive_exact10_pass=True,refused_cases=tests,
  actual_snapshot_runtime_output_unbound=True,actual_archive_members_unbound=True,parent_root_sha256=scope['parent_root_sha256'],parent_ledger_sha256=scope['parent_ledger_sha256'])
if __name__=='__main__':print(json.dumps(main(),indent=2))
