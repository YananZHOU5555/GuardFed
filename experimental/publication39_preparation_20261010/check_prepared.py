from pathlib import Path
import ast,copy,hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent;R=B.parents[1];O=R/'tmp/publication38_preparation_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
sp=importlib.util.spec_from_file_location('pub39',B/'publish_increment39.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
c=read(B/'ACTUAL_CLOSED_INPUTS.json');p=read(B/'PREPARED_INPUTS.json')
n=read(R/c['closure_pins']['native_root']['path']);s=read(R/c['closure_pins']['semantic_root']['path']);f=read(R/c['closure_pins']['FL_root']['path']);h=read(R/c['closure_pins']['Hybrid_root']['path'])
m.closed_guard(c);m.native_guard(n);m.semantic_guard(s);m.delta_guard(f,h)
refused=[]
def reject(label,fn,obj,key,value):
 changed=copy.deepcopy(obj);changed[key]=value
 try:fn(changed)
 except (AssertionError,KeyError,TypeError):refused.append(label)
 else:raise AssertionError('Accepted invalid '+label)
for key,val in [('status','PREPARED'),('parent_commit','0'*40),('counts',dict(m.COUNTS,three_view=170)),('closure_pins',{})]:reject('closed_'+key,m.closed_guard,c,key,val)
for key,val in [('status','OBSERVED'),('native_accepted',172),('added_n',9),('full_reused',0),('minus_C_accepted',60),('added_ids',[]),('ledger_previous25_entries_exact',False),('original260_raw_json_record_bytes_exact',False),('receipt_chain_verified',False),('three_view_acceptance_changed',True),('new_inference',1),('oldFull_models_repacked',1),('test',True),('whole_rebuttal_complete',True)]:reject('native_'+key,m.native_guard,n,key,val)
for key,val in [('status','DRAFT'),('findings',['unresolved']),('required_corrections',['edit']),('original_comments_verbatim_and_ordered',23),('canonical_edits',1),('new_statistics',1)]:reject('semantic_'+key,m.semantic_guard,s,key,val)
for key,val in [('accepted_before',12),('accepted_new',1),('accepted_total',17),('old_models_repacked',1),('final_test',True)]:reject('FL_'+key,lambda x:m.delta_guard(x,h),f,key,val)
for key,val in [('accepted_before',19),('accepted_new',2),('accepted_total',23),('selection_performed',True),('new_inference',1),('formal100_started',True)]:reject('Hybrid_'+key,lambda x:m.delta_guard(f,x),h,key,val)
new=(B/'publish_increment39.py').read_text('utf8');old=(O/'publish_increment38.py').read_text('utf8')
assert new[new.index('    try:\n        for name,source'):]==old[old.index('    try:\n        for name,source'):]
v=(B/'verify_increment39.py').read_text('utf8');ov=(O/'verify_increment38.py').read_text('utf8')
assert v[v.index("    payload = git('cat-file'"):v.index('    changed = set')]==ov[ov.index("    payload = git('cat-file'"):ov.index('    changed = set')]
for file in ['publish_increment39.py','verify_increment39.py']:ast.parse((B/file).read_text('utf8'))
for rel,pin in p['reference_pins'].items():assert sha(R/rel)==pin
proof=dict(status='PASS_SOURCE_METADATA_ONLY_NO_PUBLICATION',positive_guard_calls=4,refusal_checks=len(refused),refused=refused,actual_roots_used=True,copy_forceadd_attributes_index_failure_body_byteexact38=True,commit_blob_parser_byteexact38=True,expected_counts=m.COUNTS,Git_mutations=False,SSH=False,CNN=False,actual_publication=False)
(B/'SELF_CHECK.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8',newline='\n');print(json.dumps(proof))
