"""Reuse previous once-only transport, changing only exact IDs/parent bindings."""
from pathlib import Path
import difflib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1];OLD=R/'tmp/celeba_remaining620_after240_transport_20261010'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(n,x):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:f.write(x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2)+'\n')
parent=R/'tmp/celeba_mechanism_remaining620_after240_root_adoption_20261010/ROOT_ADOPTION.json';index=parent.parent/'MECHANISM251_INDEX.json'
assert sha(parent)=='edd16e71d6fef9f6e2fc7fd15a7b824d2dfb2a7f809290990878b23477a1e838'
assert sha(index)=='fd1531be09ccaa90190e174dc46296632023db38fdfc11ed249c1e188310c5e6'
storage=read(OLD/'RAW_STORAGE_INDEX.json');assert sha(OLD/'RAW_STORAGE_INDEX.json')=='f38227cd8333a1e72f479cede17cf4dbe7b0349cc55badeeea1bfc1cef252767'
assert sha(storage['receipt'])==storage['receipt_sha256']=='cbcbbed0d17aad648aa0c783119c04738e2bef312ef0ba4c4762d6174c500100'
ids=[f'minus_A_non-IID_Benign_seed{s}' for s in range(91002,91011)]
assert len(read(index)['all_ids'])==251 and len(storage['all_transported_ids'])==71
assert not set(ids)&set(read(index)['all_ids']) and not set(ids)&set(storage['all_transported_ids'])
previous_remote='/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010/exports/after240_20261010T112307001485Z/backup_receipt.json'
prepared=dict(status='PREPARED_WAITING_ROOT_NATIVE_PROOF_FOR_EXACT9',candidate_ids=ids,prior_accepted=251,prior_transported=71,prior_root_path=str(parent),prior_root_sha256=sha(parent),prior_index_path=str(index),prior_index_sha256=sha(index),previous_local_receipt=storage['receipt'],previous_remote_receipt=previous_remote,previous_receipt_sha256=storage['receipt_sha256'],previous_all_transported_ids=storage['all_transported_ids'],execution_performed=False)
save('PREPARED.json',prepared);changes={};patch=[]
def change(name,replacements):
 old=(OLD/name).read_text('utf8');new=old
 for a,b in replacements:
  assert new.count(a)==1,(name,a,new.count(a));new=new.replace(a,b,1)
 compile(new,name,'exec');save(name,new)
 changes[name]={'parent_sha256':sha(OLD/name),'new_sha256':sha(H/name),'exact_replacements':len(replacements)}
 patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='after240/'+name,tofile='after251/'+name))
pre=(OLD/'remote_preflight.py').read_text('utf8');idline=next(x for x in pre.splitlines() if x.startswith('candidate_ids='));latestline=next(x for x in pre.splitlines() if x.startswith("assert latest['all_transported_ids']"))
change('remote_preflight.py',[(idline,'candidate_ids='+repr(ids)),('assert len(ids)==11','assert len(ids)==9'),("'/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010/exports/A4_20261010T103945406866Z/backup_receipt.json'",repr(previous_remote)),("'5c988da2fe11625f949d4baee5be426b6b184dfbccc78782a2a6abcb236759ac'",repr(storage['receipt_sha256'])),(latestline,"assert latest['all_transported_ids']=="+repr(storage['all_transported_ids'])+" and latest['accepted_offserver']==0"),("assert set(latest['all_transported_ids'])<=set(allclosed)","assert ids==candidate_ids and not notready, 'Exact9 batch is not entirely closed; do not export a partial batch'\nassert set(latest['all_transported_ids'])<=set(allclosed)")])
change('execute_once.py',[("exe=json.loads((H/'EXECUTION_INPUTS.json').read_bytes());assert exe['native251_root_verified']","from verify_native_inputs import verify_native_inputs\n exe=verify_native_inputs()")])
change('export_once.py',[("execution=json.loads((HERE/'EXECUTION_INPUTS.json').read_bytes())\nassert execution['native251_root_verified'] is True","from verify_native_inputs import verify_native_inputs\nexecution=verify_native_inputs()"),("set(pre['selected_ids'])<=set(execution['exact_native_minus_prior240_ids'])","pre['selected_ids']==execution['exact_candidate_ids']"),("pre['candidate_ids']==execution['exact_native_minus_prior240_ids']","pre['candidate_ids']==execution['exact_candidate_ids']"),("1<=len(pre['selected_ids'])<=11","len(pre['selected_ids'])==9"),("tag='after240_'","tag='after251_'")])
change('download_verify_once.py',[("N=len(receipt['accepted_new_ids']);assert 1<=N<=11","N=len(receipt['accepted_new_ids']);assert N==9"),("'remaining620_after240_transport_20261010'","'remaining620_after251_transport_20261010'"),("'tmp/celeba_mechanism_remaining620_A40_20261010/RAW_STORAGE_INDEX.json'","'tmp/celeba_remaining620_after240_transport_20261010/RAW_STORAGE_INDEX.json'"),("'5c988da2fe11625f949d4baee5be426b6b184dfbccc78782a2a6abcb236759ac'",repr(storage['receipt_sha256'])),("cumulative=60+N","cumulative=71+N")])
(H/'remote_cpu111_export.py').write_bytes((OLD/'remote_cpu111_export.py').read_bytes())
changes['remote_cpu111_export.py']={'parent_sha256':sha(OLD/'remote_cpu111_export.py'),'new_sha256':sha(H/'remote_cpu111_export.py'),'exact_replacements':0}
transport=R/'tmp/celeba_mechanism_remaining_evaluation_transport_20261010/transport.py'
assert sha(transport)=='f71e6e4152625a5a0582a61ff9b3e55e4ccee65dfd70a4e657851c0247f9c9d7'
save('SOURCE_DIFF.patch',''.join(patch));save('SOURCE_REUSE.json',{'files':changes,'unchanged_transport_sha256':sha(transport),'scientific_functions_changed':False,'export_CPU110_to111_only_in_memory':True,'new_CNN':0,'new_training':0,'execution_performed':False})
