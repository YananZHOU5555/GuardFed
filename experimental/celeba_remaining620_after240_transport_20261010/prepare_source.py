from pathlib import Path
import hashlib,json,difflib
H=Path(__file__).resolve().parent;R=H.parents[1];OLD=R/'tmp/celeba_mechanism_remaining620_A40_20261010'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
priorroot=R/'tmp/celeba_mechanism_remaining620_A40_root_adoption_20261010/ROOT_ADOPTION.json'
priorindex=priorroot.parent/'MECHANISM240_INDEX.json';storage=read(OLD/'RAW_STORAGE_INDEX.json')
ids=[f'minus_A_IID_Sp-DFA_seed{s}' for s in range(91001,91011)]+['minus_A_non-IID_Benign_seed91001']
assert read(priorroot)['cumulative_accepted']==240 and sha(priorindex)=='aa0df30db62549f1be808f188fa9a0f9479b9daa812406cd98ff438b2068f8cc'
assert sha(storage['receipt'])==storage['receipt_sha256'] and not set(ids)&set(read(priorindex)['all_ids'])
bind=dict(status='PREPARED_WAITING_ACTUAL_NATIVE251_ROOT_AND_SINGLE_CLOSED_INTERSECTION',candidate_ids=ids,prior_accepted=240,prior_transported=60,prior_root_path=str(priorroot),prior_root_sha256=sha(priorroot),prior_index_path=str(priorindex),prior_index_sha256=sha(priorindex),previous_local_receipt=storage['receipt'],previous_remote_receipt='/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010/exports/A4_20261010T103945406866Z/backup_receipt.json',previous_receipt_sha256=storage['receipt_sha256'],previous_all_transported_ids=storage['all_transported_ids'],native251_root_proof=None,execution_performed=False)
def save(n,x):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:f.write(x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2)+'\n')
save('PREPARED.json',bind);changes={};patch=[]
def change(n,reps):
 raw=(OLD/n).read_text('utf8');new=raw
 for a,b in reps:
  assert new.count(a)==1,(n,a,new.count(a));new=new.replace(a,b,1)
 compile(new,n,'exec');save(n,new);patch.extend(difflib.unified_diff(raw.splitlines(True),new.splitlines(True),fromfile='A40/'+n,tofile='after240/'+n));changes[n]=dict(parent_sha256=sha(OLD/n),new_sha256=sha(H/n),replacements=len(reps))
p=(OLD/'remote_preflight.py').read_text('utf8');idline=next(x for x in p.splitlines() if x.startswith('ids='));latestline=next(x for x in p.splitlines() if x.startswith("assert latest['all_transported_ids']"));start=p.index('notready=[i for i in ids');end=p.index("assert set(latest['all_transported_ids'])",start)
change('remote_preflight.py',[(idline,'candidate_ids='+repr(ids)+'\nids=list(candidate_ids)'),('assert len(ids)==4','assert len(ids)==11'),("'/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010/exports/A8_20261010T095307961006Z/backup_receipt.json'",repr(bind['previous_remote_receipt'])),("'44ff19d96074432a154788c3b27dfa982815087ea5d000888614d0e0711f50cc'",repr(bind['previous_receipt_sha256'])),(latestline,"assert latest['all_transported_ids']=="+repr(bind['previous_all_transported_ids'])+" and latest['accepted_offserver']==0"),('assert not owners and not duplicate and not set(active)&set(ids),(owners,duplicate,active)','assert not owners and not duplicate,(owners,duplicate,active)'),(p[start:end],'''ids=[i for i in candidate_ids if i in set(allclosed)]
notready=[i for i in candidate_ids if i not in set(allclosed)]
assert not set(ids)&set(latest['all_transported_ids']) and not set(ids)&set(active)
if not ids:
    print(json.dumps(dict(status='NO_NEW_CLOSED_INTERSECTION_NO_COLLECTION',candidate_ids=candidate_ids,selected_ids=[],notready=notready,actual_remote_closed=len(allclosed),accepted_offserver=0)))
    sys.exit(0)
'''),("status='EXACT4_CLOSED_NO_SELECTED_WORKER_CPU111_AVAILABLE'","status='BOUNDED_CLOSED_INTERSECTION_NO_SELECTED_WORKER_CPU111_AVAILABLE',candidate_ids=candidate_ids,notready=notready")])
change('export_once.py',[("pre['status']=='EXACT4_CLOSED_NO_SELECTED_WORKER_CPU111_AVAILABLE' and len(pre['selected_ids'])==4","pre['status']=='BOUNDED_CLOSED_INTERSECTION_NO_SELECTED_WORKER_CPU111_AVAILABLE' and 1<=len(pre['selected_ids'])<=11"),("tag='A4_'","tag='after240_'"),("pre=json.loads((HERE/'PREFLIGHT.stdout.json').read_bytes())","pre=json.loads((HERE/'PREFLIGHT.stdout.json').read_bytes())\nexecution=json.loads((HERE/'EXECUTION_INPUTS.json').read_bytes())\nassert execution['native251_root_verified'] is True\nassert set(pre['selected_ids'])<=set(execution['exact_native_minus_prior240_ids'])\nassert pre['candidate_ids']==execution['exact_native_minus_prior240_ids']")])
(H/'remote_cpu111_export.py').write_bytes((OLD/'remote_cpu111_export.py').read_bytes());changes['remote_cpu111_export.py']=dict(parent_sha256=sha(OLD/'remote_cpu111_export.py'),new_sha256=sha(H/'remote_cpu111_export.py'),replacements=0)
change('download_verify_once.py',[("receipt=read(HERE/'EXPORT.stdout.json');export=read(HERE/'EXPORT_COMMAND.json');tag=export['tag']","receipt=read(HERE/'EXPORT.stdout.json');export=read(HERE/'EXPORT_COMMAND.json');tag=export['tag'];N=len(receipt['accepted_new_ids']);assert 1<=N<=11"),("'mechanism_remaining620_A40_20261010'","'remaining620_after240_transport_20261010'"),("'tmp/celeba_mechanism_remaining620_A36_20261010/RAW_STORAGE_INDEX.json'","'tmp/celeba_mechanism_remaining620_A40_20261010/RAW_STORAGE_INDEX.json'"),("'44ff19d96074432a154788c3b27dfa982815087ea5d000888614d0e0711f50cc'",repr(bind['previous_receipt_sha256'])),("saved['accepted_n']==4","saved['accepted_n']==N"),("==(36,96,12)","==(9*N,24*N,3*N)"),("status='F_ONLY_NEW4_ARCHIVE_RECEIPT_AND_ORIGINAL_OFFSERVER_VERIFICATION_PASS_PENDING_ROOT'","status='F_ONLY_FROZEN_CLOSED_INTERSECTION_ORIGINAL_OFFSERVER_VERIFICATION_PASS_PENDING_ROOT'"),("metrics=36,counts=96,rules=12,accepted_offserver=0,root_adopted=0","metrics=9*N,counts=24*N,rules=3*N,accepted_offserver=0,root_adopted=0"),("new=4,cumulative=60","new=N,cumulative=60+N"),("metrics=36,counts=96,rules=12,accepted_offserver=0)),flush=True)","metrics=9*N,counts=24*N,rules=3*N,accepted_offserver=0)),flush=True)")])
save('SOURCE_DIFF.patch',''.join(patch));save('SOURCE_REUSE.json',dict(status='SOURCE_ONLY_PREPARED',files=changes,unchanged_scientific_source={'transport.py':'f71e6e4152625a5a0582a61ff9b3e55e4ccee65dfd70a4e657851c0247f9c9d7'},source_fixture_is_not_scientific_run=True))
