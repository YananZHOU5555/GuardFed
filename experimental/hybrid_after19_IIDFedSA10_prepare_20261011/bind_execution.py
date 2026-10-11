"""Root-only metadata late binding. No SSH, Torch, model, array or F operations."""
from pathlib import Path
import argparse,hashlib,json,sys
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def put(name,value):
 with (H/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2);f.write('\n')
def main():
 p=argparse.ArgumentParser()
 for n in ['source-seal-sha256','snapshot-sha256','source-review-sha256']:p.add_argument('--'+n,required=True)
 for n in ['snapshot','source-review']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--root-authorize-execution',action='store_true',required=True);a=p.parse_args()
 assert not sys.flags.optimize and a.root_authorize_execution
 assert not any((H/n).exists() for n in ['EXECUTION_BINDINGS.json','FROZEN_OBSERVATION.json'])
 assert not list(H.glob('EXECUTION_FAILURE_*.json'))
 assert sha(H/'FILES_SHA256.json')==a.source_seal_sha256
 for n,v in read(H/'FILES_SHA256.json')['files'].items():
  q=(H/n).resolve();assert q.is_relative_to(H) and sha(q)==v['sha256'] and q.stat().st_size==v['bytes']
 a.snapshot=a.snapshot.resolve();a.source_review=a.source_review.resolve()
 assert a.snapshot.is_relative_to(ROOT) and a.source_review==H/'ROOT_SOURCE_REVIEW.json'
 assert sha(a.snapshot)==a.snapshot_sha256 and sha(a.source_review)==a.source_review_sha256
 assert read(a.source_review)['source_adoptable'] is True
 scope=read(H/'DELTA_SCOPE.json');fixed=read(H/'FIXED_MANIFEST.json');prior=read(H/'PRIOR_ROOT_ADOPTION.json')
 x=read(a.snapshot);h=x['Hybrid96'];ids=scope['ids']
 assert scope['accepted_before']==19 and len(ids)==10 and fixed['prior_ids']==prior['accepted_ids']
 assert h['package_sha256']==h['source']['sha256']==fixed['package_sha256']
 assert not h['source']['changed_members'] and not x['source_data_binding']['changed_paths']
 assert not h['failure_files'] and not h['queue_progress']['failed']
 assert h['service']['returncode']==0 and h['service']['stdout'].split()[1]=='RUNNING'
 assert not set(ids)&{w['id'] for w in h['workers']}
 rows=h['terminal_records'];assert len({r['id'] for r in rows})==len(rows)
 selected=[r for r in rows if r['id'] in ids];assert [r['id'] for r in selected]==ids
 assert set(prior['accepted_ids'])<=set(r['id'] for r in rows)
 for r,e in zip(selected,fixed['selected_entries']):
  assert r['acceptance']['status']=='PASS' and r['acceptance']['job_sha256']==e['job_sha256']
  assert r['acceptance']['scope_sha256']==fixed['full_scope_sha256']
 frozen=dict(source_path=str(a.snapshot),sha256=a.snapshot_sha256,observation_utc=h['checked_utc'],terminal_count=len(rows),selected_records=selected,selected_ids=ids,prior_ids=prior['accepted_ids'],excluded_other_terminal_ids=[r['id'] for r in rows if r['id'] not in prior['accepted_ids']+ids])
 put('FROZEN_OBSERVATION.json',frozen)
 put('EXECUTION_BINDINGS.json',dict(status='ROOT_EXPLICIT_ACTUAL_METADATA_BINDING_NOT_SCIENTIFIC_ACCEPTANCE',execution_authorized=True,source_seal_sha256=a.source_seal_sha256,source_review_path=str(a.source_review),source_review_sha256=a.source_review_sha256,snapshot_path=str(a.snapshot),snapshot_sha256=a.snapshot_sha256,frozen_observation_sha256=sha(H/'FROZEN_OBSERVATION.json'),allowed_ids=ids))
 print(json.dumps(dict(execution_bindings_sha256=sha(H/'EXECUTION_BINDINGS.json'),fixed_ids=ids,scientific_execution=False)))
if __name__=='__main__':main()
