"""Root adoption of one actual final-six archive; original26 remain referenced."""
from pathlib import Path,PurePosixPath
import argparse,datetime,hashlib,json,tarfile
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_flgmm_final6_closure_20261009';PARENT=ROOT/'tmp/celeba_flgmm_screen_20261009_v2_dispatch'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(p,v):
 with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=Path,required=True);parser.add_argument('--receipt-sha256',required=True);parser.add_argument('--offserver-sha256',required=True);args=parser.parse_args()
 attempt=args.attempt.resolve();assert attempt.parent==BASE.resolve() and attempt.name.startswith('actual_')
 assert not (attempt/'ROOT_ADOPTION_REVIEW.json').exists()
 assert sha(BASE/'FILES_SHA256.json')=='1d7cc11f95727a57478dd8575f65170024b36df98187030ac2de05e1e08cf6b9'
 for name,pin in read(BASE/'FILES_SHA256.json')['files'].items():assert sha(BASE/name)==pin['sha256']
 assert sha(attempt/'BACKUP_SHA256.json')==args.receipt_sha256 and sha(attempt/'OFFSERVER_ACCEPTANCE.json')==args.offserver_sha256
 receipt=read(attempt/'BACKUP_SHA256.json');off=read(attempt/'OFFSERVER_ACCEPTANCE.json');strict=read(attempt/'PARTIAL_ACCEPTANCE.json')
 assert receipt['accepted_new']==off['accepted_new']==strict['accepted_new']==6
 assert receipt['accepted_total']==off['accepted_total']==strict['accepted_total_including_previous']==32
 expected=read(BASE/'EXACT_DELTA.json')['selected_ids'];assert receipt['accepted_new_ids']==strict['accepted_new_ids']==expected
 assert off['status']=='PARTIAL_ACCEPTED_OFFSERVER_VERIFIED' and off['different_host_observed']
 assert sha(PARENT/'LATEST_BACKUP.json')==sha(BASE/'PREVIOUS_LATEST.json')
 latest=read(PARENT/'LATEST_BACKUP.json');prior_path=PARENT/latest['chain_file'];prior=read(prior_path)
 assert sha(prior_path)==receipt['previous_chain_sha256']==off['previous_chain_sha256']==sha(BASE/'PREVIOUS_CHAIN.json')
 assert prior['accepted_total']==26 and not set(prior['accepted_job_ids'])&set(expected)
 manifest={row['id']:row for row in read(BASE/'manifest.json')['jobs']}
 assert set(prior['accepted_job_ids'])|set(expected)==set(manifest) and len(manifest)==32
 assert receipt['package_sha256']==off['package_sha256']=='aec95ceb5e8c9b7aa9cec89e9d70d2700c2f242c29e918792088648d6269bad4'
 assert strict['before_source_data']==strict['after_source_data']==read(BASE/'protocol.json')['source_hashes']
 binding=read(attempt/'EXECUTION_BINDINGS.json');assert binding['status']=='ROOT_AUTHORIZED_FIXED_FINAL6_CLOSURE'
 assert sha(attempt/'AUTHORIZED_SNAPSHOT.json')==binding['authorized_snapshot_sha256']
 snapshot=read(attempt/'AUTHORIZED_SNAPSHOT.json');assert snapshot['queue']['completed']==32 and snapshot['queue']['active']==[] and snapshot['queue']['pending']==0
 archive=attempt/'accepted_final6_delta.tar.gz';assert sha(archive)==receipt['archive_sha256']==off['archive_sha256']
 assert sha(attempt/'MEMBERS.json')==receipt['inventory_sha256']==off['inventory_sha256']
 members=read(attempt/'MEMBERS.json')['members']
 with tarfile.open(archive) as bundle:
  assert len(bundle.getnames())==len(set(bundle.getnames()))==receipt['archived_member_count']==len(members)+1
  assert set(bundle.getnames())==set(members)|{'MEMBERS.json'}
  for item in bundle:
   rel=PurePosixPath(item.name);assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts
   payload=bundle.extractfile(item).read()
   pin=members[item.name] if item.name!='MEMBERS.json' else dict(sha256=sha(attempt/'MEMBERS.json'),size=(attempt/'MEMBERS.json').stat().st_size)
   assert len(payload)==pin['size'] and hashlib.sha256(payload).hexdigest()==pin['sha256']
   assert sha(attempt/'restored'/item.name)==pin['sha256']
 verified={row['id']:row for row in off['records']};assert set(verified)==set(expected)
 assert len(off['records'])==len(strict['records'])==6 and {row['id'] for row in strict['records']}==set(expected)
 for row in strict['records']:
  identity=row['id'];other=verified[identity];result=read(attempt/'restored/runs'/identity/'result.json')
  for key in ('rounds','seed','distribution','attack','metrics','evaluation_stats','checkpoint_sha256'):assert row[key]==other[key]
  assert row['rounds']==70 and row['seed']==91001 and row['job_sha256']==manifest[identity]['job_sha256']
  assert row['metrics']==result['metrics'] and result['config']['celeba_evaluation_split']=='valid'
  assert result['evaluation_stats']['prediction_count']==19867 and result['dataset']=='celeba'
  assert sha(attempt/'restored/runs'/identity/'model.pt')==row['checkpoint_sha256']
 proof=dict(status='ROOT_LOCKED_AUXILIARY_DELTA_ARCHIVE_MEMBER_AND_ORIGINAL_RECORD_ACCEPTANCE_PASS',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  accepted_before=26,accepted_new=6,accepted_total=32,previous_chain_sha256=sha(prior_path),previous_latest_sha256=sha(PARENT/'LATEST_BACKUP.json'),
  archive_sha256=sha(archive),members_verified=len(members)+1,strict_receipt_sha256=sha(attempt/'PARTIAL_ACCEPTANCE.json'),offserver_proof_sha256=sha(attempt/'OFFSERVER_ACCEPTANCE.json'),
  authorized_snapshot_sha256=sha(attempt/'AUTHORIZED_SNAPSHOT.json'),prepared_seal_sha256=sha(BASE/'FILES_SHA256.json'),
  new_inference=0,selection_performed=False,scientific_changes=False,final_test=False,formal100_started=False,negative_results_preserved=True)
 save(attempt/'ROOT_ADOPTION_REVIEW.json',proof)
 chain=dict(status='COMPLETE32_STRICT_OFFSERVER_ROOT_ADOPTED_RECIPE_NOT_YET_SUMMARIZED',checked_utc=proof['checked_utc'],
  accepted_total=32,planned=32,accepted_job_ids=prior['accepted_job_ids']+expected,accepted_new_ids=expected,previous_accepted=26,
  previous_chain_file=latest['chain_file'],previous_chain_sha256=sha(prior_path),delta_dir='../'+BASE.name+'/'+attempt.name,
  archive=archive.name,archive_sha256=sha(archive),inventory_sha256=sha(attempt/'MEMBERS.json'),archive_members=proof['members_verified'],
  server_strict_sha256=proof['strict_receipt_sha256'],offserver_proof_sha256=proof['offserver_proof_sha256'],
  root_adoption_path=(attempt/'ROOT_ADOPTION_REVIEW.json').relative_to(ROOT).as_posix(),root_adoption_sha256=sha(attempt/'ROOT_ADOPTION_REVIEW.json'),
  new_CNN_inference=0,selected_recipe=None,test_evaluated=False,formal100_started=False)
 target=PARENT/('BACKUP_CHAIN_final6_'+attempt.name+'.json');save(target,chain)
 newlatest=dict(chain_file=target.name,chain_sha256=sha(target),accepted=32,planned=32)
 (PARENT/'LATEST_BACKUP.json').write_text(json.dumps(newlatest,indent=2)+'\n',encoding='utf8',newline='\n')
 print(json.dumps(dict(status=proof['status'],root_proof_sha256=sha(attempt/'ROOT_ADOPTION_REVIEW.json'),**newlatest)))
if __name__=='__main__':main()
