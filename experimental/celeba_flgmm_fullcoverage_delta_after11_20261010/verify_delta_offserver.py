"""Existing offserver member-SHA + unchanged strict terminal checker, parameterized for96-new."""
from pathlib import Path,PurePosixPath
import argparse,datetime,hashlib,json,os,platform,sys,tarfile,traceback
if sys.flags.optimize:raise RuntimeError('Optimized Python forbidden')
PACKAGE='6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230'
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
def execute(a):
 batch=a.batch.resolve();release=a.release.resolve();out=batch/'restored'
 receipt=read(batch/'BACKUP_SHA256.json');assert sha(batch/'BACKUP_SHA256.json')==a.receipt_sha256
 assert receipt['status']=='PARTIAL_ACCEPTED_SERVER_ARCHIVE_VERIFIED_OFFSERVER_PENDING' and receipt['package_sha256']==PACKAGE
 assert (receipt['planned_new'],receipt['planned_total'],receipt['reused_separately'])==(96,100,4)
 archive=batch/'accepted_delta.tar.gz'
 assert sha(archive)==receipt['archive_sha256'] and archive.stat().st_size==receipt['archive_size']
 assert sha(batch/'MEMBERS.json')==receipt['inventory_sha256'] and sha(batch/'PARTIAL_ACCEPTANCE.json')==receipt['acceptance_sha256']
 acceptance=read(batch/'PARTIAL_ACCEPTANCE.json');expected=read(batch/'MEMBERS.json')['members']
 assert not out.exists() and not out.is_symlink()
 # Manual safe recovery is compatible with Python3.10 and does not buffer all models.
 with tarfile.open(archive) as tar:
  entries=tar.getmembers();names=[m.name for m in entries]
  assert len(names)==len(set(names))==receipt['archived_member_count'] and set(names)==set(expected)|{'MEMBERS.json'}
  assert all(m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts and '\\' not in m.name and ':' not in m.name for m in entries)
  out.mkdir()
  for member in entries:
   target=out.joinpath(*PurePosixPath(member.name).parts);assert target.resolve().is_relative_to(out.resolve())
   pin={'sha256':receipt['inventory_sha256'],'size':(batch/'MEMBERS.json').stat().st_size} if member.name=='MEMBERS.json' else expected[member.name]
   assert member.size==pin['size'];target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as f:
    stream=tar.extractfile(member)
    for chunk in iter(lambda:stream.read(8*1024**2),b''):f.write(chunk)
   assert sha(target)==pin['sha256']
 assert sha(out/'PARTIAL_ACCEPTANCE.json')==receipt['acceptance_sha256']
 assert sha(out/'collect_delta.py')==acceptance['collector_sha256']==read(out/'INPUT_BINDING.json')['collector_sha256']
 previous=read(out/'PREVIOUS_CHAIN.json');assert sha(out/'PREVIOUS_CHAIN.json')==receipt['previous_chain_sha256']
 ids=receipt['accepted_new_ids'];prior=previous['accepted_job_ids']
 assert len(ids)==len(set(ids))==receipt['accepted_new'] and len(prior)==len(set(prior)) and not set(ids)&set(prior)
 assert receipt['accepted_total']==len(prior)+len(ids)<=96 and acceptance['accepted_job_ids']==prior+ids
 assert acceptance['accepted_new_ids']==ids and {r['id'] for r in acceptance['records']}==set(ids) and len(acceptance['records'])==len(ids)
 assert sha(release/'PACKAGE_SHA256.json')==PACKAGE
 assert sha(release/'source/accept_result.py')==acceptance['original_checker_sha256'] and sha(release/'screen_common.py')==acceptance['original_screen_common_sha256']
 sys.path.insert(0,str(release));from screen_common import local_identity,accepted
 protocol,manifest=local_identity();assert set(prior+ids)<={x['id'] for x in manifest['jobs']}
 records=[]
 for identity in ids:
  item=next(x for x in manifest['jobs'] if x['id']==identity)
  result=accepted(item,out/'runs'/identity);assert result is not None and result['rounds']==70
  assert result['config']['client_alpha']==protocol['distributions'][result['distribution']]
  remote=next(x for x in acceptance['records'] if x['id']==identity)
  for key in ('seed','distribution','attack','rounds','metrics','evaluation_stats'):assert result[key]==remote[key],(identity,key)
  assert sha(out/'runs'/identity/'model.pt')==remote['checkpoint_sha256']
  assert result['revision_job']['source_hashes']==remote['source_hashes'] and result['revision_job']['adapter_source_hashes']==remote['adapter_source_hashes']
  records.append(remote)
 assert acceptance['source_host']!=platform.node() and acceptance['before_source_data']==acceptance['after_source_data']
 import torch
 proof={'status':'PARTIAL_ACCEPTED_OFFSERVER_VERIFIED','accepted_new':len(ids),'accepted_total':len(prior)+len(ids),'accepted_job_ids':prior+ids,
  'new_ids':ids,'records':records,'planned_new':96,'planned_total':100,'reused_separately':4,'package_sha256':PACKAGE,
  'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_sha256':receipt['archive_sha256'],'archived_member_count':len(names),
  'inventory_sha256':receipt['inventory_sha256'],'server_backup_receipt_sha256':a.receipt_sha256,'previous_chain_sha256':receipt['previous_chain_sha256'],
  'original_checked_result_replayed_locally':True,'source_host':acceptance['source_host'],'verification_host':platform.node(),
  'verification_runtime':{'python':sys.version,'torch':torch.__version__,'cuda_initialized':torch.cuda.is_initialized()},
  'training_runtime_unchanged':True,'source_data_rehashed_on_server_only':True,'no_old_models_repackaged':True,'CNN_calls':0,'final_test':False,'canonical_ledger_not_modified':True}
 save(batch/'OFFSERVER_ACCEPTANCE.json',proof)
 print(json.dumps({'status':proof['status'],'proof_sha256':sha(batch/'OFFSERVER_ACCEPTANCE.json'),'accepted_new':len(ids),'accepted_total_new':len(prior)+len(ids)}))
def main():
 p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--release',type=Path,required=True);p.add_argument('--receipt-sha256',required=True);a=p.parse_args()
 try:execute(a)
 except BaseException as e:
  save(a.batch/('OFFSERVER_FAILURE_'+str(__import__('time').time_ns())+'.json'),{'error':repr(e),'traceback':traceback.format_exc(),'automatic_retry':False});raise
if __name__=='__main__':main()
