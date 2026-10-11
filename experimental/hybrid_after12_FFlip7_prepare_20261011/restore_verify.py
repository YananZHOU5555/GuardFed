"""Safe Python3.10-compatible exclusive recovery, every-member SHA, original saved checks."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,os,sys,tarfile,traceback
if sys.flags.optimize:raise RuntimeError('Optimized Python forbidden')
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from verify_saved import sha,verify,PACKAGE,IDS,DELTA
from storage import guard
def write(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--receipt-sha256',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 assert a.archive.resolve().is_relative_to(Path(DELTA['local_bulk_root']).resolve()) and a.out.resolve().is_relative_to(Path(DELTA['local_bulk_root']).resolve())
 storage=guard(a.archive,a.out)
 assert sha(a.receipt)==a.receipt_sha256
 receipt=json.loads(a.receipt.read_bytes())
 assert receipt['status']=='REMOTE_STRICT_BACKUP_READY_NOT_OFFSERVER' and receipt['package_sha256']==PACKAGE
 assert receipt['scope']=='fixed_delta_70round_valid_only' and receipt['formal_table_samples']==len(IDS) and receipt['accepted_new_ids']==IDS
 assert sha(Path(__file__).resolve().parent/'DELTA_SCOPE.json')==receipt['delta_scope_sha256']
 assert sha(a.archive)==receipt['archive_sha256']
 assert not a.out.exists() and not a.out.is_symlink();a.out.mkdir()
 try:
  with tarfile.open(a.archive,'r:gz') as tar:
   members=tar.getmembers();names=[m.name for m in members]
   assert len(names)==len(set(names))==receipt['archive_members'] and all(m.isfile() for m in members)
   assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts and '\\' not in n and ':' not in n for n in names)
   raw=tar.extractfile('MEMBERS.json').read();assert hashlib.sha256(raw).hexdigest()==receipt['inventory_sha256']
   inventory=json.loads(raw);assert inventory['package_sha256']==PACKAGE and inventory['scope']==receipt['scope']
   pins=inventory['files'];assert set(names)==set(pins)|{'MEMBERS.json'} and len(pins)==receipt['content_members']
   for member in members:
    target=a.out.joinpath(*PurePosixPath(member.name).parts)
    assert target.resolve().is_relative_to(a.out.resolve())
    target.parent.mkdir(parents=True,exist_ok=True)
    if member.name!='MEMBERS.json':assert member.size==pins[member.name]['bytes']
    source=tar.extractfile(member)
    with target.open('xb') as output:
     for chunk in iter(lambda:source.read(8*1024**2),b''):output.write(chunk)
    if member.name!='MEMBERS.json':assert sha(target)==pins[member.name]['sha256']
  assert sha(a.out/'closure_source/DELTA_SCOPE.json')==receipt['delta_scope_sha256']
  remote=json.loads((a.out/'evidence/REMOTE_STRICT.json').read_bytes())
  assert remote['status']=='PASS_SAVED_ORIGINAL_FULLCOVERAGE_RECORD' and remote['source_data_before']==remote['source_data_after']
  assert remote['gate_sha256']==receipt['gate_sha256'] and remote['package_sha256']==PACKAGE
  assert remote['server_runtime']['CPU']==[108] and remote['local_CUDA_initialized'] is False
  local=verify(a.out/'stage',remote['server_runtime'],a.out/'legacy_source',offserver=True)
  for key in ('package_sha256','original_checked_source_sha256','gate_sha256','accepted_new','accepted_new_ids','accepted_before','accepted_ids','rounds','formal_table_samples','records'):assert local[key]==remote[key],key
  write(a.out/'OFFSERVER_VERIFICATION.json',{'status':'PASS_FULL_MEMBER_SHA_AND_ORIGINAL_SAVED_COMPARISON','receipt_sha256':a.receipt_sha256,'archive_sha256':sha(a.archive),'inventory_sha256':receipt['inventory_sha256'],'member_count':len(names),'local':local,'remote_runtime':remote['local_verification_runtime'],'training_runtime_unchanged':True,'source_data_rehashed_on_server_only':True,'CNN_calls':0,'formal_table_samples':len(IDS),'accepted_ids':local['accepted_ids'],'accepted_total':len(local['accepted_ids']),'storage_guard':storage})
  print(json.dumps({'status':'PASS','proof_sha256':sha(a.out/'OFFSERVER_VERIFICATION.json'),'archive_sha256':sha(a.archive)}))
 except BaseException as e:
  write(a.out/'LOCAL_FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc(),'automatic_retry':False});raise
if __name__=='__main__':main()
