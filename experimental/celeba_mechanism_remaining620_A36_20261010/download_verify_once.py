"""Download only this new export to guarded F, then run the original verifier once."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,tarfile,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'tmp'))
from guardfed_local_storage import STORAGE_ROOT,check_bulk_storage
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
save=lambda p,d:Path(p).write_text(json.dumps(d,indent=2)+'\n',encoding='utf8')
assert not (HERE/'DOWNLOAD_STARTED.json').exists() and not (HERE/'OFFSERVER_COMMAND.json').exists()
receipt=read(HERE/'EXPORT.stdout.json');export=read(HERE/'EXPORT_COMMAND.json');tag=export['tag']
remote_receipt=str(Path(receipt['archive']).with_name('backup_receipt.json')).replace('\\','/')
remote_code="from pathlib import Path\nimport hashlib,json\na=Path("+repr(receipt['archive'])+");r=Path("+repr(remote_receipt)+")\nprint(json.dumps(dict(archive_bytes=a.stat().st_size,archive_sha256=hashlib.sha256(a.read_bytes()).hexdigest(),receipt_bytes=r.stat().st_size,receipt_sha256=hashlib.sha256(r.read_bytes()).hexdigest(),receipt=json.loads(r.read_bytes()))))\n"
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 111 nice -n 10 ionice -c 3 python -B -']
r=subprocess.run(cmd,input=remote_code.encode(),capture_output=True)
save(HERE/'REMOTE_ARTIFACTS_COMMAND.json',dict(argv=cmd,code=remote_code,returncode=r.returncode,automatic_retry=False))
(HERE/'REMOTE_ARTIFACTS.stdout.json').write_bytes(r.stdout);(HERE/'REMOTE_ARTIFACTS.stderr.txt').write_bytes(r.stderr);r.check_returncode()
meta=json.loads(r.stdout);assert meta['receipt']==receipt and meta['archive_sha256']==receipt['archive_sha256']
volume=check_bulk_storage(meta['archive_bytes']*3+meta['receipt_bytes'])
dest=STORAGE_ROOT/'mechanism_remaining620_A36_20261010'/tag;dest.mkdir(parents=True,exist_ok=False)
save(HERE/'DOWNLOAD_STARTED.json',dict(destination=str(dest),fresh_F_volume=volume,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),automatic_retry=False))
scp=['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15','-P','60350','root@89.22.197.55:'+receipt['archive'],'root@89.22.197.55:'+remote_receipt,str(dest)]
save(HERE/'SCP_COMMAND.json',dict(argv=scp,automatic_retry=False))
print(json.dumps(dict(step='F_only_SCP_started',destination=str(dest),archive_bytes=meta['archive_bytes'])),flush=True)
r=subprocess.run(scp,capture_output=True)
(HERE/'SCP.stdout.txt').write_bytes(r.stdout);(HERE/'SCP.stderr.txt').write_bytes(r.stderr);save(HERE/'SCP_EXIT.json',dict(returncode=r.returncode,automatic_retry=False));r.check_returncode()
archive=dest/'incremental_valid_three_views.tar.gz';local_receipt=dest/'backup_receipt.json'
assert sha(archive)==receipt['archive_sha256'] and sha(local_receipt)==meta['receipt_sha256']
first=read(ROOT/'tmp/celeba_remaining620_A28_transport_20261010/RAW_STORAGE_INDEX.json')
assert sha(first['receipt'])==first['receipt_sha256']=='977c97a1c05ba78bc25b7402fb5c313dcd9c286b140ce4dbf9129bf7bd92aa30'
transport=ROOT/'tmp/celeba_mechanism_remaining_evaluation_transport_20261010'
for dep in read(transport/'INPUTS.json')['dependencies'].values():assert sha(ROOT/dep['local_relative'])==dep['sha256']
cache=ROOT/'tmp/celeba_final_valid_replay_20261009/verification_inputs/original_valid_cache.npz'
assert sha(cache)==read(transport/'INPUTS.json')['valid_cache_sha256']
with tarfile.open(archive,'r:gz') as t:inv=json.load(t.extractfile('backup_inventory.json'))
fresh=check_bulk_storage(sum(x['bytes'] for x in inv['members'].values()))
exe=ROOT/'tmp/celeba_baselines/remaining_20261009/group_a/.venv/Scripts/python.exe'
args=[str(exe),'-B',str(transport/'transport.py'),'verify','--source',str(ROOT/'tmp/celeba_mechanism_remaining_evaluation_v2_20261010'),
 '--source-seal',receipt['source_seal_sha256'],'--transport-seal',receipt['transport_source_seal_sha256'],
 '--archive',str(archive),'--receipt',str(local_receipt),'--receipt-sha256',meta['receipt_sha256'],
 '--previous',first['receipt'],'--previous-sha256',first['receipt_sha256'],'--cache',str(cache),'--out',str(dest/'verification')]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
save(HERE/'OFFSERVER_COMMAND.json',dict(argv=args,automatic_retry=False,fresh_F_volume=fresh,expected_new_ids=receipt['accepted_new_ids']))
print(json.dumps(dict(step='original_offserver_verifier_started_once',archive_sha256=sha(archive))),flush=True)
start=time.monotonic();r=subprocess.run(args,capture_output=True,env=env)
(HERE/'OFFSERVER.stdout.txt').write_bytes(r.stdout);(HERE/'OFFSERVER.stderr.txt').write_bytes(r.stderr)
save(HERE/'OFFSERVER_EXIT.json',dict(returncode=r.returncode,elapsed_seconds=time.monotonic()-start,automatic_retry=False));r.check_returncode()
proof=dest/'verification/OFFSERVER_TRANSPORT_VERIFICATION.json';d=read(proof);saved=d['original_saved_array_verification']
assert d['accepted_offserver']==0 and d['root_adoption_pending'] is True and saved['accepted_n']==8
assert (saved['independent_metric_checks'],saved['independent_confusion_count_checks'],saved['prediction_rule_checks'])==(72,192,24)
save(HERE/'RAW_STORAGE_INDEX.json',dict(status='F_ONLY_NEW8_ARCHIVE_RECEIPT_AND_ORIGINAL_OFFSERVER_VERIFICATION_PASS_PENDING_ROOT',directory=str(dest),
 archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,receipt=str(local_receipt),receipt_sha256=sha(local_receipt),
 offserver_verification=str(proof),offserver_verification_sha256=sha(proof),archive_member_manifest=str(dest/'verification/verified_extract/backup_inventory.json'),
 archive_member_manifest_sha256=sha(dest/'verification/verified_extract/backup_inventory.json'),accepted_new_ids=receipt['accepted_new_ids'],
 all_transported_ids=receipt['all_transported_ids'],previous_local_receipt=first['receipt'],previous_receipt_sha256=first['receipt_sha256'],
 archive_members=d['archive_member_verification']['members_verified'],metrics=72,counts=192,rules=24,accepted_offserver=0,root_adopted=0))
print(json.dumps(dict(step='original_offserver_complete',proof=str(proof),proof_sha256=sha(proof),new=8,cumulative=48,members=d['archive_member_verification']['members_verified'],metrics=72,counts=192,rules=24,accepted_offserver=0)),flush=True)
