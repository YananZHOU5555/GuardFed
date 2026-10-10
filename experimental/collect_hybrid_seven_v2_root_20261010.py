"""One approved exact-seven original strict archive and direct F-only recovery."""
from pathlib import Path
import base64,datetime,hashlib,json,subprocess,sys
from guardfed_local_storage import STORAGE_ROOT,check_bulk_storage
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tmp/celeba_hybrid_seven_canary_closure_20261010/v2'
HERE=ROOT/'tmp/celeba_hybrid_fullcoverage_root_binding_v3_20261010/seven_canary_collection'
REMOTE='/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v3_20261010/seven_canary_closure'
SEAL='e812256f42fd4493bf1709215fe349b01fdfedecbf149971cec3f2190fde1a03'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55']
def save(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2);f.write('\n')
def remote(script,receipt,timeout=90):
    r=subprocess.run(SSH+['python -B -'],input=script.encode(),capture_output=True,timeout=timeout)
    save(HERE/receipt,dict(returncode=r.returncode,stderr=r.stderr.decode(errors='replace'),stdout=r.stdout.decode(errors='replace')))
    r.check_returncode();return json.loads(r.stdout)
def main():
    assert not HERE.exists();check_bulk_storage();HERE.mkdir()
    assert sha(SOURCE/'FILES_SHA256.json')==SEAL
    review=ROOT/'tmp/celeba_hybrid_seven_canary_closure_v2_delta_review_20261010/REVIEW.json'
    assert sha(review)=='7a3f33833f5e6df6c4652ee563b45b7ce98c9624f6188c6fd0e07c2f2b3f5101'
    reviewed=read(review);assert 'PASS' in reviewed['status']
    pins=read(SOURCE/'FILES_SHA256.json')['files']
    for name,pin in pins.items():assert sha(SOURCE/name)==pin['sha256'] and (SOURCE/name).stat().st_size==pin['bytes']
    save(HERE/'ROOT_SOURCE_REVIEW.json',dict(status='ROOT_REVIEWED_HYBRID7_V2_SINGLE_COLLECTION',
        source_seal_sha256=SEAL,independent_review_sha256=sha(review),
        source_review='Read collect_once.py/restore_verify.py and actual v2 source audit. Exact original7 strict/2pair checks, no CNN, explicit real-runtime saved-record bridge, F-only safe recovery.',
        collector_CPU=107,CUDA_VISIBLE_DEVICES='',gate_completion_required=True,formal96_authorized=False,final_test=False))
    payload={name:base64.b64encode((SOURCE/name).read_bytes()).decode() for name in pins}
    payload['FILES_SHA256.json']=base64.b64encode((SOURCE/'FILES_SHA256.json').read_bytes()).decode()
    deploy='''from pathlib import Path
import base64,hashlib,json,subprocess
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
stage=Path('/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v3_20261010/stage')
assert (stage/'GATE_ACCEPTANCE.json').is_file() and not (stage/'GATE_FAILURE.json').exists()
g=json.loads((stage/'GATE_ACCEPTANCE.json').read_bytes());assert g['status']=='SEVEN_HYBRID_CANARIES_STRICT_PASS_BACKUP_PENDING' and len(g['accepted_ids'])==7 and len(g['pairs'])==2
r=subprocess.run(['supervisorctl','status','guardfed_celeba_hybrid_fullcoverage_canary'],capture_output=True,text=True)
assert r.stdout.split()[1]=='EXITED',r.stdout
target=Path(%r);assert not target.exists() and target.parent.resolve()==target.parent
data=json.loads(%r);decoded={name:base64.b64decode(value,validate=True) for name,value in data.items()}
assert hashlib.sha256(decoded['FILES_SHA256.json']).hexdigest()==%r
pins=json.loads(decoded['FILES_SHA256.json'])['files'];assert set(decoded)==set(pins)|{'FILES_SHA256.json'}
for name,pin in pins.items():
 p=Path(name);assert not p.is_absolute() and '..' not in p.parts
 assert hashlib.sha256(decoded[name]).hexdigest()==pin['sha256'] and len(decoded[name])==pin['bytes']
target.mkdir()
for name,value in decoded.items():
 p=target/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(value)
print(json.dumps(dict(status='SOURCE_DEPLOYED_AFTER_ACTUAL7_REMOTE_CLOSURE',service=r.stdout,source_seal_sha256=sha(target/'FILES_SHA256.json'),gate_sha256=sha(stage/'GATE_ACCEPTANCE.json'))))
'''%(REMOTE,json.dumps(payload),SEAL)
    remote(deploy,'DEPLOY_COMMAND.json')
    command="env CUDA_VISIBLE_DEVICES='' GUARDFED_CPU_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 taskset -c 107 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B "+REMOTE+'/collect_once.py --out '+REMOTE+'/output --source-seal-sha256 '+SEAL
    r=subprocess.run(SSH+[command],capture_output=True,timeout=300)
    save(HERE/'COLLECT_COMMAND.json',dict(command=command,returncode=r.returncode,stdout=r.stdout.decode(errors='replace'),stderr=r.stderr.decode(errors='replace')))
    r.check_returncode();receipt_stdout=json.loads(r.stdout)
    meta=remote("from pathlib import Path\nimport hashlib,json\np=Path(%r)\nprint(json.dumps({n:dict(sha256=hashlib.sha256((p/n).read_bytes()).hexdigest(),bytes=(p/n).stat().st_size) for n in ['seven_canaries.tar.gz','BACKUP_RECEIPT.json','MEMBERS.json']}))\n"%(REMOTE+'/output'),'REMOTE_FILE_IDENTITY.json')
    check_bulk_storage(sum(pin['bytes'] for pin in meta.values()))
    bulk=STORAGE_ROOT/'hybrid_seven_canary_20261010'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    assert not bulk.exists();bulk.mkdir(parents=True)
    save(HERE/'RAW_STORAGE_INDEX.json',dict(bulk_path=bulk.as_posix(),files=meta,source_seal_sha256=SEAL))
    for name,pin in meta.items():
        subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15','-P','60350','root@89.22.197.55:'+REMOTE+'/output/'+name,str(bulk/name)],check=True,timeout=180)
        assert sha(bulk/name)==pin['sha256'] and (bulk/name).stat().st_size==pin['bytes']
    assert read(bulk/'BACKUP_RECEIPT.json')==receipt_stdout
    result=subprocess.run([sys.executable,'-B',str(SOURCE/'restore_verify.py'),'--archive',str(bulk/'seven_canaries.tar.gz'),'--receipt',str(bulk/'BACKUP_RECEIPT.json'),'--receipt-sha256',meta['BACKUP_RECEIPT.json']['sha256'],'--out',str(bulk/'verified')],capture_output=True,timeout=180)
    save(HERE/'RESTORE_COMMAND.json',dict(returncode=result.returncode,stdout=result.stdout.decode(errors='replace'),stderr=result.stderr.decode(errors='replace')))
    result.check_returncode()
    proof=bulk/'verified/OFFSERVER_VERIFICATION.json'
    assert read(proof)['status']=='PASS_FULL_MEMBER_SHA_AND_ORIGINAL_SAVED_COMPARISON'
    save(HERE/'OFFSERVER_HANDOFF.json',dict(status='ACTUAL_HYBRID7_STRICT_OFFSERVER_PASS_ROOT_PENDING',bulk_path=bulk.as_posix(),offserver_sha256=sha(proof),
        receipt_sha256=sha(bulk/'BACKUP_RECEIPT.json'),archive_sha256=sha(bulk/'seven_canaries.tar.gz'),
        archive_members=read(proof)['member_count'],package_sha256=read(proof)['local']['package_sha256'],gate_sha256=read(proof)['local']['gate_sha256'],
        new_scientific70_records=0,formal96_started=False,final_test=False))
    print(json.dumps(read(HERE/'OFFSERVER_HANDOFF.json')))
if __name__=='__main__':main()
