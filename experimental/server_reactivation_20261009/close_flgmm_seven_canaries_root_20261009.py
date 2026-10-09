"""One-shot terminal seven-canary backup and saved-tensor verification; no retry."""
from pathlib import Path
import argparse,base64,datetime,hashlib,json,os,subprocess,sys,traceback
if sys.flags.optimize:raise RuntimeError('Optimized Python forbidden')
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tmp/celeba_flgmm_seven_canary_closure_20261009'
REMOTE='/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/seven_canary_closure'
SEAL='6e7acd16d2c57e11ad236de45235e7ff78b0b9633b0858bd2023fce209181d61'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--terminal-snapshot',type=Path,required=True);p.add_argument('--terminal-sha256',required=True);a=p.parse_args()
    assert sha(a.terminal_snapshot)==a.terminal_sha256
    live=read(a.terminal_snapshot)
    assert live['canary_service']['stdout'].split()[1]=='EXITED' and not live['processes']
    assert live['source_members_match'] and not live['failure_paths'] and not live['recent_log_error_matches']
    assert sum(bool(r['result_present']) for r in live['rows'] if r['kind']!='new')==7
    assert sha(SOURCE/'FILES_SHA256.json')==SEAL
    files=read(SOURCE/'FILES_SHA256.json')['files'];assert len(files)==6
    for n,pin in files.items():assert sha(SOURCE/n)==pin['sha256'] and (SOURCE/n).stat().st_size==pin['bytes']
    assert read(SOURCE/'SOURCE_CHECK.json')['status'].startswith('PASS')
    marker=SOURCE/'ROOT_CLOSURE_ATTEMPT.json';assert not marker.exists(),'Existing attempt preserved; inspect, never blindly repeat'
    attempt=SOURCE/('actual_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));attempt.mkdir()
    save(marker,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),attempt_path=attempt.relative_to(ROOT).as_posix(),terminal_path=str(a.terminal_snapshot),terminal_sha256=a.terminal_sha256,source_seal_sha256=SEAL))
    paths={n:SOURCE/n for n in files};paths['FILES_SHA256.json']=SOURCE/'FILES_SHA256.json'
    payload={n:base64.b64encode(v.read_bytes()).decode() for n,v in paths.items()}
    pins={n:sha(v) for n,v in paths.items()}
    code="""from pathlib import Path
import base64,hashlib,json,os,subprocess
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
target=Path(%r);assert not target.exists() and target.parent.resolve()==target.parent
payload=json.loads(%r);pins=json.loads(%r);assert set(payload)==set(pins)
decoded={}
for n,b in payload.items():
 assert len(Path(n).parts)==1
 data=base64.b64decode(b,validate=True);assert hashlib.sha256(data).hexdigest()==pins[n];decoded[n]=data
target.mkdir()
for n,data in decoded.items():
 with (target/n).open('xb') as f:f.write(data)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
cmd=['taskset','-c','106','ionice','-c','3','nice','-n','10','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(target/'collect_once.py'),'--out',str(target/'output'),'--source-seal-sha256',%r]
r=subprocess.run(cmd,capture_output=True,text=True,env=env)
print(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)))
"""%(REMOTE,json.dumps(payload),json.dumps(pins),SEAL)
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55']
    try:
        result=subprocess.run(ssh+['python -B -'],input=code.encode(),capture_output=True,timeout=240)
        save(attempt/'REMOTE_COMMAND.json',dict(returncode=result.returncode,stdout=result.stdout.decode(errors='replace'),stderr=result.stderr.decode(errors='replace')))
        result.check_returncode();remote=json.loads(result.stdout);assert remote['returncode']==0,remote
        receipt=json.loads(remote['stdout']);assert receipt['status']=='REMOTE_STRICT_BACKUP_READY_NOT_OFFSERVER'
        for n in ('seven_canaries.tar.gz','BACKUP_RECEIPT.json','MEMBERS.json'):
            subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15','-P','60350','root@89.22.197.55:'+REMOTE+'/output/'+n,str(attempt/n)],check=True,timeout=180)
        assert read(attempt/'BACKUP_RECEIPT.json')==receipt and sha(attempt/'seven_canaries.tar.gz')==receipt['archive_sha256']
        verified=subprocess.run([sys.executable,'-B',str(SOURCE/'restore_verify.py'),'--archive',str(attempt/'seven_canaries.tar.gz'),'--receipt',str(attempt/'BACKUP_RECEIPT.json'),'--receipt-sha256',sha(attempt/'BACKUP_RECEIPT.json'),'--out',str(attempt/'verified')],capture_output=True,timeout=180)
        save(attempt/'LOCAL_VERIFICATION_COMMAND.json',dict(returncode=verified.returncode,stdout=verified.stdout.decode(errors='replace'),stderr=verified.stderr.decode(errors='replace')))
        verified.check_returncode()
        print(json.dumps(dict(attempt=str(attempt),receipt_sha256=sha(attempt/'BACKUP_RECEIPT.json'),offserver_sha256=sha(attempt/'verified/OFFSERVER_VERIFICATION.json'),archive_sha256=sha(attempt/'seven_canaries.tar.gz'))))
    except BaseException as e:
        save(attempt/'ROOT_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False,inspect_remote_before_recovery=True));raise
if __name__=='__main__':main()
