"""Transport exact3 saved arrays and original metadata only, directly to F."""
from pathlib import Path
import argparse, datetime, hashlib, json, shlex, subprocess, zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tmp/celeba_added_cnn_exact3_root_execution_20261010'
DEST=Path('F:/YananResearchStorage/GuardFed/added_cnn_exact3_valid_20261010/attempt001')
IDS=['FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91005_fullcoverage','CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91002_fullcoverage','FedNGA_eta0.01_non-IID_Benign_seed91001_screen']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify-existing',action='store_true');a=ap.parse_args()
    vol=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command','Get-Volume -DriveLetter F | Select-Object DriveLetter,FileSystemLabel,HealthStatus,SizeRemaining | ConvertTo-Json -Compress'],text=True))
    assert vol['FileSystemLabel']=='Yanan 2TB' and vol['HealthStatus']=='Healthy' and vol['SizeRemaining']>1024**3+20_000_000
    assert (DEST.exists() if a.verify_existing else not DEST.exists()),'Preserve exact attempt; explicit verification of existing archive only'
    assert not (OUT/'TRANSPORT_VERIFICATION.json').exists(),'Do not repeat accepted transport'
    remote=r'''
import datetime,hashlib,io,json,pathlib,subprocess,sys,zipfile
base=pathlib.Path('/workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010')
out=base/'outputs/attempt001'
assert not (out/'FAILURE.json').exists()
gate=json.loads((out/'GATE_RESULT.json').read_text())
ids=['FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91005_fullcoverage','CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91002_fullcoverage','FedNGA_eta0.01_non-IID_Benign_seed91001_screen']
assert gate['status']=='EXACT3_VALID_INTERFACE_PASS_NOT_ROOT_ADOPTED' and [r['id'] for r in gate['receipts']]==ids
assert gate['package_sha256']=='49c42b90214eae57d456f8ded1d2a4ff2de2762a78d6bcf5c4ae1a5766de9434'
status=subprocess.run(['supervisorctl','status','guardfed_added_cnn_exact3_gate'],capture_output=True,text=True)
assert 'EXITED' in status.stdout and 'RUNNING' not in status.stdout,status.stdout
files={'bundle/GATE_RESULT.json':out/'GATE_RESULT.json','bundle/metadata_receipt.json':out/'metadata_receipt.json','metadata.npz':pathlib.Path('/workspace/GuardFed-celeba-expanded/data/celeba/derived/rgb64_v1/metadata.npz')}
for ident in ids:
 for name in ['receipt.json','validation_predictions.npz']:files['bundle/'+ident+'/'+name]=out/ident/name
for name in ['stdout.log','stderr.log']:files['execution/'+name]=base/'execution'/name
observed={}; payload={}
for rel,path in files.items():
 assert path.is_file() and path.stat().st_size<10_000_000
 b=path.read_bytes();h=hashlib.sha256(b).hexdigest();payload[rel]=b
 observed[rel]={'sha256':h,'bytes':len(b),'server_path':str(path),'resolved_path':str(path.resolve())}
assert sum(len(b) for b in payload.values())<20_000_000
assert observed['metadata.npz']['sha256']=='161f8028f1c29ba470afa60cbd9fb54d7bf61b3cec5c525830ad7a3ef7ab2091'
for ident,receipt in zip(ids,gate['receipts']):
 assert json.loads(payload['bundle/'+ident+'/receipt.json'])==receipt
 assert receipt['status']=='NATIVE_VALID_REPLAY_PASS' and receipt['prediction_arrays_sha256']==observed['bundle/'+ident+'/validation_predictions.npz']['sha256']
for rel,path in files.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==observed[rel]['sha256']
report={'status':'EXACT3_SAVED_ARRAY_TRANSPORT_SOURCE_MEMBERS_UNCHANGED_NOT_SCIENTIFIC_ACCEPTANCE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'members':observed,'exact_ids':ids,'service':status.stdout,'test':False,'new_CNN':0,'new_fit':0,'model_or_images_downloaded':False}
bio=io.BytesIO()
with zipfile.ZipFile(bio,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for rel,b in payload.items():z.writestr(rel,b)
 z.writestr('TRANSPORT_MANIFEST.json',json.dumps(report,indent=2)+'\n')
b=bio.getvalue();print(json.dumps({'archive_sha256':hashlib.sha256(b).hexdigest(),'archive_bytes':len(b),'members':len(payload)+1}),file=sys.stderr)
sys.stdout.buffer.write(b)
'''
    (OUT/'transport_remote.py').write_text(remote,encoding='utf-8')
    cmd='env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 taskset -c 110 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B -c '+shlex.quote(remote)
    archive=DEST/'exact3_saved_arrays_and_metadata.zip'
    if not a.verify_existing:
        DEST.mkdir(parents=True,exist_ok=False)
        with archive.open('xb') as f:
            c=subprocess.run(['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=15','root@89.22.197.55',cmd],stdout=f,stderr=subprocess.PIPE,timeout=60)
        (OUT/'TRANSPORT_STDERR.txt').write_bytes(c.stderr)
        save(OUT/'TRANSPORT_SSH_EXIT.json',{'exit':c.returncode})
        assert c.returncode==0,('Preserve failed transport',c.returncode,c.stderr.decode(errors='replace'))
    raw_stderr=(OUT/'TRANSPORT_STDERR.txt').read_text(encoding='utf-8')
    candidates=[json.loads(line) for line in raw_stderr.splitlines() if line.startswith('{')]
    assert len(candidates)==1 and set(candidates[0])=={'archive_sha256','archive_bytes','members'},'Require one exact archive record; retain SSH banner separately'
    remote_report=candidates[0]
    assert sha(archive)==remote_report['archive_sha256'] and archive.stat().st_size==remote_report['archive_bytes']
    expected={'bundle/GATE_RESULT.json','bundle/metadata_receipt.json','metadata.npz','execution/stdout.log','execution/stderr.log'}|{'bundle/'+i+'/'+n for i in IDS for n in ('receipt.json','validation_predictions.npz')}
    extract=DEST/'verified_extract';extract.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(expected)+1 and set(z.namelist())==expected|{'TRANSPORT_MANIFEST.json'}
        manifest=json.loads(z.read('TRANSPORT_MANIFEST.json'));assert set(manifest['members'])==expected
        assert manifest['exact_ids']==IDS and not manifest['test'] and not manifest['model_or_images_downloaded']
        assert z.testzip() is None
        for rel in z.namelist():
            p=extract/rel;assert p.resolve().is_relative_to(extract.resolve()) and not p.exists()
            b=z.read(rel)
            if rel in expected:
                pin=manifest['members'][rel];assert hashlib.sha256(b).hexdigest()==pin['sha256'] and len(b)==pin['bytes']
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    save(OUT/'TRANSPORT_VERIFICATION.json',dict(status='ROOT_EXACT3_F_TRANSPORT_ALL_MEMBERS_SHA_PASS_NOT_SCIENTIFIC_ACCEPTANCE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),storage=vol,archive_path=archive.as_posix(),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,members=manifest['members'],member_count=len(expected)+1,verified_extract=extract.as_posix(),explicit_existing_archive_verification=a.verify_existing,redownload=False,new_CNN=0,new_fit=0,new_training=0,test=False))
    print(json.dumps({'status':'ROOT_EXACT3_F_TRANSPORT_ALL_MEMBERS_SHA_PASS_NOT_SCIENTIFIC_ACCEPTANCE','archive_sha256':sha(archive),'extract':extract.as_posix(),'members':len(expected)+1}))

if __name__=='__main__':main()
