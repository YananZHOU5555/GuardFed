"""Root-only one-shot transport; original exact3 ZIP/member checks, exact47 scope."""
from pathlib import Path
import argparse, hashlib, json, shlex, subprocess, zipfile
from contract import HERE, IDS, digest_arg, fpath, gate_proof, linux_proof, need, read, save, sha, utc, volume

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gate-result-sha256',required=True,type=digest_arg)
    p.add_argument('--linux-proof-sha256',required=True,type=digest_arg)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--verify-existing',action='store_true',help='Read original partial transport only; never redownload')
    a=p.parse_args()
    need(__debug__,'Do not use -O')
    vol=volume(1024**3+2*128_000_000)
    dest=fpath(a.destination);out=a.report_dir.resolve()
    need(out!=HERE and not out.is_relative_to(HERE),'Prepared source stays immutable')
    out.mkdir(parents=True,exist_ok=True)
    need(not (out/'TRANSPORT_VERIFICATION.json').exists(),'Do not repeat accepted transport')
    need(dest.exists() if a.verify_existing else not dest.exists(),'Preserve existing attempt; no automatic retry')
    remote=(HERE/'transport_remote.py').read_text(encoding='utf-8')
    for marker,value in [('__GATE_RESULT_SHA256__',a.gate_result_sha256),('__LINUX_PROOF_SHA256__',a.linux_proof_sha256)]:
        need(remote.count(marker)==(2 if marker=='__GATE_RESULT_SHA256__' else 1),'Template binding differs');remote=remote.replace(marker,value)
    archive=dest/'flgmm47_saved_arrays_and_receipts.zip'
    if not a.verify_existing:
        save(out/'TRANSPORT_COMMAND.json',{'utc':utc(),'CPU':110,'threads':1,'gate_result_sha256':a.gate_result_sha256,'linux_proof_sha256':a.linux_proof_sha256,'remote_source_sha256':hashlib.sha256(remote.encode()).hexdigest(),'destination':dest.as_posix(),'storage':vol})
        (out/'TRANSPORT_REMOTE_BOUND.py').write_text(remote,encoding='utf-8')
        cmd='env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 110 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B -c '+shlex.quote(remote)
        dest.mkdir(parents=True,exist_ok=False)
        try:
            with archive.open('xb') as f:
                c=subprocess.run(['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=15','root@89.22.197.55',cmd],stdout=f,stderr=subprocess.PIPE,timeout=300)
        except subprocess.TimeoutExpired as e:
            (out/'TRANSPORT_STDERR.txt').write_bytes(e.stderr or b'')
            save(out/'TRANSPORT_SSH_EXIT.json',{'exit':None,'timeout':True,'partial_archive_preserved':archive.as_posix(),'retry_authorized':False})
            raise
        (out/'TRANSPORT_STDERR.txt').write_bytes(c.stderr)
        save(out/'TRANSPORT_SSH_EXIT.json',{'exit':c.returncode})
        need(c.returncode==0,'Transport failed; preserve attempt and stop')
    raw_stderr=(out/'TRANSPORT_STDERR.txt').read_text(encoding='utf-8')
    candidates=[json.loads(line) for line in raw_stderr.splitlines() if line.startswith('{')]
    need(len(candidates)==1 and set(candidates[0])=={'archive_sha256','archive_bytes','members'},'Require one exact archive record')
    remote_report=candidates[0]
    need(sha(archive)==remote_report['archive_sha256'] and archive.stat().st_size==remote_report['archive_bytes'],'Archive transport SHA/size differs')
    expected={'bundle/GATE_RESULT.json','bundle/metadata_receipt.json','LINUX_SAVED_CHECK.json'}|{'bundle/'+i+'/'+n for i in IDS for n in ('receipt.json','validation_predictions.npz')}
    need(remote_report['members']==len(expected)+1==98,'Archive layout differs')
    extract=dest/'verified_extract';extract.mkdir()
    with zipfile.ZipFile(archive) as z:
        need(len(z.namelist())==len(expected)+1 and set(z.namelist())==expected|{'TRANSPORT_MANIFEST.json'},'Missing/duplicate/extra ZIP member')
        manifest=json.loads(z.read('TRANSPORT_MANIFEST.json'));need(set(manifest['members'])==expected,'Member inventory differs')
        need(manifest['exact_ids']==IDS and not manifest['test'] and not manifest['model_or_images_downloaded'],'Transport scope differs')
        need(z.testzip() is None,'ZIP integrity failed')
        for rel in z.namelist():
            target=extract/rel;need(target.resolve().is_relative_to(extract.resolve()) and not target.exists(),'Unsafe/existing extraction member')
            b=z.read(rel)
            if rel in expected:
                pin=manifest['members'][rel];need(hashlib.sha256(b).hexdigest()==pin['sha256'] and len(b)==pin['bytes'],'Member SHA/size differs')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
    need(sha(extract/'LINUX_SAVED_CHECK.json')==a.linux_proof_sha256 and sha(extract/'bundle/GATE_RESULT.json')==a.gate_result_sha256,'External gate/Linux proof SHA differs')
    linux_proof(read(extract/'LINUX_SAVED_CHECK.json'));gate_proof(read(extract/'bundle/GATE_RESULT.json'))
    need(read(extract/'LINUX_SAVED_CHECK.json')['gate_result_sha256']==a.gate_result_sha256,'Whole Linux proof belongs to another gate')
    save(out/'TRANSPORT_VERIFICATION.json',dict(status='FLGMM47_F_TRANSPORT_ALL_MEMBERS_SHA_PASS_NOT_SCIENTIFIC_ACCEPTANCE',utc=utc(),storage=vol,archive_path=archive.as_posix(),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,members=manifest['members'],member_count=98,verified_extract=extract.as_posix(),gate_result_sha256=a.gate_result_sha256,linux_proof_sha256=a.linux_proof_sha256,explicit_existing_archive_verification=a.verify_existing,redownload=False,new_CNN=0,new_fit=0,new_training=0,test=False,root_adopted=False))
    print(json.dumps({'status':'TRANSPORT_PASS_NOT_ADOPTED','archive_sha256':sha(archive),'members':98,'proof_sha256':sha(out/'TRANSPORT_VERIFICATION.json')}))

if __name__=='__main__':main()
