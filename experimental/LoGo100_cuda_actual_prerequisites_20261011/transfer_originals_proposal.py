"""Future exact F/E->SSH stream; no local archive or scientific deserialization."""
from pathlib import Path
import argparse,base64,hashlib,io,json,shlex,subprocess,tarfile
HERE=Path(__file__).resolve().parent
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root-approval',type=Path,required=True);p.add_argument('--root-approval-sha256',required=True);args=p.parse_args()
    raw=args.root_approval.read_bytes();need(hashlib.sha256(raw).hexdigest()==args.root_approval_sha256,'Actual root approval SHA required');auth=json.loads(raw)
    need(auth['execution_authorized'] is True and auth['status']=='ROOT_AUTHORIZED_ORIGINAL_LOGO_ARTIFACT_TRANSFER_ONLY','No transfer authorization')
    inventory=HERE/'ACTUAL_ORIGINAL_INVENTORY.json';receiver=HERE/'receive_originals_proposal.py'
    need(sha(inventory)==auth['inventory_sha256'] and sha(receiver)==auth['receiver_sha256'] and sha(Path(__file__))==auth['sender_sha256'],'Changed source/inventory')
    guard=subprocess.run(['powershell','-NoProfile','-Command',"Get-Volume -DriveLetter F | Select-Object FileSystemLabel,HealthStatus,OperationalStatus,SizeRemaining,UniqueId | ConvertTo-Json"],capture_output=True,text=True,check=True)
    volume=json.loads(guard.stdout);need(volume['FileSystemLabel']=='Yanan 2TB' and volume['HealthStatus']=='Healthy' and volume['OperationalStatus']=='OK' and volume['UniqueId']==auth['F_volume_unique_id'],'Wrong/offline F physical volume')
    index=json.loads(inventory.read_bytes());need(index['verified_records']==100 and len(index['members'])==auth['payload_members'] and index['total_payload_bytes']==auth['payload_bytes'],'Changed original scope')
    # Source delivered as -c, while stdin remains the raw tar stream. shlex quotes
    # exact UTF-8 source and base64 approval; no shell interpolation of paths.
    command='env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ionice -c 3 python3 -B -c '+shlex.quote(receiver.read_text(encoding='utf8'))+' --approval-base64 '+shlex.quote(base64.b64encode(raw).decode())+' --approval-sha256 '+shlex.quote(args.root_approval_sha256)
    process=subprocess.Popen(['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=15','root@89.22.197.55',command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdin,mode='w|',format=tarfile.PAX_FORMAT) as archive:
            rawindex=inventory.read_bytes();entry=tarfile.TarInfo('ROOT_TRANSFER_INDEX.json');entry.size=len(rawindex);entry.mode=0o600;archive.addfile(entry,io.BytesIO(rawindex))
            for row in index['members']:
                source=Path(row['original_path']);need(source.stat().st_size==row['bytes'] and sha(source)==row['sha256'],'Original changed before stream')
                entry=tarfile.TarInfo(row['relative_destination']);entry.size=row['bytes'];entry.mode=0o600
                with source.open('rb') as f:archive.addfile(entry,f)
        process.stdin.close();process.stdin=None
        stdout,stderr=process.communicate(timeout=600)
        # Compact logs only; preserve any partial remote destination on failure.
        for name,data in [('TRANSFER.stdout',stdout),('TRANSFER.stderr',stderr)]:
            with (HERE/name).open('xb') as f:f.write(data)
        result={'returncode':process.returncode,'root_approval_sha256':args.root_approval_sha256,'inventory_sha256':auth['inventory_sha256'],'sender_sha256':sha(Path(__file__)),'receiver_sha256':sha(receiver),'stdout_sha256':hashlib.sha256(stdout).hexdigest(),'stderr_sha256':hashlib.sha256(stderr).hexdigest(),'automatic_retry':False}
        with (HERE/'ACTUAL_TRANSFER_COMMAND.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
        need(process.returncode==0,'Transfer failed; preserve remote partial evidence, no retry');print(stdout.decode())
    except BaseException:
        if process.poll() is None:process.kill();process.wait()
        raise
if __name__=='__main__':main()
