"""Future root-authorized receiver: exact original bytes only; never executed in preparation."""
from pathlib import Path,PurePosixPath
import argparse,base64,hashlib,json,shutil,sys,tarfile
def need(ok,msg):
    if not ok:raise ValueError(msg)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--approval-base64',required=True);p.add_argument('--approval-sha256',required=True);a=p.parse_args()
    raw=base64.b64decode(a.approval_base64,validate=True);need(digest(raw)==a.approval_sha256,'Root transfer approval changed');approval=json.loads(raw)
    need(digest(Path('/etc/vast-agents-guide.md').read_bytes())=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa','Guide changed')
    need(approval['status']=='ROOT_AUTHORIZED_ORIGINAL_LOGO_ARTIFACT_TRANSFER_ONLY' and approval['execution_authorized'] is True,'Template is not transfer authorization')
    destination=Path(approval['server_destination'])
    need(destination.is_absolute() and destination.parent==Path('/workspace/guardfed_checks') and destination.name.startswith('uniform_cuda_LoGo100_artifacts_') and '..' not in destination.parts and destination.resolve()==destination and not destination.exists(),'Fresh exact destination required')
    need(shutil.disk_usage(destination.parent).free>approval['payload_bytes']+1024**3,'Insufficient target space')
    with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:
        iterator=iter(archive);first=next(iterator);need(first.name=='ROOT_TRANSFER_INDEX.json' and first.isfile() and first.size<2*1024**2,'Exact compact first index required')
        payload=archive.extractfile(first).read();need(digest(payload)==approval['inventory_sha256'],'Changed exact inventory')
        inventory=json.loads(payload);need(inventory['status']=='PASS_ALL_ORIGINAL_HASHES_NOT_TRANSFERRED' and inventory['verified_records']==100,'Incomplete original inventory')
        expected={v['relative_destination']:v for v in inventory['members']}
        need(len(expected)==approval['payload_members']==712 and sum(v['bytes'] for v in expected.values())==approval['payload_bytes'],'Changed payload scope')
        destination.mkdir();(destination/'ROOT_TRANSFER_INDEX.json').write_bytes(payload)
        (destination/'ROOT_TRANSFER_APPROVAL.json').write_bytes(raw)
        actual={}
        for item in iterator:
            rel=PurePosixPath(item.name)
            need(not rel.is_absolute() and '..' not in rel.parts and item.name in expected and item.name not in actual and item.isfile(),'Foreign/duplicate/link archive member')
            row=expected[item.name];need(item.size==row['bytes'],'Changed file length')
            output=destination/item.name;need(output.resolve().is_relative_to(destination),'Escaped destination')
            output.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();total=0
            with archive.extractfile(item) as src,output.open('xb') as dst:
                for block in iter(lambda:src.read(8*1024**2),b''):dst.write(block);h.update(block);total+=len(block)
            need(total==row['bytes'] and h.hexdigest()==row['sha256'],'Received original SHA mismatch')
            # Rehash the actual received file, not only the stream.
            h2=hashlib.sha256()
            with output.open('rb') as f:
                for block in iter(lambda:f.read(8*1024**2),b''):h2.update(block)
            need(h2.hexdigest()==row['sha256'],'Disk rehash mismatch')
            actual[item.name]={'sha256':h2.hexdigest(),'bytes':total}
        need(set(actual)==set(expected),'Incomplete payload')
    receipt={'status':'ALL_ORIGINAL_MEMBERS_RECEIVED_AND_REHASHED_NOT_ROOT_ADOPTED','inventory_sha256':approval['inventory_sha256'],'approval_sha256':a.approval_sha256,'destination':str(destination),'members':actual,'member_count':len(actual),'total_payload_bytes':sum(r['bytes'] for r in actual.values()),'original_bytes_preserved':True,'science_executed':False,'root_adopted':False}
    path=destination/'TRANSPORT_VERIFICATION.json'
    with path.open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':receipt['status'],'receipt':str(path),'sha256':digest(path.read_bytes()),'member_count':len(actual),'root_adopted':False}))
if __name__=='__main__':main()
