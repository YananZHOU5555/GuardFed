"""Guard F then hash exact old seven-role evidence; do not deserialize arrays/pickle."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(n,v):(HERE/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
v=subprocess.run(['powershell','-NoProfile','-Command',"Get-Volume -DriveLetter F | Select-Object DriveLetter,FileSystemLabel,HealthStatus,OperationalStatus,Size,SizeRemaining,UniqueId | ConvertTo-Json -Depth 4"],capture_output=True,text=True,check=True)
volume=json.loads(v.stdout);assert volume['FileSystemLabel']=='Yanan 2TB' and volume['HealthStatus']=='Healthy' and volume['OperationalStatus']=='OK'
write('ACTUAL_F_GUARD.json',{'utc':datetime.now(timezone.utc).isoformat(),'volume':volume,'purpose':'read and hash frozen LoGo evidence only'})
cohort=ROOT/'tmp/LoGo100_uniform_cuda_cache_adapter_prepare_20261011/EXACT100.json'
assert sha(cohort)=='52d6a15e077eb50b1d81d2aa0e481c6e2e5fb5dfeea24e84ba6036535dffed9b'
scope=json.loads(cohort.read_bytes());records=[];members={};errors=[]
def member(path,expected,relative):
    p=Path(path);key=str(p)
    if key not in members:
        st=p.stat();actual=sha(p);after=p.stat()
        assert (st.st_size,st.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Source changed while reading'
        assert actual==expected,'SHA mismatch '+str(p)
        members[key]={'original_path':str(p),'sha256':actual,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'relative_destination':relative}
    assert members[key]['sha256']==expected and members[key]['relative_destination']==relative
    return {'original_path':str(p),'sha256':expected,'relative_destination':relative}
for r in scope['records']:
    row={'post_id':r['post_id'],'baseline_id':r['baseline_id'],'reused_screen_record':r['reused_screen_record'],'artifacts':{},'source_members':{}}
    kind='screen' if r['reused_screen_record'] else 'fullcoverage'
    try:
        for role,pin in r['original_artifacts'].items():
            name={'job':'job.json','acceptance':'acceptance.json'}.get(role,role)
            row['artifacts'][role]=member(pin['original_path'],pin['sha256'],f'original/{kind}/records/{r["post_id"]}/{name}')
        job=json.loads(Path(r['original_artifacts']['job']['original_path']).read_bytes())
        assert job['id']==r['post_id'] and set(job['local_hashes'])=={'bridge.py','prepare_reuse.py','protocol.json','reuse_manifest.json'}
        snapshot=ROOT/'tmp/celeba_logofair_screen32_20261010/snapshot' if kind=='screen' else Path('F:/YananResearchStorage/GuardFed/logofair_fullcoverage100_20261010/stage001/snapshot')
        for name,h in job['local_hashes'].items():
            rel='logofair_bridge_20261010/'+name
            row['source_members'][rel]=member(snapshot/rel,h,f'original/{kind}/snapshot/{rel}')
        for rel,h in [('logofair/adapter.py','8ac65ac38812b90ab44c71284d19184fe55f120e9d44262841ca9d925c26a5a7'),('logofair/upstream/fedlearn/models/FedFairPostClient.py','ab6bbbfb0a8d64ea99f24e890f0a5959aad2d223f0e7b2e501e1042db0eae160')]:
            row['source_members'][rel]=member(snapshot/rel,h,f'original/{kind}/snapshot/{rel}')
        row['status']='HASHED_ORIGINAL_SEVEN_ROLES_AND_SOURCE'
    except Exception as exc:row['status']='FAILED';row['error']=repr(exc);errors.append({'post_id':r['post_id'],'error':repr(exc)})
    records.append(row)
out={'status':'PASS_ALL_ORIGINAL_HASHES_NOT_TRANSFERRED' if not errors else 'INCOMPLETE_ORIGINAL_INVENTORY','utc':datetime.now(timezone.utc).isoformat(),'cohort_sha256':sha(cohort),'F_guard_sha256':sha(HERE/'ACTUAL_F_GUARD.json'),'records':records,'members':list(members.values()),'record_count':len(records),'verified_records':sum(r['status'].startswith('HASHED_') for r in records),'distinct_members':len(members),'total_payload_bytes':sum(m['bytes'] for m in members.values()),'errors':errors,'pickle_or_array_deserialization':False,'transferred':False}
write('ACTUAL_ORIGINAL_INVENTORY.json',out)
print(json.dumps({k:out[k] for k in ['status','record_count','verified_records','distinct_members','total_payload_bytes','errors']}))
