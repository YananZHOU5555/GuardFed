"""Adopt the exact FedAvg100 original-Linux plus zero-fit offserver records only."""
from pathlib import Path,PurePosixPath
import datetime,hashlib,json,sys,tarfile
R=Path(__file__).resolve().parents[1]
H=R/'tmp/FedAvg_cuda100_saved_cache_closure_prepare_20261011'
D=H/'root_adoption'
sys.path.insert(0,str(R/'tmp'))
from guardfed_local_storage import check_bulk_storage
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
assert __debug__ and not D.exists()
volume=check_bulk_storage()
seal=H/'FILES_SHA256.json'
assert sha(seal)=='daee68ef4b28a01487b1f57940d23da734e432249abd9840d4c63d0b8deca64c'
for rel,pin in read(seal)['files'].items():
    p=H/rel;assert Path(rel).name==rel and sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
audit_path=H/'actual/SAVED_OUTPUTS_AUDIT.json'
transport_path=H/'OLD800_FIXED3_ATTEMPT1_TRANSPORT_VERIFICATION.json'
assert sha(audit_path)=='daf6118b869f8887cf90d5f3d38826432e471c701a60794edf1d75f63992b23c'
assert sha(transport_path)=='e02c2efb1b1ed84952eeb87ae298690a64b130ad25be61c590d3bc709fb30c74'
audit=read(audit_path);t=read(transport_path)
inp_path=R/'tmp/FedAvg_cuda100_saved_cache_closure_prepare_20261011/ACTUAL_AUDIT_INPUTS.json'
assert sha(inp_path)==audit['inputs_sha256']=='985edcd9998031ff56c14fb3913cff0c6551f72966fa65ce9f11f624e6008048'
inp=read(inp_path);ids=inp['exact_ids']
assert audit['status']=='FEDAVG100_SAVED_OUTPUTS_AUDIT_NO_REFIT_PASS_NOT_ROOT_ADOPTED'
assert audit['exact_ids']==ids and len(ids)==len(set(ids))==100
assert (audit['metric_values_checked'],audit['integer_base_counts_checked'],audit['prediction_rules_checked'])==(900,2400,300)
assert audit['fit_calls']==audit['new_CNN']==audit['new_training']==0 and not audit['test']
assert t['status']=='PASS_ARCHIVE_AND_ALL_MEMBER_SHA' and t['all_members_verified']
archive=Path(t['archive_path']);dest=Path(t['verified_extract'])
assert all(p.resolve().is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()) for p in (archive,dest))
assert sha(archive)==audit['archive_sha256']==t['archive_sha256']=='4b0b6fc9341ae67e698927b1c37469c3e483f8026452565fefe793b4e7244401'
with tarfile.open(archive,'r:gz') as tar:
    ms=tar.getmembers();assert len(ms)==len({m.name for m in ms})==t['archive_members']==1058
    assert {m.name for m in ms}==set(t['members'])|{'ROOT_MEMBERS.json'}
    for m in ms:
        rel=PurePosixPath(m.name)
        assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts and ':' not in m.name and '\\' not in m.name
        data=tar.extractfile(m).read();p=dest.joinpath(*rel.parts)
        pin=t['members'][m.name] if m.name!='ROOT_MEMBERS.json' else {'sha256':t['member_index_sha256'],'bytes':len(data)}
        assert hashlib.sha256(data).hexdigest()==pin['sha256']==sha(p) and len(data)==pin['bytes']==p.stat().st_size
assert [r['id'] for r in audit['records']]==ids
records={}
for row in audit['records']:
    i=row['id'];producer=inp['record_producers'][i]
    task=dest/producer/'tasks'/i;out=dest/producer/'runs'/i
    a=read(task/'ACCEPTED.json');passed=read(out/'formal_saved_acceptance.json')
    assert sha(task/'ACCEPTED.json')==inp['accepted_sha256'][i]==row['accepted_sha256']
    assert a['receipt']['sha256']==row['receipt_sha256'] and a['arrays']['sha256']==row['arrays_sha256']
    assert passed['views']==row['views'] and passed['native_comparison']==row['native_comparison']
    assert passed['native_comparison']['max_abs_difference']==0 and row['original_Linux_whole_saved_accepted']
    assert row['fit_calls']==0 and sha(out/'formal_binding.json')==passed['binding_sha256']
    records[i]={'producer':producer,'acceptance':a['acceptance'],'task_acceptance_sha256':row['accepted_sha256'],
        'receipt_sha256':row['receipt_sha256'],'arrays_sha256':row['arrays_sha256'],'checkpoint_sha256':row['checkpoint_sha256'],
        'views':row['views'],'native_max_abs_difference':0.0}
failure=t['members']['attempt002/canary/QUEUE_FAILURE.json']
assert failure['sha256']==audit['attempt002_failure_sha256']==inp['attempt002_failure']['sha256']
D.mkdir()
proof={'status':'ROOT_FEDAVG100_CUDA_ORIGINAL_LINUX_WHOLE_AND_OFFSERVER_SAVED_AUDIT_ADOPTED',
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ids':ids,'accepted':100,'records':records,
    'original_Linux_saved_gate':'whole original root fit/prediction/metrics/native1e-12 accepted',
    'Windows_audit':'zero-fit saved predictions/counts/metrics; not whole refit',
    'metric_values_checked':900,'integer_base_counts_checked':2400,'prediction_rules_checked':300,
    'audit':{'path':str(audit_path),'sha256':sha(audit_path)},'transport':{'path':str(transport_path),'sha256':sha(transport_path)},
    'archive':{'path':str(archive),'sha256':sha(archive),'members':1058,'bytes':archive.stat().st_size},
    'archive_all_members_rehashed':True,'source_seal_sha256':sha(seal),'fresh_F_volume':volume,
    'record_producers':inp['record_producers'],'cohort_sha256':audit['cohort_sha256'],
    'old_failure_preserved':inp['attempt002_failure'],'historical117708_type':'UNKNOWN',
    'whole800_complete':False,'whole_comparison_complete':False,'Full_repeated':False,
    'new_training':False,'parameters_changed':False,'tolerance_changed':False,'test':False,'test_evaluation':False,'old_CPU_outputs_preserved':True}
target=D/'ROOT_SCIENTIFIC_ADOPTION.json'
with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps({'accepted':100,'root':str(target),'sha256':sha(target)}))
