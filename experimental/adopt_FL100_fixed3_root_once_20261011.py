"""Adopt the fixed three original-Linux plus zero-fit offserver records only."""
from pathlib import Path,PurePosixPath
import datetime,hashlib,json,sys,tarfile
R=Path(__file__).resolve().parents[1]
H=R/'tmp/FL100_cuda3_saved_closure_prepare_20261011'
D=H/'root_adoption'
sys.path.insert(0,str(R/'tmp'))
from guardfed_local_storage import check_bulk_storage
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
assert __debug__ and not D.exists()
volume=check_bulk_storage()
seal=H/'FILES_SHA256.json'
assert sha(seal)=='88d9b7d39df99fa1da79cc403e6009deef67f92c2f2691ad3f133fc7b7e35ac8'
for rel,pin in read(seal)['files'].items():
    p=H/rel;assert Path(rel).name==rel and sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
audit_path=H/'actual/SAVED_OUTPUTS_AUDIT.json'
transport_path=H/'OLD800_FIXED3_ATTEMPT1_TRANSPORT_VERIFICATION.json'
assert sha(audit_path)=='82a1bde7dde42468fea64fcb4016b77050fefee4a38188f16110de745179e383'
assert sha(transport_path)=='3eb4ea427fef25d5695acc808c23397aa579cdc1987f4d3f363975cd7f43c913'
audit=read(audit_path);t=read(transport_path)
inp_path=R/'tmp/FL100_cuda3_saved_closure_prepare_20261011/ACTUAL_AUDIT_INPUTS.json'
assert sha(inp_path)==audit['inputs_sha256']=='297803d2fb384ff72c217cd279884a7b67cddca99b41968adf2e18af23ab4da8'
inp=read(inp_path);ids=inp['exact_ids']
assert audit['status']=='FL100_FIXED3_CUDA_SAVED_OUTPUTS_AUDIT_NO_REFIT_PASS_NOT_ROOT_ADOPTED'
assert audit['exact_ids']==ids and len(ids)==len(set(ids))==3
assert (audit['metric_values_checked'],audit['integer_base_counts_checked'],audit['prediction_rules_checked'])==(27,72,9)
assert audit['fit_calls']==audit['new_CNN']==audit['new_training']==0 and not audit['test']
assert t['status']=='PASS_ARCHIVE_AND_ALL_MEMBER_SHA' and t['all_members_verified']
archive=Path(t['archive_path']);dest=Path(t['verified_extract'])
assert all(p.resolve().is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()) for p in (archive,dest))
assert sha(archive)==audit['archive_sha256']==t['archive_sha256']=='35f9aa7009789d4992bacbc79d451f1b344a42f81db49d50ee251695f61bcec1'
with tarfile.open(archive,'r:gz') as tar:
    ms=tar.getmembers();assert len(ms)==len({m.name for m in ms})==t['archive_members']==77
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
    i=row['id'];producer='base'
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
D.mkdir()
proof={'status':'ROOT_FIXED3_FL100_CUDA_ORIGINAL_LINUX_WHOLE_AND_OFFSERVER_SAVED_AUDIT_ADOPTED',
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ids':ids,'accepted':3,'records':records,
    'original_Linux_saved_gate':'whole original root fit/prediction/metrics/native1e-12 accepted',
    'Windows_audit':'zero-fit saved predictions/counts/metrics; not whole refit',
    'metric_values_checked':27,'integer_base_counts_checked':72,'prediction_rules_checked':9,
    'audit':{'path':str(audit_path),'sha256':sha(audit_path)},'transport':{'path':str(transport_path),'sha256':sha(transport_path)},
    'archive':{'path':str(archive),'sha256':sha(archive),'members':77,'bytes':archive.stat().st_size},
    'archive_all_members_rehashed':True,'source_seal_sha256':sha(seal),'fresh_F_volume':volume,
    'record_producers':{i:'base' for i in ids},'cohort_sha256':audit['cohort_sha256'],
    'original_Windows_CPU_failure_preserved':True,
    'whole100_complete':False,'whole_comparison_complete':False,'Full_repeated':False,
    'new_training':False,'parameters_changed':False,'tolerance_changed':False,'test':False,'old_CPU_outputs_preserved':True}
target=D/'ROOT_SCIENTIFIC_ADOPTION.json'
with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps({'accepted':3,'root':str(target),'sha256':sha(target)}))
