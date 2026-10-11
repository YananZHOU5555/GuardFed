"""Root adoption of exact FL100: new97 delta plus unchanged adopted3 F references."""
from pathlib import Path, PurePosixPath
import datetime, hashlib, json, sys, tarfile
R=Path(__file__).resolve().parents[1]
H=R/'tmp/FL100_cuda100_saved_closure_prepare_20261011'
D=H/'root_adoption'
sys.path[:0]=[str(H),str(R/'tmp')]
from guardfed_local_storage import check_bulk_storage
from prior3_overlay import join_prior
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
assert __debug__ and not D.exists()
volume=check_bulk_storage()
seal=H/'FILES_SHA256.json'
assert sha(seal)=='04f93dbb60fd74cbb059d8135f016064f9319a6444edd945582b3424b4394ae4'
for rel,pin in read(seal)['files'].items():
    p=H/rel;assert Path(rel).name==rel and sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
review=H/'ROOT_SOURCE_REVIEW.json'
assert sha(review)=='5a7492bee9e3f68d8c20850391efab1ba1f0b828fde8d885a89518d8d36f4abe'
assert read(review)['source_adoptable'] is True
ap=H/'actual/SAVED_OUTPUTS_AUDIT.json'
tp=H/'FL100_DELTA97_TRANSPORT_VERIFICATION.json'
ip=H/'actual_metadata/ACTUAL_AUDIT_INPUTS.json'
assert sha(ap)=='90ae3f6e1c12378b9158193b6f60c931f4cfcb0f2a715549c1670c561d543345'
assert sha(tp)=='156215db9ba78ab7303ac7a35105ef29697c38c5f2fa3f9f659ee46efb1f8903'
a=read(ap);t=read(tp);inp=read(ip);ids=inp['exact_ids']
assert sha(ip)==a['inputs_sha256']=='fe1652dcf42a5d04543942fbc819c16f2ed7c0cd1265985b1e9151572efdcb81'
assert a['status']=='FL100_ALL100_CUDA_SAVED_OUTPUTS_AUDIT_NO_REFIT_PASS_NOT_ROOT_ADOPTED'
assert ids==a['exact_ids']==read(H/'SELECTED100_INPUTS.json')['ids'] and len(ids)==len(set(ids))==100
assert (a['metric_values_checked'],a['integer_base_counts_checked'],a['prediction_rules_checked'])==(900,2400,300)
assert a['fit_calls']==a['new_CNN']==a['new_training']==0 and not a['test'] and not a['root_adopted']
assert (a['new_records'],a['reused_records'],a['native_max_abs_difference'])==(97,3,0.0)
assert t['status']=='PASS_ARCHIVE_AND_ALL_MEMBER_SHA' and t['all_members_verified']
archive=Path(t['archive_path']);dest=Path(t['verified_extract'])
assert all(p.resolve().is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()) for p in (archive,dest))
assert sha(archive)==a['archive_sha256']==t['archive_sha256']=='d0b35b63737806f37c9e54facc23017f0515bbe8a72671d5b2967e36f672cd8d'
assert archive.stat().st_size==t['archive_bytes']==19179655
with tarfile.open(archive,'r:gz') as tar:
    ms=tar.getmembers();assert len(ms)==len({m.name for m in ms})==t['archive_members']==959
    assert {m.name for m in ms}==set(t['members'])|{'ROOT_MEMBERS.json'}
    for m in ms:
        rel=PurePosixPath(m.name)
        assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts and ':' not in m.name and '\\' not in m.name
        data=tar.extractfile(m).read();p=dest.joinpath(*rel.parts)
        pin=t['members'][m.name] if m.name!='ROOT_MEMBERS.json' else {'sha256':t['member_index_sha256'],'bytes':len(data)}
        assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(dest.resolve())
        assert hashlib.sha256(data).hexdigest()==pin['sha256']==sha(p) and len(data)==pin['bytes']==p.stat().st_size
assert read(dest/'ROOT_MEMBERS.json')['files']==t['members']
paths,combined=join_prior(inp,dest,t['members'])
assert len(combined)==958+28 and len([n for n in t['members'] if n.endswith('validation_predictions.npz')])==97
assert [row['id'] for row in a['records']]==ids
prior=read(inp['prior3_root']['path']);assert a['prior3_root_sha256']==sha(inp['prior3_root']['path'])
records={}
for row in a['records']:
    i=row['id'];task=paths/('base/tasks/'+i);out=paths/('base/runs/'+i)
    accepted=read(task/'ACCEPTED.json');passed=read(out/'formal_saved_acceptance.json')
    assert sha(task/'ACCEPTED.json')==inp['accepted_sha256'][i]==row['accepted_sha256']
    assert accepted['receipt']['sha256']==row['receipt_sha256'] and accepted['arrays']['sha256']==row['arrays_sha256']
    assert passed['views']==row['views'] and passed['native_comparison']==row['native_comparison']
    assert passed['native_comparison']['max_abs_difference']==0 and row['original_Linux_whole_saved_accepted']
    assert row['fit_calls']==0 and sha(out/'formal_binding.json')==passed['binding_sha256']
    records[i]={'producer':'base','acceptance':accepted['acceptance'],'task_acceptance_sha256':row['accepted_sha256'],
        'receipt_sha256':row['receipt_sha256'],'arrays_sha256':row['arrays_sha256'],'checkpoint_sha256':row['checkpoint_sha256'],
        'views':row['views'],'native_max_abs_difference':0.0}
    if i in prior['ids']:assert records[i]==prior['records'][i]
assert set(prior['ids']).issubset(records) and len(set(ids)-set(prior['ids']))==97
D.mkdir()
proof={'status':'ROOT_ALL100_FL100_CUDA_ORIGINAL_LINUX_WHOLE_AND_OFFSERVER_SAVED_AUDIT_ADOPTED',
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ids':ids,'accepted':100,'new_unique':97,'reused_already_adopted':3,'records':records,
    'original_Linux_saved_gate':'whole original root fit/prediction/metrics/native1e-12 accepted',
    'Windows_audit':'zero-fit saved predictions/counts/metrics; not whole refit',
    'metric_values_checked':900,'integer_base_counts_checked':2400,'prediction_rules_checked':300,'native_max_abs_difference':0.0,
    'audit':{'path':str(ap),'sha256':sha(ap)},'transport':{'path':str(tp),'sha256':sha(tp)},
    'archive':{'path':str(archive),'sha256':sha(archive),'members':959,'bytes':archive.stat().st_size},
    'prior3_root':inp['prior3_root'],'prior3_F_member_references':t['remote']['reused_member_pins'],
    'archive_all_members_rehashed':True,'reused_F_members_rehashed':28,'source_seal_sha256':sha(seal),'fresh_F_volume':volume,
    'record_producers':{i:'base' for i in ids},'cohort_sha256':a['cohort_sha256'],
    'original_Windows_CPU_failure_preserved':True,'whole100_complete':True,'whole_comparison_complete':False,'Full_repeated':False,
    'new_training':False,'parameters_changed':False,'tolerance_changed':False,'test':False,'old_CPU_outputs_preserved':True}
target=D/'ROOT_SCIENTIFIC_ADOPTION.json'
with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps({'accepted':100,'new_unique':97,'root':str(target),'sha256':sha(target)}))
