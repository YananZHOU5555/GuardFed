"""Root adoption of original Linux frozen-post checks plus zero-fit saved decisions."""
from pathlib import Path, PurePosixPath
import datetime, hashlib, json, sys, tarfile
R=Path(__file__).resolve().parents[1]
H=R/'tmp/LoGo_cuda_fixed2_saved_audit_prepare_v2_20261011'
D=H/'root_adoption'
sys.path[:0]=[str(H),str(R/'tmp')]
import metadata_gate as g
from guardfed_local_storage import check_bulk_storage
sha=g.sha;read=g.read
assert __debug__ and not D.exists()
volume=check_bulk_storage()
seal=H/'FILES_SHA256.json'
assert sha(seal)=='eb08186667ef3ca700cd853aa4dfc919181c86f080b89e5b2b1484203051e3a3'
for rel,pin in read(seal)['files'].items():
    p=H/rel;assert Path(rel).name==rel and sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
review=H/'ROOT_SOURCE_REVIEW.json'
assert sha(review)=='6ffb69c7d7335c0d720c7b243732a7f080d766e659d3f12915cc6fa834ad7502'
assert read(review)['source_adoptable'] is True
ap=H/'actual/SAVED_AUDIT.json';tp=H/'LOGO_FIXED2_TRANSPORT_VERIFICATION.json'
assert sha(ap)=='41364281d5735fe0b10bb50ab4b9309766a82420c0f8c53aeda63f26a7d134e6'
assert sha(tp)=='ebb9b246b205057ca7f756fae727093e375fa1b33a702d7c7379adf6b783fb6d'
a=read(ap);t=read(tp)
assert a['status']=='LOGO_FIXED2_SAVED_PREDICTIONS_NO_REFIT_PASS_NOT_ROOT_ADOPTED'
assert a['ids']==g.IDS and [row['post_id'] for row in a['records']]==g.IDS
assert (a['prediction_items_checked'],a['metric_values_checked'],a['integer_confusion_counts_recomputed'])==(39734,6,16)
assert a['new_fit']==a['new_CNN']==a['pickle_loads']==0
assert not a['test'] and not a['root_adopted'] and not a['offserver_whole_predict_reexecuted']
assert not a['historical_CPU_equality_required'] and a['max_metric_error']<=1e-12
ip=H/'actual_metadata/ACTUAL_AUDIT_INPUTS.json';assert sha(ip)==a['inputs_sha256']
x=read(ip);g.input_gate(x);assert x['root_authorized_audit'] is True
assert sha(tp)==a['transport_sha256']==x['transport']['sha256']
assert t['status']=='PASS_ARCHIVE_AND_ALL_MEMBER_SHA' and t['all_members_verified'] is True
archive=Path(t['archive_path']);dest=Path(t['verified_extract'])
assert all(p.resolve().is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()) for p in (archive,dest))
assert sha(archive)==a['archive_sha256']==t['archive_sha256']==x['archive_sha256']
assert archive.stat().st_size==t['archive_bytes']
index=read(dest/'ROOT_MEMBERS.json')
assert sha(dest/'ROOT_MEMBERS.json')==t['member_index_sha256'] and index['files']==t['members'] and index['ids']==g.IDS
with tarfile.open(archive,'r:gz') as tar:
    ms=tar.getmembers();assert len(ms)==len({m.name for m in ms})==t['archive_members']
    assert {m.name for m in ms}==set(t['members'])|{'ROOT_MEMBERS.json'}
    for m in ms:
        rel=PurePosixPath(m.name)
        assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts and ':' not in m.name and '\\' not in m.name
        data=tar.extractfile(m).read();p=dest.joinpath(*rel.parts)
        pin=t['members'].get(m.name,{'sha256':t['member_index_sha256'],'bytes':len(data)})
        assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(dest.resolve())
        assert hashlib.sha256(data).hexdigest()==pin['sha256']==sha(p) and len(data)==pin['bytes']==p.stat().st_size
assert len([n for n in t['members'] if n.endswith('.npz')])==2
paths=index['server_paths']
def load(pin):
    p=dest/paths[pin['path']];assert sha(p)==pin['sha256'];return read(p)
refs=read(H/'EXACT2_REFS.json');records={}
for row,ar in zip(refs['records'],a['records']):
    i=row['post_id'];v=g.verify_record(x,row,load);aa=load(x['task_accepted'][i])
    assert ar['Linux_whole_acceptance']==x['task_accepted'][i] and ar['metrics']==v['result']['metrics']
    assert ar['scores_sha256']==v['scores']['sha256'] and ar['max_metric_error']<=1e-12
    assert ar['frozen_post_state_sha256']==row['original_artifacts']['post_state.pkl']['sha256']
    counts=ar['group_confusion_counts']
    assert sum(counts[k][n] for k in ['0','1'] for n in ['tp','fp','tn','fn'])==19867
    for p in row['original_artifacts'].values():assert sha(p['original_path'])==p['sha256']
    records[i]={'acceptance':aa['acceptance'],'binding':aa['binding'],'approval':aa['approval'],
        'task_acceptance':x['task_accepted'][i],'baseline_id':row['baseline_id'],
        'checkpoint_sha256':row['checkpoint_sha256'],'metrics':ar['metrics'],
        'scores_sha256':ar['scores_sha256'],'frozen_post_state_sha256':ar['frozen_post_state_sha256'],
        'group_confusion_counts':counts,'constant_prediction':ar['constant_prediction'],
        'historical_CPU_difference':v['accepted']['historical_CPU_difference']}
auth=load(x['root_queue_auth'])
D.mkdir()
proof={'status':'ROOT_LOGO_FIXED2_CUDA_CACHE_FROZEN_POST_AND_OFFSERVER_SAVED_AUDIT_ADOPTED',
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root_accepted':True,'accepted':2,'ids':g.IDS,
    'cohort':auth['exact_cohort'],'FedAvg_CUDA_root':auth['FedAvg_CUDA_root'],'FedAvg_CUDA_index':auth['FedAvg_CUDA_index'],
    'root_queue_auth':x['root_queue_auth'],'deployment_seal':x['deployment_seal'],'records':records,
    'offserver_audit':{'path':str(ap),'sha256':sha(ap)},'transport':{'path':str(tp),'sha256':sha(tp)},
    'archive':{'path':str(archive),'sha256':sha(archive),'members':t['archive_members'],'bytes':archive.stat().st_size},
    'archive_all_members_rehashed':True,'original_artifacts_immutable_rehashed':True,'fresh_F_volume':volume,
    'source_seal_sha256':sha(seal),'metric_values_checked':6,'integer_confusion_counts_recomputed':16,
    'prediction_items_checked':39734,'max_metric_error':a['max_metric_error'],
    'new_fit':0,'new_CNN':0,'pickle_loads_offserver':0,'whole100_complete':False,'whole_comparison_complete':False,
    'offserver_whole_predict_reexecuted':False,'historical_CPU_equality_required':False,
    'new_training':False,'parameters_changed':False,'tolerance_changed':False,'test':False,'old_CPU_outputs_preserved':True}
target=D/'ROOT_SCIENTIFIC_ADOPTION.json'
with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps({'accepted':2,'root':str(target),'sha256':sha(target)}))
