"""Root adoption of original LoGo100 saved closure, deduplicating already adopted2; no science execution."""
from pathlib import Path, PurePosixPath
import argparse, datetime, hashlib, json, math, re, sys, tarfile
R=Path(__file__).resolve().parents[1]
H=R/'tmp/LoGo100_cuda100_saved_closure_prepare_20261011'
D=H/'root_adoption'
sys.path[:0]=[str(H),str(R/'tmp')]
import metadata_gate as g
from guardfed_local_storage import check_bulk_storage
sha=g.sha;read=g.read
parser=argparse.ArgumentParser(description=__doc__)
for name in ['audit','transport','inputs','source-review']:
    parser.add_argument('--'+name,type=Path,required=True);parser.add_argument('--'+name+'-sha256',required=True)
args=parser.parse_args()
assert __debug__ and sys.platform=='win32' and not D.exists()
assert sha(R/'tmp/guardfed_local_storage.py')=='2482f2f46243abafd33ea2fadc94d555c0d80327fefe19caea3eb8f3f6bb3d63'
volume=check_bulk_storage()
seal=H/'FILES_SHA256.json'
assert sha(seal)=='40fa2578e9247d3bc746262915d4f54da470edcb980e364aa5a749babf94fd9e'
for rel,pin in read(seal)['files'].items():
    p=H/rel;assert Path(rel).name==rel and sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
review=args.source_review
assert sha(review)==args.source_review_sha256 and read(review)['source_adoptable'] is True and read(review)['source_seal_sha256']==sha(seal)
ap=args.audit;tp=args.transport;ip=args.inputs
for path,expected in [(ap,args.audit_sha256),(tp,args.transport_sha256),(ip,args.inputs_sha256)]:assert sha(path)==expected
a=read(ap);t=read(tp)
assert a['status']=='LOGO100_ALL100_CUDA_SAVED_PREDICTIONS_NO_REFIT_PASS_NOT_ROOT_ADOPTED'
assert a['ids']==g.IDS and [row['post_id'] for row in a['records']]==g.IDS
assert (a['prediction_items_checked'],a['metric_values_checked'],a['integer_confusion_counts_recomputed'])==(1986700,300,800)
assert a['new_fit']==a['new_CNN']==a['pickle_loads']==0
assert not a['test'] and not a['root_adopted'] and not a['offserver_whole_predict_reexecuted']
assert not a['historical_CPU_equality_required'] and a['max_metric_error']<=1e-12
assert sha(ip)==a['inputs_sha256'] and a['source_seal_sha256']==sha(seal)
x=read(ip);g.input_gate(x);assert x['root_authorized_audit'] is True
assert Path(x['source_review']['path']).resolve()==review.resolve() and x['source_review']['sha256']==sha(review)
assert Path(x['transport']['path']).resolve()==tp.resolve()
assert sha(tp)==a['transport_sha256']==x['transport']['sha256']
assert t['status']=='PASS_ARCHIVE_AND_ALL_MEMBER_SHA' and t['all_members_verified'] is True
archive=Path(t['archive_path']);dest=Path(t['verified_extract'])
assert all(p.resolve().drive.upper()=='F:' and p.resolve().is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()) for p in (archive,dest))
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
assert {n for n in t['members'] if n.endswith('.npz')}=={'runs/'+i+'/scores_predictions.npz' for i in g.REMAINING_IDS}
from prior2_overlay import join_prior
paths,member_roots,members=join_prior(x,dest,index,t)
prior_pins=read(H/'PRIOR2_PINS.json');prior=read(prior_pins['root']['path'])
assert prior_pins['root']['sha256']=='cdf9a578fd26a9a85ae674aff7e55932e9cd6e75b895b8e3b39adc1ad12753bc'
assert prior['ids']==g.CANARY_IDS and len(prior_pins['members'])==33
assert len(g.IDS)==len(set(g.IDS))==100 and len(g.REMAINING_IDS)==98 and set(g.REMAINING_IDS).isdisjoint(g.CANARY_IDS)
assert set(g.IDS)==set(g.REMAINING_IDS)|set(g.CANARY_IDS)

def load(pin):
    rel=paths[pin['path']];p=member_roots[rel]/rel;assert sha(p)==pin['sha256'];return read(p)
refs=read(H/'EXACT100_REFS.json');records={}
assert sha(H/'EXACT100_REFS.json')=='52d6a15e077eb50b1d81d2aa0e481c6e2e5fb5dfeea24e84ba6036535dffed9b'
assert [r['post_id'] for r in refs['records']]==g.IDS
cells=[]
for row in refs['records']:
    match=re.fullmatch(r'LoGoFair-DP_07_FedAvg_(IID|non-IID)_(Benign|F Flip|FedSA|S-DFA|Sp-DFA)_seed(910\d\d)',row['post_id']);assert match
    cells.append((match[1],match[2],int(match[3])))
assert set(cells)=={(d,attack,seed) for d in ['IID','non-IID'] for attack in ['Benign','F Flip','FedSA','S-DFA','Sp-DFA'] for seed in range(91001,91011)} and len(cells)==100
for pin,expected_ids in [(x['canary_complete'],g.CANARY_IDS),(x['remaining_complete'],g.REMAINING_IDS)]:
    complete=load(pin);assert complete['ids']==expected_ids and complete['server_saved_accepted'] is True and complete['cohort']==load(x['root_queue_auth'])['exact_cohort']
gate=load(x['root_canary_gate']);assert gate['root_accepted'] is True and gate['ids']==g.CANARY_IDS and gate['FedAvg_CUDA_root']==x['FedAvg_CUDA_root']
for i in g.CANARY_IDS:assert gate['records'][i]['task_acceptance']==x['task_accepted'][i]
metric_values=primitive_counts=decisions=0;originals=[]

for row,ar in zip(refs['records'],a['records']):
    i=row['post_id'];assert ar['post_id']==i and ar['baseline_id']==row['baseline_id']
    v=g.verify_record(x,row,load);aa=load(x['task_accepted'][i])
    command_path=x['server_base']+'/tasks/'+i+'/worker_command.json';rel=paths[command_path]
    worker=read(member_roots[rel]/rel);assert worker['exit_code']==0
    assert v['result']['checkpoint_sha256']==row['checkpoint_sha256'] and v['job']['id']==i
    for key in ['accuracy','aeod','aspd']:
        assert isinstance(ar['metrics'][key],(int,float)) and not isinstance(ar['metrics'][key],bool) and math.isfinite(ar['metrics'][key])
        metric_values+=1
    assert ar['Linux_whole_acceptance']==x['task_accepted'][i] and ar['metrics']==v['result']['metrics']
    assert ar['scores_sha256']==v['scores']['sha256'] and ar['max_metric_error']<=1e-12
    assert ar['frozen_post_state_sha256']==row['original_artifacts']['post_state.pkl']['sha256']
    counts=ar['group_confusion_counts']
    assert sum(counts[k][n] for k in ['0','1'] for n in ['tp','fp','tn','fn'])==19867
    for k in ['0','1']:
        for n in ['tp','fp','tn','fn']:
            assert isinstance(counts[k][n],int) and not isinstance(counts[k][n],bool) and counts[k][n]>=0
            primitive_counts+=1
    assert ar['metrics']['prediction_count']==19867 and ar['constant_prediction'] in (None,0,1)
    decisions+=ar['metrics']['prediction_count']
    for original in row['original_artifacts'].values():
        assert sha(original['original_path'])==original['sha256'];originals.append(original)
    records[i]={'acceptance':aa['acceptance'],'binding':aa['binding'],'approval':aa['approval'],
        'task_acceptance':x['task_accepted'][i],'baseline_id':row['baseline_id'],
        'checkpoint_sha256':row['checkpoint_sha256'],'metrics':ar['metrics'],
        'scores_sha256':ar['scores_sha256'],'frozen_post_state_sha256':ar['frozen_post_state_sha256'],
        'group_confusion_counts':counts,'constant_prediction':ar['constant_prediction'],
        'historical_CPU_difference':v['accepted']['historical_CPU_difference']}
assert (metric_values,primitive_counts,decisions)==(300,800,1986700)
assert list(records)==g.IDS
# Canonical serialization detects value/type/order drift, then retain original2 objects exactly.
for i in g.CANARY_IDS:
    assert json.dumps(records[i],ensure_ascii=False,separators=(',',':'))==json.dumps(prior['records'][i],ensure_ascii=False,separators=(',',':'))
    records[i]=prior['records'][i]
for original in originals:assert sha(original['original_path'])==original['sha256']
for rel,pin in members.items():assert sha(member_roots[rel]/rel)==pin['sha256']
for rel,pin in read(seal)['files'].items():assert sha(H/rel)==pin['sha256']
auth=load(x['root_queue_auth']);assert auth['ordered_ids']==g.IDS and auth['canary_ids']==g.CANARY_IDS
D.mkdir()
proof={'status':'ROOT_LOGO_ALL100_CUDA_CACHE_FROZEN_POST_AND_OFFSERVER_SAVED_AUDIT_ADOPTED',
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root_accepted':True,'accepted':100,'ids':g.IDS,'newly_adopted':98,'previously_adopted_reused':2,'prior2':prior_pins['root'],'prior2_records_exact':True,
    'cohort':auth['exact_cohort'],'FedAvg_CUDA_root':auth['FedAvg_CUDA_root'],'FedAvg_CUDA_index':auth['FedAvg_CUDA_index'],
    'root_queue_auth':x['root_queue_auth'],'deployment_seal':x['deployment_seal'],'records':records,
    'offserver_audit':{'path':str(ap),'sha256':sha(ap)},'transport':{'path':str(tp),'sha256':sha(tp)},
    'archive':{'path':str(archive),'sha256':sha(archive),'members':t['archive_members'],'bytes':archive.stat().st_size},
    'archive_all_members_rehashed':True,'prior33_members_rehashed':True,'original_source_review':{'path':str(review),'sha256':sha(review)},'actual_inputs':{'path':str(ip),'sha256':sha(ip)},'adopter_source_sha256':sha(__file__),'original_artifacts_immutable_rehashed':True,'fresh_F_volume':volume,
    'source_seal_sha256':sha(seal),'metric_values_checked':300,'integer_confusion_counts_recomputed':800,
    'prediction_items_checked':1986700,'max_metric_error':a['max_metric_error'],
    'new_fit':0,'new_CNN':0,'pickle_loads_offserver':0,'whole100_complete':True,'whole_comparison_complete':False,
    'offserver_whole_predict_reexecuted':False,'historical_CPU_equality_required':False,
    'new_training':False,'parameters_changed':False,'tolerance_changed':False,'test':False,'old_CPU_outputs_preserved':True}
target=D/'ROOT_SCIENTIFIC_ADOPTION.json'
with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps({'accepted':100,'newly_adopted':98,'root':str(target),'sha256':sha(target)}))
