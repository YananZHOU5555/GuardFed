"""Future fixed2 saved decisions only. No model/pickle/netcal/Torch import or fit."""
from pathlib import Path,PurePosixPath
import argparse,ast,datetime,hashlib,json,math,sys,tarfile,types,zipfile
import metadata_gate as g
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def sealed(expected):
    g.need(g.sha(HERE/'FILES_SHA256.json')==expected,'Different source seal')
    for name,item in g.read(HERE/'FILES_SHA256.json')['files'].items():g.need(Path(name).name==name and g.sha(HERE/name)==item['sha256'],'Source changed')
def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--inputs-sha256',required=True);p.add_argument('--source-seal-sha256',required=True);a=p.parse_args()
    g.need(sys.platform=='win32' and __debug__,'Original Windows audit runtime required');sealed(a.source_seal_sha256)
    g.need(g.sha(a.inputs)==a.inputs_sha256,'Changed actual inputs');x=g.read(a.inputs);g.input_gate(x);g.need(x['root_authorized_audit'] is True,'Unbound actual audit')
    review=Path(x['source_review']['path']);g.need(g.sha(review)==x['source_review']['sha256'] and g.read(review)['source_adoptable'] is True and g.read(review)['source_seal_sha256']==a.source_seal_sha256,'Unreviewed audit')
    sys.path.insert(0,str(ROOT/'tmp'));from guardfed_local_storage import check_bulk_storage
    g.need(g.sha(ROOT/'tmp/guardfed_local_storage.py')=='2482f2f46243abafd33ea2fadc94d555c0d80327fefe19caea3eb8f3f6bb3d63','Storage guard changed');volume=check_bulk_storage()
    tp=Path(x['transport']['path']);g.need(g.sha(tp)==x['transport']['sha256'],'Changed transport');t=g.read(tp)
    archive=Path(t['archive_path']).resolve();folder=Path(t['verified_extract']).resolve()
    g.need(all(p.drive.upper()=='F:' and p.is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()) for p in [archive,folder]),'Bulk outside F')
    g.need(t['status']=='PASS_ARCHIVE_AND_ALL_MEMBER_SHA' and t['all_members_verified'] is True and g.sha(archive)==t['archive_sha256']==x['archive_sha256'],'Unverified archive')
    index=g.read(folder/'ROOT_MEMBERS.json');g.need(g.sha(folder/'ROOT_MEMBERS.json')==t['member_index_sha256'] and index['files']==t['members'] and index['ids']==g.IDS,'Different archive scope')
    members=index['files'];paths=index['server_paths'];g.need(len(set(paths.values()))==len(paths) and set(paths.values())==set(members),'Missing/ambiguous server member map')
    with tarfile.open(archive,'r:gz') as tar:
        entries=tar.getmembers();g.need(len(entries)==len({m.name for m in entries})==len(members)+1 and {m.name for m in entries}==set(members)|{'ROOT_MEMBERS.json'},'Archive member mismatch')
        for member in entries:
            rel=PurePosixPath(member.name);g.need(member.isfile() and not rel.is_absolute() and '..' not in rel.parts and ':' not in member.name and '\\' not in member.name,'Unsafe archive')
            local=folder.joinpath(*rel.parts);g.need(not local.is_symlink() and local.resolve().is_relative_to(folder),'Unsafe extraction')
            wanted=members.get(member.name,{'sha256':t['member_index_sha256'],'bytes':local.stat().st_size});raw=tar.extractfile(member).read()
            g.need(hashlib.sha256(raw).hexdigest()==g.sha(local)==wanted['sha256'] and len(raw)==local.stat().st_size==wanted['bytes'],'Member changed')
            g.need(not member.name.endswith(('.pkl','.pt','.pth','.npy')) and (not member.name.endswith('.npz') or member.name in {'runs/'+i+'/scores_predictions.npz' for i in g.IDS}),'Forbidden bulk member')
    archive_folder=folder
    def localpin(pin):
        g.need(pin['path'] in paths,'Missing transported proof');p=archive_folder/paths[pin['path']];g.need(g.sha(p)==pin['sha256'],'Transported pin changed');return p
    def load(pin):return g.read(localpin(pin))
    refs=g.read(HERE/'EXACT2_REFS.json');items=[g.verify_record(x,row,load) for row in refs['records']]
    # Read current original F files only by SHA. Never unpickle or refit old state.
    originals=[]
    for row in refs['records']:
        for role,pin in row['original_artifacts'].items():
            old=Path(pin['original_path']).resolve();g.need(old.drive.upper()=='F:' or (role=='job' and old.drive.upper()=='E:'),'Unexpected original location')
            g.need(g.sha(old)==pin['sha256'],'Original backup identity changed');originals.append((old,pin['sha256']))
    rootp=Path(x['local_FedAvg_root']['path']);ip=Path(x['local_FedAvg_index']['path'])
    g.need(g.sha(rootp)==x['local_FedAvg_root']['sha256']==g.ROOT_SHA and g.sha(ip)==x['local_FedAvg_index']['sha256']==g.INDEX_SHA,'Wrong actual adopted root/index copy')
    root=g.read(rootp);cache_index=g.read(ip);g.need(root['accepted']==100 and root['status']=='ROOT_FEDAVG100_CUDA_ORIGINAL_LINUX_WHOLE_AND_OFFSERVER_SAVED_AUDIT_ADOPTED' and cache_index['root']==x['FedAvg_CUDA_root'],'Unadopted cache')
    old_transport=Path(x['local_FedAvg_transport']['path']);g.need(g.sha(old_transport)==x['local_FedAvg_transport']['sha256']==root['transport']['sha256'],'Wrong reused cache transport')
    ct=g.read(old_transport);cache_folder=Path(ct['verified_extract']).resolve();g.need(cache_folder.drive.upper()=='F:' and ct['all_members_verified'] is True,'Unverified existing cache')
    out=HERE/'actual';out.mkdir(exist_ok=True);g.need(not any((out/n).exists() for n in ['STARTED.json','SAVED_AUDIT.json','FAILURE.json']),'No overwrite/retry')
    save(out/'STARTED.json',dict(inputs_sha256=a.inputs_sha256,ids=g.IDS,new_fit=0,new_CNN=0))
    try:
        import numpy as np
        from typing import Dict,Any,List
        ns=dict(np=np,math=math,Dict=Dict,Any=Any,List=List,need=g.need,METRICS=('accuracy','aeod','aspd'),TOLERANCE=1e-12)
        exec(compile((HERE/'original_metric_functions.py.txt').read_text('utf8'),'<sealed original metric functions>','exec'),ns)
        core=types.SimpleNamespace(compute_metrics=ns['compute_metrics']);old_bridge=types.SimpleNamespace(np=np);worst=0.0;results=[];saved_metric_error=ns['saved_metric_error'];need=g.need
        for row,v in zip(refs['records'],items):
            result=v['result'];folder=localpin(v['scores']).parent
            # Exact original saved-decision block is inserted below by preparation.
            with old_bridge.np.load(folder / 'scores_predictions.npz', allow_pickle=False) as cache:
                prediction = cache['prediction']; computed = core.compute_metrics(cache['valid_y'], prediction, cache['valid_sensitive'])
                need(len(prediction) == 19867 and set(old_bridge.np.unique(prediction)) <= {0, 1}, 'Invalid saved decisions')
                constant = int(prediction[0]) if old_bridge.np.unique(prediction).size == 1 else None
            error = saved_metric_error(computed, result['metrics']); worst = max(worst, error)
            with np.load(localpin(v['scores']),allow_pickle=False) as z:
                values={k:z[k].copy() for k in ['prediction','valid_y','valid_sensitive','valid_image_id','valid_native_prediction','root_y','root_sensitive','root_image_id']}
            cache_record=next(r for r in cache_index['records'] if r['id']==row['baseline_id'])
            g.need(cache_record['members']['predictions']['sha256']==v['formal']['CUDA_cache_sha256'] and cache_record['inventory_record']['checkpoint']['sha256']==row['checkpoint_sha256'],'Wrong adopted cache record')
            producer=root['records'][row['baseline_id']]['producer'];member=producer+'/runs/'+row['baseline_id']+'/validation_predictions.npz';cp=cache_folder/member
            g.need(g.sha(cp)==ct['members'][member]['sha256']==v['formal']['CUDA_cache_sha256'],'Reused F CUDA cache changed')
            with np.load(cp,allow_pickle=False) as z:
                g.need(np.array_equal(values['valid_image_id'],z['valid_image_ids']) and np.array_equal(values['root_image_id'],z['root_image_ids']) and np.array_equal(values['valid_native_prediction'],(z['valid_margins']>0).astype(int)),'Changed cache identity/native decisions')
            metadata=cache_folder/'source/metadata.npz';g.need(g.sha(metadata)==ct['members']['source/metadata.npz']['sha256']=='161f8028f1c29ba470afa60cbd9fb54d7bf61b3cec5c525830ad7a3ef7ab2091','Wrong original labels')
            pn=dict(np=np,hashlib=hashlib,zipfile=zipfile,require=g.need)
            exec(compile((HERE/'original_prefix_functions.py.txt').read_text('utf8'),'<original label prefix>','exec'),pn)
            labels,_=pn['read_prefix'](metadata,'Smiling',202599,182637);sensitive,_=pn['read_prefix'](metadata,'Male',202599,182637)
            for split in ['root','valid']:
                ids=values[split+'_image_id'];g.need(np.issubdtype(ids.dtype,np.integer) and ((ids>=1)&(ids<=182637)).all(),'Invalid original row ids')
                g.need(np.array_equal(values[split+'_y'],labels[ids-1]) and np.array_equal(values[split+'_sensitive'],sensitive[ids-1]),'Changed label/sensitive cache')
            gm=ns['group_metrics'](values['valid_y'],values['prediction'],values['valid_sensitive']);ns['saved_metric_error'](gm,result['metrics'])
            counts=gm['group_confusion_counts'];g.need(sum(counts[k][n] for k in ['0','1'] for n in ['tp','fp','tn','fn'])==19867,'Incomplete confusion counts')
            g.need(sum(counts[k]['tp']+counts[k]['tn'] for k in ['0','1'])==int(np.sum(values['prediction']==values['valid_y'])),'Accuracy/count disagreement')
            results.append(dict(post_id=row['post_id'],baseline_id=row['baseline_id'],metrics=result['metrics'],group_confusion_counts=counts,constant_prediction=constant,max_metric_error=error,scores_sha256=v['scores']['sha256'],Linux_whole_acceptance=x['task_accepted'][row['post_id']],frozen_post_state_sha256=row['original_artifacts']['post_state.pkl']['sha256']))
        for p,s in originals:g.need(g.sha(p)==s,'Original changed during audit')
        for rel,p in members.items():g.need(g.sha(Path(t['verified_extract'])/rel)==p['sha256'],'Export changed during audit')
        sealed(a.source_seal_sha256)
        save(out/'SAVED_AUDIT.json',dict(status='LOGO_FIXED2_SAVED_PREDICTIONS_NO_REFIT_PASS_NOT_ROOT_ADOPTED',records=results,ids=g.IDS,prediction_items_checked=39734,metric_values_checked=6,integer_confusion_counts_recomputed=16,max_metric_error=worst,transport_sha256=x['transport']['sha256'],archive_sha256=x['archive_sha256'],source_seal_sha256=a.source_seal_sha256,inputs_sha256=a.inputs_sha256,F_volume=volume,new_fit=0,new_CNN=0,pickle_loads=0,test=False,root_adopted=False,offserver_whole_predict_reexecuted=False,historical_CPU_equality_required=False))
        print(json.dumps(dict(status='PASS',audit_sha256=g.sha(out/'SAVED_AUDIT.json'),records=2,metrics=6,counts=16)))
    except BaseException as exc:
        save(out/'FAILURE.json',dict(error=repr(exc),automatic_retry=False));raise
if __name__=='__main__':main()
