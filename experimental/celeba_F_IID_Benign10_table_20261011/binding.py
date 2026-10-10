"""Exact accepted304 + six F records -> one F scene; metadata gates only."""
import argparse
import hashlib
import json
from pathlib import Path

H=Path(__file__).resolve().parent
R=H.parents[1]
PARENT_ROOT='tmp/celeba_mechanism_remaining_after300_root_adoption_20261011/ROOT_ADOPTION.json'
PARENT_ROOT_SHA='748ae37ec6d343a667aabfa89362ef14660de4d8fc1a6ef4bde3b8ea4e94c5d4'
PARENT_INDEX='tmp/celeba_mechanism_remaining_after300_root_adoption_20261011/MECHANISM304_INDEX.json'
PARENT_INDEX_SHA='d7ac91496bb50c181f123fed80f305d21938d69b875eb95ce9db7e7debf3a528'
IDS=[f'minus_F_IID_Benign_seed{s}' for s in range(91001,91011)]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def need(ok,message):
    if not ok:raise ValueError(message)

def validate_scope(root,index,prior,native):
    need((root['prior_accepted'],root['new_accepted'],root['cumulative_accepted'])==(304,6,310),'Exact304+6 replay310 required')
    need(root['status'].startswith('ROOT_') and root['status'].endswith('_ADOPTED') and root['original304_unchanged'] is True,'Actual adopted replay root required')
    need(root['accepted_new_ids']==index['new_ids']==IDS[4:],'Only six remaining F IID Benign seeds permitted')
    need(prior['new_ids']==IDS[:4] and len(prior['all_ids'])==len(set(prior['all_ids']))==304,'Parent304/F4 scope differs')
    need(index['all_ids']==prior['all_ids']+IDS[4:] and len(set(index['all_ids']))==310,'Old304 order or exact310 scope differs')
    need(index['prior_index_sha256']==PARENT_INDEX_SHA and index['prior_adoption_sha256']==PARENT_ROOT_SHA,'Accepted304 parent pins differ')
    need(root['native_max_abs_difference']==0 and root['Full_inference']==root['new_CNN']==root['new_training']==0 and root.get('new_fit',0)==0 and root['test'] is False,'Forbidden science or native discrepancy')
    need(native['root_adopted'] is True and native['status']=='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS' and native['total_new_strict_and_offserver']>=310 and native['test'] is False,'Accepted native source must contain replay310; larger native cutoff is not table scope')
    for part,wanted in ((prior,IDS[:4]),(index,IDS[4:])):
        need([x['id'] for x in part['new_records']]==wanted,'Exact ordered accepted records required')
        for key in ('new_bindings','new_binding_files','new_artifacts'):
            need(set(part[key])==set(wanted),'Missing/extra exact-delta metadata: '+key)

def load():
    parser=argparse.ArgumentParser();parser.add_argument('--binding-sha256',required=True);args=parser.parse_args()
    bp=H/'ROOT_BINDING.json'
    need(bp.is_file(),'Actual adopted replay310 is not bound; numeric generation forbidden')
    need(sha(bp)==args.binding_sha256,'Root binding SHA differs')
    b=read(bp);need(b['root_adopted'] is True and b['status']=='ACTUAL_REPLAY310_BOUND_FOR_F_IID_BENIGN10','Actual root authorization binding required')
    need(sha(H/'FILES_SHA256.json')==b['source_seal_sha256'],'Reviewed source seal differs')
    for name,pin in read(H/'FILES_SHA256.json')['files'].items():
        p=(H/name).resolve();need(p.is_relative_to(H.resolve()) and sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],'Frozen table source differs: '+name)
    source=read(H/'SOURCE_INPUTS.json')
    for name,pin in source['files'].items():
        p=R/name;need(sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],'Source/input drift: '+name)
    need(sha(R/PARENT_ROOT)==PARENT_ROOT_SHA and sha(R/PARENT_INDEX)==PARENT_INDEX_SHA,'Actual parent304 drift')
    pins=dict(source);pins['files']=dict(source['files'])
    for key in ('adoption','index','native_root'):
        p=(R/b[key]).resolve();need(p.is_relative_to(R.resolve()) and sha(p)==b[key+'_sha256'],'Actual external pin differs: '+key)
        pins[key]=b[key];pins['files'][b[key]]=dict(sha256=sha(p),bytes=p.stat().st_size)
    root=read(R/b['adoption']);index=read(R/b['index']);prior=read(R/PARENT_INDEX);native=read(R/b['native_root'])
    need(root['status']==b['root_status'] and root['records_index_sha256']==b['index_sha256'],'Root/index join differs')
    need(root['native_root_sha256']==b['native_root_sha256'] and (R/root['native_root_path']).resolve()==(R/b['native_root']).resolve(),'Root/native identity differs')
    need(native['inspection_sha256']==root['native_inspection_sha256']==index['native_inspection_sha256'],'Native inspection join differs')
    validate_scope(root,index,prior,native)
    ip=(R/index['native_inspection_path']).resolve();need(ip.is_relative_to(R.resolve()) and sha(ip)==index['native_inspection_sha256'],'Native accepted metadata drift')
    pins['files'][str(ip.relative_to(R.resolve()))]=dict(sha256=sha(ip),bytes=ip.stat().st_size)
    merged=dict(index)
    merged['new_ids']=list(IDS);merged['new_records']=prior['new_records']+index['new_records']
    for key in ('new_bindings','new_binding_files','new_artifacts'):merged[key]={**prior[key],**index[key]}
    merged['record_source_roots']={rid:dict(path=PARENT_ROOT,sha256=PARENT_ROOT_SHA,index=PARENT_INDEX,index_sha256=PARENT_INDEX_SHA) for rid in IDS[:4]}
    merged['record_source_roots'].update({rid:dict(path=b['adoption'],sha256=b['adoption_sha256'],index=b['index'],index_sha256=b['index_sha256']) for rid in IDS[4:]})
    pins['root_binding_sha256']=sha(bp)
    return pins,root,merged
