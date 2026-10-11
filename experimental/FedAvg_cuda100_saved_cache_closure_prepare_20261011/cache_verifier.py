"""Root-adopted FedAvg100 cache verifier. No scientific module import until load_cache."""
from pathlib import Path
import hashlib,json,zipfile
HERE=Path(__file__).resolve().parent
ROLES={'original_result','receipt','predictions','formal_binding','saved_acceptance'}
META_SHA='161f8028f1c29ba470afa60cbd9fb54d7bf61b3cec5c525830ad7a3ef7ab2091'
VALID_SHA='64a15cf28caf1d177ac3dcf96a4408bc21091796974b243923ca947a37b554bf'
PREFIX_SHA='7be8f88970e1b3f090a55c3425cc3ca9ee3b2a5777bc2cbab956efbb88a4a883'
REFS_SHA='067b28534495601e21a2af1637a961449cf4112c76f41975b2764a924f636a8e'
ROOT_STATUS='ROOT_FEDAVG100_CUDA_ORIGINAL_LINUX_WHOLE_AND_OFFSERVER_SAVED_AUDIT_ADOPTED'
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def pinned(pin):
    need(isinstance(pin,dict) and set(pin)=={'path','sha256'} and isinstance(pin['path'],str),'Unbound pin')
    p=Path(pin['path']);need(p.is_absolute() and '..' not in p.parts and not p.is_symlink() and p.resolve()==p and p.is_file(),'Unsafe or missing pinned file')
    need(isinstance(pin['sha256'],str) and len(pin['sha256'])==64 and sha(p)==pin['sha256'],'Pinned file changed');return p
def verify_root_index(root_pin,index_pin,ordered_FedAvg_ids):
    need(sha(HERE/'EXACT100_REFS.json')==REFS_SHA,'Fixed scope reference changed')
    root=read(pinned(root_pin));index=read(pinned(index_pin));refs=read(HERE/'EXACT100_REFS.json');ids=refs['ordered_ids']
    need(root['status']==ROOT_STATUS and root['ids']==ids and root['accepted']==100,'Not root adopted FedAvg100')
    need(root['metric_values_checked']==900 and root['integer_base_counts_checked']==2400 and root['prediction_rules_checked']==300,'Incomplete original saved audit')
    need(root['archive_all_members_rehashed'] is True and not root['new_training'] and not root['test_evaluation'],'Wrong adopted scope')
    need(index['status']=='ROOT_ADOPTED_FEDAVG100_CACHE_INDEX' and index['root']==root_pin,'Index not bound to this root')
    need(len(ordered_FedAvg_ids)==len(set(ordered_FedAvg_ids))==100 and set(ordered_FedAvg_ids)==set(ids),'Missing/foreign/duplicate FedAvg selection')
    need([r['id'] for r in index['records']]==ids and set(root['records'])==set(ids),'Original100 order/completeness changed')
    audit=read(pinned(index['audit']));transport=read(pinned(index['transport']));selection=read(pinned(index['selection']))
    need(index['audit']['sha256']==root['audit']['sha256'] and index['transport']['sha256']==root['transport']['sha256'],'Root proof mismatch')
    need(audit['status']=='FEDAVG100_SAVED_OUTPUTS_AUDIT_NO_REFIT_PASS_NOT_ROOT_ADOPTED' and audit['exact_ids']==ids,'Wrong saved audit scope')
    need(audit['fit_calls']==audit['new_CNN']==audit['new_training']==0 and audit['test'] is False and len(audit['records'])==100,'Audit added science or incomplete')
    need(audit['metric_values_checked']==900 and audit['integer_base_counts_checked']==2400 and audit['prediction_rules_checked']==300,'Audit count mismatch')
    need(audit['transport_sha256']==index['transport']['sha256'] and audit['archive_sha256']==transport['archive_sha256']==root['archive']['sha256'],'Archive/audit lineage differs')
    need(transport['status']=='PASS_ARCHIVE_AND_ALL_MEMBER_SHA' and transport['all_members_verified'] is True,'Unverified transport')
    need(transport['members']['base/COHORT_SELECTION.json']['sha256']==index['selection']['sha256'] and selection['exact_ids']==ids,'Selected source not in adopted archive')
    need(selection['record_producers']==refs['record_producers'] and selection['original_inventory_sha256']==refs['inventory']['sha256'],'Original inventory/producer mismatch')
    need(index['metadata']['sha256']==META_SHA==transport['members']['source/metadata.npz']['sha256'],'Wrong metadata')
    pinned(index['metadata']);expected={r['id']:r for r in refs['records']};selected={r['id']:r for r in selection['records']};audited={r['id']:r for r in audit['records']}
    for item in index['records']:
        identity=item['id'];record=item['inventory_record'];rr=root['records'][identity];ar=audited[identity]
        need(record==selected[identity] and canonical(record)==expected[identity]['canonical_sha256'],'Changed original record')
        need(record['method']=='FedAvg' and record['terminal_round']==70 and record['original_split']=='valid' and record['original_n_eval']==19867,'Wrong native endpoint')
        need(set(item['members'])==ROLES and item['metadata']==index['metadata'],'Wrong cache member schema')
        paths={k:pinned(v) for k,v in item['members'].items()};receipt=read(paths['receipt']);binding=read(paths['formal_binding']);saved=read(paths['saved_acceptance'])
        producer=refs['record_producers'][identity];prefix=producer+'/runs/'+identity+'/'
        for role,name in [('receipt','receipt.json'),('predictions','validation_predictions.npz'),('formal_binding','formal_binding.json'),('saved_acceptance','formal_saved_acceptance.json')]:
            need(item['members'][role]['sha256']==transport['members'][prefix+name]['sha256'],'Member not in adopted archive')
        need(item['members']['original_result']['sha256']==record['result']['sha256']==transport['members']['original_results/'+identity+'.json']['sha256'],'Wrong original result')
        need(rr['producer']==producer and rr['receipt_sha256']==ar['receipt_sha256']==item['members']['receipt']['sha256'] and rr['arrays_sha256']==ar['arrays_sha256']==item['members']['predictions']['sha256'],'Root per-record link mismatch')
        need(rr['acceptance']==item['members']['saved_acceptance'],'Root acceptance link differs')
        need(receipt['id']==identity and receipt['model_inventory_record_sha256']==canonical(record),'Receipt model differs')
        need(receipt['status']=='NATIVE_VALID_REPLAY_PASS' and receipt['scope']=='FORMAL_ORIGINAL800_CUDA_PER_RECORD_V1' and receipt['runtime']['device']=='cuda:0','Not original CUDA receipt')
        for key,role in [('checkpoint_sha256','checkpoint'),('original_result_sha256','result'),('original_job_sha256','raw_job')]:need(receipt[key]==record[role]['sha256'],'Terminal artifact mismatch')
        need(receipt['prediction_arrays_sha256']==item['members']['predictions']['sha256'] and receipt['valid_n']==19867 and receipt['valid_image_ids_sha256']==VALID_SHA,'Wrong saved cache')
        need(binding['id']==identity and binding['scope']=='UNIFORM_CUDA_ORIGINAL800_FORMAL_V1' and binding['receipt_sha256']==item['members']['receipt']['sha256'],'Formal binding differs')
        need(binding['source_before']==binding['source_after'] and binding['artifact_before']==binding['artifact_after'],'Producer mutation')
        need(saved['status']=='PER_RECORD_ORIGINAL800_CUDA_SAVED_WHOLE_ACCEPTED' and saved['id']==identity and saved['binding_sha256']==item['members']['formal_binding']['sha256'],'Linux whole acceptance differs')
        need(saved['views']==receipt['views'] and rr['views']==ar['views'] and saved['native_comparison']==receipt['native_comparison']==ar['native_comparison'] and receipt['native_comparison']['accepted'] is True and receipt['native_comparison']['max_abs_difference']<=1e-12,'Native or saved audit gate failed')
    by={r['id']:dict(r,_verified_root=root_pin,_verified_index=index_pin) for r in index['records']}
    return dict(root_adopted=True,root_sha256=root_pin['sha256'],index_sha256=index_pin['sha256'],records=[by[i] for i in ordered_FedAvg_ids],index_preserves_original_order=True,returned_order_explicit_caller_order=True)
def load_cache(record):
    # Revalidate adoption and identity; zero CNN/fit, only saved arrays and train/valid labels.
    selected=verify_root_index(record['_verified_root'],record['_verified_index'],read(HERE/'EXACT100_REFS.json')['ordered_ids'])
    item=next(r for r in selected['records'] if r['id']==record['id']);need(item==record,'Cache handle changed')
    import numpy as np
    prefix=HERE/'original_prefix_functions.py.txt';need(sha(prefix)==PREFIX_SHA,'Original label reader changed')
    ns=dict(np=np,hashlib=hashlib,zipfile=zipfile,require=need);exec(compile(prefix.read_text('utf8'),str(prefix),'exec'),ns)
    metadata=pinned(item['metadata']);y,_=ns['read_prefix'](metadata,'Smiling',202599,182637);s,_=ns['read_prefix'](metadata,'Male',202599,182637)
    with np.load(pinned(item['members']['predictions']),allow_pickle=False) as z:
        need(set(z.files)=={'root_image_ids','valid_image_ids','root_margins','valid_margins','prediction_native','prediction_raw','prediction_shared_calibration'},'Changed cache keys')
        root_ids=z['root_image_ids'].copy();valid_ids=z['valid_image_ids'].copy();rm=z['root_margins'].copy();vm=z['valid_margins'].copy()
    need(root_ids.ndim==1 and np.issubdtype(root_ids.dtype,np.integer) and len(root_ids)==len(np.unique(root_ids)) and ((root_ids>=1)&(root_ids<=162770)).all(),'Wrong root sample IDs')
    need(np.array_equal(valid_ids,np.arange(162771,182638)) and ns['array_sha'](valid_ids)==VALID_SHA,'Wrong valid IDs')
    need(ns['array_sha'](root_ids)==item['inventory_record']['data_contract']['root_image_ids_sha256'],'Changed original root IDs')
    need(rm.dtype==vm.dtype==np.float32 and rm.shape==root_ids.shape and vm.shape==valid_ids.shape and np.isfinite(rm).all() and np.isfinite(vm).all(),'Invalid saved float32 margins')
    return dict(root_margins=rm,valid_margins=vm,root_y=y[root_ids-1],valid_y=y[valid_ids-1],root_sensitive=s[root_ids-1],valid_sensitive=s[valid_ids-1],root_image_id=root_ids,valid_image_id=valid_ids)
