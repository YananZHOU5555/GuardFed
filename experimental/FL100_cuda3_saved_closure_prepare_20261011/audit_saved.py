"""Future Windows offserver audit of exactly three original800 CUDA saved outputs; zero fit/CNN."""
from pathlib import Path,PurePosixPath
import argparse,ast,copy,datetime,hashlib,importlib.util,json,os,sys,tarfile,textwrap,traceback,types,zipfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE='/workspace/guardfed_checks/uniform_cuda_FL100_20261011_canary1'
REPO='/workspace/GuardFed-celeba-expanded'
IDS=['FLGMM_Tg20_L2.0_lr0.001_IID_Benign_seed91001_screen', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91001_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_Sp-DFA_seed91001_fullcoverage']
VIEWS=['native','raw','shared_calibration']
AUTHOR_SHA='a1f1be152a0aa5668b91d73d7297ea3e9275efaac8d82c3b22cfbe97b7b08fce'
FORMAL_API_SEAL_SHA='395cec9ef70de380d0f76c2acbcac88dcb3188bd388d55e4a9442f2cd0f0dc22'
METADATA_SHA='161f8028f1c29ba470afa60cbd9fb54d7bf61b3cec5c525830ad7a3ef7ab2091'
VALID_SHA='64a15cf28caf1d177ac3dcf96a4408bc21091796974b243923ca947a37b554bf'
read=lambda p:json.loads(Path(p).read_bytes())
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def load_functions(path,names,namespace):
    tree=ast.parse(path.read_text('utf8'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    need({n.name for n in nodes}==set(names),'Missing original science function')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace
def source_check(expected):
    need(sha(HERE/'FILES_SHA256.json')==expected,'Wrong audit source seal')
    for rel,p in read(HERE/'FILES_SHA256.json')['files'].items():
        q=(HERE/rel).resolve();need(q.is_relative_to(HERE) and sha(q)==p['sha256'] and q.stat().st_size==p['bytes'],'Audit source changed')
    pins=read(HERE/'SOURCE_PINS.json')
    for name in ('original_fl_audit.py.txt','original_evaluator.py.txt','original_replay.py.txt','original_full_saved_science.py.txt'):
        need(sha(HERE/name)==pins[name]['sha256'],'Original source changed')
    return pins
def verified_inputs(a):
    need(sys.platform=='win32' and __debug__,'Windows without -O required')
    pins=source_check(a.source_seal_sha256)
    need(sha(a.inputs)==a.inputs_sha256,'Actual root input SHA differs');inp=read(a.inputs)
    need(inp['root_authorized_audit'] is True and inp['exact_ids']==IDS and inp['base']==BASE,'Root audit binding absent or wrong fixed3')
    review=Path(inp['source_review']['path']);need(sha(review)==inp['source_review']['sha256'] and read(review)['source_adoptable'] is True,'Actual root source review missing')
    need(read(review)['source_seal_sha256']==a.source_seal_sha256,'Root source review is for another audit')
    guard=ROOT/pins['storage_guard']['path'];need(sha(guard)==pins['storage_guard']['sha256'],'Storage guard changed')
    spec=importlib.util.spec_from_file_location('_old800_cuda3_storage',guard);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    volume=g.check_bulk_storage()
    tp=Path(inp['transport']['path']);need(sha(tp)==inp['transport']['sha256'],'Transport proof changed');t=read(tp)
    need(t['status']=='PASS_ARCHIVE_AND_ALL_MEMBER_SHA' and t['all_members_verified'] is True,'Original transport incomplete')
    archive=Path(t['archive_path']).resolve();extract=Path(t['verified_extract']).resolve()
    if 'extract_root' in t:need(Path(t['extract_root']).resolve()==extract,'Conflicting extract path')
    need(all(p.drive.upper()=='F:' and p.is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()) for p in (archive,extract)),'Bulk outside guarded F')
    need(sha(archive)==t['archive_sha256']==inp['archive_sha256'],'Archive changed')
    index=extract/'ROOT_MEMBERS.json';need(sha(index)==t['member_index_sha256'],'Member index changed')
    members=read(index)['files'];need(members==t['members'] and len(members)==t['payload_members'],'Transport/index member disagreement')
    need(t['archive_members']==len(members)+1,'Archive member count mismatch')
    with tarfile.open(archive,'r:gz') as tar:
        mm=tar.getmembers();names=[m.name for m in mm]
        need(len(names)==len(set(names))==len(members)+1 and set(names)==set(members)|{'ROOT_MEMBERS.json'},'Missing/extra/duplicate archive member')
        for member in mm:
            rel=member.name;p=PurePosixPath(rel)
            need(member.isfile() and not p.is_absolute() and '..' not in p.parts and '\\' not in rel and ':' not in rel,'Unsafe archive member')
            local=extract.joinpath(*p.parts);need(local.resolve().is_relative_to(extract) and not local.is_symlink(),'Unsafe extracted file')
            h=hashlib.sha256();f=tar.extractfile(member)
            for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
            expected=members.get(rel,{'sha256':t['member_index_sha256'],'bytes':index.stat().st_size})
            need(h.hexdigest()==sha(local)==expected['sha256'] and member.size==local.stat().st_size==expected['bytes'],'Archive/extract/member SHA mismatch: '+rel)
            if rel.endswith('.npz'):
                allowed={'source/metadata.npz'}|{'base/runs/'+i+'/validation_predictions.npz' for i in IDS}
                need(rel in allowed,'Foreign prediction array')
            need(not rel.endswith(('.pt','.pth','.npy')),'Model/image bulk not part of fixed3 audit')
    return inp,t,extract,members,volume
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True)
    for n in ('inputs-sha256','source-seal-sha256'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=HERE/'actual';out.mkdir(exist_ok=True)
    need(not any((out/n).exists() for n in ('STARTED.json','SAVED_OUTPUTS_AUDIT.json','FAILURE.json')),'Preserve previous attempt; no retry')
    inp,t,x,members,volume=verified_inputs(a)
    save(out/'STARTED.json',dict(inputs_sha256=a.inputs_sha256,source_seal_sha256=a.source_seal_sha256,exact_ids=IDS,fit_calls=0))
    rows=[]
    try:
        def member(rel):
            need(rel in members,'Missing required transport member: '+rel);return x/rel
        from metadata_gate import verify_metadata
        frozen=read(HERE/'SELECTED3_INPUTS.json')
        need(sha(HERE/'SELECTED3_INPUTS.json')==read(HERE/'SOURCE_PINS.json')['frozen_selection_sha256'],'Frozen FL selection changed')
        verified=verify_metadata(inp,x,members,frozen);cohort_sha=frozen['cohort_sha256']
        metadata=member('source/metadata.npz');need(sha(metadata)==METADATA_SHA,'Original metadata SHA differs')
        os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        import numpy as np
        evns=load_functions(HERE/'original_evaluator.py.txt',['canonical_sha','check_binary','group_metrics','_seal_fit','predict_views','evaluate_frozen_predictions'],dict(np=np,hashlib=hashlib,json=json,VIEWS=set(VIEWS),METHODS={'FLGMM'}))
        ev=types.SimpleNamespace(**evns)
        vns=load_functions(HERE/'original_replay.py.txt',['array_sha','read_prefix','check_native'],dict(np=np,hashlib=hashlib,zipfile=zipfile,require=need,TOLERANCE=1e-12))
        v2=types.SimpleNamespace(np=np,require=need,digest=sha,VIEWS=VIEWS,check_native=vns['check_native'])
        with np.load(metadata,allow_pickle=False) as z:ids,split=z['image_id'],z['split']
        need(np.array_equal(ids,np.arange(1,202600)) and np.array_equal(np.flatnonzero(split==1),np.arange(162770,182637)),'Original metadata order/split differs')
        y,_=vns['read_prefix'](metadata,'Smiling',202599,182637);s,_=vns['read_prefix'](metadata,'Male',202599,182637)
        fn=load_functions(HERE/'original_fl_audit.py.txt',['saved_output_block'],dict(ast=ast,copy=copy,need=need))['saved_output_block']
        original=ast.parse(textwrap.dedent((HERE/'original_full_saved_science.py.txt').read_text('utf8'))).body[0]
        block=fn(original)
        from saved_fit_binding import saved_fits_for_original_predict
        for identity in IDS:
            item=verified[identity];record=item['record'];r=item['r'];receipt_path=item['receipt_path'];array_path=item['array_path'];accepted_path=item['accepted_path'];run=item['run']
            fits=saved_fits_for_original_predict(r['fits'],ev)
            with np.load(array_path,allow_pickle=False) as z:
                need(set(z.files)=={'root_image_ids','valid_image_ids','root_margins','valid_margins'}|{'prediction_'+v for v in VIEWS},'Foreign/incomplete saved array keys')
                root_ids=z['root_image_ids'].copy()
                need(root_ids.ndim==1 and len(root_ids)==r['root_reconstruction']['root_n'] and len(np.unique(root_ids))==len(root_ids) and np.issubdtype(root_ids.dtype,np.integer) and ((root_ids>=1)&(root_ids<=162770)).all(),'Wrong saved root IDs')
                need(vns['array_sha'](root_ids)==record['data_contract']['root_image_ids_sha256']==r['root_reconstruction']['root_image_ids_sha256'],'Saved root ID identity differs')
                need(z['root_margins'].shape==root_ids.shape and np.isfinite(z['root_margins']).all(),'Invalid saved root margins')
            context=dict(v2=v2,path=x/run,r=r,root_ids=root_ids,ids=ids,s=s,y=y,evaluator=ev,fits=fits,original_result={'metrics':record['prior_validation_metrics']})
            exec(compile(ast.Module(body=[block],type_ignores=[]),'<same seven original saved-output statements; fixed3; no fit>','exec'),context)
            rows.append(dict(id=identity,receipt_sha256=sha(receipt_path),arrays_sha256=sha(array_path),accepted_sha256=sha(accepted_path),checkpoint_sha256=r['checkpoint_sha256'],views=context['scored'],native_comparison=context['comparison'],original_Linux_whole_saved_accepted=True,saved_fit_hashes={k:v['fit_sha256'] for k,v in r['fits'].items()},fit_calls=0))
        for rel,pin in members.items():need(sha(x/rel)==pin['sha256'],'Member changed during audit')
        source_check(a.source_seal_sha256)
        report=dict(status='FL100_FIXED3_CUDA_SAVED_OUTPUTS_AUDIT_NO_REFIT_PASS_NOT_ROOT_ADOPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exact_ids=IDS,records=rows,transport_sha256=inp['transport']['sha256'],archive_sha256=inp['archive_sha256'],cohort_sha256=cohort_sha,source_seal_sha256=a.source_seal_sha256,inputs_sha256=a.inputs_sha256,fit_calls=0,new_CNN=0,new_training=0,test=False,root_adopted=False,uniform_FL100_complete=False,Windows_whole_refit_executed=False,Windows_exact_recalibration_claimed=False,old_CPU_failure_preserved=True,metric_values_checked=27,integer_base_counts_checked=72,prediction_rules_checked=9,F_volume=volume,scope='FL fixed3 only; original Linux whole plus SHA transport and zero-fit saved-output consistency. No CNN/fit/test/FL100-completion claim.')
        save(out/'SAVED_OUTPUTS_AUDIT.json',report);print(json.dumps(dict(status=report['status'],records=3,sha256=sha(out/'SAVED_OUTPUTS_AUDIT.json'))))
    except BaseException:
        save(out/'FAILURE.json',dict(status='OLD800_FIXED3_SAVED_AUDIT_FAILED_PRESERVED',completed=len(rows),traceback=traceback.format_exc(),root_adopted=False,automatic_retry=False));raise
if __name__=='__main__':main()
