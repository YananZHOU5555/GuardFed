"""Later root action: original Windows saved-array AST block; never whole PASS."""
from pathlib import Path
import argparse, ast, hashlib, importlib.util, json, os, sys, types, zipfile
from contract import CANDIDATE, HERE, IDS, PACKAGE, ROOT, SAVED_SOURCE_SHA, digest_arg, fpath, gate_proof, linux_proof, need, read, save, sha, utc, volume

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transport-proof',type=Path,required=True)
    p.add_argument('--transport-proof-sha256',type=digest_arg,required=True)
    p.add_argument('--linux-proof-sha256',type=digest_arg,required=True)
    p.add_argument('--metadata-npz',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-original-cached-root-refit',action='store_true',required=True)
    a=p.parse_args()
    need(sys.platform=='win32' and __debug__,'Explicit Windows F offserver host without -O required')
    vol=volume();need(not a.output.exists() and not a.output.with_suffix('.failure.json').exists(),'Preserve prior result/failure; no retries')
    need(not a.output.resolve().is_relative_to(HERE),'Prepared source stays immutable')
    need(sha(a.transport_proof)==a.transport_proof_sha256,'Actual transport proof differs')
    transport=read(a.transport_proof)
    need(transport['status']=='FLGMM10_F_TRANSPORT_ALL_MEMBERS_SHA_PASS_NOT_SCIENTIFIC_ACCEPTANCE' and transport['member_count']==24,'Transport not verified')
    need(transport['linux_proof_sha256']==a.linux_proof_sha256,'Wrong whole Linux proof binding')
    extract=fpath(transport['verified_extract']);metadata=fpath(a.metadata_npz)
    need(sha(extract/'LINUX_SAVED_CHECK.json')==a.linux_proof_sha256,'Linux proof SHA differs')
    linux_proof(read(extract/'LINUX_SAVED_CHECK.json'))
    expected={'bundle/GATE_RESULT.json','bundle/metadata_receipt.json','LINUX_SAVED_CHECK.json'}|{'bundle/'+i+'/'+n for i in IDS for n in ('receipt.json','validation_predictions.npz')}
    need(set(transport['members'])==expected,'Exact10 member inventory differs')
    for rel,pin in transport['members'].items():
        path=extract/rel
        need(path.resolve().is_relative_to(extract) and sha(path)==pin['sha256'] and path.stat().st_size==pin['bytes'],'Transport member changed: '+rel)
    gate=read(extract/'bundle/GATE_RESULT.json');gate_proof(gate)
    need(sha(extract/'bundle/GATE_RESULT.json')==transport['gate_result_sha256'],'Gate SHA differs')
    need(read(extract/'LINUX_SAVED_CHECK.json')['gate_result_sha256']==transport['gate_result_sha256'],'Whole Linux proof/gate differs')
    need(sha(metadata)=='161f8028f1c29ba470afa60cbd9fb54d7bf61b3cec5c525830ad7a3ef7ab2091','Original F metadata differs')
    saved_dir=ROOT/'tmp/celeba_added_cnn_exact3_scientific_acceptance_20261010'
    need(sha(saved_dir/'saved_binding.py')=='6915d436965cf3cf91fdc9ab566c2861c7a0becdde890722f1c4e12aa4b7cc52','Original binding differs')
    b=load('_fl47_original_saved_binding',saved_dir/'saved_binding.py')
    b.CANDIDATE,b.PACKAGE_SHA=CANDIDATE,PACKAGE  # Private metadata binding only.
    c=load('_fl47_local_candidate',CANDIDATE/'candidate.py');c.package_check(PACKAGE)
    manifest=c.read(CANDIDATE/'MANIFEST.json');c.validate_manifest(manifest)
    original_resolver=c.resolve_origin
    def local_metadata_origin(origin,m,runtime):
        pin=m['path_map'][str(origin).replace('\\','/')]
        return Path(origin) if pin['kind']=='server' else original_resolver(origin,m,runtime)
    c.resolve_origin=local_metadata_origin  # Root startup's actual local metadata adapter.
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import numpy as np,pandas as pd,torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);need(torch.cuda.device_count()==0,'CPU only')
    bridge,ev=c.bound_bridge(manifest,runtime=False,torch_module=torch,pandas_module=pd)
    dependency=ROOT/'tmp/revision-publish-20260928'
    need(sha(dependency/'src/data_loader.py')=='41723cc5dd57d7d7c4878bb93989426569279d608c0357cf0b6429c8c4a2a091','Original core dependency differs')
    sys.path.insert(0,str(dependency));source=CANDIDATE/'originals';core=c.load('_fl47_saved_core',source/'core.py')
    metadata_ns=dict(ev.rebuild_root.__globals__,zipfile=zipfile)
    prefix_node=next(n for n in ast.parse((source/'replay.py').read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='read_prefix')
    exec(compile(ast.Module(body=[prefix_node],type_ignores=[]),'<original read_prefix>','exec'),metadata_ns)
    with np.load(metadata,allow_pickle=False) as z:ids,split=z['image_id'],z['split']
    need(np.array_equal(ids,np.arange(1,202600)) and np.array_equal(np.flatnonzero(split==1),np.arange(162770,182637)),'Original ordered metadata IDs/split differ')
    y,_=metadata_ns['read_prefix'](metadata,'Smiling',202599,182637)
    sensitive,_=metadata_ns['read_prefix'](metadata,'Male',202599,182637)
    saved_text=b.SOURCE.read_text(encoding='utf-8');need(sha(b.SOURCE)==SAVED_SOURCE_SHA,'Whole saved source changed')
    saved_fn=next(n for n in ast.parse(saved_text).body if isinstance(n,ast.FunctionDef) and n.name=='check_saved')
    block=next(n for n in saved_fn.body if isinstance(n,ast.With))
    block_sha=hashlib.sha256(ast.dump(block,include_attributes=False).encode()).hexdigest()
    need(block_sha=='170bed968fecb3cd477603132b4d185a64da6f7d35ab120776449be9aae6b0ec','Original array block changed')
    binding_fn=next(n for n in ast.parse((saved_dir/'saved_binding.py').read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='verify_one')
    preamble=[]
    for statement in binding_fn.body:
        if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call) and isinstance(statement.value.func,ast.Name) and statement.value.func.id=='exec':break
        preamble.append(statement)
    diagnostic=ROOT/'tmp/diagnose_added_cnn_exact3_offserver_root_20261010.py'
    need(sha(diagnostic)=='9ef1a2f4842371eb96fb83bd1cc635f040edb1b0113eab69117e52b28064dfb7','Original receipt diff helper changed')
    diff_node=next(n for n in ast.parse(diagnostic.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='diff')
    diff_ns={};exec(compile(ast.Module(body=[diff_node],type_ignores=[]),'<original receipt diff>','exec'),diff_ns)
    rows=[]
    try:
        for row,receipt in zip(manifest['records'],gate['receipts']):
            run=extract/'bundle'/row['id'];need(read(run/'receipt.json')==receipt,'Gate/per-record receipt differs')
            ctx=dict(b.__dict__,row=row,run=run,bridge=bridge,evaluator=ev,core=core,ids=ids,y=y,sensitive=sensitive,root_authorized_cached_refit=True)
            exec(compile(ast.Module(body=preamble,type_ignores=[]),'<unchanged original verify_one guards>','exec'),ctx)
            original_result=ctx['validate_external'](None,ctx['record'],ROOT)
            cfg=core.ExperimentConfig(**ctx['record']['config'])
            root_ids,root_y,root_s,actual_root=ev.rebuild_root(core,cfg,ctx['record'],ids,y,sensitive)
            scope=dict(ctx,path=run,r=receipt,root_ids=root_ids,root_y=root_y,root_s=root_s,s=sensitive,cfg=cfg,original_result=original_result)
            exec(compile(ast.Module(body=[block],type_ignores=[]),'<exact original saved-array block>','exec'),scope)
            after=b.measure(ctx['paths'],ctx['record'])
            need(ctx['before']==ctx['second']==after,'Local original artifacts changed across check')
            differences=diff_ns['diff'](actual_root,receipt['root_reconstruction'])
            rows.append({'id':row['id'],'array_sha256':sha(run/'validation_predictions.npz'),'receipt_sha256':sha(run/'receipt.json'),'native_comparison':scope['comparison'],'root_ID_partition_original_assertions_pass':True,'root_fit_parameters_and_diagnostics_exact':True,'all_saved_predictions_and_counts_and_metrics_exact':True,'local_full_root_receipt_exact':not differences,'preserved_root_receipt_differences':differences,'artifact_observations':{'before':ctx['before'],'second_before':ctx['second'],'after':after}})
        report={'status':'FLGMM10_WINDOWS_ORIGINAL_ARRAY_BLOCK_PASS_WHOLE_NOT_EXECUTED_NO_ADOPTION','utc':utc(),'records':rows,'cached_root_refits':10,'original_saved_check_sha256':SAVED_SOURCE_SHA,'exact_saved_array_AST_block_sha256':block_sha,'linux_whole_proof_sha256':a.linux_proof_sha256,'transport_proof_sha256':a.transport_proof_sha256,'Windows_whole_check_executed':False,'Windows_whole_check_pass':None,'prior_exact3_Windows_whole_failure_preserved':True,'native_tolerance_unchanged':1e-12,'new_CNN':0,'new_training':0,'test':False,'root_adopted':False,'runtime':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'device':'cpu'},'F_volume':vol}
        save(a.output,report);print(json.dumps({'status':report['status'],'records':10,'proof_sha256':sha(a.output)}))
    except BaseException:
        import traceback
        save(a.output.with_suffix('.failure.json'),{'status':'FLGMM10_WINDOWS_ARRAY_CHECK_FAILED_PRESERVED','utc':utc(),'completed':len(rows),'traceback':traceback.format_exc(),'root_adopted':False})
        raise

if __name__=='__main__':main()
