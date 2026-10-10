"""Prepared root-invoked cached-array validation; never called during preparation."""
import argparse, ast, importlib.util, json, os, subprocess, sys, zipfile
from pathlib import Path
import saved_binding as b

def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--metadata-npz',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-original-cached-root-refit',action='store_true',required=True)
    a=p.parse_args()
    b.need(sys.platform=='win32' and __debug__, 'Root F-volume offserver host, without -O, required')
    vol=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',
        'Get-Volume -DriveLetter F | Select-Object FileSystemLabel,HealthStatus,SizeRemaining | ConvertTo-Json -Compress'],text=True))
    b.need(vol['FileSystemLabel']=='Yanan 2TB' and vol['HealthStatus']=='Healthy' and vol['SizeRemaining']>=1024**3,'Required F storage not ready')
    b.need(not a.output.exists(),'Preserve previous verification output')
    b.need(b.sha(b.CANDIDATE/'FILES_SHA256.json')==b.PACKAGE_SHA,'Candidate seal changed')
    for rel,pin in b.read(b.CANDIDATE/'FILES_SHA256.json')['files'].items():
        path=b.CANDIDATE/rel
        b.need(b.sha(path)==pin['sha256'] and path.stat().st_size==pin['bytes'],'Candidate source member changed: '+rel)
    b.need(a.bundle.resolve().is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()) and a.metadata_npz.resolve().is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()),'Bulk evidence must remain on F')
    # Root must first verify the transport archive/member inventory separately.
    b.need(b.sha(a.metadata_npz)=='161f8028f1c29ba470afa60cbd9fb54d7bf61b3cec5c525830ad7a3ef7ab2091','Original metadata SHA differs')
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import numpy as np, pandas as pd, torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    source=b.CANDIDATE/'originals'
    bridge=load('_exact3_original_offserver_bridge',source/'bridge.py')
    ev=bridge.science_bindings(torch_module=torch,pandas_module=pd)
    dependency=b.ROOT/'tmp/revision-publish-20260928'
    b.need(b.sha(dependency/'src/data_loader.py')=='41723cc5dd57d7d7c4878bb93989426569279d608c0357cf0b6429c8c4a2a091','Original core import dependency differs')
    sys.path.insert(0,str(dependency));core=load('_exact3_original_offserver_core',source/'core.py')
    ns=dict(ev.rebuild_root.__globals__,zipfile=zipfile)
    node=next(n for n in ast.parse((source/'replay.py').read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='read_prefix')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source/'replay.py'),'exec'),ns)
    with np.load(a.metadata_npz,allow_pickle=False) as z:
        ids=z['image_id'];split=z['split']
    b.need(np.array_equal(ids,np.arange(1,202600)) and np.array_equal(np.flatnonzero(split==1),np.arange(162770,182637)),'Original ordered IDs/split differ')
    y,_=ns['read_prefix'](a.metadata_npz,'Smiling',202599,182637)
    sensitive,_=ns['read_prefix'](a.metadata_npz,'Male',202599,182637)
    rows=b.read(b.CANDIDATE/'MANIFEST.json')['records'];gate=b.read(a.bundle/'GATE_RESULT.json')
    b.need(gate['status']=='EXACT3_VALID_INTERFACE_PASS_NOT_ROOT_ADOPTED' and not (a.bundle/'FAILURE.json').exists(),'Unfinished/failed exact3 gate')
    b.need([r['id'] for r in gate['receipts']]==[r['id'] for r in rows] and gate['package_sha256']==b.PACKAGE_SHA,'Gate membership/source differs')
    results=[]
    for row,receipt in zip(rows,gate['receipts']):
        b.need(b.read(a.bundle/row['id']/'receipt.json')==receipt,'Gate/per-run receipt differs')
        results.append(b.verify_one(row,a.bundle/row['id'],bridge=bridge,evaluator=ev,core=core,ids=ids,y=y,sensitive=sensitive,root_authorized_cached_refit=True))
    report={'status':'EXACT3_OFFSERVER_SAVED_ARRAY_AND_ORIGINAL_ROOT_REFIT_PASS_PENDING_ROOT_ADOPTION','records':results,'new_CNN':0,'new_training':0,'cached_root_refits':3,'test':False,'transport_verification_is_separate':True,'root_adopted':False,
            'verification_environment':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'device':'cpu'},'F_volume':vol}
    with a.output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)

if __name__=='__main__':main()
