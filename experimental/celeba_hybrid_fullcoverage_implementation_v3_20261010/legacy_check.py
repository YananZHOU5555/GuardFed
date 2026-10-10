"""Original four Hybrid records through their unchanged original device-aware checker."""
import argparse,importlib.util,json,os,sys
from pathlib import Path
def main(release,identity):
    from common import local_identity,digest,require
    _,manifest=local_identity();rows=[r for r in manifest['reused_jobs'] if r['id']==identity and Path(r['legacy_release'])==release];require(len(rows)==1,'Not one of the four exact Hybrid references')
    os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    sys.path.insert(0,str(release));spec=importlib.util.spec_from_file_location('original_hybrid_driver',release/'driver.py');driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    import body,torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    scope=driver.read(release/'screen_scope.json');scope.update(runtime_cuda_visible_device='0',runtime_gpu_uuid=driver.read(Path(__file__).parent/'BINDINGS.json')['gpu_uuid'])
    body.verify_scope(scope);worker,_=body.modules();entry=next(e for e in scope['jobs'] if e['id']==identity);job=driver.read(release/entry['job'])
    driver.validate_entry(entry,job,scope,driver.read(release/'runtime_protocol.json'),worker)
    _,checked,_=driver.functions(body,scope);result=checked(entry,scope);require(result is not None,'Original strict result missing')
    out=dict(id=identity,candidate=job['tuning_candidate'],seed=job['config']['seed'],distribution=job['distribution'],attack=job['attack'],rounds=70,metrics=result['metrics'],checkpoint_sha256=digest(release/entry['output']/'model.pt'),job_sha256=entry['job_sha256'],source_hashes=job['source_hashes'])
    for k,v in out.items():require(rows[0]['accepted_record'][k]==v,'Adopted original reference differs: '+k)
    print(json.dumps(out))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--id',required=True);a=p.parse_args();main(a.release.resolve(),a.id)
