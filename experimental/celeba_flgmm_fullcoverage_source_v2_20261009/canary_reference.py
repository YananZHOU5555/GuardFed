"""Isolated original-worker same-horizon reference; four declared horizon constants only."""
import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
from screen_common import HERE,authorized,digest,local_identity,read,repo_identity,write_json

ORIGINAL_WORKER_SHA='a3fe21a5a334b63de9c2167ed28293cd7f38d4411edf11de7d226d895b550fdc'


def horizon_functions(source):
    """Pure AST adaptation, separately testable without importing Torch or scientific code."""
    tree=ast.parse(source);output=[]
    for name,expected in [('validate_job',{70:1,71:0}),('run',{70:2,71:1})]:
        node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
        counts={70:0,71:0}
        class Horizon(ast.NodeTransformer):
            def visit_Constant(self,n):
                if type(n.value) is int and n.value in counts:
                    counts[n.value]+=1
                    return ast.copy_location(ast.Constant(3 if n.value==70 else 4),n)
                return n
        node=Horizon().visit(node);assert counts==expected,(name,counts)
        output.append(ast.fix_missing_locations(node))
    return ast.Module(body=output,type_ignores=[])


def instrument(worker,out):
    """Reuse measured v3 import/training RNG boundary; no seeds, draws or model changes."""
    import numpy as np
    import torch
    from rng_capture import RNGCapture,plain
    original_loader=worker.load_core
    def load(repo):
        recorder=RNGCapture();recorder.install()
        core=original_loader(repo);recorder.begin_training()
        write_json(out/'import_rng_evidence.json',recorder.import_evidence())
        original_run=core.run_experiment
        def save(prefix):
            torch.cuda.synchronize();captured=recorder.snapshot()
            write_json(out/(prefix+'_rng.json'),plain(dict(python=random.getstate(),numpy_legacy=np.random.get_state(),numpy_generators=captured['training_states'])))
            write_json(out/(prefix+'_rng_scope.json'),captured)
            torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),out/(prefix+'_rng.pt'))
        def run(*args,**kwargs):
            previous=kwargs.get('progress_callback')
            def progress(item):
                if previous:previous(item)
                save('round_%03d'%item['round'])
            kwargs['progress_callback']=progress
            try:
                result=original_run(*args,**kwargs);save('final');return result
            finally:recorder.restore()
        core.run_experiment=run
        return core
    worker.load_core=load


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--attack',choices=['Benign','S-DFA'],required=True)
    args=parser.parse_args();protocol,manifest=local_identity();authorized('seven_same_horizon_3round_canaries')
    from run_one import check_runtime
    check_runtime(protocol);before=repo_identity(args.repo,protocol)
    item=next(r for r in manifest['reused_jobs'] if r['distribution']=='non-IID' and r['attack']==args.attack)
    legacy=Path(item['legacy_release']);source=legacy/'source/worker.py'
    assert digest(source)==ORIGINAL_WORKER_SHA
    assert digest(legacy/'PACKAGE_SHA256.json')=='aec95ceb5e8c9b7aa9cec89e9d70d2700c2f242c29e918792088648d6269bad4'
    for name,expected in read(legacy/'PACKAGE_SHA256.json')['files'].items():assert digest(legacy/name)==expected
    original_job=legacy/'jobs'/item['job'];assert digest(original_job)==item['job_sha256']
    job=read(original_job);assert job['config']['rounds']==70
    job['config']['rounds']=3
    control=HERE/'preflight/reference_jobs';control.mkdir(parents=True,exist_ok=True)
    path=control/(item['id']+'.json');assert not path.exists();write_json(path,job)
    out=HERE/'preflight/references'/item['id'];out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists()
    spec=importlib.util.spec_from_file_location('original_flgmm_same_horizon',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    exec(compile(horizon_functions(source.read_text()),str(source)+'[CANARY3_HORIZON_ONLY]','exec'),module.__dict__)
    module.validate_job(job,read(legacy/'source/protocol.json'))
    instrument(module,out)
    module.run(args.repo,path,out)
    assert repo_identity(args.repo,protocol)==before
    write_json(out/'REFERENCE_IDENTITY.json',dict(status='PASS_CANARY_NOT_SCIENTIFIC_SCREEN',
        original_worker_sha256=ORIGINAL_WORKER_SHA,original_job_sha256=item['job_sha256'],
        package_sha256=digest(HERE/'PACKAGE_SHA256.json'),changed_constants={'validate_job':{'70_to_3':1},'run':{'70_to_3':2,'71_to_4':1}},
        original_screen_status_labels_retained=True,formal_table_eligible=False,source_before=before,source_after=before))


if __name__=='__main__':main()
