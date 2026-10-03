"""Pipeline-only old-worker reference at the SAME three-round horizon.

Only its frozen screen validator is normalized from3 to70 for verification.
The unmodified old worker, source/adapters and actual3-round execution are used.
Reference artifacts never enter the formal192+8 matrix.
"""
import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path('/workspace/GuardFed-celeba-expanded')
OLD=ROOT/'deployment/baseline_adapters_20260928/screen_20260928/lasa'
sys.path.insert(0,str(OLD))
spec=importlib.util.spec_from_file_location('original_lasa_reference',OLD/'worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
original_validate=worker.validate_job

def validate(job):
    assert job['config']['rounds']==3 and job['config']['seed']==91001
    assert job['distribution']=='non-IID' and job['attack'] in ['Benign','S-DFA']
    assert job['tuning_candidate']=='LASA_s0.3_l2_lr0.001'
    assert job['pipeline_reference_only'] is True
    normalized=copy.deepcopy(job);normalized['config']['rounds']=70
    original_validate(normalized)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    worker.validate_job=validate
    worker.run(ROOT,a.job,a.out)
    (a.out/'PIPELINE_REFERENCE_ONLY.json').write_text(json.dumps({'formal_results':False,'actual_rounds':3,'reference_worker':str(OLD/'worker.py'),'reason':'Match short-horizon extra reporting/RNG behavior; never treat as a completed70-round screen record'}))
