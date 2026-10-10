"""Pure interface fixtures and source equality; never runs a model or real gate."""
import argparse,ast,copy,hashlib,importlib.util,json,math,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
from identity import make_job,validate_job,validate_grid,require
from writer_policy import sanitize_result,check_sidecar
from runtime import functions,unchanged_science_segments
import common
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=ROOT/'tmp/celeba_hybrid_screen_execution_20261009'
read=lambda p:json.loads(p.read_bytes())
protocol=read(OLD/'runtime_protocol.json');candidate=protocol['candidates'][0]
# All values below are structural fixtures, not selected or measured results.
refs=[dict(distribution=d,attack=a,seed=91001) for d in ('IID','non-IID') for a in ('Benign','S-DFA')]
jobs=[make_job(protocol,candidate,d,a,s,'fullcoverage') for d in ('IID','non-IID') for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA') for s in range(91001,91011) if (d,a,s) not in {(r['distribution'],r['attack'],r['seed']) for r in refs}]
gates=[make_job(protocol,candidate,'non-IID',a,91002,'canary',ref) for a,ref in [('Benign',False),('Benign',True),('S-DFA',False),('S-DFA',True),('F Flip',False),('FedSA',False),('Sp-DFA',False)]]
for job in jobs+gates:validate_job(job,protocol,candidate)
validate_grid(jobs,gates,refs)
writer_tests=0;rejects=0
for attack,count in [('F Flip',4),('S-DFA',4),('Sp-DFA',2)]:
    job=make_job(protocol,candidate,'non-IID',attack,91002,'canary')
    audits=[]
    for i in range(20):
        types=[] if i>=4 else ['fflip','foe'] if attack=='S-DFA' else ['fflip'] if attack=='F Flip' or i<2 else ['foe']
        row=dict(client_id=i,is_malicious=i<4,samples=100,attack_types=types)
        if i<count:row.update(fflip_mode='all_unprivileged',fflip_overwrite_ratio=1.0,fflip_requires_full_flip=False,label_changed_count=0,fflip_label_corr_after=math.nan)
        audits.append(row)
    value=dict(config=job['config'],method=job['method'],attack=attack,attack_audit=audits,metrics={'accuracy':.5,'aeod':0.,'aspd':0.})
    clean,changes=sanitize_result(value,job,'a'*64,'a'*64)
    side=dict(status='EXPLICIT_UNDEFINED_DIAGNOSTIC_ONLY',job_sha256='a'*64,original_gate_sha256='b'*64,original_core_sha256='c'*64,undefined_values=changes)
    check_sidecar(clean,side,job,'a'*64,'b'*64,'c'*64);require(len(changes)==count and math.isnan(value['attack_audit'][0]['fflip_label_corr_after']),'Fixture altered original NaN');writer_tests+=1
    bad=copy.deepcopy(value);bad['metrics']['accuracy']=math.nan
    try:sanitize_result(bad,job,'a'*64,'a'*64)
    except ValueError:rejects+=1
    else:raise AssertionError('Unknown metric NaN accepted')
    bad=copy.deepcopy(value);bad['attack_audit'][0]['attack_types']=['other']
    try:sanitize_result(bad,job,'a'*64,'a'*64)
    except ValueError:rejects+=1
    else:raise AssertionError('Wrong attack identity accepted')
for call in [lambda:make_job(protocol,candidate,'IID','Benign',91001,'fullcoverage'),lambda:validate_grid(jobs[:-1],gates,refs),lambda:common.local_identity()]:
    try:call()
    except ValueError:rejects+=1
    else:raise AssertionError('Invalid/unbound scope accepted')
# Existing legal S-DFA policy must serialize identically; input is still a fixture.
spec=importlib.util.spec_from_file_location('old_writer_fixture',OLD/'writer_policy.py');old_writer=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_writer)
old_job=read(OLD/next(e['job'] for e in read(OLD/'screen_scope.json')['jobs'] if e['attack']=='S-DFA'))
old_value=dict(config=old_job['config'],method=old_job['method'],attack='S-DFA',attack_audit=[])
for i in range(20):
    r=dict(client_id=i,is_malicious=i<4,samples=100,attack_types=['fflip','foe'] if i<4 else [])
    if i<4:r.update(fflip_mode='all_unprivileged',fflip_overwrite_ratio=1.0,fflip_requires_full_flip=False,label_changed_count=0,fflip_label_corr_after=math.nan)
    old_value['attack_audit'].append(r)
require(sanitize_result(old_value,old_job,'a'*64,'a'*64)==old_writer.sanitize_result(old_value,old_job,'a'*64,'a'*64),'Old S-DFA serialization changed')
# Copy only two small sources to a temporary fixture; no model/output artifacts.
with tempfile.TemporaryDirectory(dir=HERE,prefix='interface_source_fixture_') as folder:
    stage=Path(folder);(stage/'body.py').write_bytes((OLD/'body.py').read_bytes());(stage/'writer_policy.py').write_bytes((HERE/'writer_policy.py').read_bytes())
    body,factory=functions(stage,Path('/UNEXECUTED_REPO'),dict(original_screen=str(OLD)))
    original=unchanged_science_segments(OLD/'body.py');copied=unchanged_science_segments(stage/'body.py')
    require(original==copied,'Whole scientific body source changed')
    scope=dict(jobs=[],scope_file='NONE');run,checked,compare=factory(body,scope)
    require(run.__code__ is body.run_one.__code__ and compare.__code__ is body.compare.__code__,'Science/comparison code object changed')
    require('torch' not in sys.modules,'Interface check must not import torch')
out=dict(status='SOURCE_AND_PURE_INTERFACE_PASS_NO_REAL_IMAGE_GATE',metadata_jobs_checked=103,new_grid=96,reused_refs=4,canary_interfaces=7,writer_roundtrips=writer_tests,old_S_DFA_serialization_exact=True,refusals=rejects,whole_body_source_byte_exact=True,run_one_and_compare_code_objects_exact=True,torch_imported=False,CNN=False,actual_recipe_selected=False,protocol_frozen=False)
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'SELF_CHECK.json');args=parser.parse_args()
with args.output.open('x',encoding='utf8',newline='\n') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
