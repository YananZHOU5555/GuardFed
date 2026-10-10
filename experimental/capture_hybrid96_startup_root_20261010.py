"""One read-only formal worker/GPU/progress observation; no acceptance."""
from pathlib import Path
import ast,datetime,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'tmp/celeba_hybrid_fullcoverage_root_binding_v3_20261010'
old=ROOT/'tmp/capture_hybrid7_startup_root_20261010.py'
source=old.read_text('utf8')
node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='script' for t in n.targets))
script=ast.literal_eval(node.value)
changes={
    "str(stage/'run_canaries.py')":"str(stage/'run_fullcoverage.py')",
    "stage/'gate_runs'/ident/'progress.json'":"stage/'runs'/ident/'progress.json'",
    "manifest['preflight_jobs']":"manifest['jobs']",
    "'READONLY_ACTUAL_HYBRID7_STARTUP_OBSERVATION_NOT_ACCEPTANCE'":"'READONLY_ACTUAL_HYBRID96_STARTUP_OBSERVATION_NOT_ACCEPTANCE'",
    "['supervisorctl','status','guardfed_celeba_hybrid_fullcoverage_canary']":"['supervisorctl','status','guardfed_celeba_hybrid_fullcoverage']",
    "stage.parent/'canary_operations/canaries.stdout.log'":"stage.parent/'fullcoverage_operations/fullcoverage.stdout.log'",
    "stage.parent/'canary_operations/canaries.stderr.log'":"stage.parent/'fullcoverage_operations/fullcoverage.stderr.log'",
}
for old_text,new_text in changes.items():
    assert script.count(old_text)>0,old_text
    script=script.replace(old_text,new_text)
extra="""
result['physical_gpu_processes']=command(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'])
result['queue_progress']=read(stage/'queue_progress.json') if (stage/'queue_progress.json').exists() else None
result['new_execution_authorization']=read(stage/'EXECUTION_AUTHORIZATION.json')
for worker in result['workers']:
 path=stage/'runs'/worker['id']/'provenance.json'
 worker['provenance']=read(path) if path.exists() else None
 worker['provenance_sha256']=sha(path) if path.exists() else None
"""
assert script.count('print(json.dumps(result))')==1
script=script.replace('print(json.dumps(result))',extra+'\nprint(json.dumps(result))')
ast.parse(script)
def main():
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55',"env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 107 ionice -c 3 nice -n 10 python -B -"]
    r=subprocess.run(command,input=script.encode(),capture_output=True,timeout=60)
    now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    with (HERE/('FORMAL_OBSERVATION_COMMAND_'+now+'.json')).open('x',encoding='utf8') as f:json.dump(dict(returncode=r.returncode,stderr=r.stderr.decode(errors='replace')),f,indent=2);f.write('\n')
    r.check_returncode();value=json.loads(r.stdout);path=HERE/('FORMAL_OBSERVATION_'+now+'.json')
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),status=value['status'],
        workers=[dict(id=w['id'],pid=w['pid'],round=w['progress'].get('round') if w['progress'] else None) for w in value['workers']],
        terminal_observed=len(value['terminal_records']),failures=value['failure_files'],main_completed=value['main_completed'],formal100_started=value['formal100_started'])))
if __name__=='__main__':main()
