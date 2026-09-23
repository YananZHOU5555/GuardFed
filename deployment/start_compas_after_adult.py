import json, pathlib, subprocess, sys, time
root=pathlib.Path('/workspace/GuardFed-next')
dep=pathlib.Path('/workspace/GuardFed-revision/results/revision_20260923/adult_ablation_v1/status.json')
start=time.monotonic()
while True:
 s=json.loads(dep.read_text())
 if s['failed']: raise RuntimeError('Adult dependency has failed runs')
 if s['new_complete']==s['new_total']: break
 if time.monotonic()-start>21600:raise TimeoutError('Adult dependency did not complete within six hours')
 time.sleep(10)
runner=root/'scripts/run_revision_ablation.py'
manifest=root/'results/revision_20260923/compas_ablation_v1/manifest.json'
for extra in [['--first-seeds','3'],[]]:
 subprocess.run([sys.executable,str(runner),'run','--manifest',str(manifest),'--concurrency','8',*extra],check=True)
