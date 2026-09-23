import json,subprocess,sys,time
from pathlib import Path
R=Path('/workspace/GuardFed-image-deterministic')
A=R/'results/revision_20260923/celeba_deterministic_acceptance_v2'
L=R/'results/revision_20260923/celeba_deterministic_learning10_v1'
while not (A/'comparison.json').exists():
    pid=int((A/'controller.pid').read_text())
    if not Path(f'/proc/{pid}').exists():raise RuntimeError('Acceptance controller stopped without passing; do not start learning pilot')
    time.sleep(10)
comparison=json.loads((A/'comparison.json').read_text())
assert comparison['first_round_equivalence']['all_passed'] and comparison['equivalence']['all_passed']
if 'extension' in comparison:assert comparison['extension']['equivalence']['all_passed']
with (L/'runner.log').open('w') as log:
    subprocess.run([sys.executable,str(R/'scripts/run_revision_ablation.py'),'run','--manifest',str(L/'manifest.json'),'--concurrency','2'],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
out=R/'deployment/celeba_checks/deterministic_learning10_analysis'
out.mkdir(parents=True,exist_ok=True)
with (out/'analysis.log').open('w') as log:
    subprocess.run([sys.executable,str(R/'deployment/celeba_checks/analyze_validation_pilot.py'),'--manifest',str(L/'manifest.json'),'--output',str(out)],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Learning10 and same-checkpoint validation analysis completed; no test or formal run.',flush=True)
