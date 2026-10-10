from pathlib import Path
import datetime,hashlib,json
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
B=Path('/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1')
q=json.loads((B/'formal_queue_progress.json').read_bytes())
ids=[f'minus_F_IID_F Flip_seed{s}' for s in range(91001,91011)]
done=[i for i in ids if i in q['completed']]
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scene='minus_F_IID_F Flip',expected_ids=ids,completed_ids=done,ten_complete=len(done)==10,main_terminal=len(q['completed']),active=len(q['active']),failed=q['failed'],scientific_acceptance=False)))
