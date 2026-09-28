"""Freeze existing worker jobs into the single bounded parent queue."""
import hashlib
import json
from pathlib import Path, PurePosixPath

HERE=Path(__file__).resolve().parent
ROOT=PurePosixPath('/workspace/GuardFed-celeba-expanded')
BASE=ROOT/'deployment/baseline_adapters_20260928/screen_20260928'
STAGE=ROOT/'results/revision_20260928/celeba_baseline_screen_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
jobs=[];sources={}
for algorithm,folder,pattern in [('FedAA','fedaa','screen_jobs/jobs/*.json'),('LASA','lasa','jobs/screen/*.json')]:
    for path in sorted((HERE/folder).glob(pattern)):
        j=json.loads(path.read_text());remote=BASE/path.relative_to(HERE).as_posix()
        jobs.append(dict(id=j['id'],algorithm=algorithm,job=str(remote),job_sha256=sha(path),output=str(STAGE/'runs'/j['id']),
            **{k:j[k] for k in ['method','distribution','attack','tuning_candidate']}))
        for name,h in j['source_hashes'].items():
            assert name not in sources or sources[name]==h
            sources[name]=h
jobs.sort(key=lambda j:(j['distribution'],j['attack'],j['tuning_candidate'],j['algorithm']))
# Interleave the two methods; all four initial scenarios for each default ordering.
fa=[j for j in jobs if j['algorithm']=='FedAA'];la=[j for j in jobs if j['algorithm']=='LASA']
jobs=[j for pair in zip(fa,la) for j in pair]
for f in HERE.rglob('*'):
    if f.is_file() and '__pycache__' not in f.parts and (f.suffix=='.py' or f.name=='protocol.json'):
        if f.name=='prepare_parent.py':continue
        sources[str((BASE/f.relative_to(HERE).as_posix()).relative_to(ROOT))]=sha(f)
manifest=dict(stage='celeba_baseline_screen_v1',concurrency=8,planned=64,rounds=70,seed=91001,evaluation_split='valid',
    protocol_sha256=sha(HERE/'PROTOCOL.md'),source_hashes=sources,jobs=jobs,
    gate_acceptance=str(STAGE/'preflight/acceptance.json'),
    gate_acceptance_sha256=sha(Path('docs/server_deployment_20260923/training_20260923/celeba_baseline_screen_v1/preflight/acceptance.json')),
    selection='One candidate per method by equal mean frozen score over IID/non-IID x Benign/S-DFA; exact ties lexicographic candidate',
    limitations='n=1 validation search; all candidate outcomes retained; not full17methods or finaltest')
assert len(jobs)==len({j['id'] for j in jobs})==64
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'jobs':len(jobs),'source_hashes':len(sources),'manifest_sha256':sha(HERE/'manifest.json')}))
