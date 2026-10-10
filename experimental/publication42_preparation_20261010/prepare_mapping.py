"""List already closed C10 inputs; do not freeze live entries or a future C80 adoption."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
C=R/'tmp/celeba_mechanism_valid_C_after70_20261010'
D=C/'execution_candidate/backups/incremental_20261010T025719Z'
PARENT='b57ae3c07e1013c17820c2eb038d701a356b27c6'
root=D/'ROOT_ADOPTION_REVIEW.json'
assert sha(root)=='3fc1e49e927a971a577d648dd9a7ff44ec7ac81552ea250026349d4f2e06d615'
r=read(root);assert r['accepted_new']==10 and r['cumulative_three_view_models']==180
files={};omissions=[]
def add(p):
    assert p.is_file() and not p.is_symlink()
    rel=p.relative_to(R);dest=('experimental/'+rel.relative_to('tmp').as_posix()) if rel.parts[0]=='tmp' else rel.as_posix()
    files[dest]=dict(path=rel.as_posix(),sha256=sha(p),bytes=p.stat().st_size)
for p in C.iterdir():
    if p.is_file():
        if p.name=='source_prepared.tar.gz':
            omissions.append(dict(path=p.relative_to(R).as_posix(),sha256=sha(p),reason='Upload-only source envelope; its exact source members are retained separately and in the unique accepted-result archive. No claim of historical tar-byte regeneration.'))
        else:add(p)
for p in (C/'execution_candidate').iterdir():
    if p.is_file():
        if p.name=='root_deployment_source.tar.gz':
            omissions.append(dict(path=p.relative_to(R).as_posix(),sha256=sha(p),reason='Upload-only deployment envelope. Exact approval, source and runtime records are separately retained and bound in the accepted archive; no old model or extra result archive.'))
        else:add(p)
for p in D.iterdir():
    if p.is_file():add(p)
for name in ['celeba_mechanism_C_after70_root_operations_20261010','celeba_mechanism_C_after70_source_review_20261010','celeba_mechanism_C_after70_root_review_20261010']:
    for p in (R/'tmp'/name).iterdir():
        if p.is_file():add(p)
assert sum(n.endswith(('.tar','.tar.gz')) for n in files)==1
out=dict(status='PREPARED_CLOSED_C10_MAPPING_ONLY_NOT_FINAL_PUBLICATION_INPUTS',parent_commit=PARENT,files=files,omissions=omissions,
    required_later=['Actual canonical C80 ROOT_VERIFICATION and its file seal','Actual independent C80 arithmetic review','Root-frozen STATE/RUNNING/completion/overview/handoff/updater bytes and measured live record','Actual41 published receipt and remote verification','Root ordering-error journal and supplementary audit paths if outside the enumerated folders'],
    recovery='verified_extract/ arrays and copied source are restored from the one original 103-member accepted archive; old native180, FL18, Hybrid23 archives remain in parent41, not repackaged.',
    no_git_mutation=True,no_new_statistics=True,actual_C80_adopted_claim=False)
with (B/'CLOSED_C10_MAPPING_DRAFT.json').open('x',encoding='utf8',newline='\n') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(mapped_files=len(files),archives=1,sha256=sha(B/'CLOSED_C10_MAPPING_DRAFT.json'))))
