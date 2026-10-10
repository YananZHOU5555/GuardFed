"""Freeze root-specified adopted C80/entry pins once. No stage or scientific computation."""
from pathlib import Path
import argparse,hashlib,json,shutil
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parent
T=Path('docs/server_deployment_20260923/training_20260923');S=T/'server_reactivation_20261009'
C=Path('tmp/celeba_mechanism_valid_C_after70_20261010');D=C/'execution_candidate/backups/incremental_20261010T025719Z'
TABLE=T/'celeba_mechanism_v1/three_view_C_eight_scenes_20261010';PARENT='b57ae3c07e1013c17820c2eb038d701a356b27c6'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root-freeze',type=Path,required=True);ap.add_argument('--root-freeze-sha256',required=True);a=ap.parse_args()
    assert sha(a.root_freeze)==a.root_freeze_sha256;f=read(a.root_freeze)
    assert f['status']=='ROOT_ACTUAL42_FREEZE_INPUTS' and f['parent_commit']==PARENT
    assert not (B/'ACTUAL_CLOSED_INPUTS.json').exists()
    rows={};snaps={}
    def pin(p):return dict(path=p.as_posix(),sha256=sha(R/p))
    def add(p):
        assert (R/p).is_file() and not (R/p).is_symlink();rows[p.as_posix()]=sha(R/p)
    def snapshot(p,digest):
        assert sha(R/p)==digest;q=B/'frozen_inputs'/p;assert not q.exists();q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/p,q);assert sha(q)==digest
        rel=q.relative_to(R).as_posix();rows[rel]=digest;snaps[rel]=('experimental/'+p.relative_to('tmp').as_posix()) if p.parts[0]=='tmp' else p.as_posix();return dict(path=rel,sha256=digest)
    draft=read(B/'CLOSED_C10_MAPPING_DRAFT.json')
    assert draft['parent_commit']==PARENT
    for row in draft['files'].values():p=Path(row['path']);assert sha(R/p)==row['sha256'];add(p)
    table_root=TABLE/'ROOT_VERIFICATION.json';assert sha(R/table_root)==f['table_root_sha256']
    table_seal=TABLE/'ACTUAL_FILES_SHA256.json';assert sha(R/table_seal)==f['table_seal_sha256']
    for p in (R/TABLE).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:add(p.relative_to(R))
    review=Path(f['table_review']['path']);assert sha(R/review)==f['table_review']['sha256']
    for p in (R/review.parent).iterdir():
        if p.is_file():add(p.relative_to(R))
    state=None
    for name,digest in f['frozen_entry_pins'].items():
        bound=snapshot(Path(name),digest)
        if name==(T/'TRAINING_STATE.json').as_posix():state=bound
    assert state is not None
    live=Path(f['formal_live']['path']);assert sha(R/live)==f['formal_live']['sha256'];add(live)
    for name,digest in f.get('extra_pins',{}).items():assert sha(R/name)==digest;add(Path(name))
    previous=T/'publication_closed_increment41_verified_20261010.json';add(previous);add(T/'publication_closed_increment41_20261010.json')
    recover={}
    for name in f['parent_recovery_paths']:
        p=Path(name);recover['experimental/'+p.relative_to('tmp').as_posix() if p.parts[0]=='tmp' else p.as_posix()]=sha(R/p)
    bindings=[pin(D/n) for n in ['incremental_valid_three_views.tar.gz','backup_receipt.json','OFFSERVER_VERIFICATION.json']]
    sealed=[]
    for p in [C/'PACKAGE_SHA256.json',TABLE/'ACTUAL_FILES_SHA256.json']:
        obj=read(R/p);members=obj.get('members') or obj.get('files');assert isinstance(members,(list,dict));sealed.append(dict(**pin(p),members=len(members)))
    out=dict(status='ROOT_CLOSED_INCREMENT42_INPUTS',parent_commit=PARENT,counts=dict(native=180,three_view=180,FL_new=18,Hybrid=23,baseline_valid=900),
        roots=dict(C10=pin(D/'ROOT_ADOPTION_REVIEW.json'),table=pin(table_root),state=state,live=pin(live),previous_publication=pin(previous)),table_review=pin(review),artifact_bindings=bindings,sealed_deliveries=sealed,parent_recovery_blobs=recover,frozen_snapshot_destinations=snaps,
        files=[dict(path=p,sha256=d) for p,d in sorted(rows.items())],omitted_members=draft['omissions'],new_archives={('experimental/'+p[4:] if p.startswith('tmp/') else p):d for p,d in rows.items() if p.endswith(('.tar','.tar.gz'))},root_freeze_sha256=a.root_freeze_sha256,measurement_utc=f['measurement_utc'],C_table_pairs=80,complete_English_reply_C_scenes=6)
    assert len(out['new_archives'])==1
    with (B/'ACTUAL_CLOSED_INPUTS.json').open('x',encoding='utf8',newline='\n') as o:json.dump(out,o,ensure_ascii=False,indent=2);o.write('\n')
    print(json.dumps(dict(files=len(rows),frozen_files=len(snaps),sha256=sha(B/'ACTUAL_CLOSED_INPUTS.json'))))
if __name__=='__main__':main()
