"""Pin existing closures and a root-frozen state; metadata only, no Git mutation."""
from pathlib import Path
import argparse,hashlib,json
R=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(n,v):
 with (B/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--formal-live',type=Path,required=True);ap.add_argument('--root-state-frozen',action='store_true');a=ap.parse_args();assert a.root_state_frozen
 T=Path('docs/server_deployment_20260923/training_20260923');C=T/'server_reactivation_20261009';V=Path('tmp/celeba_mechanism_valid_C_after60_20261010');D=V/'execution_candidate/backups/incremental_20261010T015546Z';F=T/'celeba_mechanism_v1/three_view_C_seven_scenes_20261010'
 root=read(R/F/'ROOT_VERIFICATION.json');assert sha(R/F/'ROOT_VERIFICATION.json')=='abab0188adfa2d857316b23a282fd104c50dd282ca00ffed628b78b2accaea70'
 seal=read(R/F/'ACTUAL_FILES_SHA256.json');assert sha(R/F/'ACTUAL_FILES_SHA256.json')=='c06dd986ecbb62f6a7d3cbf0ac307900404de4204eb5ae1511cf5deb881690dc' and len(seal['files'])==35
 rows={}
 def add(p):rows[p.as_posix()]=sha(R/p)
 for name,entry in seal['files'].items():assert sha(R/F/name)==entry['sha256'];add(F/name)
 add(F/'ACTUAL_FILES_SHA256.json');add(F/'ROOT_VERIFICATION.json')
 for p in (R/D).iterdir():
  if p.is_file():add(p.relative_to(R))
 cr=read(R/D/'ROOT_ADOPTION_REVIEW.json');E=V/'execution_candidate'
 terminal=[p for p in (R/E).glob('ROOT_PROGRESS_*.json') if not p.name.endswith('.RAW.json') and sha(p)==cr['remote_terminal_proof_sha256']];assert len(terminal)==1
 add(terminal[0].relative_to(R));add(terminal[0].with_suffix('.RAW.json').relative_to(R))
 for n in ['ROOT_BACKUP_ATTEMPT.json','ROOT_BACKUP_COMMAND_RESULT.json','ROOT_BACKUP_COMMAND_STDOUT.json']:add(E/n)
 add(Path('tmp/adopt_C70_table_root_20261010.py'))
 previous=read(R/'tmp/publication39_preparation_20261010/ACTUAL_CLOSED_INPUTS.json')
 extra=[n for n in previous['extra_pins'] if n in [(T/'RUNNING.md').as_posix(),(T/'TRAINING_STATE.json').as_posix(),(T/'REBUTTAL_COMPLETION_20261009.md').as_posix(),(T/'celeba_mechanism_v1/EXECUTION.md').as_posix(),(C/'MONITOR_HANDOFF.md').as_posix(),(C/'latest_formal_live.json').as_posix(),'docs/返修实验总览.md','tmp/update_reactivation_state_20261009.py','tmp/update_completion_current_20261009.py','tmp/update_overview_closure100_root_20261009.py']]
 extra += [(C/'auxiliary_screens_20261010T020618Z.json').as_posix(),(C/'auxiliary_screens_20261010T020618Z.RAW.json').as_posix(),'tmp/celeba_flgmm_fullcoverage_root_operations_20261009/observation_20261010T020613137245Z/SNAPSHOT.json']
 refs=[Path('tmp/publication39_preparation_20261010/publish_increment39.py'),Path('tmp/publication39_preparation_20261010/verify_increment39.py'),T/'publication_closed_increment39_verified_20261010.json',T/'publication_closed_increment39_20261010.json',D/'ROOT_ADOPTION_REVIEW.json',F/'ROOT_VERIFICATION.json']
 recovery={}
 for p in [V/'FILES_SHA256.json',V/'SCOPE.json',E/'EXECUTION_SOURCE_SHA256.json',E/'ROOT_STARTUP_OBSERVATION.json',Path('docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C60_20261010/ROOT_REVIEW.json')]:
  recovery['experimental/'+p.relative_to('tmp').as_posix() if p.parts[0]=='tmp' else p.as_posix()]=sha(R/p)
 prep=dict(status='SOURCE_PREPARED_ONLY',reference_pins={p.as_posix():sha(R/p) for p in refs},C70_root_sha256=sha(R/F/'ROOT_VERIFICATION.json'),C70_seal_sha256=sha(R/F/'ACTUAL_FILES_SHA256.json'),C70_review_path=root['independent_review_path'],required_extra_paths=extra,parent_recovery_blobs=recovery,ready_manifest=[dict(path=n,sha256=d) for n,d in sorted(rows.items())])
 save('PREPARED_INPUTS.json',prep)
 lp=a.formal_live.resolve().relative_to(R);assert lp.parent==C
 paths=dict(C10_root=D/'ROOT_ADOPTION_REVIEW.json',C70_root=F/'ROOT_VERIFICATION.json',state=T/'TRAINING_STATE.json',formal_live=lp,previous_publication=T/'publication_closed_increment39_verified_20261010.json')
 c=dict(status='ROOT_CLOSED_INCREMENT40_INPUTS',parent_commit='3601c9dfca63dc1c6203aceb2c7fc066faa630fa',counts=dict(native=170,three_view=170,FL_new=16,Hybrid=22,baseline_valid=900),closure_pins={k:dict(path=p.as_posix(),sha256=sha(R/p)) for k,p in paths.items()},extra_pins={n:sha(R/n) for n in extra})
 save('ACTUAL_CLOSED_INPUTS.json',c);print(json.dumps(dict(ready_files=len(rows),closed_sha256=sha(B/'ACTUAL_CLOSED_INPUTS.json'))))
if __name__=='__main__':main()
