"""Publish a bounded compact evidence increment, preserving source bytes."""
from pathlib import Path
from datetime import datetime, timezone
import subprocess, hashlib, json, shutil, sys
R=Path(__file__).resolve().parents[1]
W=Path('F:/YananResearchStorage/GuardFed/git_publication/current')
T=R/'docs/server_deployment_20260923/training_20260923'
PARENT='4c92f3d93c1a5bba01b1323c5303b80b4a0fb19e'
BRANCH='codex/revision-evidence-baselines-20260928'
H=lambda b:hashlib.sha256(b).hexdigest()
G=['git','-c','safe.directory='+W.as_posix(),'-c','core.longpaths=true','-C',str(W)]
def git(*args,data=None):
    p=subprocess.run(G+list(args),input=data,capture_output=True)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
    return p.stdout
volume=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',"Get-Volume -DriveLetter F | Select-Object FileSystemLabel,HealthStatus,SizeRemaining | ConvertTo-Json"]))
assert volume['FileSystemLabel']=='Yanan 2TB' and volume['HealthStatus']=='Healthy' and volume['SizeRemaining']>2_000_000_000
assert git('rev-parse','HEAD').decode().strip()==PARENT and not git('status','--porcelain')
assert git('branch','--show-current').decode().strip()==BRANCH
assert git('remote','get-url','origin').decode().strip()=='https://github.com/YananZHOU5555/GuardFed.git'
assert git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]==PARENT
dirs=[
 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_eight_scenes80_20261011',
 'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A80_reader_20261011',
 'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T150639Z',
 'tmp/rebuttal_A80_candidate_20261011','tmp/celeba_mechanism_A80_candidate_20261011',
 'tmp/celeba_remaining620_after260_transport_preparation_20261011',
 'tmp/celeba_mechanism_remaining620_after260_root_adoption_20261011',
 'tmp/fl_native44_exact10_20261011','tmp/celeba_native_after272_20261011',
 'tmp/A80_current_generators_review_20261011','tmp/celeba_flgmm_closed47_root_execution_20261011']
sources=[]
for directory in dirs:
    for p in sorted((R/directory).rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:continue
        if 'celeba_flgmm_closed47_root_execution_20261011' in p.parts and (p.name.startswith('LIVE_OBSERVATION') or p.name.startswith('OBSERVATION')):continue
        assert not p.is_symlink() and p.resolve().is_relative_to(R.resolve())
        assert p.suffix.lower() not in {'.gz','.npz','.npy','.pt','.pth','.zip','.pyc','.png','.pdf'}
        assert p.stat().st_size<3_000_000
        sources.append(p)
sources += [T/n for n in ['RUNNING.md','TRAINING_STATE.json','REBUTTAL_COMPLETION_20261009.md']]
sources += [R/'docs/返修实验总览.md',T/'server_reactivation_20261009/MONITOR_HANDOFF.md',
 T/'server_reactivation_20261009/latest_formal_live.json',T/'server_reactivation_20261009/root_live_20261010T152446Z.json',
 R/'tmp/celeba_flgmm_fullcoverage_incremental_20261009/LATEST_BACKUP.json']
sources += [R/'tmp'/n for n in ['adopt_rebuttal_A80_reader_root_20261011.py','adopt_A80_root_20261011.py',
 'adopt_after260_replays_root_20261011.py','update_reactivation_state_20261009.py',
 'update_completion_current_20261009.py','update_overview_closure100_root_20261009.py','publish_A80_increment52_20261011.py']]
sources=list(dict.fromkeys(sources))
pins={}
for p in sources:
    rel=p.relative_to(R).as_posix();dest='experimental/'+rel[4:] if rel.startswith('tmp/') else rel
    b=p.read_bytes();pins[dest]={'source':rel,'sha256':H(b),'bytes':len(b)}
assert sum(v['bytes'] for v in pins.values())<20_000_000
receipt=T/'publication_closed_increment52_20261011.json';assert not receipt.exists()
record=dict(status='BOUNDED_A80_EVIDENCE_PUBLICATION_READY',utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,branch=BRANCH,
    scope=dict(mechanism_native_new=280,mechanism_three_view=280,FLGMM_native_new=54,FLGMM_reused=4,FLGMM_three_view_total=48,A_pairs=80,A_scenes=8,reviewer_comments=24,final_test=False,whole_rebuttal_complete=False),
    files=pins,large_artifacts_not_in_git=True,raw_scientific_evidence_retained_on_server_and_F=True)
receipt.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
rel=receipt.relative_to(R).as_posix();pins[rel]={'source':rel,'sha256':H(receipt.read_bytes()),'bytes':receipt.stat().st_size}
changed={}
for dest,pin in pins.items():
    b=(R/pin['source']).read_bytes();assert H(b)==pin['sha256']
    p=W/dest
    if p.exists() and H(p.read_bytes())==pin['sha256']:continue
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);changed[dest]=pin
attr=W/'.gitattributes';b=attr.read_bytes()
lines=b.decode('utf8').splitlines()
for dest in changed:
    line=json.dumps(dest,ensure_ascii=False)+' -text'
    if line not in lines:lines.append(line)
attr.write_bytes(('\n'.join(lines)+'\n').encode('utf8'))
changed['.gitattributes']={'sha256':H(attr.read_bytes()),'bytes':attr.stat().st_size}
paths=list(changed)
for i in range(0,len(paths),35):git('add','--',*paths[i:i+35])
assert set(git('diff','--cached','--name-only','-z').decode('utf8').split('\0'))-{''}==set(paths)
git('commit','-m','Update accepted A80 rebuttal and native evidence cutoffs')
commit=git('rev-parse','HEAD').decode().strip();assert git('rev-parse','HEAD^').decode().strip()==PARENT
raw=git('cat-file','--batch',data=''.join('HEAD:'+p+'\n' for p in paths).encode('utf8'))
pos=0
for path in paths:
    end=raw.index(b'\n',pos);header=raw[pos:end].split();assert header[1]==b'blob';size=int(header[2]);pos=end+1
    content=raw[pos:pos+size];pos+=size;assert raw[pos:pos+1]==b'\n';pos+=1
    assert H(content)==changed[path]['sha256'] and size==changed[path]['bytes'],path
assert pos==len(raw) and not git('status','--porcelain')
prepush=dict(status='COMMITTED_BLOB_SHA_PASS_PUSH_PENDING',utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,files=changed,blob_count=len(paths))
(T/'publication_closed_increment52_committed_20261011.json').write_text(json.dumps(prepush,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
git('push','origin','HEAD:refs/heads/'+BRANCH)
remote=git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0];assert remote==commit
verified=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,remote_head=remote,branch=BRANCH,committed_blobs_checked=len(paths),manifest_sha256=H(receipt.read_bytes()),scope=record['scope'],large_artifacts_not_in_git=True)
(T/'publication_closed_increment52_verified_20261011.json').write_text(json.dumps(verified,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(verified,ensure_ascii=False))
