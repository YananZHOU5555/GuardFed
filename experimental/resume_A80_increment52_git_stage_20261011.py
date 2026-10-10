"""Resume only the explicit increment52 staging after sparse/ignore refusal."""
from pathlib import Path
from datetime import datetime,timezone
import ast,json,hashlib,subprocess
R=Path(__file__).resolve().parents[1]
original=R/'tmp/publish_A80_increment52_20261011.py'
tree=ast.parse(original.read_bytes());prefix=[]
for statement in tree.body:
    if isinstance(statement,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='volume' for x in statement.targets):break
    prefix.append(statement)
exec(compile(ast.Module(body=prefix,type_ignores=[]),str(original),'exec'))
assert git('rev-parse','HEAD').decode().strip()==PARENT
assert git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]==PARENT
receipt=T/'publication_closed_increment52_20261011.json';record=json.loads(receipt.read_bytes());pins=dict(record['files'])
pins[receipt.relative_to(R).as_posix()]={'source':receipt.relative_to(R).as_posix(),'sha256':H(receipt.read_bytes()),'bytes':receipt.stat().st_size}
for rel in ['tmp/resume_A80_increment52_git_stage_20261011.py','tmp/A80_current_generators_review_20261011/PRESERVED_GIT_STAGE_FAILURE.json']:
    p=R/rel;dest='experimental/'+rel[4:];b=p.read_bytes();target=W/dest;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
    pins[dest]={'source':rel,'sha256':H(b),'bytes':len(b)}
for dest,pin in pins.items():assert H((W/dest).read_bytes())==pin['sha256'] and H((R/pin['source']).read_bytes())==pin['sha256'],dest
attr=W/'.gitattributes';lines=attr.read_text(encoding='utf8').splitlines()
for dest in pins:
    line=json.dumps(dest,ensure_ascii=False)+' -text'
    if line not in lines:lines.append(line)
attr.write_bytes(('\n'.join(lines)+'\n').encode('utf8'))
pins['.gitattributes']={'sha256':H(attr.read_bytes()),'bytes':attr.stat().st_size}
paths=list(pins)
for i in range(0,len(paths),35):git('add','--sparse','-f','--',*paths[i:i+35])
changed_paths=[x for x in git('diff','--cached','--name-only','-z').decode('utf8').split('\0') if x]
assert changed_paths and set(changed_paths)<=set(pins)
git('commit','-m','Update accepted A80 rebuttal and native evidence cutoffs')
commit=git('rev-parse','HEAD').decode().strip();assert git('rev-parse','HEAD^').decode().strip()==PARENT
raw=git('cat-file','--batch',data=''.join('HEAD:'+p+'\n' for p in changed_paths).encode('utf8'));pos=0
for path in changed_paths:
    end=raw.index(b'\n',pos);header=raw[pos:end].split();assert header[1]==b'blob';size=int(header[2]);pos=end+1
    content=raw[pos:pos+size];pos+=size;assert raw[pos:pos+1]==b'\n';pos+=1
    assert H(content)==pins[path]['sha256'] and size==pins[path]['bytes'],path
assert pos==len(raw) and not git('status','--porcelain')
prepush=dict(status='COMMITTED_BLOB_SHA_PASS_PUSH_PENDING',utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,files={p:pins[p] for p in changed_paths},blob_count=len(changed_paths),sparse_ignore_recovery_preserved=True)
(T/'publication_closed_increment52_committed_20261011.json').write_text(json.dumps(prepush,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
git('push','origin','HEAD:refs/heads/'+BRANCH)
remote=git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0];assert remote==commit
verified=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,remote_head=remote,branch=BRANCH,committed_blobs_checked=len(changed_paths),manifest_sha256=H(receipt.read_bytes()),scope=record['scope'],large_artifacts_not_in_git=True,sparse_ignore_recovery_preserved=True)
(T/'publication_closed_increment52_verified_20261011.json').write_text(json.dumps(verified,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(verified,ensure_ascii=False))
