"""Pending increment74 inventory/derivation only; no Git/F/SSH or final source seal."""
from pathlib import Path
import argparse,hashlib,json
R=Path(__file__).resolve().parents[2];Q=Path(__file__).resolve().parent
H=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(p.read_bytes())
plan=read(Q/'FREEZE_PLAN.json')
assert plan['root_completion_received'] is False and not (Q/'FILES_SHA256.json').exists()
p=R/plan['previous_receipt']['path'];assert H(p.read_bytes())==plan['previous_receipt']['sha256'];parent=read(p)
# Receipt.parent is Git72; receipt.commit is the actual Git73 parent for74.
assert parent['commit']==plan['parent_commit'] and parent['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS'
old=R/plan['parent_publisher']['path'];raw=old.read_bytes();assert H(raw)==plan['parent_publisher']['sha256']
assert len(plan['late_mutable_names'])==len(set(plan['late_mutable_names']))==8
assert H((R/plan['draft_path']).read_bytes())==plan['draft_sha256']=='910b263339273c9d973491f1005f35ec4a64ed6a27127616c096ee7f3ffce382'
paths=set(plan['explicit_files']);missing=[];seals={};omitted={}
for folder in plan['sealed_packages']:
 sp=R/folder/'FILES_SHA256.json';seals[folder]={'sha256':H(sp.read_bytes()),'bytes':sp.stat().st_size};paths.add(folder+'/FILES_SHA256.json')
 for name,pin in read(sp)['files'].items():
  assert not Path(name).is_absolute() and '..' not in Path(name).parts
  rel=folder+'/'+name;f=R/rel;assert H(f.read_bytes())==pin['sha256'];paths.add(rel)
for rel in plan['actual_metadata_indexes']:
 sp=R/rel
 if not sp.is_file():missing.append(rel);continue
 paths.add(rel);base=sp.parent.parent
 for name,pin in read(sp)['files'].items():
  assert not Path(name).is_absolute() and '..' not in Path(name).parts
  f=base/name;assert f.is_file() and H(f.read_bytes())==pin['sha256']
  paths.add(f.relative_to(R).as_posix())
for rel in plan['future_explicit_files']:
 if (R/rel).is_file():paths.add(rel)
 else:missing.append(rel)
files={};references={};allowed={'.py','.json','.md','.patch','.txt','.stderr','.stdout','.sha256','.csv','.tex','.diff'}
for rel in sorted(paths):
 f=R/rel;assert f.is_file() and not f.is_symlink() and f.resolve().is_relative_to(R.resolve()) ,rel
 pin={'sha256':H(f.read_bytes()),'bytes':f.stat().st_size}
 if f.suffix not in allowed:omitted[rel]=dict(pin,reason='Extensionless runtime log not in inherited whitelist; exact bytes remain referenced by actual metadata seal and command proof');continue
 if pin['bytes']>=2000000:omitted[rel]=dict(pin,reason='Above compact bound; source/root SHA provenance only');continue
 assert rel not in plan['late_mutable_names']
 dest='experimental/'+rel[4:] if rel.startswith('tmp/') else rel
 oldpin=parent['files'].get(dest) or parent.get('parent_references',{}).get(rel)
 if oldpin and (oldpin['sha256'],oldpin['bytes'])==(pin['sha256'],pin['bytes']):references[rel]=pin
 else:files[rel]=pin
assert sum(v['bytes'] for v in files.values())<100000000
# Final generation will preserve these exact bodies; only73->74 names/receipt/cutoff text change.
s=raw.decode();start=s.index("G=['git'");end=s.index("cutoff=dict(",start)
body=s[start:end]
report={'status':'PROVISIONAL_INCREMENT74_INVENTORY_NOT_FROZEN','parent_commit':plan['parent_commit'],'files':files,'parent_references':references,'source_seals':seals,'omissions':omitted,'currently_missing_explicit_paths':missing,'late_mutable_names':plan['late_mutable_names'],'mutable_pins':None,'future_proof_roles':plan['future_proof_roles'],'parent_Git_F_copy_blob_clean_remote_body_sha256':H(body.encode()),'parent_Git_F_copy_blob_clean_remote_body':body,'root_completion_received':False,'final_publisher_generated':False,'Git_F_SSH':False}
with (Q/'PROVISIONAL_SELECTION.json').open('w',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'files':len(files),'bytes':sum(v['bytes'] for v in files.values()),'references':len(references),'missing':len(missing),'seal_created':False,'mutable_pinned':False}))
