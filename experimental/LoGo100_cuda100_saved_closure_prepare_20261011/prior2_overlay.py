"""Join original adopted2 F evidence with delta98 without copying or loading arrays."""
from pathlib import Path,PurePosixPath
import json,hashlib
H=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def merge_maps(new_paths,old_paths,new_members,old_members):
 need(not(set(new_paths)&set(old_paths)) and not(set(new_members)&set(old_members)),'Duplicate server/member identity')
 for paths,members in [(new_paths,new_members),(old_paths,old_members)]:
  need(len(set(paths.values()))==len(paths) and set(paths.values())==set(members),'Incomplete member map')
  for p,n in paths.items():
   rel=PurePosixPath(n);need(p.startswith('/workspace/') and not rel.is_absolute() and '..' not in rel.parts and ':' not in n and '\\' not in n,'Unsafe metadata path')
 return dict(new_paths,**old_paths),dict(new_members,**old_members)
def join_prior(x,new_root,index,new_transport):
 prior=read(H/'PRIOR2_PINS.json');ids=prior['ids']
 for k in ['root','transport','audit']:
  p=Path(prior[k]['path']);need(p.drive.upper()=='E:' and sha(p)==prior[k]['sha256'],'Changed adopted2 compact proof')
 root=read(prior['root']['path']);t=read(prior['transport']['path']);a=read(prior['audit']['path'])
 need(root['status']=='ROOT_LOGO_FIXED2_CUDA_CACHE_FROZEN_POST_AND_OFFSERVER_SAVED_AUDIT_ADOPTED' and root['root_accepted'] is True and root['accepted']==2 and root['ids']==ids,'Missing original CUDA adopted2')
 need(root['transport']['sha256']==prior['transport']['sha256'] and root['offserver_audit']['sha256']==prior['audit']['sha256'] and root['archive']['sha256']==t['archive_sha256']==a['archive_sha256'],'Broken adopted2 lineage')
 need(a['ids']==ids and a['metric_values_checked']==6 and a['integer_confusion_counts_recomputed']==16 and a['prediction_items_checked']==39734,'Wrong original saved audit')
 need(x['prior2_root_sha256']==prior['root']['sha256'] and x['root_queue_auth']==root['root_queue_auth'] and x['deployment_seal']==root['deployment_seal'],'Mixed original producer')
 need(x['FedAvg_CUDA_root']==root['FedAvg_CUDA_root'] and x['FedAvg_CUDA_index']==root['FedAvg_CUDA_index'],'Different adopted CUDA cache')
 old=Path(t['verified_extract']).resolve();need(old.drive.upper()=='F:' and old.is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()) and t['all_members_verified'] is True,'Wrong old F evidence')
 need(sha(old/'ROOT_MEMBERS.json')==t['member_index_sha256'],'Old member index changed')
 oi=read(old/'ROOT_MEMBERS.json');need(oi['ids']==ids and oi['files']==t['members']==prior['members'],'Different original2 payload')
 need(index['reused_members']==prior['members']==new_transport['remote']['reused_member_pins'] and index['reused_server_paths']==oi['server_paths']==new_transport['remote']['reused_server_paths'],'Live retained2 differs from original F')
 need(new_transport['remote']['new_arrays']==98 and new_transport['remote']['reused_arrays']==2,'Wrong delta cardinality')
 need({n for n in oi['files'] if n.endswith('.npz')}=={'runs/'+i+'/scores_predictions.npz' for i in ids},'Wrong old arrays')
 need({n for n in index['files'] if n.endswith('.npz')}=={'runs/'+i+'/scores_predictions.npz' for i in index['ids'] if i not in ids},'Wrong new arrays')
 for n,v in oi['files'].items():
  p=old/n;need(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(old) and sha(p)==v['sha256'] and p.stat().st_size==v['bytes'],'Changed reused F member')
 for i in ids:
  need(x['task_accepted'][i]==root['records'][i]['task_acceptance'] and x['task_accepted'][i]['sha256']==oi['files']['tasks/'+i+'/ACCEPTED.json']['sha256'],'Old accepted task differs')
  need(root['records'][i]['scores_sha256']==oi['files']['runs/'+i+'/scores_predictions.npz']['sha256'],'Old CUDA scores differ')
 paths,members=merge_maps(index['server_paths'],oi['server_paths'],index['files'],oi['files'])
 roots={n:Path(new_root) for n in index['files']};roots.update({n:old for n in oi['files']})
 return paths,roots,members
