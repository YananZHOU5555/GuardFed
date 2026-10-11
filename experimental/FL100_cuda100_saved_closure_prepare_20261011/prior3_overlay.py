"""Exact fixed3 F references plus new97 archive; no copying, symlinks, fitting or model loading."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def need(ok,msg):
    if not ok:raise ValueError(msg)
def old_member(rel,ids):
    return rel=='source/metadata.npz' or any(rel==prefix or rel.startswith(prefix+'/') for i in ids for prefix in ('base/tasks/'+i,'base/runs/'+i))
class Fixed3Paths:
    def __init__(self,new,old,ids):self.new=Path(new);self.old=Path(old);self.ids=ids
    def __truediv__(self,rel):
        text=str(rel).replace('\\','/');need(not text.startswith('/') and ':' not in text and '..' not in text.split('/'),'Unsafe fixed3 reference')
        return (self.old if old_member(text,self.ids) else self.new)/text
def join_prior(inp,new_root,new_members):
    frozen=read(HERE/'SELECTED100_INPUTS.json');prior=read(HERE/'PRIOR3_PINS.json');ids=frozen['canary_ids']
    for name in ['root','transport','audit']:
        pin=prior[name];p=Path(pin['path']);need(p.drive.upper()=='E:' and sha(p)==pin['sha256'],'Changed compact prior3 '+name)
    root=read(prior['root']['path']);t=read(prior['transport']['path']);a=read(prior['audit']['path'])
    need(root['status']=='ROOT_FIXED3_FL100_CUDA_ORIGINAL_LINUX_WHOLE_AND_OFFSERVER_SAVED_AUDIT_ADOPTED' and root['ids']==ids and root['accepted']==3,'Wrong adopted3')
    need(root['transport']['sha256']==prior['transport']['sha256'] and root['audit']['sha256']==prior['audit']['sha256'] and root['archive']['sha256']==t['archive_sha256']==a['archive_sha256'],'Broken adopted3 lineage')
    need(a['metric_values_checked']==27 and a['integer_base_counts_checked']==72 and a['prediction_rules_checked']==9 and a['exact_ids']==ids,'Incomplete original3 audit')
    old=Path(t['verified_extract']).resolve();need(old.drive.upper()=='F:' and old.is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()) and t['all_members_verified'] is True,'Original F transport absent')
    need(sha(old/'ROOT_MEMBERS.json')==t['member_index_sha256'] and read(old/'ROOT_MEMBERS.json')['files']==t['members'],'Changed original archive member index')
    keep={n:v for n,v in t['members'].items() if old_member(n,ids)}
    new_transport_pin=inp['transport'];need(sha(new_transport_pin['path'])==new_transport_pin['sha256'],'Changed new97 transport')
    new_transport=read(new_transport_pin['path']);need(new_transport['remote']['reused_member_pins']==keep and new_transport['remote']['new_arrays']==97 and new_transport['remote']['reused_arrays']==3,'Live original3 and retained F identities differ')
    need(not(set(keep)&set(new_members)),'New transport duplicates prior3 bulk or identities')
    need({n for n in keep if n.endswith('.npz')}=={'source/metadata.npz'}|{'base/runs/'+i+'/validation_predictions.npz' for i in ids},'Incomplete original3 arrays')
    for n,v in keep.items():
        p=old/n;need(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(old) and sha(p)==v['sha256'] and p.stat().st_size==v['bytes'],'Changed reused F member')
    for i in ids:
        need(inp['accepted_sha256'][i]==root['records'][i]['task_acceptance_sha256']==keep['base/tasks/'+i+'/ACCEPTED.json']['sha256'],'Reused task differs from root3')
        need(keep['base/runs/'+i+'/validation_predictions.npz']['sha256']==root['records'][i]['arrays_sha256'],'Reused cache differs from root3')
    combined=dict(new_members,**keep);need(len(combined)==len(new_members)+len(keep),'Duplicate member')
    return Fixed3Paths(new_root,old,ids),combined
