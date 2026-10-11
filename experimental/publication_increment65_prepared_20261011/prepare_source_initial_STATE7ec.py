"""Enumerate a bounded compact publication candidate, without Git/network/F writes."""
from pathlib import Path
import ast,difflib,hashlib,json
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
T='docs/server_deployment_20260923/training_20260923/'
P=R/'tmp/publication_increment64_prepared_20261011/publish_guardfed_increment64_20261011.py'
H=lambda b:hashlib.sha256(b).hexdigest()
J=lambda p:json.loads((R/p).read_bytes())
parent='3edbd954d91bbc5e4993546950202645d82ccbb0'
receipt_path=T+'publication_closed_increment64_verified_20261011.json'
assert H((R/receipt_path).read_bytes())=='ceb120bf2ebc7fe8d635c2fd554f1c69787f70dc0e112398600ca265cdc9add2'
previous=J(receipt_path);assert previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and previous['commit']==parent
meta_path='tmp/root_F60_current_entries_20261011/ROOT_METADATA_ADOPTION.json';meta=J(meta_path)
assert H((R/meta_path).read_bytes())=='20b5f9493d18d764168a5ca8920003f66469afd4f0968954d940f299371e3afe'
state=J(T+'TRAINING_STATE.json');assert H((R/(T+'TRAINING_STATE.json')).read_bytes())=='7ec6894aedf96dd9837db605c78e088f1f8c18020fdc8daad0c1e26c22b4644d'
roles={k:meta[v] for k,v in [('native360_root','native'),('current_three_view_root','views'),('F60_table_root','table'),('current_working_editorial_root','editorial')]}
priorroles=previous['proof_roles']
for k in ['FL_native77_root','FL_three_view81_root','FL_eight_scene80_table_root','gradient64_root','gradient64_selected_root','gradient200_frozen_source_root']:
    roles[k]=priorroles[k]
roles['working_F50_editorial_root']=priorroles['current_working_editorial_root']
fl_ed='tmp/rebuttal_FL81_F50_update_20261011/ROOT_EDITORIAL_ADOPTION.json'
roles['working_FL81_F50_editorial_root']={'path':fl_ed,'sha256':H((R/fl_ed).read_bytes())}
roles['entry_writer']=meta['source'];roles['entry_update_proof']={'path':meta_path,'sha256':H((R/meta_path).read_bytes())}
roles['latest_observation_raw']=meta['observation']
gate=state['gradient200_fullcoverage_source_preparation_20261010']['exact14_partial_progress']
assert gate['sha256']=='25b3eb5644606caba3ffc47e0da543a406ec31b2996aa19a21e749b45c14d6f6'
roles['gradient14_current_metadata']=gate
g=J(gate['path']);assert g['completed']==7 and g['total']==14 and len(g['items'])==7

allowed={'.py','.json','.md','.patch','.txt','.stderr','.stdout','.sha256','.csv','.tex','.diff'}
files={};refs={};omissions=[];groups={}
def dest(rel):return 'experimental/'+rel[4:] if rel.startswith('tmp/') else rel
def add(rel,group,parent_ok=False):
    p=R/rel
    assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(R.resolve()) and p.suffix in allowed,rel
    b=p.read_bytes();assert len(b)<2_000_000,rel
    pin={'sha256':H(b),'bytes':len(b)}
    old=previous['files'].get(dest(rel))
    if parent_ok and old and (old['sha256'],old['bytes'])==(pin['sha256'],pin['bytes']):refs[rel]=pin
    else:files[rel]=pin
    groups.setdefault(group,[]).append(rel)
def directory(rel,group,exclude=(),nested=()):
    base=R/rel
    for p in sorted(base.iterdir()):
        if not p.is_file():continue
        sub=p.relative_to(R).as_posix()
        if p.name in exclude or p.suffix not in allowed:
            omissions.append({'path':sub,'reason':'Duplicate accepted output or non-selected log; source location remains preserved','sha256':H(p.read_bytes()) if p.stat().st_size<2_000_000 else None});continue
        add(sub,group)
    for child in nested:
        for p in sorted((base/child).rglob('*')):
            if p.is_file():add(p.relative_to(R).as_posix(),group)

native=meta['native']['path'].rsplit('/',1)[0]
directory(native,'native360 root and compact acceptance')
directory(native+'/inspection','native360 accepted inspection and saved statistics')
nativeops='tmp/celeba_native_after350_nonIID_Benign10_prepared_20261011'
directory(nativeops,'native360 original collector source and actual execution',nested=['ACTUAL_COLLECTOR_001'])
directory('tmp/celeba_mechanism_remaining_F_nonIID_Benign10_prepare_20261011','views360 transport source and actual evidence',exclude=['MECHANISM_INDEX.json'],nested=['preserved_source_attempt001','preserved_source_attempt002'])
directory('tmp/celeba_F_nonIID_Benign10_root_adopter_prepare_20261011','views360 root adoption source')
directory('tmp/celeba_mechanism_remaining_F_nonIID_Benign10_root_adoption_20261011','views360 root and index')
fsource='tmp/celeba_F_six_scenes60_table_prepare_20261011'
fdup=['records.json','tables.json','TABLES.md','coverage.json','paired_per_seed.json','checks.json','SOURCE_BINDINGS.json','ROOT_BINDING.actual_candidate.json']
directory(fsource,'F60 source actual commands and preserved preparation failures',exclude=fdup,nested=['unexecuted_preparation_failure','unexecuted_preparation_v1'])
directory(T+'celeba_mechanism_v1/three_view_F_six_scenes60_20261011','canonical F60 table and original saved verification')
directory('tmp/rebuttal_FL81_F50_update_20261011','FL81 F50 editorial source and prior snapshot',nested=['before_release_pointer'])
directory('tmp/rebuttal_F60_FL81_gradient64_update_20261011','F60 editorial source',exclude=['rebuttal_working_F60_FL81_gradient64_20261011.md'])
directory('tmp/root_F60_current_entries_20261011','root metadata write and preserved old prefixes')
for p in [meta['source']['path'],'tmp/root_adopt_F60_editorial_20261011.py',T+'TRAINING_STATE.json',receipt_path,*[x['path'] for x in meta['entries']],meta['observation']['path']]:add(p,'current metadata and prior publication receipt')
for p in ['docs/server_deployment_20260923/revision_20260923/rebuttal_working_F50_FL81_gradient64_20261011.md','docs/server_deployment_20260923/revision_20260923/rebuttal_working_F60_FL81_gradient64_20261011.md']:add(p,'root adopted working manuscripts')
for item in g['items']:
    assert H((R/item['path']).read_bytes())==item['sha256'];add(item['path'],'exact7 partial CPU3 metadata only',parent_ok=True)
for k,pin in roles.items():
    assert H((R/pin['path']).read_bytes())==pin['sha256'],k
    add(pin['path'],'closed fact roles',parent_ok=k in priorroles or k=='working_F50_editorial_root')
# Reuse already published scientific inputs by parent blob; no bulk recopy.
for p in [J(fl_ed)['source_pins'][k]['path'] for k in ['FL_tables','FL_table_md','FL_records']]:add(p,'unchanged FL81 table parent references',parent_ok=True)
for p in [priorroles[k]['path'] for k in ['native350_root','current_three_view_root','F50_table_root']]:add(p,'exact prior F50/native/views root references',parent_ok=True)
parenteditor=J(priorroles['current_working_editorial_root']['path']);add(parenteditor['draft_path'],'original F50 parent draft',parent_ok=True)
assert not set(files)&set(refs)
assert sum(x['bytes'] for x in files.values())<100_000_000
stable={'status':'CLOSED_INCREMENT65_INPUTS_NOT_PUBLICATION_AUTHORIZATION','parent_commit':parent,'STATE_sha256':H((R/(T+'TRAINING_STATE.json')).read_bytes()),'files':files,'parent_references':refs,'proof_roles':roles,'groups':groups,'omissions':omissions,'scope':{'native':360,'views':360,'F_pairs':60,'F_scenes':6,'FL_native_new':77,'FL_screen_reuse_separate':4,'FL_three_views':81,'FL_complete_table_records':80,'gradient_search':64,'Hybrid_native_new':12,'Hybrid_three_views':10,'gradient200_new_scientific_accepted':0,'gradient14_partial_individual_items':7},'boundary':'Bounded selected compact evidence, not a complete raw backup. Only exact7 gate item metadata; no whole14 or192 completion. Unchanged old scientific inputs remain at verified Git64 parent; no new acceptances created.'}
(D/'STABLE_INPUTS.json').write_text(json.dumps(stable,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'new_files':len(files),'parent_references':len(refs),'new_bytes':sum(x['bytes'] for x in files.values()),'stable_sha256':H((D/'STABLE_INPUTS.json').read_bytes()),'gate_path':gate['path']}))
