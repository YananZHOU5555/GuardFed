"""Bounded E-only source preparation; no Git, SSH, F access or acceptance."""
from pathlib import Path
import json, hashlib
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
T='docs/server_deployment_20260923/training_20260923/'
H=lambda b:hashlib.sha256(b).hexdigest()
def J(p):return json.loads((R/p).read_bytes())
def save(n,o):(D/n).write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
receipt=T+'publication_closed_increment65_verified_20261011.json'
assert H((R/receipt).read_bytes())=='844ffa5ee7a42a1de54d8465edec0150dd9c9dc42677e67657aebba5e2aa671d'
prev=J(receipt);parent='8515952b668c3a22427a8d0db5cc0a2d4c7cf6e3'
assert prev['commit']==parent and prev['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS'
allowed={'.py','.json','.md','.patch','.txt','.stderr','.stdout','.sha256','.csv','.tex','.diff'}
files={};refs={};groups={};omissions=[]
def add(rel,group):
 p=R/rel
 assert ':' not in rel and '..' not in Path(rel).parts and p.is_file() and not p.is_symlink()
 assert p.resolve().is_relative_to(R.resolve()) and p.suffix in allowed and p.stat().st_size<2_000_000,rel
 pin={'sha256':H(p.read_bytes()),'bytes':p.stat().st_size}
 dest='experimental/'+rel[4:] if rel.startswith('tmp/') else rel
 old=prev['files'].get(dest) or prev.get('parent_references',{}).get(rel)
 if old and (old['sha256'],old['bytes'])==(pin['sha256'],pin['bytes']):refs[rel]=pin
 else:files[rel]=pin
 groups.setdefault(group,[]).append(rel)
def directory(rel,group,exclude=()):
 # One level only; every chosen directory is named below. No recursive tree scan.
 for p in sorted((R/rel).iterdir()):
  if not p.is_file():continue
  r=p.relative_to(R).as_posix()
  if p.name in exclude or p.suffix not in allowed or p.stat().st_size>=2_000_000:
   omissions.append({'path':r,'reason':'Not selected: duplicate outputs, original bulk/unsupported suffix or explicit old evidence; retained at original source.'});continue
  add(r,group)
native=T+'server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261011T014706Z'
directory(native,'native370 accepted delta and archive references')
for name in ['inspection.json','inspection.sha256']:
 add(native+'/inspection/'+name,'native370 original inspection includes one explicitly unaccepted pending item')
directory('tmp/mechanism_F_nonIID_FFlip10_prepare_20261011','native370 collector source and closed metadata')
directory('tmp/mechanism_F_nonIID_FFlip10_prepare_20261011/root_delta_20261011T014622Z','preserved first preparation evidence')
directory('tmp/mechanism_F_nonIID_FFlip10_saved_binding_prepare_20261011','views370 original transport and saved consumer source',exclude=['MECHANISM_INDEX.json'])
directory('tmp/mechanism_F_nonIID_FFlip10_saved_binding_prepare_20261011/preserved_EOL_guard_failure','preserved unexecuted preparation failure')
directory('tmp/celeba_mechanism_remaining_F_nonIID_FFlip10_root_adoption_20261011','views370 adopted index and root')
directory('tmp/mechanism_F_seven_scenes70_table_prepare_20261011','F70 source and actual verification commands',exclude=['records.json','tables.json','TABLES.md','coverage.json','paired_per_seed.json','checks.json','SOURCE_BINDINGS.json','ROOT_BINDING.actual_candidate.json'])
directory(T+'celeba_mechanism_v1/three_view_F_seven_scenes70_20261011','F70 canonical accepted table')
directory('tmp/fl_native_after77_SDFA9_prepared_20261011','FL86 exact9 native evidence and original source',exclude=['PREVIOUS_OFFSERVER_ACCEPTANCE.json','PREVIOUS_LATEST.json','SERVER_GUIDE.md','GUIDE_STDOUT.txt'])
directory('tmp/hybrid_after12_FFlip7_prepare_20261011','Hybrid19 native exact7 original source and accepted evidence',exclude=['GUIDE.md','PRIOR_OFFSERVER.json','PRIOR_ROOT_ADOPTION.json','FROZEN_OBSERVATION.json'])
directory('tmp/celeba_hybrid_native19_FFlip10_root_adoption_20261011','Hybrid19 native adoption')
directory('tmp/hybrid_FFlip10_three_view_prepare_20261011','Hybrid FFlip10 saved replay source and preserved failures')
directory('tmp/hybrid_FFlip10_three_view_prepare_20261011/saved','Hybrid original Linux/Windows transport source')
directory('tmp/hybrid_FFlip10_three_view_prepare_20261011/runtime','Hybrid runtime v1 preserved unexecuted schema failure')
directory('tmp/hybrid_FFlip10_three_view_prepare_20261011/runtime_v2','Hybrid actual runtime v2 source and references',exclude=['FRESH_OBSERVATION_COMMAND.json','FRESH_OBSERVATION_remote.py','START_RECEIPT_remote.py'])
directory('tmp/hybrid_FFlip10_root_adoption_20261011','Hybrid10 added saved evidence root')
for name in ['LINUX_SAVED_CHECK.json','TRANSPORT_VERIFICATION.json']:
 add('tmp/hybrid_FFlip10_saved_root_execution_20261011/saved001/'+name,'Hybrid actual compact Linux and transport proofs')
directory('outputs/guardfed_tables/celeba_hybrid_three_view_two_scenes20_20261011','Hybrid20 canonical table and checker')
directory('tmp/rebuttal_F70_Hybrid20_editorial_prepare_20261011','F70 Hybrid20 author checks (not root adoption)',exclude=['ROOT_EDITORIAL_ADOPTION.json'])
draft='docs/server_deployment_20260923/revision_20260923/rebuttal_working_F70_FL81_Hybrid20_20261011.md'
assert H((R/draft).read_bytes())=='14fcaa9d60c4ab9ca17b9964a5d1dc4f178ad5537f7ff3c1c23b06586f2dfb19'
add(draft,'preserved agent candidate provenance')
draft='docs/server_deployment_20260923/revision_20260923/rebuttal_working_F70_FL81_Hybrid20_release_20261011.md'
assert H((R/draft).read_bytes())=='9be134d9d8fbaf592274c6c4229783f3fd0dbde058e8b8714a780e0f116c5363'
add(draft,'actual root-adopted release wording; scientific cutoff unchanged')
add(receipt,'actual parent publication receipt')
directory('tmp/root_F70_Hybrid20_editorial_current_20261011','actual metadata update source history',exclude=['ROOT_METADATA_ADOPTION.json'])
def role(rel):return {'path':rel,'sha256':H((R/rel).read_bytes())}
roles={
 'native370_root':role(native+'/ROOT_DELTA_VERIFICATION.json'),
 'current_three_view_root':role('tmp/celeba_mechanism_remaining_F_nonIID_FFlip10_root_adoption_20261011/ROOT_ADOPTION.json'),
 'F70_table_root':role(T+'celeba_mechanism_v1/three_view_F_seven_scenes70_20261011/ROOT_VERIFICATION.json'),
 'FL_native86_root':role('tmp/fl_native_after77_SDFA9_prepared_20261011/ROOT_ADOPTION_REVIEW.json'),
 'Hybrid_native19_root':role('tmp/celeba_hybrid_native19_FFlip10_root_adoption_20261011/ROOT_ADOPTION.json'),
 'Hybrid_three_view20_root':role('tmp/hybrid_FFlip10_root_adoption_20261011/ROOT_SCIENTIFIC_ADOPTION.json'),
 'Hybrid_table20_root':role('outputs/guardfed_tables/celeba_hybrid_three_view_two_scenes20_20261011/ROOT_VERIFICATION.json'),
 'current_working_editorial_root':role('tmp/rebuttal_F70_Hybrid20_editorial_prepare_20261011/ROOT_EDITORIAL_ADOPTION.json'),
 'editorial_adopter':role('tmp/adopt_F70_Hybrid20_rebuttal_root_20261011.py'),
}
assert roles['current_working_editorial_root']['sha256']=='9246cd580b5cbe7a10203a646b8df878aecba30b317f0ab8d15bb40c137981a9'
for k in ['FL_native77_root','FL_three_view81_root','FL_eight_scene80_table_root','gradient64_root','gradient64_selected_root','gradient200_frozen_source_root']:
 roles[k]=prev['proof_roles'][k]
roles['previous_working_editorial_root']=prev['proof_roles']['current_working_editorial_root']
for k,p in roles.items():
 assert H((R/p['path']).read_bytes())==p['sha256']
 add(p['path'],'accepted fact roles and unchanged parent references')
add('tmp/publication_increment65_prepared_20261011/publish_guardfed_increment65_20261011.py','original publisher65 parent reference')
entry_paths=[x['path'] for x in J('tmp/root_F60_current_entries_20261011/ROOT_METADATA_ADOPTION.json')['entries']]
late_roles=['entry_writer','entry_update_proof','latest_observation_raw']
for k in late_roles:roles[k]={'path':None,'sha256':None}
assert not set(files)&set(refs)
scope={'native_new':370,'three_view_models':370,'F_paired_models':70,'F_complete_scenes':7,'FL_native_new':86,'FL_screen_reuse_separate':4,'FL_three_view_total':81,'FL_complete_table_records':80,'FL_complete_table_scenes':8,'gradient_accepted':64,'Hybrid_native_new':19,'Hybrid_three_view_total':20,'Hybrid_complete_table_records':20,'Hybrid_complete_table_scenes':2,'gradient200_new_scientific_accepted':0,'by_variant':{'minus_U':100,'minus_C':100,'minus_A':100,'minus_F':70}}
template={'status':'INCREMENT66_SOURCE_ONLY_ROOT_BINDING_REQUIRED','root_accepted':False,'parent_commit':parent,'STATE_sha256':None,'files':files,'parent_references':refs,'proof_roles':roles,'late_role_names':late_roles,'late_files':{},'entry_paths':entry_paths,'STATE_path':T+'TRAINING_STATE.json','groups':groups,'omissions':omissions,'current_accepted':scope,'draft_path':draft,'draft_sha256':H((R/draft).read_bytes()),'gradient14_evidence_scope':'Excluded from this publication; no whole14 root acceptance or scientific result is claimed.','boundary':'Native inspection may contain pending FedSA1, which is not accepted370. No models/arrays/archive/F observations. No new acceptance by publisher. Old raw and failed evidence retained externally. Future metadata/author adoption must bind actual paths and hashes.'}
save('INPUTS_TEMPLATE.json',template)
save('SELECTION_TEMPLATE.json',{'status':'CANDIDATE_NOT_AUTHORIZED','root_accepted':False,'parent_commit':parent,'STATE_sha256':None,'stable_input_sha256':None,'semantic_review_sha256':None,'files':None,'parent_references':refs,'proof_roles':roles,'root_late_bound_files':None})
save('SEMANTIC_TEMPLATE.json',{'status':'PENDING_ROOT_CURRENT_ENTRY_SEMANTIC_REVIEW','root_accepted':False,'STATE_sha256':None,'current_accepted':scope,'working_draft_content_cutoff':None,'historical_Git65_cutoff':prev['acceptance_cutoff'],'latest_observation_only':None,'entries':{p:{'sha256':None,'history_exact':None} for p in entry_paths},'source_pins':roles,'gradient14_excluded':True,'final_test':False,'whole_rebuttal_complete':False})
state_path=T+'TRAINING_STATE.json';state_sha=H((R/state_path).read_bytes())
assert state_sha=='1e9119a17226a302bf7fe33293a63dd746744d22c0480a7677b2f3533b18faa8'
state=J(state_path);metadata_path='tmp/root_F70_Hybrid20_editorial_current_20261011/ROOT_METADATA_ADOPTION.json'
assert H((R/metadata_path).read_bytes())=='2b44854aefac74d609c856e2b78751d6fd5dfbcefa53efe1bb1f715012df29b0'
meta=J(metadata_path);assert meta['STATE_after']['sha256']==state_sha and {x['path'] for x in meta['entries']}==set(entry_paths)
bound=json.loads(json.dumps(template));bound['status']='ACTUAL_INPUTS_CANDIDATE_ROOT_REVIEW_PENDING';bound['STATE_sha256']=state_sha
bound['proof_roles']['entry_writer']=meta['writer']
bound['proof_roles']['entry_update_proof']=role(metadata_path)
obs=state['latest_five_queue_readonly_observation'];bound['proof_roles']['latest_observation_raw']={k:obs[k] for k in ['path','sha256']}
for rel in [state_path,*entry_paths,*[bound['proof_roles'][k]['path'] for k in late_roles]]:
 p=R/rel;assert p.is_file() and not p.is_symlink() and p.stat().st_size<2_000_000
 pin={'sha256':H(p.read_bytes()),'bytes':p.stat().st_size};bound['late_files'][rel]=pin;bound['files'][rel]=pin
for v in meta['entries']:assert bound['files'][v['path']]['sha256']==v['sha256']
for k in late_roles:assert H((R/bound['proof_roles'][k]['path']).read_bytes())==bound['proof_roles'][k]['sha256']
save('BOUND_INPUTS_CANDIDATE.json',bound)
semantic=J(str((D/'SEMANTIC_TEMPLATE.json').relative_to(R)))
semantic.update(STATE_sha256=state_sha,working_draft_content_cutoff=state['latest_working_rebuttal_draft']['content_cutoff'],latest_observation_only=obs,source_pins=bound['proof_roles'])
semantic['entries']={v['path']:{'sha256':v['sha256'],'history_exact':None} for v in meta['entries']}
save('SEMANTIC_CANDIDATE.json',semantic)
print(json.dumps({'new_compact_files':len(files),'new_bytes':sum(v['bytes'] for v in files.values()),'parent_references':len(refs),'template_sha256':H((D/'INPUTS_TEMPLATE.json').read_bytes()),'late_roles':late_roles}))
