"""Local metadata-only derivation. No Git, external volume, network or scientific execution."""
from pathlib import Path
import ast, difflib, hashlib, json
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
T='docs/server_deployment_20260923/training_20260923/';V='docs/server_deployment_20260923/revision_20260923/'
H=lambda b:hashlib.sha256(b).hexdigest()
def read(p):return json.loads((R/p).read_bytes())
def write(n,x): (D/n).write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode())
def pin(p):
 q=R/p;assert q.is_file() and not q.is_symlink() and q.resolve().is_relative_to(R.resolve()),p
 assert q.stat().st_size<2_000_000,p
 b=q.read_bytes();return {'sha256':H(b),'bytes':len(b)}
priorrel=T+'publication_closed_increment69_verified_20261011.json';previous=read(priorrel)
assert pin(priorrel)['sha256']=='4667bab2574b294c66cb48c537e77c5586d86f54934304134ec4d862fa7b47e6'
PARENT='0a276220b447645020e8dca8ee092ce411367bb7';assert previous['commit']==PARENT
oldtemplate=read('tmp/publication_increment69_prepared_20261011/INPUTS_TEMPLATE.json')
state=read(T+'TRAINING_STATE.json');assert pin(T+'TRAINING_STATE.json')['sha256']=='8f803c7b0c4789a369dfce1997a9262909e2b5072e4f5e019cc3be652c80a2a4'
roles={
 'FL100_root':'tmp/fl_isolated_v3_saved_closure_prepare_20261011/root_closure/ROOT_SCIENTIFIC_ADOPTION.json',
 'FL100_table':'outputs/guardfed_tables/celeba_flgmm_ten_scenes100_20261011/ROOT_VERIFICATION.json',
 'NGA9_root':'tmp/NGA96_IID_Benign9_root_adopter_prepare_20261011/ROOT_ADOPTION.json',
 'Hybrid29_root':'tmp/celeba_hybrid_native29_IIDFedSA10_root_adoption_20261011/ROOT_ADOPTION.json',
 'CUDA7_root':'tmp/uniform_cuda7_saved_audit_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json',
 'CUDA7_audit':'tmp/uniform_cuda7_saved_audit_prepare_20261011/actual/SAVED_OUTPUTS_AUDIT.json',
 'CUDA7_transport':'tmp/uniform_cuda7_saved_audit_prepare_20261011/CANARY7_ATTEMPT2_TRANSPORT_VERIFICATION.json',
 'CUDA_author':T+'celeba_uniform_cuda_evaluation_v1/AUTHOR_DECISION_20261011.json',
 'resume_install':'tmp/uniform_cuda500_root_dispatch_20261011/RESUME1_INSTALL_COMMAND.json',
 'resume_observation':'tmp/uniform_cuda500_root_dispatch_20261011/RESUME1_OBSERVATION_20261011T064429Z.json',
 'current_working_editorial_root':'tmp/rebuttal_FL100_native11_prepare_20261011/ROOT_EDITORIAL_ADOPTION.json',
 'entry_update_proof':'tmp/root_CUDA7_resume1_Hybrid29_20261011/ROOT_METADATA_ADOPTION.json',
 'entry_alias_fix':'tmp/root_CUDA7_resume1_Hybrid29_20261011/ROOT_ALIAS_FIX.json',
 'entry_writer':'tmp/refresh_rebuttal_FL100_aliases_root_20261011.py'}
rolepins={k:{'path':p,'sha256':pin(p)['sha256']} for k,p in roles.items()}
metadata=read(roles['entry_update_proof']);entrypaths=[x['path'] for x in metadata['five_entries']]
draft=V+'rebuttal_working_F80_FL100_Hybrid20_native11_20261011.md'
latepaths=set(entrypaths+[T+'TRAINING_STATE.json',draft]);files={};refs={}
def add(p):
 if p in latepaths:return
 pp=pin(p);dest='experimental/'+p[4:] if p.startswith('tmp/') else p
 old=previous['files'].get(dest) or previous.get('parent_references',{}).get(p)
 if old and all(old[k]==pp[k] for k in pp):refs[p]=pp
 else:files[p]=pp
def names(base,names):
 for n in names.split():add(base+'/'+n)
for p in roles.values():add(p)
for p in [priorrel,V+'AUTHOR_REVIEW_START_HERE.md',T+'celeba_uniform_cuda_evaluation_v1/PROTOCOL.md',T+'server_reactivation_20261009/ROOT_COMPACT_20261011T064401Z.json','tmp/record_CUDA7_resume1_Hybrid29_root_20261011.py']:add(p)
names('outputs/guardfed_tables/celeba_flgmm_ten_scenes100_20261011','cells.csv records100.json SOURCE_BINDINGS.json tables.json TABLES.md VERIFICATION.json')
names('tmp/rebuttal_FL100_native11_prepare_20261011','CHECK.json EXACT_EDITS.json SOURCE_SHA256.json CHANGE.diff ROOT_EDITORIAL_GUARD_REPAIR.json ROOT_EDITORIAL_GUARD_EXECUTED_V1.py')
add('tmp/adopt_rebuttal_FL100_native11_root_20261011.py')
names('tmp/NGA96_IID_Benign9_after1_prepare_20261011','collect_delta.py pinned_collect_one.py verify_delta.py run_once.py scope.py SOURCE_FILES_SHA256.json SOURCE_ADAPTATIONS.json SOURCE_CHECK.json SCOPE.json PARENT_PINS.json DEPENDENCIES.json OFFSERVER_ACCEPTANCE.json SAVED_TENSOR_STATE_CHECK.json RAW_STORAGE_INDEX.json SERVER_COLLECTOR_RECEIPT.json ROOT_REVIEW.json ROOT_collect_COMMAND.json ROOT_verify_COMMAND.json ROOT_close_COMMAND.json DOWNLOAD.json')
names('tmp/NGA96_IID_Benign9_root_adopter_prepare_20261011','adopt_root.py FILES_SHA256.json SOURCE_CHECK.json ROOT_INPUTS.json ROOT_ADOPT_COMMAND.json')
names('tmp/hybrid_after19_IIDFedSA10_prepare_20261011','adopt_root.py collect_once.py verify_saved.py restore_verify.py storage.py bind_execution.py execute_transport.py finalize_saved.py FILES_SHA256.json SOURCE_ADAPTATIONS.json SOURCE_CHECK.json SOURCE_DATA_PINS.json DELTA_SCOPE.json FIXED_MANIFEST.json INPUT_PINS.json ROOT_SOURCE_REVIEW.json ROOT_READY_HANDOFF.json ROOT_READY_CHAIN_LINK.json ROOT_ACTUAL_PINS.json EXECUTION_BINDINGS.json RECORDS10.json SAVED_TENSOR_IDENTITY.json OFFSERVER_VERIFICATION.json REMOTE_STRICT.json RAW_STORAGE_INDEX.json MEMBERS.json BACKUP_RECEIPT.json ROOT_ADOPT_COMMAND.json COLLECT_COMMAND.json OUTER_VERIFY_COMMAND.json OUTER_FINALIZE_COMMAND.json')
names('tmp/fl_SpDFA10_final100_isolated_v3_20261011','candidate.py FILES_SHA256.json SOURCE_PREPARATION_SHA256.json SOURCE_CHECK.json SOURCE_PINS.json NAMESPACE_REVIEW.json ROOT_SOURCE_REVIEW.json MANIFEST.json BOUND_EXACT10.json ISOLATION_FIXED.json PROOF_PINS.json')
names('tmp/fl_SpDFA10_final100_isolated_v3_20261011/runtime_v5','launch_once.py deploy_once.py observer.py observation_storage.py ROOT_RUNTIME_INPUTS.json ROOT_RUNTIME_SOURCE_REVIEW.json AUTHORIZATION.json SOURCE_DEPLOYMENT.json LINUX_PREFLIGHT.json START_RECEIPT.json ROOT_LAUNCH_TOOL_COMMAND.json ROOT_DEPLOY_TOOL_COMMAND.json')
names('tmp/fl_isolated_v3_saved_closure_prepare_20261011','SOURCE_PREPARATION_SHA256.json SOURCE_CHECK.json SOURCE_REUSE.json ACTUAL_STARTUP_PINS.json ROOT_SOURCE_REVIEW.json V3_HANDOFF.md RUNTIME_RECEIPT_PATHS.json')
names('tmp/fl_isolated_v3_saved_closure_prepare_20261011/root_closure','adopt_root.py ROOT_INPUTS.json')
names('tmp/fl_isolated_v3_saved_closure_prepare_20261011/actual/saved001','LINUX_SAVED_CHECK.json LINUX_CHECK_COMMAND.json LINUX_CHECK_EXIT.json SAVED_OUTPUTS_AUDIT_NO_REFIT.json TRANSPORT_VERIFICATION.json WINDOWS_AUDIT_COMMAND.json WINDOWS_AUDIT_EXIT.json')
add('tmp/fl_isolated_v3_saved_closure_independent_review_20261011/SOURCE_REVIEW.json')
names('tmp/uniform_cuda7_saved_audit_prepare_20261011','audit_saved.py saved_fit_binding.py original_evaluator.py.txt original_fl_audit.py.txt original_full_saved_science.py.txt original_replay.py.txt backup_attempt2.py FILES_SHA256.json SOURCE_CHECK.json SOURCE_PINS.json FIT_HELPER_INVERSE.json ROOT_INPUTS.json ROOT_SOURCE_REVIEW.json ACTUAL_HANDOFF.json ACTUAL_DELIVERY_SHA256.json OUTER_AUDIT_COMMAND.json OUTER_BACKUP_ATTEMPT2_COMMAND.json CANARY7_ATTEMPT2_BACKUP_COMMAND.json CANARY7_ATTEMPT2_SCP_COMMAND.json')
add('tmp/uniform_cuda7_saved_audit_prepare_20261011/root_adoption/ROOT_ADOPTER_SCHEMA_REPAIR.json')
names('tmp/uniform_cuda500_root_dispatch_20261011','dispatch.py dispatch_resume1.py resume_queue.py DEPLOYMENT_SEAL_V2.json DEPLOYMENT_SEAL.json ROOT_SOURCE_REVIEW.json ACTIVATE_COMMAND.json DEPLOY_COMMAND.json MATERIALIZE_COMMAND.json FULL91005_DIAGNOSIS_20261011T063314Z.json CANARY7_BACKUP_COMMAND.json OBSERVATION_20261011T063233Z.json RESUME1_OBSERVATION_20261011T064329Z.json')
names('tmp/uniform_cuda500_proc_race_recovery_prepare_20261011','REVIEW_SUCCESSOR.json REVIEW.json SOURCE_CHECK.json SUCCESSOR_CHECK.json SUCCESSOR_FILES_SHA256.json PROC_RACE_ONLY.patch')
names('tmp/mechanism_uniform_cuda_scope_prepare_20261011','SOURCE_PINS.json FILES_SHA256.json SOURCE_CHECK.json')
add('tmp/mechanism_uniform_cuda_scope_independent_review_20261011/SOURCE_REVIEW.json')
add('tmp/uniform_cuda500_dispatch_independent_review_20261011/SOURCE_REVIEW.json')
accepted=dict(oldtemplate['current_accepted'],FL_three_view_total=100,FL_complete_table_records=100,FL_complete_table_scenes=10,NGA96_new_scientific_accepted=9,Hybrid_native_new=29,uniform_CUDA_root_accepted=7,uniform_CUDA_remote_observed=11,uniform_CUDA_phase=500)
template=dict(status='UNBOUND_INCREMENT70_INPUTS',root_accepted=False,parent_commit=PARENT,STATE_path=T+'TRAINING_STATE.json',STATE_sha256=None,files=dict(sorted(files.items())),parent_references=dict(sorted(refs.items())),entry_paths=entrypaths,current_accepted=accepted,draft_path=draft,draft_sha256=None,proof_roles=rolepins,late_role_names=[],late_files={p:None for p in sorted(latepaths)},omitted_large_artifacts=oldtemplate['omitted_large_artifacts'],native11_PDF_path=oldtemplate['native11_PDF_path'],publisher_creates_scientific_acceptances=False)
write('INPUTS_TEMPLATE.json',template)
semantic=dict(status='UNBOUND_LIMITED_CURRENT_ENTRY_SEMANTIC_REVIEW',root_accepted=False,STATE_sha256=None,current_accepted=accepted,working_draft_content_cutoff=state['latest_working_rebuttal_draft']['content_cutoff'],historical_Git69_cutoff=previous['acceptance_cutoff'],latest_observation_only=state['latest_current_resource_observation'],entries={p:None for p in entrypaths},source_pins=rolepins,gradient14_deployment_only=True,NGA96_scientific_completed=9,F80_root_accepted=True,replay_failure_preserved=True,uniform_CUDA_root_accepted=7,uniform_CUDA_remote_observed=11,final_test=False,whole_rebuttal_complete=False)
write('SEMANTIC_TEMPLATE.json',semantic)
write('SELECTION_TEMPLATE.json',dict(status='UNBOUND_INCREMENT70_COMPACT_SELECTION',root_accepted=False,parent_commit=PARENT,semantic_review_sha256=None,stable_input_sha256=None,STATE_sha256=None,proof_roles=rolepins,files=None,parent_references=template['parent_references'],root_late_bound_files=None))
references={}
for seal in ['tmp/fl_SpDFA10_final100_isolated_v3_20261011/FILES_SHA256.json','tmp/fl_isolated_v3_saved_closure_prepare_20261011/SOURCE_PREPARATION_SHA256.json','tmp/mechanism_uniform_cuda_scope_prepare_20261011/FILES_SHA256.json','tmp/hybrid_after19_IIDFedSA10_prepare_20261011/FILES_SHA256.json','tmp/uniform_cuda7_saved_audit_prepare_20261011/FILES_SHA256.json','tmp/NGA96_IID_Benign9_after1_prepare_20261011/SOURCE_FILES_SHA256.json']:
 members=read(seal)['files'];omitted={}
 for rel,declared in members.items():
  p=(Path(seal).parent/rel).as_posix()
  if p not in files and p not in refs and p not in latepaths:omitted[rel]=declared
 references[seal]={'seal_sha256':pin(seal)['sha256'],'omitted_members_as_declared_not_reexecuted_or_rehashed':omitted}
write('OMISSIONS.json',dict(large_artifacts=template['omitted_large_artifacts'],excluded=['models','NPZ/arrays','datasets/images','archives','raw per-round logs'],source_references=references,note='Exact source-seal dependency references only; compact publication is not a standalone scientific execution bundle.',no_external_volume_access=True))
oldpath=R/'tmp/publication_increment69_prepared_20261011/publish_guardfed_increment69_20261011.py';old=oldpath.read_bytes();assert H(old)=='227e08c9495e57fa0b5d02a00fc512a25a6682ce58a1f655aec00a0e80e35787'
s=old.decode();steps=[]
def replace(a,b):
 global s
 assert s.count(a)==1,(a[:120],s.count(a))
 before=H(s.encode());s=s.replace(a,b);steps.append(dict(old=a,new=b,before_sha256=before,after_sha256=H(s.encode())))
replace("PARENT='f252db2c041379dc9c49d6e56db559d0a2943985'",'PARENT='+repr(PARENT))
replace('Publish adopted native400/native11 compact evidence with preserved held replay failure and fixed full24 draft.','Publish fixed FL100/CUDA7/Hybrid29/NGA9 compact evidence; preserve negative and historical results.')
replace('ROOT_CLOSED_INCREMENT69_COMPACT_SELECTION','ROOT_CLOSED_INCREMENT70_COMPACT_SELECTION')
replace('ROOT_BOUND_INCREMENT69_INPUTS','ROOT_BOUND_INCREMENT70_INPUTS')
replace('f735d4edfee9a3f0944bbc2f41553dbea2ba1e9bfabfd87573787c95751e43fb',H((D/'INPUTS_TEMPLATE.json').read_bytes()))
a=s.index('previous_rel=');b=s.index("selected=selection['files']");replace(s[a:b],(D/'fact_guards.py.txt').read_text())
replace('assert template[\'omitted_large_artifacts\']["outputs/guardfed_tables/celeba_eleven_method_native_20261011/records_native_1100.json"][\'sha256\']==native11[\'files_sha256\'][\'records_native_1100.json\']\n','')
replace("'publish_guardfed_increment69_20261011.py'","'publish_guardfed_increment70_20261011.py'")
replace("assert not (T/'publication_closed_increment69_verified_20261011.json').exists()","assert not (T/'publication_closed_increment70_verified_20261011.json').exists()")
replace("git('commit','-m','Publish native400 native11 and full24 author-review evidence; preserve held replay failure')","git('commit','-m','Publish FL100 CUDA7 Hybrid29 NGA9 fixed-cutoff evidence and author review')")
a=s.index("cutoff=dict(previous[");replace(s[a:],(D/'receipt_footer.py.txt').read_text())
inverse=s
for e in reversed(steps):
 assert H(inverse.encode())==e['after_sha256'];assert inverse.count(e['new'])==1 if e['new'] else True
 if not e['new']:
  # The removed obsolete assertion is restored at its original exact boundary.
  marker="assert parent_refs==stable_input['parent_references']"
  inverse=inverse.replace(marker,e['old']+marker,1)
 else:inverse=inverse.replace(e['new'],e['old'],1)
 assert H(inverse.encode())==e['before_sha256']
assert inverse.encode()==old
compile(s,'publish_guardfed_increment70_20261011.py','exec')
# Check actual JSON facts using a clearly synthetic review in memory. No approval is saved.
fixture_entry=dict(semantic,STATE_sha256=pin(T+'TRAINING_STATE.json')['sha256'],entries={p:dict(sha256=pin(p)['sha256'],history_exact=True) for p in entrypaths})
def fixture_pinned(path,sha):
 assert H(path.read_bytes())==sha,path
 return json.loads(path.read_bytes())
env=dict(R=R,T=R/T,H=H,PARENT=PARENT,repo_path=lambda p:R/p,pinned_json=fixture_pinned,template=template,roles=rolepins,proofs={k:read(p) for k,p in roles.items() if p.endswith('.json')},entry=fixture_entry,stable_input={'draft_sha256':pin(draft)['sha256']})
exec(compile((D/'fact_guards.py.txt').read_text(),'fixed_metadata_fixture','exec'),env)
checks={}
for a,b in [("G=['git'","pins={}"),("pins={}","git('commit'"),("commit=git('rev-parse'","cutoff=dict(previous[")]:
 x=old.decode()[old.decode().index(a):old.decode().index(b)];y=s[s.index(a):s.index(b)]
 assert x==y;checks[a]=H(x.encode())
(D/'publish_guardfed_increment70_20261011.py').write_bytes(s.encode())
write('INVERSE_DIFF.json',dict(parent_path=oldpath.relative_to(R).as_posix(),parent_sha256=H(old),candidate_sha256=H(s.encode()),exact_raw_inverse=True,steps=steps))
(D/'SOURCE_DIFF.patch').write_text(''.join(difflib.unified_diff(old.decode().splitlines(True),s.splitlines(True),fromfile='increment69',tofile='increment70')),encoding='utf8')
write('SOURCE_CHECK.json',dict(status='PASS_SOURCE_ONLY_RAW_INVERSE_AND_PUBLISH_SAFETY_BODY_EXACT',compile=True,raw_inverse=True,safety_blocks_sha256=checks,actual_JSON_fact_guards_in_memory=True,synthetic_review_only_not_root_approval=True,fixed_files=len(files),fixed_bytes=sum(x['bytes'] for x in files.values()),parent_references=len(refs),future_late_files=len(latepaths),Git_executed=False,F_accessed=False,science_executed=False,root_accepted=False))
print(json.dumps(dict(fixed_files=len(files),fixed_bytes=sum(x['bytes'] for x in files.values()),parent_references=len(refs),publisher_sha256=H(s.encode()))))
