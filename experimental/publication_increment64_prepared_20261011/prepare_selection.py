"""Read exact compact candidates only; no copying, Git, network or science."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
T=R/'docs/server_deployment_20260923/training_20260923'
N=T/'server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T232753Z'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
files={};groups={}
def add(p,group):
 p=Path(p);rel=p.relative_to(R).as_posix()
 assert p.is_file() and p.stat().st_size<2_000_000 and not p.is_symlink(),rel
 assert p.suffix in {'.py','.json','.md','.patch','.txt','.stderr','.stdout','.sha256','.csv','.tex','.diff'},rel
 assert not any(x in p.parts for x in ('verified_extract','restored','__pycache__','models','arrays')),rel
 files[rel]={'sha256':sha(p),'bytes':p.stat().st_size};groups.setdefault(group,[]).append(rel)
def names(base,ns,group):
 for n in ns:add(Path(base)/n,group)
def seal_members(base,name,group):
 v=read(base/name)['files'];add(base/name,group)
 for rel,pin in v.items():
  p=base/rel;assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'];add(p,group)
def main():
 parent=read(T/'publication_closed_increment63_verified_20261011.json')
 assert parent['commit']=='ccb5207c24762d3ac92ed76b40bd40b3e2330954'
 names(N,['ROOT_DELTA_VERIFICATION.json','ARCHIVE_LOCATION.json','DELTA_VERIFICATION.json','OFFSERVER_VERIFICATION.json','REMOTE_COMMANDS.json','STORAGE_PREFLIGHT.json','verified_ledger.json','root_delta_20261010T232753Z.tar.gz.receipt.json','ROOT_REMOTE_LEDGER_PROMOTION.stdout.json','ROOT_REMOTE_LEDGER_PROMOTION.stderr.txt','inspection/inspection.json','inspection/inspection.sha256'],'native350')
 names(R/'tmp/celeba_native_after341_20261011',['run_once.py','adopt_root.py','COLLECTOR_SOURCE_REVIEW.json','ROOT_ACTUAL_INPUTS_REVIEW.json','ACTUAL_HANDOFF.json','ACTUAL_DELIVERY_SHA256.json','ACTUAL_COLLECT_COMMAND.json','ACTUAL_COLLECT_EXIT.json','SOURCE_CHECK.json','SOURCE_DIFF.patch','FILES_SHA256.json'],'native350_source')
 names(R/'tmp/gradient64_final_acceptance_prepared_20261011',['ROOT_ADOPTION_REVIEW.json','COMPLETE64_SELECTION.json','SELECTED64_ROOT.json','ROOT_READY_HANDOFF.json','ROOT_READY_CHAIN_LINK.json','OFFSERVER_ACCEPTANCE.json','SAVED_TENSOR_STATE_CHECK.json','RAW_STORAGE_INDEX.json','SERVER_COLLECTOR_RECEIPT.json','SOURCE_FILES_SHA256.json','SOURCE_CHECK.json','SOURCE_DIFF.patch','collect_delta.py','pinned_collect_one.py','verify_delta.py','finalize.py','run_once.py','adopt_root_prepared.py','COLLECT_COMMAND.json','VERIFY_COMMAND.json','COLLECTOR_RELEASE.json','ROOT_EXECUTION_REVIEW.json'],'gradient64')
 names(R/'tmp/fl_native_after67_FedSA10_prepared_20261011',['ROOT_ADOPTION_REVIEW.json','ROOT_READY_HANDOFF.json','ROOT_READY_CHAIN_LINK.json','RAW_STORAGE_INDEX.json','SAVED_TENSOR_STATE_CHECK.json','SERVER_COLLECTOR_RECEIPT.json','COLLECTOR_CLOSED.json','ROOT_EXECUTION_REVIEW.json'],'FL_native77')
 F=R/'tmp/fl_FedSA10_three_view_prepared_20261011'
 seal_members(F,'FILES_SHA256.json','FL_exact10_source');seal_members(F/'transport_v2','FILES_SHA256.json','FL_transport_V2_source')
 names(F,['ROOT_SCIENTIFIC_ADOPTION.json','ROOT_INPUTS_v2.json','ROOT_SOURCE_REVIEW.json','saved_v2/STATIC_SOURCE_SHA256.json','saved_v2/FILES_SHA256.json','saved_v2/SOURCE_PINS.json','runtime/AUTHORIZATION.json','runtime/LINUX_PREFLIGHT.json','runtime/SOURCE_DEPLOYMENT.json','runtime/START_RECEIPT.json','runtime/TERMINAL_GATE.json'],'FL_actual_roots')
 seal_members(F/'saved_v2','STATIC_SOURCE_SHA256.json','FL_saved_original_source')
 D=F/'actual/saved002';v=read(D/'ACTUAL_DELIVERY_FILES_SHA256.json')['files'];add(D/'ACTUAL_DELIVERY_FILES_SHA256.json','FL_actual_saved')
 for rel,pin in v.items():
  p=F/rel;assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'];add(p,'FL_actual_saved')
 names(F/'actual/saved001',['TRANSPORT_COMMAND.json','TRANSPORT_SSH_EXIT.json','TRANSPORT_STDERR.txt','TRANSPORT_REMOTE_BOUND.py'],'FL_preserved_transport_failure')
 B=R/'outputs/guardfed_tables/celeba_flgmm_eight_scenes80_20261011'
 names(B,['ROOT_VERIFICATION.json','SOURCE_BINDINGS.json','VERIFICATION.json','records81.json','tables.json','TABLES.md','cells.csv'],'FL_table80')
 G=R/'tmp/gradient_source_freeze_deployment_index_20261011'
 names(G/'freeze_actual001',['ROOT_SOURCE_APPROVAL.json','COMMAND.json','SHA_RECEIPT.json','REMOTE.py'],'gradient200_source_freeze')
 receipt=read(G/'freeze_actual001/SHA_RECEIPT.json');assert len(receipt['files'])==221
 for rel,pin in receipt['files'].items():
  p=G/'freeze_actual001/code'/rel;assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'];add(p,'gradient200_frozen_code221')
 names(G/'bind14_actual001/metadata',['scope.json','shared_cache_bindings.json'],'gradient14_fixed_metadata')
 for row in read(G/'bind14_actual001/metadata/scope.json')['jobs']:add(G/'bind14_actual001/metadata'/row['job'],'gradient14_fixed_jobs')
 names(G/'authorize14_actual001',['ROOT_AUTHORIZATION.json','COMMAND.json','SHA_RECEIPT.json'],'gradient14_authorization_metadata')
 names(G/'gate14_job00_actual002',['ITEM_PROOF.json'],'gradient14_single_actual_item_only')
 names(R/'tmp/gradient14_resource_preflight_prepared_20261011',['preflight.py','README.md','FILES_SHA256.json','SOURCE_CHECK.json','METADATA_CHECKS.json'],'gradient14_source')
 names(R/'tmp/rebuttal_working_F40_gradient64_update_20261011',['ROOT_EDITORIAL_ADOPTION.json','CHANGESET.json','DIFF.patch','EDITORIAL_VERIFICATION.json','HANDOFF.md'],'working_F40_gradient64')
 add(R/'docs/server_deployment_20260923/revision_20260923/rebuttal_working_F40_gradient64_20261011.md','working_F40_gradient64')
 add(T/'publication_closed_increment63_verified_20261011.json','parent63_receipt')
 parent_by_source={p.get('source'):{'sha256':p['sha256'],'bytes':p['bytes']} for p in parent['files'].values() if 'source' in p}
 refs={rel:pin for rel,pin in files.items() if parent_by_source.get(rel)==pin}
 role_paths={'native350_root':N/'ROOT_DELTA_VERIFICATION.json','FL_native77_root':R/'tmp/fl_native_after67_FedSA10_prepared_20261011/ROOT_ADOPTION_REVIEW.json','FL_three_view81_root':F/'ROOT_SCIENTIFIC_ADOPTION.json','FL_eight_scene80_table_root':B/'ROOT_VERIFICATION.json','gradient64_root':R/'tmp/gradient64_final_acceptance_prepared_20261011/ROOT_ADOPTION_REVIEW.json','gradient64_selected_root':R/'tmp/gradient64_final_acceptance_prepared_20261011/SELECTED64_ROOT.json','gradient200_frozen_source_root':G/'freeze_actual001/ROOT_SOURCE_APPROVAL.json','working_editorial_root':R/'tmp/rebuttal_working_F40_gradient64_update_20261011/ROOT_EDITORIAL_ADOPTION.json'}
 roles={role:{'path':p.relative_to(R).as_posix(),'sha256':sha(p)} for role,p in role_paths.items()}
 for pin in roles.values():refs.pop(pin['path'],None)
 stable={'status':'ACTUAL_STABLE_SOURCE_CANDIDATES_ONLY_NOT_FROZEN_OR_PUBLISHED','parent_commit':parent['commit'],'files':files,'proof_roles':roles,'groups':groups,'omissions':[{'scope':'FL previous70 duplicate copies','reason':'Canonical new81/80 table selected once; prior70 remains parent evidence.'},{'scope':'F/server bulk and active gate14 outputs','reason':'No archives, NPZ, models, raw logs or running item directories are selected.'}]}
 candidate={'status':'PENDING_ROOT_BINDING_NOT_EXECUTABLE','root_accepted':False,'parent_commit':parent['commit'],'semantic_review_sha256':None,'STATE_sha256':None,'files':{r:p for r,p in files.items() if r not in refs},'parent_references':refs,'root_late_bound_files':{},'proof_roles':roles,'pending':['final STATE + five entries','actual entry writer/update proof','final root semantic review','root-selected exact14 compact metadata, never192 results'],'acceptance_cutoff_target':{'native':350,'three_view':340,'FL_native_new':77,'FL_three_view_total':81,'FL_complete_table_records':80,'FL_complete_table_scenes':8,'gradient_accepted':64,'Hybrid_native_new':12,'gradient200_new_scientific_accepted':0},'working_draft_content_cutoff':{'native':341,'three_view':340,'F_pairs':40,'F_scenes':4,'gradient_search':64},'copied_F_or_Git_files':0}
 for name,value in [('STABLE_INPUTS.json',stable),('SELECTION_CANDIDATE.json',candidate)]:
  with (H/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
 print(json.dumps({'stable_candidates':len(files),'bytes':sum(v['bytes'] for v in files.values()),'samebyte_parent_refs_suggested':len(refs),'frozen_code_members':221,'STABLE_INPUTS_sha256':sha(H/'STABLE_INPUTS.json')}))
if __name__=='__main__':main()
