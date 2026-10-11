"""Prepare only the explicit increment69 sources/list on E; no F/Git/network calls."""
from pathlib import Path
import ast, difflib, hashlib, json

R=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
T='docs/server_deployment_20260923/training_20260923/'
V='docs/server_deployment_20260923/revision_20260923/'
PARENT='f252db2c041379dc9c49d6e56db559d0a2943985'
OLD=R/'tmp/publication_increment68_prepared_20261011/publish_guardfed_increment68_20261011.py'
H=lambda b:hashlib.sha256(b).hexdigest()
save=lambda n,x:(HERE/n).write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode())
state=json.loads((R/(T+'TRAINING_STATE.json')).read_bytes())
assert H((R/(T+'TRAINING_STATE.json')).read_bytes())=='ed689b5750d41c8954b36c2672b141b67d40ced750231eebd99fd0ce16a982f9'
prev_path=T+'publication_closed_increment68_verified_20261011.json';previous=json.loads((R/prev_path).read_bytes())
assert previous['commit']==PARENT and H((R/prev_path).read_bytes())=='cc07ed80bf338e062a7ced2e5e8b83502b91abaa5fd9fc29263837dcfb6cc964'
prior_template=json.loads((OLD.parent/'INPUTS_TEMPLATE.json').read_bytes())
draft=state['latest_working_rebuttal_draft'];roles={
 'previous68':prev_path,
 'native400_root':state['celeba_mechanism_v1']['latest_native_root_proof_path'],
 'native380_parent_root':T+'server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261011T024637Z/ROOT_DELTA_VERIFICATION.json',
 'FL96_native_root':'tmp/fl_native_final10_prepared_20261011/ROOT_ADOPTION_REVIEW.json',
 'native11_table_root':'outputs/guardfed_tables/celeba_eleven_method_native_20261011/ROOT_REVIEW.json',
 'native11_PDF_root':'outputs/guardfed_tables/celeba_eleven_method_native_pdf_20261011/ROOT_REVIEW.json',
 'current_working_editorial_root':draft['root_proof']['path'],
 'statistical_scope_correction_root':'tmp/rebuttal_statistics_scope_root_20261011/ROOT_EDITORIAL_ADOPTION.json',
 'method_appendix_root':V+'method_information_access_paper_appendix_20261011/ROOT_EDITORIAL_REVIEW.json',
 'NGA_first1_root':state['NGA96_coverage_acceptance']['root_proof']['path'],
 'remaining213_numerical_failure':state['mechanism_replay_failure_20261011']['evidence']['path'],
 'entry_writer':state['current_entry_writer']['path'],
 'entry_update_proof':'tmp/root_native400_native11_replay_failure_20261011/ROOT_METADATA_ADOPTION.json',
 'latest_observation_raw':state['latest_five_queue_readonly_observation']['snapshot']['path'],
}
roles={k:{'path':p,'sha256':H((R/p).read_bytes())} for k,p in roles.items()}
files={};refs={};omissions=dict(prior_template['omitted_large_artifacts'])
allowed={'.py','.json','.md','.patch','.txt','.stderr','.stdout','.sha256','.csv','.tex','.diff'}
pdf=state['celeba_native_eleven_method_PDF_20261011']['pdf']['path']
mandatory={pin['path'] for pin in roles.values()}
def add(rel):
 p=R/rel
 assert ':' not in rel and '\\' not in rel and '..' not in p.relative_to(R).parts
 assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(R.resolve()),rel
 assert (p.suffix in allowed or rel==pdf) and p.stat().st_size<2_000_000,rel
 pin={'sha256':H(p.read_bytes()),'bytes':p.stat().st_size}
 dest='experimental/'+rel[4:] if rel.startswith('tmp/') else rel
 old=previous['files'].get(dest) or previous.get('parent_references',{}).get(rel)
 if rel not in mandatory and old and (old['sha256'],old['bytes'])==(pin['sha256'],pin['bytes']):refs[rel]=pin
 else:files[rel]=pin
def flat(folder,exclude=()):
 # Enumerate only this named folder; never recurse into runtime, bulk or history.
 for p in sorted((R/folder).iterdir()):
  if p.is_file() and p.name not in exclude and (p.suffix in allowed or p.relative_to(R).as_posix()==pdf):add(p.relative_to(R).as_posix())
for folder in [
 'tmp/mechanism_F_final20_native_prepare_20261011','tmp/native400_adopter_prepare_20261011',
 'tmp/native400_adopter_independent_review_20261011','tmp/F_final20_native_independent_review_20261011',
 'tmp/celeba_eleven_method_native_table_prepare_20261011','tmp/eleven_native_actual_binder_review_20261011',
 'outputs/guardfed_tables/celeba_eleven_method_native_20261011',
 'outputs/guardfed_tables/celeba_eleven_method_native_pdf_20261011',
 'tmp/rebuttal_native11_increment_prepare_20261011','tmp/rebuttal_native11_root_adoption_20261011',
 V+'method_information_access_paper_appendix_20261011',
 'tmp/NGA96_incremental_acceptance_prepare_20261011',
]:flat(folder,{'records_native_1100.json'})
# All65 original compact delivery members; root adoption and the seal are additional actual files.
fl='tmp/fl_native_final10_prepared_20261011/'
fl_seal=json.loads((R/(fl+'DELIVERY_FILES_SHA256.json')).read_bytes())
for name,pin in fl_seal['files'].items():
 assert '/' not in name and '\\' not in name
 p=R/(fl+name);assert (H(p.read_bytes()),p.stat().st_size)==(pin['sha256'],pin['bytes'])
 add(fl+name)
add(fl+'DELIVERY_FILES_SHA256.json');add(fl+'ROOT_ADOPTION_REVIEW.json')
native_delta=T+'server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261011T050221Z/'
for name in ['ARCHIVE_LOCATION.json','DELTA_VERIFICATION.json','OFFSERVER_VERIFICATION.json','REMOTE_COMMANDS.json','root_delta_20261011T050221Z.tar.gz.receipt.json','ROOT_DELTA_VERIFICATION.json','ROOT_REMOTE_LEDGER_PROMOTION.stderr.txt','ROOT_REMOTE_LEDGER_PROMOTION.stdout.json','STORAGE_PREFLIGHT.json','verified_ledger.json','inspection/inspection.sha256']:add(native_delta+name)
Hdir='tmp/fl_final10_completed_aux_runtime_prepare_20261011/'
for name in ['ACTUAL_DIAGNOSIS.json','ACTUAL_DIAGNOSIS.stderr','ACTUAL_DIAGNOSIS_COMMAND.json','ACTUAL_NATIVE_ARTIFACTS.json','ACTUAL_NATIVE_ARTIFACTS.stderr','ACTUAL_NATIVE_ARTIFACTS_COMMAND.json','ACTUAL_ROUND70.json','ACTUAL_ROUND70.stderr','ACTUAL_ROUND70_COMMAND.json','diagnose_remaining620_executed_v1.py','EXECUTED_SOURCE_PREIMAGE_GAP.json','EXECUTED_SOURCE_PREIMAGE_RECOVERED.json','read_failed_native_artifacts.py','read_failed_round70.py','READ_ONLY_CAUSAL_DIAGNOSIS.json']:add(Hdir+name)
assert H((R/(Hdir+'diagnose_remaining620_executed_v1.py')).read_bytes())=='8248febae90a3824f30bdc2dfda2f76ac40c1e3a7b25824ad5ffeeb5907f13b2'
for rel in [OLD.relative_to(R).as_posix(),prior_template['draft_path'],draft['entry'],V+'AUTHOR_REVIEW_START_HERE.md',V+'rebuttal_working_F80_FL90_Hybrid20_statistics_clarified_20261011.md',V+'rebuttal_working_F80_FL90_Hybrid20_clarified_20261011.md','tmp/rebuttal_statistics_scope_root_20261011/ROOT_METADATA_ADOPTION.json','tmp/rebuttal_substantive_submission_audit_20261011/REVIEW.md','tmp/adopt_NGA96_first_delta_root_20261011.py','tmp/celeba_eleven_method_native_table_prepare_20261011/actual_native1100/NUMERIC_VERIFICATION.json']:
 add(rel)
for pin in roles.values():add(pin['path'])
# Small source-reference report replaces duplicate raw records in the public subset.
nr=json.loads((R/roles['native11_table_root']['path']).read_bytes())
record_rel='outputs/guardfed_tables/celeba_eleven_method_native_20261011/records_native_1100.json'
omissions[record_rel]={'sha256':nr['files_sha256']['records_native_1100.json'],'bytes':(R/record_rel).stat().st_size,'reason':'Above2MB; full immutable record source remains local and hash-bound by the adopted native11 root/SOURCE_BINDINGS. No truncation.','same_bytes_source':'tmp/celeba_eleven_method_native_table_prepare_20261011/actual_native1100/records_native_1100.json'}
save('OMISSIONS.json',{'large_artifacts':omissions,'other_compact_omissions':{'native400_full_inspection_and_statistics':'Root/DELTA/archive location/receipt/ledger/inspection.sha256 retain recovery identity; do not repeat500 historical native objects or old CSV/statistic copies.','PDF_preview_PNGs_and_attempt1':'PDF plus original numeric/spatial/visual QA and file seal selected; previews and superseded attempt stay local. The public subset is not a complete source-seal mirror.','duplicate_actual_native1100_summary_copies':'Canonical native11 summary/seed-first statistics/NUMERIC_VERIFICATION selected; duplicate actual_native1100 outputs not repackaged.','runtime_v3_v4_and_FL_final10_execution':'Unaccepted/current development and unstarted FL replay excluded.','models_npz_rawlogs_archives_F_bulk':'No archive, model, array, dataset or F bulk is selected; original proof indices retain actual archive/member paths and hashes.'}})
current=dict(prior_template['current_accepted'],native_new=400,FL_native_new=96,NGA96_new_scientific_accepted=1,native_methods=11,native_evaluation_records=1100)
template=dict(status='INCREMENT69_SOURCE_ONLY_ACTUAL_NATIVE400_NATIVE11_FAILURE_PRESERVED_UNBOUND',root_accepted=False,parent_commit=PARENT,STATE_path=T+'TRAINING_STATE.json',STATE_sha256=None,files=files,parent_references=refs,entry_paths=prior_template['entry_paths'],current_accepted=current,draft_path=draft['entry'],draft_sha256=None,proof_roles=roles,late_role_names=[],late_files={},omitted_large_artifacts=omissions,native11_PDF_path=pdf,publisher_creates_scientific_acceptances=False)
save('INPUTS_TEMPLATE.json',template)
semantic=dict(status='UNBOUND_NOT_APPROVED',root_accepted=False,STATE_sha256=None,current_accepted=current,working_draft_content_cutoff=draft['content_cutoff'],historical_Git68_cutoff=previous['acceptance_cutoff'],latest_observation_only=state['latest_five_queue_readonly_observation'],entries={},source_pins=roles,gradient14_deployment_only=True,NGA96_scientific_completed=1,F80_root_accepted=True,replay_failure_preserved=True,FL_final10_started=False,final_test=False,whole_rebuttal_complete=False)
save('SEMANTIC_TEMPLATE.json',semantic)
save('SELECTION_TEMPLATE.json',dict(status='UNBOUND_NOT_APPROVED',root_accepted=False,parent_commit=PARENT,semantic_review_sha256=None,STATE_sha256=None,stable_input_sha256=None,files={},parent_references=refs,proof_roles=roles,root_late_bound_files={}))
parent=OLD.read_bytes();assert H(parent)=='d60b479ffd8ae0c87968917979eab135f9485fb9d50e4df60f0dd8f20057cbf8'
source=parent.decode();changes=[]
def replace(old,new):
 global source
 assert old!=new and old in source,old[:90]
 count=source.count(old);source=source.replace(old,new);changes.append({'before':old,'after':new,'occurrences':count})
replace('increment68','increment69')
replace('INCREMENT68','INCREMENT69')
replace('Publish root-adopted saved380/F80 compact metadata with future root-bound draft and entries.','Publish adopted native400/native11 compact evidence with preserved held replay failure and fixed full24 draft.')
replace("PARENT='3ffad7675c82c4ec7a8e29db541c437e93969590'",f"PARENT='{PARENT}'")
replace("'3a2b70394bd7705943c45bff9d793ade580325cae286643fe5c475a0839fe7f4'",repr(H((HERE/'INPUTS_TEMPLATE.json').read_bytes())))
replace("'omitted_large_artifacts']:","'omitted_large_artifacts','native11_PDF_path']:")
start=source.index("previous_rel=");end=source.index("selected=selection['files']")
replace(source[start:end],(HERE/'fact_guards.py.txt').read_bytes().decode())
start=source.index("for rel,pin in template['omitted_large_artifacts'].items():");end=source.index('assert parent_refs==',start)
replace(source[start:end],"for rel,pin in template['omitted_large_artifacts'].items():\n    assert rel not in selected and re.fullmatch('[0-9a-f]{64}',pin['sha256']) and pin['bytes']>2_000_000\nassert template['omitted_large_artifacts'][\"outputs/guardfed_tables/celeba_eleven_method_native_20261011/records_native_1100.json\"]['sha256']==native11['files_sha256']['records_native_1100.json']\n")
old_names=source[source.index('own_names='):source.index('\nown_paths=')]
names=['README.md','SELECTION_TEMPLATE.json','INPUTS_TEMPLATE.json','STABLE_INPUTS.json','SEMANTIC_TEMPLATE.json','INVERSE_DIFF.json','SOURCE_CHECK.json','prepare_source.py','SOURCE_DIFF.patch','FILES_SHA256.json','HANDOFF.json','publish_guardfed_increment69_20261011.py','fact_guards.py.txt','receipt_footer.py.txt','OMISSIONS.json']
replace(old_names,'own_names='+repr(set(names)))
replace("p.relative_to(R).as_posix() in NDIFF_ALLOWED_PATHS)","p.relative_to(R).as_posix() in NDIFF_ALLOWED_PATHS or p.relative_to(R).as_posix()==template['native11_PDF_path'])")
replace("git('commit','-m','Publish adopted saved380 F80 compact evidence and complete24 author-review update')","git('commit','-m','Publish native400 native11 and full24 author-review evidence; preserve held replay failure')")
start=source.index('cutoff=dict(previous[')
replace(source[start:],(HERE/'receipt_footer.py.txt').read_bytes().decode())
raw=source.encode();compile(raw,str(HERE/'publish_guardfed_increment69_20261011.py'),'exec')
inverse=source
for change in reversed(changes):
 assert inverse.count(change['after'])==change['occurrences'],change['before'][:90]
 inverse=inverse.replace(change['after'],change['before'])
assert inverse.encode()==parent
blocks=[('Git_F_guard',b"G=['git'",b'pins={}'),('copy_stage_source_hash',b'pins={}',b"git('commit'"),('committed_blob_push_remote',b"commit=git('rev-parse'",b'cutoff=dict(previous[')]
checks={}
for name,start,end in blocks:
 a=parent[parent.index(start):parent.index(end,parent.index(start))];b=raw[raw.index(start):raw.index(end,raw.index(start))]
 assert a==b,name
 assert ast.dump(ast.parse(a.decode()),include_attributes=False)==ast.dump(ast.parse(b.decode()),include_attributes=False),name
 checks[name]={'raw_bytes_exact':True,'AST_exact':True,'sha256':H(a),'bytes':len(a)}
(HERE/'publish_guardfed_increment69_20261011.py').write_bytes(raw)
save('INVERSE_DIFF.json',changes)
(HERE/'SOURCE_DIFF.patch').write_bytes(''.join(difflib.unified_diff(parent.decode().splitlines(keepends=True),source.splitlines(keepends=True),fromfile=OLD.relative_to(R).as_posix(),tofile=(HERE/'publish_guardfed_increment69_20261011.py').relative_to(R).as_posix())).encode())
save('SOURCE_CHECK.json',dict(status='SOURCE_ONLY_COMPILE_RAW_INVERSE_CORE_PASS_NOT_EXECUTED',parent_publisher_sha256=H(parent),publisher_sha256=H(raw),parent_commit=PARENT,full_source_raw_inverse_exact=True,core_blocks=checks,fixed_new_files=len(files),fixed_new_bytes=sum(p['bytes'] for p in files.values()),exact_parent_references=len(refs),all_selected_below2MB=True,explicit_PDF_only=pdf,omitted_large_artifacts=omissions,late_fields=['STATE_sha256','draft_sha256','five_entry_hashes_and_history','STABLE_INPUTS_root_accepted','COMPACT_SELECTION_root_accepted','semantic_actual_path_sha256'],actual_STATE_at_preparation='ed689b5750d41c8954b36c2672b141b67d40ced750231eebd99fd0ce16a982f9',publisher_executed=False,Git=False,F_access=False,SSH=False,root_accepted=False))
print(json.dumps({'source_sha256':H(raw),'selected':len(files),'bytes':sum(p['bytes'] for p in files.values()),'parent_references':len(refs),'inverse':True,'core_raw_AST':True}))
