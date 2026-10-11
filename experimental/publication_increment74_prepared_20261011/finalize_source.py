"""Close the source proposal only after the actual metadata adoption exists. No Git/F/SSH."""
from pathlib import Path
import ast,difflib,hashlib,json,argparse
R=Path(__file__).resolve().parents[2];Q=Path(__file__).resolve().parent
H=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(p.read_bytes())
parser=argparse.ArgumentParser();parser.add_argument('--metadata-sha256',required=True);a=parser.parse_args()
assert not (Q/'FILES_SHA256.json').exists()
plan=read(Q/'FREEZE_PLAN.json');inventory=read(Q/'PROVISIONAL_SELECTION.json')
oldtemplate=read(R/'tmp/publication_increment73_prepared_20261011/INPUTS_TEMPLATE.json')
oldpath=R/plan['parent_publisher']['path'];old=oldpath.read_bytes();assert H(old)==plan['parent_publisher']['sha256']
previous=read(R/plan['previous_receipt']['path']);assert H((R/plan['previous_receipt']['path']).read_bytes())==plan['previous_receipt']['sha256']
assert previous['commit']==plan['parent_commit'] and previous['acceptance_cutoff']['uniform_CUDA_root_accepted']==211
assert inventory['currently_missing_explicit_paths']==[] and inventory['mutable_pins'] is None
assert plan['scientific_closure_received'] is True
def pin(rel):
 p=R/rel;assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(R.resolve()) and p.stat().st_size<2000000,rel
 return {'sha256':H(p.read_bytes()),'bytes':p.stat().st_size}
def write(name,value):
 p=Q/name;assert not p.exists(),name
 p.write_bytes(value.encode() if isinstance(value,str) else (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())
meta_rel=plan['metadata_root_pending'];assert pin(meta_rel)['sha256']==a.metadata_sha256
meta=read(R/meta_rel);assert meta['status']=='ROOT_CUDA407_FULL100_LOGO100_GIT73_CURRENT_ENTRIES_RECORDED'
assert meta['root_adopted_CUDA']==407 and meta['history_tails_exact'] is True and meta['Git74_published'] is False
assert pin(plan['STATE_path'])['sha256']==meta['state_after']['sha256']
for p in meta['five_entries']+[meta['reader'],meta['protocol']]:assert pin(p['path'])['sha256']==p['sha256']
files=inventory['files'].copy();refs=inventory['parent_references'].copy()
for rel,p in files.items():assert pin(rel)==p,rel
roles={k:v for k,v in plan['future_proof_roles'].items() if k!='metadata'}
roles['metadata']={'path':meta_rel,**pin(meta_rel)}
for k in ['health','cuda500','cuda800']:
 p=meta['evidence'][k];roles[k]={'path':p['path'],**pin(p['path'])};assert roles[k]['sha256']==p['sha256']
for p in roles.values():
 assert pin(p['path'])=={k:p[k] for k in ['sha256','bytes']};files[p['path']]={k:p[k] for k in ['sha256','bytes']}
for p in meta['evidence'].values():
 if p['path'] in plan['late_mutable_names']:continue
 actual=pin(p['path']);assert actual['sha256']==p['sha256'];dest='experimental/'+p['path'][4:] if p['path'].startswith('tmp/') else p['path']
 prev=previous['files'].get(dest) or previous.get('parent_references',{}).get(p['path'])
 if prev and (prev['sha256'],prev['bytes'])==(actual['sha256'],actual['bytes']) and p['path'] not in {v['path'] for v in roles.values()}:refs[p['path']]=actual
 else:files[p['path']]=actual
for rel,p in oldtemplate['parent_references'].items():assert pin(rel)==p;refs[rel]=p
for rel in list(refs):
 if rel in files:del refs[rel]
current=dict(oldtemplate['current_accepted']);current.update(uniform_CUDA_root_accepted=407,uniform_CUDA_remote_observed=108,uniform_CUDA_old800_remote_observed=277,uniform_CUDA_LoGo_root_accepted=100,uniform_CUDA_Full100=100,uniform_CUDA_first500_unique=105,uniform_CUDA_partial_four_method_records=400)
own={'prepare_source.py','finalize_source.py','current_guards.py.txt','FREEZE_PLAN.json','PROVISIONAL_SELECTION.json','publish_guardfed_increment74_20261011.py','INPUTS_TEMPLATE.json','SELECTION_TEMPLATE.json','SEMANTIC_TEMPLATE.json','EXPLICIT_SELECTION_PROPOSAL.json','SOURCE_DIFF.patch','INVERSE_DIFF.json','SOURCE_REUSE.json','SOURCE_CHECK.json','OMISSIONS.json','README.md','HANDOFF.json','FILES_SHA256.json','STABLE_INPUTS.json'}
template={k:oldtemplate[k] for k in ['entry_paths','STATE_path','reader_path','protocol_path','draft_path','draft_sha256','native11_PDF_path']}
large={k:v for k,v in inventory['omissions'].items() if v['bytes']>2000000}
template.update(status='SOURCE_ONLY_INCREMENT74_INPUTS_TEMPLATE',root_accepted=False,parent_commit=plan['parent_commit'],files=files,parent_references=refs,proof_roles=roles,current_accepted=current,fixed_observed_cutoff=plan['fixed_observed_cutoff'],late_role_names=[],late_mutable_names=plan['late_mutable_names'],omitted_large_artifacts=large,own_names=sorted(own),STATE_sha256=None,semantic_review_sha256=None,scope='CLOSED_CUDA407_FULL100_LOGO100_PARTIAL_NATIVE400_NOT_COMPLETE_COMPARISON',scientific_results_created_by_preparation=0)
write('INPUTS_TEMPLATE.json',template)
write('EXPLICIT_SELECTION_PROPOSAL.json',dict(status='SOURCE_ONLY_CLOSED_FILES_NOT_GIT_STAGED',files=files,parent_references=refs,source_seals=inventory['source_seals'],future_mutable_pins=None,no_directory_expansion=True))
write('SELECTION_TEMPLATE.json',dict(status='ROOT_CLOSED_INCREMENT74_COMPACT_SELECTION',root_accepted=False,parent_commit=plan['parent_commit'],STATE_sha256=None,semantic_review_sha256=None,stable_input_sha256=None,proof_roles=roles,files=None,parent_references=refs,root_late_bound_files=None,own_names=sorted(own),binding_rule='Root binds exact8 mutable bytes to STABLE_INPUTS.late_files and files; selection is stable files plus exact own_names. No extra paths.'))
write('SEMANTIC_TEMPLATE.json',dict(status='PASS_LIMITED_CURRENT_ENTRY_SEMANTIC_REVIEW',root_accepted=False,STATE_sha256=None,source_pins=roles,current_accepted=current,fixed_observed_cutoff=plan['fixed_observed_cutoff'],latest_observation_only=None,entries={p:dict(sha256=None,history_exact=False) for p in template['entry_paths']},reader=dict(path=template['reader_path'],sha256=None),protocol=dict(path=template['protocol_path'],sha256=None),working_draft_content_cutoff=None,no_final_test_or_complete_comparison_claim=False,historical_Git73_cutoff=previous['acceptance_cutoff']))
write('OMISSIONS.json',dict(omitted=inventory['omissions'],bulk='No F/remote arrays, models, pickle, images, archives, dataset or environment payload. Exact archive/member/root pins retained.',not_complete_source_mirror=True))
text=old.decode().replace('increment73','increment74').replace('INCREMENT73','INCREMENT74')
text=text.replace("PARENT='51a27b50c6c1880d3fb3c6b5c45b750d0cc68d12'","PARENT="+repr(plan['parent_commit']))
text=text.replace('Publish closed CUDA211/FL100/LoGo2 compact proofs; remaining98 start is not acceptance.','Publish closed CUDA407/Full100/LoGo100 and partial four-method native400 proofs.')
text=text.replace('04a82a823623c070ef58a5292d825c81fcffff402da9f6b0292e517afa09874f',H((Q/'INPUTS_TEMPLATE.json').read_bytes()))
start=text.index("for key in ['parent_references'");end=text.index("selected=selection['files']",start)
text=text[:start]+(Q/'current_guards.py.txt').read_text(encoding='utf8')+'\n'+text[end:]
start=text.index('own_names=');end=text.index('\nown_paths=',start)
text=text[:start]+'own_names='+repr(set(sorted(own)))+"\nassert own_names==set(template['own_names'])"+text[end:]
text=text.replace('Publish CUDA211 FL100 LoGo2 and preserved prerequisite failures','Publish CUDA407 Full100 LoGo100 and partial four-method validation table')
start=text.index('cutoff=dict(')
text=text[:start]+'''cutoff=dict(previous['acceptance_cutoff'],uniform_CUDA_root_accepted=407,uniform_CUDA_first500_root_accepted=105,uniform_CUDA_Full100=100,uniform_CUDA_old800_root_accepted=102,uniform_CUDA_FL100_root_accepted=100,uniform_CUDA_LoGo_root_accepted=100,uniform_CUDA_partial_four_method_records=400)
proof=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',verified_utc=datetime.now(timezone.utc).isoformat(),parent=PARENT,commit=commit,branch=BRANCH,committed_blobs_sha256_verified=len(changed),files={p:pins[p] for p in changed},acceptance_cutoff=cutoff,proof_roles=roles,STATE_sha256=entry['STATE_sha256'],semantic_review_sha256=a.semantic_review_sha256,compact_selection_sha256=a.compact_selection_sha256,parent_references=parent_refs,observed_only_cutoff=entry['fixed_observed_cutoff'],historical_Git73_cutoff=previous['acceptance_cutoff'],working_draft_content_cutoff=entry['working_draft_content_cutoff'],new_scientific_acceptances=0,scientific_acceptances_created_by_publisher=0,new_adopted_views_published={'Full100_new_unique':98,'LoGo100_new_unique':98},whole_comparison_complete=False,whole_rebuttal_complete=False,final_test=False,primary_endpoint_selected=False,original_negative_results_preserved=True,scientific_scope='CUDA407=105 first500 +102 unique old800 +100 FL +100 LoGo. Full100 and LoGo100 each reuse prior2 and add98. Four-method native400 validation table is partial and adds zero scientific records. Observations108/277/main447/NGA27/Hybrid37 are separate from native400/NGA19/Hybrid29 acceptance. Fixed24 draft and old CPU tables/failures preserved. Compact proof publication only, no final comparison or test claim.')
(T/'publication_closed_increment74_verified_20261011.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\\n',encoding='utf8')
print(json.dumps({k:v for k,v in proof.items() if k!='files'}))
'''
new=text.encode();compile(new,'publisher74','exec');write('publish_guardfed_increment74_20261011.py',text)
checks=[]
for start,end in [(b"G=['git'",b'\n# No source mutation'),(b'# No source mutation',b"\ngit('commit'"),(b"raw=git('cat-file'",b'\ncutoff=dict(')]:
 x=old[old.index(start):old.index(end,old.index(start))];y=new[new.index(start):new.index(end,new.index(start))]
 assert x==y and ast.dump(ast.parse(x),include_attributes=False)==ast.dump(ast.parse(y),include_attributes=False)
 checks.append(dict(start=start.decode(),bytes=len(x),sha256=H(x),raw_bytes_exact=True,AST_exact=True))
ndiff=list(difflib.ndiff(old.decode().splitlines(keepends=True),text.splitlines(keepends=True)))
assert ''.join(difflib.restore(ndiff,1)).encode()==old and ''.join(difflib.restore(ndiff,2)).encode()==new
write('INVERSE_DIFF.json',dict(parent_path=plan['parent_publisher']['path'],parent_sha256=H(old),new_sha256=H(new),ndiff=ndiff,old_raw_byte_restore_exact=True,new_raw_byte_restore_exact=True))
write('SOURCE_DIFF.patch',''.join(difflib.unified_diff(old.decode().splitlines(True),text.splitlines(True),fromfile='increment73',tofile='increment74')))
write('SOURCE_REUSE.json',dict(parent_source_path=plan['parent_publisher']['path'],parent_source_sha256=H(old),parent_commit=plan['parent_commit'],three_original_blocks=checks,changes='Only parent/namespace/template/closed407 metadata/partial400 guards, exact whitelist and receipt wording; Git/F/copy/blob/push blocks raw exact.',scientific_execution=False))
node=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='own_names' for t in n.targets));assert ast.literal_eval(node.value)==own==set(template['own_names'])
assert sum(p['bytes'] for p in files.values())<100000000
write('SOURCE_CHECK.json',dict(status='SOURCE_ONLY_STATIC_COMPILE_RAW_INVERSE_AND_CLOSED_SELECTION_PASS',publisher_sha256=H(new),parent_sha256=H(old),core_blocks=checks,both_full_raw_inverse_exact=True,own_names_source_templates_exact=True,fixed_files=len(files),fixed_bytes=sum(p['bytes'] for p in files.values()),parent_refs=len(refs),proof_roles=len(roles),late_mutable_files=8,future_root_authorization_in_templates=False,future_pins_are_null=True,Git_F_SSH_science_execution=False))
readme='''# Increment74 source-only handoff

Closed evidence is CUDA407 = first500 unique105 + old800 unique102 + FL100 + LoGo100. Full100 and LoGo100 each add98 to prior2. Partial four-method native400 table uses only adopted CUDA records, includes FedAvg1/LoGo1 constant predictors, and adds zero scientific records. Observed500108/800277/main447/NGA27/Hybrid37 do not increase accepted native400/NGA19/Hybrid29. Fixed full24 draft SHA910b263 is unchanged.

Root must read current eight mutable files, verify metadata.state_after plus five suffixes/reader/protocol, then bind STABLE_INPUTS from INPUTS_TEMPLATE with status ROOT_BOUND_INCREMENT74_INPUTS and root_accepted true. Copy fixed files/proof roles/parent references/cutoff exactly; late_files is a mapping for exact late_mutable_names, not a template field named late_files. Fill actual semantic review from SEMANTIC_TEMPLATE. COMPACT_SELECTION must equal stable files plus exact own_names; root_late_bound_files is exact own_names. Hash original bytes and reject any stale metadata/source before F/Git operations. Receipt73.commit is parent74; receipt73.parent is older72. INVERSE_DIFF uses parent_path/parent_sha256, not parent.

Root alone executes publisher74 with --semantic-review PATH --semantic-review-sha256 SHA --compact-selection tmp/publication_increment74_prepared_20261011/COMPACT_SELECTION.json --compact-selection-sha256 SHA. False/null templates cannot execute. Actual bind/review/selection are not fabricated by preparation.

The three original73 Git/F/copy/blob/clean/push/remote blocks are raw-byte and AST exact. Compact explicit selection only, each file below2MB and total below100MB. Models/NPZ/datasets/archives/images/environments remain excluded; extensionless runtime logs are referenced by original command/index SHA without renaming. Source preparation performs no Git, F, SSH or scientific execution. Old refusal/failure sources remain preserved.
'''
assert (Q/'README.md').exists();(Q/'README.md').write_text(readme,encoding='utf8')
write('HANDOFF.json',dict(status='SOURCE_ONLY_INCREMENT74_CLOSED_CUDA407_READY_FOR_ROOT_BINDING',publisher=dict(path=(Q/'publish_guardfed_increment74_20261011.py').relative_to(R).as_posix(),sha256=H(new)),parent_commit=plan['parent_commit'],previous_receipt=plan['previous_receipt'],fixed_files=len(files),fixed_bytes=sum(p['bytes'] for p in files.values()),parent_references=len(refs),proof_roles=len(roles),accepted_CUDA=407,Full100=100,LoGo100=100,partial_four_method_native400=400,metadata=roles['metadata'],late_binding=['eight current mutable byte pins','actual semantic review','STABLE_INPUTS and COMPACT_SELECTION'],scientific_results_created=0,Git_F_SSH_execution=False,root_authorized_publisher_execution=False))
for p in [Path(__file__),Q/'prepare_source.py']:compile(p.read_bytes(),str(p),'exec')
write('FILES_SHA256.json',dict(files={name:dict(sha256=H((Q/name).read_bytes()),bytes=(Q/name).stat().st_size) for name in sorted(own-{'FILES_SHA256.json','STABLE_INPUTS.json'})},source_only=True))
print(json.dumps(dict(publisher_sha256=H(new),seal_sha256=H((Q/'FILES_SHA256.json').read_bytes()),handoff_sha256=H((Q/'HANDOFF.json').read_bytes()),fixed_files=len(files),fixed_bytes=sum(p['bytes'] for p in files.values()),refs=len(refs),roles=len(roles))))
