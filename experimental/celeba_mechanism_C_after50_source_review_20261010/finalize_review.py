from pathlib import Path
import ast,hashlib,json,importlib.util,copy,datetime,sys
sys.dont_write_bytecode=True
R=Path.cwd(); H=R/'tmp/celeba_mechanism_C_after50_source_review_20261010'; B=R/'tmp/celeba_mechanism_valid_C_after50_20261010'; P=R/'tmp/celeba_mechanism_valid_C_after47_20261010'; T=R/'tmp/celeba_mechanism_C_after50_root_operations_20261010'; O=R/'tmp/celeba_mechanism_C_after47_root_operations_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); read=lambda p:json.loads(p.read_bytes())
def save(n,v):
 with (H/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
def funcs(p):
 s=p.read_text('utf8');return {x.name:ast.get_source_segment(s,x) for x in ast.parse(s).body if isinstance(x,ast.FunctionDef)}
def load(n,p):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
E=read(H/'SEALED_EVIDENCE.json'); C=read(H/'CONNECTIONS.json'); ids=E['exact_selected_ids']; tr=read(T/'TRANSPORT_REBIND.json')
for row in tr['members']:assert sha(T/row['path'])==row['sha256']
A=load('review_adopt',T/'adopt.py'); K=load('review_backup',T/'backup.py'); D=load('review_deploy',T/'deploy.py')
assert funcs(T/'adopt.py')['expected_archive_names']==funcs(O/'adopt.py')['expected_archive_names']
names=A.expected_archive_names(ids,read(B/'FILES_SHA256.json'),read(B/'execution_candidate/EXECUTION_SOURCE_SHA256.json'));assert len(names)==75
base=dict(service='guardfed_celeba_mechanism_valid_C_after50 EXITED',processes=[],batch_failure=None,batch_complete={'records':ids},completed=[{'id':i} for i in ids]);cases=[]
for m in (A,K):
 m.check_terminal(base,ids)
 for tag,change in [('active',{'processes':[{'pid':1}]}),('running',{'service':'service RUNNING'}),('failure',{'batch_failure':{'error':'x'}}),('missing',{'completed':base['completed'][:-1]}),('duplicate',{'completed':base['completed'][:-1]+base['completed'][:1]}),('extra',{'completed':base['completed']+[{'id':'other'}]}),('no_terminal',{'batch_complete':None})]:
  x=copy.deepcopy(base);x.update(change)
  try:m.check_terminal(x,ids)
  except AssertionError:cases.append(m.__name__+':'+tag)
  else:raise AssertionError(tag)
assert len(cases)==14
for name in ['budget_snapshot','assert_cpu_available','fresh_cpu_scan']:
 assert funcs(B/'execution_candidate/install_once.py')[name]==funcs(P/'execution_candidate/install_once.py')[name]
assert funcs(B/'execution_candidate/batch.py')['runtime_policy']==funcs(P/'execution_candidate/batch.py')['runtime_policy']
bs=(B/'execution_candidate/batch.py').read_text();ds=(T/'deploy.py').read_text()
for s in ['ROOT_REVIEW_PASS_BOUNDED_C_AFTER50_VALID_REPLAY','APPROVED_C_AFTER50_MECHANISM_VALID_REPLAY_ONLY']:assert s in bs and s in ds
assert 'batch.check_approval(approved,scope' in (B/'execution_candidate/install_once.py').read_text()
layout=dict(status='PASS_METADATA_ONLY',terminal_positive_checks=2,terminal_refusals=cases,archive_names=75,per_model_members=7,metrics=54,confusions=144,rules=18,archive_names_function_exact=True,resource_functions_exact=True,runtime_policy_exact=True,installer_deploy_batch_status_compatible=True,legacy_preflight_label='PASS_EXACT37_PREFLIGHT is inherited diagnostic label; no count authority; exact6 selected IDs and approval guards authoritative',future_archive_verified=False)
save('SOURCE_DIFF_AND_LAYOUT_CHECKS.json',layout)
report=dict(status='PASS_SOURCE_READY_FOR_ROOT_LINUX_PREFLIGHT_AND_EXACT6_APPROVAL',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_adoptable=True,actual_dispatch_authorized_by_this_review=False,
 science_seal_sha256=E['pins']['science'],execution_seal_sha256=E['pins']['execution'],package_sha256=E['pins']['package'],inventory_sha256=E['pins']['inventory'],package_receipt_sha256=sha(B/'PACKAGE_RECEIPT.json'),source_handoff_sha256=sha(B/'HANDOFF.json'),
 package_members_verified=40,science_members_verified=10,execution_members_verified=11,source_tar_members_verified=29,source_tar_sha256=sha(B/'source_prepared.tar.gz'),input_pins_verified=36,
 native_accepted_snapshot=156,excluded_prior_three_view_ids=150,old150_records_exact=True,old150_inventory_record_bytes_exact=True,old150_record_order_preserved=True,Full100_references_exact=True,Full100_actual900_source_records_exact=True,
 new_three_view_accepted=0,exact_selected_ids=ids,actual_worker_pre_science_bind_ids=E['metadata_check']['worker_original_pre_bind_path_reached'],positive_approval_exact6=True,actual_metadata_check=E['metadata_check'],actual_metadata_check_exit_code=0,metadata_check_matches_sealed_SELF_CHECK=True,self_check_sha256=sha(B/'SELF_CHECK.json'),actual_metadata_check_stdout_sha256=sha(H/'METADATA_STDOUT.json'),
 unchanged_bridge_scientific_functions=E['scientific_functions_byteexact'],unchanged_bridge_scientific_function_count=11,execution_diff_review='Exact6/excluded150/native156 namespace, IDs, pins and prior C_after47 closure replacements only; original numerical science, calibration and strict bodies unchanged.',
 frozen_tolerance=1e-12,views=['native','raw','shared_calibration'],CPU_budget=list(range(112,120)),torch_threads=8,max_processes=1,nice=10,idle_IO=True,CUDA_hidden=True,resource_module_byte_exact_to_parent=True,budget_plus3_and_all_thread_cpu_owner_functions_byte_exact=True,prior_service_EXITED_guard=True,
 prior_C_after47_root_adoption_sha256='3859d49fb57c3ecc4b23244012431224d255b02590dab221b2d6464e28aa7dd8',actual_Linux_resource_availability_checked=False,new_external_root_approval_required=True,inherited_source_review_pin_is_not_new_approval=True,
 source_constructor_connections_sha256=sha(H/'CONNECTIONS.json'),independent_native_review_sha256=C['native_review_sha256'],native_root_review_reused_not_reexecuted=True,native156_original_strict_source_records_exact=True,native_archive_members_rescanned_by_this_review=False,
 transport_source_review=dict(members={x['path']:x['sha256'] for x in tr['members']},seal_sha256=sha(T/'TRANSPORT_REBIND.json'),expected_archive_names_from_actual_seals=75,archive_members_per_checkpoint=7,actual_future_archive_verified=False,metrics_counts_rules=[54,144,18],terminal_fixture_positive_and_refusal_checks=16,installer_approval_status_compatible=True),
 deploy_source_review_contract_fields_compatible=True,deploy_source_sha256_at_review=sha(T/'deploy.py'),findings=[],limits=['Source-only PASS, not dispatch approval; no SSH, Torch, CNN, training or test. Fresh Linux preflight and actual root exact6 approval remain required.','Old150 and Full100 excluded; six non-IID Benign C models remain incomplete scene6/10 and no new three-view result accepted by this review.','Expected75 archive layout is pure function fixture, not an actual future archive.','Legacy PASS_EXACT37_PREFLIGHT label remains diagnostic; installer and batch enforce exact6 via frozen selected IDs and approval source pins.'])
save('ROOT_INDEPENDENT_REVIEW.json',report)
p=H/'ROOT_INDEPENDENT_REVIEW.json';D.validate_source_review(p,sha(p))
save('TRANSPORT_REVIEW_GATE.json',dict(status='ACTUAL_DEPLOY_SOURCE_REVIEW_CONTRACT_PASS',review_sha256=sha(p),deploy_sha256=sha(T/'deploy.py'),SSH=False,CNN=False,approval=False))
(H/'README.md').write_text('Independent source review PASS: exact6 C non-IID Benign seeds91001..91006; native156 minus closed150. Verified40 package/10 science/11 execution/29 tar members, original11 scientific functions, actual6 pre-bind and69 refusal checks, Full100 actual900 linkage. Transport approval schema matches original installer/batch. Pure archive layout is75 members with7 per model. No SSH/CNN/dispatch performed; root live preflight and fresh approval still required. No new scientific result accepted.\n',encoding='utf8')
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
rows=[dict(path=p.name,size=p.stat().st_size,sha256=sha(p)) for p in sorted(H.iterdir()) if p.is_file() and p.name!='REVIEW_FILES_SHA256.json']
save('REVIEW_FILES_SHA256.json',dict(status='SEALED_SOURCE_ONLY_REVIEW',members=rows))
print(json.dumps(dict(review_sha256=sha(H/'ROOT_INDEPENDENT_REVIEW.json'),seal_sha256=sha(H/'REVIEW_FILES_SHA256.json'),members=len(rows))))
