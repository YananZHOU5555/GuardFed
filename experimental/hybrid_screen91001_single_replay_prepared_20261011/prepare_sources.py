# Source-only singleton identity binding. No models, arrays, scientific calls or remote execution.
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,sys,difflib
H=Path(__file__).resolve().parent;P=H.parent/'hybrid_missing8_pool32_runtime_prepared_20261011';B=H.parent/'celeba_hybrid_three_view_missing8_prepared_20261011';OLD=H.parent/'celeba_hybrid_three_view_bridge_20261011'
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();pin=lambda p:dict(path=Path(p).resolve().as_posix(),sha256=sha(p),bytes=Path(p).stat().st_size)
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def funcs(s):return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
sourcefile=H.parent/'hybrid_screen91001_identity_gap_review_20261011/MINIMAL_INPUT_PINS.json';assert sha(sourcefile)=='a71896812f2065856c4991a181ad9fc935a5a57aca0282b690eb2790a6e0c848'
q=read(sourcefile);rid=q['id'];mapping={};sources={};diff=[]
def cp(v,rel):
 src=Path(v['path']);assert src.suffix in {'.json','.py'} and src.stat().st_size<2_000_000 and sha(src)==v['sha256']
 dst=H/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes());sources[rel]=pin(src)
 mapping[src.resolve().as_posix()]=dict(kind='package',relative=rel,sha256=sha(dst),bytes=dst.stat().st_size)
for rel in ['originals/fl_candidate.py','originals/bridge.py','originals/SOURCE_REUSE.json','originals/evaluator.py','originals/replay.py','originals/core.py','originals/saved_science.py','originals/saved_output_audit.py','originals/receipt_diff.py']:cp(pin(B/rel),rel)
for k,v in read(B/'MANIFEST.json')['path_map'].items():
 if v.get('relative') in sources:mapping[k]=copy.deepcopy(v)
for k in ['native_root','strict','offserver','members','original_record_checker']:cp(q[k],'proofs/'+k+'.json')
for name in ['screen_scope.json','runtime_protocol.json','body.py']:cp(q['source_metadata'][name],'proofs/'+name)
inputs=dict(scope='ONE_ALREADY_ADOPTED_HYBRID_SCREEN_IDENTITY_ONLY',exact_ids=[rid],root=q['native_root'],strict=q['strict'],offserver=q['offserver'],members=q['members'],record_checker=q['original_record_checker'],scope_pin=q['source_metadata']['screen_scope.json'],protocol_pin=q['source_metadata']['runtime_protocol.json'],body_pin=q['source_metadata']['body.py'],artifacts=q['artifacts'],checkpoint=q['checkpoint'],original_bridge=pin(B/'originals/bridge.py'),original_reuse=pin(B/'originals/SOURCE_REUSE.json'),config_canonical_sha256=q['config_canonical_sha256'],source_role='accepted_screen_reuse',selection_seed=True,original_phase='screen',new_formal_training=0,full_client_ID_partition_recomputed=False)
save(H/'originals/SCREEN_INPUTS.json',inputs)
f=funcs((OLD/'bridge.py').read_text())
identity='''def identity_record(rid, *, checkpoint_sha256=None):
    m = inputs()
    require(m['exact_ids'] == [RID] and rid == RID, 'Only accepted screen91001')
    root, strict, off, members, local = [read_pin(m[k]) for k in ('root','strict','offserver','members','record_checker')]
    require(root['status'] == 'ROOT_LOCKED_AUXILIARY_DELTA_ARCHIVE_MEMBER_AND_ORIGINAL_RECORD_ACCEPTANCE_PASS' and (root['accepted_before'],root['accepted_new'],root['accepted_total']) == (23,4,27), 'Wrong original adoption')
    require(root['strict_receipt_sha256'] == m['strict']['sha256'] and root['offserver_proof_sha256'] == m['offserver']['sha256'] and root['record_check_sha256'] == m['record_checker']['sha256'], 'Broken root chain')
    require(off['status'] == 'ORIGINAL_SERVER_STRICT_PLUS_OFFSERVER_ALL_MEMBERS_AND_CPU_TENSORS_VERIFIED' and local['status'] == 'RECORD_BOUND_ORIGINAL_SCIENTIFIC_AND_WRITER_CHECKS_PASS', 'Original accepted checks missing')
    require(off['inventory_sha256'] == m['members']['sha256'] and off['archive_sha256'] == root['archive_sha256'] and off['server_acceptance_sha256'] == m['strict']['sha256'], 'Broken member/archive links')
    require(rid in strict['accepted_new_ids'] and rid in off['accepted_new_ids'], 'Screen ID absent from accepted delta')
    sr, ore, lr = [one(doc['records'], rid) for doc in (strict,off,local)]
    docs = {}
    for key, artifact in m['artifacts'].items():
        require(members['members'][artifact['member']] == {'sha256':artifact['sha256'],'size':artifact['bytes']}, 'Wrong member')
        docs[key] = read_pin(artifact)
    job, result, prov, acceptance, native = [docs[k] for k in ('job','result','provenance','acceptance','native_replay')]
    read_pin(m['scope_pin']); read_pin(m['protocol_pin'])
    require(prov['scope_sha256'] == acceptance['scope_sha256'] == m['scope_pin']['sha256'] and job['runtime_protocol_sha256'] == m['protocol_pin']['sha256'], 'Scope/protocol changed')
    require(hashlib.sha256(read_pin(m['body_pin'],decode=False)).hexdigest() == prov['local_hashes']['body.py'], 'Original strict body changed')
    require(job['id'] == rid and job['method'] == METHOD and job['phase'] == 'screen' and job['evidence_stage'] == result['evidence_stage'] == 'validation_screen', 'Do not relabel screen as fullcoverage')
    require(job['config'] == result['config'] and job['config']['seed'] == 91001 and job['config']['rounds'] == 70 and job['config']['client_alpha'] == result['alpha'] == 5000.0, 'Config/partition/round drift')
    require(job['config']['celeba_evaluation_split'] == 'valid' and job['config']['celeba_train_limit'] == job['config']['celeba_eval_limit'] == 0 and job['config']['device'] == 'cuda', 'Full valid/historical CUDA required')
    require(job['adapter'] == {'fairness_lambda':20.0,'threshold':0.1} and job['config']['learning_rate'] == .001, 'Original recipe changed')
    canonical = hashlib.sha256(json.dumps(job['config'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    require(canonical == m['config_canonical_sha256'], 'Original config SHA changed')
    require(prov == sr['original_provenance'] and prov['job_sha256'] == acceptance['job_sha256'] == m['artifacts']['job']['sha256'], 'Raw-job/provenance drift')
    require(job['source_hashes'].items() <= prov['source_hashes'].items(), 'Source/data provenance mismatch')
    model=m['checkpoint']; member=members['members'][model['member']]
    require(member == {'sha256':model['sha256'],'size':model['bytes']} and model['sha256'] == sr['model_sha256'] == ore['model_sha256'] == lr['checkpoint_sha256'] == acceptance['artifact_hashes']['model.pt'], 'Checkpoint mismatch')
    require(checkpoint_sha256 is None or checkpoint_sha256 == model['sha256'], 'Caller checkpoint mismatch')
    require(native['checkpoint_tensor_sha256'] == acceptance['checkpoint_tensor_sha256'] == ore['checkpoint_tensor_sha256'], 'Tensor mismatch')
    require([r['round'] for r in result['trajectory_metrics']] == list(range(1,71)) and [r['round'] for r in result['round_summaries']] == list(range(1,71)), 'Incomplete70')
    require(result['metrics'] == result['trajectory_metrics'][-1]['metrics'] == native['metrics'] == sr['metrics'] == lr['metrics'], 'Terminal native mismatch')
    image=result['data_contract']['image_data_contract']
    require(image['actual_train_rows'] == 162770 and image['actual_evaluation_rows'] == native['prediction_count'] == 19867 and native['root_group_label_total'] == result['data_contract']['root_clean_rows'] == 16277, 'Split/root count mismatch')
    require(image['root_image_ids_sha256'] == native['root_image_ids_sha256'] and image['evaluation_image_ids_sha256'] == native['evaluation_image_ids_sha256'] and image['train_eval_disjoint'] and image['root_client_disjoint'], 'Root/valid identity mismatch')
    require(prov['torch'] == '2.11.0+cu128' and prov['cuda_build'] == '12.8' and prov['device'] == 'cuda:0' and prov['cpu_threads'] == 1 and prov['cpu_affinity'] == [104], 'Training environment changed')
    selected=dict(metadata=copy.deepcopy(m['artifacts']),checkpoint=copy.deepcopy(model),source_role='accepted_screen_reuse',selection_seed=True,phase='screen')
    return dict(id=rid,method=METHOD,source_method=job['method'],config=copy.deepcopy(job['config']),distribution='IID',attack='Benign',seed=91001,actual_alpha=5000.0,terminal_round=70,original_split='valid',config_canonical_sha256=canonical,data_contract=copy.deepcopy(image),prior_validation_metrics=copy.deepcopy(result['metrics']),checkpoint=copy.deepcopy(model),result=copy.deepcopy(m['artifacts']['result']),raw_job=copy.deepcopy(m['artifacts']['job']),source_hashes=copy.deepcopy(prov['source_hashes']),adapter_source_hashes=copy.deepcopy(prov['local_hashes']),training_torch=prov['torch'],original_training_provenance=copy.deepcopy(prov),original_remote_output=model['server_path'].rsplit('/',1)[0],external_proof_sha256={k:m[k]['sha256'] for k in ('root','strict','offserver','members','record_checker')},original_artifact_pins=selected,source_role='accepted_screen_reuse',selection_seed=True,phase='screen',full_client_ID_partition_recomputed=False,status='PRIVATE_IDENTITY_ONLY_NO_NEW_PREDICTION_OR_FIT',dispatch_authorized=False)
'''
bridge='from __future__ import annotations\nimport ast,copy,hashlib,importlib.util,json,sys\nfrom pathlib import Path\nHERE=Path(__file__).resolve().parent\nMETHOD="CosineFairnessHybrid"\nRID='+repr(rid)+'\nINPUTS_SHA256='+repr(sha(H/'originals/SCREEN_INPUTS.json'))+'\n\n'
for name in ['require','read_pin','inputs','original_bridge','science_bindings','one']:bridge+=f[name].replace("HERE / 'INPUTS.json'","HERE / 'SCREEN_INPUTS.json'")+'\n\n'
bridge+=identity;(H/'originals/hybrid_bridge.py').write_text(bridge,encoding='utf8');compile(bridge,'bridge','exec')
spec=importlib.util.spec_from_file_location('_singleton_metadata',H/'originals/hybrid_bridge.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
record=b.identity_record(rid,checkpoint_sha256=q['checkpoint']['sha256'])
for rel in ['originals/hybrid_bridge.py','originals/SCREEN_INPUTS.json']:
 v=pin(H/rel);mapping[v['path']]=dict(kind='package',relative=rel,sha256=v['sha256'],bytes=v['bytes'])
for key,v in q['artifacts'].items():mapping[Path(v['path']).resolve().as_posix()]=dict(kind='server',server_path=v['server_path'],sha256=v['sha256'],bytes=v['bytes'])
model=dict(path=q['checkpoint']['local_path'],sha256=q['checkpoint']['sha256'],bytes=q['checkpoint']['bytes'],member=q['checkpoint']['member'],server_path=q['checkpoint']['server_path'])
mapping[Path(model['path']).resolve().as_posix()]=dict(kind='server',server_path=model['server_path'],sha256=model['sha256'],bytes=model['bytes'])
artifacts=copy.deepcopy(q['artifacts']);artifacts['model']=model
row=dict(id=rid,method='CosineFairnessHybrid',distribution='IID',attack='Benign',seed=91001,terminal_round=70,split='valid',n_eval=19867,actual_alpha=5000.,identity=record,runtime_output=record['original_remote_output'],runtime_artifacts=artifacts,original_training_torch=record['training_torch'],original_training_device='cuda')
m=dict(scope='HYBRID_ADOPTED_SCREEN91001_SINGLE_SOURCE_ONLY',status='PREPARED_NOT_EXECUTED_NOT_AUTHORIZED',exact_ids=[rid],canary_id=rid,previously_accepted_skip_ids=[],source_role='accepted_screen_reuse',selection_seed=True,phase='screen',private_inputs_pin=pin(H/'originals/SCREEN_INPUTS.json'),native_root_pin=q['native_root'],records=[row],path_map=mapping,server_repo='/workspace/GuardFed-celeba-expanded',server_python='/workspace/guardfed_envs/celeba-cu128-20261009/bin/python',server_namespace='/workspace/guardfed_checks/celeba_hybrid_screen91001_single_20261011',runtime_repo_hashes={r:h for r,h in record['source_hashes'].items() if r.startswith(('scripts/','src/','data/celeba/'))},cpu_affinity=None,threads=8,device='cpu',dtype='float32',views=['native','raw','shared_calibration'],native_tolerance=1e-12,new_three_view_accepted=0,new_formal_training=0,dispatch_authorized=False,full_client_ID_partition_recomputed=False)
for v in [inputs[k] for k in ['root','strict','offserver','members','record_checker','scope_pin','protocol_pin','body_pin','original_bridge','original_reuse']]:
 literal=v['path'].replace('\\','/');resolved=Path(v['path']).resolve().as_posix()
 if literal!=resolved:m['path_map'][literal]=copy.deepcopy(m['path_map'][resolved])
save(H/'MANIFEST.json',m)
repls=[("IDS = ['CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed%d_fullcoverage' % n for n in range(91003, 91011)]","IDS = ["+repr(rid)+"]"),('celeba_hybrid_three_view_missing8_20261011','celeba_hybrid_screen91001_single_20261011'),('guardfed_hybrid_missing8_valid','guardfed_hybrid_screen91001_valid'),('HYBRID_MISSING8','HYBRID_SCREEN91001'),('_hybrid_missing8','_hybrid_screen91001'),('HYBRID_ADOPTED_BENIGN_MISSING8_CANARY_FIRST_SOURCE_ONLY','HYBRID_ADOPTED_SCREEN91001_SINGLE_SOURCE_ONLY'),("m['previously_accepted_skip_ids'] == [IDS[0].replace('91003', '91002')]","m['previously_accepted_skip_ids'] == [] and m['phase'] == 'screen' and m['selection_seed'] is True"),('1434b40de5116bf3d53bc5a6ae2bd3b90f4e54222f23ad099ff72b1fd3ef1775',q['native_root']['sha256']),
("    require(m['existing_replay_root_pin']['sha256'] == '631d7ee2acf1cbe3453d849523e262f29ff5b354b5ddb56c60e5779e94364456', 'Prior replay root changed')\n",''),('91003 + IDS.index(row[\'id\'])','91001'),("p['FL_replay_runtime_verified'] is True","p['FL_replay_service_exited'] is True and p['previous_Hybrid8_service_exited'] is True and p['previous_replay_workers_absent'] is True"),('Exactly8 adopted Hybrid IID Benign checkpoints; first91003 canary then seven sequentially; not full100, test, ranking, final-primary or CUDA equivalence','Exactly1 adopted Hybrid screen91001 checkpoint; selection history retained, not new formal training, full100, test, ranking, final-primary or CUDA equivalence'),('Canary failed; remaining seven forbidden','Single native canary failed; stop without retry'),('Prepared exact8 Hybrid replay. First91003 must pass before the remaining seven.','Prepared single accepted Hybrid screen91001 replay. No new formal training.'),('Exact8 order required','Exact1 screen identity required'),('Native8 root changed','Original screen native root changed')]
s=(P/'candidate.py').read_text();old=s
for a,z in repls:assert a in s,a;s=s.replace(a,z)
(H/'candidate.py').write_text(s,encoding='utf8');compile(s,'candidate','exec');diff.extend(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile=str(P/'candidate.py'),tofile=str(H/'candidate.py')))
savedrepls=[('HYBRID_MISSING8','HYBRID_SCREEN91001'),('guardfed_hybrid_missing8_valid','guardfed_hybrid_screen91001_valid'),("'hybrid_three_view_missing8' in t","'hybrid_screen91001_single' in t"),('Exact19 saved members required','Exact5 saved members required'),("linux['cached_root_refits'] == 8","linux['cached_root_refits'] == 1"),('Whole exact8 incomplete','Whole exact1 incomplete'),('cached_root_refits=8 if linux_mode else 0, fit_calls=8 if linux_mode else 0','cached_root_refits=1 if linux_mode else 0, fit_calls=1 if linux_mode else 0'),('records=8, fit_calls=','records=1, fit_calls='),("pre['hybrid_replay_service_exited'] is True","pre['hybrid_replay_service_exited'] is True and pre['previous_Hybrid8_service_exited'] is True and pre['previous_replay_workers_absent'] is True")]
s=(B/'check_saved.py').read_text();old=s
for a,z in savedrepls:assert a in s,a;s=s.replace(a,z)
(H/'check_saved.py').write_text(s,encoding='utf8');compile(s,'check_saved','exec');diff.extend(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile=str(B/'check_saved.py'),tofile=str(H/'check_saved.py')))
spec=importlib.util.spec_from_file_location('_singleton_candidate_metadata',H/'candidate.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);c.validate_manifest(m)
for bad,cp in [(rid.replace('91001','91002'),None),(rid,'0'*64)]:
 try:b.identity_record(bad,checkpoint_sha256=cp)
 except ValueError:pass
 else:raise AssertionError('Missing singleton/checkpoint rejection')
for k in ['root','strict','offserver','members','record_checker','scope_pin','protocol_pin','body_pin','original_bridge','original_reuse']:assert Path(inputs[k]['path']).resolve().as_posix() in mapping,k
expected=funcs((P/'candidate.py').read_text())['run']
for a,z in repls:expected=expected.replace(a,z)
assert funcs((H/'candidate.py').read_text())['run']==expected
expected=(B/'check_saved.py').read_text()
for a,z in savedrepls:expected=expected.replace(a,z)
assert expected==(H/'check_saved.py').read_text()
science=[]
for item,filename in zip(read(B/'SOURCE_PINS.json')['scientific_functions'],['evaluator.py','replay.py']):
 s=(H/'originals'/filename).read_text();assert (H/'originals'/filename).read_bytes()==(B/'originals'/filename).read_bytes()
 for n in ast.parse(s).body:
  if isinstance(n,ast.FunctionDef) and n.name in item['functions']:
   assert hashlib.sha256(ast.get_source_segment(s,n).encode()).hexdigest()==item['functions'][n.name];science.append(dict(member='originals/'+filename,name=n.name,sha256=item['functions'][n.name]))
assert len(science)==17
for p in H.rglob('*.py'):compile(p.read_bytes(),str(p),'exec')
save(H/'SOURCE_CHECK.json',dict(status='SINGLE_SCREEN_METADATA_JOIN_COMPILE_SCOPE_REUSE_PASS_NO_SCIENCE',id=rid,checkpoint_sha256=record['checkpoint']['sha256'],config_canonical_sha256=record['config_canonical_sha256'],original_science17=science,per_record_replay_body_exact_after_scope_inverse=True,saved_whole_and_output_body_exact_after_scope_count_inverse=True,single_ID_and_wrongcheckpoint_refusal=True,original_screen_phase_retained=True,selection_seed=True,full_client_ID_partition_recomputed=False,new_acceptance=0,SSH=0,models_opened=0,arrays_opened=0,forward=0,fit=0))
save(H/'SOURCE_EDITS.json',dict(candidate_parent=pin(P/'candidate.py'),candidate_replacements=repls,saved_parent=pin(B/'check_saved.py'),saved_replacements=savedrepls,private_bridge_parent=pin(OLD/'bridge.py'),private_bridge_scope='identity_record binds original screen proof schema; science delegation unchanged',input_review=pin(sourcefile)))
(H/'SOURCE_DIFF.patch').write_text(''.join(diff),encoding='utf8')
files={p.relative_to(H).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(H.rglob('*')) if p.is_file()};save(H/'FILES_SHA256.json',dict(status='PREPARED_SINGLE_SCREEN_NOT_DISPATCH_AUTHORIZED',files=files))
save(H/'HANDOFF.json',dict(status='SOURCE_ONLY_SINGLE_SCREEN_REPLAY_ROOT_REVIEW_REQUIRED',package_sha256=sha(H/'FILES_SHA256.json'),manifest_sha256=sha(H/'MANIFEST.json'),source_check_sha256=sha(H/'SOURCE_CHECK.json'),id=rid,checkpoint_sha256=record['checkpoint']['sha256'],native_root_sha256=q['native_root']['sha256'],source_role='original_screen_reuse',selection_seed=True,threads=8,fixed_pool=list(range(64,96)),required_siblings=list(range(320,352)),existing_Hybrid8_and_FL_must_be_EXITED_and_noowner=True,new_observer_or_dispatch_scripts_created=False,actual_authorization=False,new_three_views_accepted=0,new_formal_training=0,full100=False,test=False,commands='README.md; actual root review and post-completion resource proof hashes required'))
(H/'README.md').write_text('单条已接受screen91001三视图源码；原ID/phase/选参历史/环境均保留，不是新增formal训练。原17函数、replay_one、whole saved、Windows zero-fit output body复用，仅改身份/数量/namespace。无模型数组读取或新接受。逐client image IDs未重算，不造partitionhash。\n\n资源晚绑定：64–95池、8threads、32physicalpairs/64logical/至少16quietpairs，现Hybrid8与FL必须正常EXITED且无owner，原source/data/checkpoint/current资源fresh gate通过；复用完成后的原helper由root绑定，没有新观察器。candidate preflight新增必需布尔FL_replay_service_exited、previous_Hybrid8_service_exited、previous_replay_workers_absent；原300秒期和资源门保留。Linux whole随后只CPU110，单条service也须EXITED。\n\nroot审查与实际资源绑定后唯一候选CLI：\n`python -B <deployed-source>/candidate.py --cpus 64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95 --package-sha256 <HANDOFF实际seal> --authorization <actual> --authorization-sha256 <actual> --preflight <actual> --preflight-sha256 <actual> --source-review <actual> --source-review-sha256 <actual> --output /workspace/guardfed_checks/celeba_hybrid_screen91001_single_20261011/outputs/attempt001`\n\n后续`check_saved.py --mode linux-whole --allow-original-cached-root-refit`及`--mode windows-saved-output --allow-saved-output-zero-fit`原CLI，需要实际gate、package、whole/transportSHA。最终5个saved members（gate/metadata/whole+本ID receipt/npz），不得提前生成proof。源码尚待root独审；部署/启动不在本次范围。\n',encoding='utf8')
print(json.dumps(dict(package=sha(H/'FILES_SHA256.json'),manifest=sha(H/'MANIFEST.json'),handoff=sha(H/'HANDOFF.json'),source_check=sha(H/'SOURCE_CHECK.json'),files=len(files))))
