from __future__ import annotations
import ast,copy,hashlib,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
METHOD="CosineFairnessHybrid"
RID='CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91001_screen'
INPUTS_SHA256='9e5a559d18e54726df3a1c98e8130a6c590542c06445447931477f2d339a48a0'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read_pin(pin, *, decode=True):
    """Only small source/JSON bytes: never open models, archives or arrays."""
    path = Path(pin['path'])
    require(path.suffix in {'.json', '.py'} and path.stat().st_size <= 2_000_000,
            'Not a compact metadata/source input: ' + str(path))
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == pin['sha256'], 'Pinned bytes changed: ' + str(path))
    require(len(raw) == pin['bytes'], 'Pinned size changed')
    return json.loads(raw) if decode else raw

def inputs():
    raw = (HERE / 'SCREEN_INPUTS.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == INPUTS_SHA256, 'Private registry bytes changed')
    return json.loads(raw)

def original_bridge():
    pin = inputs()['original_bridge']
    read_pin(pin, decode=False)
    spec = importlib.util.spec_from_file_location('_hybrid_private_original_bridge', pin['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def science_bindings(*, torch_module=None, pandas_module=None):
    """Original loader/code objects and frozen recipe; constructs, never calls science."""
    return original_bridge().science_bindings(torch_module=torch_module, pandas_module=pandas_module)

def one(records, rid):
    matches = [r for r in records if r['id'] == rid]
    require(len(matches) == 1, 'Missing/duplicate proof ID')
    return matches[0]

def identity_record(rid, *, checkpoint_sha256=None):
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
