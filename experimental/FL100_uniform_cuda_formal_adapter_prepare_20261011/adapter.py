"""FL100 formal CUDA per-record binding. Import is stdlib only; no CLI/dispatch."""
from pathlib import Path, PurePosixPath
import ast
import copy
import hashlib
import inspect
import json
import os
import sys
import time
import types

HERE = Path(__file__).resolve().parent
SCOPE = 'UNIFORM_CUDA_FL100_FORMAL_V1'
AUTHOR_SHA = 'a1f1be152a0aa5668b91d73d7297ea3e9275efaac8d82c3b22cfbe97b7b08fce'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def pinned(pin):
    need(isinstance(pin, dict) and set(pin) == {'path', 'sha256'}, 'Missing exact Linux path/SHA')
    p = Path(pin['path'])
    need(p.is_absolute() and '..' not in p.parts and ':' not in str(p) and p.resolve() == p
         and p.is_file() and not p.is_symlink() and sha(p) == pin['sha256'], 'Unbound or changed Linux dependency')
    return p


def metadata_gate(approval, cohort, row, now):
    need(approval.get('status') == 'ROOT_AUTHORIZED_UNIFORM_CUDA_FL100_FORMAL_V1'
         and approval.get('execution_authorized') is True, 'No actual formal root approval')
    need(cohort['status'] == 'FIXED_FL100_ACCEPTED_NATIVE_REFERENCES_NOT_CUDA_RESULTS'
         and len(cohort['records']) == 100 and len({r['id'] for r in cohort['records']}) == 100
         and [r['native_index'] for r in cohort['records']] == list(range(1000, 1100)), 'Exact FL100 scope required')
    need(row in cohort['records'] and row['family'] == 'FL100' and approval.get('id') == row['id']
         and approval.get('native_index') == row['native_index']
         and approval.get('identity_canonical_sha256') == row['identity_canonical_sha256'], 'Wrong accepted FL identity')
    need(approval.get('ordered_input_ids') == [r['id'] for r in cohort['records']], 'Root must freeze full ordered100')
    need(approval.get('decision') == 'B' and approval.get('author_decision_sha256') == AUTHOR_SHA, 'Wrong author decision')
    need(approval.get('native_tolerance') == 1e-12 and approval.get('target_split') == 'valid'
         and approval.get('compute_threads') == approval.get('max_processes') == 1, 'Science/budget changed')
    need(approval.get('views') == ['native', 'raw', 'shared_calibration'], 'Incomplete or changed views')
    for k in ['training', 'test', 'automatic_retry', 'diagnostic_reuse', 'old_CPU_reuse_as_CUDA']:
        need(approval.get(k) is False, 'Forbidden operation: ' + k)
    need(approval.get('device') == 'cuda:0' and isinstance(approval.get('gpu_uuid'), str)
         and approval['gpu_uuid'].startswith('GPU-'), 'Actual GPU required')
    cpus = approval.get('allowed_cpus')
    need(isinstance(cpus, list) and cpus and all(type(c) is int and c >= 0 for c in cpus)
         and len(set(cpus)) == len(cpus), 'Actual CPU binding required')
    ns = approval.get('output_namespace')
    need(isinstance(ns, str) and ns.startswith('/workspace/guardfed_checks/uniform_cuda_FL100_')
         and '..' not in PurePosixPath(ns).parts and approval.get('output') == str(PurePosixPath(ns) / 'runs' / row['id']), 'New FL formal namespace required')
    need(approval.get('receipt_version') == SCOPE and approval.get('old_evidence_preserved') is True
         and type(approval.get('deadline_unix')) in (int, float) and now < approval['deadline_unix'], 'Expired/incomplete approval')


def bind_fl(candidate, args, exactcohortpin, actualapprovalpin, resource_gate):
    """Root-only future API, matching old800's explicit bind/replay/accept contract.

    args.repo is the original server repo; args.torch_module/pandas_module are
    the root-bootstrapped real modules. No scientific import occurs at module import.
    """
    need(sys.platform == 'linux' and not sys.flags.optimize, 'Original strict Linux runtime only')
    approval_path = pinned(actualapprovalpin)
    approval = json.loads(approval_path.read_bytes())
    cohort_path = pinned(exactcohortpin)
    need(approval.get('exact_cohort') == exactcohortpin and sha(cohort_path) == sha(HERE / 'EXACT100.json'), 'Changed FL100 scope')
    cohort = json.loads(cohort_path.read_bytes())
    row = next((r for r in cohort['records'] if r['id'] == approval.get('id')), None)
    need(row is not None, 'Foreign FL checkpoint')
    metadata_gate(approval, cohort, row, time.time())
    pins = json.loads((HERE / 'SOURCE_PINS.json').read_bytes())
    need(sha(__file__) == approval.get('adapter_sha256'), 'Changed formal adapter')
    seal = json.loads(pinned(approval['source_seal']).read_bytes())
    for name, item in seal['files'].items():
        need(Path(name).name == name and sha(HERE / name) == item['sha256'], 'Formal source seal changed')
    author_path = pinned(approval['author_decision'])
    need(sha(author_path) == AUTHOR_SHA, 'Wrong author approval bytes')
    author = json.loads(author_path.read_bytes())
    need(author['approved']['uniform_CUDA_evaluation'] and author['approved']['corresponding_Full_included']
         and not author['existing_diagnostic_retroactively_adopted'], 'Author scope changed')
    package = pins['packages'][row['identity_source']]
    deps = approval.get('runtime_dependencies')
    need(isinstance(deps, dict) and set(deps) == set(package), 'All actual Linux dependency roles required')
    paths = {name: pinned(pin) for name, pin in deps.items()}
    for name, pin in package.items():
        need(deps[name]['sha256'] == pin['sha256'], 'Wrong frozen runtime source: ' + name)
    need(Path(inspect.getsourcefile(candidate)).resolve() == paths['candidate'], 'Wrong real candidate module')
    need(Path(candidate.HERE).resolve() == paths['candidate'].parent
         and paths['manifest'] == Path(candidate.HERE) / 'MANIFEST.json'
         and paths['package_seal'] == Path(candidate.HERE) / 'FILES_SHA256.json', 'Mixed FL package')
    candidate.package_check(deps['package_seal']['sha256'])
    manifest = json.loads(paths['manifest'].read_bytes())
    candidate.validate_manifest(manifest)
    original_row = manifest['records'][row['identity_index']]
    need(original_row['id'] == row['id'] and original_row['method'] == 'FLGMM'
         and canonical(original_row['identity']) == row['identity_canonical_sha256'], 'FL identity object changed')
    repo = Path(args.repo).resolve()
    need(str(repo) == manifest['server_repo'], 'Original FL repository differs')
    need(paths['core'] == repo / 'scripts/reproduce_paper_tables.py'
         and paths['cnn'] == repo / 'src/celeba_data.py', 'Wrong actual model/core source')
    for role, name in [('replay', 'replay.py'), ('evaluator', 'evaluator.py'), ('bridge', 'bridge.py'),
                       ('source_reuse', 'SOURCE_REUSE.json')]:
        need(paths[role] == Path(candidate.HERE) / 'originals' / name, 'Mixed original source: ' + role)
    registry_relative = Path(package['proof_registry']['path']).relative_to(Path(package['candidate']['path']).parent)
    need(paths['proof_registry'] == Path(candidate.HERE) / registry_relative, 'Wrong per-package proof registry')
    preflight_path = pinned(approval['Linux_preflight'])
    preflight = json.loads(preflight_path.read_bytes())
    need(preflight.get('pass') is True and 0 <= time.time() - preflight.get('observed_unix', 0) <= 90
         and preflight.get('input_scope_sha256') == sha(cohort_path)
         and preflight.get('adapter_sha256') == sha(__file__)
         and preflight.get('gpu_uuid') == approval['gpu_uuid']
         and preflight.get('allowed_cpus') == approval['allowed_cpus'], 'Fresh root resource gate missing')
    gate_path = pinned(approval['resource_gate_source'])
    need(callable(resource_gate) and Path(inspect.getsourcefile(resource_gate)).resolve() == gate_path, 'Unreviewed live resource gate')
    torch, pandas = args.torch_module, args.pandas_module
    need(sys.modules.get('torch') is torch and sys.modules.get('pandas') is pandas, 'Actual bootstrapped modules required')

    def fresh(snapshot=None):
        need(torch.__version__ == '2.11.0+cu128' and torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0
             and os.environ.get('CUDA_VISIBLE_DEVICES') == approval['gpu_uuid'], 'Actual CUDA execution identity changed')
        need(torch.get_num_threads() == torch.get_num_interop_threads() == 1
             and set(os.sched_getaffinity(0)) == set(approval['allowed_cpus']), 'Thread/CPU budget changed')
        need(torch.are_deterministic_algorithms_enabled() and torch.backends.cudnn.deterministic
             and not torch.backends.cudnn.benchmark and not torch.backends.cudnn.allow_tf32
             and not torch.backends.cuda.matmul.allow_tf32 and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'FP32 deterministic contract changed')
        need(time.time() < approval['deadline_unix'], 'Root authorization expired')
        resource_gate(snapshot, approval)

    fresh()
    # Same original_fl_identity sequence; each of the seven candidates owns its registry.
    bridge, evaluator = candidate.bound_bridge(manifest, runtime=True, torch_module=torch, pandas_module=pandas)
    identity = bridge.identity_record('FLGMM', row['id'])
    need(identity == original_row['identity'], 'Original strict/offserver/artifact identity changed')
    record = copy.deepcopy(identity)
    record.update(distribution=original_row['distribution'], attack=original_row['attack'], seed=original_row['seed'],
                  result=identity['original_artifact_pins']['result'], raw_job=identity['original_artifact_pins']['job'],
                  config_canonical_sha256=canonical(identity['config']), training_torch=original_row['original_training_torch'])
    import datetime, platform, resource, signal, subprocess, zipfile
    ns = dict(evaluator.rebuild_root.__globals__, datetime=datetime, platform=platform, resource=resource,
              signal=signal, subprocess=subprocess, time=time, zipfile=zipfile, os=os, sys=sys)
    replay_text = paths['replay'].read_text(encoding='utf-8')
    helpers = {'read', 'save', 'canonical', 'read_prefix', 'metadata', 'live_snapshot'}
    nodes = [n for n in ast.parse(replay_text).body if isinstance(n, ast.FunctionDef) and n.name in helpers]
    need(len(nodes) == len(helpers), 'Original runtime helper source changed')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(paths['replay']), 'exec'), ns)
    ns.update(TRAIN_IDS_SHA='46d42484d5b5f53af8747fcf44ee11a0b051af27f10383d32254faee0311bc99',
              VALID_IDS_SHA='64a15cf28caf1d177ac3dcf96a4408bc21091796974b243923ca947a37b554bf', VIEWS=evaluator.VIEWS)
    core = candidate.load('_formal_fl_core', paths['core'])
    cnn = candidate.load('_formal_fl_cnn', paths['cnn'])
    artifacts = {Path(p['server_path']): p['sha256'] for p in original_row['runtime_artifacts'].values()}
    artifact_paths = {k: Path(original_row['runtime_artifacts'][j]['server_path']) for k, j in [('checkpoint','model'), ('result','result'), ('raw_job','job')]}
    source_paths = {p: deps[k]['sha256'] for k,p in paths.items()}
    source_paths.update({repo / p:h for p,h in manifest['runtime_repo_hashes'].items()})
    source_paths.update({HERE / n:p['sha256'] for n,p in seal['files'].items()})
    source_paths.update({approval_path:actualapprovalpin['sha256'], cohort_path:exactcohortpin['sha256'],
                         author_path:AUTHOR_SHA, preflight_path:sha(preflight_path), gate_path:sha(gate_path)})
    package_seal = json.loads(paths['package_seal'].read_bytes())
    source_paths.update({Path(candidate.HERE)/n:p['sha256'] for n,p in package_seal['files'].items()})

    def measure(mapping):
        out = {}
        for p,h in mapping.items():
            need(sha(p) == h, 'Frozen input changed: ' + str(p))
            out[str(p)] = {'sha256':h, 'bytes':p.stat().st_size}
        return out

    def validate(_original, rec, _repo):
        need(rec == record and Path(_repo).resolve() == repo, 'Wrong bound FL record')
        need(bridge.identity_record('FLGMM', row['id']) == identity, 'Original FL identity changed')
        return candidate.read(artifact_paths['result'])

    before_source, before_artifacts = measure(source_paths), measure(artifacts)
    body_path = pinned(approval['formal_GPU_body'])
    need(body_path == HERE / 'gpu_body.py' and sha(body_path) == pins['derived']['gpu_body_sha256'], 'Wrong formal FL GPU body')
    ns.update(validate_external=lambda r:validate(None,r,repo), runtime_checkpoint=lambda r:artifact_paths['checkpoint'], resource_gate=fresh)
    exec(compile(body_path.read_bytes(), str(body_path), 'exec'), ns)
    inference = ns['replay_one']
    output = Path(approval['output']);failure = output.with_name(output.name + '.formal_failure.json')
    need(output.resolve() == output and not output.resolve().is_relative_to(repo), 'Output escapes formal scope')

    def preserve_failure(exc):
        need(not failure.exists(), 'Preserve first failure')
        failure.parent.mkdir(parents=True, exist_ok=True)
        with failure.open('x', encoding='utf8') as f:
            json.dump(dict(scope=SCOPE,id=record['id'],error_type=type(exc).__name__,error=str(exc),automatic_retry=False),f)

    def replay_one(wall_seconds=1800):
        need(not output.exists() and not failure.exists(), 'Preserve prior output/failure; single attempt')
        need(type(wall_seconds) is int and 1 <= wall_seconds <= 3600
             and time.time()+wall_seconds <= approval['deadline_unix'], 'Bounded original wall time required')
        def timed_out(signum, frame):
            raise TimeoutError('Fixed original per-record wall limit; preserve failure and stop')
        prior_handler = signal.signal(signal.SIGALRM, timed_out)
        try:
            fresh();need(measure(source_paths)==before_source and measure(artifacts)==before_artifacts,'Inputs drifted')
            ids,y,s,_ = ns['metadata'](repo)
            receipt = inference(core,None,cnn,evaluator,record,repo,ids,y,s,output,wall_seconds)
            need(measure(source_paths)==before_source and measure(artifacts)==before_artifacts,'Inputs changed')
            ns['save'](output/'formal_binding.json',dict(scope=SCOPE,id=record['id'],approval_sha256=actualapprovalpin['sha256'],
                input_scope_sha256=exactcohortpin['sha256'],receipt_sha256=sha(output/'receipt.json'),
                source_before=before_source,source_after=measure(source_paths),artifact_before=before_artifacts,
                artifact_after=measure(artifacts),cohort_complete=False))
            return receipt
        except BaseException as exc:
            preserve_failure(exc);raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prior_handler)

    def accept_saved_predictions():
        need(not failure.exists(), 'Failed formal record cannot be accepted')
        try:
            fresh()
            binding=ns['read'](output/'formal_binding.json');r=ns['read'](output/'receipt.json')
            need(binding['scope']==SCOPE and binding['id']==r['id']==record['id']
                 and binding['approval_sha256']==actualapprovalpin['sha256']
                 and binding['input_scope_sha256']==exactcohortpin['sha256']
                 and binding['receipt_sha256']==sha(output/'receipt.json'), 'Mixed formal receipt')
            need(r['scope']=='FORMAL_FL100_CUDA_PER_RECORD_V1' and r['status']=='NATIVE_VALID_REPLAY_PASS'
                 and r['runtime']['device']=='cuda:0' and r['runtime']['cuda_device_count']==1
                 and not r['optimizer_created'] and not r['gradients_created']
                 and not r['test_labels_accessed'] and not r['test_inference_performed'], 'Wrong CUDA operation')
            need(r['external_identity_record_sha256']==ns['canonical'](record)
                 and all(r[k]==record[a]['sha256'] for k,a in [('checkpoint_sha256','checkpoint'),('original_result_sha256','result'),('original_job_sha256','raw_job')]), 'Mixed terminal identity')
            need(binding['source_before']==binding['source_after']==measure(source_paths)==before_source
                 and binding['artifact_before']==binding['artifact_after']==measure(artifacts)==before_artifacts, 'Inputs changed')
            ids,y,s,_=ns['metadata'](repo)
            # Original Linux whole check_saved and its v2/context shape remain exact.
            v2=types.SimpleNamespace(np=ns['np'],require=need,rebuild_root=evaluator.rebuild_root,digest=sha,
                canonical=ns['canonical'],VIEWS=evaluator.VIEWS,check_native=evaluator.check_native)
            context=dict(v2=v2,mapping_paths=lambda rec,bind:artifact_paths,
                full_hashes=lambda mapping:measure(mapping),mapped_functions=lambda rec,p,orig:(validate,None),
                KINDS=('checkpoint','result','raw_job'))
            exec(compile(paths['saved_science'].read_bytes(),str(paths['saved_science']),'exec'),context)
            proof={'artifact_before':measure({p:record[k]['sha256'] for k,p in artifact_paths.items()}),
                   'artifact_after':measure({p:record[k]['sha256'] for k,p in artifact_paths.items()})}
            comparison=context['check_saved'](record,output,r,proof,types.SimpleNamespace(repo=repo),None,core,None,evaluator,ids,y,s)
            need(measure(source_paths)==before_source and measure(artifacts)==before_artifacts,'Acceptance inputs changed')
            return dict(scope=SCOPE,id=record['id'],status='PER_RECORD_FL100_CUDA_SAVED_WHOLE_ACCEPTED',
                        views=r['views'],native_comparison=comparison,cohort_complete=False,new_training=False,test=False,
                        binding_sha256=sha(output/'formal_binding.json'),original_saved_science_sha256=deps['saved_science']['sha256'],
                        Windows_original_refit_pass=False,legacy_CPU_reused=False)
        except BaseException as exc:
            preserve_failure(exc);raise

    return dict(record=record,validate_original=validate,replay_one=replay_one,accept_saved_predictions=accept_saved_predictions)
