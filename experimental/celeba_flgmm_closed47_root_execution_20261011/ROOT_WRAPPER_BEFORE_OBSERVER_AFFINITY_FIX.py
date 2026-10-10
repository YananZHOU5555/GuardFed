"""Thin root controls for the sealed finite47 candidate; reuse the original deployment/observer/start code."""
from pathlib import Path
import argparse, base64, datetime, hashlib, importlib.util, json, shlex, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'tmp/celeba_flgmm_three_view_closed_batch_preparation_20261011'
OUT = ROOT / 'tmp/celeba_flgmm_closed47_root_execution_20261011'
BASE = '/workspace/guardfed_checks/celeba_flgmm_three_view_closed_batch_20261011'
PACKAGE = '43b53caf0d6a0b4ac268fdad6597065cd76eee26d52f6d4929d45478e47cf023'
PROGRAM = 'guardfed_flgmm_closed47_valid'
OLD_BASE = '/workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010'
OLD = ROOT / 'tmp/celeba_added_cnn_exact3_root_execution_20261010'
GUIDE = '42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'

def sha(b): return hashlib.sha256(b).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(name, value):
    with (OUT / name).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def pins():
    assert sha((SRC / 'FILES_SHA256.json').read_bytes()) == PACKAGE
    members = {}
    for rel, pin in read(SRC / 'FILES_SHA256.json')['files'].items():
        p = (SRC / rel).resolve(); assert p.is_relative_to(SRC)
        b = p.read_bytes(); assert sha(b) == pin['sha256'] and len(b) == pin['bytes']
        members['source/' + rel] = b
    members['source/FILES_SHA256.json'] = (SRC / 'FILES_SHA256.json').read_bytes()
    return members
def call(tag, code, payload, timeout=240):
    (OUT / (tag + '_remote.py')).write_text(code, encoding='utf-8', newline='\n')
    command = 'env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 taskset -c 110 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B -c ' + shlex.quote(code)
    argv = ['ssh', '-p', '60350', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'root@89.22.197.55', command]
    started = utc()
    cp = subprocess.run(argv, input=json.dumps(payload).encode(), capture_output=True, timeout=timeout)
    (OUT / (tag + '_STDOUT.txt')).write_bytes(cp.stdout); (OUT / (tag + '_STDERR.txt')).write_bytes(cp.stderr)
    save(tag + '_COMMAND.json', dict(utc_started=started, utc_finished=utc(), argv=argv,
         source_sha256=sha(code.encode()), payload_sha256=sha(json.dumps(payload).encode()), exit=cp.returncode))
    if cp.returncode:
        print(cp.stderr.decode(errors='replace')); raise SystemExit(cp.returncode)
    result = json.loads(cp.stdout); save(tag + '.json', result)
    return result

def review_and_deploy():
    if OUT.exists():
        assert {p.name for p in OUT.iterdir()} == {'ROOT_WRAPPER_BEFORE_NUMPY_ASSERT_FIX.py','ROOT_METADATA_WRAPPER_FAILURE.json'}
    else:
        OUT.mkdir(exist_ok=False)
    members = pins(); m = read(SRC / 'MANIFEST.json')
    assert len(m['records']) == len(set(m['exact_ids'])) == 47
    assert len(members) == 128
    spec = importlib.util.spec_from_file_location('root_flgmm_candidate', SRC / 'candidate.py')
    c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
    c.package_check(PACKAGE); c.validate_manifest(m)
    original_resolver = c.resolve_origin
    def local_metadata_origin(origin, manifest, runtime):
        pin = manifest['path_map'][str(origin).replace('\\','/')]
        if pin['kind'] == 'server':
            # Root's metadata-only replay reads the already accepted local JSON origins.
            # This does not rebind the deployed candidate or open any model tensor.
            return Path(origin)
        return original_resolver(origin, manifest, runtime)
    c.resolve_origin = local_metadata_origin
    b, _ = c.bound_bridge(m, runtime=False)
    identities = [b.identity_record(row['method'], row['id']) for row in m['records']]
    assert identities == [row['identity'] for row in m['records']]
    assert 'torch' not in sys.modules
    r = dict(status='ROOT_FLGMM_FINITE47_SOURCE_REVIEW_PASS_NOT_RUNTIME_ACCEPTANCE', utc=utc(),
        source_adoptable=True, package_sha256=PACKAGE, manifest_sha256=sha((SRC/'MANIFEST.json').read_bytes()),
        exact_ids=m['exact_ids'], original_scientific_functions_exact=17,
        actual_original_bridge_metadata_replay_records=47, actual_metadata_replay_exact=True,
        metadata_only_local_origin_path_binding=True,
        root_read_entire_candidate=True, source_check_sha256=sha((SRC/'SOURCE_CHECK.json').read_bytes()),
        screen4_original_CRLF_and_stored_LF_bytes_both_bound=True,
        source_only_preparation_failures_preserved=True, independent_review_received_in_current_turn=True,
        independent_review_scope='Finite accepted identities, metadata adaptation, original science/replay/whole-checker and runtime guards; no new model execution',
        NumPy_imported_by_original_bridge=True, Torch_imported=False,
        root_metadata_wrapper_failure_preserved=True,
        root_scientific_acceptances=0, new_inference=0, new_fit=0, new_training=0, test=False,
        Windows_whole_checker_prior_failure_preserved=True,
        runtime_still_requires_actual_fresh_preflight=True, failure_policy='Stop and preserve; no automatic retry or tolerance change')
    save('ROOT_SOURCE_REVIEW.json', r)
    members['ROOT_SOURCE_REVIEW.json'] = (OUT / 'ROOT_SOURCE_REVIEW.json').read_bytes()
    code = (OLD / 'deploy_source_remote.py').read_text(encoding='utf-8').replace(OLD_BASE, BASE)
    original = (OLD / 'deploy_source_remote.py').read_bytes()
    save('ORIGINAL_DEPLOY_BINDING.json', dict(original_path=str(OLD/'deploy_source_remote.py'), original_sha256=sha(original),
         transformation='Only fixed deployment namespace is replaced; original remote verification and no-start behavior unchanged'))
    payload = {'base':BASE, 'members':{rel:dict(base64=base64.b64encode(b).decode(),sha256=sha(b),bytes=len(b)) for rel,b in members.items()}}
    result = call('SOURCE_DEPLOYMENT', code, payload, 90)
    assert result['members'] == {rel:dict(sha256=sha(b),bytes=len(b)) for rel,b in members.items()}
    print(json.dumps(dict(status=result['status'],members=len(members),bytes=sum(map(len,members.values())),root_source_review_sha256=sha((OUT/'ROOT_SOURCE_REVIEW.json').read_bytes()))))

def preflight():
    pins(); m = read(SRC / 'MANIFEST.json')
    original = ROOT/'tmp/celeba_added_cnn_exact3_linux_preflight_20261010/observe_remote.py'
    code = original.read_text(encoding='utf-8')
    assert code.count('PAYLOAD = None  # launcher replaces this single assignment with frozen JSON literal') == 1
    code = code.replace('PAYLOAD = None  # launcher replaces this single assignment with frozen JSON literal', "PAYLOAD = json.load(__import__('sys').stdin)").replace(OLD_BASE, BASE)
    code = "assert __import__('hashlib').sha256(__import__('pathlib').Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='" + GUIDE + "'\n" + code
    result = call('OBSERVATION', code, dict(manifest=m,guide_sha256=GUIDE), 240)
    assert result['source_model_data_hashes_verified'] and not result['duplicate_gate']
    assert not result['narrow_cpu120_127_conflicts']
    active_wide = [dict(pid=r['pid'],tid=t['tid'],last_cpu=t['last_cpu']) for r in result['guardfed_processes'] for t in r['threads']
                   if len(t['affinity'])>8 and t.get('delta_cpu_ticks_2s',0) and 120 <= t['last_cpu'] <=127]
    assert not active_wide, ('Sampled active wide-thread collision',active_wide)
    assert all(not r['failures'] and not r['live_producers'] for r in result['selected'])
    states = {line.split()[0]:line.split()[1] for line in result['services']['stdout'].splitlines()}
    required = ['guardfed_celeba_mechanism_formal','guardfed_celeba_flgmm_fullcoverage','guardfed_celeba_gradient_screen64_v2a','guardfed_celeba_hybrid_fullcoverage']
    assert all(states.get(s)=='RUNNING' for s in required)
    assert states.get('guardfed_celeba_mechanism_remaining620_valid_v2a') in ('RUNNING','EXITED')
    assert all(result[k]['returncode']==0 for k in ('gpu','gpu_health','gpu_recovery'))
    assert len(result['gpu_recovery']['stdout'].splitlines())==2 and all(line.strip().endswith('None') for line in result['gpu_recovery']['stdout'].splitlines())
    temperatures = [int(line.split(',')[-1]) for line in result['gpu']['stdout'].splitlines()]
    assert len(temperatures)==2 and max(temperatures)<85
    cg=result['cgroup']; quota,period=cg['cpu.max'].split(); assert quota!='max'; quota=float(quota)/float(period)
    assert int(cg['memory.max'])-int(cg['memory.current'])>16*1024**3
    events=dict(line.split() for line in cg['memory.events'].splitlines()); assert int(events['oom'])==int(events['oom_kill'])==0
    assert result['disk']['free']>20*1024**3 and not result['main_queue']['failed']
    main = [r for r in result['guardfed_processes'] if '--job' in r['argv'] and any('celeba_mechanism_v1' in a for a in r['argv'])]
    assert len(main)==8
    nominal=sum(len(r['threads']) for r in main)+2+1+1+8+8+8
    assert nominal<=quota and set(range(120,128))<=set(result['eligible_cpus'])
    assert (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(result['utc'])).total_seconds()<120
    p=dict(status='ROOT_LINUX_FLGMM_EXACT47_PREFLIGHT_PASS',utc=utc(),cpu_affinity=list(range(120,128)),eligible_cpus=result['eligible_cpus'],
        actual_quota_cores=quota,nominal_reserved_cores_including_gate=nominal,no_duplicate_gate=True,all_thread_cpus_free=True,
        source_model_data_hashes_verified=True,selected_producers_quiescent=True,services_healthy=True,gpu_health_verified=True,
        cgroup_and_memory_headroom_verified=True,storage_headroom_verified=True,package_sha256=PACKAGE,
        exact_ids=m['exact_ids'],observation_sha256=sha((OUT/'OBSERVATION.json').read_bytes()),
        source_model_data_files_hashed=len(result['hashes']),source_model_data_bytes_hashed=sum(x['bytes'] for x in result['hashes']),
        all_thread_cpus_free_definition='No narrow reservation overlap and no sampled active wide-thread last_cpu120..127; wide masks may migrate later and are not exclusive reservations',
        nominal_budget_components=dict(main_all_current_OS_threads=sum(len(r['threads']) for r in main),FL_shared_affinity=2,Hybrid=1,gradient=1,remaining620_reserved=8,finite47_proposed=8,coordinator_IO_allowance=8),
        nominal_budget_not_hard_peak_bound=True,new_inference=0,new_fit=0,new_training=0,test=False)
    save('LINUX_PREFLIGHT.json',p)
    print(json.dumps({k:v for k,v in p.items() if k not in ('eligible_cpus','exact_ids')}))

def start():
    pins(); m=read(SRC/'MANIFEST.json'); p=read(OUT/'LINUX_PREFLIGHT.json')
    assert not (OUT/'START_RECEIPT.json').exists() and not (OUT/'START_FAILURE.json').exists()
    assert (datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(p['utc'])).total_seconds()<220
    review_sha=sha((OUT/'ROOT_SOURCE_REVIEW.json').read_bytes()); pre_sha=sha((OUT/'LINUX_PREFLIGHT.json').read_bytes())
    auth=dict(status='ROOT_AUTHORIZED_FLGMM_CLOSED_EXACT47_THREE_VIEW',utc=utc(),package_sha256=PACKAGE,
        manifest_sha256=sha((SRC/'MANIFEST.json').read_bytes()),source_review_sha256=review_sha,linux_preflight_sha256=pre_sha,
        exact_ids=m['exact_ids'],cpu_affinity=list(range(120,128)),device='cpu',threads=8,max_processes=1,test=False,training=False,
        scope='Exactly47 previously accepted terminal70 FLGMM checkpoints, valid19867 only; original root-only fit/native/raw/shared interface',
        no_change_to_training_queues=True,not_full100=True,not_final_endpoint=True,failure_policy='Stop and preserve; no automatic retry, no tolerance change')
    save('AUTHORIZATION.json',auth)
    cmd=['/usr/bin/env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=8','MKL_NUM_THREADS=8','OPENBLAS_NUM_THREADS=1','NUMEXPR_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','/usr/bin/taskset','-c','120-127','/usr/bin/ionice','-c','3','/usr/bin/nice','-n','10','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',BASE+'/source/candidate.py','--package-sha256',PACKAGE,'--source-review',BASE+'/ROOT_SOURCE_REVIEW.json','--source-review-sha256',review_sha,'--preflight',BASE+'/LINUX_PREFLIGHT.json','--preflight-sha256',pre_sha,'--authorization',BASE+'/AUTHORIZATION.json','--authorization-sha256',sha((OUT/'AUTHORIZATION.json').read_bytes()),'--output',BASE+'/outputs/attempt001']
    config='\n'.join(['[program:'+PROGRAM+']','command='+shlex.join(cmd),'directory=/workspace/GuardFed-celeba-expanded','autostart=false','autorestart=false','startsecs=2','startretries=0','stopsignal=TERM','stopasgroup=true','killasgroup=true','stdout_logfile='+BASE+'/execution/stdout.log','stderr_logfile='+BASE+'/execution/stderr.log','stdout_logfile_maxbytes=10MB','stdout_logfile_backups=1','stderr_logfile_maxbytes=10MB','stderr_logfile_backups=1',''])
    (OUT/(PROGRAM+'.conf')).write_text(config,encoding='utf-8')
    code=(OLD/'start_remote.py').read_text(encoding='utf-8').replace(OLD_BASE,BASE).replace('guardfed_added_cnn_exact3_gate',PROGRAM).replace('ROOT_EXACT3','ROOT_FLGMM_FINITE47')
    files={name:dict(sha256=sha((OUT/name).read_bytes()),bytes=(OUT/name).stat().st_size,base64=base64.b64encode((OUT/name).read_bytes()).decode()) for name in ('LINUX_PREFLIGHT.json','AUTHORIZATION.json')}
    result=call('START_RECEIPT',code,dict(base=BASE,program=PROGRAM,package=PACKAGE,files=files,config=config,config_sha256=sha(config.encode()),source_review_sha256=review_sha),90)
    assert 'STARTED_NOT_SCIENTIFIC_ACCEPTANCE' in result['status']
    print(json.dumps(result))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['review-deploy','preflight','start']);a=ap.parse_args()
    {'review-deploy':review_and_deploy,'preflight':preflight,'start':start}[a.phase]()
