"""Root-approved one launcher-only recovery; original writer9 and approval stay sealed."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
APPROVED_DIR = HERE.parent
STAGE = APPROVED_DIR.parent
REPAIR = STAGE / 'execution_repair_v1'

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def save(path, value):
    assert not path.exists()
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())

assert digest(REPAIR / 'FILES_SHA256.json') == '54c67847b44a26a654b380c6c4d03863279261126a61de29489bfa78572e332b'
assert digest(APPROVED_DIR / 'APPROVED.json') == '59b0c3b4298050ee55d9dd2b0d9e9e28c4b3b90390cd3e0c3b79c4a73c6d3bd1'
assert not (REPAIR / 'runtime_overlay').exists() and not (HERE / 'RECOVERY_AUTHORIZED.json').exists()
failure = APPROVED_DIR / 'launcher_failure_v1'
seal = read(failure / 'FILES_SHA256.json')
assert all(digest(failure / name) == sha for name, sha in seal['files'].items())
proof = read(failure / 'failure.json')
assert proof['status'] == 'TERMINAL_LAUNCHER_FAILURE_BEFORE_PYTHON' and proof['new_prediction_or_model_files'] == 0
assert all(digest(Path(name)) == sha for name, sha in proof['source_approval_and_launcher_sha_before_after_identical'].items())
sys.path.insert(0, str(REPAIR))
import repair_execute
gate, scope, original = repair_execute.inspect(); gate.verify_scope(original)
for row in gate.read(REPAIR / 'reused_IID_references.json')['records']:
    out = STAGE / row['output']; assert digest(out / 'acceptance.json') == row['acceptance_sha256']
    assert all(digest(out / name) == sha for name, sha in row['artifact_hashes'].items())
preflight = APPROVED_DIR / 'dispatch_preflight.py'
assert digest(preflight) == read(APPROVED_DIR / 'APPROVED.json')['launch_attachment_hashes']['dispatch_preflight.py']
spec = importlib.util.spec_from_file_location('hybrid_launch_v2_resource_preflight', preflight)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
resources = module.resources()
fl_stage = STAGE.parent / 'celeba_flgmm_screen_20261009/release_v2'
fl_script = fl_stage / 'run_one.py'
fl_seal = read(fl_stage / 'PACKAGE_SHA256.json')
assert fl_seal['status'] == 'FROZEN_PENDING_EXECUTION'
assert digest(fl_script) == fl_seal['files']['run_one.py'] == 'e55fbfc78c82617e8226e700597f9b76eb48ab18006fd9fd11ce8be5433a48c9'
fl_manifest = read(fl_stage / 'jobs/manifest.json')
fl_workers = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name) == os.getpid():
        continue
    try:
        argv = [x.decode() for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
        if len(argv) != 7 or argv[1:3] != ['-u', str(fl_script)]:
            continue
        assert argv[3:6] == ['--repo', str(gate.REPO), '--job-id'] and 'python' in Path(argv[0]).name
        assert str((proc / 'cwd').resolve()) == str(fl_stage)
        item = next(row for row in fl_manifest['jobs'] if row['id'] == argv[6])
        job_path = fl_stage / 'jobs' / item['job']; job = read(job_path)
        assert digest(job_path) == item['job_sha256'] and job['id'] == argv[6]
        env = dict(x.split('=', 1) for x in (proc / 'environ').read_text().split('\0') if '=' in x)
        assert all(env.get(k) == '1' for k in ('GUARDFED_CPU_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
        cpus = sorted(os.sched_getaffinity(int(proc.name)))
        assert not (len(cpus) <= 16 and set(cpus).intersection(range(8, 16)))
        fl_workers.append(dict(pid=int(proc.name), argv=argv, cwd=str(fl_stage), cpus=cpus, compute_threads=1,
            job_sha256=digest(job_path), run_one_source_sha256=digest(fl_script),
            package_sha256=digest(fl_stage / 'PACKAGE_SHA256.json'), thread_environment_verified_one=True))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
assert len(fl_workers) <= 2 and len({row['argv'][-1] for row in fl_workers}) == len(fl_workers)
assert not {row['pid'] for row in fl_workers}.intersection(row['pid'] for row in resources['tracked_compute'])
effective_nominal = resources['nominal_compute_threads_including_this8'] + len(fl_workers)
assert effective_nominal <= resources['actual_quota_cores']
save(HERE / 'RECOVERY_AUTHORIZED.json', dict(status='ROOT_APPROVED_ONE_LAUNCHER_ONLY_RECOVERY',
    authority='Root explicitly authorized one independent v2 launcher after sealed pre-science shell failure, current agent task 2026-10-09',
    original_approved_sha256=digest(APPROVED_DIR / 'APPROVED.json'), failure_seal_sha256=digest(failure / 'FILES_SHA256.json'),
    failed_launcher_sha256=digest(failure / 'guardfed_celeba_hybrid_writer_repair.sh'),
    launcher_v2_sha256=digest(HERE / 'service.sh'), source_writer9_unchanged=True,
    only_delta='nounset after shared logging/environment setup; own v2 log/service identity',
    exact_jobs=read(APPROVED_DIR / 'APPROVED.json')['jobs'], resources=resources,
    known_FLscreen_workers=fl_workers, total_effective_nominal=effective_nominal,
    no_new_scientific_protocol=True, no_automatic_retry=True, install_script_sha256=digest(__file__)))
name = 'guardfed_celeba_hybrid_writer_repair_v2'
script = Path('/opt/supervisor-scripts') / (name + '.sh')
config = Path('/etc/supervisor/conf.d') / (name + '.conf')
assert not script.exists() and not config.exists()
script.write_bytes((HERE / 'service.sh').read_bytes()); script.chmod(0o755)
config.write_bytes((HERE / 'supervisor.conf').read_bytes())
steps = []
for command in (['supervisorctl', 'reread'], ['supervisorctl', 'add', name], ['supervisorctl', 'start', name]):
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    steps.append(dict(argv=command, stdout=result.stdout, stderr=result.stderr))
save(HERE / 'start_receipt.json', dict(status='ONE_AUTHORIZED_RECOVERY_START_SENT_NOT_ACCEPTANCE', steps=steps))
print(json.dumps(dict(status='RECOVERY_START_SENT_NOT_ACCEPTANCE', service=name,
    launcher_v2_sha256=digest(script), original_failure_launcher_sha256=proof['source_approval_and_launcher_sha_before_after_identical'][str(Path('/opt/supervisor-scripts/guardfed_celeba_hybrid_writer_repair.sh'))],
    nominal_cpu_threads=resources['nominal_compute_threads_including_this8'])), flush=True)
