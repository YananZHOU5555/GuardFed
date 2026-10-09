"""Execute the parent's explicit bounded resource authorization, never a scientific freeze."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

from gate import HERE, REPO, digest, dispatch, hashes, read, require, snapshot, validate_scope, write

SERVICE = 'guardfed_celeba_gradient_realimage_gate'
CPUS = list(range(104, 112))


def cpu_processes():
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            exe = path.joinpath('exe').resolve().name
            command = path.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode()
            if 'python' in exe and '/workspace/guardfed_checks/' in command:
                found.append(dict(pid=int(path.name), command=command, cpu_affinity=sorted(os.sched_getaffinity(int(path.name)))))
        except (OSError, UnicodeError):
            continue
    return found


def main():
    require(str(HERE) == '/workspace/guardfed_checks/celeba_gradient_realimage_gate_20261009', 'Wrong exclusive gate directory')
    setup = read(HERE / 'SETUP_SHA256.json')
    require(digest(HERE / 'SETUP_SHA256.json') == 'c2c9325f7c2be4fc9dbd5164990f6e880a132bd26661e79e2b9d524ffbaca0cc', 'Parent-reviewed setup changed')
    require(len(setup['file_sha256']) == 22, 'Unexpected setup inventory'); hashes(HERE, setup['file_sha256'])
    scope = read(HERE / 'scope.json'); validate_scope(scope)
    require(digest('/etc/vast-agents-guide.md') == scope['guide_sha256'], 'Guide changed')
    require(set(CPUS).issubset(os.sched_getaffinity(0)), 'Approved CPUs unavailable')
    require(not (HERE / 'runs').exists() and not (HERE / 'dispatch_receipt.APPROVED.json').exists(), 'Existing outputs/dispatch require review')
    live = cpu_processes()
    require(not any('celeba_gradient_realimage_gate_20261009/gate.py' in x['command'] for x in live), 'Duplicate gradient gate')
    require(not any(set(CPUS).intersection(x['cpu_affinity']) for x in live), 'Approved CPUs overlap a live CPU gate')
    source = hashes(REPO, scope['protected_source_hashes'])
    before = snapshot(); require(not before['failed'], 'Protected formal800 queue has failures')
    write(HERE / 'launch_preflight.json', dict(at_unix=time.time(), source_hashes=source,
        local_setup_member_sha256=setup['file_sha256'], live_cpu_gates=live, formal_resource=before,
        approved_cpu_ids=CPUS, no_duplicate_or_cpu_overlap=True))
    additions = {name: digest(HERE / name) for name in ('EXECUTION_NOTES.md', 'service.sh',
        'guardfed_celeba_gradient_realimage_gate.conf', 'approve_dispatch.py', 'shared_cache_wrapper.py', 'shared_cache_bindings.json')}
    receipt = dict(status='APPROVED_BOUNDED_EXPLORATORY_GATE_ONLY', scope_sha256=digest(HERE / 'scope.json'),
        jobs={x['id']: x['job_sha256'] for x in scope['jobs']}, source_hashes=source, local_hashes=scope['local_hashes'],
        exclusive_cpu_ids=CPUS, verified_no_overlap_with_live_cpu_gates=True,
        formal_decisions_approved=False, screen64_authorized=False, test_authorized=False,
        authorized_by='Parent /root explicit NEW_TASK 2026-10-09: execute four exploratory pilots, CPU104..111, supervisor fail-stop',
        authorized_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), execution_attachment_sha256=additions,
        shared_cache_bindings=read(HERE / 'shared_cache_bindings.json'),
        metadata_access_disclosure='Frozen src/celeba_data.py line44 materializes full Smiling/Male metadata including test rows. '
            'Training uses train images; scoring uses valid images. No test-image inference/fitting/scoring/selection. Not untouched test.',
        test_metadata_materialized=True, test_images_inferred=False, test_used_for_fitting_or_selection=False,
        niceness=10, io_class='idle', cpu_threads=8, max_concurrent_cpu_processes=1,
        supervisor=dict(service=SERVICE, autostart=False, autorestart=False, startretries=0),
        launch_preflight_sha256=digest(HERE / 'launch_preflight.json'))
    write(HERE / 'dispatch_receipt.APPROVED.json', receipt); dispatch(scope, HERE / 'dispatch_receipt.APPROVED.json')
    config = Path('/etc/supervisor/conf.d') / (SERVICE + '.conf')
    require(not config.exists(), 'Do not replace existing supervisor service')
    config.write_bytes((HERE / config.name).read_bytes())
    # Targeted update cannot change another program's registration/state.
    reread = subprocess.run(['supervisorctl', 'reread'], capture_output=True, text=True, check=True)
    update = subprocess.run(['supervisorctl', 'update', SERVICE], capture_output=True, text=True, check=True)
    start = subprocess.run(['supervisorctl', 'start', SERVICE], capture_output=True, text=True, check=True)
    status = subprocess.run(['supervisorctl', 'status', SERVICE], capture_output=True, text=True, check=True)
    write(HERE / 'launch_receipt.json', dict(status='STARTED_NOT_ACCEPTED', at_unix=time.time(),
        dispatch_receipt_sha256=digest(HERE / 'dispatch_receipt.APPROVED.json'), execution_attachment_sha256=additions,
        supervisor_config_sha256=digest(config), reread=reread.stdout, update=update.stdout,
        start=start.stdout, supervisor_status=status.stdout, real_image_gates_accepted=0,
        formal_protocol_modified=False, other_services_changed=False))
    print(json.dumps(dict(status='STARTED_NOT_ACCEPTED', service=status.stdout.strip(), cpus=CPUS)), flush=True)


if __name__ == '__main__':
    main()
