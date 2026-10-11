"""Root-only initial measurement, then exec the unchanged NGA96 runner once."""
import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = Path('/workspace/guardfed_checks/celeba_gradient_fullcoverage_selected_20261011/Fed-NGA-gradient')
REPO = Path('/workspace/GuardFed-celeba-expanded')
INITIAL_SHA = 'f5b8d252df52d9afd276015fa197af7a521280e64f674405b9c8320cd34d9aef'
SOURCE_APPROVAL_SHA = '79edd0dde78c9dc23583a22953d3babebb357a4fcebb7197941cc5726b2ff961'
FROZEN = {
    'FILES_SHA256.json': '06e3d638aea70123d7fcf73e5a219a0c0d3165c8cf60acb9dee4eace5faad09b',
    'run_queue.py': '5d75a33e0a9a434bc4bd03369ece898183e8d6679234929adbdea6f616010cab',
    'coverage_contract.py': '2ccc116bf67976b33028ff6f4f35a6327056fe8415db98c4d149df810a3edc7c',
    'job_preflight.py': '2e4d8dd68bccbb1ef2756a62c4e7e086419c064f9272073b6b43cefed4d24697',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--receipt-dir', type=Path, required=True)
    parser.add_argument('--source-approval', type=Path, required=True)
    parser.add_argument('--source-approval-sha256', required=True)
    parser.add_argument('--gate14-adoption', type=Path, required=True)
    parser.add_argument('--gate14-adoption-sha256', required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux' and not sys.flags.optimize
    out, receipts = args.out.resolve(), args.receipt_dir.resolve()
    assert out.is_relative_to(Path('/workspace')) and out != Path('/workspace') and not out.exists()
    assert receipts.is_relative_to(Path('/workspace')) and receipts != Path('/workspace') and not receipts.exists()
    assert not receipts.is_relative_to(out), 'Receipts must not create the new queue output'
    receipts.mkdir(parents=True, exist_ok=False)
    phase = 'bootstrap_identity'
    save(receipts/'BOOTSTRAP_COMMAND.json', dict(argv=sys.argv, pid=os.getpid(),
        source_sha256=sha(__file__), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        automatic_retry=False, role='ROOT_BOOTSTRAP_NOT_YET_QUEUE_CONTROLLER'))
    try:
        assert all(sha(BASE/name) == expected for name, expected in FROZEN.items())
        assert sha(HERE/'initial_preflight.py') == INITIAL_SHA
        assert args.source_approval_sha256 == SOURCE_APPROVAL_SHA == sha(args.source_approval)
        assert len(args.gate14_adoption_sha256) == 64 and all(c in '0123456789abcdef' for c in args.gate14_adoption_sha256)
        assert sha(args.gate14_adoption) == args.gate14_adoption_sha256
        probe_args = [str(HERE/'initial_preflight.py'), '--base', str(BASE), '--repo', str(REPO),
            '--queue-out', str(out), '--proof-dir', str(receipts),
            '--source-approval', str(args.source_approval), '--source-approval-sha256', args.source_approval_sha256,
            '--gate14-adoption', str(args.gate14_adoption), '--gate14-adoption-sha256', args.gate14_adoption_sha256]
        phase = 'initial_resource_probe'
        save(receipts/'INITIAL_PREFLIGHT_COMMAND.json', dict(call_kind='runpy_in_same_real_bootstrap_process',
            argv=probe_args, pid=os.getpid(), source_sha256=INITIAL_SHA, fake_controller=False))
        actual_argv = sys.argv
        try:
            # /proc cmdline stays this bootstrap; no run_queue controller is impersonated.
            sys.argv = probe_args
            with (receipts/'INITIAL_PREFLIGHT.stdout.json').open('x', encoding='utf8') as stdout:
                with (receipts/'INITIAL_PREFLIGHT.stderr.txt').open('x', encoding='utf8') as stderr:
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        namespace = runpy.run_path(str(HERE/'initial_preflight.py'), run_name='__main__')
        finally:
            sys.argv = actual_argv
        save(receipts/'INITIAL_PREFLIGHT_EXIT.json', dict(returncode=0, pid=os.getpid()))
        proof_path = receipts/'INITIAL_RESOURCE.json'
        proof = json.loads(proof_path.read_bytes())
        assert proof['producer_role'] == 'ROOT_INITIAL_RESOURCE_PROBE_NOT_QUEUE_CONTROLLER'
        assert proof['bootstrap_probe_pid'] == os.getpid() and 'controller_pid' not in proof
        assert proof['common_lock_probe_pass'] is True and 0 <= time.time()-proof['at_unix'] <= 90
        command = [sys.executable, '-B', str(BASE/'run_queue.py'), '--repo', str(REPO), '--out', str(out),
            '--source-approval', str(args.source_approval), '--source-approval-sha256', args.source_approval_sha256,
            '--gate14-adoption', str(args.gate14_adoption), '--gate14-adoption-sha256', args.gate14_adoption_sha256,
            '--resource-preflight', str(proof_path), '--resource-sha256', sha(proof_path)]
        save(receipts/'RUNNER_HANDOFF.json', dict(argv=command, pid=os.getpid(),
            source_sha256=FROZEN['run_queue.py'], mode='os.execv_same_PID_no_live_bootstrap_parent',
            initial_resource_sha256=sha(proof_path), note='Runner exit belongs to supervisor; not prefilled here'))
        phase = 'exec_original_runner'
        # The original runner reacquires and owns its existing common lock. A race refuses.
        namespace['common_lock'].close()
        os.execv(sys.executable, command)
    except BaseException as error:
        save(receipts/'BOOTSTRAP_FAILURE.json', dict(phase=phase, error=repr(error),
            traceback=traceback.format_exc(), automatic_retry=False, scientific_acceptance=False))
        raise


if __name__ == '__main__':
    main()
