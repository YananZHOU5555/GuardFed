"""Execute root's explicit conditional authorization for this exact 32-job release."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

BASE = Path(__file__).resolve().parent
RELEASE = BASE / 'release_v2'
SERVICE = 'guardfed_celeba_flgmm_screen'
EXPECTED = 'aec95ceb5e8c9b7aa9cec89e9d70d2700c2f242c29e918792088648d6269bad4'
sys.path.insert(0, str(RELEASE))
from screen_common import digest, local_identity, read, write_json

steps = []
try:
    protocol, manifest = local_identity()
    assert digest(RELEASE / 'PACKAGE_SHA256.json') == EXPECTED
    preflight = read(BASE / 'PREFLIGHT_V2.json')
    assert preflight['status'] == 'RELEASE_UPLOADED_HASHED_NOT_AUTHORIZED_OR_STARTED'
    assert preflight['package_sha256'] == EXPECTED
    then = datetime.datetime.fromisoformat(preflight['after']['utc'])
    assert (datetime.datetime.now(datetime.timezone.utc) - then).total_seconds() < 300
    assert not (BASE / 'PREFLIGHT_V2_FAILURE.json').exists()
    assert not (RELEASE / 'EXECUTION_AUTHORIZATION.json').exists()
    assert not (RELEASE / 'runs').exists()
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if int(path.parent.name) == os.getpid():
            continue
        try:
            args = path.read_bytes().decode().strip('\0').split('\0')
        except (OSError, UnicodeError):
            continue
        assert not (args and 'python' in Path(args[0]).name and any('flgmm' in a.lower() for a in args)), args
    config = Path('/etc/supervisor/conf.d') / (SERVICE + '.conf')
    wrapper = Path('/opt/supervisor-scripts') / (SERVICE + '.sh')
    assert not config.exists() and not wrapper.exists()
    formal_before = subprocess.check_output(['supervisorctl', 'status', 'guardfed_celeba_mechanism_formal'], text=True).strip()
    assert 'RUNNING' in formal_before and 'pid 9179' in formal_before
    authorization = dict(status='AUTHORIZED', scope='32_valid_only_screen',
        package_sha256=EXPECTED, resource_review_utc=preflight['after']['utc'],
        preflight_sha256=digest(BASE / 'PREFLIGHT_V2.json'), no_duplicate_workers_verified=True,
        authorization_basis='Root explicitly authorized reviewed v2 freeze/upload and conditional launch after complete live preflight',
        jobs=32, rounds=70, seed=91001, max_gpu_workers=2, cpu_threads_per_worker=1, nice=10,
        final_test=False, formal100=False, automatic_retry=False,
        written_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(RELEASE / 'EXECUTION_AUTHORIZATION.json', authorization)
    shutil.copyfile(RELEASE / 'dispatch' / config.name, config)
    shutil.copyfile(RELEASE / 'dispatch' / wrapper.name, wrapper)
    wrapper.chmod(0o755)
    for command in [['supervisorctl', 'reread'], ['supervisorctl', 'update', SERVICE], ['supervisorctl', 'start', SERVICE]]:
        result = subprocess.run(command, capture_output=True, text=True)
        steps.append(dict(command=command, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr))
        assert result.returncode == 0, steps[-1]
    service = subprocess.check_output(['supervisorctl', 'status', SERVICE], text=True).strip()
    formal_after = subprocess.check_output(['supervisorctl', 'status', 'guardfed_celeba_mechanism_formal'], text=True).strip()
    assert 'RUNNING' in service
    assert 'RUNNING' in formal_after and 'pid 9179' in formal_after
    write_json(BASE / 'START_V2_RECEIPT.json', dict(status='STARTED_NOT_ACCEPTED',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), service=service,
        package_sha256=EXPECTED, authorization_sha256=digest(RELEASE / 'EXECUTION_AUTHORIZATION.json'),
        installed_config_sha256=digest(config), installed_wrapper_sha256=digest(wrapper),
        formal_before=formal_before, formal_after=formal_after, steps=steps))
    print(service)
except BaseException as error:
    path = BASE / 'START_V2_FAILURE.json'
    if not path.exists():
        write_json(path, dict(error=repr(error), traceback=traceback.format_exc(), time=time.time(), steps=steps))
    raise
