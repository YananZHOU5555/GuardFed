"""Stop only the pinned metadata reader before stage creation, preserving approved inputs."""
import datetime, hashlib, json, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OPS = ROOT/'tmp/celeba_logofair100_root_operations_20261010'
BASE = Path('F:/YananResearchStorage/GuardFed/logofair_fullcoverage100_20261010')
H = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(name, data):
    with (HERE/name).open('x', encoding='utf8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write('\n')

def run():
    review = ROOT/'tmp/celeba_logofair100_reader_v2_review_20261010/REVIEW.json'
    assert H(review) == '436a5b8a64a450d2883005b0adbdfb4d6b0b08e1b2e7be85d04355d79f67f4ce'
    assert H(HERE/'bind_with_read_guard.py') == 'd8158c1ab988e10f650aab994a2815aee15e74d9af8fdbef5e4c14e28702ad1e'
    stage = BASE/'stage001'
    assert not stage.exists(), 'Original stage already exists: leave original process alone'
    def query():
        p = subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',
            'Get-CimInstance Win32_Process -Filter "ProcessId=133008" | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress'],
            capture_output=True, text=True, check=True, timeout=30)
        return json.loads(p.stdout.lstrip('\ufeff')) if p.stdout.strip() else None
    proc = query()
    assert proc and proc['ProcessId'] == 133008 and proc['ParentProcessId'] == 134584
    command = proc['CommandLine']
    assert 'prepare_inputs_and_bind_approval.py' in command and 'bind_attempt001' in command
    for pin in ['67a595e14e5fc208800dc5f1b7556d53590c52ca30f1edcd9f75b2fad09f138c',
                '2805bb5c77cd0f5972d9240fbb5ef67b1ddde6e2384582bbc77cf0e5ca4983e5',
                '30e1607d8c94b2992f9b390a7a177983f502fe93436ca6921431d08fda0d112d']:
        assert pin in command
    started = OPS/'bind_attempt001/STARTED.json'
    inputs = BASE/'root_approved_inputs001/ROOT_INPUTS.json'
    approval = BASE/'root_approved_inputs001/BIND_APPROVAL.json'
    evidence = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        process=proc, original_started_sha256=H(started), inputs_sha256=H(inputs), approval_sha256=H(approval),
        stage_absent_before_stop=True, source_review_sha256=H(review), original_attempt_preserved=True)
    save('PRE_STOP.json', evidence)
    assert not stage.exists(), 'Stage creation raced: do not stop'
    stop = subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',
        "$ErrorActionPreference='Stop'; $p=Get-CimInstance Win32_Process -Filter 'ProcessId=133008'; "
        "if ($null -eq $p -or $p.ParentProcessId -ne 134584 -or "
        "$p.CommandLine -notmatch 'prepare_inputs_and_bind_approval.py' -or "
        "$p.CommandLine -notmatch 'bind_attempt001') { throw 'Process identity changed' }; "
        "Stop-Process -Id 133008 -ErrorAction Stop; Wait-Process -Id 133008 -Timeout 10 -ErrorAction SilentlyContinue"],
        capture_output=True, text=True, timeout=30)
    save('STOP_COMMAND.json', dict(exit_code=stop.returncode, stdout=stop.stdout, stderr=stop.stderr))
    assert stop.returncode == 0 and query() is None, 'Stop not confirmed; no continuation'
    assert not stage.exists(), 'Stage creation raced: preserve existing stage, no continuation'
    assert H(inputs)==evidence['inputs_sha256'] and H(approval)==evidence['approval_sha256']
    proof = dict(evidence, status='ROOT_READONLY_BIND_CANCELLED_BEFORE_STAGE_CREATION',
        original_process_stopped=True, stage_absent_after_stop=True, CNN_calls=0, fits=0,
        stop_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), automatic_retry=False)
    save('CANCELLATION.json', proof)
    print(json.dumps(dict(path=str(HERE/'CANCELLATION.json'), sha256=H(HERE/'CANCELLATION.json'))))

if __name__ == '__main__':
    run()
