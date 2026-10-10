"""Read the actual fixed-recipe launch and first original strict fit; no fitting."""
import argparse, datetime, hashlib, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / 'tmp/celeba_logofair100_root_operations_20261010'
BASE = Path('F:/YananResearchStorage/GuardFed/logofair_fullcoverage100_20261010')
H = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
R = lambda p: json.loads(Path(p).read_bytes())

def run(a):
    assert H(a.startup) == a.sha256
    start = R(a.startup)
    assert start['status'] == 'WINDOWS_HIDDEN_PROCESS_CREATED_NOT_ACCEPTED'
    assert start['automatic_retry'] is False and start['accepted_offserver'] == start['root_adopted'] == 0
    assert start['environment_overrides']['CUDA_VISIBLE_DEVICES'] == ''
    assert all(start['environment_overrides'][k] == '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
    stage, out = BASE/'stage001', BASE/'attempt001'
    assert Path(start['output']).resolve() == out.resolve()
    binding_path = a.bind_result.resolve()
    assert binding_path.parent.parent == CONTROL.resolve()
    binding = R(binding_path)
    assert H(binding_path) == start['bind_result_sha256']
    manifest = R(stage/'manifest.json')
    assert H(stage/'manifest.json') == binding['manifest_sha256']
    assert H(stage/'SOURCE_SHA256.json') == binding['source_sha256']
    for name, wanted in R(stage/'SOURCE_SHA256.json').items():
        assert H(stage/name) == wanted
    jobs, reuse = manifest['jobs'], manifest['reused_jobs']
    assert len(jobs) == 96 and len(reuse) == 4
    assert len({r['id'] for r in jobs+reuse}) == 100
    assert manifest['candidate']['id'] == 'LoGoFair-DP_07' and manifest['new_CNN'] == 0 and not manifest['final_test']
    protocol = R(stage/'snapshot/logofair_bridge_20261010/protocol.json')
    assert protocol['seeds'] == list(range(91001, 91011))
    assert protocol['candidates'] == [manifest['candidate']]
    for row in jobs:
        job = R(stage/row['job'])
        assert H(stage/row['job']) == row['job_sha256']
        assert job['seed'] in protocol['seeds'] and job['fit_seed'] == 1719
        assert job['evaluation_split'] == 'valid' and job['settings']['post_rounds'] == 30
    assert not (out/'QUEUE_FAILURE.json').exists()
    marker = R(stage/'FULLCOVERAGE_STARTED.json')
    assert marker['approval_sha256'] == start['approval_sha256']
    command = "$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'logofair_fullcoverage_20261010[/\\\\]run96\\.py|logofair_fullcoverage100_20261010[/\\\\]stage001' } | ForEach-Object { $p = Get-Process -Id $_.ProcessId -ErrorAction Stop; [pscustomobject]@{pid=$_.ProcessId;parent=$_.ParentProcessId;argv=$_.CommandLine;priority=[string]$p.PriorityClass;cpu_seconds=$p.CPU} }) | ConvertTo-Json -Compress"
    proc = subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',command], check=True, capture_output=True, text=True, timeout=30)
    processes = json.loads(proc.stdout.lstrip('\ufeff'))
    assert isinstance(processes, list) and processes
    owner = next(p for p in processes if p['pid'] == marker['pid'])
    assert owner['priority'] == 'Idle'
    progress_path = out/'PROGRESS.json'
    assert progress_path.exists(), 'Wait for the first original strict fit; do not restart'
    progress = R(progress_path)
    assert progress['records'] and len(progress['records']) <= 96
    first = progress['records'][0]
    result, acceptance = R(first['result']), R(first['acceptance'])
    assert H(first['result']) == first['result_sha256'] and H(first['acceptance']) == first['acceptance_sha256']
    assert first['id'] == jobs[0]['id'] and acceptance['status'] == 'PASS'
    assert acceptance['job_sha256'] == jobs[0]['job_sha256']
    assert [r['round'] for r in result['history']] == list(range(1,31))
    for name, wanted in acceptance['artifact_hashes'].items():
        assert H(Path(first['result']).parent/name) == wanted
    proof = dict(status='ROOT_LOGOFAIR100_FIXED_RECIPE_ACTUAL_STARTUP_AND_FIRST_STRICT_FIT_PASS',
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),startup_path=a.startup.relative_to(ROOT).as_posix(),
        startup_sha256=a.sha256,bind_result_sha256=H(binding_path),manifest_sha256=H(stage/'manifest.json'),
        source_sha256=H(stage/'SOURCE_SHA256.json'),approval_sha256=start['approval_sha256'],
        stage=stage.as_posix(),output=out.as_posix(),processes=processes,coordinator_pid=marker['pid'],
        selected_recipe='LoGoFair-DP_07',model_seeds=protocol['seeds'],fit_seed=1719,new_fits_planned=96,reused=4,
        original_strict_closed_at_observation=len(progress['records']),first_original_strict=first,
        first_rounds=30,first_metrics=result['metrics'],offserver_accepted=0,root_adopted=0,
        CNN_training=0,CNN_inference=0,final_test=False,automatic_retry=False,
        scope='Local first strict fit and actual startup only; complete100 awaits independent saved-array and identity acceptance')
    assert not a.output.exists() and a.output.resolve().parent == CONTROL.resolve()
    a.output.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(path=str(a.output),sha256=H(a.output),pid=marker['pid'],local_strict=len(progress['records']),root_adopted=0)))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--startup',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--bind-result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
