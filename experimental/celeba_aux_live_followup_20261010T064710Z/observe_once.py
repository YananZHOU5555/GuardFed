"""Reuse the preceding read-only remote body once; preserve this bounded snapshot."""
from pathlib import Path
import datetime, hashlib, json, runpy, subprocess, sys, traceback
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
R = H.parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())

def save(name, value):
    with (H/name).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')

def main():
    original = R/'tmp/celeba_aux_live_after_20261010/observe_once.py'
    namespace = runpy.run_path(str(original), run_name='saved_readonly_body')
    remote = namespace['REMOTE']
    guide = R/'tmp/celeba_flgmm_fullcoverage_delta_after22_20261010/SERVER_GUIDE.md'
    assert sha(guide) == '42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
    flbase = R/'tmp/celeba_flgmm_fullcoverage_incremental_20261009'
    fl_latest = read(flbase/'LATEST_BACKUP.json')
    flroot = R/fl_latest['root_adoption_path']
    fl = read(flroot)
    assert fl_latest['accepted_total'] == fl['accepted_total'] == 28
    assert sha(flroot) == fl_latest['root_adoption_sha256'] == 'e51007c549f8cc95970ce06cb61b4a477c11eff5d00e05a9aeac9ffb30dd449c'
    assert sha(fl_latest['next_collector_previous_path']) == fl_latest['next_collector_previous_sha256']
    hybase = R/'tmp/celeba_hybrid_screen_execution_20261009'
    hy_latest = read(hybase/'LATEST_BACKUP.json')
    hychain = hybase/hy_latest['chain_file']
    hy = read(hychain)
    assert hy_latest['accepted'] == hy['accepted_total'] == 27
    assert sha(hychain) == hy_latest['chain_sha256']
    hyroot = R/hy['root_adoption_path']
    assert sha(hyroot) == hy['root_adoption_sha256'] and read(hyroot)['accepted_total'] == 27
    save('INPUT_PINS.json', dict(original_reader_path=original.relative_to(R).as_posix(), original_reader_sha256=sha(original), remote_body_sha256=hashlib.sha256(remote.encode()).hexdigest(), guide_sha256=sha(guide), FL_latest_path=(flbase/'LATEST_BACKUP.json').relative_to(R).as_posix(), FL_latest_sha256=sha(flbase/'LATEST_BACKUP.json'), FL_root_path=flroot.relative_to(R).as_posix(), FL_root_sha256=sha(flroot), FL_previous_path=fl_latest['next_collector_previous_path'], FL_previous_sha256=fl_latest['next_collector_previous_sha256'], Hybrid_latest_path=(hybase/'LATEST_BACKUP.json').relative_to(R).as_posix(), Hybrid_latest_sha256=sha(hybase/'LATEST_BACKUP.json'), Hybrid_chain_path=hychain.relative_to(R).as_posix(), Hybrid_chain_sha256=sha(hychain), Hybrid_root_sha256=sha(hyroot)))
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', '60350', 'root@89.22.197.55', 'python -B -']
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    p = subprocess.run(command, input=remote.encode(), capture_output=True, timeout=75)
    for name, content in [('STDOUT.txt',p.stdout),('STDERR.txt',p.stderr)]:
        with (H/name).open('xb') as f:
            f.write(content)
    save('COMMAND.json', dict(command=command, start=start, finish=datetime.datetime.now(datetime.timezone.utc).isoformat(), exit_code=p.returncode, single_attempt=True, automatic_retry=False))
    p.check_returncode()
    snapshot = json.loads(p.stdout)
    save('SNAPSHOT.json', snapshot)
    f = snapshot['FLGMM']; y = snapshot['Hybrid']
    ft = [row['id'] for row in f['rows'] if row['observed_terminal']]
    ht = [row['id'] for row in y['rows'] if row['observed_terminal']]
    assert set(fl['accepted_job_ids']) <= set(ft) and set(hy['accepted_job_ids']) <= set(ht)
    nf = [job for job in ft if job not in fl['accepted_job_ids']]
    nh = [job for job in ht if job not in hy['accepted_job_ids']]
    assert type(f['queue']['pending']) is int
    clean = not f['changed_source_members'] and not f['failure_paths'] and f['queue']['failed'] is False and not y['changed_source_members'] and not y['screen_failure'] and not any(row['failures'] for row in y['rows'])
    result = dict(status='BOUNDED_READONLY_OBSERVATION_NOT_ACCEPTANCE', utc=snapshot['utc'], snapshot_sha256=sha(H/'SNAPSHOT.json'), FL_accepted=28, FL_observed_terminal=len(ft), FL_new_ids=nf, FL_active=[dict(id=row['id'],round=row['round']) for row in f['rows'] if row['active']], FL_pending=f['queue']['pending'], FL_reused_separate=f['separately_reused'], FL_service=f['service'], Hybrid_accepted=27, Hybrid_observed_terminal=len(ht), Hybrid_new_ids=nh, Hybrid_incomplete_progress=[dict(id=row['id'],round=row['round']) for row in y['rows'] if not row['observed_terminal'] and row['round'] is not None], Hybrid_pending=sum(row['round'] is None for row in y['rows']), Hybrid_service=y['service'], source_and_failure_clean=clean, CPU106_free=snapshot['CPU106_all_thread_free'], existing_collectors=snapshot['existing_collectors'], Hybrid_complete32_boundary=len(ht)==32, FL_new_at_least4=len(nf)>=4, priority_candidate='Hybrid32_COMPLETE' if len(ht)==32 and clean else 'FL_DELTA' if len(nf)>=4 and clean else 'NO_THRESHOLD_MET', collect_performed=False, download_performed=False, new_accepted=0, no_bulk_written=True, observation_is_atomic=False)
    save('FINDINGS.json', result)
    print(json.dumps(result, ensure_ascii=False))

if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        save('FAILURE.json', dict(error=repr(e), traceback=traceback.format_exc(), automatic_retry=False))
        raise
