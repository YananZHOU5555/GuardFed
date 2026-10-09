"""Future explicit root-approved materialization. No defaults select a recipe or authorize execution."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

if sys.flags.optimize:
    raise RuntimeError('Optimized Python is forbidden: scientific identity/comparison assertions must execute')

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def validate_selection(summary, adoption, old_protocol, manifest):
    assert summary['status'] == 'ROOT_ACCEPTED32_RECORD_ONLY_RECIPE_SUMMARY' and summary['accepted'] == 32
    assert summary['final_test'] is False and summary['formal100_started'] is False and summary['seed_n'] == 1
    rows = summary['records']; expected = {r['id']: r for r in manifest['jobs']}
    assert len(rows) == len({r['id'] for r in rows}) == len(expected) == 32
    assert {r['id'] for r in rows} == set(expected)
    from frozen_score import score
    scored=[]
    for recipe in old_protocol['candidates']:
        group=[r for r in rows if r['candidate']==recipe['id']]
        assert len(group)==4 and {(r['distribution'],r['attack']) for r in group}=={(d,a) for d in ('IID','non-IID') for a in ('Benign','S-DFA')}
        scored.append((statistics.mean(score(r['metrics']) for r in group),recipe['id']))
    candidate = summary['selected_candidate_recipe']
    assert candidate in old_protocol['candidates']
    assert candidate['id']==min(scored,key=lambda x:(-x[0],x[1]))[1]
    assert summary['selected_per_method']['FLGMM-author-code']['candidate'] == candidate['id']
    assert adoption['status'] == 'ROOT_APPROVED_FLGMM100_RECIPE_BINDING'
    assert adoption['scope'] == '96_new_plus_4_reused_70round_valid_only'
    assert adoption['selected_recipe'] == candidate
    assert adoption['final_test'] is False and adoption['execute_authorized'] is False
    reused = []
    for row in rows:
        item = expected[row['id']]
        assert (row['candidate'], row['distribution'], row['attack'], row['seed'], row['rounds']) == (
            item['tuning_candidate'], item['distribution'], item['attack'], 91001, 70)
        if row['candidate'] == candidate['id']:
            reused.append(dict(item, accepted_record=row))
    assert len(reused) == 4 and {(r['distribution'], r['attack']) for r in reused} == {
        (d, a) for d in ('IID', 'non-IID') for a in ('Benign', 'S-DFA')}
    return candidate, reused


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('old-release', 'summary', 'final-root-proof', 'adoption', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    for key in ('summary-sha256', 'final-root-proof-sha256', 'adoption-sha256', 'source-seal-sha256'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    assert not sys.flags.optimize and not args.output.exists()
    assert sha(HERE / 'FILES_SHA256.json') == args.source_seal_sha256
    for name, pin in read(HERE / 'FILES_SHA256.json')['files'].items():
        assert sha(HERE / name) == pin['sha256']
    for path, digest in [(args.summary, args.summary_sha256), (args.final_root_proof, args.final_root_proof_sha256), (args.adoption, args.adoption_sha256)]:
        assert sha(path) == digest
    old = args.old_release.resolve()
    assert sha(old / 'PACKAGE_SHA256.json') == 'aec95ceb5e8c9b7aa9cec89e9d70d2700c2f242c29e918792088648d6269bad4'
    for name, digest in read(old / 'PACKAGE_SHA256.json')['files'].items():
        assert sha(old / name) == digest
    summary, approval, proof = read(args.summary), read(args.adoption), read(args.final_root_proof)
    assert proof['status'] == 'ROOT_LOCKED_AUXILIARY_DELTA_ARCHIVE_MEMBER_AND_ORIGINAL_RECORD_ACCEPTANCE_PASS'
    assert (proof['accepted_before'], proof['accepted_new'], proof['accepted_total']) == (26, 6, 32)
    assert proof['new_inference'] == 0 and proof['final_test'] is False
    assert proof['previous_chain_sha256']=='3844e24261b7e3a89a550b0fcb0016bbf2fb1b8a579f6e51b6a3e5afff23032b'
    assert summary['source_bindings']['final_root_proof_sha256'] == args.final_root_proof_sha256
    assert summary['source_bindings']['final_archive_sha256']==proof['archive_sha256']
    assert summary['source_bindings']['final_strict_sha256']==proof['strict_receipt_sha256']
    assert summary['source_bindings']['final_offserver_sha256']==proof['offserver_proof_sha256']
    assert approval['summary_sha256'] == args.summary_sha256
    assert approval['final_root_proof_sha256'] == args.final_root_proof_sha256
    assert approval['source_seal_sha256'] == args.source_seal_sha256
    assert Path(approval['output']).resolve() == args.output.resolve()
    protocol, manifest = read(old / 'source/protocol.json'), read(old / 'jobs/manifest.json')
    candidate, reused = validate_selection(summary, approval, protocol, manifest)
    for item in reused:
        path = old / 'jobs' / item['job']; assert sha(path) == item['job_sha256']
        job = read(path); row = item['accepted_record']
        assert job['config']['rounds'] == 70 and job['config']['seed'] == 91001
        assert job['config']['client_alpha'] == protocol['distributions'][item['distribution']]
        assert job['adapter'] == candidate['adapter'] and row['job_sha256'] == item['job_sha256']
        assert row['source_hashes'] == job['source_hashes'] == protocol['source_hashes']
    output = args.output.resolve(); output.mkdir()
    for name in read(HERE / 'FILES_SHA256.json')['files']:
        if name == 'source/protocol.json': continue
        target = output / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(HERE / name, target)
    shutil.copyfile(HERE / 'FILES_SHA256.json', output / 'PREPARED_SOURCE_SEAL.json')
    for name, source in [('SELECTED_SUMMARY.json', args.summary), ('FINAL32_ROOT_PROOF.json', args.final_root_proof), ('BIND_APPROVAL.json', args.adoption)]:
        shutil.copyfile(source, output / name)
    selected = copy.deepcopy(protocol)
    selected.update(status='FROZEN', version='celeba_flgmm_selected_fullcoverage_20261009_v1',
                    scope='flgmm_selected_100_valid_only', selected_recipe=candidate, candidates=[candidate],
                    attacks=['Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA'], seeds=list(range(91001, 91011)))
    selected['execution'] = dict(protocol['execution'], status='NOT_AUTHORIZED_TO_START', formal100=True)
    save(output / 'source/protocol.json', selected)
    source_names = ['flgmm_adapter.py', 'worker.py', 'prepare_jobs.py', 'protocol.json',
                    'sources/flgmm_pinned.py', 'sources/flgmm_fedavg.py', 'sources/flgmm_license.txt']
    hashes = {name: sha(output / 'source' / name) for name in source_names}
    sys.path.insert(0, str(output / 'source'))
    from prepare_jobs import definitions, make_job
    from worker import validate_job
    new, gates = [], []
    for job in list(definitions(selected, hashes)) + [make_job(selected, hashes, 'non-IID', attack, 91001, 'preflight') for attack in selected['attacks']]:
        validate_job(job, selected)
        path = output / 'jobs' / (job['id'] + '.json'); save(path, job)
        item = dict(id=job['id'], job=path.relative_to(output / 'jobs').as_posix(), job_sha256=sha(path),
                    tuning_candidate=job['tuning_candidate'], distribution=job['distribution'], attack=job['attack'], seed=job['config']['seed'])
        (new if job['phase'] == 'fullcoverage' else gates).append(item)
    assert len(new) == 96 and len(gates) == 5 and len(reused) == 4
    for item in reused:
        item.update(legacy_release=str(old), original_output=str(old / 'runs' / item['id']))
    save(output / 'manifest.json', dict(scope='flgmm_selected_100_valid_only', planned_new=96, planned_total=100,
         jobs=new, reused_jobs=reused, preflight_jobs=gates, protocol_sha256=sha(output / 'source/protocol.json'),
         summary_sha256=args.summary_sha256, final_root_proof_sha256=args.final_root_proof_sha256,
         adoption_sha256=args.adoption_sha256, execution_authorized=False))
    files = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    save(output / 'PACKAGE_SHA256.json', dict(status='FROZEN_PENDING_EXECUTION', files=files))
    print(json.dumps(dict(stage=str(output), package_sha256=sha(output / 'PACKAGE_SHA256.json'), new=96, reused=4, gates=5, execute_authorized=False)))


if __name__ == '__main__': main()
