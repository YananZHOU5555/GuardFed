"""Bind one actually complete64 observation; no transport, selection or adoption."""
from pathlib import Path
import argparse, hashlib, json, sys

H = Path(__file__).resolve().parent
R = H.parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())


def save(name, value):
    with (H / name).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def validate_observation(raw, contract):
    g = raw['gradient64']
    expected = contract['expected_all64_ids']
    assert g['terminal_ids'] == expected and len(set(expected)) == 64
    assert g['queue_completed_ids'] == expected and not g['active']
    assert g['service']['returncode'] == 3
    assert g['service']['stdout'].split()[:2] == ['guardfed_celeba_gradient_screen64_v2a', 'EXITED']
    assert not g['failure_paths'] and not g['source']['changed_members']
    assert g['source']['sha256'] == contract['source_seal_sha256']
    assert g['manifest_sha256'] == contract['manifest_sha256']
    assert [r['id'] for r in g['rows']] == expected
    for row in g['rows']:
        assert row['round'] == 70 and row['observed_terminal']
        assert not row['active'] and row['acceptance_present']
        assert row['identity_checks'] and all(row['identity_checks'].values())
    assert raw['guide_sha256'] == '42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
    assert not raw['source_data_binding']['changed_paths']
    assert raw['source_data_binding']['all_gradient_registered_targets_match'] is True
    return g


if __name__ == '__main__':
    assert not sys.flags.optimize
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--snapshot-sha256', required=True)
    a = p.parse_args()
    for name, pin in read(H / 'PREPARATION_FILES_SHA256.json')['files'].items():
        assert sha(H / name) == pin['sha256'] and (H / name).stat().st_size == pin['bytes']
    contract = read(H / 'PENDING_BINDING.json')
    assert sha(R / contract['prior_root_path']) == contract['prior_root_sha256']
    assert sha(R / contract['prior_offserver_path']) == contract['prior_offserver_sha256']
    prior = read(H / 'PREVIOUS_ROOT_ADOPTION.json')
    assert prior['accepted_total'] == 46
    assert prior['accepted_ids'] + contract['remaining18_ids'] == contract['expected_all64_ids']
    assert sha(a.snapshot) == a.snapshot_sha256
    raw = read(a.snapshot)
    g = validate_observation(raw, contract)
    save('SNAPSHOT.json', dict(status='ACTUAL_COMPLETE64_OBSERVATION_NOT_SCIENTIFIC_ADOPTION',
        original_snapshot_path=str(a.snapshot.resolve()), original_snapshot_sha256=a.snapshot_sha256,
        utc=raw['utc'], guide_sha256=raw['guide_sha256'], terminal_ids=g['terminal_ids'],
        original_strict_acceptance_present64=True, observed_terminal64=True,
        all_producers_inactive_in_observation=True, source_changed_members=[], failure_paths=[],
        actual_runtime_EXITED_and_all_thread_resource_check_still_required=True))
    save('AUTHORIZED_SNAPSHOT.json', dict(status='BOUND_ACTUAL64_MINUS_ACCEPTED46_RUNTIME_REVIEW_REQUIRED',
        snapshot_sha256=sha(H / 'SNAPSHOT.json'), snapshot_utc=raw['utc'],
        accepted_prior_ids=prior['accepted_ids'], accepted_prior_count=46,
        authorized_ids=contract['remaining18_ids'], terminal_count=64,
        manifest_sha256=contract['manifest_sha256'], source_seal_sha256=contract['source_seal_sha256'],
        prior_root_path=contract['prior_root_path'], prior_root_sha256=contract['prior_root_sha256'],
        prior_offserver_path=contract['prior_offserver_path'], prior_offserver_sha256=contract['prior_offserver_sha256'],
        future_completions_excluded=True, parent_authorization='Source preparation only; actual root source review and explicit execution authorization required'))
    members = list(read(H / 'PREPARATION_FILES_SHA256.json')['files']) + ['PREPARATION_FILES_SHA256.json', 'SNAPSHOT.json', 'AUTHORIZED_SNAPSHOT.json']
    save('SOURCE_FILES_SHA256.json', dict(status='BOUND64_SOURCE_ONLY_NOT_EXECUTED_NOT_ADOPTED',
        files={n: dict(sha256=sha(H/n), bytes=(H/n).stat().st_size) for n in members}))
    print(json.dumps(dict(source_files_sha256=sha(H/'SOURCE_FILES_SHA256.json'),
        authorization_sha256=sha(H/'AUTHORIZED_SNAPSHOT.json'), exact_new_ids=contract['remaining18_ids'], execution_authorized=False)))
