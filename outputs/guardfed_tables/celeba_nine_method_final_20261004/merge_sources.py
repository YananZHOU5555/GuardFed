"""Merge two strictly accepted, immutable validation snapshots; never train."""
from collections import Counter
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

def read(name):
    raw = (OUT / name).read_bytes()
    return json.loads(raw.decode('utf-8-sig')), hashlib.sha256(raw).hexdigest()

old, old_sha = read('source_stageA700_snapshot.json')
new, new_sha = read('source_baseline200_snapshot.json')
audit, audit_sha = read('identity_evidence.json')
assert old['complete'] and old['accepted_total'] == 700
assert new['complete'] and new['accepted'] == 200
assert not old['missing_ids'] and not old['failed_ids']
assert not new['invalid_records'] and not new['preserved_failures']
assert audit['pass']
assert audit['source_snapshots']['old700']['sha256'] == old_sha
assert audit['source_snapshots']['new200']['sha256'] == new_sha

names = {'FedAA-DDPG-adapted-v1': 'FedAA-DDPG', 'LASA-official': 'LASA'}
identities = audit['atomic_identity_environment']
assert set(identities) == {r['id'] for r in new['records']}
records = [dict(r) for r in old['all_conditions']]
for original in new['records']:
    r = dict(original)
    proof = identities[r['id']]
    assert proof['off_server_verified']
    assert proof['rounds'] == 70 and proof['split'] == 'valid'
    assert proof['n_train'] == 162770 and proof['n_eval'] == 19867
    assert proof['actual_alpha'] == (5000.0 if r['distribution'] == 'IID' else 5.0)
    assert proof['checkpoint_sha256'] == r['checkpoint_sha256']
    assert all(proof['raw_metrics'][k] == r[k] for k in ('accuracy', 'aeod', 'aspd'))
    r['source_method'] = r['method']
    r['method'] = names[r['source_method']]
    r['torch_version'] = proof['torch_version']
    r['origin'] = 'baseline_fullcoverage_reused' if r['reused'] else 'baseline_fullcoverage_new'
    r['identity_evidence'] = 'identity_evidence.json:atomic_identity_environment:' + r['id']
    records.append(r)

keys = {(r['method'], r['distribution'], r['attack'], r['seed']) for r in records}
assert len(records) == len(keys) == 900
assert len({r['id'] for r in records}) == 900
assert Counter(r['torch_version'] for r in records) == audit['environment_counts']
subsets = [('all_ten', 'ten_seed', list(range(91001, 91011))),
           ('nonselection_nine', 'exclude_selection_nine_seed', list(range(91002, 91011))),
           ('matching_six', 'matching_six_seed', list(range(91005, 91011)))]
summaries = {}
for old_label, new_label, seeds in subsets:
    old_key = 'prospective_six' if old_label == 'matching_six' else old_label
    summaries[old_label] = [dict(g) for g in old['summaries'][old_key]]
    for g in new['cohorts'][new_label]['cells']:
        assert g['complete'] and g['n'] == len(seeds)
        method = 'FedAA-DDPG' if g['algorithm'] == 'FedAA' else 'LASA'
        summaries[old_label].append(dict(method=method, distribution=g['distribution'],
                                        attack=g['attack'], n=g['n'], seeds=seeds,
                                        **g['metrics']))
    assert len(summaries[old_label]) == 90

data = dict(checked_utc=audit['checked_utc'], complete=True, accepted_total=900,
            accepted_new=836, reused_verified=64, planned_total=900,
            missing_ids=[], failed_ids=[], all_conditions=records, summaries=summaries,
            environments=dict(Counter(r['torch_version'] for r in records)),
            source_manifest_sha256=dict(StageA=old['source_manifest_sha256'],
                                       baseline_fullcoverage=new['manifest_sha256']),
            source_snapshots=dict(StageA700=dict(file='source_stageA700_snapshot.json', sha256=old_sha),
                                  baseline200=dict(file='source_baseline200_snapshot.json', sha256=new_sha),
                                  identity_evidence=dict(file='identity_evidence.json', sha256=audit_sha)),
            historical_failures=new['historical_failures'],
            note='Nine adapted/native implementations; validation only; fixed recipes; '
                 'new200 identity fields come from verified raw result metadata; '
                 '10/9/6 shared-seed summaries; all outcomes retained.')
(OUT / 'source_acceptance_snapshot.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(records=900, groups=90, environments=data['environments'], sources=data['source_snapshots'])))
