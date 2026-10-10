"""Bounded source-only checks. No Git, network, bulk writes, or scientific execution."""
from pathlib import Path
import ast, copy, json
import publish_increment43 as pub

HERE = Path(__file__).resolve().parent
for name in ['publish_increment43.py', 'verify_increment43.py', 'check_prepared.py']:
    ast.parse((HERE / name).read_text(encoding='utf8'))
spec = json.loads((HERE / 'CURATED_INPUTS_DRAFT.json').read_bytes())
try:
    pub.plan(spec)
except AssertionError as e:
    assert str(e) == 'Missing actual binding: remaining620_startup', repr(e)
else:
    raise AssertionError('Draft must refuse missing actual startup')
# In-memory metadata only: never an actual startup or accepted result.
seal = json.dumps({'files': {}}).encode()
fact = json.dumps({'fixture_ready': True, 'cumulative_new': 188, 'three_view_accepted_unchanged': 180,
                  'final_test': False, 'accepted_total': 22, 'accepted_new': 4,
                  'celeba_mechanism_v1': {'scientific_results_strictly_accepted': 188,
                  'three_view_new_models_accepted': 180}}).encode()
original = pub.source
pub.source = lambda name, allowed: (None, seal if name.endswith('FILES_SHA256.json') else fact)
fixture = copy.deepcopy(spec)
fixture['files'] = ['tmp/fixture/a.json']
fixture['sealed_sources'] = [dict(directory='tmp/' + n, sha256=pub.sha(seal)) for n in sorted(pub.SOURCES)]
fixture['bindings'] = {r: dict(path='tmp/fixture/a.json', sha256=pub.sha(fact), expect={'/fixture_ready': True}) for r in pub.ROLES}
for role in ['native188', 'FL22', 'current_state']:
    fixture['bindings'][role]['expect'].update(spec['bindings'][role]['expect'])
assert pub.plan(fixture)['total_bytes'] > 0
for field, value in [('sha256', '0' * 64), ('expect', {'/fixture_ready': False})]:
    bad = copy.deepcopy(fixture); bad['bindings']['native188'][field] = value
    try:
        pub.plan(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError(field)
pub.source = original
for path in ['tmp/x/../z.json', 'tmp/x/__pycache__/z.py', 'tmp/x/model.pt',
             'tmp/x/data.npy', 'tmp/x/bulk.tar.gz', 'tmp/x/attempt001/a.json']:
    try:
        pub.relative(path)
    except AssertionError:
        pass
    else:
        raise AssertionError(path)
result = dict(status='SOURCE_ONLY_STATIC_AND_BOUNDED_BYTE_CHECKS_PASS_NO_STAGE', syntax_files=3,
    actual_source_seals=5, actual_source_members=280,
    missing_actual_620_startup_refused=True, wrong_SHA_and_wrong_root_fact_refused=True,
    bulk_traversal_runtime_path_refusals=6, synthetic_metadata_fixture_only=True,
    no_Git_commands_or_bulk_writes_called=True, source_packages_modified=False,
    actual_stage_commit_push_or_remote_blob_verification=False)
# The 85+91+57+29+18 count is measured from the actual five seals, not a result count.
result['actual_source_members'] = sum(len(pub.read(pub.ROOT / e['directory'] / 'FILES_SHA256.json')['files']) for e in spec['sealed_sources'])
print(json.dumps(result, indent=2))
