"""Build and seal the C50 table only after the actual exact-three replay adoption."""
from pathlib import Path
import argparse, datetime, hashlib, json, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'tmp/celeba_mechanism_three_view_C_five_scenes_prepared_20261010'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--adoption', type=Path, required=True)
parser.add_argument('--sha256', required=True)
args = parser.parse_args()
adoption = args.adoption.resolve()
assert not sys.flags.optimize and sha(adoption) == args.sha256
assert adoption.parent.parent == ROOT/'tmp/celeba_mechanism_valid_C_after47_20261010/execution_candidate/backups'
proof = read(adoption)
assert proof['status'] == 'ROOT_C_AFTER47_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
assert (proof['prior_three_view_models'],proof['accepted_new'],proof['cumulative_three_view_models']) == (147,3,150)
assert proof['original147_unchanged'] and proof['all_native_differences_zero'] and not proof['test_inference']
assert sha(BASE/'FILES_SHA256.json') == '06e58da21084ebc2b1c83d73cd10a1f01b9a6d4eed45baea6e99d4a3c4fd242f'
for row in read(BASE/'FILES_SHA256.json')['files']:
    assert sha(BASE/row['path']) == row['sha256'] and (BASE/row['path']).stat().st_size == row['bytes']
assert not (BASE/'snapshot').exists() and not (BASE/'ACTUAL_HANDOFF.json').exists()
build = subprocess.run([sys.executable,'-B',str(BASE/'build.py'),'--C3-adoption',str(adoption),'--C3-adoption-sha256',args.sha256,'--output',str(BASE/'snapshot')],capture_output=True)
(BASE/'ROOT_BUILD_STDOUT.log').write_bytes(build.stdout)
(BASE/'ROOT_BUILD_STDERR.log').write_bytes(build.stderr)
assert build.returncode == 0, 'Preserve the failed attempt; inspect evidence before recovery'
verify = subprocess.run([sys.executable,'-B',str(BASE/'verify_numeric.py'),'--snapshot',str(BASE/'snapshot')],capture_output=True)
(BASE/'ROOT_NUMERIC_STDOUT.log').write_bytes(verify.stdout)
(BASE/'ROOT_NUMERIC_STDERR.log').write_bytes(verify.stderr)
assert verify.returncode == 0, 'Preserve the failed numeric check'
handoff = dict(status='C50_ACTUAL_BUILT_PENDING_INDEPENDENT_ROOT_REVIEW',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_prepared_seal_sha256=sha(BASE/'FILES_SHA256.json'),actual_C3_adoption_path=adoption.relative_to(ROOT).as_posix(),
    actual_C3_adoption_sha256=sha(adoption),unique_records=100,Full=50,minus_C=50,complete_scenes=5,
    scene_statistics=810,display_cells=405,count_metrics=900,cross_scene_seed_first_statistics=162,
    snapshot_files={p.name:sha(p) for p in (BASE/'snapshot').iterdir() if p.is_file()},
    old80_records_unchanged=True,old648_statistics_unchanged=True,old324_display_cells_unchanged=True,
    inference=0,training=0,test=False,canonical_modified=False,negative_results_preserved=True)
with (BASE/'ACTUAL_HANDOFF.json').open('x',encoding='utf8') as stream:
    json.dump(handoff,stream,indent=2);stream.write('\n')
members = {}
for path in BASE.rglob('*'):
    if path.is_file() and '__pycache__' not in path.parts and path.name != 'ACTUAL_FILES_SHA256.json':
        members[path.relative_to(BASE).as_posix()] = dict(sha256=sha(path),bytes=path.stat().st_size)
with (BASE/'ACTUAL_FILES_SHA256.json').open('x',encoding='utf8') as stream:
    json.dump(dict(files=members),stream,indent=2);stream.write('\n')
print(json.dumps(dict(actual_handoff_sha256=sha(BASE/'ACTUAL_HANDOFF.json'),actual_seal_sha256=sha(BASE/'ACTUAL_FILES_SHA256.json'),members=len(members),canonical_modified=False)))
