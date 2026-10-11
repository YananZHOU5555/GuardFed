"""Read-only text/pointer check. Does not rebuild the candidate or statistics."""
import hashlib
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
read = lambda name: json.loads((HERE / name).read_bytes())
sha = lambda data: hashlib.sha256(data).hexdigest()
pins = read('SOURCE_PINS.json')
for path, expected in pins.items():
    assert sha((ROOT / path).read_bytes()) == expected, path
parent_path = next(path for path in pins if path.endswith('statistics_clarified_20261011.md'))
parent = (ROOT / parent_path).read_bytes()
candidate = (HERE / 'rebuttal_working_F80_FL90_Hybrid20_native11_candidate_20261011.md').read_bytes()
result = read('CHECK_RESULTS.json')
assert sha(candidate) == result['candidate_sha256']
text = candidate.decode('utf-8')
inverse = text
edits = read('REPLACEMENTS.json')
for edit in reversed(edits):
    assert inverse.count(edit['after']) == 1, edit['label']
    inverse = inverse.replace(edit['after'], edit['before'], 1)
assert inverse.encode('utf-8') == parent
titles = lambda raw: [line for line in raw.splitlines(keepends=True) if line.startswith(b'### ') or line.strip() == b'## Associate Editor']
quotes = lambda raw: [line for line in raw.splitlines(keepends=True) if line.startswith(b'>')]
assert titles(candidate) == titles(parent) and len(titles(candidate)) == 24
assert quotes(candidate) == quotes(parent) and len(quotes(candidate)) == 38
resolve = lambda obj, ptr: __import__('functools').reduce(lambda value, key: value[int(key)] if isinstance(value, list) else value[key], ptr.strip('/').split('/'), obj)
points = read('SOURCE_POINTERS.json')
for point in points['displayed_means']:
    value = resolve(json.loads((ROOT / point['path']).read_bytes()), point['pointer'])
    assert value == point['actual_value']
    display = f'{value * 100:.2f}%' if point['metric'] == 'accuracy' else f'{value:.4f}'
    assert display == point['display'] and display in text
base = ROOT / 'outputs/guardfed_tables/celeba_eleven_method_native_20261011'
scenes = json.loads((base / 'summary_statistics.json').read_bytes())
for point in points['saved_scene_mean_directions']:
    gf, fl = resolve(scenes, point['GF_pointer']), resolve(scenes, point['FL_pointer'])
    assert gf == point['GF_saved_mean'] and fl == point['FL_saved_mean'] and gf < fl
agg = json.loads((base / 'seed_first_aggregates.json').read_bytes())
for point in points['retained_tradeoffs']:
    b, g = resolve(agg, point['baseline_pointer']), resolve(agg, point['GF_pointer'])
    assert b == point['baseline_saved_mean'] and g == point['GF_saved_mean']
    assert (b > g) if point['relation_to_GuardFed'] == 'higher' else (b < g)
local = [url for url in re.findall(r'\]\(([^)]+)\)', text) if url.startswith('E:/')]
assert len(local) == result['local_link_occurrences_exist']
assert all(pathlib.Path(url.split('#', 1)[0]).is_file() for url in local)
print(json.dumps({'status': 'PASS_READ_ONLY_TEXT_AND_SAVED_POINTER_CHECK', 'candidate_sha256': sha(candidate), 'reversible_spans': len(edits), 'titles': 24, 'quotes': 38, 'display_pointers': len(points['displayed_means']), 'scene_direction_comparisons': len(points['saved_scene_mean_directions']), 'statistics_recomputed': 0}))
