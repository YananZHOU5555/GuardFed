"""Adopt only the frozen validation-search recipe; preserve all original records."""
from pathlib import Path
import datetime
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tmp/celeba_flgmm_final6_closure_20261009'
SUMMARY = BASE / 'summary32_final/SUMMARY32.json'
INDEPENDENT = ROOT / 'tmp/celeba_flgmm_summary32_root_independent_20261009'
OUTPUT = BASE / 'ROOT_SUMMARY_ADOPTION.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())

assert not OUTPUT.exists()
assert sha(SUMMARY) == 'd8f32441402f0ce7870bacc34e94de2a334133b2881d0a59ccf5123ef0781291'
assert sha(INDEPENDENT / 'ROOT_INDEPENDENT_REVIEW.json') == '4b25891a75ddfd99c356c23dbc6422610ea217204a27ebc44dd89c343da959e4'
assert sha(INDEPENDENT / 'FILES_SHA256.json') == '14326cef49bed537f9552fe787d4bb3e7ddecab95d1bb0ba64c499f20dff59cb'
for name, pin in read(INDEPENDENT / 'FILES_SHA256.json')['files'].items():
    assert sha(INDEPENDENT / name) == pin['sha256']
independent = read(INDEPENDENT / 'ROOT_INDEPENDENT_REVIEW.json')
assert independent['status'] == 'PASS' and independent['accepted_records'] == 32
assert independent['maximum_abs_numeric_difference'] == 0 and independent['original26_records_exact'] and independent['final6_ids_exact']
assert independent['legacy_reader']['nonlegacy_missing_host_refused'] and independent['legacy_reader']['current_missing_host_refused']
summary = read(SUMMARY)
assert summary['accepted'] == 32 and summary['seed_n'] == 1 and not summary['recipe_adopted']
assert not summary['sample_SD_reported'] and not summary['significance_claimed'] and not summary['final_test']
manifest = {r['id']: r for r in read(BASE / 'manifest.json')['jobs']}
protocol = read(BASE / 'protocol.json')
rows = summary['records']
assert len(rows) == len({r['id'] for r in rows}) == len(manifest) == 32
assert {r['id'] for r in rows} == set(manifest)
for row in rows:
    item = manifest[row['id']]
    assert (row['candidate'], row['distribution'], row['attack'], row['seed'], row['rounds']) == (item['tuning_candidate'], item['distribution'], item['attack'], 91001, 70)
    assert row['job_sha256'] == item['job_sha256'] and row['source_hashes'] == protocol['source_hashes']
    m = row['metrics']; gap = max(m['aeod'], m['aspd'])
    actual = m['accuracy'] - .35 * (.45 * m['aeod'] + .45 * m['aspd'] + .10 * gap) - .10 * max(0, gap - .06)
    assert row['score'] == actual
calculated = []
for candidate in sorted({r['candidate'] for r in rows}):
    group = [r for r in rows if r['candidate'] == candidate]
    assert len(group) == 4 and {(r['distribution'], r['attack']) for r in group} == {(d, a) for d in ('IID', 'non-IID') for a in ('Benign', 'S-DFA')}
    calculated.append(dict(candidate=candidate, n_seeds=1, **{key: statistics.mean(r[key] for r in group) for key in ('accuracy', 'aeod', 'aspd', 'score')}))
assert calculated == summary['candidates']
ranked = sorted(calculated, key=lambda r: (-r['score'], r['candidate']))
selected = ranked[0]
assert selected == summary['selected_per_method']['FLGMM-author-code'] == independent['frozen_score_leader']
recipe = next(r for r in protocol['candidates'] if r['id'] == selected['candidate'])
assert recipe == summary['selected_candidate_recipe']
champion = min(calculated, key=lambda r: (-r['accuracy'], r['candidate']))
assert champion == summary['accuracy_champion'] == independent['accuracy_champion']
def dominates(a, b):
    return a['accuracy'] >= b['accuracy'] and a['aeod'] <= b['aeod'] and a['aspd'] <= b['aspd'] and any(a[k] != b[k] for k in ('accuracy', 'aeod', 'aspd'))
pareto = [r for r in calculated if not any(dominates(other, r) for other in calculated)]
assert pareto == summary['three_metric_pareto'] == independent['three_metric_pareto']
proof = dict(status='ROOT_FROZEN_VALIDATION32_RECIPE_SUMMARY_ADOPTED', checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    accepted_records=32, candidates=8, condition_scores_checked=32, candidate_mean_scalars_checked=32,
    summary_path=SUMMARY.relative_to(ROOT).as_posix(), summary_sha256=sha(SUMMARY),
    independent_review_path=(INDEPENDENT / 'ROOT_INDEPENDENT_REVIEW.json').relative_to(ROOT).as_posix(),
    independent_review_sha256=sha(INDEPENDENT / 'ROOT_INDEPENDENT_REVIEW.json'), independent_seal_sha256=sha(INDEPENDENT / 'FILES_SHA256.json'),
    final32_root_proof_sha256=summary['source_bindings']['final_root_proof_sha256'], selected_recipe=recipe,
    selected_four_condition_mean=selected, accuracy_champion=champion, three_metric_pareto=pareto,
    score_gap_to_second=selected['score'] - ranked[1]['score'], exact_tie=False,
    selection_rule='Frozen per-condition score, then four-condition mean, exact ties candidate lexical order',
    seed_n=1, sample_SD_reported=False, significance_claimed=False, all_candidates_retained=True,
    original_summary_unchanged=True, legacy_missing_host_field_fabricated=False,
    recipe_selected_for_next_valid_coverage=True, formal100_binding_or_execution=False,
    new_inference=0, final_test=False, scientific_goal_complete=False,
    limitation='Exposed validation seed91001; four conditions are not independent seeds. Near score tie does not establish stable superiority. Author-code FLGMM adaptations and zero-SD extension remain disclosed. Formal coverage requires separate source-bound binding and actual canaries.')
with OUTPUT.open('x', encoding='utf8', newline='\n') as stream:
    json.dump(proof, stream, indent=2, allow_nan=False); stream.write('\n')
print(json.dumps(dict(status=proof['status'], proof_sha256=sha(OUTPUT), selected_recipe=recipe, score_gap=proof['score_gap_to_second'], formal100_started=False)))
