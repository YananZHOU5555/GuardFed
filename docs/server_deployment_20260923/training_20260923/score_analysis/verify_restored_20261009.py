#!/usr/bin/env python3
"""Read-only raw-result audit of the restored Adult score diagnostics.

Writes only this directory. Does not import training code or alter any input.
The original four restored members remain byte-identical to their archive.
"""
import ast
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tarfile

import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
ARCHIVE = OUT.parent / 'revision_verified740_20260923T090026Z.tar.gz'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b, label, atol=1e-12):
    if a is None or b is None:
        require(a is b, label)
    elif isinstance(a, bool) or isinstance(b, bool):
        require(a == b, label)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        require(math.isfinite(a) and math.isfinite(b), label + ': nonfinite')
        require(abs(a - b) <= atol + 1e-12 * max(abs(a), abs(b)), label + ': mismatch')
    else:
        require(a == b, label + ': mismatch')


def csv_value(value):
    if value == '':
        return None
    if value in ('True', 'False'):
        return value == 'True'
    try:
        return float(value)
    except ValueError:
        return value


def robust(values, higher=True, clip=0):
    values = np.asarray(values, dtype=float)
    med = float(np.median(values))
    scale = max(1.4826 * float(np.median(np.abs(values - med))), 1e-6)
    z = (values - med) / scale
    if not higher:
        z = -z
    if clip > 0:
        z = np.clip(z, -clip, clip)
    return z.tolist()


def subset(scores, malicious, indices, temperature, prefix):
    good = [scores[i] for i in indices if not malicious[i]]
    bad = [scores[i] for i in indices if malicious[i]]
    finite = all(math.isfinite(x) for x in good + bad)
    defined = bool(good and bad and finite)
    margin = min(good) - max(bad) if defined else None
    return {
        prefix + '_n': len(indices), prefix + '_benign_n': len(good),
        prefix + '_malicious_n': len(bad),
        prefix + '_min_benign_score': min(good) if good and finite else None,
        prefix + '_max_malicious_score': max(bad) if bad and finite else None,
        prefix + '_strict_margin': margin,
        prefix + '_logit_margin': margin / max(temperature, 1e-6) if defined else None,
        prefix + '_strictly_separated': margin > 0 if defined else None,
        prefix + '_status': 'defined' if defined else (
            'nonfinite_scores' if not finite else 'missing_benign' if not good else 'missing_malicious'),
    }


def positive_mass_bound(h, m, margin, temperature):
    if m == 0:
        return 0.0
    if h == 0 or margin is None or margin < 0:
        return None
    x = math.log(h / m) + margin / max(temperature, 1e-6)
    # Stable evaluation of m/(h*exp(margin/temperature)+m).
    if x >= 0:
        e = math.exp(-x)
        return e / (1 + e)
    return 1 / (1 + math.exp(x))


def group_summary(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['source_kind'], r['distribution'], r['component'])].append(r)
    result = []
    for key, group in sorted(groups.items()):
        g = {'source_kind': key[0], 'distribution': key[1], 'component': key[2],
             'runs': len({r['run_id'] for r in group}), 'rounds': len(group),
             'seeds': sorted({r['seed'] for r in group})}
        for prefix in ('all_pre_gate', 'hard_gate', 'selected'):
            margins = [r[prefix + '_strict_margin'] for r in group
                       if r[prefix + '_strict_margin'] is not None]
            g[prefix] = {'defined': len(margins), 'positive': sum(v > 0 for v in margins),
                         'zero': sum(v == 0 for v in margins), 'negative': sum(v < 0 for v in margins),
                         'missing_malicious': sum(r[prefix + '_malicious_n'] == 0 for r in group),
                         'missing_benign': sum(r[prefix + '_benign_n'] == 0 for r in group),
                         'min': min(margins, default=None), 'max': max(margins, default=None)}
        g['selected_outside_gate_rounds'] = sum(r['selected_outside_hard_gate_n'] > 0 for r in group)
        g['selected_outside_gate_clients'] = sum(r['selected_outside_hard_gate_n'] for r in group)
        g['malicious_mass_positive_rounds'] = sum(r['actual_malicious_weight_sum'] > 0 for r in group)
        g['malicious_mass_max'] = max(r['actual_malicious_weight_sum'] for r in group)
        g['nonvacuous_nonnegative_premise_rounds'] = sum(r['nonvacuous_nonnegative_premise'] for r in group)
        g['nonvacuous_bound_failure_rounds'] = sum(r['nonvacuous_bound_failure'] for r in group)
        g['candidate_counts'] = dict(collections.Counter(r['candidate'] for r in group))
        seed_rows = []
        for seed in g['seeds']:
            sr = [r for r in group if r['seed'] == seed]
            seed_rows.append({'seed': seed, 'observed_rounds': len(sr),
                              'all_pre_gate_positive_fraction': sum(r['all_pre_gate_strict_margin'] > 0 for r in sr) / len(sr),
                              'selected_contains_malicious_fraction': sum(r['selected_malicious_n'] > 0 for r in sr) / len(sr),
                              'malicious_mass_mean': float(np.mean([r['actual_malicious_weight_sum'] for r in sr]))})
        g['per_seed'] = seed_rows
        result.append(g)
    return result


def main():
    receipt = json.loads((OUT / 'restore_receipt.json').read_text(encoding='utf-8'))
    require(sha(ARCHIVE) == receipt['archive_sha256'], 'Archive changed')
    for member in receipt['members']:
        require(sha(Path(member['restored_path'])) == member['sha256'], 'Restored member changed')
    old_summary = json.loads((OUT / 'coverage_summary.json').read_text())
    original_rows = list(csv.DictReader((OUT / 'strict_score_rounds.csv').open(newline='')))
    expected_rows = {(r['source_file'], int(r['round'])): r for r in original_rows}
    require(len(expected_rows) == len(original_rows) == 8440, 'CSV duplicate/coverage')
    frozen = subprocess.check_output(['git', 'show', 'ec419b7:scripts/reproduce_paper_tables.py'], cwd=REPO)
    require(hashlib.sha256(frozen).hexdigest() == old_summary['formula_reference']['source_sha256'], 'Frozen source SHA')
    live = (REPO / 'tmp/celeba_shared_calibration_20260928/reproduce_paper_tables.py').read_text()
    old_functions = {n.name: n for n in ast.parse(frozen.decode()).body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: n for n in ast.parse(live).body if isinstance(n, ast.FunctionDef)}
    ast_checks = {}
    for name in ('softmax_weights', 'weighted_average', 'select_top_scores', 'robust_median_mad'):
        ast_checks[name] = ast.dump(old_functions[name], include_attributes=False) == ast.dump(new_functions[name], include_attributes=False)
    require(all(ast_checks.values()), 'Changed formula AST')
    # Execute only the two small frozen numeric helpers, never a training module.
    namespace = {'np': np, 'math': math}
    for name in ('softmax_weights', 'select_top_scores'):
        exec('from __future__ import annotations\n' + ast.get_source_segment(frozen.decode(), old_functions[name]), namespace)
    frozen_softmax = namespace['softmax_weights']
    frozen_topk = namespace['select_top_scores']
    entries = {f['file']: f for f in old_summary['input_files']}
    require(len(entries) == 140, 'Input count')
    rows, members, wrappers = [], [], []
    max_weights_error = 0.0
    max_independent_math_weights_error = 0.0
    exact_topk_mismatches = []
    checked = collections.Counter()
    with tarfile.open(ARCHIVE, 'r|gz') as archive:
        for member in archive:
            relative = member.name.removeprefix('GuardFed-revision/')
            if relative not in entries:
                continue
            require(member.isfile() and not member.issym() and not member.islnk(), 'Nonregular source')
            payload = archive.extractfile(member).read()
            digest = hashlib.sha256(payload).hexdigest()
            meta = entries[relative]
            require(digest == meta['sha256'], relative + ': source SHA')
            wrapper = json.loads(payload)
            d = wrapper.get('result', wrapper)
            kind = meta['kind']
            require(d['dataset'] == 'adult' and d['attack'] == 'S-DFA' and d['method'] == 'GuardFed-AD2+', 'Outside diagnostic scope')
            require(d['rounds'] == 70 and len(d['trajectory_metrics']) == 70, 'Metric coverage')
            raw_rounds = d['round_summaries']
            round_numbers = [r['round'] for r in raw_rounds]
            require(round_numbers == meta['rounds_present'], 'Round list differs')
            require(set(round_numbers) == (set(range(1, 71)) if kind == 'new_completed' else {1, 70}), 'Round coverage')
            require(len(round_numbers) == len(set(round_numbers)), 'Repeated round')
            require(d['num_clients'] == 20 and d['num_malicious'] == 4, 'Client budget')
            require(len({a['client_id'] for a in d['attack_audit']}) == 20, 'Audit identity')
            if kind == 'historical_full':
                wrappers.append({'member': member.name, 'source': wrapper['source'],
                                 'source_line': wrapper['source_line'], 'source_line_sha256': wrapper['source_line_sha256'],
                                 'result': d})
            for rr in raw_rounds:
                label = relative + ':' + str(rr['round'])
                a = rr['aggregate']
                scores = [float(v) for v in a['trust_scores']]
                require(len(scores) == 20 and all(math.isfinite(v) for v in scores), label + ': finite scores')
                audit = {v['client_id']: bool(v['is_malicious']) for v in d['attack_audit']}
                ids = rr.get('client_ids', list(range(20)))
                require(len(set(ids)) == 20 and set(ids) == set(audit), label + ': IDs')
                malicious = [audit[i] for i in ids]
                require(sum(malicious) == 4, label + ': malicious count')
                if 'malicious_mask' in rr:
                    require([bool(x) for x in rr['malicious_mask']] == malicious, label + ': round malicious mask')
                selected, gate = a['selected_clients'], a['hard_gate_clients']
                for indices in (selected, gate):
                    require(len(indices) == len(set(indices)) and all(isinstance(i, int) and 0 <= i < 20 for i in indices), label + ': subset indices')
                candidates = a['ad2_plus_candidates']
                require(len(candidates) == len({c['name'] for c in candidates}) == 10, label + ': candidates')
                winner = next(c for c in candidates if c['name'] == a['ad2_plus_selected_candidate'])
                temperature = float(winner['config']['act_temperature'])
                require(math.isfinite(temperature) and temperature > 0, label + ': temperature')
                if 'aggregation_temperature' in a:
                    close(a['aggregation_temperature'], temperature, label + ': logged temperature')
                floor = max(c['root_metrics']['accuracy'] for c in candidates) - min(max(0., d['config']['ad2_calibration_max_acc_drop']), .005)
                close(floor, a['ad2_plus_acc_floor'], label + ': accuracy floor')
                candidate_objectives = {}
                for c in candidates:
                    acc, eo, sp = (c['root_metrics'][k] for k in ('accuracy', 'aeod', 'aspd'))
                    loss = .45 * eo + .45 * sp + .10 * max(eo, sp)
                    q = acc - .35 * loss - 6. * max(0., floor - acc) - .10 * max(0., max(eo, sp) - d['config']['ad2_calibration_budget'])
                    close(loss, c['root_fairness_loss'], label + ': candidate fairness')
                    close(q, c['root_selection_score'], label + ': candidate objective')
                    candidate_objectives[c['name']] = (q, -loss, acc)
                feasible = [c for c in candidates if c['root_metrics']['accuracy'] >= floor] or candidates
                expected_candidate = max(feasible, key=lambda c: candidate_objectives[c['name']])
                require(winner['name'] == expected_candidate['name'], label + ': candidate winner')
                close(a['ad2_plus_selected_root_score'], winner['root_selection_score'], label + ': selected root score')
                require(a['ad2_plus_selected_root_metrics'] == winner['root_metrics'], label + ': selected metrics')
                checked['candidate_choices'] += 1
                cfg = dict(d['config']); cfg.update(winner['config'])
                if 'component_raw' in a:
                    zkeys = {'U': 'utility_z', 'C': 'centrality_z', 'A': 'alignment_z', 'F': 'risk_z', 'V': 'violation_z'}
                    for key, zkey in zkeys.items():
                        z = robust(a['component_raw'][key], key not in ('F', 'V'), cfg['ad2_score_clip'])
                        for x, y in zip(z, a[zkey]):
                            close(x, y, label + ': standardization ' + key)
                    distances = [-x for x in a['component_raw']['C']]
                    med = float(np.median(distances)); scale = max(1.4826 * float(np.median(np.abs(np.asarray(distances) - med))), 1e-6)
                    recomputed_gate = [i for i, (dist, alignment) in enumerate(zip(distances, a['component_raw']['A'])) if dist <= med + 2.5 * scale or alignment > 0]
                    if len(recomputed_gate) < max(1, 20 - cfg['num_malicious']):
                        recomputed_gate = list(range(20))
                    require(recomputed_gate == gate, label + ': recomputed gate')
                    checked['gate_reconstructed_from_geometry'] += 1
                terms = []
                for i in range(20):
                    parts = {'U': cfg['ad2_utility_weight'] * a['utility_z'][i],
                             'C': cfg['ad2_centrality_weight'] * a['centrality_z'][i],
                             'A': cfg['ad2_alignment_weight'] * a['alignment_z'][i],
                             'F': cfg['act_risk_weight'] * a['risk_z'][i],
                             'V': cfg['act_violation_weight'] * a['dual_lambda'] * a['violation_z'][i]}
                    component = d['config'].get('ablation_component', 'none')
                    if component in parts:
                        parts[component] = 0.
                    combined = parts['U'] + (parts['C'] + parts['A']) + (parts['F'] + parts['V'])
                    close(combined, scores[i], label + ': additive score')
                    if 'component_contributions' in a:
                        for key, value in parts.items():
                            close(value, a['component_contributions'][i][key], label + ': contribution')
                    terms.append(parts)
                checked['score_reconstruction_rounds'] += 1
                gated_scores = [scores[i] if i in gate else -1e9 for i in range(20)]
                keep_ratio = min(max(cfg['act_keep_ratio'], .05), 1.)
                expected_selected = frozen_topk(gated_scores, keep_ratio, max(1, 20 - cfg['num_malicious']))
                if selected != expected_selected:
                    exact_topk_mismatches.append({'source_file': relative, 'round': rr['round'], 'recorded': selected, 'local_frozen_formula': expected_selected})
                k = max(1, 20 - cfg['num_malicious'], min(20, math.ceil(20 * keep_ratio)))
                require(len(selected) == k, label + ': retained count')
                # Semantic gate accepts any tie ordering at the cutoff; record exact mismatches separately.
                outside = [i for i in range(20) if i not in selected]
                require(not outside or min(gated_scores[i] for i in selected) >= max(gated_scores[i] for i in outside), label + ': top-k order')
                checked['topk_semantic_rounds'] += 1
                u = frozen_softmax([scores[i] for i in selected], temperature)
                weights = [0.] * 20
                for i, value in zip(selected, u):
                    weights[i] = float(value / sum(u))
                stable = [math.exp((scores[i] - max(scores[j] for j in selected)) / max(temperature, 1e-6)) for i in selected]
                alt = [x / sum(stable) for x in stable]
                max_independent_math_weights_error = max(max_independent_math_weights_error, max(abs(weights[i] - v) for i, v in zip(selected, alt)))
                if 'client_weights' in a:
                    raw_weights = a['client_weights']
                    require(len(raw_weights) == 20 and all(math.isfinite(w) and w >= 0 for w in raw_weights), label + ': logged weights')
                    error = max(abs(x - y) for x, y in zip(raw_weights, weights))
                    max_weights_error = max(max_weights_error, error)
                    require(error <= 1e-12, label + ': weight formula')
                    weights = raw_weights
                    checked['direct_logged_weight_rounds'] += 1
                else:
                    error = None
                    checked['historical_reconstructed_weight_rounds'] += 1
                close(sum(weights), 1., label + ': normalized weight')
                require(all(weights[i] == 0 for i in range(20) if i not in selected), label + ': unselected zero weight')
                row = {'source_kind': kind, 'source_file': relative, 'source_sha256': digest,
                       'run_id': d['run_id'], 'dataset': d['dataset'], 'distribution': d['distribution'],
                       'component': component, 'seed': d['seed'], 'round': rr['round'], 'expected_rounds': 70,
                       'candidate': winner['name'], 'temperature': temperature,
                       'identity_source': 'round_log_verified_against_attack_audit' if 'client_ids' in rr else 'attack_audit_with_frozen_range_order',
                       'selected_outside_hard_gate_n': len(set(selected) - set(gate)),
                       'weights_source': 'direct_logged' if 'client_weights' in a else 'reconstructed_frozen_softmax',
                       'actual_malicious_weight_sum': sum(w for w, flag in zip(weights, malicious) if flag),
                       'weights_total': sum(weights), 'reconstruction_max_abs_error': error,
                       'formula_check': ('pass' if error <= 1e-12 else 'FAIL') if error is not None else 'frozen_formula_candidate_temperature'}
                for prefix, indices in (('all_pre_gate', list(range(20))), ('hard_gate', gate), ('selected', selected)):
                    row.update(subset(scores, malicious, indices, temperature, prefix))
                archived = expected_rows.pop((relative, rr['round']))
                for key, value in archived.items():
                    close(row[key], csv_value(value), label + ': CSV field ' + key)
                bound = positive_mass_bound(row['selected_benign_n'], row['selected_malicious_n'], row['selected_strict_margin'], temperature)
                row['nonvacuous_nonnegative_premise'] = row['selected_benign_n'] > 0 and row['selected_malicious_n'] > 0 and row['selected_strict_margin'] >= 0
                row['conditional_mass_bound'] = bound
                row['nonvacuous_bound_failure'] = row['nonvacuous_nonnegative_premise'] and row['actual_malicious_weight_sum'] > bound + 1e-12
                require(not row['nonvacuous_bound_failure'], label + ': conditional mass bound')
                rows.append(row)
            members.append({'member': member.name, 'sha256': digest, 'bytes': len(payload), 'kind': kind,
                            'run_id': d['run_id'], 'seed': d['seed'], 'rounds': round_numbers})
    require(len(members) == 140 and len(rows) == 8440 and not expected_rows, 'Raw/CSV full join')
    # Independently reconnect the 20 reused wrappers to their original source lines.
    historical_path = REPO / 'results/attack_strength/raw_results.jsonl'
    requested = {w['source_line']: w for w in wrappers}
    historical_matches = []
    with historical_path.open('rb') as stream:
        for number, line in enumerate(stream, 1):
            if number not in requested:
                continue
            w = requested.pop(number)
            variants = {'bytes_with_line_ending': line, 'bytes_strip_line_ending': line.rstrip(b'\r\n'), 'bytes_strip_whitespace': line.strip()}
            modes = [mode for mode, blob in variants.items() if hashlib.sha256(blob).hexdigest() == w['source_line_sha256']]
            require(modes, 'Historical source-line hash mismatch')
            original = json.loads(line)
            # JSON canonicalization preserves legacy NaN strings on both sides.
            require(json.dumps(original, sort_keys=True) == json.dumps(w['result'], sort_keys=True), 'Historical result body mismatch')
            historical_matches.append({'wrapper_member': w['member'], 'source_line': number,
                                       'source_line_sha256': w['source_line_sha256'], 'hash_matching_modes': modes,
                                       'result_body_equal': True})
    require(len(historical_matches) == 20 and not requested, 'Historical line coverage')
    groups = group_summary(rows)
    for old in old_summary['groups']:
        g = next(g for g in groups if (g['source_kind'], g['distribution'], g['component']) == (old['source_kind'], old['distribution'], old['component']))
        checks = {'completed_run_n': g['runs'], 'seeds': g['seeds'], 'round_records': g['rounds'],
                  'strict_margin_defined_n': g['all_pre_gate']['defined'], 'strict_positive_n': g['all_pre_gate']['positive'],
                  'strict_zero_n': g['all_pre_gate']['zero'], 'strict_negative_n': g['all_pre_gate']['negative'],
                  'strict_margin_min': g['all_pre_gate']['min'], 'strict_margin_max': g['all_pre_gate']['max'],
                  'selected_no_malicious_n': g['selected']['missing_malicious'],
                  'selected_outside_hard_gate_round_n': g['selected_outside_gate_rounds'],
                  'selected_strict_margin_defined_n': g['selected']['defined'],
                  'malicious_weight_defined_n': g['rounds'], 'malicious_weight_positive_n': g['malicious_mass_positive_rounds'],
                  'malicious_weight_max': g['malicious_mass_max']}
        for key, value in checks.items():
            close(value, old[key], 'Group summary ' + str((g['source_kind'], g['distribution'], g['component'])) + ': ' + key)
    totals = []
    for kind in ('new_completed', 'historical_full'):
        cohort = [r for r in rows if r['source_kind'] == kind]
        totals.append({'source_kind': kind, 'runs': len({r['run_id'] for r in cohort}), 'rounds': len(cohort),
                       'pre_gate_positive': sum(r['all_pre_gate_strict_margin'] > 0 for r in cohort),
                       'pre_gate_zero': sum(r['all_pre_gate_strict_margin'] == 0 for r in cohort),
                       'pre_gate_negative': sum(r['all_pre_gate_strict_margin'] < 0 for r in cohort),
                       'selected_no_malicious': sum(r['selected_malicious_n'] == 0 for r in cohort),
                       'selected_no_benign': sum(r['selected_benign_n'] == 0 for r in cohort),
                       'selected_both_classes': sum(r['selected_status'] == 'defined' for r in cohort),
                       'selected_margin_positive': sum(r['selected_strict_margin'] > 0 for r in cohort if r['selected_strict_margin'] is not None),
                       'selected_margin_zero': sum(r['selected_strict_margin'] == 0 for r in cohort if r['selected_strict_margin'] is not None),
                       'selected_margin_negative': sum(r['selected_strict_margin'] < 0 for r in cohort if r['selected_strict_margin'] is not None),
                       'selected_outside_gate_rounds': sum(r['selected_outside_hard_gate_n'] > 0 for r in cohort),
                       'nonvacuous_nonnegative_premise': sum(r['nonvacuous_nonnegative_premise'] for r in cohort),
                       'nonvacuous_bound_failures': sum(r['nonvacuous_bound_failure'] for r in cohort),
                       'malicious_mass_positive': sum(r['actual_malicious_weight_sum'] > 0 for r in cohort),
                       'malicious_mass_max': max(r['actual_malicious_weight_sum'] for r in cohort)})
    rows.sort(key=lambda r: (r['source_file'], r['round']))
    with (OUT / 'independent_round_checks_20261009.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    audit = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'PASS',
             'archive_sha256': receipt['archive_sha256'], 'raw_input_files_verified': len(members),
             'raw_input_files_by_kind': dict(collections.Counter(m['kind'] for m in members)),
             'csv_round_rows_verified': len(rows), 'csv_field_count': len(original_rows[0]),
             'csv_fields_compared': len(rows) * len(original_rows[0]), 'archived_group_summaries_verified': len(groups),
             'comparison_numeric_tolerance': 'absolute 1e-12 plus relative 1e-12; categorical/identity fields exact',
             'all_raw_source_hashes_match': True, 'all_csv_rows_match_recomputed_raw': True,
             'all_archived_summary_groups_match': True, 'frozen_source_sha256': hashlib.sha256(frozen).hexdigest(),
             'frozen_source_commit': 'ec419b7', 'live_formula_helper_ast_checks': ast_checks,
             'numpy_local_version': np.__version__, 'checks': dict(checked),
             'direct_logged_weight_max_abs_error': max_weights_error,
             'independent_scalar_math_weight_max_abs_error': max_independent_math_weights_error,
             'exact_topk_local_mismatches': exact_topk_mismatches,
             'historical_source_path': str(historical_path), 'historical_source_sha256': sha(historical_path),
             'historical_original_line_checks': historical_matches, 'groups': groups, 'cohort_totals': totals, 'source_members': members,
             'limitations': [
                 'Only Adult S-DFA ablation diagnostics: 120 deletion runs and 20 historical Full wrappers.',
                 'Historical Full logs only rounds 1 and 70: 40 observed/1400 possible rounds, no intermediate reconstruction.',
                 'Geometry-derived hard-gate reconstruction possible for 8400 new rounds only; historical 40 lack raw distance/alignment inputs.',
                 'Candidate winner is checked against logged candidate root metrics; this does not independently re-evaluate candidate checkpoints on root data.',
                 'Scalar score generation is checked against logged standardized/raw components; raw client update tensors and norm transformations are not independently replayed by this audit.',
                 'Historical 40 weights are reconstructed, not directly logged. Their source bodies match original historical JSONL lines.',
                 'Conditional coefficient-mass checks do not establish accuracy, fairness, convergence or success of adaptive selection/calibration.',
                 'Round records are serially correlated. Descriptive counts are not independent statistical sample sizes.'
             ], 'training_started': False, 'original_results_modified': False}
    (OUT / 'independent_acceptance_20261009.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: audit[k] for k in ('status', 'raw_input_files_verified', 'csv_round_rows_verified', 'csv_fields_compared', 'checks', 'direct_logged_weight_max_abs_error', 'independent_scalar_math_weight_max_abs_error', 'exact_topk_local_mismatches')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
