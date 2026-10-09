#!/usr/bin/env python3
"""Offline conditional score/softmax diagnostics; reads only completed E1 results."""
import argparse
import ast
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FROZEN = 'ec419b7'


def formula_reference():
    source = subprocess.check_output(['git', 'show', FROZEN + ':scripts/reproduce_paper_tables.py'], cwd=ROOT, text=True)
    live = (ROOT / 'scripts/reproduce_paper_tables.py').read_text()
    def functions(text):
        return {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    old, current = functions(source), functions(live)
    names = ['softmax_weights', 'weighted_average']
    identical = all(ast.dump(old[n], include_attributes=False) == ast.dump(current[n], include_attributes=False) for n in names)
    namespace = {'np': np}
    exec('from __future__ import annotations\n' + ast.get_source_segment(source, old['softmax_weights']), namespace)
    return namespace['softmax_weights'], {'frozen_commit': FROZEN, 'current_functions_ast_identical': identical,
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'formula': 'w_i=exp(score_i/max(T,1e-6)-max_selected_logit)/sum_selected(exp(...)); unselected weights=0',
        'scope': 'Weights multiply norm-scaled updates, not raw updates; no claim about final influence magnitude.'}


def subset_diagnostics(scores, mask, subset, temperature, prefix):
    benign = [scores[i] for i in subset if not mask[i]]
    malicious = [scores[i] for i in subset if mask[i]]
    finite = all(math.isfinite(x) for x in benign + malicious)
    valid = bool(benign and malicious and finite)
    margin = min(benign) - max(malicious) if valid else None
    status = 'defined' if valid else ('nonfinite_scores' if not finite else 'missing_benign' if not benign else 'missing_malicious')
    return {prefix + '_n': len(subset), prefix + '_benign_n': len(benign), prefix + '_malicious_n': len(malicious),
        prefix + '_min_benign_score': min(benign) if benign and finite else None,
        prefix + '_max_malicious_score': max(malicious) if malicious and finite else None,
        prefix + '_strict_margin': margin,
        prefix + '_logit_margin': margin / max(temperature, 1e-6) if valid and temperature is not None else None,
        prefix + '_strictly_separated': margin > 0 if valid else None,
        prefix + '_status': status}


def analyze_round(d, r, kind, path, digest, softmax, formula_ok):
    info = r['aggregate']; scores = [float(x) for x in info['trust_scores']]; n = len(scores)
    ids = r.get('client_ids', list(range(n)))
    audit = {a['client_id']: bool(a['is_malicious']) for a in d['attack_audit']}
    assert len(ids) == n and len(set(ids)) == n and set(ids) == set(audit), 'Missing or ambiguous client identity'
    mask = r.get('malicious_mask', [audit[i] for i in ids])
    assert len(mask) == n and all(bool(v) == audit[i] for i, v in zip(ids, mask)), 'Malicious labels disagree'
    selected = info['selected_clients']; gate = info['hard_gate_clients']
    assert len(set(selected)) == len(selected) and all(0 <= i < n for i in selected), 'Invalid selected indices'
    assert len(set(gate)) == len(gate) and all(0 <= i < n for i in gate), 'Invalid gate indices'
    candidate = info.get('ad2_plus_selected_candidate')
    candidate_rows = [c for c in info.get('ad2_plus_candidates', []) if c['name'] == candidate]
    temperature = info.get('aggregation_temperature')
    if temperature is None and len(candidate_rows) == 1:
        temperature = candidate_rows[0]['config'].get('act_temperature')
    if temperature is not None:
        temperature = float(temperature)
        assert math.isfinite(temperature), 'Nonfinite temperature'
    row = {'source_kind': kind, 'source_file': str(path.relative_to(ROOT)), 'source_sha256': digest,
        'run_id': d['run_id'], 'dataset': d['dataset'], 'distribution': d['distribution'],
        'component': d['config'].get('ablation_component', 'none'), 'seed': d['seed'], 'round': r['round'],
        'expected_rounds': d['rounds'], 'candidate': candidate, 'temperature': temperature,
        'identity_source': 'round_log_verified_against_attack_audit' if 'client_ids' in r else 'attack_audit_with_frozen_range_order',
        'selected_outside_hard_gate_n': len(set(selected) - set(gate)),
        'weights_source': 'unavailable', 'actual_malicious_weight_sum': None,
        'weights_total': None, 'reconstruction_max_abs_error': None, 'formula_check': None}
    for prefix, subset in [('all_pre_gate', list(range(n))), ('hard_gate', gate), ('selected', selected)]:
        row.update(subset_diagnostics(scores, mask, subset, temperature, prefix))
    reconstructed = None
    if formula_ok and temperature is not None and selected and all(math.isfinite(s) for s in scores):
        unnormalized = softmax([scores[i] for i in selected], temperature)
        reconstructed = [0.0] * n
        for i, w in zip(selected, unnormalized):
            reconstructed[i] = float(w / sum(unnormalized))
    weights = info.get('client_weights')
    if weights is not None:
        row['weights_source'] = 'direct_logged'
        assert len(weights) == n and all(math.isfinite(w) and w >= 0 for w in weights), 'Invalid weights'
        assert abs(sum(weights) - 1) <= 1e-12, 'Weights not normalized'
        assert all(weights[i] == 0 for i in range(n) if i not in selected), 'Weight outside selection'
        if reconstructed is not None:
            error = max(abs(a-b) for a,b in zip(weights, reconstructed))
            row['reconstruction_max_abs_error'] = error
            row['formula_check'] = 'pass' if error <= 1e-12 else 'FAIL'
    elif reconstructed is not None:
        weights = reconstructed; row['weights_source'] = 'reconstructed_frozen_softmax'
        row['formula_check'] = 'frozen_formula_candidate_temperature'
    if weights is not None:
        row['weights_total'] = sum(weights)
        row['actual_malicious_weight_sum'] = sum(w for w, malicious in zip(weights, mask) if malicious)
    return row


def summarize(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['source_kind'], r['distribution'], r['component'])].append(r)
    output = []
    for key, group in sorted(groups.items()):
        runs = {r['run_id']: r['expected_rounds'] for r in group}
        valid = [r for r in group if r['all_pre_gate_status'] == 'defined']
        masses = [r['actual_malicious_weight_sum'] for r in group if r['actual_malicious_weight_sum'] is not None]
        output.append({'source_kind': key[0], 'distribution': key[1], 'component': key[2], 'completed_run_n': len(runs),
            'seeds': sorted({r['seed'] for r in group}), 'round_records': len(group),
            'expected_rounds_for_completed_runs': sum(runs.values()), 'strict_margin_defined_n': len(valid),
            'strict_positive_n': sum(r['all_pre_gate_strict_margin'] > 0 for r in valid),
            'strict_zero_n': sum(r['all_pre_gate_strict_margin'] == 0 for r in valid),
            'strict_negative_n': sum(r['all_pre_gate_strict_margin'] < 0 for r in valid),
            'strict_margin_min': min((r['all_pre_gate_strict_margin'] for r in valid), default=None),
            'strict_margin_max': max((r['all_pre_gate_strict_margin'] for r in valid), default=None),
            'selected_no_malicious_n': sum(r['selected_malicious_n'] == 0 for r in group),
            'selected_outside_hard_gate_round_n': sum(r['selected_outside_hard_gate_n'] > 0 for r in group),
            'selected_strict_margin_defined_n': sum(r['selected_status'] == 'defined' for r in group),
            'malicious_weight_defined_n': len(masses), 'malicious_weight_positive_n': sum(m > 0 for m in masses),
            'malicious_weight_max': max(masses, default=None),
            'weight_sources': dict(collections.Counter(r['weights_source'] for r in group))})
    return output


def self_check():
    diagnostic = subset_diagnostics([1., 2., 3.], [True, False, False], [0,1,2], .5, 'x')
    assert diagnostic['x_strict_margin'] == 1 and diagnostic['x_logit_margin'] == 2
    negative = subset_diagnostics([4., 2., 3.], [True, False, False], [0,1,2], .5, 'x')
    assert negative['x_strict_margin'] == -2 and not negative['x_strictly_separated']
    empty = subset_diagnostics([4., 2., 3.], [True, False, False], [1,2], .5, 'x')
    assert empty['x_strict_margin'] is None and empty['x_status'] == 'missing_malicious'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'results/revision_20260923/adult_ablation_v1')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True); self_check()
    softmax, formula = formula_reference(); rows=[]; files=[]; rejected=[]
    snapshot = [('new_completed', p) for p in sorted(args.input.glob('runs/*/result.json'))]
    snapshot += [('historical_full', p) for p in sorted(args.input.glob('reused_full/*.json'))]
    for kind, path in snapshot:
        try:
            payload = path.read_bytes(); wrapper=json.loads(payload); d=wrapper.get('result', wrapper)
            assert d['dataset'] == 'adult' and d['attack'] == 'S-DFA', 'Outside Adult E1 scope'
            assert len(d['trajectory_metrics']) == d['rounds'], 'Incomplete metric trajectory'
            rounds=d['round_summaries']; assert len({r['round'] for r in rounds}) == len(rounds), 'Duplicate rounds'
            digest=hashlib.sha256(payload).hexdigest()
            current=[analyze_round(d,r,kind,path,digest,softmax,formula['current_functions_ast_identical']) for r in rounds]
            if kind == 'new_completed': assert {r['round'] for r in current} == set(range(1,d['rounds']+1)), 'Missing new diagnostics'
            rows.extend(current);files.append({'file':str(path.relative_to(ROOT)), 'sha256':digest, 'kind':kind, 'rounds_present':[r['round'] for r in rounds]})
        except Exception as exc:
            rejected.append({'file':str(path), 'error':f'{type(exc).__name__}: {exc}'})
    errors=[r for r in rows if r['formula_check'] == 'FAIL']
    # If direct new logs contradict the formula, never retain inferred old weights.
    if errors:
        for r in rows:
            if r['weights_source'] == 'reconstructed_frozen_softmax':
                r.update(weights_source='unavailable_formula_validation_failed', actual_malicious_weight_sum=None, weights_total=None)
    checks=[r['reconstruction_max_abs_error'] for r in rows if r['reconstruction_max_abs_error'] is not None]
    summary={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'input_directory':str(args.input), 'snapshot_input_file_n':len(snapshot),'accepted_completed_run_n':len(files),
        'round_record_n':len(rows), 'rejected':rejected, 'formula_reference':formula,
        'direct_weight_crosscheck_round_n':len(checks), 'direct_weight_crosscheck_max_abs_error':max(checks,default=None),
        'direct_weight_crosscheck_failure_n':len(errors), 'groups':summarize(rows), 'input_files':files,
        'interpretation':[
            'Strict margin=min(benign trust_scores)-max(malicious trust_scores); every negative and zero margin is retained.',
            'trust_scores is the additive score from normalized components; logit margin additionally divides by the chosen candidate temperature. No extra score normalization is invented.',
            'All-pre-gate, hard-gate and final selected sets are reported separately. Empty-class margins are undefined, not positive separation.',
            'The implementation can select gate-excluded clients when the top-k count exceeds gate size, then applies softmax to original scores. Therefore selected need not be a subset of hard_gate; selected_outside_hard_gate_n explicitly records this existing behavior.',
            'Historical Full contains only rounds 1 and 70. Its missing rounds are neither imputed nor treated as failed/successful separation.',
            'Historical weights are reconstructed only from stored scores, selected indices and chosen-candidate temperature, using the unchanged frozen softmax and normalization formula.',
            'Malicious weight is on norm-scaled updates; it is not total adversarial influence, nor a model accuracy/fairness guarantee.',
            'These are conditional softmax diagnostics, not a complete robustness guarantee. Round observations are correlated and are not independent statistical replicates.',
            'Coverage is completed files at script start, not all planned experiments. Re-run after more runs complete.']}
    with (args.output/'strict_score_rounds.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []);writer.writeheader();writer.writerows(rows)
    (args.output/'coverage_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# Strict score separation: offline diagnostics','',f"Snapshot: {summary['created_utc']}",
        f"Accepted completed runs: {len(files)}; available round records: {len(rows)}; rejected files: {len(rejected)}.",
        f"Direct-weight formula checks: {len(checks)} rounds; maximum absolute difference: {max(checks,default=None)}; failures: {len(errors)}.",'',
        '| Source | Distribution | Component | Runs | Rounds available / expected | Positive / zero / negative margin | Selected with no malicious | Max malicious weight |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for g in summary['groups']:
        lines.append(f"| {g['source_kind']} | {g['distribution']} | {g['component']} | {g['completed_run_n']} | {g['round_records']} / {g['expected_rounds_for_completed_runs']} | {g['strict_positive_n']} / {g['strict_zero_n']} / {g['strict_negative_n']} | {g['selected_no_malicious_n']} | {g['malicious_weight_max']} |")
    lines+=['']+['- '+x for x in summary['interpretation']]
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ['accepted_completed_run_n','round_record_n','direct_weight_crosscheck_round_n','direct_weight_crosscheck_max_abs_error','direct_weight_crosscheck_failure_n','rejected','groups']},indent=2))
    if errors or rejected: raise SystemExit(1)


if __name__ == '__main__':
    main()
