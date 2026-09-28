"""Reproduce the completed validation cohort and prespecified seed subsets."""
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev
import hashlib
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / 'docs/server_deployment_20260923/training_20260923/celeba_fullcoverage_v1/acceptance_summary.json'
snapshot = OUT / 'source_acceptance_snapshot.json'
raw = snapshot.read_bytes() if snapshot.exists() else SOURCE.read_bytes()
data = json.loads(raw.decode('utf-8-sig'))
if not snapshot.exists():
    snapshot.write_bytes(raw)
methods = ['FedAvg', 'FairFed', 'Median', 'FLTrust', 'FairGuard', 'FLTrust+FairGuard', 'GuardFed-AD2+']
categories = ['Vanilla FL', 'Fairness-aware', 'Robust FL', 'Robust FL', 'Robust + fair', 'Robust + fair', 'Ours']
dists = ['IID', 'non-IID']
attacks = ['Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA']
metrics = [('accuracy', 'ACC (%) ↑', 100, 2), ('aeod', 'AEOD ↓', 1, 4), ('aspd', 'ASPD ↓', 1, 4)]
adapted = {'FairFed', 'FairGuard', 'FLTrust+FairGuard'}
groups = defaultdict(list)
keys = set()
for r in data['all_conditions']:
    key = (r['method'], r['distribution'], r['attack'])
    identity = key + (r['seed'],)
    assert identity not in keys
    keys.add(identity)
    assert r['checkpoint_sha256'] and 91001 <= r['seed'] <= 91010
    assert all(math.isfinite(r[k]) and 0 <= r[k] <= 1 for k, _, _, _ in metrics)
    groups[key].append(r)
assert len(keys) == data['accepted_total'] and len(groups) == 70
common = sorted(set.intersection(*(set(r['seed'] for r in rows) for rows in groups.values())))
assert data['complete'] and data['accepted_total'] == 700 and not data['missing_ids'] and not data['failed_ids']
assert common == list(range(91001, 91011))
assert all(len(rows) == 10 for rows in groups.values())
verified_pairs = 0
for subset, summaries in data['summaries'].items():
    for g in summaries:
        rows = [r for r in groups[g['method'], g['distribution'], g['attack']] if r['seed'] in g['seeds']]
        assert len(rows) == g['n']
        for metric, _, _, _ in metrics:
            values = [r[metric] for r in rows]
            assert math.isclose(mean(values), g[metric]['mean'], abs_tol=1e-12)
            if len(values) > 1:
                assert math.isclose(stdev(values), g[metric]['sample_sd'], abs_tol=1e-12)
            verified_pairs += 1

def values_for(method, dist, attack, seeds, formal=False):
    rows = groups[method, dist, attack]
    if formal and {r['seed'] for r in rows} != set(range(91001, 91011)):
        return [f'Pending ({len(rows)}/10)'] * 3
    rows = [r for r in rows if r['seed'] in seeds]
    assert len(rows) == len(seeds)
    return [f'{mean(r[k] for r in rows)*scale:.{precision}f} ± {stdev(r[k] for r in rows)*scale:.{precision}f}'
            for k, _, scale, precision in metrics]

notes = [
    'Round 70; validation only (19,867 images). Mean ± sample SD (ddof = 1); ACC in %, gaps on [0, 1].',
    'AEOD denotes the implemented absolute TPR gap. No significance claim follows from mean ranks alone.',
    '* Project adaptations. GuardFed includes training-root group calibration; baseline outputs are uncalibrated.',
    'Recipes were selected on non-IID Benign/S-DFA, including seed 91001; this is coverage transfer, not IID-specific tuning.',
    'The source cohort includes 14 reused seed-91001 records on cu130; all other records use cu128. No 70-round equivalence claim.',
    'Stage A covers seven implementations; ten further manuscript baselines and mechanism controls remain outstanding.'
]
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})

def render(name, distributions, seeds, formal=False, images=True):
    conditions = [(d, a) for d in distributions for a in attacks]
    title = 'CelebA: ' + ' / '.join(distributions)
    status = ('TEN-SEED COMPLETION TEMPLATE - incomplete cells withheld' if formal else
              f'COMPLETED VALIDATION COHORT - {len(seeds)} shared seeds ({seeds[0]}-{seeds[-1]})')
    headers = ['Category', 'Method', 'Metric'] + [(d + ': ' if len(distributions) > 1 else '') + a.replace('F Flip', 'F-Flip') for d, a in conditions]
    md = ['# ' + title, '', status + '.', '', f"Snapshot: {data['checked_utc']}; {data['accepted_total']}/700 accepted records.", '', '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    tex = [r'% Requires multirow and graphicx.', r'\begin{table*}[t]', r'\centering',
           r'\caption{' + title + '. ' + status.replace('-', '--') + r'. Validation-only; mean $\pm$ sample SD at round 70.}',
           r'\label{tab:' + name.replace('_', '-') + '}', r'\setlength{\tabcolsep}{3pt}', r'\resizebox{\textwidth}{!}{%',
           r'\begin{tabular}{l|l|c|' + '|'.join(['ccccc'] * len(distributions)) + '}', r'\hline\hline',
           ' & '.join(headers).replace('↑', r'$\uparrow$').replace('↓', r'$\downarrow$') + r' \\', r'\hline']
    if images:
        wide = len(conditions) == 10
        fig, ax = plt.subplots(figsize=(23 if wide else 15.7, 11.3))
        fig.subplots_adjust(left=.018, right=.985, top=.985, bottom=.015)
        ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
        ax.text(.5, .985, title, ha='center', va='top', fontsize=16)
        ax.text(.5, .95, status, ha='center', va='top', fontsize=11.5)
        ax.text(.5, .925, f"Accepted snapshot: {data['accepted_total']}/700 records, {data['checked_utc'][:19]} UTC", ha='center', va='top', fontsize=10)
        left = .30 if wide else .35
        bounds = [0, .105 if wide else .125, .25 if wide else .29, left] + [left + (1-left)*i/len(conditions) for i in range(1, len(conditions)+1)]
        centers = [(a+b)/2 for a,b in zip(bounds, bounds[1:])]
        top, hh, rh = .885, .034, .026
        body = top - 2*hh; bottom = body - 21*rh
        ax.hlines([top+.005, top, top-hh, body, bottom, bottom-.005], 0, 1, color='black', linewidth=.65)
        ax.text(left/2, top-hh/2, 'Data distribution', ha='center', va='center', fontsize=11)
        for di, dist in enumerate(distributions):
            start = bounds[3+5*di]; stop = bounds[3+5*(di+1)]
            ax.text((start+stop)/2, top-hh/2, dist + (' (α = 5000)' if dist == 'IID' else ' (α = 5)'), ha='center', va='center', fontsize=11)
        for x, h in zip(centers, ['Category', 'Method', 'Metric'] + [a.replace('F Flip', 'F-Flip') for _,a in conditions]):
            ax.text(x, top-1.5*hh, h, ha='center', va='center', fontsize=10.5)
        for x in bounds[1:-1]: ax.vlines(x, bottom, top if x in [bounds[3], bounds[8] if wide else -1] else top-hh, color='black', linewidth=.4)
    for i, (method, category) in enumerate(zip(methods, categories)):
        display = method + ('*' if method in adapted else '')
        values = [values_for(method, d, a, seeds, formal) for d,a in conditions]
        if images:
            y = body-3*i*rh
            if i: ax.hlines(y, 0, 1, color='black', linewidth=.5)
            ax.text(centers[0], y-1.5*rh, category, ha='center', va='center', fontsize=10)
            ax.text(centers[1], y-1.5*rh, display, ha='center', va='center', fontsize=10, fontweight='bold' if method == 'GuardFed-AD2+' else 'normal')
        if i: tex.append(r'\hline')
        for k, (_, metric_label, _, _) in enumerate(metrics):
            cells = [category if k == 0 else '', display if k == 0 else '', metric_label] + [v[k] for v in values]
            md.append('| ' + ' | '.join(cells) + ' |')
            if images:
                for x, text in zip(centers[2:], cells[2:]): ax.text(x, y-(k+.5)*rh, text, ha='center', va='center', fontsize=9.7)
            category_tex = r'\multirow{3}{*}{' + category + '}' if k == 0 else ''
            method_tex = r'\multirow{3}{*}{' + display.replace('*', r'$^{*}$') + '}' if k == 0 else ''
            label_tex = metric_label.replace('%', r'\%').replace('↑', r'$\uparrow$').replace('↓', r'$\downarrow$')
            nums = [('$' + v[k].replace(' ± ', r' \pm ') + '$') if '±' in v[k] else v[k] for v in values]
            tex.append(' & '.join([category_tex, method_tex, label_tex] + nums) + r' \\')
    table_notes = notes.copy()
    if not formal:
        used_rows = [r for r in data['all_conditions'] if r['distribution'] in distributions and r['seed'] in seeds]
        n130 = sum('cu130' in r['torch_version'] for r in used_rows)
        table_notes[4] = (f'This table uses {len(used_rows)-n130} cu128 and {n130} cu130 records; the source cohort contains 14 cu130 records. No 70-round equivalence claim.' if n130 else
                          'All records in this table use torch 2.11/cu128. The source cohort contains 14 cu130 records, excluded from this table.')
    local_notes = [('Pending cells have fewer than ten accepted seeds; no partial-seed values are substituted.' if formal else
                    f'All 70 method/distribution/scenario cells use the same {len(seeds)} seeds. Seed subsets follow the frozen protocol; all cells within each table use identical seeds.')] + notes
    local_notes[1:] = table_notes
    md += ['', *[n + '  ' for n in local_notes]]
    tex += [r'\hline\hline', r'\end{tabular}}', r'\par\smallskip', r'\begin{minipage}{\textwidth}\scriptsize'] + [n.replace('±', r'$\pm$').replace('%', r'\%').replace('* Project', r'$^{*}$ Project') + r'\par' for n in local_notes] + [r'\end{minipage}', r'\end{table*}']
    (OUT / (name+'.md')).write_text('\n'.join(md)+'\n', encoding='utf-8')
    (OUT / (name+'.tex')).write_text('\n'.join(tex)+'\n', encoding='utf-8')
    if images:
        for j,note in enumerate(local_notes): ax.text(0, bottom-.025-j*.026, note, va='top', fontsize=9.5)
        fig.savefig(OUT / (name+'.png'), dpi=190, facecolor='white')
        fig.savefig(OUT / (name+'.pdf'), facecolor='white')
        plt.close(fig)

for dist in dists:
    suffix = dist.lower().replace('-', '')
    render('celeba_'+suffix+'_ten_seed', [dist], common)
render('celeba_iid_noniid_ten_seed', dists, common)
for label, seeds in [('exclude_selection', [s for s in common if s != 91001]), ('prospective', [s for s in common if s >= 91005])]:
    render('celeba_iid_noniid_'+label, dists, seeds, images=False)

coverage = ['# Accepted coverage (not a performance ranking)', '', f"{data['accepted_total']}/700 records accepted; common fully paired seeds: {common}.", '', '| Distribution | Method | ' + ' | '.join(attacks) + ' |', '| --- | --- | ' + ' | '.join(['---']*5) + ' |']
for dist in dists:
    for method in methods:
        coverage.append('| ' + ' | '.join([dist, method]+[str(len(groups[method,dist,a]))+'/10' for a in attacks])+' |')
(OUT/'coverage.md').write_text('\n'.join(coverage)+'\n', encoding='utf-8')
provenance = {'source_path': str(SOURCE), 'source_sha256': hashlib.sha256(raw).hexdigest(), 'source_checked_utc': data['checked_utc'],
              'manifest_sha256': data['source_manifest_sha256'], 'accepted_total': data['accepted_total'], 'common_seeds': common,
              'main_table_records_used': len(common)*70, 'accepted_records_preserved_in_snapshot': len(keys),
              'summary_mean_sd_pairs_verified': verified_pairs, 'complete_ten_seed_groups': sum(len(v)==10 for v in groups.values()),
              'source_environments': data['environments'], 'statistics': 'mean and sample SD, ddof=1; all metrics from each accepted final checkpoint',
              'limitations': notes}
(OUT/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
print(json.dumps(provenance, ensure_ascii=True, indent=2))

# Average the ten scenarios within each seed before estimating between-seed spread.
paired_summary = {}
for subset, seeds in [('ten', range(91001, 91011)), ('nonselection_nine', range(91002, 91011)), ('prospective_six', range(91005, 91011))]:
    per_method = {}
    seed_values = {}
    for method in methods:
        seed_values[method] = {s: {k: mean(r[k] for r in data['all_conditions'] if r['method'] == method and r['seed'] == s)
                                  for k, _, _, _ in metrics} for s in seeds}
        per_method[method] = {k: {'mean': mean(v[k] for v in seed_values[method].values()),
                                  'sample_sd': stdev(v[k] for v in seed_values[method].values())}
                              for k, _, _, _ in metrics}
    paired = {}
    for k, _, _, _ in metrics:
        differences = [seed_values['GuardFed-AD2+'][s][k] - seed_values['FLTrust'][s][k] for s in seeds]
        paired[k] = {'GF_minus_FLTrust_mean': mean(differences), 'sample_sd': stdev(differences),
                     'GF_better_seeds': sum(x > 0 if k == 'accuracy' else x < 0 for x in differences), 'n': len(seeds)}
    paired_summary[subset] = {'methods': per_method, 'paired_GuardFed_vs_FLTrust': paired}
(OUT/'seed_paired_summary.json').write_text(json.dumps(paired_summary, indent=2)+'\n', encoding='utf-8')
