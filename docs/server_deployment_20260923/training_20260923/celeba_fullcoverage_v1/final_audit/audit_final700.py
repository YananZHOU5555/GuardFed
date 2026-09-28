"""Independent arithmetic and matrix audit; no training or source mutation."""
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
source = BASE / 'acceptance_summary.json'
data = json.loads(source.read_text(encoding='utf-8-sig'))
rows = data['all_conditions']
methods = sorted({r['method'] for r in rows})
distributions = ['IID', 'non-IID']
attacks = ['Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA']
seeds = list(range(91001, 91011))
metrics = ['accuracy', 'aeod', 'aspd', 'score']
subsets = {'all_ten': seeds, 'nonselection_nine': seeds[1:], 'prospective_six': seeds[4:]}
checks = []
def check(name, passed):
    checks.append({'check': name, 'passed': bool(passed)})
    if not passed:
        raise AssertionError(name)
def key(r):
    return (r['method'], r['distribution'], r['attack'], r['seed'])
def stats(values):
    return {'n': len(values), 'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)}
expected = set(itertools.product(methods, distributions, attacks, seeds))
check('700 unique exact method/distribution/attack/seed keys', len(rows) == 700 and len({key(r) for r in rows}) == 700 and {key(r) for r in rows} == expected)
check('644 new and 56 reused', Counter(r['origin'] for r in rows) == {'new': 644, 'reused': 56})
check('Complete accepted700 and no failures or missing', data['complete'] and data['accepted_total'] == 700 and not data['failed_ids'] and not data['missing_ids'])
check('All raw values finite', all(math.isfinite(r[m]) for r in rows for m in metrics))
check('Every record has checkpoint SHA256', all(len(r['checkpoint_sha256']) == 64 for r in rows))
check('Environment counts 686 cu128 and 14 cu130', Counter(r['torch_version'] for r in rows) == {'2.11.0+cu128': 686, '2.11.0+cu130': 14})
check('Manifest digest unchanged', hashlib.sha256((BASE/'manifest.json').read_bytes()).hexdigest() == data['source_manifest_sha256'])
groups = defaultdict(list)
lookup = {}
for r in rows:
    groups[key(r)[:3]].append(r)
    lookup[key(r)] = r
    score = r['accuracy'] - .35*(.45*r['aeod']+.45*r['aspd']+.10*max(r['aeod'],r['aspd']))-.10*max(0,max(r['aeod'],r['aspd'])-.06)
    check('Frozen score '+r['id'], math.isclose(score, r['score'], rel_tol=0, abs_tol=1e-14))
max_error = 0.0
for subset, selected in subsets.items():
    summary = data['summaries'][subset]
    check(subset+' has 70 distinct complete groups', len(summary) == 70 and len({(g['method'],g['distribution'],g['attack']) for g in summary}) == 70)
    for group in summary:
        gkey = (group['method'], group['distribution'], group['attack'])
        rr = [r for r in groups[gkey] if r['seed'] in selected]
        check(subset+' membership '+str(gkey), group['n'] == len(selected) and group['expected_n'] == len(selected) and group['complete'] and group['seeds'] == selected and sorted(r['seed'] for r in rr) == selected)
        for metric in metrics:
            recomputed = stats([r[metric] for r in rr])
            for stat in ['mean', 'sample_sd']:
                error = abs(recomputed[stat] - group[metric][stat])
                max_error = max(max_error, error)
                check(subset+' ddof1 '+str(gkey)+' '+metric+' '+stat, error < 1e-12)
averages = {}
overall = {}
paired = {}
for subset, selected in subsets.items():
    averages[subset] = {}
    overall[subset] = {}
    for method in methods:
        perseed = {seed: {metric: statistics.mean(lookup[(method, dist, attack, seed)][metric] for dist in distributions for attack in attacks) for metric in metrics} for seed in selected}
        averages[subset][method] = perseed
        overall[subset][method] = {metric: stats([perseed[seed][metric] for seed in selected]) for metric in metrics}
    paired[subset] = {}
    for method in methods:
        if method == 'GuardFed-AD2+': continue
        paired[subset][method] = {}
        for metric in metrics:
            deltas = [averages[subset]['GuardFed-AD2+'][seed][metric]-averages[subset][method][seed][metric] for seed in selected]
            wins = sum(v > 0 if metric in ['accuracy','score'] else v < 0 for v in deltas)
            paired[subset][method][metric] = {**stats(deltas), 'guardfed_better_seeds': wins, 'tied_seeds': sum(v == 0 for v in deltas), 'seed_differences': dict(zip(selected, deltas))}
scenario_differences = []
for dist in distributions:
    for attack in attacks:
        g = next(x for x in data['summaries']['all_ten'] if (x['method'],x['distribution'],x['attack']) == ('GuardFed-AD2+',dist,attack))
        for method in methods:
            if method == 'GuardFed-AD2+': continue
            b = next(x for x in data['summaries']['all_ten'] if (x['method'],x['distribution'],x['attack']) == (method,dist,attack))
            scenario_differences.append({'distribution': dist, 'attack': attack, 'baseline': method, 'guardfed_minus_baseline': {m:g[m]['mean']-b[m]['mean'] for m in metrics}})
zeros = Counter(r['method'] for r in rows if r['aeod'] == 0 and r['aspd'] == 0)
report = {'checked_utc':datetime.now(timezone.utc).isoformat(), 'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(), 'scope':'Independent local summary matrix/arithmetic audit, not revalidation of remote raw checkpoints', 'pass':True, 'checks_count':len(checks), 'max_statistic_error':max_error, 'matrix':{'records':700,'groups':70,'seeds_per_group':10}, 'environment_counts':data['environments'], 'overall_seed_first':overall, 'paired_guardfed_minus_baseline':paired, 'scenario_differences':scenario_differences, 'zero_aeod_and_aspd_runs':dict(zeros), 'checks':checks, 'limitations':['valid-only; not untouched test', 'seed91001 used for recipe selection', '14 reused runs cu130; 686 cu128; first-round canaries do not establish 70-round equivalence', 'FairFed/FairGuard/FLTrust+FairGuard are project adaptations', 'Seven methods only; ten manuscript rows, calibration attribution, mechanism ablations and frozen final evaluation remain', 'AEOD is implemented absolute TPR gap, not full equalized odds', 'Zero fairness gap alone cannot establish useful predictions']}
(OUT/'independent_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
lines = ['# CelebA 阶段A：独立验收与结果分析','', '**矩阵与统计检查通过：700条唯一记录，70组均具有完整10个共享种子。** 644条新增与56条复用完整合并；10/9/6种子三个子集的均值与样本标准差（ddof=1）均独立复算通过。', '', '本审计核对本地验收汇总中的键、来源、checkpoint标识及数值计算；服务器原始文件和模型的严格身份验收、增量备份由主流程负责。', '', '以下跨场景结果先对每个种子的两分布×五场景取均值，再跨种子计算均值±样本标准差；独立样本数为10，不能将100个场景记录当100个种子。','', '| 方法 | ACC (%) | AEOD | ASPD |','|---|---:|---:|---:|']
for method in methods:
    s = overall['all_ten'][method]
    lines.append(f"| {method} | {100*s['accuracy']['mean']:.3f} ± {100*s['accuracy']['sample_sd']:.3f} | {s['aeod']['mean']:.5f} ± {s['aeod']['sample_sd']:.5f} | {s['aspd']['mean']:.5f} ± {s['aspd']['sample_sd']:.5f} |")
lines += ['', '结果支持稳定的准确率—公平性权衡：GuardFed相对FLTrust的整体准确率低1.209个百分点，AEOD低0.03618、ASPD低0.05351；十个场景均为同一方向。GuardFed整体AEOD在七方法中最低，但准确率低于FLTrust，ASPD高于FairGuard。FairGuard整体ACC仅74.044%，同时有27次两种差距均为零，应保留并进一步核对预测分布，不能仅凭低差距作公平能力结论。', '', '下表按每场景十种子均值比较，记录GuardFed优于该基线的场景数（分母10）。这些是方向计数，不是显著性结论。', '', '| 基线 | ACC更高 | AEOD更低 | ASPD更低 |', '|---|---:|---:|---:|']
for method in methods:
    if method == 'GuardFed-AD2+': continue
    differences = [r for r in scenario_differences if r['baseline'] == method]
    wins = {metric: sum(r['guardfed_minus_baseline'][metric] > 0 if metric == 'accuracy' else r['guardfed_minus_baseline'][metric] < 0 for r in differences) for metric in ['accuracy', 'aeod', 'aspd']}
    lines.append(f"| {method} | {wins['accuracy']}/10 | {wins['aeod']}/10 | {wins['aspd']}/10 |")
lines += ['', '## GuardFed 与 FLTrust 的配对比较', '', '| 种子子集 | ΔACC (百分点) | ΔAEOD | ΔASPD | ACC/AEOD/ASPD 优于 FLTrust 的种子数 |', '|---|---:|---:|---:|---|']
for subset in subsets:
    p = paired[subset]['FLTrust']
    lines.append(f"| {subset} (n={len(subsets[subset])}) | {100*p['accuracy']['mean']:+.3f} ± {100*p['accuracy']['sample_sd']:.3f} | {p['aeod']['mean']:+.5f} ± {p['aeod']['sample_sd']:.5f} | {p['aspd']['mean']:+.5f} ± {p['aspd']['sample_sd']:.5f} | {p['accuracy']['guardfed_better_seeds']}/{p['aeod']['guardfed_better_seeds']}/{p['aspd']['guardfed_better_seeds']} |")
lines += ['', 'Δ=GuardFed−FLTrust；ACC正值有利，AEOD/ASPD负值有利。此处±为共享种子差值的样本标准差，没有进行显著性检验。', '', '| 分布 | 场景 | ΔACC (百分点) | ΔAEOD | ΔASPD |', '|---|---|---:|---:|---:|']
for item in scenario_differences:
    if item['baseline'] != 'FLTrust': continue
    d = item['guardfed_minus_baseline']
    lines.append(f"| {item['distribution']} | {item['attack']} | {100*d['accuracy']:+.3f} | {d['aeod']:+.5f} | {d['aspd']:+.5f} |")
lines += ['', '## 表述边界与剩余实验', '', '- 完整保留负结果。准确率与公平性分别报告，不能用单一综合评分替代原始三指标，也不能声称所有方法、所有指标全面胜出。', '- 零AEOD/ASPD不自动代表方法更好：需结合ACC和预测分布判断退化。各方法两个公平差距同时为零的单次记录数：'+json.dumps(dict(zeros),ensure_ascii=False)+'。该计数本身不证明恒定预测。', '- 此阶段为valid-only。seed91001参与配置选择；9种子和此前未观察的6种子结果另列。14条旧结果使用cu130，其余686条cu128；首轮迁移一致不能外推为70轮完全等价。', '- FairFed、FairGuard及组合仍须标明项目适配。700条只完成七方法覆盖；剩余十个正文方法的可信集成、共享校准归因、CelebA机制消融、最终冻结评价仍待完成。', '- AEOD在当前实现中是绝对TPR差距，而非完整equalized odds。', '']
(OUT/'analysis_zh.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps({'pass':True,'checks':len(checks),'max_statistic_error':max_error,'zero_gap_runs':dict(zeros),'overall':overall['all_ten'],'guardfed_vs_fltrust':paired['all_ten']['FLTrust']},ensure_ascii=False,indent=2))
