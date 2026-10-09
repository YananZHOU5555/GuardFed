#!/usr/bin/env python3
"""Audit pre-attack CelebA partitions without images or attribute metadata.

Reads 20 restored GuardFed-AD2+ Sp-DFA results and the frozen source.
Writes only this directory; never imports the training module.
"""
import ast
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import tarfile

import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
STAGE = OUT.parent / 'celeba_fullcoverage_v1'


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def approx(a, b, label):
    require(math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-12, label)


def reconstruct_sensitive_counts(group_lengths, alpha, seed, clients=20):
    """Replay only the frozen int64 shuffle and Dirichlet split operations."""
    rng = np.random.default_rng(seed)
    counts = {}
    for group in [1, 0]:
        indices = np.arange(group_lengths[group], dtype=np.int64)
        rng.shuffle(indices)
        cuts = (np.cumsum(rng.dirichlet([alpha] * clients)) * len(indices)).astype(int)[:-1]
        counts[group] = np.diff(np.concatenate([[0], cuts, [len(indices)]])).tolist()
    return counts


def shuffle_content_independence(length, seed):
    one, two = np.random.default_rng(seed), np.random.default_rng(seed)
    a = np.arange(length, dtype=np.int64)
    b = np.arange(length, dtype=np.int64) * 7 + 31
    one.shuffle(a)
    two.shuffle(b)
    require(one.bit_generator.state == two.bit_generator.state, 'Shuffle depends on int64 values')
    require(np.array_equal(b, a * 7 + 31), 'Shuffle swap pattern differs by values')


def stat(values):
    a = np.asarray(values, dtype=float)
    constant = bool(np.all(a == a[0]))
    return {'n_seeds': len(a), 'mean': float(a[0] if constant else a.mean()), 'sample_sd': 0. if constant else float(a.std(ddof=1)),
            'min': float(a.min()), 'max': float(a.max())}


def dump_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_report(acceptance, partitions, root_unique):
    def mean_sd(s, percent=False, decimals=3):
        scale = 100 if percent else 1
        return f"{scale*s['mean']:.{decimals}f}±{scale*s['sample_sd']:.{decimals}f}"
    summary = acceptance['partition_summaries_across_10_seeds']
    rstats = acceptance['root_summary_across_10_unique_seeds']
    lines = [
        '# CelebA实际分区与root支持审计', '',
        '2026-10-09。**20份分区、400个client样本总数逐个精确匹配；实际敏感组分配已经恢复，未运行新训练。** 本审计取Stage A中GuardFed-AD2+的Sp-DFA、IID/non-IID、seed91001–91010共20份70轮原结果。它报告的是攻击前的真实分区，而非运行时被改写的敏感注释。', '',
        '## 证据与精确重建', '',
        f"- [acceptance.json](acceptance.json)记录20个member及原inventory SHA/bytes、{len(acceptance['archives'])}份增量归档SHA、20个冻结job SHA、源码与AST/helper SHA，以及全部门检。原result只安全恢复到本目录raw_results，未恢复/改动模型或旧结果。",
        '- [audit_partitions.py](audit_partitions.py)可复现统计；[冻结函数摘录](frozen_source_function_excerpts.txt)保留create_client_data_dict、load、root和runtime的具体源码。core与CelebA loader的完整SHA均匹配Stage A冻结manifest。',
        '- 由global_group_counts减root的敏感组计数得客户端池：Male=0共85058、Male=1共61435，合计146493。对每个seed重新建立default_rng，按敏感组1、0顺序，先shuffle同长度int64数组，再dirichlet/cumsum/floor切分。shuffle的随机状态只与数组长度/类型有关；40个同长度不同取值的精确RNG状态及swap-pattern检验通过。无需图像或属性metadata。',
        '- 400个重建样本总数与原image_data_contract.client_sample_counts完全一致，也与attack_audit.samples一致。另40个原Sp-DFA属性攻击客户端的fflip_changed与重建Male=1数量精确一致：冻结all_unprivileged模式将敏感注释设0，因此该计数提供额外的原始组数交叉核验。',
        '- load_bundle/load_celeba_bundle及分区函数不接收attack参数，所读config也不依赖攻击模式或experiment tag；run_experiment先load_bundle，再调用client_runtime_data。root/分区先于攻击，且敏感数组在runtime中复制后才改写。相同seed/alpha/配置的攻击前分区由此不依赖attack选择；这不等于另行提取了所有方法/场景的原文件。', '',
        '## 跨seed实际分布', '',
        '每个分区先在20个client内计算描述量，再对10个共享seed报告mean±sampleSD(ddof=1)。以下n=10，不能把400个client或两个分布中重复的root当成独立seed。百分比的SD单位为百分点。', '',
        '| 分区描述量 | IID（α=5000） | non-IID（α=5） |',
        '|---|---:|---:|',
    ]
    specifications = [
        ('每seed最小client样本数', 'client_sample_min', False, 1),
        ('每seed最大client样本数', 'client_sample_max', False, 1),
        ('20client样本数CV（populationSD/mean）', 'client_sample_cv', False, 4),
        ('每seed最小client男组比例（%）', 'client_Male1_fraction_min', True, 3),
        ('每seed最大client男组比例（%）', 'client_Male1_fraction_max', True, 3),
        ('20client男组比例populationSD（百分点）', 'client_Male1_fraction_population_sd', True, 3),
        ('前4恶意client覆盖客户端池样本（%）', 'first4_malicious_sample_fraction', True, 3),
        ('前4覆盖客户端池Male=0样本（%）', 'first4_malicious_Male0_coverage_fraction', True, 3),
        ('前4覆盖客户端池Male=1样本（%）', 'first4_malicious_Male1_coverage_fraction', True, 3),
    ]
    for name, key, percent, decimals in specifications:
        lines.append(f"| {name} | {mean_sd(summary['IID'][key], percent, decimals)} | {mean_sd(summary['non-IID'][key], percent, decimals)} |")
    lines += ['',
        '全部200个IID client观测的男组比例范围为40.495%–42.969%，non-IID为9.832%–74.303%；样本数分别为7126–7532、2926–15638。400个client中没有缺Male=0或Male=1的情况。非IID下前4恶意client的样本覆盖率范围为15.173%–24.661%，所以“4/20恶意client”不是每个seed恰好20%的样本覆盖。不能把这些敏感组统计扩展为未经恢复的client标签联合异质性。', '',
        '## 原20分区明细', '',
        '[per_partition_20.csv](per_partition_20.csv)给出完整数值/来源SHA；[per_client_400.csv](per_client_400.csv)给出每个client的两组计数。下表root四格顺序为Male0/Smiling0、0/1、1/0、1/1。', '',
        '| 分布 | seed | client样本min–max | client男组% min–max | 前4样本覆盖% | root四格支持 |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for r in partitions:
        cells = '/'.join(str(r[key]) for key in ('root_0_0', 'root_0_1', 'root_1_0', 'root_1_1'))
        lines.append(f"| {r['distribution']} | {r['seed']} | {r['client_sample_min']}–{r['client_sample_max']} | {100*r['client_Male1_fraction_min']:.3f}–{100*r['client_Male1_fraction_max']:.3f} | {100*r['first4_malicious_sample_fraction']:.3f} | {cells} |")
    lines += ['',
        '## root四格与代表性', '',
        '每个root有16277图像；敏感组计数Male=0为9451、Male=1为6826，均由敏感组分层抽样固定。训练总体四格为43688/50821/41002/27259。相同seed在两分布中的root image-id SHA、四格和代表性指标完全相同，因此下表及统计使用10个独立seed，而非重复的20份。', '',
        '| root描述量 | 10seed mean±sampleSD | 所有seed范围 |',
        '|---|---:|---:|',
        f"| 四格中最小支持数 | {mean_sd(rstats['root_joint_min_support'], decimals=1)} | {int(rstats['root_joint_min_support']['min'])}–{int(rstats['root_joint_min_support']['max'])} |",
        f"| sensitive×label联合TVD | {mean_sd(rstats['root_joint_tvd'], decimals=6)} | {rstats['root_joint_tvd']['min']:.6f}–{rstats['root_joint_tvd']['max']:.6f} |",
        f"| label边际TVD | {mean_sd(rstats['root_label_tvd'], decimals=6)} | {rstats['root_label_tvd']['min']:.6f}–{rstats['root_label_tvd']['max']:.6f} |",
        f"| sensitive边际TVD | {mean_sd(rstats['root_sensitive_tvd'], decimals=9)} | 固定 {rstats['root_sensitive_tvd']['mean']:.9f} |",
        f"| root Smiling=1比例（%） | {mean_sd(rstats['root_Smiling1_fraction'], True, 3)} | {100*rstats['root_Smiling1_fraction']['min']:.3f}–{100*rstats['root_Smiling1_fraction']['max']:.3f} |", '',
        '四格均有支持，最小2666；这些是已测root分层抽样的代表性描述，不证明任意root具有同样支持，也不消除真实部署获取root的隐私/可用性假设。所有已存group/sensitive/label TVD均从计数重算并在绝对1e-12内吻合。10个root原始明细见 [root_support_10_unique_seeds.csv](root_support_10_unique_seeds.csv)。', '',
        '## 不能恢复的部分', '',
        '**没有恢复每client的Smiling标签数或Male×Smiling四格。** 总体/root四格和client样本数不足以唯一确定client标签联合计数；没有加载metadata，不能据此虚构标签分布、子群TPR/FPR分母或预测指标。当前审计只关闭CelebA敏感组分区及root支持这部分P4；Adult/COMPAS强异质性实测分区及历史synthetic图实现链仍需另外证据。', '',
        '## Manuscript paragraph candidate', '',
        'We audited the realized pre-attack CelebA partitions, rather than interpreting heterogeneity solely from the Dirichlet parameter. Replaying the frozen sensitive-group split for 20 partitions (two distributions and ten shared seeds) exactly matched all 400 archived client sample counts. Across the 200 client observations per distribution, the Male=1 fraction ranged from 40.495% to 42.969% under IID and from 9.832% to 74.303% under non-IID; no client lacked either sensitive group. Across seeds, the within-partition sample-count CV was 0.00946±0.00129 versus 0.29870±0.06038. The first four nominally malicious clients covered 19.964±0.102% versus 18.874±3.132% of client-held examples. The 16,277-example root had nonzero support in all four sensitive/label cells (minimum 2,666), with joint total-variation distance from training-population proportions 0.003486±0.002155 across ten unique root seeds. These are sensitive-allocation and root-support diagnostics; client label-by-sensitive counts were not recovered and are not inferred from alpha.', '',
        '未访问服务器、未训练、未更改旧结果或冻结协议；归档、源码、helper和产物SHA均可追溯。',
    ]
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    manifest_path = STAGE / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    snapshot_path = REPO / 'outputs/guardfed_tables/celeba_nine_method_final_20261004/source_stageA700_snapshot.json'
    snapshot = json.loads(snapshot_path.read_text())
    require(file_sha(manifest_path) == snapshot['source_manifest_sha256'], 'Stage A manifest snapshot mismatch')
    accepted = {r['id']: r for r in snapshot['all_conditions']}
    sources_path = OUT / 'restored_sources_with_inventories.json'
    sources = json.loads(sources_path.read_text(encoding='utf-8'))
    require(len(sources) == 20, 'Expected exactly 20 sources')
    archives = {}
    for record in sources:
        require(file_sha(Path(record['output'])) == record['sha256'] == record['inventory_entry']['sha256'], 'Restored/member/inventory SHA mismatch')
        require(Path(record['output']).stat().st_size == record['bytes'] == record['inventory_entry']['bytes'], 'Member byte count')
        archives.setdefault(record['archive'], {'archive': record['archive'], 'sha256': file_sha(Path(record['archive']))})
    latest = json.loads((STAGE / 'latest_backup.json').read_text())
    require(archives[str((STAGE / Path(latest['archive']).name).resolve())]['sha256'] == latest['sha256'], 'Latest archive receipt SHA')
    prior_name = Path(latest['prior_archive']).name
    require(archives[str((STAGE / prior_name).resolve())]['sha256'] == latest['prior_sha256'], 'Prior archive receipt SHA')
    # Verify the frozen jobs within their original archives, without restoring them.
    job_checks = []
    for archive_path in archives:
        selected_sources = [r for r in sources if r['archive'] == archive_path]
        with tarfile.open(archive_path, 'r:gz') as archive:
            for record in selected_sources:
                run_id = Path(record['output']).stem
                job_member = f'results/revision_20260926/celeba_fullcoverage_v1/jobs/{run_id}.json'
                job_blob = archive.extractfile(job_member).read()
                job_sha = hashlib.sha256(job_blob).hexdigest()
                remote = '/workspace/GuardFed-celeba-expanded/' + job_member
                require(job_sha == manifest['job_sha256s'][remote], 'Frozen job SHA mismatch')
                raw = json.loads(Path(record['output']).read_text())
                job = json.loads(job_blob)
                require(job['config'] == raw['config'], 'Frozen job config differs from result')
                require(job['source_hashes'] == raw['revision_job']['source_hashes'], 'Frozen job source differs from result')
                job_checks.append({'id': run_id, 'member': job_member, 'sha256': job_sha})
    source_files = {
        'scripts/reproduce_paper_tables.py': REPO / 'tmp/celeba_shared_calibration_20260928/reproduce_paper_tables.py',
        'src/celeba_data.py': REPO / 'tmp/celeba_shared_calibration_20260928/celeba_data.py',
    }
    pinned = {}
    snippets, ast_records = [], []
    node_map = {}
    wanted = {'create_client_data_dict', 'sample_server_dataframe', 'apply_root_noise', 'load_bundle',
              'run_experiment', 'client_runtime_data', 'load_celeba_bundle'}
    for name, path in source_files.items():
        digest = file_sha(path)
        require(digest == manifest['source_hashes'][name], 'Frozen code SHA mismatch')
        pinned[name] = {'path': str(path), 'sha256': digest}
        text = path.read_text()
        for node in ast.parse(text).body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                segment = ast.get_source_segment(text, node)
                node_map[node.name] = node
                ast_records.append({'file': name, 'function': node.name, 'line': node.lineno,
                                    'end_line': node.end_lineno,
                                    'source_sha256': hashlib.sha256(segment.encode()).hexdigest(),
                                    'ast_sha256': hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()})
                snippets.append(f'# {name}:{node.lineno} [{digest}]\n{segment}')
    require(set(node_map) == wanted, 'Missing required source functions')
    create = node_map['create_client_data_dict']
    loop = next(n for n in create.body if isinstance(n, ast.For))
    require(ast.literal_eval(loop.iter) == [1, 0], 'Sensitive group order differs')
    calls = [(n.lineno, ast.unparse(n.func)) for n in ast.walk(create) if isinstance(n, ast.Call)]
    shuffle_line = next(line for line, name in calls if name == 'rng.shuffle')
    dirichlet_line = next(line for line, name in calls if name == 'rng.dirichlet')
    require(shuffle_line < dirichlet_line, 'Dirichlet/shuffle RNG order differs')
    run = node_map['run_experiment']
    run_calls = [(n.lineno, ast.unparse(n.func)) for n in ast.walk(run) if isinstance(n, ast.Call)]
    load_line = next(line for line, name in run_calls if name == 'load_bundle')
    attack_line = next(line for line, name in run_calls if name == 'client_runtime_data')
    require(load_line < attack_line, 'Attack applied before split')
    forbidden_config = {'fflip_mode', 'foe_mode', 'sdfa_foe_mode', 'spdfa_foe_mode', 'experiment_tag', 'experiment_suite'}
    for fn in ('load_bundle', 'load_celeba_bundle', 'create_client_data_dict', 'sample_server_dataframe', 'apply_root_noise'):
        args = [a.arg for a in node_map[fn].args.args]
        attrs = {n.attr for n in ast.walk(node_map[fn]) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'config'}
        require('attack' not in args and not (attrs & forbidden_config), 'Partition code consumes attack information')
    (OUT / 'frozen_source_function_excerpts.txt').write_text('\n\n'.join(snippets) + '\n', encoding='utf-8')
    by_key, client_rows, partition_rows = {}, [], []
    shuffle_checks = set()
    expected_keys = {(dist, seed) for dist in ('IID', 'non-IID') for seed in range(91001, 91011)}
    additional_attack_count_checks = 0
    for record in sorted(sources, key=lambda r: Path(r['output']).name):
        path = Path(record['output']); d = json.loads(path.read_text()); cfg = d['config']
        key = d['distribution'], d['seed']
        require(key in expected_keys and key not in by_key, 'Partition key coverage/duplication')
        require(d['dataset'] == 'celeba' and d['method'] == 'GuardFed-AD2+' and d['attack'] == 'Sp-DFA', 'Outside source cohort')
        require(d['rounds'] == 70 and len(d['trajectory_metrics']) == 70, 'Not completed final-round source')
        require(d['num_clients'] == 20 and d['num_malicious'] == 4, 'Client budget differs')
        require(d['alpha'] == cfg['client_alpha'] == (5000. if key[0] == 'IID' else 5.), 'Real alpha differs')
        require(cfg['celeba_train_limit'] == cfg['celeba_eval_limit'] == 0 and cfg['celeba_evaluation_split'] == 'valid', 'Wrong image split')
        require(cfg['server_sampling'] == 'stratified_sensitive' and cfg['server_ratio'] == .1, 'Wrong root protocol')
        require(cfg['root_label_noise'] == cfg['root_sensitive_noise'] == cfg['synthetic_ratio'] == 0., 'Root perturbed in source')
        require(d['revision_job']['source_hashes'] == manifest['source_hashes'], 'Run frozen source mismatch')
        require(d['revision_job']['checkpoint_sha256'] == accepted[path.stem]['checkpoint_sha256'], 'Accepted checkpoint identity mismatch')
        dc = d['data_contract']; image = dc['image_data_contract']; root_audit = dc['server_sampling_audit']
        require(dc['train_rows'] == image['actual_train_rows'] == 162770 and image['actual_evaluation_rows'] == 19867, 'Row counts differ')
        require(image['cache_manifest_sha256'] == manifest['source_hashes']['data/celeba/derived/rgb64_v1/manifest.json'], 'Image source identity differs')
        global_joint = {k: int(v) for k, v in root_audit['global_group_counts'].items()}
        root_joint = {k: int(v) for k, v in root_audit['server_group_counts'].items()}
        require(set(global_joint) == set(root_joint) == {'0|0', '0|1', '1|0', '1|1'}, 'Missing joint root/global categories')
        require(sum(global_joint.values()) == 162770 and sum(root_joint.values()) == dc['root_clean_rows'] == 16277, 'Joint totals')
        global_sensitive = {g: sum(v for k, v in global_joint.items() if k.startswith(str(g) + '|')) for g in (0, 1)}
        root_sensitive = {g: sum(v for k, v in root_joint.items() if k.startswith(str(g) + '|')) for g in (0, 1)}
        root_label = {y: sum(v for k, v in root_joint.items() if k.endswith('|' + str(y))) for y in (0, 1)}
        require({str(g): n for g, n in root_sensitive.items()} == root_audit['server_sensitive_counts'], 'Root sensitive counts disagree')
        require({str(y): n for y, n in root_label.items()} == root_audit['server_label_counts'], 'Root label counts disagree')
        require(root_joint == dc['root_noise_audit']['clean_group_label_counts'] == dc['root_noise_audit']['observed_group_label_counts'], 'Zero-noise root support differs')
        lengths = {g: global_sensitive[g] - root_sensitive[g] for g in (0, 1)}
        for length in lengths.values():
            if (length, d['seed']) not in shuffle_checks:
                shuffle_content_independence(length, d['seed'])
                shuffle_checks.add((length, d['seed']))
        group_counts = reconstruct_sensitive_counts(lengths, d['alpha'], d['seed'])
        totals = [group_counts[0][i] + group_counts[1][i] for i in range(20)]
        require(totals == image['client_sample_counts'], 'Client-by-client exact total match failed')
        require(sum(totals) == 146493 and all(n > 0 for n in totals), 'Client pool totals/empty client')
        audits = {a['client_id']: a for a in d['attack_audit']}
        require(set(audits) == set(range(20)), 'Client audit identities missing')
        male_fraction = [group_counts[1][i] / totals[i] for i in range(20)]
        for i in range(20):
            require(audits[i]['samples'] == totals[i] and audits[i]['is_malicious'] == (i < 4), 'Attack audit sample/identity mismatch')
            if i < 2:
                require(audits[i]['attack_types'] == ['fflip'] and audits[i]['fflip_mode'] == 'all_unprivileged', 'Sp-DFA role mismatch')
                # This frozen mode sets sensitive values to zero, so changed count equals pre-attack Male=1.
                require(audits[i]['fflip_changed'] == group_counts[1][i], 'Independent changed-Male count disagreement')
                additional_attack_count_checks += 1
            elif i < 4:
                require(audits[i]['attack_types'] == ['foe'], 'Sp-DFA update role mismatch')
            client_rows.append({'distribution': key[0], 'alpha': d['alpha'], 'seed': key[1], 'client_id': i,
                                'sample_count': totals[i], 'sensitive_Male0_count': group_counts[0][i],
                                'sensitive_Male1_count': group_counts[1][i], 'Male1_fraction_pre_attack': male_fraction[i],
                                'nominal_malicious': i < 4, 'client_label_joint_counts_available': False})
        global_positive = global_joint['0|1'] + global_joint['1|1']
        joint_tvd = .5 * sum(abs(root_joint[k] / 16277 - global_joint[k] / 162770) for k in root_joint)
        sensitive_tvd = abs(root_sensitive[1] / 16277 - global_sensitive[1] / 162770)
        label_tvd = abs(root_label[1] / 16277 - global_positive / 162770)
        for value, name in ((joint_tvd, 'group_tvd'), (sensitive_tvd, 'sensitive_tvd'), (label_tvd, 'label_tvd')):
            approx(value, root_audit[name], 'Stored root representativeness metric mismatch: ' + name)
        pr = {'distribution': key[0], 'alpha': d['alpha'], 'seed': key[1], 'source_attack': 'Sp-DFA',
              'samples_total': sum(totals), 'client_sample_min': min(totals), 'client_sample_max': max(totals),
              'client_sample_mean': float(np.mean(totals)), 'client_sample_population_sd': float(np.std(totals, ddof=0)),
              'client_sample_cv': float(np.std(totals, ddof=0) / np.mean(totals)),
              'client_Male1_fraction_min': min(male_fraction), 'client_Male1_fraction_max': max(male_fraction),
              'client_Male1_fraction_unweighted_mean': float(np.mean(male_fraction)),
              'client_Male1_fraction_population_sd': float(np.std(male_fraction, ddof=0)),
              'client_Male1_fraction_weighted': lengths[1] / sum(totals),
              'clients_missing_Male0': sum(n == 0 for n in group_counts[0]),
              'clients_missing_Male1': sum(n == 0 for n in group_counts[1]),
              'first4_malicious_sample_fraction': sum(totals[:4]) / sum(totals),
              'first4_malicious_Male0_coverage_fraction': sum(group_counts[0][:4]) / lengths[0],
              'first4_malicious_Male1_coverage_fraction': sum(group_counts[1][:4]) / lengths[1],
              'first4_malicious_Male1_fraction': sum(group_counts[1][:4]) / sum(totals[:4]),
              'root_rows': 16277, 'root_0_0': root_joint['0|0'], 'root_0_1': root_joint['0|1'],
              'root_1_0': root_joint['1|0'], 'root_1_1': root_joint['1|1'],
              'root_joint_min_support': min(root_joint.values()),
              'root_Male1_fraction': root_sensitive[1] / 16277,
              'global_Male1_fraction': global_sensitive[1] / 162770,
              'root_Smiling1_fraction': root_label[1] / 16277,
              'global_Smiling1_fraction': global_positive / 162770,
              'root_joint_tvd': joint_tvd, 'root_sensitive_tvd': sensitive_tvd, 'root_label_tvd': label_tvd,
              'root_image_ids_sha256': image['root_image_ids_sha256'], 'result_sha256': record['sha256'],
              'all20_sample_counts_exact_match': True, 'client_label_joint_counts_available': False}
        partition_rows.append(pr)
        by_key[key] = pr
    require(set(by_key) == expected_keys, 'Incomplete 2x10 partitions')
    for seed in range(91001, 91011):
        iid, noniid = by_key['IID', seed], by_key['non-IID', seed]
        for key in iid:
            if key.startswith('root_') or key.startswith('global_'):
                require(iid[key] == noniid[key], 'Root differs across alpha for a shared seed')
    numeric_keys = ['client_sample_min', 'client_sample_max', 'client_sample_cv',
                    'client_Male1_fraction_min', 'client_Male1_fraction_max', 'client_Male1_fraction_population_sd',
                    'first4_malicious_sample_fraction', 'first4_malicious_Male0_coverage_fraction',
                    'first4_malicious_Male1_coverage_fraction', 'first4_malicious_Male1_fraction']
    summaries = {dist: {key: stat([r[key] for r in partition_rows if r['distribution'] == dist]) for key in numeric_keys}
                 for dist in ('IID', 'non-IID')}
    root_unique = [by_key['IID', seed] for seed in range(91001, 91011)]
    root_summary = {key: stat([r[key] for r in root_unique]) for key in ('root_joint_min_support', 'root_joint_tvd', 'root_label_tvd',
                                                                     'root_sensitive_tvd', 'root_Smiling1_fraction', 'root_Male1_fraction')}
    dump_csv(OUT / 'per_partition_20.csv', partition_rows)
    dump_csv(OUT / 'per_client_400.csv', client_rows)
    dump_csv(OUT / 'root_support_10_unique_seeds.csv', root_unique)
    helper_node = next(n for n in ast.parse(Path(__file__).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'reconstruct_sensitive_counts')
    acceptance = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'PASS',
                  'source_method': 'GuardFed-AD2+', 'source_attack': 'Sp-DFA', 'scope': 'pre-attack partition/support only',
                  'partitions_verified': 20, 'shared_seeds': list(range(91001, 91011)), 'clients_per_partition': 20,
                  'client_total_count_exact_checks': 400, 'attack_audit_sample_count_exact_checks': 400,
                  'extra_original_Male1_counts_from_fflip_changed_checks': additional_attack_count_checks,
                  'shuffle_content_independence_checks': len(shuffle_checks),
                  'missing_sensitive_group_client_occurrences': sum(r['clients_missing_Male0'] + r['clients_missing_Male1'] for r in partition_rows),
                  'actual_alpha': {'IID': 5000., 'non-IID': 5.},
                  'root_same_across_distributions_per_seed': True, 'root_unique_seed_n': 10,
                  'source_manifest_sha256': file_sha(manifest_path), 'stageA_snapshot_sha256': file_sha(snapshot_path),
                  'source_pins': pinned, 'source_function_ast_records': ast_records,
                  'helper_ast_sha256': hashlib.sha256(ast.dump(helper_node, include_attributes=False).encode()).hexdigest(),
                  'helper_script_sha256': file_sha(Path(__file__)), 'numpy_local_version': np.__version__,
                  'source_execution_order': {'load_bundle_line': load_line, 'runtime_attack_line': attack_line,
                                             'load_precedes_attack': True, 'partition_signatures_have_no_attack': True,
                                             'partition_config_has_no_attack_or_tag_dependency': True,
                                             'group_order': [1, 0], 'shuffle_precedes_dirichlet': True},
                  'archives': list(archives.values()), 'restored_result_sources': sources, 'frozen_job_member_checks': job_checks,
                  'partition_summaries_across_10_seeds': summaries, 'root_summary_across_10_unique_seeds': root_summary,
                  'client_label_joint_counts_reconstructed': False,
                  'limitations': [
                      'Group counts reconstructed from frozen shuffle/Dirichlet logic and archived global/root counts; all 400 archived client totals match exactly.',
                      'Only pre-attack sensitive-group counts are reconstructed. Runtime attack annotations differ for the first two Sp-DFA clients.',
                      'No client label-by-sensitive joint counts or within-client Smiling rate can be recovered without the original attribute metadata.',
                      'Shared root repeated under IID/non-IID is one root sample per seed, not two independent observations.',
                      'The audited source is 20 GuardFed-AD2+ Sp-DFA records. Transfer to other attack choices follows frozen attack-independent loading, not a new extraction of every method/scenario.',
                      'This audit does not measure heterogeneity beyond sensitive-group allocation or establish realistic device populations.'
                  ], 'training_started': False, 'image_metadata_loaded': False, 'original_results_modified': False}
    (OUT / 'acceptance.json').write_text(json.dumps(acceptance, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    write_report(acceptance, partition_rows, root_unique)
    print(json.dumps({key: acceptance[key] for key in ('status', 'partitions_verified', 'client_total_count_exact_checks',
                      'extra_original_Male1_counts_from_fflip_changed_checks', 'missing_sensitive_group_client_occurrences',
                      'root_unique_seed_n')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
