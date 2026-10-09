#!/usr/bin/env python3
"""Render the accepted offline diagnostics and pin delivery member hashes."""
import csv
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def read_csv(name):
    with (OUT / name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    acc = json.loads((OUT / 'acceptance.json').read_text(encoding='utf-8'))
    assert acc['status'] == 'PASS'
    partitions, clients = read_csv('per_partition_60.csv'), read_csv('per_client_1200.csv')
    assert len(partitions) == 60 and len(clients) == 1200
    summaries = acc['partition_summaries_across_10_shared_seeds']
    def ms(s, percent=False, digits=3):
        scale = 100 if percent else 1
        return f"{scale*s['mean']:.{digits}f}±{scale*s['sample_sd_ddof1']:.{digits}f}"
    lines = ['# Adult/COMPAS 强异质性实际分区审计', '',
             '**60/60 分区通过；1,200 个 client 样本总数、2,400 个敏感组边际及240个 S-DFA 属性翻转计数精确匹配。** 本次仅离线读取2026-09-23原始归档及冻结源码，没有连接服务器、训练模型、改动原实验或已发布快照。', '',
             '## 来源和验收范围', '',
             '- 对象是Adult/COMPAS × α=1/0.5/0.1 × 10个预定义seed，共60个攻击前分区；原结果均为GuardFed-AD2+、Benign、70轮。每个分区另与同seed/alpha的S-DFA完整原记录交叉核对。seed为123、456、789、1001、2024、3141、4242、5050、6060、7070；不是重新挑选的seed。',
             '- 来源归档：[revision_verified740_20260923T090026Z.tar.gz](../revision_verified740_20260923T090026Z.tar.gz)，SHA256 `55b50d489a4ee1824a8637c7da61e25157b50f500ddbd2fc5119711ba16c399b`。仅选择普通文件安全恢复；每个归档member的字节数/SHA、120原job、两个manifest与恢复路径保存在[restoration_receipt.json](restoration_receipt.json)。S-DFA保存的是明确标注的完整原record摘录，其member SHA覆盖原始完整JSON，复核可由原归档再读取。',
             '- Adult使用git `9e62b7887be73a4c3ce600bb8d6df79596fba0e1`的core/loader/runner；COMPAS使用`b1a808b0c8016634e032a3d1914e55782da44fdd`。六份源码完整SHA与各自manifest逐一一致。COMPAS是train_only预处理，未与旧legacy结果混合。源码、函数AST与count helper SHA见[acceptance.json](acceptance.json)和[函数摘录](frozen_source_function_excerpts.txt)。',
             '- 由global四格减root四格取得客户端池两敏感组长度。按冻结的fresh default_rng(seed)、敏感组1→0、int64 shuffle→Dirichlet→cumsum/floor切分顺序重放。对60份分区另直接执行冻结分区函数AST，输入明确标识的虚构索引/占位标签，只比较敏感计数；未把占位标签当成原标签。40个不同数组取值的RNG状态及swap-pattern检验通过。',
             '- 1,200个重建样本总数均与Benign、S-DFA原attack_audit及当时的原分区审计一致；2,400个敏感边际与原client四格边际一致；前4个S-DFA client全部240个fflip_changed等于原敏感组1数，label_changed_count为0。两种攻击的root/data_contract完全相同，120个原job配置/源码数据身份一致。',
             '- 此次还找回了两套当时直接从load_bundle统计的**原client标签×敏感组四格**。它们有独立归档来源和生产脚本；已核每client总数/敏感边际、全体四格=训练总体−root四格、原标签边际及空组标记。下文标签/四格字段标为“归档实测恢复”，**未从敏感边际推导标签，也未以原特征数据独立重算四格或tensor SHA**。这与CelebA目前缺少client标签metadata的证据边界不同。', '',
             '## 分区描述量：每条件10seed', '',
             '先在每seed的20个client内统计，再跨10个共享seed报告mean±sampleSD(ddof=1)。CV的分母是20client样本均值，分子是populationSD；空端保留在样本CV和空端数中。非空client的组比例才有定义。四格缺失统计来自原归档实测，不是新生成标签。', '',
             '| 数据集 | 实际α | 样本数CV | 空端/20 | 非空单敏感组/20 | 名义前4端样本覆盖（%） | 覆盖率范围（%） |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for d in ['adult', 'compas']:
        for alpha in [1., .5, .1]:
            s = summaries[f'{d}|alpha={alpha:g}']; coverage = s['first4_nominal_sample_fraction']
            lines.append(f"| {d.upper() if d == 'compas' else 'Adult'} | {alpha:g} | {ms(s['client_samples_cv_population_sd'])} | {ms(s['empty_clients'], digits=2)} | {ms(s['nonempty_single_sensitive_group_clients'], digits=2)} | {ms(coverage, True, 2)} | {100*coverage['min']:.2f}–{100*coverage['max']:.2f} |")
    lines += ['',
              '下面是每条件200个client观测的计数和范围，**仅作描述，200不是独立seed样本量**。缺任一敏感组包括空端；非空单组数单列，避免把空端当作组比例为0。四格缺失也包括空端。Adult敏感1为Male，COMPAS敏感1为African-American、0为Others，按冻结loader实际编码，不另外赋予社会群体“特权”含义。', '',
              '| 数据集 | α | 空端/200 | 缺任一敏感组/200（含空端） | 非空单组/200 | 缺任一四格/200（归档） | 非空敏感1比例范围（%） |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for d in ['adult', 'compas']:
        for alpha in [1., .5, .1]:
            rows = [r for r in clients if r['dataset'] == d and float(r['alpha']) == alpha]
            nonempty = [r for r in rows if r['empty_client'] == 'False']
            empty = sum(r['empty_client'] == 'True' for r in rows)
            missing = sum(r['missing_sensitive0'] == 'True' or r['missing_sensitive1'] == 'True' for r in rows)
            single = sum(r['nonempty_single_sensitive_group'] == 'True' for r in rows)
            joint = sum(int(r['archived_missing_joint_cells']) > 0 for r in rows)
            ratios = [float(r['sensitive1_fraction_nonempty']) for r in nonempty]
            lines.append(f"| {d} | {alpha:g} | {empty} | {missing} | {single} | {joint} | {100*min(ratios):.3f}–{100*max(ratios):.3f} |")
    lines += ['',
              '**实测限制：α=0.1并非“所有client都得到较少但完整的数据”。** Adult每seed0–6空端、COMPAS每seed2–7空端；非空单组端大量存在。因此不能声称每端四格都有支持，不能把4/20恶意ID等同于20%恶意样本，也不能假定local TPR/FPR处处有非零分母。空端的组比例保存为undefined；没有重分区、重采样或剔除seed。冻结train_local_model对n=0返回原global state，但后续攻击/聚合是另外步骤；本审计不由“空端”推断最终聚合系数或更新贡献必为0。', '',
              '## root支持和代表性', '',
              '每个数据集同seed的三个α使用相同root计数和TVD；COMPAS还核对clean_root_tensor SHA相同。因此每个数据集只按10个unique root统计，而非30份重复root。Adult训练/测试为30162/15059，root3016、客户端池27146（sensitive0=8804、sensitive1=18342）；COMPAS为4320/1852，root432、客户端池3888（0=1888、1=2000）。COMPAS的train/test划分随seed，四格总体不被假定完全不变。', '',
              '| root描述量 | Adult（10seed mean±SD） | COMPAS（10seed mean±SD） |',
              '|---|---:|---:|']
    rsum = acc['root_summaries_across_10_unique_seeds']
    for title, key, digits in [('四格中最小支持数', 'root_min_joint_support', 2), ('sensitive×label联合TVD', 'root_joint_tvd', 6), ('sensitive边际TVD', 'root_sensitive_tvd', 8), ('label边际TVD', 'root_label_tvd', 6)]:
        lines.append(f"| {title} | {ms(rsum['adult'][key], digits=digits)} | {ms(rsum['compas'][key], digits=digits)} |")
    lines += ['',
              '所有已测root四格非空，最小支持数Adult90、COMPAS68；180个已存TVD逐项重算误差≤1e-12。以上只说明当前stratified_sensitive抽样的root支持，不能证明任意现实获取的root具有同样代表性，也不产生隐私保证。原20个unique root见[root_support_20_unique_seeds.csv](root_support_20_unique_seeds.csv)。', '',
              '## 原60份分区明细', '',
              '完整来源SHA、两组覆盖份额、非空最小样本量及标签范围见[per_partition_60.csv](per_partition_60.csv)；全部1,200个client敏感计数及标为archived的四格见[per_client_1200.csv](per_client_1200.csv)。', '',
              '| 数据集 | α | seed | 样本min–max（含空端） | 空端 | 非空单组 | 前4样本覆盖% |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(partitions, key=lambda r: (r['dataset'], -float(r['alpha']), int(r['seed']))):
        lines.append(f"| {r['dataset']} | {float(r['alpha']):g} | {r['seed']} | {r['client_samples_min_including_empty']}–{r['client_samples_max']} | {r['empty_clients']} | {r['nonempty_single_sensitive_group_clients']} | {100*float(r['first4_nominal_sample_fraction']):.3f} |")
    lines += ['', '## 复核命令与边界', '',
              '在仓库根目录执行：', '', '```powershell',
              'python docs/server_deployment_20260923/training_20260923/tabular_partition_audit_20261009/restore_sources.py',
              'python docs/server_deployment_20260923/training_20260923/tabular_partition_audit_20261009/audit_partitions.py',
              'python docs/server_deployment_20260923/training_20260923/tabular_partition_audit_20261009/build_report.py', '```', '',
              '脚本只写本目录。restore需要本地已备份归档和已存在Git对象；audit只编译并执行冻结的分区函数AST，使用CPU占位张量，不导入/运行训练模块。恢复原result约132MB，不需服务器或真实图像。验收覆盖分区来源与计数，不是重新验收checkpoint数值性能；模型SHA仍由原正式训练验收负责。', '',
              '当前交付关闭P4的Adult/COMPAS强α实际分区/support子项。它没有补齐剩余8方法、CelebA机制消融、最终冻结评价，未解决历史synthetic生成器及Fig.3/PCA来源。本文候选段落见[manuscript_candidate.md](manuscript_candidate.md)。[integrity.json](integrity.json)记录本目录交付member SHA（不递归自哈希）。']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    paragraph = '''# Manuscript/rebuttal insertion candidate: realized strong-alpha partitions

This is a supported candidate paragraph, not an automatic change to the submitted manuscript.

We audited the realized pre-attack partitions for Adult and COMPAS at Dirichlet concentrations 1, 0.5, and 0.1 over ten shared predefined seeds (60 partitions). Replaying the frozen sensitive-group allocation exactly matched all 1,200 client sample counts and all 2,400 sensitive-group marginals in the archived partition audits; 240 original S-DFA attribute-flip counts provided an additional check. The archived client label-by-sensitive tables were separately recovered and checked for marginal and population-minus-root conservation; they were not inferred from the sensitive counts or independently regenerated from raw metadata in this audit. At alpha=0.1, the 200 Adult client observations contained 29 empty clients and 104 nonempty single-sensitive-group clients; the corresponding COMPAS counts were 44 and 78. Across the ten seeds, the first four nominal attacked IDs held 1.912%–66.680% of client examples on Adult and 1.337%–65.458% on COMPAS. These allocations are retained without resampling or seed exclusion. They show that a 4/20 attacked-client setting does not fix the attacked-example share and that local group or group/label support can vanish under severe allocation skew. The sampled roots retained all four sensitive/label cells (minimum support 90 on Adult and 68 on COMPAS), but this describes the implemented stratified root sample rather than guaranteeing availability or representativeness of a deployment root. COMPAS uses the separately documented train-only preprocessing cohort; these sensitivity results are not pooled with historical legacy tables.

Report the table as descriptive ten-seed diagnostics, with sample SD computed across seeds. Empty-client group proportions are undefined. The partition is by sensitive group, not an independently specified label-skew process. No final coefficient, attack success rate, or per-client fairness denominator is inferred from sample coverage alone.
'''
    (OUT / 'manuscript_candidate.md').write_text(paragraph, encoding='utf-8')
    members = [{'path': str(p.relative_to(OUT)).replace('\\', '/'), 'bytes': p.stat().st_size, 'sha256': sha(p)}
               for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'integrity.json' and '__pycache__' not in p.parts]
    (OUT / 'integrity.json').write_text(json.dumps({'status': 'PASS', 'files': members,
        'note': 'Each listed member is pinned; integrity.json does not hash itself. Original result/model files outside this directory were not changed.'}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'files': len(members), 'integrity_sha256': sha(OUT / 'integrity.json'),
                      'readme_sha256': sha(OUT / 'README.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
