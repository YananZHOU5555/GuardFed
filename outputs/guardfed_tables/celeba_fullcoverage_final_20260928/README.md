# CelebA 阶段 A 完整验证集表

固定来源快照：2026-09-28 10:20:44 UTC，700/700 条严格验收通过，644 条新增 + 56 条明确复用，零缺失、零失败。七方法 × IID / non-IID × 五场景 × 十种子，全部 70 组均已完成。

本目录是独立完整版本，不覆盖先前八种子中期表。所有结果为第 70 轮、19,867 张验证图像，不称未触碰测试集表现，也不代表整个返修实验已结束。

- `celeba_iid_ten_seed`：IID 五场景完整表，PNG / PDF / LaTeX / Markdown。
- `celeba_noniid_ten_seed`：non-IID 五场景完整表，PNG / PDF / LaTeX / Markdown。
- `celeba_iid_noniid_ten_seed`：双分布合并完整表，PNG / PDF / LaTeX / Markdown。
- `celeba_iid_noniid_exclude_selection`：排除配置选择种子 91001，九种子 91002–91010；LaTeX / Markdown。
- `celeba_iid_noniid_prospective`：此前未观察的六种子 91005–91010；LaTeX / Markdown。
- `coverage.md`：每方法 × 分布 × 场景均 10/10。
- `seed_paired_summary.json`：跨场景先在每个种子内平均，再跨种子汇总；附 GuardFed 与 FLTrust 的逐种子配对差值与胜率。
- `source_acceptance_snapshot.json`、`provenance.json`：冻结全部 700 条记录、checkpoint 身份及来源 SHA。

统计口径：均值 ± 样本标准差（ddof=1）；ACC 为百分数，AEOD / ASPD 为 0–1。每条记录全部指标来自同一终轮 checkpoint。AEOD 为代码实际实现的绝对 TPR 差，不是完整 equalized odds；不根据均值名次单独宣称显著性。

IID 表 350 条均为 cu128；non-IID 表含 336 条 cu128 + 14 条 cu130；合并表含 686 条 cu128 + 14 条 cu130。九种子和六种子子表全部 cu128。迁移首轮一致不证明 70 轮完全等价。seed 91001 曾参与配置选择，其他前四种子结果也已看过；六种子表用于检查此前未观察种子的表现。

三个带星号方法是项目适配。GuardFed 已有训练-root组阈值校准，基线尚未施加相同校准，仍需共享校准对照。阶段 A 仅七种实现；其余正文基线、CelebA机制对照和最后冻结测试评价仍为后续工作。

数据摘要：以每 seed 内平均全部十场景，再跨 seed 汇总，GuardFed 对比 FLTrust 的 ACC 平均低 1.209 个百分点，AEOD 低 0.03618，ASPD 低 0.05351。十个种子均呈现相同的准确率与公平性取舍；六种子子集亦如此。不能写成三项指标全面击败所有方法。攻击下低准确率伴近零 gap 的记录原样保留，不能将这种结果直接解释为模型有效公平。

验证：700 个独特运行身份与完整种子覆盖通过；630 个已有汇总指标均值 / 样本SD与原始 `all_conditions` 重算一致；五份 LaTeX 已实际通过本机 XeLaTeX 编译；三张成品 PDF 经过 Poppler 渲染目视检查及单页文本核验。PDF 提取器会拆开英文标题中的字距，文本核验先规范化空白，不改变产物。`verify_tex.*`、`*_pdfcheck.png` 是 QA 产物。

生成器读取本目录的固定快照以保证复现，只写本目录，不修改训练、旧表或实验协议。
