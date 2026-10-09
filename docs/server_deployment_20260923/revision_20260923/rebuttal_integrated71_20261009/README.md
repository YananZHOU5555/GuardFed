# 完整英文稿最小更新：七场景固定快照

**DO_NOT_SUBMIT_BEFORE_FULL_COHORT。** 仅将旧完整六场景稿更新到根正式接受的 `three_view_interim71_20261009`；旧包原字节保持。

- [完整24意见回复](rebuttal_integrated_20261009.md)
- [完整正文插入稿](manuscript_insertions_integrated_20261009.md)
- [六→七场景完整diff](UPDATE_DIFF.patch)、[来源映射](SOURCE_MAP.json)、[主表数字引用](NUMERIC_REFERENCES.json)、[评论核对](COMMENT_CONSISTENCY.json)、[未改段落哈希](UNCHANGED_PARAGRAPHS.json)、[链接核对](LINK_CHECKS.json)和[验证](verification.json)。

新快照 ROOT_REVIEW SHA `c616d97c9eb6ad0112773925f563791ad9d9154261f7258acadc9c331a754281`，接受于 2026-10-09T15:35:42.401065+00:00。七场景是原六场景加 non-IID F Flip。70 完整配对/140 展示模型进入相同规则 n=10/9/6 面板；71 配对/142 原 receipt 全部保留，额外 non-IID FedSA91001 一对因未齐十seed而排除所有均值/SD面板。

n=10 删除U的 ACC 七场景均更低0.309–1.384pp、ASPD七场景更低；native/shared AEOD六场景更高，仅IID Sp-DFA更低；raw AEOD五场景更低，IID Sp-DFA及non-IID F Flip更高。新增F Flip原native配对δ自动引用原JSON，不计算新指标或统计。native/shared140展示模型全部指标及组混淆计数字典相同，不是独立校准增益。Full5CPU65GPU、69cu1281cu130；controls70CPU/cu128，driver与91001选择及valid曝光限制保留。

首个预写检查拒收了任务摘要中的“raw六场景降低”；固定原tables.json实际为五降低、两升高，根已确认是摘要解读错误。该拒收和准确原表值保留于PREWRITE_REFUSAL.json，没有修改源表或统计。Python首次导入旧文本helper产生了非sealed `tmp/guardfed_rebuttal_integrated_20261009/__pycache__/integrate.cpython-310.pyc`；后续显式禁用缓存写入。该单缓存清理被automatic approval review以“blocked by policy”拒绝，未绕过重试删除；根已明确要求保留该非sealed缓存，不删除、不移动、不发布；不需要用户处理。旧11成员源封条未变。

原24评论、tabular280、COMPAS全部负结果和P1–P6保留；P2仍pending，native/shared主口径不选择。900模型valid重放完成不等于完整机制实验或final test完成。未联网、未新推理、未训练、未改Git/canonical/旧sealed源。外部URL仅语法核对，不宣称在线访问核验。更新脚本只调用旧 integrator 的文本/hash/link只读帮助函数，复用原接受表和记录，不新增统计框架；拒绝覆盖交付。
