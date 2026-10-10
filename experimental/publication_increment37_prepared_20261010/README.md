# Increment37 prepared publisher / verifier

状态：SOURCE_ONLY_PREPARED。未执行 Git、stage、commit、push、SSH、CNN 或新统计；本包不授予发布权限。

父提交固定 `277be091be761249372c1da17c97d2bf83ef62ef`，分支 `codex/revision-evidence-baselines-20260928`，既有发布 worktree `tmp/revision-publish-20260928`。闭合口径 native160 / three-view160 / FL12 / Hybrid19 / baseline900；005703快照164终轮只是观察。C60含六场景，non-IID仅Benign完整，未选主终点、未运行final test。

固定来源160文件、9,950,360 bytes，逐项列于 `PREPARED_INPUTS.json`。包含native160 exact4档案/inspection/ledger、after56 source/review/transport及其reader失败与部署后补充审阅时序、FL12 exact1、C60独立证据与短英文增量。publisher另读取显式根闭合输入，核验并收取实际C4备份、C60 canonical 24-member seal/ROOT、运行记录和最新状态入口。旧Hybrid档案、旧C50全文不重包；不复制任何restored/verified_extract/verified权重。

来源恢复仅发布8个自写报告或HTTP元数据文件，完整18-member seal及第三方PDF/提取文本/图片/HTML/NEXT_DATA正文留本机；0方法解锁。`ready_manifest`不是完整最终stage清单，动态闭合部分只在根提供实际SHA后加入。总量仍必须低于100MB，归档SHA不得重复，sealed bytes使用逐路径`-text`并复核index和commit blob SHA。

`ROOT_CLOSED_INPUTS_TEMPLATE.json`保留空值，不是接受/授权记录。根已准备自己的 `tmp/bind_publication37_actual_root_20261010.py`；应先独立审本包，再由根运行该绑定入口，确认生成的actual inputs SHA。实际兼容检查只在内存读取binder构造段，在首个Git调用前停止，没有生成actual闭合输入或调用Git。

根后续操作入口（本次均未执行）：

```powershell
python -B tmp/bind_publication37_actual_root_20261010.py
python -B tmp/publication_increment37_prepared_20261010/publish_increment37.py --closed-inputs tmp/publication37_actual_closed_inputs_root_20261010.json --closed-inputs-sha256 ACTUAL_ROOT_INPUT_SHA --output tmp/publication_increment37_prepared_20261010/actual_stage --execute-stage
```

publisher只stage并验证index，不commit/push。缺闭合、错SHA、分支/父提交不符或worktree不干净均拒绝。任何失败保留现场，不自动清理/重试。root在自己实际commit后调用以下verifier；只有实际push后才加`--remote`：

```powershell
python -B tmp/publication_increment37_prepared_20261010/verify_increment37.py --receipt tmp/publication_increment37_prepared_20261010/actual_stage/publication_closed_increment37_20261010.json --receipt-sha256 ACTUAL_RECEIPT_SHA --output tmp/publication_increment37_prepared_20261010/actual_verified.json --remote
```

`SELF_CHECK_FINAL.json`记录3个纯metadata正向fixture/25个拒收；`ACTUAL_INPUT_COMPATIBILITY.json`记录真正闭合字段及C60 24成员字节检查。Git分支/index/commit/远端检查留待根实际执行，未把source-only结果写成已发布。
