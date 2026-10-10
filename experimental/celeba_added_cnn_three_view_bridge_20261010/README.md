# 新增四 CNN 方法三视图：身份桥准备

本目录仅完成源码接线和小型门检。`TEST_RESULTS.json` 的 64 项检查通过；没有 CNN、阈值拟合、训练、SSH 或派发，没有新增三视图科学结果，也没有采用或启动授权。

`bridge.py` 提供两个入口：

- `identity_record(method, job_id)`：只读已注册的原 strict → offserver → root 接受链及 F 盘原件，验证字节 SHA、job/config/seed、source、checkpoint 和同终轮指标，返回独立私有 metadata。原 GPU provenance 放在 `original_training_provenance`，不描述本机运行环境；从不在 CPU 调用原 GPU checker，也不替换 CUDA 查询。
- `science_bindings(torch_module=..., pandas_module=...)`：从冻结原文件 AST 构造 evaluator 和 replay 的 17 个原函数；只把独立 globals 的 `METHODS` 设为 `FLGMM`、`CosineFairnessHybrid`、`Fed-NGA-gradient`、`Huber-BRFL-gradient`。原模块和原函数体不变。仅构造函数时不需要 Torch；本次未调用 `rebuild_root`、`extract_and_predict`、`fit_views` 或 `thresholds_from_root`。

CLI 只输出身份 metadata，例如在仓库根目录执行：

```powershell
python -B tmp/celeba_added_cnn_three_view_bridge_20261010/bridge.py FLGMM FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91005_fullcoverage
python -B tmp/celeba_added_cnn_three_view_bridge_20261010/check_bridge.py
```

此接口不是执行授权。后续真实门检须经 root 独审与授权，由独立执行层加载本票据绑定的 terminal checkpoint、真实原 core 和实际运行环境，使用原 `rebuild_root` 恢复 root ID/order 与 client counts，再使用原 `extract_and_predict` 及 `check_native`。本目录不提供 inference/fit runner、资源调度、cohort 生成器或 authorization receipt。

## 科学规则与身份差异

原 evaluator、`rebuild_root`、`check_native` 的来源及逐函数 SHA 在 `SOURCE_REUSE.json`。测试将 17 个实际绑定函数的 bytecode/flags 与原整文件编译结果比对。原 core 的 margin、shared fitter、root 采样/分区/noise 函数只记录冻结来源，供后续实际 core 复用。本次未执行这些科学函数。

native/raw 保留二分类 argmax 对应的 `margin > 0`；shared 保留 `margin >= group_threshold`，因此阈值为 0 时 tie 不同。shared recipe 保持原七字段：weight=1、budget=.06、temperature=.03、quantiles=41、max_acc_drop=.005、objective=acc_floor、enabled=True。原 fitter 仅接收 clean train root margins/labels/sensitive；valid 标签不参与 fit。原 metrics 及 native tolerance `1e-12` 不变。

FLGMM 的报告标签映射到原件 `FLGMM-author-code`，其余三方法保留原 source method。原件不会重标记成旧九方法的 `checked_result`。

两处 schema 接线均保留原值：gradient 的原始 `source/package/jobs/<id>.json` SHA 绑定 provenance/strict；其 `runs/<id>/job.json` 序列化字节不同，另 pin 并验证语义 exact。Hybrid 的 candidate 位于 `revision_job`；provenance 的 28 source 与 `full_scope` exact，job 的 21 source 为其 exact 子集；scope/local/job SHA 都接到原链。沿用现有 Hybrid seed 适配后的外部 strict 证明，不在本机重造运行时证明。

## 当前范围与拒收

`PROOF_PINS.json` 只注册三个实际已 root 采用的最新精确 chunk：FLGMM 6 条、Hybrid 首 1 条、Fed-NGA screen 8 条。三个 root SHA 分别为 `5e95872f…`、`04d62c36…`、`83d22e58…`；这不是全部已采用历史 prefix 的重收或完整 100 cohort 声明。扩展 registry 必须先有对应原 strict/offserver/root 证明并重新审查 pin，不能只增加 job ID。

Huber 已具备原 evaluator 方法标签及零 margin 门检；当前没有已注册、已 root 采用的 70 轮原 strict/offserver 证明，因此所有 Huber 身份票据都拒收。3 轮 canary 不能填补这一缺口。Fed-NGA 的 screen 记录不代表已选择获胜配置或已完成 fullcoverage。

拒收门检覆盖三种真实证据格式的 wrong method/source/config/seed/checkpoint/proof、部分 horizon、root 字节改变和非 adopted ID；另检查 Huber 缺 proof 与 LoGoFair 拒收。零 margin 测试仅使用四元素合成 prediction-stage fixture，没有拟合 threshold，也不是科学输入 receipt。现有源码 smoke 调整过上述 gradient/Hybrid schema 接线后才得到最终 PASS，未改变原证据。

本目录不注册 LoGoFair 普通 CNN native。其 DP/cohort 原作者口径按 [原 scope 报告](../celeba_added_baseline_three_view_scope_20261010/REPORT.md) 保持待决。旧 900、共享 STATE、Git、队列和原证据均未修改；不选择主终点或 test，不声称 runtime equivalence、最终论文结论或科研完成。F 盘本次只读检查确认为 `Yanan 2TB`，bulk 写入为 0。
