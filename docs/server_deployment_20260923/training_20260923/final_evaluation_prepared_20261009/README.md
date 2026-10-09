# 最终评价准备包

状态 **PREPARED_NOT_FROZEN**。仅当前九方法900个已验收valid模型的来源清单及可审阅方案；没有执行test、推理或训练。

- `PROTOCOL.md`：历史暴露、native/raw/shared校准选项、待决主表/评价终点、执行与验收门禁。
- `protocol.json`：机器可读的准备状态；重大选择留空。
- `model_inventory.json` / `.csv`：900个模型/result/config/job/source/data/seed/alpha/attack身份及备份链。
- `evaluation_jobs_draft.json`：900条终轮模型引用，全部不可dispatch。
- `identity_issues.json`：逐项缺口及未来服务器/target split/协议门禁。
- `prepared_acceptance.json`：本地实测身份检查，不是最终评价验收。
- `acceptance_contract.json`：未来同模型/同样本/同视图指标验收条件。
- `prepare_inventory.py`：只读已接受原产物，核archive/成员/model字节与终轮metadata；不导入torch、不加载标签。
- `check_prepared.py`：轻量清单及未冻结状态检查；拒绝把本稿当可运行阶段。

以`prepared_acceptance.json`的实测计数为准。未决选择未填写、服务器模型/数据未实时核验、目标split ID SHA未核定时，不得开启test。900只是九方法部分，不能冒充17方法或整个返修已完成。

本次实测：900/900终轮model/result/raw job全部定位，25份archive总SHA通过；56条旧复用的原config/source/result/model另与显式reuse核对通过，20个seed×distribution的root身份在九方法之间一致。训练环境为14cu130/886cu128，`identity_issues.json`当前逐条身份缺口为0；服务器现文件、target ID SHA和新评价门检仍未完成。`history_and_limits.json`保存具体暴露/选择历史。独立准备检查通过900网格及两个误派发拒绝案例；这些检查不等于test结果验收。
