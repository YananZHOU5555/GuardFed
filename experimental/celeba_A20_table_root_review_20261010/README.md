# A20 独立算术审查入口（源码交付）

本包仅准备独立审查器，未执行实际表审查、未采用表、未修改 canonical/STATE/Git。实际交付封存后由 root 单次运行：

```powershell
python -B tmp/celeba_A20_table_root_review_20261010/review.py --table-dir docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_IID_two_scenes20_20261010 --delivery-seal 3841a7498dad4a3dc6a0bf3b8262510082cd042671850b5f56320a079c012ef8
```

`--delivery-seal` 是 root 外部核对的实际 FILES_SHA256.json SHA256。审查器先核整个交付封条与固定来源；`-O/-OO` 被拒绝。成功仅在本目录写 `ROOT_ARITHMETIC_REVIEW.json`；首次失败写 `REVIEW_FAILURE.json` 并停止。已有任一输出即拒绝覆盖，不自动重试。

独立 oracle 复用原 root `verify_A_Benign10_root_20261010.py` 的 `math.fsum` 均值、sample SD(ddof1) 与混淆计数公式，不导入作者 builder/statistic。唯一范围变化是 Benign 一场景扩展为 IID Benign/F Flip 两场景，10/9/6 相同 seed 面板：40唯一记录、20对、324均值/SD标量、162展示单元、360混淆重算指标、960基础计数检查。旧24规范JSON记录及过滤后顺序、Benign162标量/81展示逐字保留。

来源连接固定实际 A12→A20 的 root adoption/index 恢复链，逐A记录核 binding JSON、六类小凭据、科学receipt的views/fits/config/checkpoint/result/rawjob、70轮/valid19867和原native1e-12；Full只连接已接受900 JSON及原Full100引用。读取现有F盘小JSON，不读取NPZ或模型，不重包归档、不执行CNN、fit、训练或test。

这不是新的科学验收器或采用器；预测数组、模型恢复链沿用原root接受证明，本检查独立核表算术与已接受来源。不产生显著性、必要性或最终测试主张。AEOD为绝对TPR差；ACC为百分比、配对差为pp。两场景证据仍受混合CPU/GPU重放、训练/driver、seed91001选择与valid/test历史限制；其它A场景/控制不由此完成。

`SOURCE_CHECK.json` 只记录源码语法、原独立算术AST保持与元数据范围检查，不代表实际324/162表已通过。源码封条不包含未来审查输出，也不包含 root 文件。
