# C 三完整场景表：仅准备，尚未生成统计

当前真实 native136 包含 U100 + C36；这里已经接受的三视图来源为旧两场景C20、after20的5条、after25的3条。after28的8条必须实际严格验收、离机并由根采用后，才允许build；RUNNING、远端完成或 prepared package均不能替代 ROOT_ADOPTION_REVIEW。

现有两场景包原字节不动。本生成器仅沿旧 build 的已接受receipt连接，读取 after20 5 + after25 3 + after28 8；从中10条FedSA与900已接受Full逐模型配对，追加至旧40条。最终表60条=Full30+C30，仅 IID Benign/F Flip/FedSA。新增S-DFA六条严格来源记录另存 excluded_partial_C_records.json，包含配对Full引用，不推理Full、不进入均值、不丢弃负结果。

原 statistic/summarize、receipt_identity/normalized、Full normalizer 和原严格 archive/receipt join 均复用原封源码。面板仍 raw/native/shared × 10/9/6同一seed集合，配对差始终 minus_C−Full。预计81行、486均值/SD标量、243展示单元、540原receipt三指标及1440计数结构检查。数字按3场景×3视图×3seed面板×3行×3指标推导；它们是未来验收范围，不是已经产出的统计。

每次build前核126输入绑定及自身seal。三个archive链均逐成员SHA，并使用原join的配置、source/data/checkpoint、native1e-12、root-only校准、receipt、strict、offserver一致性门。旧40记录值及JSON片段逐字相同，旧324统计标量完全相同、162展示单元整行逐字相同；旧独立numeric verifier再回归。新 arithmetic body及confusion公式沿旧实现，仅场景及断言计数扩展。

## 根实际采用 after28 之后才能执行

```powershell
python -B tmp/celeba_mechanism_three_view_C_three_scenes_prepared_20261009/build.py --C8-adoption ACTUAL_AFTER28_BACKUP/ROOT_ADOPTION_REVIEW.json --C8-adoption-sha256 ACTUAL_ROOT_SHA --output tmp/celeba_mechanism_three_view_C_three_scenes_prepared_20261009/snapshot
python -B tmp/celeba_mechanism_three_view_C_three_scenes_prepared_20261009/verify_numeric.py --snapshot tmp/celeba_mechanism_three_view_C_three_scenes_prepared_20261009/snapshot
```

外部路径必须位于 after28 原backups内，根凭据明确128+8=136、exact8、source_scope_complete、旧128不变、负结果保留、前after25根SHA。缺失/错误凭据拒收。未生成未来proof、审批文件、snapshot或统计结果。准备自检只运行metadata/内存fixture；其正向fixture只说明门形状可满足，不能作为根审批。

展示继续披露 AEOD=绝对TPR差（不是完整equalized odds）；native含各自原生校准；shared为共同root-only规则；模型三视图同终轮checkpoint；ACC(%)与ΔACC(pp)，gap越低越好。保留CPU/GPU、CUDA/driver混合及seed91001配置选择、validation/test历史暴露。无显著性/必要性因果/最终test主张；主终点等待作者，七个其他C场景不完整。本包无SSH/CNN/重新拟合/训练/正文或canonical修改。
