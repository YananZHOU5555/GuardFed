# A20 root binding — SOURCE ONLY

本入口仅准备根实际采用的薄身份绑定；当前没有执行采用、更新STATE/Git或生成论文表。新范围严格为`minus_A_IID_F Flip_seed91003..91010`八项，排除已采用U100+C100+A12共212项。若根实际执行全部原检查通过，预期新增8、累计220，remaining620中新增采用40；A只完成IID Benign/F Flip两场景，不能称A100或全部机制完成。

`bind_A20.py`以SHA `c6295afa5cea744f5ef18d1ca96d6c3ca3400d5aee5921019b3d8f6816f3f3af`读取原A12采用器，只在内存中替换身份元数据，然后根运行时执行。`METADATA_DIFF.patch`给出完整差异。原逐记录saved-count/阈值/checkpoint/native-member验证块逐字相同，原evidence_v4.verify_archive调用保留。源记录/native差异容差1e-12未变；并非新增科学接受器。

变化限于8个ID、212→220前驱/索引恢复链、两个已实际闭合native归档（070642Z六项＋072729Z两项）、新namespace、实际PREPARED字段、74成员/72指标/192计数/24规则，以及CLI外部运输pin。原A12采用proof `1221482d…`、index `9a90f3d7…`和运输前32项保持。旧212/Full不重包或重推，native只核新8的model/result共16成员。全部实际采用输出只写全新`tmp/celeba_mechanism_remaining620_A20_root_adoption_20261010`，存在即拒绝。

根审阅后的一次实际命令（本任务未运行）：

```powershell
python -B tmp/celeba_mechanism_A20_root_binding_20261010/bind_A20.py --delivery-seal-sha256 66b9e054a005786ca3d8d0e32547bbd83d7a8c66520b31b611a2bde759d6f9cc --storage-index-sha256 d9f3c06dba68212dafe719c5df66c881fdcb1ec4d4d8b03ce0563ff5fde64378 --archive-sha256 6c36e08cc27e9269491cda34e2aecc50aa258430c00148dfb2363cc5a0195847 --receipt-sha256 428fa89806721228f9deca04030218b819102d5d4f6c7595b23a29dba2fe7669 --offserver-sha256 5e75a5fd7ae606f97457478b72fe633454528faea585a68d3a7288fe815b5bd2 --archive-members 74
```

实际执行先完整核运输小目录封条、外部RAW_STORAGE_INDEX及原F卷guard，再复用原归档/保存输出/proof/native恢复链检查；原TRAINING_STATE必须仍对应旧212顺序。它不自动改STATE、retry、拟合或执行CNN。根自行决定执行及采用，不能把source-ready视为已采用220。

已实际运行`python -B tmp/celeba_mechanism_A20_root_binding_20261010/check_source.py`，exit0：实际小目录封条/schema、旧212唯一ID与prospective220组成、科学检查块逐字、编译/AST唯一字段，以及6项错误metadata拒收通过。未执行被绑定body、F卷查询、SSH、模型或数组读取。源入口/输入pins/check/diff由本目录封条绑定；原A12代码、proof、transport目录均未改。
