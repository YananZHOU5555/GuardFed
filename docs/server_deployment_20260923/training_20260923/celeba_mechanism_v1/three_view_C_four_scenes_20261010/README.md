# C40 四场景表：只准备，不等同重放闭合

目标为IID Benign/F Flip/FedSA/S-DFA各10共享seed，Full40+C40=80条模型记录。现有C30三场景表已经root采用；原单列S-DFA6具有真实三视图证据，但不能在补足之前构成均值。native140仅证明原生终轮已验收，after36四条三视图必须实际strict、离机及root采用后才能build。

这里复用C30构建源码和原accepted_increment/receipt_identity/normalized/Full normalizer/statistic/summarize。仅更改精确增量身份（136→140，IID S-DFA91007..91010）、旧表入口、场景数和断言计数；预测、阈值、metric、1e-12容差、种子集合与统计规则不变。既有旧60记录及原S-DFA6记录片段逐字保留；旧486统计标量精确相同、243展示单元整行及顺序不变；未来新4读取原strict/offserver归档所有成员SHA，同checkpoint三指标和source/data/config绑定。原Full只按已采用900引用，不推理、拟合或重包权重。

三视图native/raw/shared ×10/9/6seed面板；每个场景并列Full、minus_C、minus_C−Full。预计108行、648均值/样本SD标量、324展示单元、720receipt指标和1920计数结构检查，均按实际范围推导。ACC以百分数、差值为百分点；ACC高/gap低较好。全部方向与负结果保留，不改为score、不择优seed、不作显著性或C必要性/因果主张。

## 等根提供实际after36采用凭据并授权后

```powershell
python -B tmp/celeba_mechanism_three_view_C_four_scenes_prepared_20261010/build.py --C4-adoption ACTUAL_AFTER36_BACKUP/ROOT_ADOPTION_REVIEW.json --C4-adoption-sha256 ACTUAL_ROOT_SHA --output tmp/celeba_mechanism_three_view_C_four_scenes_prepared_20261010/snapshot
python -B tmp/celeba_mechanism_three_view_C_four_scenes_prepared_20261010/verify_numeric.py --snapshot tmp/celeba_mechanism_three_view_C_four_scenes_prepared_20261010/snapshot
```

门禁要求实际ROOT_C_AFTER36_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS、原136保持、准确4与总140、完整源码范围、负结果保留以及前after28根SHA。不接受native完成数、RUNNING、准备文件或缺失凭据；自检中的正向审批fixture只存在进程内存，不落地、不授权build。

AEOD是绝对TPR差，不是完整equalized odds；native包含各自原生root-only校准，shared使用原共同root-only规则。保留mixed CPU/GPU、CUDA/driver来源、seed91001配置选择与历史validation/test暴露；9/6是描述性子集。四场景不会表示全部C场景或全返修完成；主终点仍待作者。

本包不SSH、CNN、训练、阈值重拟合、部署、canonical/STATE/Git修改。原三场景封包不变。
