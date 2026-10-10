# exact3 root adoption — independent review

**PASS：可采用这3项 validation 三视图接口记录；没有发现阻止该有限范围采用的实质缺口。** 本结论不代表所有新增方法、100项多种子比较、17方法全集或最终测试已经完成，也不把原251项机制三视图截点扩为254。

固定输入 `ROOT_SCIENTIFIC_ADOPTION.json` SHA256 `631d7ee2acf1cbe3453d849523e262f29ff5b354b5ddb56c60e5779e94364456`。其13个证明文件和 adopter 源 SHA 均一致。独立审查只读取源码及紧凑 JSON；没有重跑 CNN、root fit、科学全量 verifier、SSH、数组加载、STATE/Git操作或F写入。

- **Linux完整原检查通过。** 当前保留的远程 helper 与调用脚本中的 literal 完全相等；原 saved checker 文件 SHA `d512e5…` 与 check_saved 函数片段 SHA `ed8dc5…` 固定，执行的是原函数整段 AST。实际退出0，stdout与CHECK证明逐字相同，stdout/stderr SHA与退出记录一致。映射仅包装 artifact/identity/path；模型、result和job在检查前后实测 SHA，原 root receipt 相等门、weights门、数组顺序、fit/prediction/metrics 门保留。Linux绑定冻结 package/manifest/source以及metadata，隐藏CUDA、CPU单线程，无图像推理入口调用。
- **Windows完整检查失败没有被改写为通过。** 退出1和原始stderr完整保留，失败发生在 root receipt/weights 门；FLGMM唯一差异是 `/server_sampling_audit/group_kl`，差 `-2.168404344971009e-19`。另外两项完整root receipt的重建诊断相等。这不能证明 Windows整条checker对三项完整成功。精确平台/库原因尚未实验隔离。
- **Windows原数组代码块通过有单独边界。** 后续 helper 从原 check_saved 取唯一 `with np.load(...)` AST，不复制新公式；AST SHA `170bed…` 与证明一致。它保留root/valid ID顺序、arraySHA、root-only fit完全相等、预测数组完全相等、完整metrics字典和native对比检查。使用经封存的同一setup和actual core/evaluator；外部重建诊断提供root/partition检查，原whole-root receipt检查没有假称通过。这里的 `1e-12` 是未改动native容差，不是抹平Windows审计字段差异的容差。
- **同checkpoint与身份成立。** 三项精确ID顺序与冻结manifest、gate、Linux、Windows、root adoption和F回执一致；各自checkpoint/result/job/config身份沿既有绑定，模型前后张量hash相同，rounds=70，split=valid，valid n=19,867，clean root n=16,277。回执和数组SHA与原transport清单一致。此次只读取F上的小JSON，没有重复取模型/数组或改动科学证据。
- **27/72/9计数准确。** 3记录×3视图×3主指标=27；3×3×2敏感组×4基础TP/FP/TN/FN=72；3×3预测规则=9。独立从回执整数计数复核27个指标，算术差均≤1e-15（只作本次review额外核对，未改任何原验收容差）。72不表示所有派生计数；n/positives/negatives另核代数一致。native差均为0。native/raw用严格 `>0`，shared用组阈值 `>=`，shared仅clean train root拟合。

接口仅限 FLGMM IID S-DFA seed91005、CosineFairnessHybrid IID Benign seed91002、Fed-NGA eta0.01 non-IID Benign seed91001。最后一项是搜索候选，不是已选配方。小规模接口通过不能支持总体胜出或显著性主张。原 inference的3次root校准和后续两平台共6次验证refit有分别记账；本review新增fit/inference均0。

本review将当前Linux及Windows helper源码hash另存EVIDENCE，以便发布清单绑定：原root adoption proof_files并没有包括所有helper源，不能把输出receipt hash单独视为源码完整发布包。后续Git范围应保留这些helper、失败证明及本review；这是发布完整性要求，不改变已测结果。
