# Git44实际输入草案

原Git44十成员封包保持不变。本目录仅准备实际路径/SHA清单，不运行Git、SSH、freeze、stage或fit。

当前范围：native200、三视图200、Hybrid27、FL新增28（4个复用单列）、LoGo32完整搜索、C100表和完整作者审阅稿。主队列05:37:24UTC观测204项终轮不计作验收204。LoGo100尚无本目录绑定的ROOT_STARTUP_REVIEW，字段为null。

`refresh_inputs_v3.py`复用v2实际读取器，仅把已根采用FL22→28接入元数据，并加入本批55封条、原严格/离机/张量证明和原始SHA恢复索引。它需要同目录`refresh_inputs_v2.py`。v1误读STATE的LoGo32键名导致KeyError，已保存ACTUAL_SPEC_FAILURE；v2仅改为实际root_adopted字段，无科学工作。ACTUAL_SPEC_v2是FL22当时快照；v3及后续快照才包含FL28。

新Hybrid交付全部本地封条成员先核SHA，再剔除日志、guide、tar和重复record body；后者以父提交的after18原字节路径显式恢复。FL55同样先全核本地成员，再只纳小文件。四份FL原始JSON receipt/member/strict/offserver逐字复制到FL28_compact，来源与SHA记在FL28_COMPACT_PROVENANCE；不复制模型或压缩包。

根最后刷新状态后，可写一个新输入快照（不要覆盖旧文件）：

```powershell
python tmp/publication_increment44_root_inputs_20261010/refresh_inputs_v3.py --main-live <实际root_live相对路径> --output tmp/publication_increment44_root_inputs_20261010/ACTUAL_SPEC_final.json --extra tmp/publication_increment44_root_inputs_20261010/refresh_inputs.py --extra tmp/publication_increment44_root_inputs_20261010/refresh_inputs_v2.py --extra tmp/publication_increment44_root_inputs_20261010/ACTUAL_SPEC_FAILURE.json --extra tmp/publication_increment44_root_inputs_20261010/REFRESH_V2_DIFF.patch --extra tmp/publication_increment44_root_inputs_20261010/FL28_COMPACT_PROVENANCE.json --extra tmp/publication_increment44_root_inputs_20261010/README.md
```

仅在实际启动被根审阅后追加`--startup <实际ROOT_STARTUP_REVIEW相对路径> --startup-sha256 <实际SHA>`及必要小型`--extra`凭据。缺启动文件时不传这两参数，保持null。该脚本只接受存在于项目内的安全相对路径，最终发布仍由根使用原Git44 `plan`核父提交恢复引用后进行。没有在这里执行parent Git blob核验；它是正式plan的必过门。
