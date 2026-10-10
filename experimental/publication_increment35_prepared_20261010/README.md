# Increment35：source only

复用成功34-v2的精简发布器与51行commit verifier；父HEAD固定`ae94e17a7b1c3b3f6298eaa9d7f09bbf31cca17d`，branch仍`codex/revision-evidence-baselines-20260928`。本目录仅源准备，未调用Git、stage、commit、push、SSH或科学计算。

目标闭合口径为baseline valid900、native150/three-view150、FL新9、Hybrid19。native150已接受来源及after47源包SHA固定于PREPARED_INPUTS；after47实际ROOT_ADOPTION/source review、FL after7、Hybrid after18、状态/只读live、previous34 remote proof均由root向外部输入JSON注入实际路径/SHA。不会用模板或source-ready状态替代实际闭合。准备阶段未读正在变化的这些新运行证据。

Root将ROOT_CLOSED_INPUTS_TEMPLATE复制到自己的输入文件，填实际闭合后令status=`ROOT_CLOSED_INCREMENT35_INPUTS`、counts取required_closed_counts。closure_pins中的7项均必须真实存在且逐SHA通过。previous_publication使用已接受的increment34远端证明。extra_pins为精确allowlist：填全部required_extra_paths SHA，并加入本批实际after47 root operations/独立source review封条及helper、FL/Hybrid链接与观察、root updater等源文件；不加入已发布34源审查/prepare-repair包、旧归档、模型或恢复目录。

全部闭合与源检查通过后才会创建output或修改发布工作树/index。原`git add -f`精确allowlist、`-text`+renormalize、index/commit blob SHA保持；拒绝已在父commit同路径发布的tar，当前批归档SHA唯一，总文件体积严格<100MB。排除restored/verified/verified_extract/pycache、raw `.pt/.pth`与凭据。

root未来实际CLI（本阶段未执行）：

```powershell
python -B tmp/publication_increment35_prepared_20261010/publish_increment35.py --closed-inputs ROOT_ACTUAL_CLOSED_INPUTS.json --closed-inputs-sha256 ACTUAL_SHA256 --output tmp/publication_increment35_prepared_20261010/actual_stage_once --execute-stage
```

审index proof后，root自行commit；本源码不commit/push。提交后及push后分别调用：

```powershell
python -B tmp/publication_increment35_prepared_20261010/verify_increment35.py --receipt tmp/publication_increment35_prepared_20261010/actual_stage_once/publication_closed_increment35_20261010.json --receipt-sha256 ACTUAL_RECEIPT_SHA256 --output tmp/publication_increment35_prepared_20261010/committed_verification.json
python -B tmp/publication_increment35_prepared_20261010/verify_increment35.py --receipt tmp/publication_increment35_prepared_20261010/actual_stage_once/publication_closed_increment35_20261010.json --receipt-sha256 ACTUAL_RECEIPT_SHA256 --output tmp/publication_increment35_prepared_20261010/publication_closed_increment35_verified_20261010.json --remote
```

C50表可选。尚无actual交付时保持`C50_table=null`，其他闭合证据可单独发布。若root已实际采用C50，填实际directory/root_proof/seal路径及SHA、实际ROOT proof中绑定C3 adoption的字段名。仅接受100记录/50对/5场景、810 mean/SD标量、405展示cell、900计数指标、10/9/6面板及negative/valid-only/pending-author限制的真实root采用证明；不重算统计、不预填未来hash/数值。表封条成员被逐SHA加入。

SELF_CHECK仅无Git/no-network结构检查与明确标注的内存closure fixture；不是实际C3 adoption、Linux核验或发布。首次只读探针遇尚未存在的source review路径，已保留READ_PROBE_NOT_READY；未改变科学或索引。任何实际执行失败保留output/index，不自动重试、不回滚旧证据。
