# Exact14 CPU3：仅离机闭合源准备

本包没有运行SSH、compare、collector、模型/数组读取、forward、fit或训练。全部14尚未实际闭合；所有完成/comparison/archive/member/offserver哈希在ROOT_INPUTS.template.json为null，不能作为通过证明。192覆盖仍未授权。

实际scope：/workspace/guardfed_checks/celeba_gradient_fullcoverage_exact14_20261011；SHA22eba673a701d569b35736875245373d15696bca6a1ee9421165a5ed0e7232ff。14有序job来自原scope，不另造库存；两方法各F Flip/FedSA/Sp-DFA的screen/coverage配对及Benign参考，seed91002、每项3轮/42总轮仅是短接口合同，绝非70轮结果或CPU/GPU训练等价。

复用入口：
- 原四项close_realimage_gates_20261009.py的records循环及tar写入循环raw/AST逐字保留。export14只消费已存在且SHA晚绑定的原GATE_COMPARISON，不再调用compare或科学strict。
- 原verify_completed_four_offserver.py使用的evidence_v4.verify_archive原函数不改，继续检查safe member/唯一regular文件/归档、inventory和逐member SHA；它本身不做scientific strict。
- verify14额外调用原adapter.render_gate_functions()['checked']，绑定原gate的HERE到F恢复目录、STAGE到原14stage，复用原ARTIFACTS/数学/tensor_sha/check_attack_assignment。没有调用load_context，没有将Linux absolute external identities重写成Windows路径，没有共享cache/image加载。
- compare_saved.py原嵌套exact函数及八组配对循环从SHA固定原文件按AST/source取出原样执行；同checkpoint tensor、保存metrics/round/audit/RNG及F Flip metadata-only null规则均不变。
- 原after42 CPU fullstate检查块仅缩进/路径变量绑定，raw恢复及AST相同，八tensor布局/类型/finite检查无forward。原CNN类源仍按原job source SHA核实；不是重新训练或CPU/GPU等价门。

服务器先等待全部14 item_receipt/acceptance无failure、没有adapter/compare producer。root按原guide、source/data及全线程CPU owner门实核空闲collector CPU（例如110并不代表目前可用），记录fresh preflight和全部实际单项命令/退出码；原index0与index4首轮资源拒收/未启动adapter证据保持，不重试科学失败。然后仅一次调用原比较器并保存原命令/stdout/stderr/exit：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c <actual-free-CPU> nice -n 10 ionice -c 3 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B /workspace/GuardFed-celeba-expanded/tmp/celeba_gradient_fullcoverage_gates_prepare_20261010/compare_saved.py --stage /workspace/guardfed_checks/celeba_gradient_fullcoverage_exact14_20261011 --repo /workspace/GuardFed-celeba-expanded
```

原comparison已有输出或failure不能再次运行。本包不实现新的14摘要框架。其结果只允许EXACT14_CPU3_GATE_SAVED_COMPARISON_PASS/14jobs/3rounds/42rounds/scientific_table_records0/coverage_authorizedfalse。

root将export14.py及SOURCE_PINS.json按原SHA部署到stage/closure_source；实际RUNTIME_MEMBERS.json放同目录，shape为{files:{runtime/<actual-file>:{sha256,bytes}}}，仅精确列原ROOT_AUTHORIZATION、实际成功审批/命令/退出及两次原资源拒收凭据；不能加入未来或不相关文件。export输入文件必须是该固定路径。无producer时一次：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c <actual-free-CPU> nice -n 10 ionice -c 3 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B /workspace/guardfed_checks/celeba_gradient_fullcoverage_exact14_20261011/closure_source/export14.py --comparison-sha256 <actual-comparison-SHA> --runtime-manifest /workspace/guardfed_checks/celeba_gradient_fullcoverage_exact14_20261011/closure_source/RUNTIME_MEMBERS.json --runtime-manifest-sha256 <actual-runtime-manifest-SHA> --destination /workspace/guardfed_checks/celeba_gradient_fullcoverage_exact14_20261011/offserver_closure_<actual-tag> --cpu <actual-free-CPU>
```

导出器自身核guide、原source/scope/external identity、CPU实际affinity/nice/CUDAhidden单线程、所有任务线程的窄mask<=32冲突及producer；宽mask线程只计数，不能称独占host。原protected data hash在服务器核实；成员数从实际files推导，不写预测基数。归档只新14的完整output、原job/scope/cache路径metadata、comparison、精确runtime receipt、source与strict_delivery，不重打包旧4/64模型或图像缓存。任何失败保留stdout/stderr、已生成目录或部分归档，禁止盲重发。

下载前必须先调用原tmp/guardfed_local_storage.py:check_bulk_storage(actual_archive_bytes)，确认F=Yanan 2TB、Healthy与容量+1GB余量；不允许E fallback。仅SCP该新archive和receipt到F:/YananResearchStorage/GuardFed/gradient14_offserver_closure_20261011/<actual-tag>/。本包不执行下载器。下载SHA核对后，原CPU保存物检查命令如下（先在owned目录下创建精简actual output子目录）：

```powershell
python -B tmp/gradient14_offserver_closure_prepared_20261011/verify14.py --archive F:/YananResearchStorage/GuardFed/gradient14_offserver_closure_20261011/<actual-tag>/exact14_saved_gradient_canaries.tar.gz --receipt F:/YananResearchStorage/GuardFed/gradient14_offserver_closure_20261011/<actual-tag>/backup_receipt.json --receipt-sha256 <actual-receipt-SHA> --comparison-sha256 <actual-comparison-SHA> --restore F:/YananResearchStorage/GuardFed/gradient14_offserver_closure_20261011/<actual-tag>/restored --output tmp/gradient14_offserver_closure_prepared_20261011/actual_<tag>/OFFSERVER_VERIFICATION.json
```

verify14在读取模型或解包前复用fresh F guard；归档/receipt/restored均必须真实F路径。输出/failure/restored任何已存在均fail-stop，模型只map_location=cpu，CUDA隐藏；原source SHA先核。离机证明限定saved-artifact strict/model/RNG及source/data receipt lineage，**不**声称离机重新读取全图像数据、再次native forward、GPU等价或70轮性能。CPU torch版本另记，原run provenance保持2.11.0+cu128/8threads，不伪装本机CUDA。

只有root再逐actual proof采用，才能给coverage_contract消费ROOT_GRADIENT14_SAME_HORIZON_STRICT_OFFSERVER_ADOPTED（含all14_offserver_verified与0table records等）；本包不生成该根采用或启动192。scientific和资源失败完整保留，没有自动重试或参数/规则更改。
