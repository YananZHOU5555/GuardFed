# V3 saved closure — source-only handoff

本次只派生一次 v3 closure。原 v1/v2 closure、失败证据、旧90条与4个screen引用保持不动。源自封存v2 closure `b8e3d5a20b98a96d8ed742096a40b4758a934998b5e652e2e1fa133f1e11d126`，绑定candidate package `43534f4770eebae89becc2a132fe087ecc716479d77d3f12af9f2e8462ec82ec`，runtime仅允许v5。

已核实际 START 原stdout SHA `eee1ec8f14ba152f2e341f4d5cd5be20cc03be4852fc5d5ca763ee4a284f352c`；JSON与stdout内容完全一致，记录2026-10-11 05:45:42 UTC服务 `guardfed_flgmm_SpDFA10_isolated_v3_valid` RUNNING/PID111602。这里只证明该时刻已启动，不证明当前仍运行、canary已过或exact10完成。

实际deployment/preflight/auth/start及candidate root review逐个路径、字节数、SHA在ACTUAL_STARTUP_PINS.json，RUNTIME_RECEIPT_PATHS.json已绑定真实v5路径。binder同时硬绑定两份文件的SHA，并要求CLI回执SHA与已实测pins一致；不能用旧candidate或新一轮回执替代。本closure ROOT_SOURCE_REVIEW仍是待根审核模板，source_adoptable=false。

20份文件有完整逆变换；saved_science、verify_arrays、原table/build、verify、input_contract及OLD_SCREEN4_REUSE保持整文件逐字，其他源只适配namespace/package/service/已存在启动metadata闭包。Linux原whole checker的10次cached-root refit、Windows零fit计数循环、同checkpoint/native1e-12/10个seed与统计规则均未修改。所有将来inference/gate/Linuxwhole/transport/Windowsaudit/root100/table100证据仍null。

## 根执行门与顺序

必须先实际核 exact10 GATE_RESULT全通过、服务EXITED、无FAILURE/producer；单独RUNNING/EXITED都不是完成。随后根审核closure source并填ROOT_SOURCE_REVIEW.json。原README中的Linux checker、最小transport、Windows零fit、root closure和table命令已全部改到v3 namespace，按顺序执行即可，不重启candidate。执行Linux whole前重新核CPU110与资源；transport/根采用前重新核F卷Yanan 2TB。禁止用本包制作未来科学回执。

以下binder命令中的启动SHA已是实际值，ACTUAL_GATE/LINUX/TRANSPORT/CLOSURE_REVIEW仍必须等实际文件：

```powershell
python -B tmp/fl_isolated_v3_saved_closure_prepare_20261011/actual/bind_audit.py --runtime-paths-sha256 0d70c5cd0e39843ee146b331ec01c59d83361068a6cfb2d65074ffe7cbdbfe43 --gate-result-sha256 ACTUAL_GATE_SHA --linux-proof-sha256 ACTUAL_LINUX_SHA --transport-proof-sha256 ACTUAL_TRANSPORT_SHA --source-review-sha256 2af6b581e33e6de99ac186d5581ebbaa9977c5e39efa49120aedd635de3c0a8e --linux-preflight-sha256 90a6a4977e6ec5d0533fa098cebd57c3e6a50692480be2507051ff748c62c0c4 --source-deployment-sha256 64b04b3849bafda0855b043a29e07e4488c66d532dde1c12c874ea086fd0f377 --authorization-sha256 0c78adfc16f9282583e5aee9c80e6b221821179f46372bace0f03b344488df8a --start-receipt-sha256 4ec5bc6e14f88e46d88dfa5b1b385e9b8c9ba07847dcd4e084154c6e9939ee6e --closure-source-review-sha256 ACTUAL_CLOSURE_REVIEW_SHA
```

根source review的 `runtime_paths_sha256` 为 `0d70c5cd0e39843ee146b331ec01c59d83361068a6cfb2d65074ffe7cbdbfe43`，`saved_static_seal_sha256` 为 `244f2fb8634d21da31b61293bc88f325822ed50abc3a236074432a6a781a4184`。不需要再填一个runtime路径模板；不要重新运行derive_v3.py。

root_closure模板19角色中仅既存静态/parent90/旧failure及真实candidate root review已绑定；其余未来科学角色留空。table模板仍root_adopted=false：完成100条三视图后才运行原表统计，要求旧90对象/486标量/243展示格exact，目标100记录/540标量/270展示格；不改变10/9/6seed视图，不择seed。

211项静态检查通过，20个完整inverse；只执行源编译、哈希、JSON闭包及纯metadata路径fixture。无SSH、F访问、模型/数组读取、科学模块import、CNN、fit、统计或STATE/Git修改。源码准备不是结果验收，旧Windows47失败与remaining620失败仍有效保留。
