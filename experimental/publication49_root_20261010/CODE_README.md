# Git49薄接线与实际冻结入口

三个新入口共约200行：publish_increment49.py、prepare_spec49.py、verify_increment49.py。parent固定ebf8436126687e044262fabb90a0848f39275f52，F输出increment49。原48封包/旧准备3文件不改。原plan/stage只有scope/schema/comment metadata变化；prepare原build保留source SHA、seal逐成员与receipt45–48连接/路径映射/父恢复引用循环，仅换49 roles/pins/目录选择。Hybrid旧members(path/size)封条只读归一为原files(sha256/bytes)，逐成员校验不删。

原48 freeze只冻清单，无法在后续STATE变化时重现旧bytes。本次F/source001冻结所有选中compact文件；原source函数__code__原样，以独立globals仅绑定其ROOT到该F副本。base.ROOT、base.ns ROOT保持实际E仓库；storage/Git alternates/env/CHECKOUT/ORIGIN/BRANCH安全守卫不变。stage读取F，未来不会读取八个E共享入口。源快照本身SHA必须绑定，不能静默回退E。

actual closed inputs要求native236/views236/FL38/gradient18/Hybrid32，Hybrid100147源/config、7个3-round gate与96正式启动单列，正式70轮新增接受0。A36是FedSA补2和S-DFA6，不是A100；不增统计/recipe/test/goalcomplete。源码检查9个actual roles共46条pointer；2旧cutoff和未closed拒收，3文件compile通过。未重复旧科学验收/48全207blob。

后续root可审真实ACTUAL_SPEC.json、SOURCE_SNAPSHOT/INPUT_FREEZE_RECEIPT、ACTUAL_HANDOFF。prepare命令仅在ROOT_CLOSED_INPUTS真实SHA齐备后执行：

```powershell
python -B tmp/publication49_root_20261010/prepare_spec49.py --closed-inputs tmp/publication49_root_20261010/ROOT_CLOSED_INPUTS.json --sha256 CLOSED_INPUT_SHA --out tmp/publication49_root_20261010/ACTUAL_SPEC.json
python -B tmp/publication49_root_20261010/publish_increment49.py freeze --input tmp/publication49_root_20261010/ACTUAL_SPEC.json --sha256 ACTUAL_SPEC_SHA --output-name inputs001
```

stage/commit/push由root另行审查执行；stage使用交付的F/inputs001/FROZEN_INPUTS.json和其真实SHA。远端验证仍为原函数：verify_increment49.py --receipt ACTUAL_RECEIPT --receipt-sha256 ACTUAL_SHA --commit ACTUAL_COMMIT --remote。每次F写入继续fresh卷标Healthy与容量守卫；不采用POST49观测刷新截止范围。
