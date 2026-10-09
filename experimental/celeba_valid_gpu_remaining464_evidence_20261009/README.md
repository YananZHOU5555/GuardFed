本包仅本机实现/无CNN验证，未连接服务器、下载新chunk、登记cohort或修改旧source/collector。根审阅后可一次处理一个464队列已远端闭合chunk。没有自动循环、跳过、覆盖、训练、推理或test。

evidence.py复用首1 root verifier的归档/保存数组流程，并直接提取冻结evaluator、原v2 check_native/canonical、sealed recovery.receipt_guard函数，不导入Torch。INPUT_PINS.json绑定436原字节、当前完整464 review、原900 inventory、完整旧implementation/queue seals、evaluator/valid标签cache39e5等；inputs/只有两个原metadata快照，无模型/数组重包。

每次下载前只读核guide SHA，然后获取当前chunk的原archive、remote_archive_inventory和REMOTE_PENDING_OFFSERVER收据。核archive及每成员SHA/size/重复/路径、前一chunk receipt SHA、原strict/batch/source-before-after、完整sourcefreeze、storage-map三个artifact身份、原record/config/seed/70round/weights/root/train/valid及明确cuda:0/UUID/单线程/nice10。三视图预测用原evaluator从保存margin重算（只还原JSON threshold整数键并核canonical fit），9指标及24混淆计数与receipt逐项一致；原native逐项复现阈值仍1e-12。root重新拟合与70round原result验收引用已绑定的原remote strict，本地不重新拟合、不读取旧模型。

--chunk 0默认从冻结436来源开始，成功另存单chunk ROOT_OFFSERVER_VERIFICATION.json与cumulative_447_accepted.json。之后每次显式--prior和--prior-sha256；只接受从436延续的已验收proof/collector SHA链及恰好之前chunk的ID前缀。任何差异停止并保留FAILURE_PRESERVED.json；缺失或partial不出collector，旧输入始终只读。父目录必须是本包目录内真实存在的目录，--output必须全新。

未来root审阅封存后首11命令（本次未执行）：

```powershell
python tmp/celeba_valid_gpu_remaining464_evidence_20261009/evidence.py --chunk 0 --package-sha256 <本包PACKAGE_SHA> --output E:/OneDrive/文档/GuardFed/tmp/celeba_valid_gpu_remaining464_evidence_20261009/chunk_000
```

下一chunk需显式绑定新collector，不能直接省略prior：

```powershell
python tmp/celeba_valid_gpu_remaining464_evidence_20261009/evidence.py --chunk 1 --package-sha256 <本包PACKAGE_SHA> --prior E:/OneDrive/文档/GuardFed/tmp/celeba_valid_gpu_remaining464_evidence_20261009/chunk_000/cumulative_447_accepted.json --prior-sha256 <该原文件SHA> --output E:/OneDrive/文档/GuardFed/tmp/celeba_valid_gpu_remaining464_evidence_20261009/chunk_001
```

--bundle DIR只核已经本地保存的三个同名输入文件，不联网；仍新建输出并完整验收，不跳过任何检查。下载路径固定当前已批准remote剩余464输出，只读SCP，不执行服务器命令。root须独立复核新proof后再将新collector纳入主状态；本包不写canonical状态/Git或旧账本。

selfcheck只读取实际已接受首1保存数组和metadata，9指标/24计数/3规则回归通过，18项篡改/错误来源/容差/重复/跳chunk拒收；436→447仅合成metadata结构检查，不是新11接受。所有临时文件限独占目录并清理；0网络/0CNN/0登记。一次本地检查夹具误写groups键引起KeyError，已按真实group_confusion_counts键修正，原科学函数未改；日志保留说明。

混合CPU/GPU逐IDdevice/source provenance保留，不据此称统一设备最终公平比较；正式native/shared主终点和final协议继续pending。只有900唯一科学ID及全部真实严格离机闭环才可完成；checkpoint内容SHA相同不去重。
