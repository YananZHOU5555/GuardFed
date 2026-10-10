# root审阅后的可执行入口；本包没有执行以下三步

使用已完成真实gate的同一隔离Python，沿用netcal1.3.6。所有新输入副本/产物均在F；不要把后面的路径改为E或相对cwd。

```powershell
$logofairPython = 'E:/OneDrive/文档/GuardFed/tmp/celeba_baselines/remaining_20261009/group_a/.venv/Scripts/python.exe'
$logofairSource = 'E:/OneDrive/文档/GuardFed/tmp/celeba_logofair_screen32_20261010'
$logofairStorage = 'F:/YananResearchStorage/GuardFed/logofair_screen32_20261010'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
& $logofairPython -B "$logofairSource/stage_inputs.py" --out "$logofairStorage/inputs"
# 确认exit0后下一步；不自动重试或清理已有attempt。
& $logofairPython -B "$logofairSource/run_queue.py" --repo 'E:/OneDrive/文档/GuardFed/tmp/revision-publish-20260928' --inputs "$logofairStorage/inputs" --out "$logofairStorage/attempt001"
# 仅当32原strict全部闭合、无QUEUE_FAILURE后执行。
& $logofairPython -B "$logofairSource/summarize.py" --index "$logofairStorage/attempt001/STRICT32_INDEX.json" --out "$logofairStorage/attempt001/SUMMARY32.json"
```

上面是独立步骤，root须检查每步返回码；切勿在失败后继续下一步。程序亦有完整输入receipt/源seal/已有目录/failure/32集合守卫。stage_inputs只提取4×model/result/source_job/cache及原mapping，不拟合、不重新打包旧归档。queue的同源启动marker保留，不支持重入。模型加载仅用于验证原checkpoint有限性，CNN调用仍为0。

完成证据位于F：inputs/INPUT_RECEIPT.json → attempt001/DISPATCH.json → 逐job的acceptance.json/result.json/post_state.pkl/scores_predictions.npz/原mapping → STRICT32_INDEX.json → SUMMARY32.json。原strict与所有artifactSHA保存，root独立复核/采用尚待实际结果；E源包不写未来accepted或recipe。

本机已经执行过的唯一检查：`python -B tmp/celeba_logofair_screen32_20261010/check_prepared.py`，只是source/job/mapping/存储及内存MOCK回归，未运行上述stage/queue/summary。
