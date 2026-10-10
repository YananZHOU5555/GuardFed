本次只处理固定 2026-10-10 01:07:18 UTC 观察快照相对已采用12项的两个终轮 ID：IID F Flip seeds91004、91005。完整ID及原观察SHA见 AUTHORIZED_SNAPSHOT.json。原96个新任务及4个历史复用的科学范围、recipe、seed、70轮、valid口径不变；本次不选择recipe、不生成场景均值。

实际新增2项通过原远端接受器、27成员归档逐SHA和本机原接受器复核。旧12项完整有序前缀保留；供root采用后的新任务累计为14/96，4个历史复用仍单列。本目录不是root采用记录，LATEST_BACKUP、STATE、Git及原科学包均未改。

collect_delta.py 仅在原wanted列表后添加固定2-ID过滤，完整观察仍保存，其他严格接受、source/data前后守卫和归档体逐字保留。verify_delta_offserver.py 与原冻结版本逐字相同。COLLECTOR_SCOPE_DIFF.patch、SOURCE_REUSE.json 和 TRANSPORT_REBIND.patch 给出确切变化。远端一次性 collector 使用CPU107、单线程、nice10、idle IO、CUDA隐藏；结束后CPU107已释放，FL服务继续RUNNING，双GPU100%，cgroup OOM为0。这是资源观察，未测定训练性能影响。

check_saved_tensors.py 在原接受器之外补核新2个保存checkpoint的全部16个张量（187012元素），对照原SHA冻结CelebACNN类的key/shape/dtype并核finite、文件前后SHA；只实例化类取得布局，无forward、optimizer、数据加载或训练。远端接受器Python3.12/Torch2.11.0+cu128，本机复核Python3.10.4/Torch2.8.0+cpu；验证环境不等于重建训练环境，不声称跨设备训练等价。

已执行入口（保留实际command/exit/stdout/stderr；不可盲重跑）：

```
python -B tmp/celeba_flgmm_fullcoverage_delta_after12_20261010/prepare_once.py
python -B tmp/celeba_flgmm_fullcoverage_delta_after12_20261010/dispatch_once.py
python -B tmp/celeba_flgmm_fullcoverage_delta_after12_20261010/transfer_verify_once.py
python -B tmp/celeba_flgmm_fullcoverage_delta_after12_20261010/check_saved_tensors.py
```

preflight.py 与 close_observation.py 实际通过短SSH argv+stdin调用，确切命令在对应 *_COMMAND.json。没有创建服务或启动新CNN。batch/accepted_delta.tar.gz 与 MEMBERS.json 是恢复链；batch/restored 是本机安全恢复的衍生树，不纳入交付seal或重复发布模型。ROOT_READY_CHAIN_LINK.json 供root独立审阅后连接原链；本阶段不root-adopt。
