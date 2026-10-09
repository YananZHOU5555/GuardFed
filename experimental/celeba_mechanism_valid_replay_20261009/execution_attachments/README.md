# 单项执行附件：等待root明示freeze

只准备 `minus_U_IID_Benign_seed91002` 的一次70轮终轮valid三视图重放；CPU112..119、单进程8线程、nice10/idleIO、cu128 CPU。原Full同条件引用父代理phase3严格接受/离机证明，不重新推理或备份。不开另7条、不训练、不读test图像。

原prepared桥13成员seal保持原样，本目录另封。`dispatch_receipt.PREPARED.json` 是拒绝执行的模板。父代理审阅实际source、CPU预算后，才可创建独立 `dispatch_receipt.APPROVED.json` 和其真实SHA文件；服务不生成批准、不自动重试。当前未注册/启动supervisor或实际图像重放。

运行入口复用桥的 `bind_runtime -> replay_one -> accept_saved_predictions`，成功还必须生成 `strict_acceptance.json`；仅进程退出不算通过。既有partial、failure与收据不覆盖。原科学body receipt独立保留，bridge proof额外绑定source/data/artifact前后、完整三视图和same-checkpoint。

运行前检查原prepared和execution两套seal/外部receipt SHA、单ID/CPU/输出，识别当前正式GPU1线程worker、baseline valid8线程worker及3个CPU门检预约，名义合计再加8不得超实际cgroup-v2配额。phase5的88线程不得在FLGMM释放8核或本任务退出前叠加；父代理仍须串行协调，不能只凭CPU不重叠就认为配额足够。

父代理批准后的正常supervisor配置是 `supervisor.conf`，autostart/autorestart均false、无额外巡检/端口。`service.sh`使用原镜像normal logging/environment wrapper。资源检查不会终止其他进程或改变其affinity。

准备期本地检查仅source/schema/批准门，实际运行、source路径可达性、native误差/耗时/RSS仍待授权后实测。任何真实数值/身份错误保留证据停止，不放宽1e-12。
