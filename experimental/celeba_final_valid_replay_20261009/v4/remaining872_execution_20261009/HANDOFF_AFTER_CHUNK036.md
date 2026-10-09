原872队列因 FairGuard_IID_F-Flip_seed91009 原生指标不一致而自动 failstop；11:14:50Z 实测服务 EXITED，PID19149不存在，无 replay_v4 进程。未重启、未改科学源/阈值/1e-12容差，主800/FL/Hybrid服务未改，主STATE/RUNNING和共享latest未写；原3h monitor按parent记录仍PAUSED。

实际离机接受424个唯一冻结ID（旧28 + 新396），native最大误差均为0。033–035通过原工具独立闭合33项：3×64成员、297指标、792计数、99规则。cumulative_424_accepted.json与collection_inputs_424.json记录完整版本/输入/归档/证明SHA，不以checkpoint字节SHA去重。全900尚未完成，未生成完整性能表。

036的65成员失败归档已离机逐SHA核验，保存11项预测数组/收据与10/11 partial原记录；10项不计接受。failed_id的原model/result/job与weights均未变，44源/数据身份before-after一致。其ACC/AEOD/ASPD差为-5.03347259274145e-5、+0.0002902757619738239、+0.00011823126034521403。原result没有group confusion或逐图prediction；仅在单翻转假设下，差值兼容sensitive1/y1的原1→CPU0。CPU最近零margin为image_id172599、-2.60770320892334e-7，仅是候选，无法唯一确认实际翻转点或CPU-GPU根因。

476个missing = 036未注册partial10 + 数值失败1 + 从037起未派发465。下一历史计划路径为 /workspace/guardfed_checks/celeba_final_valid_replay_20261009/v4/remaining872_attempt1/chunk_037，当前尚不存在；不能直接启动该路径或恢复原872服务。独立GPU/原环境有界诊断须另冻结输入并获执行授权；不得连续重试。已有verify-chunk/collector完整chunk规则不改，不把partial批次伪装完整。此agent按parent要求结束本轮，不再追新组。

精确路径与SHA集中在 HANDOFF_AFTER_CHUNK036.json；失败输入映射为 ../remaining872_attempt1/chunk_036/failure_input_bindings.json，24项失败封存清单为该目录FILES_SHA256。输入映射含从原已核归档精确提取的单模型/job诊断副本与7项科学源码、CPU数组/receipt、valid-only标签缓存，供独立PREPARED诊断使用，不构成执行授权。
