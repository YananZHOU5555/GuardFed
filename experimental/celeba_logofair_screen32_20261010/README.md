# LoGoFair32：固定虚拟人口的正式验证搜索准备包

**FROZEN 科学配置、SOURCE_ONLY、未启动。** 原8个候选×IID/non-IID×Benign/S-DFA，训练seed91001，fit seed1719。实读原协议每项 **30 post-rounds**，local/global各20步、beta1000、calibration=True；不是100轮、不是100 fullcoverage，也不新增fit-seed搜索。原FedAvg的70训练轮身份不变，CNN推理0。

作者裁定 AUTHOR_DECISIONS SHA aee89e8210b5aa83d8ee814d5afde4bb655f3ba559bb53ae6ebcd1ca0d46851d 已绑定。L由用户委托root决定，采用固定20个image-ID哈希虚拟cohort；只root labels进入官方DP/BetaCalibration拟合及校准prior，valid只用于评价/择recipe。不能声称真实训练client公平性，也不是EO分支或上游main.py无修改全轨迹。

原桥bridge.py、官方adapter.py、FedFairPostClient.py原字节复用，fit_predict/save_post/load_post/checked_output不改；只prepare的过期limitation/打印状态更新。新protocol解决4项状态，新reuse_manifest仅保留准确4个原接受引用，原100清单/旧proposal/门检/全部封条均不改。metadata批准副本绑定实际作者记录，mapping数组SHA保持原值；准备阶段只引用原数组，不把数组复制到E。SOURCE_DIFF.patch/SOURCE_REUSE.json给最小差异。

四引用准确ID为 FedAvg_IID_Benign_seed91001、FedAvg_IID_S-DFA_seed91001、FedAvg_lr0.002_Benign_seed91001、FedAvg_lr0.002_S-DFA_seed91001。后两条实际non-IID，保留原ID别名/source job，绝不按名字猜条件。REFERENCE_INPUTS记录原checkpoint/result/source_job/cache的archive/member/SHA，原reuse entries逐条相同。root执行stage_inputs时才重新hash原archive、逐成员提取到F；目前未重验大归档或加载模型。原始E备份只读保留。

真实3 post-round接口门检实有PASS：40Beta、root16277/valid19867、同保存重载预测exact，CNN/训练0。恒定负预测 ACC0.5166859616449389、AEOD0、ASPD0、positive_rate0 完整保留；**只证明接口，不是性能、被选recipe或正式32接受**。GATE_ACCEPTANCE原SHA e0508ea7f1eecc4e0e3e53c8b0bf13443cd63a2801aa97e7e8a01947181b0b4d。它仅覆盖首候选、一个条件、3轮，不保证其他候选/30轮收敛；后续原失败规则不变。

run_queue为一个CPU1线程串行worker，每job独立Python进程避免interop线程生命周期冲突，隐藏CUDA；原bridge.run完成原checked_output后才进入local strict进度链。全部模型/数组/pkl/日志/运行记录/汇总只到F。stage_inputs、queue启动/每job及summary前实核卷标Yanan 2TB、健康、容量和路径，无C/D/E回退；stage或输出已存在/partial/任何failure即拒绝覆盖，输入上的独占启动marker阻止重复32队列，无自动重试。没有新SSH/服务/STATE/Git更改。

逐项严格检查原FedAvg70轮、train162770/valid19867、同checkpoint/source/config/cache、root/valid ID与映射、原30轮history/有限阈值、40Beta状态、保存重载预测逐数组一致、全部三指标1e-12；精确tie、缺人口/标签支持和不收敛/nonfinite原样失败，不池化、不随机tie、不改beta。分数路径仍为原float32 sigmoid(margin)，可能与fresh softmax差1ULP；不做新CNN。

只有实际32全部原strict完成，summarize才读完整结果：**每候选每条件先调用原frozen_score，再平均四条件，最高score、精确并列candidate字典序**。保留所有候选/负结果/ACC冠军/三指标Pareto，n=1，无SD/显著性，四条件不是独立seed；汇总不自动采用recipe或启动100。原loader已暴露test属性/划分元数据，不能称untouched test，没有test图像推理/拟合/择参。

SELF_CHECK实际通过32覆盖、8候选/30轮、4原引用、原科学bytes、批准映射身份、少量metadata/存储拒收、内存MOCK精确tie/Pareto/31项拒绝选择；原14项未变checker结构门检继续复用。准备阶段实际F卷标/容量读取通过，但新bulk写0、拟合0、正式结果0；未来真实stage与运行均待root审阅后执行。
