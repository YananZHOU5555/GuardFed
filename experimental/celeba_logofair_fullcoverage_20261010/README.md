# LoGoFair 100 格完整覆盖：源码准备，尚未绑定或启动

已连接 100 个真实已接受 FedAvg 70 轮 checkpoint、原始结果和旧 margin cache 的身份；原 900 三视图中的相同 checkpoint/ID 数组作为人口顺序来源。没有重训、CNN、拟合、缓存生成、归档解包或大文件复制。本次只读 JSON 身份，未重新审计旧归档。

目标固定 IID(alpha5000)/non-IID(alpha5) × Benign/F Flip/FedSA/S-DFA/Sp-DFA × 91001..91010。完整32经原 summarize 的四条件平均 frozen score/候选字典序选出唯一 recipe 后，只新增96个30轮后处理，旧4个同 recipe/同四条件/seed91001结果显式引用。fit_seed一直1719，十seed描述的是原模型/分区种子，不能称十个独立后处理fit seed。

## 当前事实与缺口

- CACHE_IDENTITIES100.json逐格记录原 checkpoint/result/job archive/member/SHA、已接受 margin cache archive/member/SHA、历史服务器路径、已接受ID数组 archive/member/SHA。所有100身份与既有FedAvg100/900记录吻合；服务器路径是既有记录，本次未SSH确认现存。
- F盘现有4个参考目录与原mapping入口见 EXISTING_F_PATHS.json；旧E盘档案保持只读。其余96只需将既有字节按记录恢复/复用到F，不计算新的CNN/cache。本包不再实现一套归档运输器；沿用原stage_inputs的按archive分组/memberSHA取数方式，按该文件清单运输即可。
- 实际有10种root ID顺序、1种valid ID顺序。seed91001映射原字节必须保留；另外9个映射还未生成，SHA明确null。population.py只从已接受预测数组读取root_image_ids/valid_image_ids，复用原cohorts/check_mapping/arrsha，生成PREPARED_NOT_APPROVED映射。不会取valid标签、Male或score来定义cohort；文件本身含这些字段，代码不读取。
- 原20虚拟cohort已获作者授权，扩展时须由root对实际9个映射及100输入身份进行具体绑定批准；不是原真实训练client，不作EO公平性或训练client公平性主张。
- 32搜索尚未作为本包已闭合结果；BINDING_TEMPLATE全部未来选择/approval为null，未选recipe。仅凭目录/部分result不能执行。

## 最小适配

metadata.py：外部完整32摘要+实际ROOT独立接受SHA门，原summary函数重新算四条件分数并检查32原job/指标/完整artifact链接；100输入和10个实际mapping全部检查后才生成96新job/4引用、协议、源封条。新stage科学源码从冻结32 snapshot复制；仅validate_job的seed谓词改变，另外15个函数源字节完全相同。run/fit_predict/checked_output/序列化/1e-12/官方Beta/阈值规则不改。原 per-result evidence_stage=validation_postprocessing_screen 为兼容原strict保留；外层 manifest 明确 fixed_recipe_validation_postprocessing100，不能把100说成新的参数搜索。

run96.py：复用原顺序fresh-child、单CPU线程、CUDA隐藏、F-only、fail-stop结构；先验外部实际执行批准及封存source/manifest，再执行96。4旧结果只核字节引用，不重拟合、不重新推断。只报告local strict与root待验收，offserver/root计数0。出现任何partial/failure不自动继续，FULLCOVERAGE_STARTED排斥重启和别输出目录重复启动。无service/新监督机制。

## 待执行顺序和限制

1. 完整32原strict及独立离机接受→root摘要采用；按schema提供实际summary/index/adoption SHA。当前不存在的未来SHA不得填示例值。
2. 用既有恢复工具仅恢复所需100参考与9个ID数组到F；每文件原SHA核验。population.py分别准备9映射，root核身份与原域/支持/缺组/threshold tie政策后批准元数据；seed91001的原mapping/meta不改。
3. root提供明确100 inputs receipt与绑定批准→metadata.py生成新stage；审生成source/job/protocol差异。无需重复旧CNN/已过3轮接口gate。新seed身份/人口支持尚未实测，首个新seed真实30轮后处理本身保留可验收结果；若要额外短门检，须root另行明确范围，不能当70轮CNN或性能证据。
4. root实时资源核验并提供execution批准，run96.py只用一个本地CPU线程/原screen环境；保存全部负结果、constant-negative、tie/缺支持/数值失败，不自动重试或换规则。全部100最终按共享10/9/6seed统计原ACC/AEOD/ASPD；场景不作为独立seed，跨场景先seed内平均。

既有seed91001参与验证选参；原FedAvg环境有混合cu128/cu130/CPU-GPU重放历史。margin→sigmoid沿原浮点转换，不宣称与新softmax逐位相等。AEOD沿原绝对TPR差定义；DP后处理而非EO。原loader曾可见test属性/分区元数据，不称未触碰test；本包无test拟合/推理/选择。不保证胜过其他方法，不更换负结果。
