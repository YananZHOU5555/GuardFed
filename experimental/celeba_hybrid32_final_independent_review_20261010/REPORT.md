# Hybrid32 最终增量独立审查

**PASS，可由root采用；本审查未采用、未改共享账本。** 27条既有接受记录与本批精确5条差集组成原32条冻结网格：8候选×IID/non-IID×Benign/S-DFA，全部seed91001、round70、valid19867。

实际固定交付源为 `tmp/celeba_hybrid32_final_collection_20261010`，82成员封条SHA `4370962e33b24e7fe9c5ffb461b1a99adc39621bc46efaaf25deb28c64b26e59`。独立核对82个小交付文件、69个原源文件、146个F索引文件、新71个归档成员及SHA/大小；13段既有root/strict/offserver链恢复27条原记录，未重验旧模型或旧归档成员。新5条与原manifest/job/config/source/data合同、round70、checkpoint/result/acceptance SHA相符。数据身份依托已绑定的原strict before/after收据；未在本机读取图像或重新验证原大数据集。

首collector失败是 `supervisorctl status` 对正常EXITED返回3，被check_output拒绝，发生于Torch/strict/archive之前。原FAILURE及非零运输凭据完整保留。独立源码比较确认v2仅改为精确服务名、空stderr及RUNNING/0或EXITED/3状态读取，并使用独立运输namespace；原科学strict循环字节和AST相同。32项终轮/无failure状态由实际快照绑定，不以EXITED本身代替接受。

独立使用math.fsum复算全部8候选的四条件分数均值与三指标均值，共32个标量；最大差为 `1.1102230246251565e-16`。冻结规则为先逐条件计算score，再平均4条件，完全并列按candidate ID字典序。排序、ACC冠军和四条件均值三指标Pareto集合均与原SUMMARY32一致：

- 候选：`CosineFairness_lam20.0_tau0.1_lr0.001`。
- 四条件平均ACC/AEOD/ASPD：`0.863429304877435 / 0.03960062449126897 / 0.08554014846132205`。
- 冻结score：`0.8381717130951234`；与第二名差 `0.00346216988895387`。

所有8候选及32条原记录保留，32条终轮的保存positive-rate元数据显示0个恒定预测。审查JSON中`negative_candidates_preserved=8`表示全部8候选（包括不利候选）未筛掉，不是8个恒定或失败实验。上述胜出及Pareto仅适用于这4条件均值和原固定score，不代表每条件/每指标、其他seed或正式100的优势。n=1，无sample SD、显著性或把四场景当四独立seed的结论。没有保存预测数组，故没有数组指标复算；本次没有运行原strict、Torch、模型/数组加载、fit、CNN、训练、SSH或Git。

审查工具兼容历史保留：V1在任何校验前遇到Python3.10缺少hashlib.file_digest；V2误将最终交付根目录内成功收据副本视为原失败现场，并因x模式保留V1错误文件；V3错误假定外层运输rc也为3，实际外层Python因内部CalledProcessError退出1。最终V4仅修正流式SHA、真实收据布局/返回码和独立失败文件名；算术、归档、来源与身份检查不变。相关源版本和失败记录均封存，没有科学失败被重试，也未改作者包。

最终证书：`ROOT_INDEPENDENT_REVIEW.json`，SHA `40f411717385b14c6d5f8885dba3c0d2c9757a8a67a7d41f2afc709490405789`。原归档SHA `ddf22d96348918af02d22db1be7007d6c7400048fc3908cf6ad127af6177cfb6`；SUMMARY32 SHA `46b5f8fdca9536166ed868e50d4c7bc2578f8a1100ffea97878cf044c95748ae`。设备、环境和日志prefix的原限制保持，不声称CPU/CUDA等价、正式100已执行或final test完成。
