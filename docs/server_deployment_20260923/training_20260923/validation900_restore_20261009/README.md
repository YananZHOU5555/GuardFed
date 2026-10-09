# 九方法900终轮模型：独立恢复与验证重放入口

2026-10-09已在用户指定的89.22.197.55:60350恢复完成900模型、900原结果及900原job，共2700文件；这是字节身份验收，不等于900原图评价已完成。

- 原库存SHA：`3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd`，900行原字节保持不变。
- 100 Full的300现有文件只读核验，未重复打包。其他800从21份已验收历史归档抽取2400精确成员，恢复到服务器独立`/workspace/guardfed_checks/celeba_validation900_restore_20261009/artifact_store/`。
- `storage_map.json`显式绑定每个原id/kind、历史archive/member/hash/size及当前实际路径；包括8个原tuning产出与带空格的`F Flip`原id。未覆盖原部分输出、失败证据或历史job。
- bundle SHA：`fc3af63dc360648afbd8436b19adb945f4f2b0fa09fbb44e1853c77e13c414da`；369441467字节，2400成员本机及远端逐项核验。
- 远端实际恢复收据：`restore_acceptance.json`，SHA`5114c2cd96e5b8ffaf46e40a341619dd8b3547f89d417263c19c6e7f1f33bf77`；2400创建、300复用核验、2700 SHA通过、原output修改0、训练/推理0。

两条Full实测CPU原图重放已通过：IID Benign seed91001和non-IID Benign seed91002，每条完整valid19867/root16277。native三指标与原终轮结果误差0。主代理独立核52项封存文件、27归档成员、三个视图的18指标与48个混淆计数；收据`root_canary_independent_verification.json`。两次耗时108.022/121.936秒，单进程限制8核，仅两条门检，不能外推其余898条。

原图重放实现及原始收据在`../../../../tmp/celeba_final_valid_replay_20261009/README.md`（本地工作区入口）；后续v3需要显式核本目录storage map与恢复收据SHA，不得改写原库存路径或放宽1e-12。全部900仍待受限吞吐测量、完整运行及验收。正式最终评价协议仍`PREPARED_NOT_FROZEN`；没有test推理或测试集择优。

准备时两项路径守卫错误保存在`initial_path_guard_failure.json`：原路径还包括tuning、原id保留`F Flip`空格。均发生在bundle生成前；修正只用于忠实保留原身份。主代理首次独立计数helper错误比较完整支持字典与四项计数字典，停在断言后；修正helper后独立核验通过，没有重跑模型或改变原收据/容差。
