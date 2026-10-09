原32条验证搜索的接续与备份边界

当前队列只包含原8候选×4条件，seed91001、70轮、valid-only。源69成员封条与每项job/runtime协议绑定，已有CPU4/CUDA4仅作为管线先决证据，不计为新的正式实验。不得自动启动100、多seed、test或失败重试；normal supervisor 的 autostart/autorestart=false 不变。

启动证据只证明队列实际启动和首轮推进，不等于70轮接受。服务退出也不等于32项成功。巡检读取本阶段screen_scope、APPROVED_screen、screen_dispatch、各任务progress/acceptance/failure、screen_failure与screen_complete，并与上次真实轮次比较。不要写或覆盖其他阶段状态入口。

部分完成时复用本包未修改的 driver.approve(...,dispatch=False)、driver.functions(body,scope) 返回的 checked；scope 只补入批准的 runtime_cuda_visible_device 和 runtime_gpu_uuid。校验完整70轮、原config/seed/alpha/attack/source/data、同终轮checkpoint所有指标、全部9个scientific artifact及sidecar/acceptance；body.checked 会CPU加载模型并查询GPU名称，需要保留CUDA_VISIBLE_DEVICES=0但不进行CNN推理。只检查真正已有acceptance的完成目录；活跃任务的中间result/缺失acceptance不是终态失败。发现实际failure或身份/数值错误保留证据并停，不盲重启或覆盖部分输出。

检查进程使用已核空闲CPU单线程/nice10/idleIO，不与健康screen worker的CPU104绑定重叠，不改变训练并发。读取其他已退出队列的旧资源快照不能作为当前占用事实。GPU瞬时低利用率不能作为重启原因。

关键批次按已严格接受ID减去已离机验收ledger的ID，得到差集；每个新ID只归档一次model.pt、result.json、diagnostics.json、provenance.json、resource_before/after.json、rng_final.json、native_replay.json、undefined_diagnostics.json、acceptance.json、progress.json。保留实际失败/partial及负结果，单独附证，不以删除失败换取成功。raw job/config由本次源码启动恢复链中的screen_jobs和result.revision_job绑定，无需重复包旧源码、旧门检或旧模型。

每个增量必须保存inventory（成员相对路径、SHA256、size、实际accepted_new_ids）、archive SHA、前一receipt SHA/source-startup引用、当前source69与scope/protocol/APPROVED身份；BlueBook使用已封evidence_v4.verify_archive逐archive/member核验后才能追加ledger。相同ID/model/未变产物不重复备份。恢复链以本次source/startup增量和先前CUDA source/startup81（archive SHA1876769e6073b00ef1131d57a855611d60ab6863f144fefb3bbd20e7ac07d791）的显式reused_members映射重建；reuse仅是已核原字节，不代表复用旧实验结果。

全部32项终轮且无失败后，仍复用未修改的 summarize.py：

    /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B /workspace/guardfed_checks/celeba_hybrid_screen_execution_20261009/summarize.py --approved /workspace/guardfed_checks/celeba_hybrid_screen_execution_20261009/APPROVED_screen.json --approved-sha256 <实际64hex>

调用前为该只读检查选择当时已核空闲CPU并nice10/idleIO；summarize本身固定Torch1线程、CUDA0，逐条原checked而不重新推理。summary必须含all32/8候选/四条件综合分平均、精确并列candidate字典序、accuracy冠军、三指标Pareto。n=1，不报sampleSD/显著性，四条件不是4个seed；score不替代原三指标。只有strict+全32增量离机闭合后才报告搜索阶段完成。当前仍无formal100/test或方法胜出保证。

沿用原loader可能物化含test尾部的属性/划分元数据；无test图像/推理/拟合/选参，不声称untouched test。
