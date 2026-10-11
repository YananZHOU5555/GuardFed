from pathlib import Path
import ast,datetime,hashlib,json,re
R=Path.cwd();T=R/'docs/server_deployment_20260923/training_20260923';V=R/'docs/server_deployment_20260923/revision_20260923';H=R/'tmp/root_CUDA7_resume1_Hybrid29_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_bytes())
ref=lambda p:dict(path=Path(p).relative_to(R).as_posix(),sha256=sha(p))
sp=T/'TRAINING_STATE.json';before=sp.read_bytes();assert sha(sp)=='79f051b5c824b4c91b3a1a496d60e936394c76662b8e480f8ea1717d9ea9a50b' and not H.exists()
s=json.loads(before)
cudaroot=R/'tmp/uniform_cuda7_saved_audit_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json';hybridroot=R/'tmp/celeba_hybrid_native29_IIDFedSA10_root_adoption_20261011/ROOT_ADOPTION.json'
obs=T/'server_reactivation_20261009/ROOT_COMPACT_20261011T064401Z.json';cudaobs=R/'tmp/uniform_cuda500_root_dispatch_20261011/RESUME1_OBSERVATION_20261011T064429Z.json'
install=R/'tmp/uniform_cuda500_root_dispatch_20261011/RESUME1_INSTALL_COMMAND.json';protocol=T/'celeba_uniform_cuda_evaluation_v1/PROTOCOL.md'
for p,h in [(cudaroot,'3ed2e1fe79edbc5c196d428305bf27201856f2e6d5589c6e4752255accec4dda'),(hybridroot,'2d9c518b72a2c7baf672ee9b3231c9e9c96831053bdf2048281148ba95e42f56'),(obs,'97fcd632a29df8593a739a179113278df0282bb32ee39b0aee034d90fa6fa4f6'),(cudaobs,'cc9d592785478dd61074d0d926f08e58353e8b08b193d7c2913cbae1a7042204')]:assert sha(p)==h
assert read(install)['exit_code']==0
o=read(obs);co=read(cudaobs);hp=read(hybridroot);cr=read(cudaroot)
assert cr['accepted']==7 and hp['cumulative_accepted']==29 and len(co['progress.json']['remote_saved_accepted_ids'])==11 and not co.get('QUEUE_FAILURE.json')
u=s['uniform_CUDA_evaluation_v1'];u.update(status='FIRST7_STRICT_OFFSERVER_ROOT_ADOPTED_RESUME490_ACTUAL_PROGRESS',protocol=ref(protocol),uniform_CUDA_scientific_accepted=7,first7_root=ref(cudaroot),first7_native_max_difference=0,actual_phase_frozen=500,current_remote_saved_accepted=11,current_remote_new_after_resume=1,current_remote_active=co['progress.json']['active'],current_remote_pending=488,current_remote_failed=0,current_observation=ref(cudaobs),current_service='guardfed_uniform_cuda500_resume1_valid',current_base='/workspace/guardfed_checks/uniform_cuda_formal_20261011_resume1',current_mechanism_base='/workspace/guardfed_checks/mechanism_cuda_formal_20261011_resume1',resume_source_review=ref(R/'tmp/uniform_cuda500_proc_race_recovery_prepare_20261011/REVIEW_SUCCESSOR.json'),actual_resume_command=ref(install),prior10_original_CUDA_reused=True,CPU_outputs_reused_as_CUDA=False,one_failed_record_replay_completed=True,whole500_complete=False,whole_comparison_complete=False)
s['uniform_CUDA_infrastructure_interruption_20261011']=dict(status='PRESERVED_AFTER10_RESOURCE_PROC_PERMISSION_ERROR_EXPLICIT_RECOVERY_SAME_MODEL_PASSED',failed_id='GuardFed-AD2+_IID_Benign_seed91005',diagnosis=ref(R/'tmp/uniform_cuda500_root_dispatch_20261011/FULL91005_DIAGNOSIS_20261011T063314Z.json'),original_partial_native_difference=0,original_partial_formally_adopted=False,historical_zombie_cause_proven=False,fix='PermissionError re-read proc stat;skip only verified disappeared/Z/X;otherwise fail closed',scientific_worker_raw_text_unchanged=True,old10_source_artifact_stat_gate_passed=True,original_failure_and_backup_failure_preserved=True,new_namespace=True,recovered_full_native_difference=0,automatic_retry=False)
h=s['hybrid100_fullcoverage_20261010'];assert h['new_accepted']==19;h.update(status='FORMAL96_RUNNING_NATIVE29_STRICT_OFFSERVER_ROOT_ADOPTED',new_accepted=29,accepted_ids=hp['accepted_ids'],latest_delta_root_path=ref(hybridroot)['path'],latest_delta_root_sha256=sha(hybridroot),latest_delta_archive_sha256=hp['archive_sha256'],latest_delta_offserver_sha256=hp['offserver_sha256'],latest_delta_utc=hp['utc'],limits='29 of96 new native records accepted;4 screen reuses separate. IID FedSA10 strict offserver closed;three-view20 still legacy CPU,not whole100 or finaltest.',latest_measured_observation=dict(utc=o['utc'],terminal_observed=32,source=ref(obs),acceptance_unchanged_by_observation=True))
m=s['celeba_mechanism_v1'];m.update(queue_completed_observed=416,queue_active_observed=8,queue_pending_observed=376,queue_failures_observed=0,queue_observation_utc=o['utc'],queue_observation_path=str(obs),queue_observation_sha256=sha(obs))
s['latest_current_resource_observation']=dict(utc=o['utc'],source_ref=ref(obs),main_terminal=416,main_active=8,main_pending=376,main_failed=0,NGA_terminal_observed=15,NGA_new_accepted=9,Hybrid_terminal_observed=32,Hybrid_native_new_accepted=29,GPU_utilization=[100,100],GPU_C=[67,62],GPU_RecoveryAction=['None','None'],memory_current_B=int(o['cgroup']['memory.current']),memory_max_B=int(o['cgroup']['memory.max']),disk_free_B=o['disk_free_B'],cgroup_CPU_quota_cores=122.87999,cgroup_CPU_measured_cores=o['measured_cgroup_CPU_cores'],cgroup_OOM=0,not_new_scientific_acceptance=True)
s['latest_NGA96_readonly_observation']=dict(**ref(obs),utc=o['utc'],terminal_observed=15,accepted_new=9,active=o['NGA']['active_progress'],Huber_started=False)
s['NGA96_runtime_dispatch_20261011'].update(latest_readonly_observation=ref(obs),terminal_result_directories_observed=15,latest_observed_rounds=o['NGA']['active_progress'])
reader=V/'AUTHOR_REVIEW_START_HERE.md';oldreader=reader.read_bytes();r=oldreader.decode('utf8')
a='新CUDA结果尚未验收。';b='新CUDA首批7项已完成原Linux严格验收、离机零拟合核对、F盘110成员SHA备份及root采用；服务器首阶段累计11/500验收，进程检查中断已在新目录有限恢复。完整CUDA表尚未完成。'
assert r.count(a)==1;r=r.replace(a,b)
sentence='本页不是实时监控。';assert r.count(sentence)==1;r=r.replace(sentence,'本页不是实时监控。统一CUDA协议与恢复边界见[本阶段协议](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_uniform_cuda_evaluation_v1/PROTOCOL.md)。')
for link in re.findall(r'\[[^\]]+\]\((E:/[^)]+)\)',r):assert Path(link.split('#',1)[0]).is_file(),link
tree=ast.parse((R/'tmp/refresh_native370_FL86_Hybrid20_root_20261011.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='entries' for x in n.targets));ns={'R':R,'T':T};exec(compile(ast.Module(body=[node],type_ignores=[]),'<tailguards>','exec'),ns)
entries=[];prefix=None
for p,marker,pin in ns['entries']:
    raw=p.read_bytes();i=raw.index(marker.encode());tail=raw[i:];assert hashlib.sha256(tail).hexdigest()==pin
    if prefix is None:prefix=raw[:i]
    assert raw[:i]==prefix;entries.append((p,raw,tail,pin))
now=datetime.datetime.now(datetime.timezone.utc).isoformat();H.mkdir();(H/'PRIOR_STATE.json').write_bytes(before);(H/'PRIOR_PREFIX.md').write_bytes(prefix);(H/'PRIOR_READER.md').write_bytes(oldreader)
newprefix=f'''# CURRENT: GuardFed返修实验

入口更新：{now}。服务器 ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded；先遵守/etc/vast-agents-guide.md。结果采用与实时观测分开。

| 阶段 | 严格验收、F盘SHA与root采用 | 最新实测及边界 |
|---|---:|---|
| 机制训练 | 新400/800，U/C/A/F各100；Full100复用 | {o['utc']}：416终轮、8活动、376等待、训练失败0；其余4类继续 |
| 统一CUDA首阶段 | 已采用7/500；服务器严格保存验收11/500 | {co['utc']}：新目录已恢复Full/IID/Benign/91005，diff0；当前91006，488等待，当前失败0 |
| 历史CPU三视图 | 机制380（U/C/A100、F80）、FL100、Hybrid20 | 原620数值失败与旧表保留，不重启原CPU队列；新CUDA表独立产生 |
| 主比较表 | 11方法native1100，三页PDF | IID/non-IID各5场景×10seed，另9/6seed；历史评价协议，非最终17方法或1100独立训练 |
| 组合控制训练 | native29/96新；4screen复用另计 | IID FedSA10增量严格离机闭合；最新32终轮，不将观察计入接受 |
| Fed-NGA/Huber | NGA新9/96采用，screen4另计 | 最新15终轮，已有9条恒定负类预测保留；Huber96尚未启动 |

作者实际批准完整比较范围统一CUDA，包括Full。不重训、不改参数/容差、不访问test。统一原CNN clean-root及valid确定性FP32 CUDA前向；原后处理实现与设备保留，包括GuardFed NumPy阈值和LoGo netcal CPU。协议：{ref(protocol)['path']}。首阶段严格冻结500=400控制+Full100；Full与方法表共享同一CUDA结果。完整目标2500逻辑记录仍待六方法/四控制及适配，不能冒充已冻结完整清单或已完成。

首固定7原Linux whole验收native三指标最大差均0；离机零拟合63指标/168计数一致，F归档110成员逐SHA，root {ref(cudaroot)['path']}。此前minus_F/non-IID/Sp-DFA/91004已正式通过，不是把诊断回填。旧CPU模型阈值差导致的30张预测变化及原1e-12失败完整保留，不证明所有模型环境等价。

第一次CUDA队列10项完成后，Full/IID/Benign/91005后置资源检查读取/proc/113052/cwd发生PermissionError，receipt原生差0但缺binding/ACCEPTED，旧项未采用。修复仅当场确认进程消失/Z/X才跳过，存活不可读仍拒绝；未证明历史zombie根因。独审+16边界检查/9元数据检查通过，原科学worker文本不变。新service guardfed_uniform_cuda500_resume1_valid，BASE /workspace/guardfed_checks/uniform_cuda_formal_20261011_resume1；原10CUDA源/模型/数组/binding/stat全核后复用，失败项仅一次有限恢复已通过，剩余原顺序490不自动重试。源/实际启动与观察：{ref(install)['path']}，{ref(cudaobs)['path']}。runner的root_offserver_accepted=0只是runner不管理本地采用；root真实已接受7，分开记录。

Hybrid原19+新增IID FedSA10=29已采用，root {ref(hybridroot)['path']}；304归档成员/10模型80张量核验，CPU108释放；三视图仍20，不能称全100完成。FLGMM完整native100与CPU三视图100已闭合，Windows零拟合的一项root group_kl诊断差-2.1684e-19保留，不称Windows whole/refit通过。主表 outputs/guardfed_tables/celeba_eleven_method_native_20261011/TABLES.md；FL表 outputs/guardfed_tables/celeba_flgmm_ten_scenes100_20261011/TABLES.md。

完整24条英文回复为 docs/server_deployment_20260923/revision_20260923/rebuttal_working_F80_FL100_Hybrid20_native11_20261011.md，固定380/F80、FL100、Hybrid20/native11截止，未用首7局部结果替代完整多seed证据。仍保留准确率代价、LoGo较低ASPD、FLTrust较高准确率；不称每个评分项不可或缺或全面领先。旧TableII全部480格有来源但294格n1的SD不可计算；FairGuard/IID/FedSA投稿59.13/历史54.13/round70候选53.94须区分。数值追溯不等于忠实17方法/统一十seed。

Git69提交0a276220b447645020e8dca8ee092ce411367bb7已核251blob；固定native400/views380/FLnative100/views90/NGA1/Hybrid19。其后NGA9、GPU诊断、FL100、新回复、作者CUDA批准、首7与Hybrid29尚未发布，不改旧回执。

{o['utc']} 双5090 100/100%、67/62°C，Recovery None；CPU实测11.03核/122.87999配额，OOM0，内存70.20/519.17GB，磁盘1.059TB余。GPU计算限制CPU有效占用，独立可用任务继续；不以无意义负载填占用。大文件留服务器，最小离机证据仅写前核Yanan2TB的F:/YananResearchStorage/GuardFed，禁止内置盘批量回退。三小时聊天automation仍PAUSED，无可调用更新工具，未新建替代监督；现存supervisor队列正常执行不等于聊天持续巡检。

尚待六基线、四控制、完整统一CUDA表、主端点/最终评价、FedWA/SmartFL/FedDNA忠实资料、匹配投稿LaTeX整合及最终发布。当前valid，官方test历史暴露，不能称未触碰test。Huber恒等投影为CNN适配；LoGo20虚拟cohort仅人口适配；DP指demographic parity。整个返修仍未完成。

本次原STATE/入口/reader备份在tmp/root_CUDA7_resume1_Hybrid29_20261011；历史尾逐SHA保持。

'''.encode('utf8')
reader.write_text(r,encoding='utf8',newline='\n');s['author_review_reading_entry_20261011'].update(utc=now,reader=ref(reader),uniform_CUDA_first7_adopted=True)
s['current_entry_writer']=dict(**ref(__file__),utc=now);s['local_work_verified_utc']=now;s['updated_unix']=datetime.datetime.now(datetime.timezone.utc).timestamp()
assert sp.read_bytes()==before;sp.write_text(json.dumps(s,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8',newline='\n')
for p,raw,tail,pin in entries:
    assert p.read_bytes()==raw;p.write_bytes(newprefix+tail);assert hashlib.sha256(p.read_bytes()[len(newprefix):]).hexdigest()==pin
proof=dict(status='ROOT_CUDA7_RESUME1_HYBRID29_CURRENT_ENTRIES_RECORDED',utc=now,source=ref(__file__),state_before_sha256=hashlib.sha256(before).hexdigest(),state_after=ref(sp),reader=ref(reader),five_entries=[ref(x[0]) for x in entries],history_tails_exact=True,evidence=[ref(p) for p in [cudaroot,hybridroot,obs,cudaobs,install,protocol]],new_scientific_adoptions_in_this_metadata_writer=0,whole_rebuttal_complete=False)
with (H/'ROOT_METADATA_ADOPTION.json').open('x',encoding='utf8',newline='\n') as f:json.dump(proof,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps(dict(status=proof['status'],state_sha256=sha(sp),root=ref(H/'ROOT_METADATA_ADOPTION.json'))))
