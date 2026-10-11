"""Record measured runtime and already adopted results; preserve historical tails."""
from pathlib import Path
import ast, datetime, hashlib, json, re

R=Path.cwd();T=R/'docs/server_deployment_20260923/training_20260923'
V=R/'docs/server_deployment_20260923/revision_20260923'
H=R/'tmp/root_CUDA10_two_queues_Git70_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
ref=lambda p:dict(path=Path(p).relative_to(R).as_posix(),sha256=sha(p))
sp=T/'TRAINING_STATE.json';before=sp.read_bytes()
assert sha(sp)=='8f803c7b0c4789a369dfce1997a9262909e2b5072e4f5e019cc3be652c80a2a4'
assert not H.exists()
s=json.loads(before)
paths={
 'health':T/'server_reactivation_20261009/ROOT_COMPACT_20261011T075500Z.json',
 'cuda500':R/'tmp/uniform_cuda500_root_dispatch_20261011/RESUME2_OBSERVATION_20261011T075524Z.json',
 'cuda800':R/'tmp/baseline_old800_runtime_root_20261011_attempt003/OBSERVATION_20261011T075457Z.json',
 'root7':R/'tmp/uniform_cuda7_saved_audit_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json',
 'root3':R/'tmp/old800_cuda3_saved_audit_prepare_20261011/mixed_producer_v2/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json',
 'git70':T/'publication_closed_increment70_verified_20261011.json',
 'resume2_start':R/'tmp/uniform_cuda500_root_dispatch_20261011/RESUME2_INSTALL_COMMAND.json',
 'remaining797_start':R/'tmp/baseline_old800_runtime_root_20261011_attempt003/REMAINING797_START_COMMAND.json',
 'resume1_failure_diagnosis':R/'tmp/uniform_cuda500_root_dispatch_20261011/RESUME1_INTERRUPTION_DIAGNOSIS_20261011T073527Z.json',
 'resume2_review':R/'tmp/uniform_cuda500_resume2_independent_review_20261011/SOURCE_REVIEW.json',
}
pins={
 'health':'2d130fd4e1e319147135cf7a1145dcd8e575202bdcce20fd8d78fd68dd52f38a',
 'cuda500':'9ac19c18782c77a7c6a07dbc5b964519f58812030665f46e0893b538e25dca65',
 'cuda800':'f95e546eb48ff606f41a4ef7bcac23cb9d9d23af71e6f0389182262409378e12',
 'root7':'3ed2e1fe79edbc5c196d428305bf27201856f2e6d5589c6e4752255accec4dda',
 'root3':'f700def3a48621f132ee3ac6ed7257f52ba65fc9598a4202e18174dbba6f39c5',
 'resume1_failure_diagnosis':'d3eb976c2dcf74b7aa69d7025b490a7c9ee80f9d37737b0b6038689432555b46',
 'resume2_review':'88b22ac6eb63ac1171d4c2b04f929276c41db8f47b7c19cb867a8e652502f6d3',
}
for k,h in pins.items():assert sha(paths[k])==h,k
o=read(paths['health']);c=read(paths['cuda500']);b=read(paths['cuda800']);g=read(paths['git70'])
assert read(paths['root7'])['accepted']==7
assert read(paths['root3'])['status'] and g['commit']=='4fccc8b8d1d56492ac9cffd0cf27ca390e78bd20'
assert g['committed_blobs_sha256_verified']==197
assert read(paths['resume2_start'])['exit_code']==0 and read(paths['remaining797_start'])['exit_code']==0
cp=c['progress.json'];bp=b['remaining_progress'];q=o['main_queue']
assert cp['accepted_prior_cuda']==57 and len(c['accepted_receipts'])==1
assert len(cp['remote_saved_accepted_ids'])==58 and cp['pending']==442
assert c['accepted_receipts'][0]['data']['acceptance']['native_comparison']['max_abs_difference']==0
assert c['accepted_receipts'][0]['id']=='GuardFed-AD2+_lr0.0005_drop0.005_Benign_seed91002'
assert not c['failure_files'] and not c.get('QUEUE_FAILURE.json') and c['service']['rc']==0
assert len(b['accepted'])+len(b['reused'])==23 and len(bp['new_accepted_ids'])==20
assert not b['failures'] and b['services']['guardfed_uniform_cuda800_remaining_v3_valid']['exit_code']==0
counts={k:len(v) if isinstance(v,(list,dict)) else v for k,v in q.items()}
assert (counts['completed'],counts['active'],counts['pending'],counts['failed'])==(427,8,365,0)

protocol=T/'celeba_uniform_cuda_evaluation_v1/PROTOCOL.md';oldprotocol=protocol.read_bytes()
assert sha(protocol)=='6f0502f95c5179bff11771ee3859d44314f879b7e4dd5ed0d83634bc45a02827'
append='''
## 2026-10-11 实际执行增量（07:55 UTC 截止）

以下为实际执行后的追加记录，不追改上述首阶段冻结时间。首阶段后续于 Full/non-IID/Benign/91002 的后置资源检查再次发生 `/proc/117589/cwd` PermissionError。其 receipt 和 binding 已生成，但无最终 ACCEPTED，未作为已完成项复用。原首次10项和 resume1 新47项合计 **57项**；47是新增数，不能作为累计数。故障当时的进程类别与状态未能回溯证明，原因不称 zombie 或非Python。

经独立审查，资源检查仅对明确非Python进程免读 cwd，所有PID/task的CPU affinity和名义预算仍核查；存活但不可读的Python及未知进程仍拒绝。科学worker原文、参数、容差、GPU0/CPU107及单线程未变。新 namespace `uniform_cuda_formal_20261011_resume2` 于07:50 UTC实际启动，逐项重核原57项来源、模型、数组、binding、SHA和stat后复用，只执行原顺序剩余443项。07:55实测累计服务器保存验收58/500，失败模型已在新目录通过、原生三指标最大差0；本地严格离机采用仍为首7项。旧失败证据与部分输出全部保留，无自动重试。

另批原八方法800条库存已冻结，cohort SHA `d39e29333c40e8a5b63f1344487225abac4ba83216dc32e75f64badc87bc0b71`：FedAvg、FairFed、Median、FLTrust、FairGuard、FLTrust+FairGuard、FedAA、LASA各100条。Full不在此800内，不与首500重复计数。原788条cu128、12条cu130训练身份保留；本次评价统一设备不等于训练环境等价。原生实现和第70轮checkpoint不变。

该800批的启动模块命名冲突与一次资源读取失败均保留在旧目录。第三目录 `uniform_cuda_formal_original800_20261011_attempt003` 复用第一次真正通过的FedAvg单例，仅新做固定FedAA/LASA两例。三例均通过原Linux完整保存验收（native三指标差0）、离机零拟合27指标/72计数/9预测规则核对及F盘85成员SHA备份，root记录 `tmp/old800_cuda3_saved_audit_prepare_20261011/mixed_producer_v2/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json`。其余797项于07:46 UTC实际启动；07:54实测服务器累计23/800、无失败。本地严格采用3/800，不能把服务器进度直接等同论文验收。

以上两批共冻结1300个互不重复的逻辑记录；本地严格CUDA采用共10（7+3），完整目标2500尚未齐备。FL100运行调用器和LoGo100冻结后处理缓存适配仍在源审查/输入准备阶段，未据方案称实际运行。LoGo需要完整、根采用的FedAvg CUDA100缓存后才可评价；原已冻结CPU netcal后处理状态保持不变，不通过新缓存重新择优或拟合。其余正在训练的方法和四控制需各自完成后增量接入，不启动未经冻结的finaltest。
'''
newprotocol=oldprotocol+append.encode('utf8')
reader=V/'AUTHOR_REVIEW_START_HERE.md';oldreader=reader.read_bytes();r=oldreader.decode('utf8')
a='11方法表、三页PDF与上一版FL90回复稿已推送Git（0a276220）；当前FL100回复稿及此后增量尚未发布。'
z='十一方法表、三页PDF、当前FL100回复稿、作者CUDA批准与首7项/Hybrid29证据已在Git70发布（4fccc8b8，197个文件blob核SHA）；本页后续07:55运行增量尚未发布。'
assert r.count(a)==1;r=r.replace(a,z)
a='新CUDA首批7项已完成原Linux严格验收、离机零拟合核对、F盘110成员SHA备份及root采用；服务器首阶段累计11/500验收，进程检查中断已在新目录有限恢复。完整CUDA表尚未完成。'
z='新CUDA已有10项完成原Linux严格验收、离机零拟合核对、F盘成员SHA备份及root采用（首阶段7/500、原八方法3/800）。07:55 UTC实测服务器分别58/500和23/800通过保存验收；旧中断证据保留，恢复只做未完成项。服务器进度不等于全部离机采用，完整CUDA表尚未完成。'
assert r.count(a)==1;r=r.replace(a,z)
for link in re.findall(r'\[[^\]]+\]\((E:/[^)]+)\)',r):assert Path(link.split('#',1)[0]).is_file(),link

tree=ast.parse((R/'tmp/refresh_native370_FL86_Hybrid20_root_20261011.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='entries' for x in n.targets))
ns={'R':R,'T':T};exec(compile(ast.Module(body=[node],type_ignores=[]),'<tailguards>','exec'),ns)
entries=[];prefix=None
for p,marker,pin in ns['entries']:
 raw=p.read_bytes();i=raw.index(marker.encode());tail=raw[i:]
 assert hashlib.sha256(tail).hexdigest()==pin
 if prefix is None:prefix=raw[:i]
 assert raw[:i]==prefix
 entries.append((p,raw,tail,pin))
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
newprefix=f'''# CURRENT: GuardFed返修实验

入口更新：{now}。服务器 ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded；先遵守/etc/vast-agents-guide.md。实际结果采用与运行观测分开。

| 阶段 | 严格验收、F盘SHA与root采用 | 最新实测及边界 |
|---|---:|---|
| 机制训练 | 新400/800，U/C/A/F各100；Full100复用 | {o['utc']}：427终轮、8活动、365等待、训练失败0；其余4类继续 |
| 统一CUDA首阶段500 | 本地采用7/500；服务器保存验收58/500 | {c['utc']}：原57逐SHA/stat复用，剩余443有限恢复，已新验收1；当前失败0 |
| 统一CUDA原八方法800 | 本地采用3/800；服务器保存验收23/800 | {b['utc']}：固定3闭合后其余797已启动，新通过20、777等待，当前失败0；不含Full/FL/LoGo |
| 历史CPU三视图 | 机制380（U/C/A100、F80）、FL100、Hybrid20 | 原620数值失败与旧表保留；新CUDA表独立产生 |
| 主比较表 | 11方法native1100，三页PDF | IID/non-IID各5场景×10seed，另9/6seed；历史评价协议，非最终17方法或1100独立训练 |
| 组合控制训练 | native29/96新；4screen复用另计 | 最新34终轮，不把观察计入接受；三视图20 |
| Fed-NGA/Huber | NGA新9/96采用，screen4另计 | 最新20终轮；9条已采用恒定负类预测保留；Huber96尚未启动 |

作者批准完整比较范围统一CUDA，包括Full。不重训、不改参数/容差、不访问test。统一原CNN clean-root及valid确定性FP32 CUDA前向；原冻结后处理实现与设备保留。协议及追加运行记录：docs/server_deployment_20260923/training_20260923/celeba_uniform_cuda_evaluation_v1/PROTOCOL.md。目前两批实际冻结1300条互不重复记录，本地CUDA采用10=7+3。Full100仅在首500中，与方法表共享一次。完整目标2500仍待六方法/四控制及适配，不能冒充齐备或已完成。

首7原Linux whole验收native三指标最大差0，离机零拟合63指标/168计数一致，F归档110成员逐SHA；root tmp/uniform_cuda7_saved_audit_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json。原八方法固定3（FedAvg、FedAA、LASA）亦均native差0，离机27指标/72计数/9预测规则一致，F归档85成员逐SHA；root tmp/old800_cuda3_saved_audit_prepare_20261011/mixed_producer_v2/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json。其余尚只计服务器保存验收。旧CPU阈值差造成30张预测变化及原1e-12失败保留，不称所有环境等价。

500队列resume1在累计57项后遇/proc/117589/cwd资源读取错误；47为该次新增而非累计。失败Full/non-IID/Benign/91002有receipt/binding但无ACCEPTED，旧部分未采用。独审后的resume2只对明确非Python进程免读cwd，所有PID/task affinity及名义预算仍检查，未知/不可读Python仍拒绝；科学worker原文不变。旧57全核后复用，失败项新目录有限重评已通过、native差0。service guardfed_uniform_cuda500_resume2_valid，BASE /workspace/guardfed_checks/uniform_cuda_formal_20261011_resume2；原失败证据不覆盖，无自动重试，历史进程类别/状态未知。

800批原启动模块命名冲突、第二次资源检查失败均保留。第三目录复用已有FedAvg单例、新做FedAA/LASA两例，固定3本地采用后才实际启动797。service guardfed_uniform_cuda800_remaining_v3_valid，BASE /workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt003。FL100调用器及LoGo100缓存适配仍是源准备/审查，不称实际运行；LoGo须等FedAvg CUDA100完整采用缓存，原冻结netcal后处理不重新择优/拟合。

Hybrid29、NGA9、机制native400、历史CPU三视图380/FL100/Hybrid20严格采用量不因运行观测增加。主表 outputs/guardfed_tables/celeba_eleven_method_native_20261011/TABLES.md；FL表 outputs/guardfed_tables/celeba_flgmm_ten_scenes100_20261011/TABLES.md。完整24条英文回复为 docs/server_deployment_20260923/revision_20260923/rebuttal_working_F80_FL100_Hybrid20_native11_20261011.md，固定380/F80、FL100、Hybrid20/native11截止。保留准确率代价、LoGo较低ASPD、FLTrust较高准确率，不称每个评分项不可或缺或全面领先。

Git70提交4fccc8b8d1d56492ac9cffd0cf27ca390e78bd20已核197文件blob及remote，父0a276220；含上述固定截止稿、NGA9、作者CUDA批准、CUDA首7与Hybrid29等代码/精简证据。发布回执 docs/server_deployment_20260923/training_20260923/publication_closed_increment70_verified_20261011.json。07:55实时观察、800固定3及其后恢复源尚未随此旧提交发布，不改旧回执。无模型/NPZ/数据/大归档入Git。

{o['utc']} 双5090 99/100%、65/62°C，Recovery None；CPU实测12.02核/122.87999配额，OOM0，内存70.15/519.17GB，磁盘1.059TB余。GPU计算限制CPU有效占用，独立可用任务继续。大文件留服务器，最小离机证据只写核验Yanan2TB后的F:/YananResearchStorage/GuardFed，不回退内置盘。三小时聊天automation仍PAUSED，无可调用更新工具，未新建替代监督；supervisor实际运行不等于聊天持续巡检。

仍待六基线、四控制、完整统一CUDA表、主端点/最终评价、FedWA/SmartFL/FedDNA忠实资料、匹配投稿LaTeX整合及最终发布。旧TableII480格有来源，294格n1的SD不可计算；FairGuard/IID/FedSA投稿59.13/历史54.13/round70候选53.94区分保留。当前valid，官方test历史暴露，不称未触碰test；Huber恒等投影仅CNN适配，LoGo20虚拟cohort仅人口适配。整个返修尚未完成。

本次原STATE/入口/reader/protocol备份在tmp/root_CUDA10_two_queues_Git70_20261011；五处历史尾逐SHA保持。

'''.encode('utf8')

H.mkdir();(H/'PRIOR_STATE.json').write_bytes(before);(H/'PRIOR_PREFIX.md').write_bytes(prefix)
(H/'PRIOR_READER.md').write_bytes(oldreader);(H/'PRIOR_PROTOCOL.md').write_bytes(oldprotocol)
assert protocol.read_bytes()==oldprotocol;protocol.write_bytes(newprotocol)
reader.write_text(r,encoding='utf8',newline='\n')
u=s['uniform_CUDA_evaluation_v1'];u.update(status='ROOT10_ADOPTED_TWO_FROZEN_QUEUES_RUNNING',uniform_CUDA_scientific_accepted=10,actual_phase_frozen=500,actual_all_frozen=1300,protocol=ref(protocol),current_remote_saved_accepted=58,current_remote_new_after_resume=1,current_remote_active=cp['active'],current_remote_pending=cp['pending'],current_remote_failed=0,current_observation=ref(paths['cuda500']),current_service='guardfed_uniform_cuda500_resume2_valid',current_base='/workspace/guardfed_checks/uniform_cuda_formal_20261011_resume2',current_mechanism_base='/workspace/guardfed_checks/mechanism_cuda_formal_20261011_resume2',resume_source_review=ref(paths['resume2_review']),actual_resume_command=ref(paths['resume2_start']),prior_CUDA_reused=57,original10_plus_resume1_new47=True,root_adopted_first500=7,root_adopted_old800=3,old800_root=ref(paths['root3']),old800_observation=ref(paths['cuda800']),old800_remote_saved_accepted=23,old800_remaining_queue_new_accepted=20,old800_current_pending=777,old800_service='guardfed_uniform_cuda800_remaining_v3_valid',old800_actual_remaining_command=ref(paths['remaining797_start']),FL100_runtime_started=False,LoGo100_runtime_started=False,whole500_complete=False,whole_comparison_complete=False)
s['uniform_CUDA_resume1_interruption_20261011']=dict(status='PRESERVED_TOTAL57_EXPLICIT_RESUME2_MISSING_FULL_PASSED',diagnosis=ref(paths['resume1_failure_diagnosis']),total_accepted_before_interruption=57,new_accepted_before_interruption=47,prior_original_accepted=10,failed_id=c['accepted_receipts'][0]['id'],original_partial_adopted=False,historical_process_type_and_state_proven=False,scientific_worker_unchanged=True,new_namespace=True,automatic_retry=False,recovered_native_difference=0)
s['latest_publication_verification']=dict(**ref(paths['git70']),commit=g['commit'],branch=g['branch'],verified_utc=g['verified_utc'],committed_blobs_sha256_verified=197,subsequent_runtime_changes_published=False)
m=s['celeba_mechanism_v1'];m.update(queue_completed_observed=427,queue_active_observed=8,queue_pending_observed=365,queue_failures_observed=0,queue_observation_utc=o['utc'],queue_observation_path=str(paths['health']),queue_observation_sha256=sha(paths['health']))
s['active_services']=['guardfed_celeba_mechanism_formal','guardfed_celeba_hybrid_fullcoverage','guardfed_celeba_fednga_fullcoverage_v1','guardfed_uniform_cuda500_resume2_valid','guardfed_uniform_cuda800_remaining_v3_valid']
s['latest_current_resource_observation']=dict(utc=o['utc'],source_ref=ref(paths['health']),main_terminal=427,main_active=8,main_pending=365,main_failed=0,NGA_terminal_observed=20,NGA_new_accepted=9,Hybrid_terminal_observed=34,Hybrid_native_new_accepted=29,GPU_utilization=[99,100],GPU_C=[65,62],GPU_RecoveryAction=['None','None'],memory_current_B=int(o['cgroup']['memory.current']),memory_max_B=int(o['cgroup']['memory.max']),disk_free_B=o['disk_free_B'],cgroup_CPU_quota_cores=122.87999,cgroup_CPU_measured_cores=o['measured_cgroup_CPU_cores'],cgroup_OOM=0,not_new_scientific_acceptance=True)
s['latest_NGA96_readonly_observation']=dict(**ref(paths['health']),utc=o['utc'],terminal_observed=20,accepted_new=9,active=o['NGA']['active_progress'],Huber_started=False)
s['NGA96_runtime_dispatch_20261011'].update(latest_readonly_observation=ref(paths['health']),terminal_result_directories_observed=20,latest_observed_rounds=o['NGA']['active_progress'])
s['hybrid100_fullcoverage_20261010'].update(latest_measured_observation=dict(utc=o['utc'],terminal_observed=34,source=ref(paths['health']),acceptance_unchanged_by_observation=True))
s['author_review_reading_entry_20261011'].update(utc=now,reader=ref(reader),uniform_CUDA_root_adopted=10)
s['current_entry_writer']=dict(**ref(__file__),utc=now);s['local_work_verified_utc']=now;s['updated_unix']=datetime.datetime.now(datetime.timezone.utc).timestamp()
assert sp.read_bytes()==before;sp.write_text(json.dumps(s,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8',newline='\n')
for p,raw,tail,pin in entries:
 assert p.read_bytes()==raw;p.write_bytes(newprefix+tail)
 assert hashlib.sha256(p.read_bytes()[len(newprefix):]).hexdigest()==pin
proof=dict(status='ROOT_CUDA10_TWO_QUEUES_GIT70_CURRENT_ENTRIES_RECORDED',utc=now,source=ref(__file__),state_before_sha256=hashlib.sha256(before).hexdigest(),state_after=ref(sp),reader=ref(reader),protocol=ref(protocol),five_entries=[ref(x[0]) for x in entries],history_tails_exact=True,evidence={k:ref(p) for k,p in paths.items()},root_adopted_CUDA=10,remote500_saved_accepted=58,remote800_saved_accepted=23,new_scientific_adoptions_in_metadata_writer=0,whole_rebuttal_complete=False)
with (H/'ROOT_METADATA_ADOPTION.json').open('x',encoding='utf8',newline='\n') as f:json.dump(proof,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps(dict(status=proof['status'],state=ref(sp),proof=ref(H/'ROOT_METADATA_ADOPTION.json'))))
