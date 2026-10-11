"""Refresh current entries from adopted evidence, with historical tails left byte-exact."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,re
R=Path.cwd();T=R/'docs/server_deployment_20260923/training_20260923';V=T.parent/'revision_20260923'
D=R/'tmp/root_CUDA211_FL100_LoGo2_Git72_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
ref=lambda p:dict(path=Path(p).relative_to(R).as_posix(),sha256=sha(p))
sp=T/'TRAINING_STATE.json';before=sp.read_bytes()
assert sha(sp)=='087f0ad964f9a284cec5b5867b3743e34c598a96c11193896dc6089c448d188e' and not D.exists()
paths={
 'FL_root':R/'tmp/FL100_cuda100_saved_closure_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json',
 'LoGo_root':R/'tmp/LoGo_cuda_fixed2_saved_audit_prepare_v2_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json',
 'LoGo_start98':R/'tmp/LoGo100_remaining98_root_start_prepare_20261011/ROOT_REMAINING98_START_COMMAND.json',
 'LoGo_deployment':R/'tmp/LoGo100_cuda_actual_root_20261011/ROOT_DEPLOYMENT_RECORD.json',
 'LoGo_env':R/'tmp/LoGo100_cuda_actual_runtime_prerequisites_20261011/ACTUAL_REPAIRED_HANDOFF.json',
 'LoGo_restore':R/'tmp/LoGo100_cuda_actual_runtime_prerequisites_20261011/ACTUAL_TRANSPORT_VERIFICATION.json',
 'health':T/'server_reactivation_20261009/ROOT_COMPACT_20261011T091928Z.json',
 'cuda500':R/'tmp/uniform_cuda500_root_dispatch_20261011/RESUME2_OBSERVATION_20261011T091914Z.json',
 'cuda800':R/'tmp/baseline_old800_runtime_root_20261011_attempt003/OBSERVATION_20261011T091920Z.json',
 'Git72':T/'publication_closed_increment72_verified_20261011.json'}
assert sha(paths['FL_root'])=='ea935ead7da50ba7eff0dd7e4e4b289ca9365427096ad8431808ae4d76acb751'
assert sha(paths['LoGo_root'])=='cdf9a578fd26a9a85ae674aff7e55932e9cd6e75b895b8e3b39adc1ad12753bc'
assert sha(paths['LoGo_start98'])=='4c200beebdc55c69ef62155060829dfe7c5f4d0235f3a416e43ff062c5e51130'
fl=read(paths['FL_root']);lg=read(paths['LoGo_root']);o=read(paths['health']);c=read(paths['cuda500']);b=read(paths['cuda800']);git=read(paths['Git72'])
assert fl['accepted']==100 and fl['new_unique']==97 and fl['reused_already_adopted']==3
assert lg['accepted']==2 and lg['root_accepted'] is True and read(paths['LoGo_start98'])['exit_code']==0
assert git['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and git['commit']=='51a27b50c6c1880d3fb3c6b5c45b750d0cc68d12' and git['committed_blobs_sha256_verified']==258
cp=c['progress.json'];n500=len(cp['remote_saved_accepted_ids']);n800=len(b['accepted'])+len(b['reused'])
assert n500==95 and n800==220 and not cp['failed'] and not b['failures']
nmain=len(o['main_queue']['completed']);nnga=len(o['NGA']['terminal_ids_observed']);nhybrid=len(o['Hybrid']['terminal_ids_observed'])
assert (nmain,nnga,nhybrid)==(440,25,37) and not o['main_queue']['failed']
now=datetime.now(timezone.utc).isoformat();s=json.loads(before);u=s['uniform_CUDA_evaluation_v1']
assert u['root_adopted_first500']==7 and u['root_adopted_old800_unique']==102 and 7+102+100+2==211
u.update(status='ROOT211_ADOPTED_FL100_COMPLETE_LOGO2_ADOPTED_REMAINING98_STARTED',uniform_CUDA_scientific_accepted=211,
 actual_all_frozen=1500,actual_frozen_cohorts={'first500':500,'old800_excluding_Full_FL_LoGo':800,'FL100':100,'LoGo100':100},
 root_adopted_old800=102,old800_fixed_canary_adopted=3,root_adopted_FL100=100,FL100_root=ref(paths['FL_root']),
 FL100_server_saved_accepted=100,FL100_whole_complete=True,FL100_remaining97_complete=True,
 FL100_actual_completion=ref(R/'tmp/FL100_cuda100_saved_closure_prepare_20261011/actual_metadata/OBSERVATION.json'),
 current_remote_saved_accepted=n500,current_remote_new_after_resume=cp['accepted_new_cuda'],current_remote_pending=cp['pending'],current_remote_failed=0,current_observation=ref(paths['cuda500']),
 old800_remote_saved_accepted=n800,old800_remaining_queue_new_accepted=len(b['remaining_progress']['new_accepted_ids']),old800_current_pending=b['remaining_progress']['remaining'],old800_observation=ref(paths['cuda800']),
 LoGo100_runtime_started=True,LoGo100_actual_frozen=100,root_adopted_LoGo100=2,LoGo100_root=ref(paths['LoGo_root']),
 LoGo_remaining98_started=True,LoGo_actual_remaining98_command=ref(paths['LoGo_start98']),LoGo_whole100_complete=False,
 LoGo_service='guardfed_uniform_cuda_LoGo100_remaining_valid',LoGo_base='/workspace/guardfed_checks/uniform_cuda_LoGo100_20261011_attempt001',
 LoGo_Linux_index_pending=False,LoGo_original_restore_and_environment_pending=False,
 LoGo_environment=ref(paths['LoGo_env']),LoGo_original_restore=ref(paths['LoGo_restore']),LoGo_deployment=ref(paths['LoGo_deployment']),
 LoGo_Python_31020_vs_original_3104_disclosed=True,LoGo_new_fit=False,LoGo_new_CNN=False)
s['celeba_mechanism_v1'].update(queue_observation_utc=o['utc'],queue_observation_path=str(paths['health']),queue_observation_sha256=sha(paths['health']))
s['NGA96_runtime_dispatch_20261011'].update(latest_readonly_observation=ref(paths['health']),terminal_result_directories_observed=nnga,latest_observed_rounds=o['NGA']['active_progress'])
s['latest_NGA96_readonly_observation']=dict(**ref(paths['health']),utc=o['utc'],terminal_observed=nnga,accepted_new=19,active=o['NGA']['active_progress'],Huber_started=False)
s['hybrid100_fullcoverage_20261010'].update(latest_measured_observation=dict(utc=o['utc'],terminal_observed=nhybrid,source=ref(paths['health']),acceptance_unchanged_by_observation=True))
s['latest_current_resource_observation']=dict(utc=o['utc'],source_ref=ref(paths['health']),main_terminal=nmain,main_active=8,main_pending=o['main_queue']['pending'],main_failed=0,NGA_terminal_observed=nnga,NGA_new_accepted=19,Hybrid_terminal_observed=nhybrid,Hybrid_native_new_accepted=29,GPU_utilization=[98,100],GPU_C=[66,64],GPU_RecoveryAction=['None','None'],memory_current_B=int(o['cgroup']['memory.current']),memory_max_B=int(o['cgroup']['memory.max']),disk_free_B=o['disk_free_B'],cgroup_CPU_quota_cores=122.87999,cgroup_CPU_measured_cores=o['measured_cgroup_CPU_cores'],cgroup_OOM=0,not_new_scientific_acceptance=True)
s['latest_publication_verification']=dict(**ref(paths['Git72']),commit=git['commit'],files=258,verified_utc=git['verified_utc'],later_CUDA211_LoGo_runtime_pending_Git73=True)
tree=ast.parse((R/'tmp/refresh_native370_FL86_Hybrid20_root_20261011.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='entries' for x in n.targets));ns={'R':R,'T':T};exec(compile(ast.Module(body=[node],type_ignores=[]),'<tails>','exec'),ns)
entries=[];prefix=None
for p,marker,h in ns['entries']:
 raw=p.read_bytes();at=raw.index(marker.encode());assert hashlib.sha256(raw[at:]).hexdigest()==h
 if prefix is None:prefix=raw[:at]
 assert prefix==raw[:at];entries.append((p,raw,raw[at:],h))
text=f'''# CURRENT: GuardFed返修实验

入口更新：{now}。当前服务器 ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded；先遵守/etc/vast-agents-guide.md。以下将实际采用与运行观测分开。

| 阶段 | 严格验收、F盘SHA与root采用 | 最新实测及边界 |
|---|---:|---|
| 机制训练 | 新400/800，U/C/A/F各100；Full100共享复用 | {o['utc']}：440终轮、8活动、352等待、失败0；其余4类继续 |
| 统一CUDA首阶段500 | 本地采用7/500；服务器保存验收95/500 | {c['utc']}：旧57复用、新38通过、405等待；失败0 |
| 统一CUDA原八方法800 | 本地去重采用102/800；服务器保存验收220/800 | {b['utc']}：原固定3加剩余217通过，580等待；失败0；不含Full/FL/LoGo |
| 统一CUDA FLGMM100 | 全部100已采用；新97+原3，非103 | 原Linux完整验收；离机900指标/2400计数/300规则，native最大差0；959新归档成员SHA与28旧F成员核验 |
| 统一CUDA LoGo100 | 固定2已采用，剩98实际启动 | 09:23 UTC原队列启动；2条39734预测/6指标/16计数重算误差0；CPU后处理保持冻结，不重新fit |
| 历史CPU三视图 | 机制380、FL100、Hybrid20 | 旧620数值失败与旧表完整保留，新CUDA表独立产生 |
| 主比较表 | 11方法native1100，三页PDF | IID/non-IID各5场景×10seed，另9/6seed；非最终17方法或1100独立训练 |
| 组合控制训练 | native29/96新；screen4另计 | 最新37终轮，仅观察未增采用；三视图20 |
| Fed-NGA/Huber | NGA新19/96采用；screen4另计 | 最新25终轮；已采用19条恒定负类预测保留；Huber96尚未启动 |

作者批准完整比较范围统一CUDA，包括Full。不重训、不改参数/容差、不访问test。统一原CNN clean-root及valid确定性FP32 CUDA前向；原冻结后处理实现与设备保留。协议：celeba_uniform_cuda_evaluation_v1/PROTOCOL.md。目前实际冻结1500条互不重复记录（500+800+FL100+LoGo100），本地CUDA去重采用211=7+102+100+2。Full100仅在首500中，与方法表共享一次。完整目标2500仍待六方法/四控制及适配，不能称全部冻结或完成。

FedAvg100根采用：tmp/FedAvg_cuda100_saved_cache_closure_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json。900指标/2400计数/300规则和1058归档成员SHA通过；与旧800固定3重叠一个FedAvg，唯一采用102。FL100根采用：tmp/FL100_cuda100_saved_closure_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json；只运输97新NPZ，旧3原F引用，不重复下载。LoGo固定2根采用：tmp/LoGo_cuda_fixed2_saved_audit_prepare_v2_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json。原Linux完整预测与离机零拟合算术检查均通过；不是离机重做整个netcal预测。Benign恒负类结果保留，不能把零差异单独解释为公平成功。

LoGo712份原产物已逐SHA恢复；FedAvg100 Linux索引actual002已实际通过。独立CPU环境首次安装错误引入torch2.14.1，已保留失败并以原包约束修复；实际Python3.10.20与原3.10.4差别披露，netcal1.3.6/torch2.8CPU保持原直接包版本，两例实际检查通过但不声称全部环境等价。LoGo部署原科学代码未变；fixed2采用后才启动原序98，service guardfed_uniform_cuda_LoGo100_remaining_valid，BASE /workspace/guardfed_checks/uniform_cuda_LoGo100_20261011_attempt001。既有712与FedAvg缓存不重复运输，不重新选择/拟合原后处理。

500旧两次/proc资源读取失败保留；resume2仅对明确非Python进程免读cwd，所有PID/task affinity及名义预算仍核，未知不可读Python仍拒绝。旧57逐SHA/stat复用，失效项新目录通过native差0。service guardfed_uniform_cuda500_resume2_valid，BASE /workspace/guardfed_checks/uniform_cuda_formal_20261011_resume2。800旧模块冲突与资源检查失败保留；attempt003原科学worker不变，service guardfed_uniform_cuda800_remaining_v3_valid。无自动重试或失败证据覆盖。

Git72已提交51a27b50c6c1880d3fb3c6b5c45b750d0cc68d12，258个blob、remote分支和clean核验通过，父ba302df5；含FedAvg100、FL3、NGA19与08:38截止入口等。72发布前清单漏SUFFIX_FIX的拒绝发生于Git写入之前，补精确封条成员后发布成功，失败和修复证据保留。后续FL100完整采用、LoGo实际闭环及本次入口待Git73；不追改旧回执。模型、NPZ、数据和大归档不进Git。

{o['utc']} 双5090 98/100%、66/64°C，Recovery None；CPU实测12.02核/122.87999配额，OOM0，内存75.02/519.17GB，服务器磁盘1.0445TB余。活动机制轮次持续增长。CPU低有效占用已定位GPU计算限制，继续独立有效任务，不改协议或重复作业提高占用。大文件优先留服务器；最小离机证据只写核验Yanan2TB后的F:/YananResearchStorage/GuardFed，不回退内置盘。三小时聊天automation仍PAUSED，无可调用更新工具，未创建替代监督；supervisor持续运行不等于聊天持续巡检。

完整24条英文回复：docs/server_deployment_20260923/revision_20260923/rebuttal_working_F80_FL100_Hybrid20_native11_20261011.md，固定380/F80、FL100、Hybrid20/native11截止，原SHA910b263保留，不把后续部分结果偷偷写入该截止稿。IID/non-IID十一方法表：outputs/guardfed_tables/celeba_eleven_method_native_20261011/TABLES.md。保留准确率代价、LoGo较低ASPD和FLTrust较高准确率，不称每评分项不可或缺或全面领先。

仍待六基线、四控制、完整统一CUDA表、主端点/最终评价、FedWA/SmartFL/FedDNA忠实来源、匹配投稿LaTeX整合和最终发布。旧TableII480格有来源，294格n1的SD不可计算；FairGuard/IID/FedSA投稿59.13/历史54.13/round70候选53.94区分保留。当前valid，官方test历史暴露，不称未触碰test；Huber恒等投影仅CNN适配，LoGo20虚拟cohort仅人口适配。整个返修尚未完成。

本次STATE/入口/reader/protocol旧快照在tmp/root_CUDA211_FL100_LoGo2_Git72_20261011；五处历史尾逐SHA保持。

'''
reader=V/'AUTHOR_REVIEW_START_HERE.md';oldreader=reader.read_bytes();r=oldreader.decode('utf8')
r=r.replace('FL3/97、NGA19和FedAvg100等后续增量尚未发布。','FL3、NGA19和FedAvg100已在Git72发布（51a27b50，258个blob/remote核SHA）；新FL100完整CUDA采用、LoGo2与剩98启动及最新入口待下一增量。')
start=r.index('| 图像机制消融 |');end=r.index('\n| 新增基线与简单组合 |',start)
r=r[:start]+'''| 图像机制消融 | 训练U/C/A/F各100新结果已采用，共400/800；历史CPU三视图380（F80）保留。作者批准完整比较范围统一CUDA重评（含Full），不重训、不改参数/容差、不访问test；原CNN确定性FP32 CUDA，冻结后处理的数学与设备保持。统一CUDA去重采用211：首500为7，原八方法102，FLGMM100全部，LoGo2。09:19实测服务器95/500、220/800保存验收，不能等同离机采用。FL100离机900指标/2400计数/300规则通过、native差0；LoGo2共39734预测/6指标/16计数重算误差0，原冻结CPU netcal不refit，剩98在09:23实际启动。旧CPU失败与中断证据不覆盖。当前实际冻结1500条，完整2500目标尚未齐备；完整CUDA表待完成。Full100共享一次，不称800控制全部完成或每评分项不可或缺。 |'''+r[end:]
for link in re.findall(r'\[[^\]]+\]\((E:/[^)]+)\)',r):assert Path(link.split('#',1)[0]).is_file(),link
protocol=T/'celeba_uniform_cuda_evaluation_v1/PROTOCOL.md';oldprotocol=protocol.read_bytes()
add=f'''\n\n## 2026-10-11 actual LoGo100 deployment and closure increment\n\nRecorded {now}. Author approval a1f1be… remains unchanged. Actual disjoint frozen scope is now1500=first500+original800+FL100+LoGo100. This is an execution addition within the approved uniform-CUDA scope, not a new training/test campaign or a changed metric/fit/selection rule. LoGo reuses root-adopted FedAvg100 CUDA caches and original frozen CPU netcal postprocessors. 712 original artifacts SHA matched; independent Python3.10.20 versus original3.10.4 is disclosed. Failed environment attempt retained; repaired pinned direct CPU packages and actual2 checks passed.\n\nRoot adopted FL100=prior3+new97 (900 metrics/2400 counts/300 decision rules/native max0), and LoGo fixed2 (39734 predictions/6 metrics/16 recomputed confusion counts/error0). LoGo remaining98 started only after fixed2 offserver audit/root adoption and fresh unchanged CPU113 guard. Exact root proofs: {ref(paths['FL_root'])} and {ref(paths['LoGo_root'])}; actual start {ref(paths['LoGo_start98'])}. LoGo native frozen-post view is not an independent newly trained model; no new CNN/fit/refit/test. Historical CPU differences are retained diagnostics, not a required equality gate.\n\nTotal unique root adoption211=first7+old800 unique102+FL100+LoGo2; Full appears only once. Pending server saved results do not become adopted evidence. Target2500 and complete comparison remain incomplete.\n'''
D.mkdir();(D/'PRIOR_STATE.json').write_bytes(before);(D/'PRIOR_READER.md').write_bytes(oldreader);(D/'PRIOR_PREFIX.md').write_bytes(prefix);(D/'PRIOR_PROTOCOL.md').write_bytes(oldprotocol)
reader.write_bytes(r.encode());protocol.write_bytes(oldprotocol+add.encode())
u['protocol']=ref(protocol);s['author_review_reading_entry_20261011'].update(utc=now,reader=ref(reader),uniform_CUDA_root_adopted=211)
s['current_entry_writer']=dict(**ref(__file__),utc=now);s['local_work_verified_utc']=now;s['updated_unix']=datetime.now(timezone.utc).timestamp()
assert sp.read_bytes()==before;sp.write_text(json.dumps(s,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8',newline='\n')
for p,raw,tail,h in entries:
 assert p.read_bytes()==raw;p.write_bytes(text.encode()+tail);assert hashlib.sha256(p.read_bytes()[len(text.encode()):]).hexdigest()==h
proof=dict(status='ROOT_CUDA211_FL100_LOGO2_GIT72_CURRENT_ENTRIES_RECORDED',utc=now,source=ref(__file__),state_before_sha256=hashlib.sha256(before).hexdigest(),state_after=ref(sp),reader=ref(reader),protocol=ref(protocol),five_entries=[ref(x[0]) for x in entries],history_tails_exact=True,evidence={k:ref(p) for k,p in paths.items()},root_adopted_CUDA=211,old800_unique_accepted=102,FL100_accepted=100,LoGo2_accepted=2,LoGo98_started=True,remote500_saved_accepted=n500,remote800_saved_accepted=n800,new_scientific_adoptions_in_metadata_writer=0,whole_rebuttal_complete=False,root_FL_adopter_initial_schema_error='KeyError exact_ids while frozen scope uses ids; before output mutation; corrected key only; original science and data unaffected')
p=D/'ROOT_METADATA_ADOPTION.json';p.write_text(json.dumps(proof,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n');print(json.dumps({'state':ref(sp),'metadata':ref(p)}))
