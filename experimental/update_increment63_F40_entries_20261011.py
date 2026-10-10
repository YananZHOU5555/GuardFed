"""Refresh current pointers to already adopted341/340/F40; preserve history bytes."""
from pathlib import Path
import copy, datetime, hashlib, json, os

R=Path(__file__).resolve().parents[1]
T=R/'docs/server_deployment_20260923/training_20260923'
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
rel=lambda p:Path(p).relative_to(R).as_posix()
sp=T/'TRAINING_STATE.json'
prior=read(R/'tmp/increment62_postpublication_pointer_refresh_20261011.json')
assert H(sp)==prior['STATE_sha256']=='8a8f58019b9472d8da119ea4277be5b315ecc56340fc7a9adaa9acf15937bc56'
s=read(sp); old=copy.deepcopy(s)
np=T/'server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T222658Z/ROOT_DELTA_VERIFICATION.json'
vp=R/'tmp/celeba_mechanism_remaining_F_SDFA10_root_adoption_20261011/ROOT_ADOPTION.json'
fp=T/'celeba_mechanism_v1/three_view_F_IID_four_scenes40_20261011/ROOT_VERIFICATION.json'
ep=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_working_20261011.root_editorial_review.json'
dp=R/'tmp/cpu_throughput_readonly_20261011/ROOT_DIAGNOSIS.json'
pins={np:'755f996594675c858b696ec1e0ac7038de9e8ad2a1fd011212fc78a38b736ba4',vp:'80a7f38890c2a7f8c75c0472489315bb2acb9e43c9e28d686a799cfea9709a02',fp:'a57b5d235a39190e6a2a88e1ebe291813f976efbe91fcf571f17357837304f23',ep:'0cf74dba2727092895f6792d0aaea91c27957791aa3ba1bd93be2fc3e943833b',dp:'a7950b1020b77fccbc9f7fb539a0ec8fe3c7e25e7495798db84a9b80cc6860bd'}
for p,h in pins.items():assert H(p)==h
n,v,f,e=map(read,[np,vp,fp,ep])
assert n['root_adopted'] and n['canonical_local_and_remote_ledger_promoted'] and n['total_new_strict_and_offserver']==341 and n['old433_records_exact'] and n['root_checked_archive_members']==94
assert (v['prior_accepted'],v['new_accepted'],v['cumulative_accepted'],v['remaining620_new_accepted'])==(330,10,340,160)
assert v['native_root_sha256']==H(np) and v['original330_unchanged'] and v['test'] is False
idx=R/v['records_index_path'];ix=read(idx);assert H(idx)==v['records_index_sha256'] and len(ix['all_ids'])==340
assert f['root_adoption'] and (f['paired_models'],f['complete_scenes'],f['displayed_records'])==(40,4,80)
assert f['source_acceptance_sha256']==H(vp) and f['source_native_root_sha256']==H(np) and f['final_test'] is False
assert e['original_comment_blocks']==24 and (e['native_accepted'],e['three_view_accepted'])==(333,330) and e['author_review_only']
op=T/'server_reactivation_20261009/root_five_queue_20261010T223921Z.raw.json';o=read(op)
assert H(op)=='3bc6db9fcd3b5d2ed350d92b5144fb90fc101fddcbd382cd38aababbb4e62a71'
assert o['guide_sha256']=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert o['source_data_binding']['changed_paths']==[]
for k in ['main_mechanism','FLGMM','gradient64','Hybrid96','remaining620']:assert 'RUNNING' in o[k]['service']['stdout'] and o[k]['service']['returncode']==0
q=o['main_mechanism']['queue'];assert not q['failed'] and len(q['completed'])==344 and len(q['active'])==8 and q['pending']==448
assert not o['FLGMM']['queue_failed'] and not o['FLGMM']['failure_paths'] and not o['gradient64']['failure_paths'] and not o['Hybrid96']['failure_files'] and not o['remaining620']['failures']
assert (len(o['FLGMM']['terminal_ids']),len(o['gradient64']['terminal_ids']),len(o['Hybrid96']['terminal_records']),o['remaining620']['remote_closed_n'])==(77,62,19,164)
m=s['celeba_mechanism_v1'];assert (m['scientific_results_offserver_verified'],m['three_view_new_models_offserver_verified'])==(333,330)
for k in ['new_completed','scientific_results_completed','scientific_results_strictly_accepted','scientific_results_offserver_verified']:m[k]=341
m.update(science_acceptance_inspection=rel(np.parent/'inspection/inspection.json')[len(rel(T))+1:],latest_science_backup_sha256=n['archive_sha256'],latest_science_backup_members_verified=94,latest_native_root_proof_path=rel(np),latest_native_root_proof_sha256=H(np))
back=m['incremental_science_backups'];assert len(back)==47 and len({i for b in back for i in b['new_ids']})==333
assert set(n['new_ids']).isdisjoint(i for b in back for i in b['new_ids'])
back.append(dict(archive_sha256=n['archive_sha256'],new_ids=n['new_ids'],members_verified=94,root_proof_path=rel(np),root_proof_sha256=H(np)))
assert back[:47]==old['celeba_mechanism_v1']['incremental_science_backups'] and len({i for b in back for i in b['new_ids']})==341
m['native_after333_root_adoption']=dict(new_accepted=8,total_accepted=341,complete_added_scene='minus_F IID S-DFA ten seeds',partial_Sp_DFA_seeds=[91001],root_path=rel(np),root_sha256=H(np),archive_members_verified=94,final_test=False)
for k in ['three_view_new_models_accepted','three_view_new_models_offserver_verified']:m[k]=340
m.update(three_view_accepted_ids=ix['all_ids'],three_view_counts_by_variant=dict(minus_U=100,minus_C=100,minus_A=100,minus_F=40),three_view_root_proof_sha256=H(vp),latest_three_view_index=rel(idx),latest_three_view_index_sha256=H(idx))
m['three_view_scope_limit']='U/C/A each100; F40 covers IID Benign/F Flip/FedSA/S-DFA; fixed10/9/6 panels retain all reversals. Six F scenes60 plus four controls400 remain; native341 and saved340 differ. Validation only, no universal necessity, significance or final-test claim.'
m['F_three_view_four_scene_table']=dict(root_proof_path=rel(fp),root_proof_sha256=H(fp),table_path=rel(fp.with_name('TABLES.md')),paired_models=40,complete_scenes=4,displayed_records=80,seed_panels=[10,9,6],mean_SD_scalars=648,display_cells=324,validation_only=True,final_test=False,whole_rebuttal_complete=False)
m.update(queue_completed_observed=344,queue_active_observed=8,queue_pending_observed=448,queue_failures_observed=0,new_started=352,queue_observation_utc=o['utc'],queue_observation_path=rel(op),new_started_observation_utc=o['utc'])
rem=s['mechanism_remaining620_valid_20261010'];assert rem['new_offserver_accepted']==150 and set(v['accepted_new_ids']).isdisjoint(rem['accepted_ids'])
rem.update(new_offserver_accepted=160,accepted_ids=rem['accepted_ids']+v['accepted_new_ids'],latest_accepted_new_ids=v['accepted_new_ids'],root_adoption_path=rel(vp),root_adoption_sha256=H(vp),F_complete_scenes=v['complete_F_scenes'],F_partial_scenes=[],F_partial_seed_ids=[],F_IID_four_scene_table_adopted=True)
obs=dict(path=rel(op),sha256=H(op),utc=o['utc'],main_completed=344,main_active=8,main_pending=448,failed=0,FLGMM_new_terminal=77,gradient_terminal=62,Hybrid_new_terminal=19,remaining620_remote_closed=164,counts_are_observation_only=True,new_acceptance=0,source_hash_scope='Small frozen source/data pins unchanged; large images historical full SHA plus stat only',scope='Five live service/worker identities and source/data pins checked; terminal observations do not replace strict/offserver acceptance. FL and Hybrid screen4 reuse separate.')
s['latest_five_queue_readonly_observation']=obs
s['latest_main_queue_readonly_observation']=dict(path=rel(op),sha256=H(op),utc=o['utc'],completed=344,active=8,pending=448,failed=0,observation_only=True,science_acceptance_created=0)
for k,c in [('gradient64_validation_search_20261010',62),('flgmm_fullcoverage_v2_20261009',77),('hybrid100_fullcoverage_20261010',19),('mechanism_remaining620_valid_20261010',164)]:s[k]['latest_measured_observation']=dict(checked_utc=o['utc'],terminal_observed=c,snapshot_path=rel(op),snapshot_sha256=H(op),acceptance_unchanged_by_observation=True)
s['latest_working_rebuttal_draft']=dict(status=e['status'],entry=e['draft_path'],sha256=e['draft_sha256'],root_proof_path=rel(ep),root_proof_sha256=H(ep),comments_verbatim=24,content_cutoff=dict(native=333,three_view=330,F_pairs=30,F_scenes=3,Hybrid_table_checkpoints=10),F40_incorporated=False,author_review_only=True,submitted_manuscript_edited=False,final_test=False,whole_rebuttal_complete=False)
s['latest_bounded_CPU_diagnosis']=dict(path=rel(dp),sha256=H(dp),science_or_resource_settings_changed=False,scope='15.18s/4 snapshots; ~12 active single-thread workers/~12.09 cgroup cores; bothGPU100% in samples, no sampled disk reads or new throttle. Consistent with frozen worker/thread structure; not a proved operator bottleneck or optimal throughput.')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
prefix=f'''# CURRENT: GuardFed返修实验

本段更新时间：{now}。服务器 ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded；先遵守/etc/vast-agents-guide.md。sglang已按授权停止，文件保留。

## 已接纳结果与实时观测

| 阶段 | 原strict、离机SHA及root验收 | 22:39:26 UTC实测（不代替验收） |
|---|---:|---|
| 机制训练 | 341/800新增；100 Full另复用 | 完成344、活动8、等待448、失败0；活动第5–23轮 |
| 机制三视图 | 340终轮checkpoint：U100/C100/A100/F40 | remaining620远端闭合164，离机接受160另列 |
| FLGMM完整覆盖 | 67/96新增；4screen另复用 | 新增终轮77；三视图接受71，七完整场景表70，另保留一个未齐场景记录 |
| Fed-NGA/Huber搜索 | 46/64 | 终轮62/64；未完成全部搜索或选择recipe |
| 组合基线覆盖 | 12/96新增；4screen另复用 | 新增终轮19；IID Benign三视图十seed表已接纳 |

同次资源实测：双GPU利用率100/100%，温度66/62°C，显存6040/6020MiB，RecoveryAction None；cgroup内存76.45/519.17GB、OOM0，磁盘余1.060TB。五队列服务与worker身份通过，small source/data hash未变，大图像只核stat及历史全量SHA。记录：{rel(op)}。瞬时采样不证明持续占用或完成全部实验，冻结方法、seed、并发、FP32和统计口径保持。

本批机制native新接纳8项：补齐去F的IID S-DFA十seed，并接纳Sp-DFA seed91001，总341/800。94归档成员SHA及终轮/配置/源码/数据/checkpoint身份通过，旧433记录和47条恢复链前缀不变，当前48条链已索引。S-DFA10保存数组随后核验并接纳，三视图330→340；Sp-DFA一项native尚不计为三视图完成。原始大文件留服务器及F盘，增量证据待Git63发布。

## 新补齐的四个IID场景F消融表

{rel(fp.with_name('TABLES.md'))}

IID Benign/F Flip/FedSA/S-DFA各10共享seed；raw/native/shared三视图及固定10/9/6面板。80记录/40配对，root独立核648均值SD标量、324展示格、720计数派生指标和1920基础计数；旧60记录、486标量、243展示值原样保留。表格制作没有新增拟合、CNN推理或训练。

S-DFA十seed native：Full为ACC88.689±0.809%、AEOD0.00769±0.00463、ASPD0.06377±0.00697；去F为88.412±1.202%、0.01054±0.00697、0.06941±0.01477。去F减Full配对均值为ACC−0.277pp、AEOD+0.00284、ASPD+0.00564；raw/native/shared的10/9/6九个面板三指标均值方向均支持Full，不作显著性、普遍必要性或单独因果主张。

其他场景的取舍保留：F Flip的9/6seed准确率差值反向；FedSA十seed去F准确率均值+0.136pp，六seed native/shared的ASPD差值−0.00560；Benign也有取舍。native/shared相同不作独立确认，混合设备、环境和选择史均披露。F仍缺六场景60项，另四类机制控制400项：共460三视图未接纳；native未接纳459项，不能称这些任务全部尚未训练。

FL七完整场景表70及单列部分记录已接纳，表：outputs/guardfed_tables/celeba_flgmm_seven_scenes70_20261011/TABLES.md。non-IID F Flip十seed native ACC89.28±0.98%、AEOD0.0525±0.0077、ASPD0.1185±0.0068；shared为88.90±1.04%、0.0091±0.0073、0.0713±0.0111，存在准确率代价。旧Windows全量重拟合失败、五次资源拒绝、Linux原校准通过及Windows零拟合计数审计均保留，不称跨平台逐位一致。

Hybrid IID Benign三视图十seed表已接纳，表：outputs/guardfed_tables/celeba_hybrid_three_view_IID_Benign10_20261011/TABLES.md。原screen seed91001身份与选择史、已验收seed91002及八个覆盖checkpoint显式保留；其余场景继续。native ACC88.027%、AEOD0.02920、ASPD0.09251；shared87.708%、0.02321、0.03383，有准确率代价。Hybrid为项目控制，不称外部原方法；原Windows resource导入失败及采样审计微小差异保留。

## 返修边界与接续

十方法native验证表1000格，IID/non-IID各五场景已交付；九方法三视图900记录和2052项归因已接纳。U/C/A各100配对、十场景表已接纳。尚缺七方法完整覆盖、其余机制控制、冻结最终评价及提交版正文/rebuttal收尾。FedWA/SmartFL/FedDNA忠实规格、最终主终点/测试边界、提交版LaTeX源及Fig3执行来源仍待解决；已有作者问题不重复询问。

24条原意见统一英文工作稿：{e['draft_path']}；已root核原话与顺序、F30/Hybrid数值及链接，供作者审阅。该稿内容截止native333/views330/F30；本次F40为单独新证据表，尚未写进该稿。F20旧稿、正文插入候选及F30补充稿保持各自截止；未应用提交版正文，不能称整份返修完成。

Huber按用户同意采用恒等投影，标为CNN项目适配且不继承原约束域理论。已验收14条Huber搜索结果均恒负预测，ACC0.5166859616449389、AEOD/ASPD0，作为退化负结果保留，不称公平性获胜。LoGoFair采用固定图像ID20虚拟cohort，标为人口适配，不称真实client公平性；其100个native结果已接纳。

最终候选partition2的19962个image_id元数据已核，未读标签/像素或作最终模型推理；validation选择史、旧test暴露、cu128/cu130及driver差异披露。NGA/Huber须64项完整验收后按冻结四条件评分规则选recipe，再过真实图像门检，不能以62项观测启动192覆盖。FL下一批non-IID FedSA10已终轮，先按原strict和离机核验接纳，不把77终轮当77验收。

CPU有界诊断：{rel(dp)}。15.18秒四采样观察约12个活动单线程worker及12.09/122.88核的cgroup平均用量，双GPU均100%；未采到磁盘读取或新增配额节流，符合冻结worker/thread结构。尚不能据此证明具体算子瓶颈或最优吞吐；未改数值、并发或实例设置，利用空闲CPU处理已闭合证据。

## 保存、Git与巡检

大文件只写F:/YananResearchStorage/GuardFed，写前核F为Yanan 2TB/Healthy且容量足；E仅代码、配置、索引与精简报告。模型/原始数组/归档不进Git，服务器大文件优先原地保留。

最近已验证推送fb1f31031aad89206f252969a6c26af6cde9efa4，截止以publication_closed_increment62_verified_20261011.json为准（245提交文件哈希及远端分支通过）。本次341/340/F40、统一工作稿和入口刷新在该提交之后，待Git63实际发布，不能提前称已上传。三小时聊天任务guardfed-training-health仍PAUSED，本会话无automation_update接口；未建替代cron/Windows任务。supervisor训练与聊天巡检分开。

入口写入器：tmp/update_increment63_F40_entries_20261011.py。旧生成器只保留历史截止，下方历史原字节保留。

'''
markers={'RUNNING.md':b'# HISTORICAL:','REBUTTAL_COMPLETION_20261009.md':b'## Historical accepted increment','返修实验总览.md':'以下为 2026-10-04'.encode(),'MONITOR_HANDOFF.md':b'# Historical handoff snapshots','EXECUTION.md':b'# HISTORICAL PREPARATION SNAPSHOT'}
pending=[];proof=dict(status='CURRENT_ENTRIES_ADOPTED_NATIVE341_VIEWS340_F40_AND_UNIFIED_WORKING_DRAFT_CUTOFF333_330',utc=now,source_sha256=H(__file__),entries={},new_science_created=0,new_training=0,new_inference=0,final_test=False)
for path,pin in prior['entries'].items():
 p=R/path;assert H(p)==pin['sha256'];b=p.read_bytes();history=b[b.index(markers[p.name]):]
 assert hashlib.sha256(history).hexdigest()==pin['history_sha256'];pending.append((p,prefix.encode()+history,history,pin))
assert s['latest_rebuttal_draft']==old['latest_rebuttal_draft'] and m['F_three_view_three_scene_table']==old['celeba_mechanism_v1']['F_three_view_three_scene_table']
assert (s['flgmm_fullcoverage_v2_20261009']['new_accepted'],s['gradient64_validation_search_20261010']['offserver_accepted'],s['hybrid100_fullcoverage_20261010']['new_accepted'])==(67,46,12)
for p,content,history,pin in pending:
 p.write_bytes(content);assert p.read_bytes().endswith(history);proof['entries'][rel(p)]=dict(sha256=H(p),history_sha256=pin['history_sha256'],history_exact=True)
s['current_entry_writer']=dict(path=rel(Path(__file__)),sha256=H(__file__),utc=now)
s['updated_unix']=datetime.datetime.now(datetime.timezone.utc).timestamp()
temp=sp.with_suffix('.increment63.tmp');assert not temp.exists();temp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(temp,sp)
proof.update(STATE_sha256=H(sp),native_root_sha256=H(np),view_root_sha256=H(vp),F40_root_sha256=H(fp),backup_index_entries=48,backup_index_ids=341,old47_backup_index_entries_exact=True,observation_sha256=H(op))
out=R/'tmp/increment63_F40_entry_update_proof_20261011.json';assert not out.exists();out.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(STATE_sha256=H(sp),proof_sha256=H(out),entries=len(proof['entries']))))
