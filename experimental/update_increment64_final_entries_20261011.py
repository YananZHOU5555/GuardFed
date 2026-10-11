"""Root metadata-only adoption of already verified350/F50; preserve historical bytes."""
from pathlib import Path
import datetime, hashlib, json

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / 'docs/server_deployment_20260923/training_20260923'
H = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
R = lambda p: json.loads(Path(p).read_bytes())
def pin(rel, sha):
    p = ROOT / rel
    assert p.is_file() and H(p) == sha, rel
    return p
def ref(p): return {'path': p.relative_to(ROOT).as_posix(), 'sha256': H(p)}
def create(p, v):
    with p.open('x', encoding='utf8', newline='\n') as f:
        json.dump(v, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')

sp = T / 'TRAINING_STATE.json'
assert H(sp) == '9f0bf995521ab882ad0155e6847cc2828eb74731ca38f32d3a72c2cbe21fcc44'
before = sp.read_bytes(); s = json.loads(before)
vpath = pin('tmp/celeba_mechanism_remaining_F_SpDFA10_root_adoption_20261011/ROOT_ADOPTION.json', '3f7b436e0b2fec0401d7bf6117d6dca48aef21c185b6f170d37954e2cafee5a4')
v = R(vpath)
ixp = pin(v['records_index_path'], '5b24956b4f14707e2a284fe73ffb538545b4aad218c833e31427720402ff02b3'); ix = R(ixp)
tp = pin('docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_five_scenes50_20261011/ROOT_VERIFICATION.json', 'cf89d438b80f48df617c994084a69d8d913cf7dc35c65c762d8800517b89590b'); tv=R(tp)
ep = pin('tmp/rebuttal_working_F50_gradient64_update_20261011/ROOT_EDITORIAL_ADOPTION.json', '72fe35446ef5efc6f39455b56466803f5123fe0dce4f25b6ca654d1a6eee09cb'); ev=R(ep)
dp = pin(ev['draft_path'], 'b6e1c35be9dd0bea4867ff8486c23f7b7d39d0a6966e449dda5ec0664d515670')
op = pin('docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/root_five_queue_20261010T235053Z.raw.json', 'af956c7e32e6bf3496d6e4a7b6b4ce11effbb0d27ac67c26b9d4ea1684cca992'); ov=R(op)
gp2 = pin('tmp/gradient_source_freeze_deployment_index_20261011/GATES_PROGRESS02.json', 'ead7b62bf3b3f4ab043c7d66ac9019a7b06911fb68566dff11c9385b9289409a'); g=R(gp2)
gp = pin('tmp/gradient_source_freeze_deployment_index_20261011/gate14_job02_actual001/ITEM_PROOF.json', '5562dc15e2c46b8b9311226af82932be9711edf382ec711584842fafb05d0c24'); item=R(gp)
assert item['status']=='ONE_ACTUAL_ORIGINAL_CPU3_GATE_PASS_NOT_WHOLE_ACCEPTANCE' and item['index']==2 and item['acceptance']['status']=='PASS'
assert v['cumulative_accepted']==350 and v['original340_unchanged'] and v['remaining620_new_accepted']==170
assert tv['paired_models']==50 and tv['records']==100 and tv['actual_views350_root_sha256']==H(vpath)
assert ev['accepted_native']==350 and ev['accepted_three_view']==350 and ev['F_complete_scenes']==5
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
g.update(utc=now, completed=3, index2_manually_started=False, index3_manually_started=True)
g['items'].append(dict(index=2, job_id=item['job_id'], **ref(gp), resource_sha256=item['resource_sha256']))
g['scope_note']='Three original individual CPU3 passes; whole14 comparison/offserver/root closure pending. Index3 running at metadata cutoff. Not paper records or GPU/70round equivalence.'
gp3=ROOT/'tmp/gradient_source_freeze_deployment_index_20261011/GATES_PROGRESS03.json'
assert not gp3.exists()

m=s['celeba_mechanism_v1']; r=s['mechanism_remaining620_valid_20261010']
assert m['three_view_new_models_offserver_verified']==340 and len(m['three_view_accepted_ids'])==340
assert len(ix['all_ids'])==350 and ix['all_ids'][:340]==m['three_view_accepted_ids']
assert ix['all_ids'][340:]==v['accepted_new_ids'] and r['new_offserver_accepted']==160
assert not set(r['accepted_ids']) & set(v['accepted_new_ids'])
m.update(three_view_new_models_accepted=350, three_view_new_models_offserver_verified=350,
    three_view_accepted_ids=ix['all_ids'], three_view_counts_by_variant={'minus_U':100,'minus_C':100,'minus_A':100,'minus_F':50},
    three_view_root_proof_sha256=H(vpath), latest_three_view_index=ref(ixp)['path'], latest_three_view_index_sha256=H(ixp),
    three_view_scope_limit='U/C/A each100 plus F50; native350 and saved-view350 accepted. Five nonIID F scenes50 plus four controls400 remain450. Validation only; prediction-rule tradeoffs retained, no universal necessity/significance/final-test claim.',
    queue_observation_utc=ov['utc'], queue_observation_path=ref(op)['path'], queue_active_observed=8, queue_pending_observed=440, queue_failures_observed=0)
m['F_three_view_five_scene_table']=dict(root_proof_path=ref(tp)['path'],root_proof_sha256=H(tp),table_path=tv['table_path'],paired_models=50,complete_scenes=5,displayed_records=100,seed_panels=[10,9,6],mean_SD_scalars=810,display_cells=405,validation_only=True,final_test=False,whole_rebuttal_complete=False)
r.update(new_offserver_accepted=170,root_adoption_path=ref(vpath)['path'],root_adoption_sha256=H(vpath),latest_accepted_new_ids=v['accepted_new_ids'],F_complete_scenes=[['IID',a] for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')],F_partial_scenes=[],F_partial_seed_ids=[],F_IID_five_scene_table_adopted=True)
r['accepted_ids']=r['accepted_ids']+v['accepted_new_ids']; assert len(r['accepted_ids'])==170
s['prior_working_F40_draft_fixed_cutoff']=s['latest_working_rebuttal_draft']
s['latest_working_rebuttal_draft']=dict(status='ROOT_UNIFIED_COMPLETE24_F50_GRADIENT64_WORKING_REBUTTAL_ADOPTED_FOR_AUTHOR_REVIEW',entry=ref(dp)['path'],sha256=H(dp),root_proof=ref(ep),comments_verbatim=24,content_cutoff=dict(native=350,three_view=350,F_pairs=50,F_scenes=5,Hybrid_three_view=10,gradient_search=64),F50_incorporated=True,FL81_or_gate14_incorporated=False,author_review_only=True,submitted_manuscript_edited=False,final_test=False,whole_rebuttal_complete=False)
s['latest_five_queue_readonly_observation']=dict(**ref(op),utc=ov['utc'],main_completed=352,main_active=8,main_pending=440,failed=0,FLGMM_new_terminal=79,gradient_terminal=64,Hybrid_new_terminal=21,remaining620_remote_closed=172,counts_are_observation_only=True,new_acceptance=0,source_hash_scope='Small frozen source/data pins unchanged; large images historical full SHA plus stat only',scope='Terminal observation distinct from strict/offserver acceptance; FL/Hybrid screen4 reuse separate.')
r['latest_measured_observation']=dict(**ref(op),utc=ov['utc'],remote_closed=172,offserver_accepted=170)
s['flgmm_fullcoverage_v2_20261009']['latest_readonly_terminal_observation']=dict(**ref(op),utc=ov['utc'],new_terminal=79,new_accepted=77,reuse_separate=4)
s['hybrid100_fullcoverage_20261010']['latest_readonly_terminal_observation']=dict(**ref(op),utc=ov['utc'],new_terminal=21,new_accepted=12,reuse_separate=4)
gs=s['gradient200_fullcoverage_source_preparation_20261010'];gs.update(exact14_completed_at_cutoff=3,exact14_partial_progress={'path':gp3.relative_to(ROOT).as_posix()},limitations='192new+8reuse source frozen; original CPU3 individual gates3/14 pass, index3 running. Whole saved comparison/strict offserver/root closure and fresh dispatch review pending. No192 training, final test or short-gate scientific table claim. Initial metadata preflight refusal retained; one fresh resource check passed, no scientific retry.')
s['updated_unix']=datetime.datetime.now(datetime.timezone.utc).timestamp()
s['local_work_verified_utc']=now;s['current_entry_writer']=dict(**ref(Path(__file__)),utc=now)

prefix=f'''# CURRENT: GuardFed返修实验

更新时间：{now}。服务器 ssh -p60350 root@89.22.197.55，实例52183675；repo /workspace/GuardFed-celeba-expanded。先遵守/etc/vast-agents-guide.md。sglang按授权停止，其他项目不动。

## 已验收证据与实测

| 阶段 | 原strict、离机SHA和root已采用 | 23:50:58 UTC实测（不代替验收） |
|---|---:|---|
| 机制训练 | 350/800新增；Full100另复用 | 352完成、8活动、440等待、失败0；活动28–49轮 |
| 机制三视图 | U100/C100/A100/F50，共350 | remaining620远端闭合172，离机采用170 |
| FLGMM覆盖 | 新增77/96；screen4另复用 | 新增终轮79；三视图81，八完整场景表80，部分场景1另存 |
| Fed-NGA/Huber搜索 | 全64验收并按原规则选参 | 服务自然结束，未重启 |
| 组合基线覆盖 | 新增12/96；screen4另复用 | 新增终轮21；IID Benign三视图10已采用 |

原生机制350：{m['latest_native_root']['path']}。最新增量9项补齐去F IID Sp-DFA十seed，实际102归档成员，旧441记录与48账本前缀不变，当前49条链。三视图350：{ref(vpath)['path']}。原340对象/顺序保持，新10项原数组重放、零新拟合/训练/CNN，实际92归档成员与离机SHA通过。

F50论文表：{tv['table_path']}。IID Benign/F Flip/FedSA/S-DFA/Sp-DFA各10配对；raw/native/shared及固定10/9/6面板，均值±sampleSD(ddof1)。原F40对象、648统计量与324显示单元不变。新Sp-DFA的native/shared去F−Full均值：ACC+0.1173百分点、AEOD−0.00661、ASPD+0.01725；raw两种公平差距均变差。S-DFA的Full均值优势及其他场景的取舍/反向保留，说明预测规则依赖，不作普遍必要性、显著性或独立因果主张。仍缺五个non-IID F场景50条及四类控制400条，共450三视图未接纳；不等于全部尚未训练。

FLGMM八完整场景表：outputs/guardfed_tables/celeba_flgmm_eight_scenes80_20261011/TABLES.md。non-IID FedSA10：native ACC89.15±0.88%、AEOD0.0428±0.0049、ASPD0.1088±0.0055；shared88.77±0.81%、0.0087±0.0079、0.0689±0.0061，准确率代价保留。原71记录不变。Linux原校准通过，Windows仅零拟合审计保存输出；原Windows47重拟合失败、资源拒绝、首个FedSA传输ID拒绝及单个groupKL差−2.168404344971009e−19保留，不称跨平台逐位等价。

梯度64搜索：8配置/方法×IID/non-IID Benign/S-DFA×seed91001，70轮valid。冻结四条件score均值、精确并列字典序选择NGA eta0.001及Huber eta0.3/T0=0.01/M0。Huber27/32、NGA12/32恒负预测，全部负结果、准确率冠军和Pareto保留。n=1无sampleSD/显著性；零差距不当公平性胜出，不推断原算法普遍失效。

192新增+8复用覆盖源码已冻结。原真实图像CPU三轮门检当前3/14单项通过，index3运行；整体验收/离机保存及启动资源复核仍待完成，192覆盖未启动。短门检不是70轮性能或GPU等价证明。FFlip在此梯度桥接中是元数据空效应，保留其边界；初次资源元数据拒绝保留，未更改科学函数。

## 回复稿与边界

当前24条原话统一英文工作稿：{ref(dp)['path']}，固定native350/views350/F50/gradient64/Hybrid表10。原24块38行、标题与顺序精确保持；未纳入FL81或门检结果，供作者审阅，尚未应用提交版正文。历史F40稿固定341/340/F40截面，保留原字节。

Huber按用户决定采用恒等投影，明确CNN项目适配，不沿用原约束域理论保证。LoGoFair固定图像ID映射20虚拟cohort并用官方DP后处理；DP为人口统计平等，标为人口适配，不称真实训练client公平性。

十方法native1000及九方法三视图900/2052项归因已接纳；目标17方法仍未齐。尚需剩余基线忠实覆盖、机制控制、冻结最终评价和提交版正文/rebuttal收尾。FedWA/SmartFL/FedDNA忠实规格、主终点/测试边界、匹配LaTeX和Fig3来源问题待解决，已有问题不重复询问。最终partition2仅核19962 image_id元数据，未读标签/像素或作最终推理。validation选参、旧test暴露及cu128/cu130/driver差异继续披露。

双GPU此次100/100%，65/61°C，6038/5037MiB，RecoveryAction None；cgroup RAM76.27/519.17GB、OOM0，磁盘余1.060TB。CPU有界诊断见tmp/cpu_throughput_readonly_20261011/ROOT_DIAGNOSIS.json；冻结worker/thread结构可解释低总CPU占比，未证明算子瓶颈或最优吞吐，未改数值/并发/实例。

## 保存与接续

大文件只写F:/YananResearchStorage/GuardFed，写前核Yanan 2TB/Healthy/容量；E仅代码/配置/索引/精简报告。模型、数组、数据和归档不进Git，优先留服务器，不重复历史备份。

最近已验证Git63：ccb5207c24762d3ac92ed76b40bd40b3e2330954，191 blob/远端通过；其固定截止native341/views340/F40、FL67/71/table70、gradient46。上述新350/F50、FL77/81/table80、gradient64和新工作稿待下一紧凑增量发布，不能冒称已上传。三小时聊天任务guardfed-training-health仍PAUSED，当前无automation_update接口，未建替代cron/Windows任务；supervisor训练独立继续。

入口写入器：{Path(__file__).relative_to(ROOT).as_posix()}。下方历史原字节保持。

'''.encode('utf8')
entries=[
 (T/'RUNNING.md',b'# HISTORICAL:','203700dc6c349a10fc18f09e271df6db05f0aba0fcdf42617cce1f0bf6328a44'),
 (T/'REBUTTAL_COMPLETION_20261009.md',b'## Historical accepted increment','3aa956a830219050d59e9577c87d5b29b66520d26bee018d54335b9cd1630080'),
 (ROOT/'docs/返修实验总览.md','以下为 2026-10-04'.encode('utf8'),'08b94e001b4ff78474706db0a63cc2c1b626fe5664c01edfafbaefb53fe9b029'),
 (T/'server_reactivation_20261009/MONITOR_HANDOFF.md',b'# Historical handoff snapshots','f9dfedaf289de16f1226d2356ccfa451034e2e5e05b19348f0c08095e5a0a1ee'),
 (T/'celeba_mechanism_v1/EXECUTION.md',b'# HISTORICAL PREPARATION SNAPSHOT','f1056563ae8ac5d19dd0d8a076541298ebff298ccde73ac951954b9e83dde7cd')]
pending=[]
for p,mark,sha in entries:
    old=p.read_bytes();tail=old[old.index(mark):];assert hashlib.sha256(tail).hexdigest()==sha
    pending.append((p,old,prefix+tail,mark,sha))
assert sp.read_bytes()==before
create(gp3,g);gs['exact14_partial_progress']['sha256']=H(gp3)
sp.write_bytes((json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))
for p,old,new,mark,sha in pending:
    assert p.read_bytes()==old;p.write_bytes(new)
for p,old,new,mark,sha in pending:
    actual=p.read_bytes();assert actual[:actual.index(mark)]==prefix
    assert hashlib.sha256(actual[actual.index(mark):]).hexdigest()==sha
proof=dict(status='ROOT_CLOSED_INCREMENT64_FINAL350_F50_METADATA_AND_FIVE_ENTRIES_UPDATED',utc=now,state_before_sha256=hashlib.sha256(before).hexdigest(),state_after_sha256=H(sp),writer_sha256=H(__file__),prefix_sha256=hashlib.sha256(prefix).hexdigest(),entries={p.relative_to(ROOT).as_posix():{'sha256':H(p),'history_exact':True} for p,_,_,_,_ in pending},all_five_prefixes_equal=True,historical_tails_byte_exact=True,accepted=dict(native_new=350,three_view_models=350,F_paired_models=50,FL_native_new=77,FL_three_view_total=81,FL_complete_table_records=80,gradient_accepted=64,Hybrid_native_new=12),remaining620_adopted=170,latest_observation=ref(op),gate14_partial_progress=ref(gp3),new_scientific_checks=0,final_test=False,whole_rebuttal_complete=False)
pp=ROOT/'tmp/update_increment64_final_entries_20261011.proof.json';create(pp,proof)
print(json.dumps({'status':proof['status'],'STATE_sha256':H(sp),'proof_sha256':H(pp),'gate_progress_sha256':H(gp3)}))
