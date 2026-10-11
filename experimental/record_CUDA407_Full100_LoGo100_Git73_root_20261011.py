"""Source-only successor of the CUDA211 writer; root executes only after source review.

All evidence, suffix, link, serialization and stale-input checks precede the first write.
No training, inference, fitting, archive/array reads, Git or scheduler operations.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, hashlib, json, re

R = Path.cwd()
T = R / 'docs/server_deployment_20260923/training_20260923'
V = T.parent / 'revision_20260923'
D = R / 'tmp/root_CUDA407_Full100_LoGo100_Git73_20261011'
parser = argparse.ArgumentParser()
parser.add_argument('--partial-table-review', type=Path, default=R / 'tmp/celeba_uniform_cuda_fourmethod_partial_independent_review_20261011/SOURCE_REVIEW.json')
parser.add_argument('--partial-table-review-sha256', default='154c53a04f6216f0e8a651506d9ff648986f9cf0a422636b8e1432a74ddb09d7')
args = parser.parse_args()
args.partial_table_review = args.partial_table_review.resolve()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
ref = lambda p: dict(path=Path(p).relative_to(R).as_posix(), sha256=sha(p))
ref_bytes = lambda p, b: dict(path=Path(p).relative_to(R).as_posix(), sha256=hashlib.sha256(b).hexdigest())
sp = T / 'TRAINING_STATE.json'
before = sp.read_bytes()
assert hashlib.sha256(before).hexdigest() == '49bae8ad6f2ce2529151d198edf44d8e05639d395f68fb82a5e3d82dd788ebfe'
assert not D.exists()
paths = {
    'Full_root': R / 'tmp/Full100_cuda_saved_closure_prepare_v2_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json',
    'LoGo_root': R / 'tmp/LoGo100_cuda100_saved_closure_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json',
    'first7_root': R / 'tmp/uniform_cuda7_saved_audit_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json',
    'FL_root': R / 'tmp/FL100_cuda100_saved_closure_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json',
    'health': T / 'server_reactivation_20261009/ROOT_COMPACT_20261011T094318Z.json',
    'cuda500': R / 'tmp/uniform_cuda500_root_dispatch_20261011/RESUME2_OBSERVATION_20261011T094856Z.json',
    'cuda800': R / 'tmp/baseline_old800_runtime_root_20261011_attempt003/OBSERVATION_20261011T094316Z.json',
    'Git73': T / 'publication_closed_increment73_verified_20261011.json',
    'partial400_build': R / 'tmp/celeba_uniform_cuda_fourmethod_partial_20261011/actual_partial400/BUILD_RECEIPT.json',
    'partial400_table': R / 'tmp/celeba_uniform_cuda_fourmethod_partial_20261011/actual_partial400/TABLES.md',
    'partial400_review': args.partial_table_review,
    'partial400_root': R / 'tmp/celeba_uniform_cuda_fourmethod_partial_20261011/ROOT_TABLE_ADOPTION.json',
    'fixed24_draft': V / 'rebuttal_working_F80_FL100_Hybrid20_native11_20261011.md',
}
pins = {
    'Full_root': '0d33ecb624323f0f334debcb12f888886f85410cd9b0d5607c687dbb6575cc08',
    'LoGo_root': 'c926f7ee3cd6dea31f178d3a6eab3661826f9ecd4e7c90e54d3027e6b94c1317',
    'first7_root': '3ed2e1fe79edbc5c196d428305bf27201856f2e6d5589c6e4752255accec4dda',
    'FL_root': 'ea935ead7da50ba7eff0dd7e4e4b289ca9365427096ad8431808ae4d76acb751',
    'health': 'ffb3aaa73818b58f85df6aaf25bb91cc7c1809fa57a7939d977bb5c2c030180c',
    'cuda500': '01fd7d02525cbfa2774184fde00081e23930c0b8a4ce33e6dc54814808b2cbbf',
    'cuda800': '0795208e228236bd66b7bbf0a1205190a7ddf3275844f1f13dd11a16c1f8e750',
    'Git73': 'dd7145de8cc8b9ec311e545836204e7dfd83b2e96e018195a086bcf7f3efcdd6',
    'partial400_build': '14ab89c336f540dd1109f1bd8d07b706096dd52db78e7bd0e8acf25885328f83',
    'partial400_table': '09157fd0dad7047fd05e7fb0e54ff7c852532a5cdcf3743e26ef91a6ccd2b1b8',
    'partial400_review': args.partial_table_review_sha256,
    'partial400_root': '515724881bcad4903b0e2dd011d5c9646916c7e8c3eed7141aa879891f2e697e',
    'fixed24_draft': '910b263339273c9d973491f1005f35ec4a64ed6a27127616c096ee7f3ffce382',
}
for k, p in paths.items():
    assert p.is_file() and sha(p) == pins[k], k
full = read(paths['Full_root']); lg = read(paths['LoGo_root']); first = read(paths['first7_root'])
fl = read(paths['FL_root']); o = read(paths['health']); c = read(paths['cuda500']); b = read(paths['cuda800'])
git = read(paths['Git73']); table_build = read(paths['partial400_build'])
table_review = read(paths['partial400_review']); table_root = read(paths['partial400_root'])
assert full['root_accepted'] is True and (full['accepted'], full['new_unique'], full['reused_already_adopted']) == (100, 98, 2)
assert (full['metric_values_checked'], full['integer_base_counts_checked'], full['prediction_rules_checked']) == (900, 2400, 300)
assert full['native_max_abs_difference'] == 0 and full['whole100_complete'] is True and full['test'] is False
assert lg['root_accepted'] is True and (lg['accepted'], lg['newly_adopted'], lg['previously_adopted_reused']) == (100, 98, 2)
assert (lg['metric_values_checked'], lg['integer_confusion_counts_recomputed'], lg['prediction_items_checked']) == (300, 800, 1986700)
assert lg['new_fit'] == lg['new_CNN'] == lg['max_metric_error'] == 0 and lg['whole100_complete'] is True and lg['test'] is False
assert lg['prior2_records_exact'] is True and lg['offserver_whole_predict_reexecuted'] is False
assert len(full['ids']) == len(set(full['ids'])) == len(full['records']) == 100
assert len(lg['ids']) == len(set(lg['ids'])) == len(lg['records']) == 100
assert set(full['ids']) == set(full['records']) and set(lg['ids']) == set(lg['records'])
first_ids = set(first['ids']); full_ids = set(full['ids'])
assert first['accepted'] == len(first_ids) == 7 and len(first_ids & full_ids) == 2
assert len(first_ids - full_ids) == 5 and len(first_ids | full_ids) == 105
constant_ids = [k for k, v in lg['records'].items() if v['constant_prediction'] is not None]
assert constant_ids == ['LoGoFair-DP_07_FedAvg_IID_Benign_seed91001']
assert lg['records'][constant_ids[0]]['constant_prediction'] == 0
assert (fl['accepted'], fl['new_unique'], fl['reused_already_adopted']) == (100, 97, 3)
assert git['status'] == 'COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS'
assert git['commit'] == 'f1148225d99e5b5f469e18bcd3d44f0a1c817875' and git['committed_blobs_sha256_verified'] == 246
assert git['STATE_sha256'] == hashlib.sha256(before).hexdigest()
assert table_build['status'] == 'PARTIAL_FOUR_METHOD_UNIFORMLY_CUDA_VALIDATION_NOT_FINAL_COMPARISON'
assert table_build['records'] == 400 and table_build['view'] == 'native' and table_build['ddof'] == 1
assert table_build['methods'] == ['FedAvg', 'GuardFed-AD2+', 'FLGMM', 'LoGoFair-DP']
assert table_build['files']['TABLES.md']['sha256'] == pins['partial400_table']
assert table_review['status'] == 'PASS_INDEPENDENT_ARITHMETIC_AND_FORMAT_VERIFICATION' and table_review['identity_grid_records'] == 400
assert table_review['build_receipt_sha256'] == pins['partial400_build'] and not table_review['unit_findings'] and not table_review['blocking_findings']
assert table_root['root_accepted'] is True and table_root['records'] == 400 and table_root['new_scientific_adoptions'] == 0
assert table_root['build_receipt']['sha256'] == pins['partial400_build'] and Path(table_root['build_receipt']['path']) == paths['partial400_build']
assert table_root['independent_review']['sha256'] == pins['partial400_review'] and Path(table_root['independent_review']['path']) == paths['partial400_review']
assert table_root['table']['sha256'] == pins['partial400_table'] and table_root['whole_comparison_complete'] is False
cp = c['progress.json']; n500 = len(cp['remote_saved_accepted_ids']); n800 = len(b['accepted']) + len(b['reused'])
assert n500 == 108 and n800 == 277 and not cp['failed'] and not b['failures']
nmain = len(o['main_queue']['completed']); nnga = len(o['NGA']['terminal_ids_observed']); nhybrid = len(o['Hybrid']['terminal_ids_observed'])
assert (nmain, nnga, nhybrid) == (447, 27, 37) and not o['main_queue']['failed']
assert len(o['main_active_progress']) == 8 and o['main_queue']['pending'] == 345
gpu_rows = [x.split(', ') for x in o['GPU']['stdout'].splitlines()]
assert o['GPU']['exit'] == 0 and len(gpu_rows) == 2 and all(len(x) == 5 and x[4] == 'None' for x in gpu_rows)
gpu_util = [int(x[1]) for x in gpu_rows]; gpu_temp = [int(x[3]) for x in gpu_rows]
assert gpu_util == [100, 100] and gpu_temp == [67, 62]
memory_events = dict(x.split() for x in o['cgroup']['memory.events'].splitlines())
assert all(int(memory_events[k]) == 0 for k in ('oom', 'oom_kill', 'oom_group_kill'))
quota, period = map(int, o['cgroup']['cpu.max'].split()); quota_cores = quota / period
now = datetime.now(timezone.utc).isoformat(); s = json.loads(before); prior_s = json.loads(before); u = s['uniform_CUDA_evaluation_v1']
assert (u['uniform_CUDA_scientific_accepted'], u['root_adopted_first500'], u['root_adopted_old800_unique'], u['root_adopted_FL100'], u['root_adopted_LoGo100']) == (211, 7, 102, 100, 2)
assert 105 + 102 + 100 + 100 == 407 and u['actual_all_frozen'] == 1500 and u['future_target_logical_records'] == 2500
assert u['complete_target_inventory_frozen'] is False and u['test'] is False
u.update(status='ROOT407_ADOPTED_FULL100_FL100_LOGO100_COMPLETE_COMPARISON_PENDING', uniform_CUDA_scientific_accepted=407,
    root_adopted_first500=105, root_adopted_Full100=100, Full100_root=ref(paths['Full_root']),
    Full100_new_unique=98, Full100_reused_already_adopted=2, Full100_whole_complete=True,
    first500_other_controls_root_adopted=5, Full100_shared_once=True,
    current_remote_saved_accepted=n500, current_remote_new_after_resume=cp['accepted_new_cuda'],
    current_remote_active=cp['active'], current_remote_pending=cp['pending'], current_remote_failed=0, current_observation=ref(paths['cuda500']),
    old800_remote_saved_accepted=n800, old800_remaining_queue_new_accepted=len(b['remaining_progress']['new_accepted_ids']),
    old800_current_pending=b['remaining_progress']['remaining'], old800_observation=ref(paths['cuda800']),
    LoGo_prior2_root=u['LoGo100_root'], root_adopted_LoGo100=100, LoGo100_root=ref(paths['LoGo_root']),
    LoGo_whole100_complete=True, LoGo_remaining98_complete=True, LoGo_newly_adopted=98,
    LoGo_previously_adopted_reused=2, LoGo_constant_negative_ids=constant_ids,
    LoGo_new_fit=False, LoGo_new_CNN=False, LoGo_offserver_whole_predict_reexecuted=False,
    partial_fourmethod_native400=dict(status='ROOT_PARTIAL_FOUR_METHOD_NATIVE400_TABLE_ADOPTED',
        build=ref(paths['partial400_build']), table=ref(paths['partial400_table']), records=400, methods=4, view='native',
        independent_review=ref(paths['partial400_review']), root=ref(paths['partial400_root']),
        independent_check_complete=True, root_adopted=True, new_scientific_adoptions=0, final_17_method_table=False))
s['celeba_mechanism_v1'].update(queue_observation_utc=o['utc'], queue_observation_path=str(paths['health']),
    queue_observation_sha256=sha(paths['health']), queue_completed_observed=nmain, queue_active_observed=8,
    queue_pending_observed=345, queue_failures_observed=0, queue_failed_observed=0,
    native_observed_not_yet_accepted=nmain - 400)
s['NGA96_runtime_dispatch_20261011'].update(latest_readonly_observation=ref(paths['health']), terminal_result_directories_observed=nnga, latest_observed_rounds=o['NGA']['active_progress'])
s['latest_NGA96_readonly_observation'] = dict(**ref(paths['health']), utc=o['utc'], terminal_observed=nnga, accepted_new=19, active=o['NGA']['active_progress'], Huber_started=False)
s['hybrid100_fullcoverage_20261010'].update(latest_measured_observation=dict(utc=o['utc'], terminal_observed=nhybrid, source=ref(paths['health']), acceptance_unchanged_by_observation=True))
s['latest_current_resource_observation'] = dict(utc=o['utc'], source_ref=ref(paths['health']), main_terminal=nmain, main_active=8,
    main_pending=345, main_failed=0, NGA_terminal_observed=nnga, NGA_new_accepted=19, Hybrid_terminal_observed=nhybrid,
    Hybrid_native_new_accepted=29, GPU_utilization=gpu_util, GPU_C=gpu_temp, GPU_RecoveryAction=[x[4] for x in gpu_rows],
    memory_current_B=int(o['cgroup']['memory.current']), memory_max_B=int(o['cgroup']['memory.max']), disk_free_B=o['disk_free_B'],
    cgroup_CPU_quota_cores=quota_cores, cgroup_CPU_measured_cores=o['measured_cgroup_CPU_cores'], cgroup_OOM=0, not_new_scientific_acceptance=True)
s['latest_publication_verification'] = dict(**ref(paths['Git73']), commit=git['commit'], files=246, verified_utc=git['verified_utc'], later_CUDA407_pending_Git74=True)

# Exact historical-tail discovery from the existing CUDA211 writer, unchanged.
tree=ast.parse((R/'tmp/refresh_native370_FL86_Hybrid20_root_20261011.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='entries' for x in n.targets));ns={'R':R,'T':T};exec(compile(ast.Module(body=[node],type_ignores=[]),'<tails>','exec'),ns)
entries=[];prefix=None
for p,marker,h in ns['entries']:
 raw=p.read_bytes();at=raw.index(marker.encode());assert hashlib.sha256(raw[at:]).hexdigest()==h
 if prefix is None:prefix=raw[:at]
 assert prefix==raw[:at];entries.append((p,raw,raw[at:],h))

text = f'''# CURRENT: GuardFed返修实验

入口更新：{now}。当前服务器 ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded；先遵守/etc/vast-agents-guide.md。以下将实际采用与带时间的运行观测分开。

| 阶段 | 严格验收、F盘SHA与root采用 | 最新实测及边界 |
|---|---:|---|
| 机制训练 | 新400/800，U/C/A/F各100；Full100共享复用 | {o['utc']}：447终轮、8活动、345等待、失败0；其余4类继续，观测不增采用 |
| 统一CUDA首阶段500 | 本地去重采用105/500=Full100+旧5条控制 | {c['utc']}：服务器保存验收108/500（旧57+新51），392等待；失败0，不等同离机采用 |
| 统一CUDA原八方法800 | 本地去重采用102/800；不含Full/FL/LoGo | {b['utc']}：服务器277/800（原固定3+后续274），523等待；失败0，不等同离机采用 |
| 统一CUDA FLGMM100 | 全部100已采用；新97+原3，非103 | 原Linux完整验收；离机900指标/2400计数/300规则，native最大差0；旧F引用不重复运输 |
| 统一CUDA LoGo100 | 全部100已采用；新98+原2，非102 | 原冻结CPU后处理，1986700预测/300指标/800计数误差0；无新CNN/fit/refit；1条恒负结果保留 |
| 历史三视图 | 机制CPU380、FL CPU100、Hybrid CPU20 | 既有CPU或混合路径基线表和原620数值失败保留；新CUDA表独立产生 |
| 主比较表 | 历史11方法native1100，三页PDF；新四方法CUDA native400阶段表 | 新阶段表已独立算术检查并根采用；两表均非最终17方法表，本次表采用不新增科学记录 |
| 组合控制训练 | native29/96新；screen4另计 | {o['utc']}：37终轮，仅观察；历史三视图20 |
| Fed-NGA/Huber | NGA新19/96采用；screen4另计 | {o['utc']}：27终轮；已采用19条恒负预测保留；Huber96尚未启动 |

作者批准完整比较范围统一CUDA，包括Full。不重训、不改参数/容差、不访问test。原CNN clean-root及valid使用确定性FP32 CUDA前向；原冻结后处理实现与设备保留。协议：celeba_uniform_cuda_evaluation_v1/PROTOCOL.md。实际冻结1500条互不重复记录（500+800+FL100+LoGo100）；本地CUDA去重采用407=105+102+100+100，不是411。首500内原7与Full100重叠2条，因此首阶段为105，不能将7+100直接相加。Full100与方法表共享一次。完整目标2500仍待六方法/四控制及适配，不能称全部冻结或完成。

Full100根采用：{ref(paths['Full_root'])}。新98+旧2、原Linux whole saved gate与离机900指标/2400计数/300规则、609新归档成员SHA通过，native最大差0；不是完整500采用。FedAvg100与旧800固定3重叠一个FedAvg，因此该800范围唯一采用102。FL100根采用：{ref(paths['FL_root'])}，只运输97新NPZ并保留旧3引用。LoGo100根采用：{ref(paths['LoGo_root'])}，原Linux冻结后处理与离机零拟合保存预测算术检查通过；不是离机重新执行整个netcal预测。LoGoFair-DP使用20个固定image-ID虚拟人口、fitseed1719及原CPU netcal1.3.6，DP指demographic parity，不能称真实client公平性或差分隐私。IID Benign seed91001恒定负预测保留，零差异不能单独解释为公平成功。

LoGo712份原产物、FedAvg100 Linux index actual002和独立CPU环境的既有恢复链保留。首次环境安装错误引入torch2.14.1，失败未覆盖；修复后保持冻结直接包版本，实际Python3.10.20相对原3.10.4的差别仍披露，不声称全部环境等价。原fixed2采用后才启动98；本次100完整闭合不重新选择/拟合原后处理、不重复CNN。500旧/proc读取失败及800模块冲突/资源读取失败原件保留；原科学worker未改，无自动重试或追认诊断。

四方法部分表：tmp/celeba_uniform_cuda_fourmethod_partial_20261011/actual_partial400/TABLES.md，build receipt {pins['partial400_build']}，TABLES SHA{pins['partial400_table']}，独立检查{pins['partial400_review']}及根表采用{pins['partial400_root']}。它只包含FedAvg、GuardFed-AD2+、FLGMM、LoGoFair-DP各100条CUDA native记录，已生成并独立验收；不是三视图全比较或最终17方法表，不设新主端点，不新增科学采用。FedAvg与LoGo各1条恒定预测保留。

Git73已实际提交{git['commit']}，246个blob及remote分支核验通过，回执SHA{pins['Git73']}，截止仍211采用。Git72 pre-Git漏SUFFIX_FIX的拒绝和精确清单修复保留；旧回执不追改。本次Full100/LoGo100完整采用及407入口未发布Git74。模型、NPZ、数据和大归档不进Git。

{o['utc']} 资源快照：双5090利用率{gpu_util}%，温度{gpu_temp}°C，Recovery None；CPU实测{o['measured_cgroup_CPU_cores']:.2f}/{quota_cores:.5f}核，cgroup OOM0，RAM {int(o['cgroup']['memory.current'])/1e9:.2f}/{int(o['cgroup']['memory.max'])/1e9:.2f}GB，服务器磁盘{o['disk_free_B']/1e12:.4f}TB余。它与500/800快照时间不同，不作持续巡检。CPU低有效占用保留GPU瓶颈判断，不造重复负载。大文件仅留服务器或核验Yanan2TB后的F:/YananResearchStorage/GuardFed，不回退内置盘。三小时聊天automation仍PAUSED，无可调用更新工具，未创建替代监督。

完整24条英文回复：docs/server_deployment_20260923/revision_20260923/rebuttal_working_F80_FL100_Hybrid20_native11_20261011.md，固定380/F80、FL100、Hybrid20/native11截止，原SHA910b263不变；后续部分结果未偷偷写入旧稿。十一方法native1100历史表仍保留准确率代价、LoGo较低ASPD和FLTrust较高准确率，不称每评分项不可或缺或全面领先。

仍待六基线、四控制、完整统一CUDA比较、主端点/最终评价、FedWA/SmartFL/FedDNA忠实来源、匹配投稿LaTeX整合和最终发布。旧TableII480格有来源，294格n1的SD不可计算；FairGuard/IID/FedSA投稿59.13/历史54.13/同终轮候选53.94区分保留。当前valid，官方test有历史暴露，不能称未触碰test；Huber恒等投影仅CNN适配，不继承原理论保证。整个返修尚未完成。

本次STATE/五入口/reader/protocol旧bytes将在tmp/root_CUDA407_Full100_LoGo100_Git73_20261011备份；五历史尾逐SHA保持。

'''
reader = V / 'AUTHOR_REVIEW_START_HERE.md'; oldreader = reader.read_bytes(); r = oldreader.decode('utf8')
assert hashlib.sha256(oldreader).hexdigest() == '9adef3f0425b8e1c43b8d395c7b6b917f15fd074a396b01a79e92bdbe644edb6'
old_git = '新FL100完整CUDA采用、LoGo2与剩98启动及最新入口待下一增量。'
assert r.count(old_git) == 1
r = r.replace(old_git, 'FL100完整CUDA采用、LoGo2与剩98启动及09:19截止入口已在Git73发布（f1148225，246个blob/remote核SHA）；本次Full100与LoGo100完整采用、CUDA去重407及新入口尚未发布Git74。', 1)
start = r.index('| 图像机制消融 |'); end = r.index('\n| 新增基线与简单组合 |', start)
r = r[:start] + '''| 图像机制消融 | 训练U/C/A/F各100新结果已采用，共400/800；历史CPU三视图380（F80）及旧失败保留。统一CUDA去重采用407=首阶段105（Full100+旧5控制）+原八方法102+FLGMM100+LoGo100；Full100只共享一次，旧7与Full100重叠2条，不是411。Full100原Linux whole saved gate及离机900指标/2400计数/300规则、native差0通过；LoGo100使用已采用FedAvg CUDA缓存与原冻结CPU netcal后处理，无新fit/refit，1986700预测/300指标/800计数重算误差0，保留1条恒负结果。09:48:54服务器108/500、09:43:14服务器277/800仅为保存验收观测，不增加离机采用。[四方法CUDA native400阶段表](E:/OneDrive/文档/GuardFed/tmp/celeba_uniform_cuda_fourmethod_partial_20261011/actual_partial400/TABLES.md)已生成并独立验收、根采用，非最终17方法表且无新增科学记录。实际冻结1500条，完整2500目标尚未齐备；不重训、不改参数/容差、不访问test。 |''' + r[end:]
for link in re.findall(r'\[[^\]]+\]\((E:/[^)]+)\)', r):
    assert Path(link.split('#', 1)[0]).is_file(), link
newreader = r.encode('utf8')
protocol = T / 'celeba_uniform_cuda_evaluation_v1/PROTOCOL.md'; oldprotocol = protocol.read_bytes()
assert hashlib.sha256(oldprotocol).hexdigest() == '83004a75f62612994ec64c4d6953cb46e762746ae67f44b90381a72ef1da4154'
add = f'''\n\n## 2026-10-11 actual Full100 and LoGo100 closure (CUDA407)\n\nRecorded {now}. The author-approved uniform-CUDA root/valid scope and all original mathematical/device rules remain unchanged. Full100 root proof {ref(paths['Full_root'])} adopts100=prior2+new98; original Linux whole saved checks plus complementary offserver zero-fit audit cover900 metric values/2400 counts/300 rules, with native maximum difference0. First500 unique adoption is105=prior7+Full100−overlap2, including5 previously adopted control records; Full is shared once with the baseline comparison. It is not complete500.\n\nLoGo100 root proof {ref(paths['LoGo_root'])} adopts100=prior2+new98, with1986700 saved decisions/300 metrics/800 confusion counts and maximum error0. Frozen CPU netcal1.3.6 postprocessors, fitseed1719 and20 image-ID virtual cohorts are retained; DP means demographic parity. No new CNN, fit/refit, training or test. The single IID Benign seed91001 constant-negative result is retained. Offserver audit does not rerun the whole netcal predictor or establish cross-platform bitwise equivalence. Existing Python/environment limitations, original failures and CPU or mixed-path historical tables remain preserved.\n\nUnique root adoption407=first105+old800102+FL100100+LoGo100100, not411. Actual frozen scope remains1500; full target2500 is not yet completely frozen or accepted. Observed108/500 at{c['utc']} and277/800 at{b['utc']} are separate saved-result snapshots, not extra adoptions. Main447/NGA27/Hybrid37 at{o['utc']} likewise change no scientific acceptance. Native400/NGA19/Hybrid29 and the fixed24-comment draft SHA910b263 remain unchanged. Git73 {git['commit']} verified246 blobs at{git['verified_utc']}; it has the earlier211 cutoff. Git74 is not published.\n\nFour-method native400 table build {ref(paths['partial400_build'])} and TABLES {ref(paths['partial400_table'])} were independently checked ({ref(paths['partial400_review'])}) and root adopted ({ref(paths['partial400_root'])}). These are partial four-method native validation results, not final17 or a complete three-view comparison; table adoption adds no new scientific records. No new primary endpoint or scientific gate is created by this metadata writer.\n'''
newprotocol = oldprotocol + add.encode('utf8')
u['protocol'] = ref_bytes(protocol, newprotocol)
s['author_review_reading_entry_20261011'].update(utc=now, reader=ref_bytes(reader, newreader), uniform_CUDA_root_adopted=407)
s['current_entry_writer'] = dict(**ref(__file__), utc=now)
s['local_work_verified_utc'] = now; s['updated_unix'] = datetime.now(timezone.utc).timestamp()
assert s['celeba_mechanism_v1']['scientific_results_strictly_accepted'] == 400
assert s['celeba_mechanism_v1']['three_view_new_models_accepted'] == 380
assert s['NGA96_runtime_dispatch_20261011']['scientific_accepted'] == 19 and s['hybrid100_fullcoverage_20261010']['new_accepted'] == 29
assert s['health_monitor'] == prior_s['health_monitor'] and s['native_monitor_20261009'] == prior_s['native_monitor_20261009']
assert s['uniform_CUDA_evaluation_v1']['complete_target_inventory_frozen'] is False
newstate = (json.dumps(s, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf8')
newprefix = text.encode('utf8')
outputs = {sp: newstate, reader: newreader, protocol: newprotocol}
for p, raw, tail, h in entries:
    outputs[p] = newprefix + tail
    assert hashlib.sha256(outputs[p][len(newprefix):]).hexdigest() == h
assert len(entries) == 5 and len(outputs) == 8
assert newprotocol.startswith(oldprotocol)
proof = dict(status='ROOT_CUDA407_FULL100_LOGO100_GIT73_CURRENT_ENTRIES_RECORDED', utc=now, source=ref(__file__),
    parent_writer=ref(R / 'tmp/record_CUDA211_FL100_LoGo2_Git72_root_20261011.py'), state_before_sha256=hashlib.sha256(before).hexdigest(),
    state_after=ref_bytes(sp, newstate), reader=ref_bytes(reader, newreader), protocol=ref_bytes(protocol, newprotocol),
    five_entries=[dict(**ref_bytes(p, outputs[p]), historical_suffix_sha256=h, historical_suffix_bytes=len(tail)) for p, raw, tail, h in entries],
    history_tails_exact=True, all_preconditions_before_writes=True, evidence={k:ref(p) for k,p in paths.items()},
    root_adopted_CUDA=407, first500_unique_accepted=105, first500_Full100_overlap_prior7=2, first500_other_controls=5,
    old800_unique_accepted=102, FL100_accepted=100, LoGo100_accepted=100, Full100_accepted=100,
    remote500_saved_accepted=n500, remote800_saved_accepted=n800, remote_observations_do_not_adopt=True,
    partial400_status='ROOT_PARTIAL_FOUR_METHOD_NATIVE400_TABLE_ADOPTED', LoGo_constant_negative_ids=constant_ids,
    new_scientific_adoptions_in_metadata_writer=0, whole_comparison_complete=False, whole_rebuttal_complete=False, final_test=False,
    Git74_published=False, frozen_scope=1500, complete_target_inventory_frozen=False)
proof_bytes = (json.dumps(proof, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf8')
# Final stale-input guards: no output/backup directory or file has been written yet.
assert not D.exists() and sp.read_bytes() == before and reader.read_bytes() == oldreader and protocol.read_bytes() == oldprotocol
for k, p in paths.items():
    assert sha(p) == pins[k], k
for p, raw, tail, h in entries:
    assert p.read_bytes() == raw

# Mutation boundary. Root runs this once after source review; all checks above.
D.mkdir()
(D / 'PRIOR_STATE.json').write_bytes(before)
(D / 'PRIOR_READER.md').write_bytes(oldreader)
(D / 'PRIOR_PREFIX.md').write_bytes(prefix)
(D / 'PRIOR_PROTOCOL.md').write_bytes(oldprotocol)
for i, (p, raw, tail, h) in enumerate(entries):
    (D / ('PRIOR_ENTRY_%02d.md' % i)).write_bytes(raw)
for p, body in outputs.items():
    p.write_bytes(body)
proof_path = D / 'ROOT_METADATA_ADOPTION.json'
proof_path.write_bytes(proof_bytes)
print(json.dumps(dict(state=ref(sp), metadata=ref(proof_path)), ensure_ascii=False))
