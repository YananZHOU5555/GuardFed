"""Update current pointers from actual live and launch records; preserve histories."""
from pathlib import Path
import datetime, hashlib, json, os, re

R = Path(__file__).resolve().parents[1]
T = R / 'docs/server_deployment_20260923/training_20260923'
C = R / 'tmp/fl_FFlip10_capacity_pool32_20261011'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
sp = T / 'TRAINING_STATE.json'
assert sha(sp) == '36b8b5c9dd7d3f20a1566e0f527f2ce0de9c31d400ecc09c312a58d862a32edb'
s = read(sp)
hp = T / 'server_reactivation_20261009/ROOT_FIVE_QUEUE_GROWTH_20261010T2039.json'
assert sha(hp) == 'ade8ba6808a7536bf788301fd716b8dbb11cc0a95c57ec094ec1e431e30fd1e2'
h = read(hp)
assert h['status'] == 'PASS_LIVE_FIVE_QUEUE_GROWTH_NO_NEW_SCIENTIFIC_ACCEPTANCE'
pre = read(C / 'runtime/LINUX_PREFLIGHT.json')
start = read(C / 'runtime/START_RECEIPT.json')
assert pre['resources_eligible'] and pre['cpu_affinity'] == list(range(32, 64))
assert start['status'] == 'ROOT_FLGMM_FINITE10_SUPERVISOR_STARTED_NOT_SCIENTIFIC_ACCEPTANCE'
assert start['new_scientific_acceptances'] == 0
assert start['authorization_sha256'] == sha(C / 'runtime/AUTHORIZATION.json')
assert start['linux_preflight_sha256'] == sha(C / 'runtime/LINUX_PREFLIGHT.json')
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
s['latest_five_queue_readonly_observation'] = dict(
    path=h['raw_path'], sha256=h['raw_sha256'], utc=h['utc'],
    growth_proof_path=hp.relative_to(R).as_posix(), growth_proof_sha256=sha(hp),
    main_completed=327, main_active=8, main_pending=465, failed=0,
    FLGMM_terminal=71, gradient_terminal=55, Hybrid_terminal=16,
    remaining_remote_closed=147, new_acceptance=0, counts_are_observation_only=True)
s['FLGMM_FFlip10_resource_rebind_20261011']['latest_pool_launch'] = dict(
    status=start['status'], utc=start['utc'], source_root=(C/'ROOT_SOURCE_REVIEW.json').relative_to(R).as_posix(),
    source_root_sha256=sha(C/'ROOT_SOURCE_REVIEW.json'),
    preflight=(C/'runtime/LINUX_PREFLIGHT.json').relative_to(R).as_posix(), preflight_sha256=sha(C/'runtime/LINUX_PREFLIGHT.json'),
    start_receipt=(C/'runtime/START_RECEIPT.json').relative_to(R).as_posix(), start_receipt_sha256=sha(C/'runtime/START_RECEIPT.json'),
    CPU_affinity=list(range(32,64)), torch_threads=8, scientific_acceptances_added=0,
    requires_original_Linux_whole_and_Windows_saved_only_audit=True, prior_five_refusals_preserved=True, test=False)
s['Huber_constant_prediction_review_20261011'] = dict(
    report='tmp/huber_constant_prediction_review_20261011/REVIEW.md',
    report_sha256=sha(R/'tmp/huber_constant_prediction_review_20261011/REVIEW.md'),
    accepted_candidates=14, seed=91001, eta=.03, constant_negative=True,
    failure_cause_proven=False, new_training=0, final_test=False)
s['Hybrid_private_identity_bridge_20261011'] = dict(
    metadata_records=12, check='tmp/celeba_hybrid_three_view_bridge_20261011/CHECK.json',
    check_sha256=sha(R/'tmp/celeba_hybrid_three_view_bridge_20261011/CHECK.json'),
    reused_existing_seed91002='tmp/celeba_hybrid_three_view_canary1_prepared_20261011/CHECK.json',
    reuse_check_sha256=sha(R/'tmp/celeba_hybrid_three_view_canary1_prepared_20261011/CHECK.json'),
    new_scientific_acceptances=0, missing_Benign_views=9, whole100_complete=False)
files = [(T/'RUNNING.md',b'# HISTORICAL:'),
    (T/'REBUTTAL_COMPLETION_20261009.md',b'## Historical accepted increment'),
    (R/'docs/返修实验总览.md','以下为 2026-10-04'.encode()),
    (T/'server_reactivation_20261009/MONITOR_HANDOFF.md',b'# Historical handoff snapshots'),
    (T/'celeba_mechanism_v1/EXECUTION.md',b'# HISTORICAL PREPARATION SNAPSHOT')]
proof = dict(status='CURRENT_ENTRIES_UPDATED_ACTUAL_POOL_LAUNCH_F20_AUTHOR_DRAFTS',utc=now,
    source_sha256=sha(__file__),entries={},new_scientific_acceptances=0,test=False)
for p,marker in files:
    b=p.read_bytes(); i=b.index(marker); prefix=b[:i].decode(); history=b[i:]
    prefix,n=re.subn(r'本段更新时间：[^。]+。','本段更新时间：'+now+'。',prefix,count=1); assert n==1
    prefix=prefix.replace('2026-10-10T19:58:26.072343+00:00','2026-10-10T20:39:34.249713+00:00')
    replacements={
        '完成320、活动8、等待472、失败0，活动第17–46轮':'完成327、活动8、等待465、失败0，活动第8–69轮',
        'remaining620远端闭合140，与离机范围单列':'remaining620远端闭合147，与离机接受140单列',
        '终轮69；三视图仍61':'终轮71；三视图仍61',
        '终轮53；未完成全部搜索或选择recipe':'终轮55；未完成全部搜索或选择recipe',
        '终轮15；IID Benign十seed native表已接纳':'终轮16；IID Benign十seed native表已接纳',
        '温度68/64°C':'温度65/63°C',
        '78.71/519.17GB':'77.41/519.17GB',
        'ROOT_FIVE_QUEUE_GROWTH_20261010T1958.json':'ROOT_FIVE_QUEUE_GROWTH_20261010T2039.json'}
    for old,new in replacements.items():
        assert prefix.count(old)==1,(p,old);prefix=prefix.replace(old,new)
    paragraphs=prefix.split('\n\n')
    idx=[k for k,v in enumerate(paragraphs) if v.startswith('FL新十项评价原CPU120–127')]
    assert len(idx)==1
    paragraphs[idx[0]]=(
        'FL exact10评价保留五次未启动的资源拒绝记录：原CPU120–127两次、CPU136–143宽mask tick门一次、'
        '修正容量门后CPU136–143及CPU11–18各一次。后两次实测出现系统层CPU忙碌，不能定位具体所有者；'
        '容器cgroup使用约12核、配额122.87999核。固定八核全空闲要求会随共享宿主负载迁移而过期。'
        '本次只调整运行调度：8 Torch线程在32–63的32核池内调度，单进程、FP32、方法/seed/recipe及native1e-12容差不变；'
        '原科学17函数、Linux whole及Windows零fit saved审计保持。'
        f"实际fresh容量门及supervisor启动已通过（{start['utc']}），证据为tmp/fl_FFlip10_capacity_pool32_20261011/runtime/START_RECEIPT.json。"
        '这是已启动事实，不是新科学结果；FL三视图仍接受61。完成后须原whole检查、F盘离机SHA及root采用，不能直接据RUNNING补表。'
        '采样余量不保证持续独占或跨CPU数值等价，不重启健康训练、不改其他服务。')
    p.write_bytes('\n\n'.join(paragraphs).encode()+history)
    assert p.read_bytes().endswith(history)
    proof['entries'][p.relative_to(R).as_posix()]=dict(sha256=sha(p),history_sha256=hashlib.sha256(history).hexdigest(),history_exact=True)
s['current_entry_writer'] = dict(path=Path(__file__).relative_to(R).as_posix(),sha256=sha(__file__),utc=now)
s['updated_unix']=datetime.datetime.now(datetime.timezone.utc).timestamp()
tmp=sp.with_suffix('.increment60.tmp');assert not tmp.exists()
tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(tmp,sp)
proof['STATE_sha256']=sha(sp)
out=R/'tmp/publication_increment60_prepared_20261011/ENTRY_UPDATE_ACTUAL.json'
with out.open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(dict(status=proof['status'],STATE_sha256=sha(sp),proof_sha256=sha(out))))
