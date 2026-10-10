"""Refresh current entry pointers from adopted evidence; preserve history bytes."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, re

R = Path(__file__).resolve().parents[1]
T = R / 'docs/server_deployment_20260923/training_20260923'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
a = argparse.ArgumentParser()
a.add_argument('--hybrid-root', required=True)
a.add_argument('--hybrid-root-sha256', required=True)
args = a.parse_args()
sp = T / 'TRAINING_STATE.json'
assert sha(sp) == 'bd33f2b21e51436ecb9e36a8a31d0b74781a1d8efa45ee0074b7837a3d773ad4'
s = read(sp)
fl = R / 'tmp/fl_FFlip10_capacity_pool32_20261011/ROOT_SCIENTIFIC_ADOPTION.json'
ft = R / 'outputs/guardfed_tables/celeba_flgmm_seven_scenes70_20261011/ROOT_VERIFICATION.json'
hy = R / args.hybrid_root
assert sha(fl) == 'fe400060961fb923cbac79443f735fab6824b06689d45fb3bbf8d56307c95421'
assert sha(ft) == 'bcf0b16a6838114444080f95b6bd282b16f79f0052bee0598c7c0728e54fc8d8'
assert sha(hy) == args.hybrid_root_sha256
assert read(fl)['root_adoption'] and read(ft)['complete_scene_records'] == 70
assert read(hy)['root_adoption']
hp = R / 'tmp/hybrid_screen91001_single_replay_prepared_20261011/runtime/FRESH_OBSERVATION_REF.json'
hr = read(hp); h = read(hr['path'])
assert sha(hr['path']) == hr['sha256'] == 'a9cfbbf8527380fa15c120314d2a9e28dd22db0555776b9ae42affe937ec4b06'
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
s['FLGMM_after61_valid_three_view_20261011'] = dict(
    root_proof_path=fl.relative_to(R).as_posix(), root_proof_sha256=sha(fl),
    new_three_view_records_accepted=10, FLGMM_total_three_view_records=71,
    original_Windows47_failure_preserved=True, Windows_saved_output_fit_calls=0,
    cross_platform_bitwise_recalibration_claimed=False, final_test=False,
    seven_scene_table=dict(root_proof_path=ft.relative_to(R).as_posix(), root_proof_sha256=sha(ft),
        table_path='outputs/guardfed_tables/celeba_flgmm_seven_scenes70_20261011/TABLES.md',
        complete_scene_records=70, retained_partial_records=1, scenes=7,
        views=['raw','native','shared_calibration'], fixed_seed_panels=[10,9,6],
        validation_only=True, final_test=False))
s['Hybrid_IID_Benign_three_view10_20261011'] = dict(
    root_proof_path=hy.relative_to(R).as_posix(), root_proof_sha256=sha(hy),
    new_three_view_records_accepted=9, total_three_view_records=10,
    prior_seed91002_reused=1, screen_seed91001_phase_preserved=True,
    original_Windows8_import_failure_preserved=True, Linux_cached_root_refits=9,
    Windows_saved_output_fit_calls=0, cross_platform_bitwise_recalibration_claimed=False,
    validation_only=True, final_test=False)
s['latest_five_queue_readonly_observation'] = dict(
    path=hr['path'], sha256=hr['sha256'], utc=h['utc'],
    main_completed=328, main_active=8, main_pending=464, failed=0,
    counts_are_observation_only=True, new_acceptance=0,
    source_hash_scope='18 current small source/model/data pins; large images prior full hash plus exact stat only')
pub = T / 'publication_closed_increment60_verified_20261011.json'; pj = read(pub)
assert pj['status'] == 'COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS'
assert pj['commit'] == '798ce6ca670a1ca8cd573c6bfaf4850d46f6d258'
s['latest_publication_verification'] = dict(
    commit=pj['commit'], branch=pj['branch'], verified_utc=pj['verified_utc'],
    committed_blobs_sha256_verified=pj['committed_blobs_sha256_verified'],
    acceptance_cutoff=pj['acceptance_cutoff'], proof_path=pub.name, proof_sha256=sha(pub),
    earlier_publication_snapshot_is_historical=True)
files = [(T/'RUNNING.md',b'# HISTORICAL:'),
    (T/'REBUTTAL_COMPLETION_20261009.md',b'## Historical accepted increment'),
    (R/'docs/返修实验总览.md','以下为 2026-10-04'.encode()),
    (T/'server_reactivation_20261009/MONITOR_HANDOFF.md',b'# Historical handoff snapshots'),
    (T/'celeba_mechanism_v1/EXECUTION.md',b'# HISTORICAL PREPARATION SNAPSHOT')]
proof = dict(status='CURRENT_ENTRIES_ACTUAL_FL71_TABLE70_HYBRID10',utc=now,
    source_sha256=sha(__file__),entries={},final_test=False,whole_revision_complete=False)
prefix_text = None
for p,marker in files:
    b=p.read_bytes(); i=b.index(marker); prefix=b[:i].decode(); history=b[i:]
    prefix,n=re.subn(r'本段更新时间：[^。]+。','本段更新时间：'+now+'。',prefix,count=1); assert n==1
    replacements={
        '2026-10-10T20:39:34.249713+00:00':'2026-10-10T21:15:50.604577+00:00',
        '完成327、活动8、等待465、失败0，活动第8–69轮':'完成328、活动8、等待464、失败0，活动第44–65轮',
        '终轮71；三视图仍61，六场景表60，未把新native计作已评价':'终轮71；三视图接受71，七完整场景表70，另保留一个未齐场景记录',
        '终轮16；IID Benign十seed native表已接纳':'终轮16；IID Benign十seed native及三视图结果已接纳',
        '该实测双GPU均100%，温度65/63°C':'该实测双GPU利用率100/99%，温度64/61°C',
        '77.41/519.17GB':'77.46/519.17GB',
        '磁盘余1.060TB':'磁盘余1.060TB',
        '五队列真实终轮集合增长通过：docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/ROOT_FIVE_QUEUE_GROWTH_20261010T2039.json':'当前五队列服务及worker身份已核：tmp/hybrid_screen91001_single_replay_prepared_20261011/runtime/FRESH_OBSERVATION_REF.json；此前完整五队列增长记录保留',
        '清晰候选已在Git27ebed9，后新增详细稿及本地采用记录尚未算已推送':'清晰候选、详细稿及正文插入候选均已在Git798ce6；本次新FL/Hybrid证据待下一增量推送',
        '完成FL exact10评价后再补其七场景表':'FL exact10及七场景表已闭合，继续其剩余覆盖',
        '最近已验证推送27ebed940ce822f87976e0d37b90576e9a43f36a':'最近已验证推送798ce6ca670a1ca8cd573c6bfaf4850d46f6d258',
        'publication_closed_increment59_verified_20261011.json':'publication_closed_increment60_verified_20261011.json',
        '314提交文件哈希及远端分支通过':'120提交文件哈希及远端分支通过',
        '入口写入器：tmp/update_increment60_entries_20261011.py':'入口写入器：tmp/update_increment61_entries_20261011.py'}
    for old,new in replacements.items():
        assert prefix.count(old)==1,(p,old);prefix=prefix.replace(old,new)
    paras=prefix.split('\n\n'); ids=[k for k,v in enumerate(paras) if v.startswith('FL exact10评价保留五次')]; assert len(ids)==1
    paras[ids[0]]=('FL新增10记录已通过原生指标复核、Linux原校准检查、F盘归档及成员SHA、Windows零拟合计数审计，并由root采用；旧61记录原样保留。七完整场景表按同一10/9/6种子面板展示，另保留一个未齐场景记录。non-IID F Flip十seed native为ACC89.28±0.98%、AEOD0.0525±0.0077、ASPD0.1185±0.0068；共享校准为88.90±1.04%、0.0091±0.0073、0.0713±0.0111，存在准确率代价。旧Windows重拟合失败、五次资源拒绝和采样审计浮点差异均保留，不宣称跨平台逐位重校准一致。表：outputs/guardfed_tables/celeba_flgmm_seven_scenes70_20261011/TABLES.md。\n\n'
        'Hybrid IID Benign三视图现已补齐十共享seed：八个既有覆盖checkpoint、一个原screen seed91001及一个已验收seed91002显式复用。原screen身份及选择史保持，不改名为正式训练；同一checkpoint全部指标。新九项通过Linux原校准检查、F盘归档及Windows零拟合审计，原Windows首次Linux resource导入失败及采样审计微小差异保留。仅这一场景齐备，其他场景和完整100覆盖仍在继续。')
    newprefix='\n\n'.join(paras)
    if prefix_text is None: prefix_text=newprefix
    else: assert prefix_text==newprefix
    p.write_bytes(newprefix.encode()+history);assert p.read_bytes().endswith(history)
    proof['entries'][p.relative_to(R).as_posix()]=dict(sha256=sha(p),history_sha256=hashlib.sha256(history).hexdigest(),history_exact=True)
s['current_entry_writer']=dict(path=Path(__file__).relative_to(R).as_posix(),sha256=sha(__file__),utc=now)
s['updated_unix']=datetime.datetime.now(datetime.timezone.utc).timestamp()
tmp=sp.with_suffix('.increment61.tmp');assert not tmp.exists()
tmp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(tmp,sp)
proof['STATE_sha256']=sha(sp)
out=R/'tmp/increment61_entry_update_proof_20261011.json';assert not out.exists()
out.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'status':proof['status'],'STATE_sha256':sha(sp),'proof_sha256':sha(out)}))
