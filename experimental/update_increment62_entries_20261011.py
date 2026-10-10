"""Refresh only current pointers from already adopted evidence; preserve history bytes."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, re

R = Path(__file__).resolve().parents[1]
T = R / 'docs/server_deployment_20260923/training_20260923'
H = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
a = argparse.ArgumentParser()
a.add_argument('--addendum-proof', required=True)
a.add_argument('--addendum-proof-sha256', required=True)
args = a.parse_args()
sp = T / 'TRAINING_STATE.json'
prior = read(R/'tmp/native333_increment61_pointer_refresh_20261011.json')
assert H(sp) == prior['STATE_sha256'] == 'b8e30bfd0154c9eb1f410c853e37d44f4be8e5f233146939de189a46a5d838b8'
s = read(sp)
vr = R/'tmp/celeba_mechanism_remaining_F_FedSA10_root_adoption_20261011/ROOT_ADOPTION.json'
fr = T/'celeba_mechanism_v1/three_view_F_IID_three_scenes30_20261011/ROOT_VERIFICATION.json'
hr = R/'outputs/guardfed_tables/celeba_hybrid_three_view_IID_Benign10_20261011/ROOT_VERIFICATION.json'
assert H(vr) == '62475bb89325d0b229977629bb879d99994c4d313cb7924db9a5efbce6c9ec25'
assert H(fr) == 'ea4a3ea8caaa3c05a3bbd857eb2ae660785b6c9177a1b970f0f7ebf7cc3937d5'
assert H(hr) == '7f64be4721f1eaeeb42881c4633fa03de5f1ae19e32fc01842de4da0867a0b32'
v, f, h = read(vr), read(fr), read(hr)
idx = R/v['records_index_path']; ix = read(idx)
assert H(idx) == v['records_index_sha256'] and len(ix['all_ids']) == 330
assert v['cumulative_accepted'] == 330 and f['paired_models'] == 30 and h['unique_checkpoints'] == 10
ap = R/args.addendum_proof
assert H(ap) == args.addendum_proof_sha256
ad = read(ap)
addendum = R/'docs/server_deployment_20260923/revision_20260923/rebuttal_F30_addendum_20261011.md'
assert ad['status'] == 'PASS_LIMITED_EDITORIAL_NUMERIC_LINK_AND_SCOPE_VERIFICATION_AFTER_ONE_TERMINOLOGY_FIX'
assert ad['after_sha256'] == H(addendum) and ad['displayed_numeric_scalars_checked'] == 30
growth = T/'server_reactivation_20261009/ROOT_FIVE_QUEUE_GROWTH_20261010T2144.json'
g = read(growth); raw = R/g['raw_path']
assert H(growth) == '21601909a64e9c1681765b45d6f34fe02d72a635f8d367e5736b407f4b5e917c'
assert H(raw) == g['raw_sha256'] and (g['main_completed'],g['FLGMM_terminal'],g['gradient_terminal'],g['Hybrid_terminal']) == (336,73,59,17)
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
m = s['celeba_mechanism_v1']
assert (m['scientific_results_offserver_verified'],m['three_view_new_models_offserver_verified']) == (333,320)
for key in ['three_view_new_models_accepted','three_view_new_models_offserver_verified']: m[key] = 330
m['three_view_accepted_ids'] = ix['all_ids']
m['three_view_counts_by_variant'] = dict(minus_U=100,minus_C=100,minus_A=100,minus_F=30)
m['three_view_root_proof_sha256'] = H(vr)
m['latest_three_view_index'] = idx.relative_to(R).as_posix()
m['latest_three_view_index_sha256'] = H(idx)
m['three_view_scope_limit'] = 'U/C/A each100; F30 covers IID Benign/F Flip/FedSA, fixed10/9/6 panels and all paired reversals. Seven F scenes and four other variants pending; no universal necessity, significance, final-test or whole-rebuttal claim.'
m['F_three_view_three_scene_table'] = dict(root_proof_path=fr.relative_to(R).as_posix(),root_proof_sha256=H(fr),table_path=fr.with_name('TABLES.md').relative_to(R).as_posix(),paired_models=30,complete_scenes=3,displayed_records=60,seed_panels=[10,9,6],mean_SD_scalars=486,display_cells=243,validation_only=True,final_test=False,whole_rebuttal_complete=False)
m.update(queue_completed_observed=336,queue_active_observed=8,queue_pending_observed=456,queue_failures_observed=0,new_started=344,queue_observation_utc=g['utc'],queue_observation_path=g['raw_path'],new_started_observation_utc=g['utc'])
rem = s['mechanism_remaining620_valid_20261010']
assert rem['new_offserver_accepted'] == 140 and set(v['accepted_new_ids']).isdisjoint(rem['accepted_ids'])
rem['new_offserver_accepted'] = 150
rem['accepted_ids'] += v['accepted_new_ids']
rem['latest_accepted_new_ids'] = v['accepted_new_ids']
rem['root_adoption_path'] = vr.relative_to(R).as_posix(); rem['root_adoption_sha256'] = H(vr)
rem['F_complete_scenes'] = v['complete_F_scenes']; rem['F_partial_scenes'] = []; rem['F_partial_seed_ids'] = []
rem['F_IID_three_scene_table_adopted'] = True
hy = s['Hybrid_IID_Benign_three_view10_20261011']
hy['IID_Benign_table'] = dict(root_proof_path=hr.relative_to(R).as_posix(),root_proof_sha256=H(hr),table_path=hr.with_name('TABLES.md').relative_to(R).as_posix(),unique_checkpoints=10,complete_scenes=1,seed_panels=[10,9,6],mean_SD_scalars=54,validation_only=True,final_test=False)
obs = dict(path=g['raw_path'],sha256=g['raw_sha256'],utc=g['utc'],main_completed=336,main_active=8,main_pending=456,failed=0,FLGMM_new_terminal=73,gradient_terminal=59,Hybrid_new_terminal=17,remaining620_remote_closed=156,counts_are_observation_only=True,new_acceptance=0,source_hash_scope='Small frozen source/data pins unchanged; large images historical full SHA plus stat only',scope='Full five-queue snapshot and identity/growth checks; FL and Hybrid terminal counts are new jobs, with four screen reuses separate.',growth_path=growth.relative_to(R).as_posix(),growth_sha256=H(growth))
s['latest_five_queue_readonly_observation'] = obs
s['latest_main_queue_readonly_observation'] = dict(path=g['raw_path'],sha256=g['raw_sha256'],utc=g['utc'],completed=336,active=8,pending=456,failed=0,observation_only=True,science_acceptance_created=0)
for key, terminal in [('gradient64_validation_search_20261010',59),('flgmm_fullcoverage_v2_20261009',73),('hybrid100_fullcoverage_20261010',17),('mechanism_remaining620_valid_20261010',156)]:
    s[key]['latest_measured_observation'] = dict(checked_utc=g['utc'],terminal_observed=terminal,snapshot_path=g['raw_path'],snapshot_sha256=g['raw_sha256'],acceptance_unchanged_by_observation=True)
s['latest_rebuttal_evidence_addendum'] = dict(path=addendum.relative_to(R).as_posix(),sha256=H(addendum),editorial_proof_path=ap.relative_to(R).as_posix(),editorial_proof_sha256=H(ap),F_paired_models=30,F_complete_scenes=3,Hybrid_table_checkpoints=10,author_review_only=True,original24_comment_draft_unchanged=True,submitted_manuscript_edited=False,final_test=False,whole_rebuttal_complete=False)
proof = dict(status='CURRENT_ENTRIES_ADOPTED_NATIVE333_VIEWS330_F30_HYBRID10_TABLE',utc=now,source_sha256=H(__file__),entries={},source_science_already_adopted=True,new_science_created=0,new_training=0,new_inference=0,final_test=False)
markers = {'RUNNING.md':b'# HISTORICAL:','REBUTTAL_COMPLETION_20261009.md':b'## Historical accepted increment','返修实验总览.md':'以下为 2026-10-04'.encode(),'MONITOR_HANDOFF.md':b'# Historical handoff snapshots','EXECUTION.md':b'# HISTORICAL PREPARATION SNAPSHOT'}
prefix_same = None
pending = []
for rel, pin in prior['entries'].items():
    p = R/rel; assert H(p) == pin['sha256']
    b = p.read_bytes(); i = b.index(markers[p.name]); history = b[i:]; prefix = b[:i].decode()
    assert hashlib.sha256(history).hexdigest() == pin['history_sha256']
    prefix,n = re.subn(r'本段更新时间：[^。]+。','本段更新时间：'+now+'。',prefix,count=1); assert n == 1
    replacements = {
      '21:26:25：完成331、活动8、等待461、失败0；随后验收333':'21:44:05：完成336、活动8、等待456、失败0；活动第9–27轮',
      '320终轮checkpoint：U100/C100/A100/F20':'330终轮checkpoint：U100/C100/A100/F30',
      '20:39:34：remaining620远端闭合147，与离机接受140单列':'21:44:05：remaining620远端闭合156，与离机接受150单列',
      '20:39:34：终轮71；三视图接受71':'21:44:05：新增终轮73，另复用4screen；三视图接受71',
      '20:39:34：终轮55；未完成全部搜索或选择recipe':'21:44:05：终轮59/64；未完成全部搜索或选择recipe',
      '20:39:34：终轮16；IID Benign十seed native及三视图结果已接纳':'21:44:05：新增终轮17，另复用4screen；IID Benign三视图十seed表已接纳',
      '21:15:50资源实测：双GPU利用率100/99%，温度64/61°C，RecoveryAction None；cgroup内存77.46/519.17GB、OOM0，磁盘余1.060TB。源码/数据身份未变，当前五队列服务及worker身份已核：tmp/hybrid_screen91001_single_replay_prepared_20261011/runtime/FRESH_OBSERVATION_REF.json；此前完整五队列增长记录保留。':'21:44:05资源实测：双GPU利用率100/100%，温度64/62°C，RecoveryAction None；cgroup内存77.30/519.17GB、OOM0，磁盘余1.060TB。五队列服务、worker身份与相对增长通过；small source/data hash未变，大图像只核stat及历史全量SHA。增长记录：docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/ROOT_FIVE_QUEUE_GROWTH_20261010T2144.json。两采样间整个cgroup平均使用13.69/122.88 CPU核，不能称GuardFed专属效率或持续占用。',
      '三视图仍320，未把新native当校准完成或生成新F表。该批root证据654ce595在Git61截止后形成，下一增量发布待纳入':'其中FedSA10三视图已增量接纳，累计330；另外三项S-DFA native仍不计作三视图完成。native333及views330均在Git61截止后形成，下一增量发布待纳入',
      '## 新补齐的F消融表':'## 新补齐的三个场景F消融表',
      'three_view_F_IID_two_scenes20_20261011/TABLES.md':'three_view_F_IID_three_scenes30_20261011/TABLES.md',
      'IID Benign/F Flip两完整场景，每场景10共享seed；raw/native/shared三视图、固定10/9/6面板。40记录/20配对，独立核算324均值SD标量、162展示格、360计数派生指标、960基础计数；旧Benign20对象、162标量及81展示值原样保留。没有新增推理、拟合或训练。':'IID Benign/F Flip/FedSA三个完整场景，每场景10共享seed；raw/native/shared三视图、固定10/9/6面板。60记录/30配对，独立核算486均值SD标量、243展示格、540计数派生指标、1440基础计数；旧40对象、324标量及162展示值原样保留。表格制作没有新增推理、拟合或训练。',
      '仅两个IID场景，其他八个F场景和其余控制未齐。':'现为三个IID场景，其他七个F场景和四类其他控制未齐。FedSA十seed去F较Full的配对均值为ACC+0.136pp、AEOD+0.00322、ASPD+0.00393；六seed native/shared的ASPD差值反转为−0.00560。全部结果保留。',
      '仅这一场景齐备，其他场景和完整100覆盖仍在继续。':'仅这一场景齐备，其他场景和完整100覆盖仍在继续。三视图表：outputs/guardfed_tables/celeba_hybrid_three_view_IID_Benign10_20261011/TABLES.md。十seed native ACC88.027%、AEOD0.02920、ASPD0.09251；shared为87.708%、0.02321、0.03383，存在准确率代价。Hybrid为项目控制，不称外部原方法。',
      '三份作者候选已对齐A100及F20，原话与顺序保留，F20取舍和固定面板反转已写入；未应用提交版正文，不代表返修完成。':'三份作者候选已对齐A100及F20，原话与顺序保留；单独新增F30/Hybrid证据补充稿：docs/server_deployment_20260923/revision_20260923/rebuttal_F30_addendum_20261011.md。未把原24条旧截止稿暗改称F30，未应用提交版正文，不代表返修完成。',
      '入口写入器：tmp/update_increment61_entries_20261011.py':'入口写入器：tmp/update_increment62_entries_20261011.py',
    }
    for old, new in replacements.items():
        assert prefix.count(old) == 1,(rel,old)
        prefix = prefix.replace(old,new)
    if prefix_same is None: prefix_same = prefix
    else: assert prefix == prefix_same
    pending.append((p, prefix.encode()+history, history, pin))
for p, content, history, pin in pending:
    p.write_bytes(content)
    assert p.read_bytes().endswith(history)
    proof['entries'][p.relative_to(R).as_posix()] = dict(sha256=H(p),history_sha256=pin['history_sha256'],history_exact=True)
s['current_entry_writer'] = dict(path=Path(__file__).relative_to(R).as_posix(),sha256=H(__file__),utc=now)
s['updated_unix'] = datetime.datetime.now(datetime.timezone.utc).timestamp()
temp = sp.with_suffix('.increment62.tmp'); assert not temp.exists()
temp.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf8'); os.replace(temp,sp)
proof['STATE_sha256'] = H(sp)
out = R/'tmp/increment62_entry_update_proof_20261011.json'; assert not out.exists()
out.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(STATE_sha256=H(sp),proof_sha256=H(out),entries=len(proof['entries']))))
