"""Update current prose only; preserve both historical suffixes byte for byte."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'docs/server_deployment_20260923/training_20260923'
items = [
    (ROOT / 'docs/返修实验总览.md', '以下为 2026-10-04', '08b94e001b4ff78474706db0a63cc2c1b626fe5664c01edfafbaefb53fe9b029'),
    (TRAIN / 'server_reactivation_20261009/MONITOR_HANDOFF.md', '# Historical handoff snapshots', 'f9dfedaf289de16f1226d2356ccfa451034e2e5e05b19348f0c08095e5a0a1ee'),
]
for path, marker, expected in items:
    marker_bytes = marker.encode('utf8')
    prefix, suffix = path.read_bytes().split(marker_bytes, 1)
    history = marker_bytes + suffix
    assert hashlib.sha256(history).hexdigest() == expected
    text = prefix.decode('utf8')
    text = text.replace('17:10 UTC', '17:32 UTC').replace('92项观测到终轮，82/800项', '97项观测到终轮，92/800项')
    text = text.replace('8项活动、700项等待', '8项活动、695项等待')
    text = text.replace('双5090均100%，70/68℃', '双5090利用率100%/96%，69/64℃')
    text = text.replace('CPU11.02/122.88核，RAM75.23GB', 'CPU11.11/122.88核，RAM75.06GB')
    text = text.replace('17:10运行状态', '17:32运行状态')
    if path.name == '返修实验总览.md':
        assert '| 机制native消融 |' not in text
        text = text.replace('| 机制三视图 |', '| 机制native消融 | 92/800新增已严格离机，9完整场景×10/9/6种子；486统计独立复算、旧7场景数值保留 | [九场景native表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_interim92_20261009/TABLES.md) |\n| 机制三视图 |')
        text = text.replace('当前机制消融也体现取舍：', '新10份三视图评价在推理前因旧11项审批基数检查而停止，0新接受、无结果输出；现场保留，独立修复须通过有效10项审批门检，主训练及旧82不变。\n\n当前机制消融也体现取舍：')
    else:
        start, end = text.index('- 主机制：'), text.index('\n- FLGMM：')
        text = text[:start] + ('- 主机制：97观测终轮、92/800严格验收离机、8活动、695等待、0训练失败；100Full显式复用。'
            '真实轮次69/67/60/27/25/23/23/新任务尚无round；双GPU100%/96%，69/64℃，RecoveryNone/OOM0，'
            'CPU11.11/122.88核、RAM75.06GB、余量1.061TB。root_live_20261009T173249Z.json，'
            'SHA1672c97f331fa3e7902462e4ba35661cf2a21dfd0a8f5e33155387eecde56e53。观测97与接受92区分，冻结70round/valid-only/8并发保持。') + text[end:]
        text += ('\n- 新native92九场景表已验收：celeba_mechanism_v1/native_interim92_20261009/TABLES.md，486独立统计，旧7场景不变；三视图仍82接受/固定7场景。'
            'after82精确10首worker在require_approval旧11项基数断言失败，EXITED/0worker/0完成/空输出树，CNN前停止。'
            '失败现场及ROOT_FAILURE_REVIEW.json SHA947e809b1db1585ea41622532730c243e86aae4b5e29212d5d6984a5ba2af169保留，旧after82禁止重启；'
            '仅允许独立审阅修复版本，不能将准备计为接受。\n\n')
    path.write_bytes(text.encode('utf8') + history)
    assert path.read_bytes().split(marker_bytes, 1)[1] == suffix
    print(path.name + ': current updated; historical suffix byte-exact')
