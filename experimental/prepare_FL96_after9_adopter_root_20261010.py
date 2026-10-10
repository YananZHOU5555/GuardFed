"""Bind the unchanged linked-delta adopter to the actual 9-to-11 FL96 increment."""
from pathlib import Path
import ast, re
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'tmp/adopt_FL96_after7_delta_root_20261010.py').read_text(encoding='utf8')
changes={
 'sixth FL96 delta to the actual fifth backup':'seventh FL96 delta to the actual sixth backup',
 'tmp/celeba_flgmm_fullcoverage_delta_after7_20261010':'tmp/celeba_flgmm_fullcoverage_delta_after9_20261010',
 '98ca2be9f0dea5cc1961e24ceeb9104e6acbfd76a71ea094a54985a042c4033b':'deb6640101fe4176ac7deb4dc0345e6ba2de9566b33a0fe92fdb250bd43f5eb2',
 '547249468817c87cc08cb1d37c69d15f07fe4004b420da365ed7b2ace2a0149c':'70ee3144b9e67d0000f22f25202c9fdda0437db17d2a1e628b28887be2c12687',
 '97bc8a2f41751603026aaf2260dc495e7aaf2b2b28ba21df830b7503541655ae':'89f5fded09b3d1af61fd29d4d02eb6c2145ef0b4ca7edbf6917320e4c6222b3d',
 '4403439d39196206e169f14428d68b59b779e1cdd4a5a9fb7d0dd0a3b13dabcf':'ecaaa936289589c2b8b28ff42fa81e7e4eb09206fce49a187781be9ed4143577',
 "== handoff['previous_actual_offserver_sha256'] == '89e6e45173fa0e514a97dbf8cd99072189dabd6c8f00c07cc4bd330fbb41ebf1'":"== handoff['previous_actual_offserver_sha256'] == '396543a8effb8f7e0a25ad6130c0f6bc8b957c88bfbe7e862aaa019e14a73684'",
 "== handoff['offserver_acceptance_sha256'] == '396543a8effb8f7e0a25ad6130c0f6bc8b957c88bfbe7e862aaa019e14a73684'":"== handoff['offserver_acceptance_sha256'] == '40e14dc03414462e091f14c7f3d1241fb11ca1766cbb54479f3bbc459f9de7f2'",
 "expected = [f'FLGMM_Tg20_L2.0_lr0.001_IID_Benign_seed{s}_fullcoverage' for s in (91009, 91010)]":"expected = [f'FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed{s}_fullcoverage' for s in (91001, 91002)]",
 "prior['accepted_total'] == latest['accepted_total'] == 7":"prior['accepted_total'] == latest['accepted_total'] == 9",
 "proof['accepted_total'] == handoff['accepted_new_cumulative'] == 9":"proof['accepted_total'] == handoff['accepted_new_cumulative'] == 11",
 'accepted_before=7, accepted_new=2, accepted_total=9':'accepted_before=9, accepted_new=2, accepted_total=11',
 'next_latest = dict(accepted_total=9':'next_latest = dict(accepted_total=11',
}
for old in changes: assert source.count(old)==1, (old, source.count(old))
pattern='|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True))
updated=re.sub(pattern,lambda m:changes[m.group()],source)
ast.parse(updated)
with (ROOT/'tmp/adopt_FL96_after9_delta_root_20261010.py').open('x',encoding='utf8',newline='\n') as f:f.write(updated)
print('Prepared actual 9-to-11 original linked-delta adopter; shared state unchanged.')
