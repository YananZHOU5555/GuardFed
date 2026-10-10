"""Bind the existing FL96 root adopter to the actual sealed seven-to-nine delta."""
from pathlib import Path
import ast, re

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT/'tmp/adopt_FL96_after5_delta_root_20261010.py').read_text(encoding='utf8')
changes = {
    'fifth FL96 delta to the actual fourth backup': 'sixth FL96 delta to the actual fifth backup',
    'tmp/celeba_flgmm_fullcoverage_delta_after5_20261010': 'tmp/celeba_flgmm_fullcoverage_delta_after7_20261010',
    '1d765295d8485ee8fd49cf48a1b372a648b16488b9c18004f48a1e78aeb5a987': '98ca2be9f0dea5cc1961e24ceeb9104e6acbfd76a71ea094a54985a042c4033b',
    'b667d2422e593da2d88d18dc1c6fbef7638d54570b2cf8ebd8741ca5f6327982': '547249468817c87cc08cb1d37c69d15f07fe4004b420da365ed7b2ace2a0149c',
    '8738e8ab72204b2094b277564198f4802d50c1d4c69000b45e461ebb1d7cc494': '97bc8a2f41751603026aaf2260dc495e7aaf2b2b28ba21df830b7503541655ae',
    '51ae9a0798d763b8bac6ef92022ae028f60bdc059ac9d0f0f7d114b597450288': '4403439d39196206e169f14428d68b59b779e1cdd4a5a9fb7d0dd0a3b13dabcf',
    "== handoff['previous_actual_offserver_sha256'] == 'e1c4ec50fbbfb1fabd2d97c6286ee08a0360472d1431f84066873c15267c75f9'": "== handoff['previous_actual_offserver_sha256'] == '89e6e45173fa0e514a97dbf8cd99072189dabd6c8f00c07cc4bd330fbb41ebf1'",
    "== handoff['offserver_acceptance_sha256'] == '89e6e45173fa0e514a97dbf8cd99072189dabd6c8f00c07cc4bd330fbb41ebf1'": "== handoff['offserver_acceptance_sha256'] == '396543a8effb8f7e0a25ad6130c0f6bc8b957c88bfbe7e862aaa019e14a73684'",
    'for s in (91007, 91008)': 'for s in (91009, 91010)',
    "prior['accepted_total'] == latest['accepted_total'] == 5": "prior['accepted_total'] == latest['accepted_total'] == 7",
    "proof['accepted_total'] == handoff['accepted_new_cumulative'] == 7": "proof['accepted_total'] == handoff['accepted_new_cumulative'] == 9",
    'accepted_before=5, accepted_new=2, accepted_total=7': 'accepted_before=7, accepted_new=2, accepted_total=9',
    'next_latest = dict(accepted_total=7': 'next_latest = dict(accepted_total=9',
}
for old in changes:
    assert source.count(old) == 1, (old, source.count(old))
pattern = '|'.join(re.escape(k) for k in sorted(changes, key=len, reverse=True))
updated = re.sub(pattern, lambda match: changes[match.group()], source)
ast.parse(updated)
with (ROOT/'tmp/adopt_FL96_after7_delta_root_20261010.py').open('x', encoding='utf8', newline='\n') as stream:
    stream.write(updated)
print('Prepared actual 7-to-9 strict adopter; shared state unchanged.')
