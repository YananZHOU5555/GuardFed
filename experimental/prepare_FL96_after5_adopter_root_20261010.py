"""Bind the existing strict delta adopter to the sealed 5-to-7 delivery."""
from pathlib import Path
import ast, re

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT/'tmp/adopt_FL96_after3_delta_root_20261010.py').read_text(encoding='utf8')
changes = {
    'fourth FL96 delta to the actual third backup': 'fifth FL96 delta to the actual fourth backup',
    "tmp/celeba_flgmm_fullcoverage_delta_after3_20261010": "tmp/celeba_flgmm_fullcoverage_delta_after5_20261010",
    '8603b35e77f2b2975f71f3191f35a108395a88b44dd913847142219a3fce5ed9': '1d765295d8485ee8fd49cf48a1b372a648b16488b9c18004f48a1e78aeb5a987',
    'b47567e33e48e49427a0730de7a1251fb30cce304c01391d7a45824f663d6021': 'b667d2422e593da2d88d18dc1c6fbef7638d54570b2cf8ebd8741ca5f6327982',
    'a265d9213e4e1e7fe7d90dcedb82bdbb9eaee7299b770441c2e1a1358cc6ddb0': '8738e8ab72204b2094b277564198f4802d50c1d4c69000b45e461ebb1d7cc494',
    'c9aacd305eedf737f313ddcef9ab0b2c2c7ededc9b5aea2230f9205ee45be638': '51ae9a0798d763b8bac6ef92022ae028f60bdc059ac9d0f0f7d114b597450288',
    'bdb72d40fdb99a582ab0e6b346900cd92202611eedb25692c4ba05c91fe7b333': 'e1c4ec50fbbfb1fabd2d97c6286ee08a0360472d1431f84066873c15267c75f9',
    "== handoff['offserver_acceptance_sha256'] == 'e1c4ec50fbbfb1fabd2d97c6286ee08a0360472d1431f84066873c15267c75f9'": "== handoff['offserver_acceptance_sha256'] == '89e6e45173fa0e514a97dbf8cd99072189dabd6c8f00c07cc4bd330fbb41ebf1'",
    'for s in (91005, 91006)': 'for s in (91007, 91008)',
    "prior['accepted_total'] == latest['accepted_total'] == 3": "prior['accepted_total'] == latest['accepted_total'] == 5",
    "proof['accepted_total'] == handoff['accepted_new_cumulative'] == 5": "proof['accepted_total'] == handoff['accepted_new_cumulative'] == 7",
    'accepted_before=3, accepted_new=2, accepted_total=5': 'accepted_before=5, accepted_new=2, accepted_total=7',
    'next_latest = dict(accepted_total=5': 'next_latest = dict(accepted_total=7',
}
for before in changes:
    assert source.count(before) == 1, (before, source.count(before))
pattern = '|'.join(re.escape(k) for k in sorted(changes, key=len, reverse=True))
source = re.sub(pattern, lambda m: changes[m.group()], source)
ast.parse(source)
with (ROOT/'tmp/adopt_FL96_after5_delta_root_20261010.py').open('x', encoding='utf8', newline='\n') as stream:
    stream.write(source)
print('Prepared the sealed 5-to-7 adopter; no shared state changed.')
