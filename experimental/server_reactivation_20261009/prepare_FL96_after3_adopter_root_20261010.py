"""Rebind the already checked FL delta adopter to the delivered two-record increment."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
old = ROOT/'tmp/adopt_FL96_third_delta_root_20261009.py'
target = ROOT/'tmp/adopt_FL96_after3_delta_root_20261010.py'
source = old.read_text(encoding='utf8')
changes = {
    'third FL96 delta to the actual second backup': 'fourth FL96 delta to the actual third backup',
    "ATTEMPT = BASE / 'attempt_20261009T214720181886Z'": "ATTEMPT = ROOT / 'tmp/celeba_flgmm_fullcoverage_delta_after3_20261010'",
    '61a0d8be0ad7544290b01a16b24f2192ff2b23ff3c55e279a70e0ec85db4c85c': '8603b35e77f2b2975f71f3191f35a108395a88b44dd913847142219a3fce5ed9',
    'cf8498b9872efe337e46b3061733c3da923f683ab0bd8b4a16ca8cbe6445df83': 'b47567e33e48e49427a0730de7a1251fb30cce304c01391d7a45824f663d6021',
    '0eb5f92ca214ab4c2fde1c12fc40279e26f62a547397d07e6a1964251ab38eb3': 'a265d9213e4e1e7fe7d90dcedb82bdbb9eaee7299b770441c2e1a1358cc6ddb0',
    'd3accbbaeaa6ff34e526c9c9a6daac46c4dad9eb1a69014328295140fb2f20cb': 'c9aacd305eedf737f313ddcef9ab0b2c2c7ededc9b5aea2230f9205ee45be638',
    '9e9591bf3bede29b74a6ea34a944ef1b63b0584edd3ad95802d3725f77bd510e': 'bdb72d40fdb99a582ab0e6b346900cd92202611eedb25692c4ba05c91fe7b333',
    "== handoff['offserver_acceptance_sha256'] == 'bdb72d40fdb99a582ab0e6b346900cd92202611eedb25692c4ba05c91fe7b333'": "== handoff['offserver_acceptance_sha256'] == 'e1c4ec50fbbfb1fabd2d97c6286ee08a0360472d1431f84066873c15267c75f9'",
    "expected = ['FLGMM_Tg20_L2.0_lr0.001_IID_Benign_seed91004_fullcoverage']": "expected = [f'FLGMM_Tg20_L2.0_lr0.001_IID_Benign_seed{s}_fullcoverage' for s in (91005, 91006)]",
    "prior['accepted_total'] == latest['accepted_total'] == 2": "prior['accepted_total'] == latest['accepted_total'] == 3",
    "proof['accepted_new'] == handoff['accepted_new'] == 1": "proof['accepted_new'] == handoff['accepted_new'] == 2",
    "proof['accepted_total'] == handoff['accepted_new_cumulative'] == 3": "proof['accepted_total'] == handoff['accepted_new_cumulative'] == 5",
    "handoff['archive_members'] == 17": "handoff['archive_members'] == 27",
    'accepted_before=2, accepted_new=1, accepted_total=3': 'accepted_before=3, accepted_new=2, accepted_total=5',
    'archive_members_verified=17': 'archive_members_verified=27',
    'next_latest = dict(accepted_total=3': 'next_latest = dict(accepted_total=5',
}
# Apply simultaneously so the old offserver SHA can safely become the new predecessor.
import re
for before in changes:
    assert before in source, before
pattern = '|'.join(re.escape(k) for k in sorted(changes, key=len, reverse=True))
source = re.sub(pattern, lambda m: changes[m.group()], source)
start = source.index('    def load(name):')
end = source.index('\n\nreview = ', start)
block = source[start:end].replace('expected[0]', 'identity').replace("row = proof['records'][0]", "row = next(r for r in proof['records'] if r['id'] == identity)").replace('== 91004', "== int(identity.split('_seed')[1].split('_')[0])")
source = source[:start] + '    for identity in expected:\n' + '\n'.join('    '+line for line in block.splitlines()) + source[end:]
ast.parse(source)
assert "prior['accepted_total'] == latest['accepted_total'] == 3" in source
assert "proof['accepted_total'] == handoff['accepted_new_cumulative'] == 5" in source
assert "for identity in expected:" in source
with target.open('x', encoding='utf8', newline='\n') as stream:
    stream.write(source)
print(target)
