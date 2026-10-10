"""Rebind the established FLGMM adoption checks to the actual exact-one delta."""
from pathlib import Path
import ast,difflib,json,re
R=Path(__file__).resolve().parents[1]
p=R/'tmp/adopt_FL96_after9_delta_root_20261010.py'
before=p.read_text('utf8')
changes={
 'seventh FL96 delta to the actual sixth backup':'eighth FL96 delta to the actual seventh backup',
 'celeba_flgmm_fullcoverage_delta_after9_20261010':'celeba_flgmm_fullcoverage_delta_after11_20261010',
 'deb6640101fe4176ac7deb4dc0345e6ba2de9566b33a0fe92fdb250bd43f5eb2':'ec2aa481c79624c602bf6de97ecc9fc9f6a5dc2a41a55bb7c9c6a4361d056ffb',
 '70ee3144b9e67d0000f22f25202c9fdda0437db17d2a1e628b28887be2c12687':'5edcd043a1eb693116246f439d7d3db048dee7d38702b80713e916c528bddf1e',
 '89f5fded09b3d1af61fd29d4d02eb6c2145ef0b4ca7edbf6917320e4c6222b3d':'423e177734d0177e67056680e80c2251f0913b0d347faa4574fe67d87a445283',
 'ecaaa936289589c2b8b28ff42fa81e7e4eb09206fce49a187781be9ed4143577':'6feb41c9f2f06980d29865ca03d59e5d2cffeeb2a6f0d6f0209f065cf80caf80',
 '396543a8effb8f7e0a25ad6130c0f6bc8b957c88bfbe7e862aaa019e14a73684':'40e14dc03414462e091f14c7f3d1241fb11ca1766cbb54479f3bbc459f9de7f2',
 '40e14dc03414462e091f14c7f3d1241fb11ca1766cbb54479f3bbc459f9de7f2':'83bca8eef7ed5423315ad5bef4b42ccd0335bb750c4214f2e50abcd2db9e8e34',
 'for s in (91001, 91002)':'for s in (91003,)',
 "prior['accepted_total'] == latest['accepted_total'] == 9":"prior['accepted_total'] == latest['accepted_total'] == 11",
 "proof['accepted_new'] == handoff['accepted_new'] == 2":"proof['accepted_new'] == handoff['accepted_new'] == 1",
 "proof['accepted_total'] == handoff['accepted_new_cumulative'] == 11":"proof['accepted_total'] == handoff['accepted_new_cumulative'] == 12",
 "handoff['archive_members'] == 27":"handoff['archive_members'] == 17",
 'accepted_before=9, accepted_new=2, accepted_total=11':'accepted_before=11, accepted_new=1, accepted_total=12',
 'archive_members_verified=27':'archive_members_verified=17',
 'next_latest = dict(accepted_total=11':'next_latest = dict(accepted_total=12',
}
assert all(before.count(k)==1 for k in changes),[(k,before.count(k)) for k in changes if before.count(k)!=1]
after=re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda m:changes[m.group()],before)
ast.parse(after)
target=R/'tmp/adopt_FL96_after11_delta_root_20261010.py'
with target.open('x',encoding='utf8',newline='\n') as f:f.write(after)
patch=R/'tmp/FL96_after11_adopter_root_20261010.patch'
with patch.open('x',encoding='utf8',newline='\n') as f:f.write(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=p.name,tofile=target.name)))
print(json.dumps({'status':'SOURCE_REBOUND_ONLY','exact_replacements':len(changes),'adoption_executed':False}))
