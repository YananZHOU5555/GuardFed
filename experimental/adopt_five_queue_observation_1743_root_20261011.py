"""Run the unchanged bounded growth checks against the next measured snapshots."""
from pathlib import Path
import hashlib
R=Path(__file__).resolve().parents[1]
p=R/'tmp/adopt_five_queue_observation_root_20261010.py'
s=p.read_text(encoding='utf8')
replacements={
 'root_five_queue_20261010T113833Z.raw.json':'root_five_queue_20261010T172543Z.raw.json',
 'root_five_queue_20261010T123338Z.raw.json':'root_five_queue_20261010T174308Z.raw.json',
 '480ac705d3fdbcefc6feb9be3f5ba0906b159a9b1625953b3b9631a9116e9bbe':'c3bd0e481f06569bd48b74c061023593699fb28410e71ccf1fe82781bce8e450',
 'a62f9cd234d93c8d244e047c57d0f7f034d89e969a4746152e9fb84df7404f2e':'454e544c7a74766a2b29c4a2a1db3f496947592e7093dffd0e1d68c198eb0ac0',
 'ROOT_FIVE_QUEUE_GROWTH_20261010T1233.json':'ROOT_FIVE_QUEUE_GROWTH_20261010T1743.json',
 'Observed terminal-set growth with correct source/worker identities, not scientific result acceptance; gradient active metadata was sampled during handover.':'Observed terminal-set growth with correct source/worker identities; counts are observation only and do not increase scientific acceptance.'}
for i,(a,b) in enumerate(replacements.items()):
    assert s.count(a)==1
    s=s.replace(a,'__SNAPSHOT_PIN_'+str(i)+'__')
for i,b in enumerate(replacements.values()):s=s.replace('__SNAPSHOT_PIN_'+str(i)+'__',b)
exec(compile(s,str(p)+':NEXT_MEASURED_OBSERVATION','exec'),dict(__name__='__main__',__file__=str(p)))
