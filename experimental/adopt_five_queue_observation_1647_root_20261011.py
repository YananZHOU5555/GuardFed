"""Run the unchanged bounded growth checks against the next measured snapshots."""
from pathlib import Path
import hashlib
R=Path(__file__).resolve().parents[1]
p=R/'tmp/adopt_five_queue_observation_root_20261010.py'
s=p.read_text(encoding='utf8')
replacements={
 'root_five_queue_20261010T113833Z.raw.json':'root_five_queue_20261010T160612Z.raw.json',
 'root_five_queue_20261010T123338Z.raw.json':'root_five_queue_20261010T164728Z.raw.json',
 '480ac705d3fdbcefc6feb9be3f5ba0906b159a9b1625953b3b9631a9116e9bbe':'a4bbad3eda8fe5a3a01341d233013d6debbd9660de7b3d52e7fde3a7bb30c162',
 'a62f9cd234d93c8d244e047c57d0f7f034d89e969a4746152e9fb84df7404f2e':'cf27044d202579b17a7ee185207e7ff1f999c486d975260fbeda7af855b6c3e3',
 'ROOT_FIVE_QUEUE_GROWTH_20261010T1233.json':'ROOT_FIVE_QUEUE_GROWTH_20261010T1647.json',
 'Observed terminal-set growth with correct source/worker identities, not scientific result acceptance; gradient active metadata was sampled during handover.':'Observed terminal-set growth with correct source/worker identities; counts are observation only and do not increase scientific acceptance.'}
for i,(a,b) in enumerate(replacements.items()):
    assert s.count(a)==1
    s=s.replace(a,'__SNAPSHOT_PIN_'+str(i)+'__')
for i,b in enumerate(replacements.values()):s=s.replace('__SNAPSHOT_PIN_'+str(i)+'__',b)
exec(compile(s,str(p)+':NEXT_MEASURED_OBSERVATION','exec'),dict(__name__='__main__',__file__=str(p)))
