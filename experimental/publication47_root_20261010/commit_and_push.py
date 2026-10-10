"""Bind Git47 metadata to the unchanged verified Git46 commit/push procedure."""
from pathlib import Path
import hashlib,re,ast
ROOT=Path(__file__).resolve().parents[2]
original=ROOT/'tmp/publication46_root_20261010/commit_and_push.py'
assert hashlib.sha256(original.read_bytes()).hexdigest()=='f4bc8cb630ecb8e07fead6c8d95cf559ebd3931012401ad00a5a760f9819d237'
source=original.read_text('utf8')
bindings={
 'tmp/publication_increment46_20261010':'tmp/publication_increment47_20261010',
 'from publish_increment46 import':'from publish_increment47 import',
 "{'native':212,'three_view':212}":"{'native':218,'three_view':212}",
 'verify_increment46.py':'verify_increment47.py',
 'Record mechanism212 and paired A-ablation validation table':'Record mechanism218, FLGMM32 and prepared gradient validation gates',
 'Join twelve new saved three-view results to the accepted native checkpoint restore chain. ':'Accept six more terminal mechanism records and four FLGMM coverage records with strict offserver restore chains. ',
 'Add the complete ten-seed IID Benign Full/minus-A comparison, retaining raw/native/shared ':'Retain the previously adopted A/Full comparison and all raw/native/shared ',
 'Prepare the two gradient methods for future fixed-recipe coverage without selecting ':'Prepare fourteen common-horizon new-attack validation gates and review added-method view compatibility without selecting ',
}
for key in bindings:assert source.count(key)==1,key
bound=re.sub('|'.join(re.escape(k) for k in sorted(bindings,key=len,reverse=True)),lambda m:bindings[m.group()],source)
ast.parse(bound)
exec(compile(bound,str(original),'exec'),{'__file__':__file__,'__name__':'__main__'})
