"""Use the original commit/push checks for Git48's actual compact evidence."""
from pathlib import Path
import ast,hashlib,re
ROOT=Path(__file__).resolve().parents[2]
original=ROOT/'tmp/publication46_root_20261010/commit_and_push.py'
assert hashlib.sha256(original.read_bytes()).hexdigest()=='f4bc8cb630ecb8e07fead6c8d95cf559ebd3931012401ad00a5a760f9819d237'
source=original.read_text('utf8')
bindings={
 'tmp/publication_increment46_20261010':'tmp/publication_increment48_v2_20261010',
 'from publish_increment46 import':'from publish_increment48 import',
 "{'native':212,'three_view':212}":"{'native':220,'three_view':220}",
 'verify_increment46.py':'verify_increment48.py',
 'Record mechanism212 and paired A-ablation validation table':'Complete ten-method CelebA validation tables and paired A ablations',
 'Join twelve new saved three-view results to the accepted native checkpoint restore chain. ':'Join eight new saved three-view results to native220 and adopt the complete two-scene paired A table. ',
 'Add the complete ten-seed IID Benign Full/minus-A comparison, retaining raw/native/shared ':'Extend IID/non-IID five-scenario native tables to ten methods with fixed-recipe LoGoFair100, retaining raw/native/shared ',
 'views, paired differences, partial F Flip records and negative outcomes.':'evidence for the original nine methods, paired differences and negative outcomes. LoGoFair virtual cohorts and the ASPD/accuracy tradeoff remain explicit.',
 'Prepare the two gradient methods for future fixed-recipe coverage without selecting ':'Record five more accepted gradient-search results and full reviewer-response candidates without selecting ',
}
for key in bindings:assert source.count(key)==1,key
bound=re.sub('|'.join(re.escape(k) for k in sorted(bindings,key=len,reverse=True)),lambda m:bindings[m.group()],source)
ast.parse(bound)
exec(compile(bound,str(original),'exec'),{'__file__':__file__,'__name__':'__main__'})
