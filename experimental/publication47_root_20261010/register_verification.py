"""Register actual Git47 remote bytes using the original registration checks."""
from pathlib import Path
import hashlib,re,ast
ROOT=Path(__file__).resolve().parents[2]
original=ROOT/'tmp/publication46_root_20261010/register_verification.py'
assert hashlib.sha256(original.read_bytes()).hexdigest()=='8dc65215b6969ceb7def0c9f12f9b931e17f53b11c5c1a97f2e27af8979b2726'
source=original.read_text('utf8')
bindings={
 'Git46':'Git47',
 'e7ea15c5d2e77c40be932e20f0efe941e584fb13':'d93f87a0f267e4d3d9c79a89c0853020c25c91ba',
 "{'native':212,'three_view':212}":"{'native':218,'three_view':212}",
 'FLGMM_new96=28':'FLGMM_new96=32',
 'publication_closed_increment46_verified_20261010.json':'publication_closed_increment47_verified_20261010.json',
 'A_IID_Benign_paired_table=10,':'A_IID_Benign_paired_table_retained_via_parent=10,gradient_new_attack_gates_source_only=14,actual_image_gates=0,',
}
for key in bindings:assert source.count(key)==1,key
bound=re.sub('|'.join(re.escape(k) for k in sorted(bindings,key=len,reverse=True)),lambda m:bindings[m.group()],source)
ast.parse(bound)
exec(compile(bound,str(original),'exec'),{'__file__':__file__,'__name__':'__main__'})
