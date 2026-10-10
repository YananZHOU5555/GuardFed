"""Register the actual remote Git48 cutoff, with LoGoFair100 and native1000."""
from pathlib import Path
import ast,hashlib,re
ROOT=Path(__file__).resolve().parents[2]
original=ROOT/'tmp/publication46_root_20261010/register_verification.py'
assert hashlib.sha256(original.read_bytes()).hexdigest()=='8dc65215b6969ceb7def0c9f12f9b931e17f53b11c5c1a97f2e27af8979b2726'
source=original.read_text('utf8')
bindings={
 'Git46':'Git48',
 'e7ea15c5d2e77c40be932e20f0efe941e584fb13':'d7ecf9f266f4d7ff6824b8cf1efd6483768b7281',
 "{'native':212,'three_view':212}":"{'native':220,'three_view':220}",
 'FLGMM_new96=28':'FLGMM_new96=32',
 'gradient_screen64=5':'gradient_screen64=10',
 'publication_closed_increment46_verified_20261010.json':'publication_closed_increment48_verified_20261010.json',
 'A_IID_Benign_paired_table=10,':'A_two_IID_scenes_paired_table=20,native_method_table_records=1000,native_method_table_methods=10,',
 'LoGoFair100_new_independent_accepted=0':'LoGoFair100_new_independent_accepted=96,LoGoFair100_total_adopted=100',
}
for key in bindings:assert source.count(key)==1,key
bound=re.sub('|'.join(re.escape(k) for k in sorted(bindings,key=len,reverse=True)),lambda m:bindings[m.group()],source)
ast.parse(bound)
exec(compile(bound,str(original),'exec'),{'__file__':__file__,'__name__':'__main__'})
