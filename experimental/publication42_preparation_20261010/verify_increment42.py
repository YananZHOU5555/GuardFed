"""Original committed-blob verifier; only parent, counts, receipt/table labels rebound."""
from pathlib import Path
import hashlib
H=Path(__file__).resolve().parent;R=H.parents[1]
p=R/'tmp/publication40_preparation_20261010/verify_increment40.py'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='160b5f4b56a302d69fbff698e5d5c2252d6edf58a43f00eebbf39b24201e363d'
text=p.read_text('utf8')
for old,new in [('increment40','increment42'),('3601c9dfca63dc1c6203aceb2c7fc066faa630fa','b57ae3c07e1013c17820c2eb038d701a356b27c6'),('[900, 170, 170, 16, 22]','[900, 180, 180, 18, 23]'),('C70_table','C80_table'),('C70_in_full','C80_in_full'),('mechanism_offserver_verified=170','mechanism_offserver_verified=180'),('mechanism_three_view_offserver_verified=170','mechanism_three_view_offserver_verified=180'),('FLGMM_fullcoverage_new_offserver_verified=16, Hybrid_offserver_verified=22','FLGMM_fullcoverage_new_offserver_verified=18, Hybrid_offserver_verified=23')]:
    assert old in text;text=text.replace(old,new)
namespace=dict(__name__='sealed40_commit_verifier_rebound42',__file__=str(__file__))
exec(compile(text,str(p)+':metadata42','exec'),namespace)
if __name__=='__main__':namespace['main']()
