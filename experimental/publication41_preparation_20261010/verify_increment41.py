"""Run sealed40 commit verifier with only explicit41 parent/count/receipt metadata substitutions."""
from pathlib import Path
import hashlib
B=Path(__file__).resolve().parent;R=B.parents[1]
source=R/'tmp/publication40_preparation_20261010/verify_increment40.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='160b5f4b56a302d69fbff698e5d5c2252d6edf58a43f00eebbf39b24201e363d'
text=source.read_text('utf8')
for old,new in [('increment40','increment41'),('3601c9dfca63dc1c6203aceb2c7fc066faa630fa','59ec6455c1402ff3bfdac454cbf8de9daf0d216d'),('[900, 170, 170, 16, 22]','[900, 180, 170, 18, 23]'),("receipt['C70_table_included'] is True","receipt['C70_table_included'] is False"),('mechanism_offserver_verified=170','mechanism_offserver_verified=180'),('FLGMM_fullcoverage_new_offserver_verified=16, Hybrid_offserver_verified=22','FLGMM_fullcoverage_new_offserver_verified=18, Hybrid_offserver_verified=23')]:
    assert old in text;text=text.replace(old,new)
namespace=dict(__name__='sealed40_metadata_rebound_for41',__file__=str(__file__))
exec(compile(text,str(source)+':metadata41','exec'),namespace)
if __name__=='__main__':namespace['main']()
