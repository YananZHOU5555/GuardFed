"""Only the sealed exporter CPU110 literal is adapted in memory; original source is untouched."""
from pathlib import Path
import ast,hashlib,os,sys
p=Path('/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010/transport.py')
assert os.sched_getaffinity(0)=={111} and os.environ.get('CUDA_VISIBLE_DEVICES')==''
raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=='f71e6e4152625a5a0582a61ff9b3e55e4ccee65dfd70a4e657851c0247f9c9d7'
source=raw.decode('utf8');old='os.sched_setaffinity(0, [110])';new='os.sched_setaffinity(0, [111])'
assert source.count(old)==1
adapted=source.replace(old,new);assert adapted.count(new)==1 and adapted.replace(new,old)==source
compile(adapted,str(p)+':CPU111_ONLY','exec')
exec(compile(adapted,str(p)+':CPU111_ONLY','exec'),dict(__name__='__main__',__file__=str(p)))
