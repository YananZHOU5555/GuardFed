"""Exact original fail-stop/skip/count/statistics runner, only one Hybrid slot and bridge import."""
from pathlib import Path
import argparse,os
from common import HERE,digest,read,require,local_identity
def original_runner():
    source=HERE/'fl_queue_original.py';require(digest(source)==read(HERE/'SOURCE_PINS.json')['fl_queue_sha256'],'Original lifecycle source differs')
    text=source.read_text('utf8')
    for old,new in [('from screen_common import','from common import'),('for gpu in (0,1):','for gpu in (0,):')]:
        require(text.count(old)==1,'Original lifecycle interface changed');text=text.replace(old,new)
    env=dict(__name__='hybrid100_original_failstop_lifecycle',__file__=str(__file__));exec(compile(text,str(source)+':one_hybrid_slot','exec'),env);return env
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','summarize']);p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
    local_identity();os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    f=original_runner()
    if a.action=='run':f['run'](a.repo.resolve())
    else:f['summarize'](local_identity()[1])
