"""Exact new-scene metadata contract; no numeric statistics or scientific execution."""
import re
from pathlib import PurePosixPath

IDS=['FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed'+str(s)+'_fullcoverage' for s in range(91001,91011)]
PACKAGE='tmp/fl_FFlip10_capacity_pool32_20261011'
PACKAGE_SHA='699e24a9421e684f446346d0eb46252020806ee3250a3d677a3d90c7c821de39'

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(value):
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None,'Missing/invalid actual SHA')

def relative(value):
    require(isinstance(value,str) and '\\' not in value and ':' not in value and not value.startswith('/'),'Repository-relative path required')
    require('..' not in PurePosixPath(value).parts and PurePosixPath(value).as_posix()==value,'Noncanonical path refused')

def validate_binding(binding):
    require(binding.get('root_adopted') is True,'Actual root71 adoption must precede table construction')
    for key in ('root71','candidate','transport'):relative(binding.get(key))
    for key in ('root71_sha256','candidate_seal_sha256','transport_sha256'):sha(binding.get(key))
    require(binding['root71']==PACKAGE+'/ROOT_SCIENTIFIC_ADOPTION.json','Root adoption proof required')
    require(binding['transport'].endswith('/TRANSPORT_VERIFICATION.json'),'Actual verified F transport proof required')
    require(binding['candidate']==PACKAGE and binding['candidate_seal_sha256']==PACKAGE_SHA,'Only actual Pool32 exact10 source package is in scope')

def validate_new_rows(rows):
    require(isinstance(rows,list) and len(rows)==10,'Exactly ten new rows required')
    require([r.get('id') for r in rows]==IDS,'Exact new10 IDs and order required')
    for seed,r in zip(range(91001,91011),rows):
        require((r.get('method'),r.get('distribution'),r.get('attack'),r.get('seed'))==('FLGMM','non-IID','F Flip',seed),'Only non-IID F Flip ten-seed scope allowed')
        for key in ('checkpoint_sha256','array_sha256','receipt_sha256'):sha(r.get(key))
