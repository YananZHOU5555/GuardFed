"""Small root-operation helpers; all science remains in the original100 package."""
import json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE=ROOT/'tmp/celeba_logofair_fullcoverage_20261010'
sys.path.insert(0,str(SOURCE))
import metadata as original
from metadata import require,digest,read,pinned,bulk_path
SOURCE_SHA='89f3e4b4fd02352fa3c0f5080ce00ab6854eaabb4e3803a2635a1d3ba51196c3'
ADOPTION=ROOT/'tmp/celeba_logofair32_root_adoption_20261010/ROOT_ADOPTION.json'
ADOPTION_SHA='145f30628270b457d84d47aeab60825c9d4fdb23a0ae27fcd44a35a19e62ae90'
REVIEW=ROOT/'tmp/celeba_logofair100_independent_review_20261010/REVIEW.json'
REVIEW_SHA='483b971a7ef8c56707585dd3c25fbc2c32f51252d01a2912e2666b529ea24d68'
FROOT=Path('F:/YananResearchStorage/GuardFed/logofair_fullcoverage100_20261010')

def sources():
    require(not sys.flags.optimize and digest(SOURCE/'FILES_SHA256.json')==SOURCE_SHA,'Original source seal differs')
    original.verify_sources()
    review=pinned(REVIEW,REVIEW_SHA)
    require(review['source_adoptable'] and review['source_seal_sha256']==SOURCE_SHA,'Independent source review differs')
    for name,pin in read(HERE/'FILES_SHA256.json')['files'].items():
        p=HERE/name;require(p.resolve().is_relative_to(HERE.resolve()) and digest(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],'Operation source changed')

def save(path,value):
    path=Path(path)
    if path.drive.upper()=='F:':bulk_path(path,1024*1024)
    else:require(path.resolve().is_relative_to(HERE.resolve()),'Only owned small root records on E')
    with path.open('x',encoding='utf8',newline='\n') as out:out.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def verify_seal(path,wanted):
    seal=pinned(path,wanted);parent=Path(path).resolve().parent
    for name,pin in seal['files'].items():
        p=(parent/name).resolve();require(p.is_relative_to(parent) and p.is_file(),'Unsafe/missing staged seal member')
        require(digest(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],'Staged member drift: '+name)
    return seal
