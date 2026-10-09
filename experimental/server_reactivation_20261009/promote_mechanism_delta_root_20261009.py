"""Promote a closed root delta to the established flat ledger convention."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009'
parser=argparse.ArgumentParser(); parser.add_argument('--tag',required=True); args=parser.parse_args()
assert args.tag.startswith('root_delta_') and '/' not in args.tag and '\\' not in args.tag
folder=BASE/args.tag
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
proof=read(folder/'ROOT_DELTA_VERIFICATION.json')
assert proof['status']=='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS' and proof['tag']==args.tag
assert sha(BASE/'verified_ledger.json')==proof['ledger_sha256']
for source,digest in [(folder/(args.tag+'.tar.gz'),proof['archive_sha256']),
                      (folder/(args.tag+'.tar.gz.receipt.json'),proof['receipt_sha256'])]:
    target=BASE/source.name
    assert source.resolve().is_relative_to(BASE.resolve()) and target.resolve().is_relative_to(BASE.resolve())
    assert sha(source)==digest and not target.exists()
    source.rename(target)
    assert sha(target)==digest
offserver=BASE/(args.tag+'_offserver_verification.json')
assert not offserver.exists() and sha(folder/'OFFSERVER_VERIFICATION.json')==proof['offserver_proof_sha256']
shutil.copyfile(folder/'OFFSERVER_VERIFICATION.json',offserver)
inspection=BASE/('mechanism_inspection_v4_'+args.tag)
assert not inspection.exists(); inspection.mkdir()
for source in (folder/'inspection').iterdir():
    assert source.is_file()
    shutil.copyfile(source,inspection/source.name)
    assert sha(source)==sha(inspection/source.name)
assert sha(inspection/'inspection.json')==proof['inspection_sha256']
print(json.dumps({'status':'CLOSED_DELTA_PROMOTED_NO_REPACK','tag':args.tag,'total':proof['total_new_strict_and_offserver'],
                  'inspection':str(inspection),'archive_sha256':proof['archive_sha256']}))
