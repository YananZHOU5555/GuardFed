"""Incremental accepted-result/source/gate backup; never starts training.

Deploy outside the immutable worker directory. Download archive and receipt,
verify every member off-server, then upload the verified receipt to backups/.
Only receipts with off_server_verified=true suppress an unchanged member.
"""
import hashlib
import importlib.util
import json
from datetime import datetime,timezone
from pathlib import Path
import tarfile

ROOT=Path('/workspace/GuardFed-celeba-expanded')
BASE=ROOT/'deployment/baseline_adapters_20260928/fullcoverage_20261003'
STAGE=ROOT/'results/revision_20261003/celeba_baseline_fullcoverage_v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

if __name__=='__main__':
    spec=importlib.util.spec_from_file_location('coverage_backup_acceptance',BASE/'run_fullcoverage.py')
    coverage=importlib.util.module_from_spec(spec);spec.loader.exec_module(coverage)
    manifest=json.loads((STAGE/'manifest.json').read_text())
    summary=coverage.summarize(manifest)
    backups=STAGE/'backups';backups.mkdir(exist_ok=True)
    previous={};backed_ids=set();receipts=[]
    for path in sorted(backups.glob('*_receipt.json')):
        r=json.loads(path.read_text())
        if r.get('off_server_verified'):
            previous.update(r.get('member_hashes',{}));backed_ids.update(r.get('accepted_new_ids',[]));receipts.append(path.name)
    new_ids={r['id'] for r in summary['records'] if not r['reused']}-backed_ids
    paths={p for p in BASE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    paths.update(p for p in (STAGE/'preflight').rglob('*') if p.is_file())
    paths.update(p for p in (STAGE/'failed_attempts').rglob('*') if p.is_file())
    paths.update(STAGE/n for n in ['manifest.json','PROTOCOL.md','summary.json','per_seed.csv'] if (STAGE/n).exists())
    paths.add(Path(__file__).resolve())
    for item in manifest['jobs']:
        if item['id'] not in new_ids:continue
        paths.add(Path(item['job']))
        paths.update(p for p in Path(item['output']).rglob('*') if p.is_file())
        log=STAGE/'logs'/f"{item['id']}.log"
        if log.exists():paths.add(log)
    files={}
    for p in sorted(paths):
        assert p.resolve().is_relative_to(ROOT.resolve()),p
        name=p.relative_to(ROOT).as_posix();h=sha(p)
        if previous.get(name)!=h:files[name]=dict(sha256=h,bytes=p.stat().st_size)
    tag=f"incremental_new{len(new_ids)}_total{summary['accepted_new']}_"+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    inventory=backups/(tag+'_inventory.json')
    inventory.write_text(json.dumps(dict(files=files,accepted_new_ids=sorted(new_ids),accepted_total=summary['accepted'],accepted_new_total=summary['accepted_new'],reused8_from_original_chain=True,previous_verified_receipts=receipts),indent=2))
    archive=backups/(tag+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for name in files:tar.add(ROOT/name,arcname=name,recursive=False)
        tar.add(inventory,arcname=inventory.relative_to(ROOT).as_posix(),recursive=False)
    receipt=dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,inventory_member=inventory.relative_to(ROOT).as_posix(),content_members=len(files),member_hashes={n:v['sha256'] for n,v in files.items()},accepted_new_ids=sorted(new_ids),accepted_new_total=summary['accepted_new'],accepted_total=summary['accepted'],previous_verified_receipts=receipts,off_server_verified=False,receipt_path=str(backups/(tag+'_receipt.json')))
    Path(receipt['receipt_path']).write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='member_hashes'}))
