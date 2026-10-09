"""Preserve new deployment/freeze receipts; old Full models are not duplicated."""
from pathlib import Path
import hashlib
import json
import tarfile

CHECKS=Path('/workspace/guardfed_checks/server_reactivation_20261009')
STAGE=Path('/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
files=[STAGE/'dispatch_receipt.json',CHECKS/'full_inspection_v1/full_inspection.json']
files += [CHECKS/name for name in ('formal_launch.json','cu128_environment.json','evidence.py','model_inventory.json',
    'preflight_acceptance_cu128.json','preflight_cu128_offserver_verification.json','preflight_acceptance_cu130.json',
    'preflight_cu130_offserver_verification.json','freeze_mechanism_dispatch_20261009.py','start_mechanism_formal_20261009.py',
    'check_mechanism_live_20261009.py')]
files += [Path('/etc/supervisor/conf.d/guardfed_celeba_mechanism_formal.conf'),
          Path('/opt/supervisor-scripts/guardfed_celeba_mechanism_formal.sh')]
archive=CHECKS/'mechanism_frozen_startup_20261009.tar.gz'
assert not archive.exists()
members=[]
with tarfile.open(archive,'w:gz',compresslevel=3) as tf:
    for p in sorted(files):
        name=str(p).lstrip('/')
        members.append({'name':name,'bytes':p.stat().st_size,'sha256':sha(p)})
        tf.add(p,arcname=name,recursive=False)
receipt={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'members':members,
    'new_formal_queue':800,'reused_Full':100,'reused_models_duplicated':False,'test_started':False,
    'original_prepared_setup_sha256':'4669a4986219adf7348a46cd2bbe11291825a246cd6d0284f47eb595f8f90008',
    'exact_restore_sha256':'2b4f427600d7cc36f3f169ad7167dd57866d0c1703fae30465305b6860a1948a',
    'cu128_gate_archive_sha256':'b0facf3681cc208d3697c52eea5fdc235b076774bdace4bc6acfc3a5292d3464',
    'off_server_verified':False}
(CHECKS/'mechanism_frozen_startup_backup.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ('archive','bytes','sha256')}))
