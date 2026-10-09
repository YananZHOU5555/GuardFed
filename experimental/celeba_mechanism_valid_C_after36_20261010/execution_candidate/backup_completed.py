"""Archive only new strictly accepted, producer-closed replay outputs; no inference."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,io,json,os,socket,sys,tarfile
sys.dont_write_bytecode=True
import batch
HERE=Path(__file__).resolve().parent
parent_sha=batch.digest(HERE/'APPROVED.json')
scope,parent=batch.approved(HERE/'APPROVED.json',parent_sha)
previous_path=HERE/'BACKUP_LATEST.json'
previous=batch.read(previous_path) if previous_path.exists() else None
prior=set(previous['all_accepted_ids']) if previous else set()
complete={i:batch.read(HERE/('completed_'+i+'.json')) for i in batch.SELECTED if (HERE/('completed_'+i+'.json')).exists()}
ids=[i for i in batch.SELECTED if i in complete and i not in prior]
batch.require(ids,'No new strictly accepted IDs; do not duplicate archive')
active=set()
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():continue
    try:
        argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        if str(HERE/'batch.py') in argv and '--id' in argv:active.add(argv[argv.index('--id')+1])
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
batch.require(not set(ids)&active,'Producer still alive')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
dest=HERE/'backups'/('incremental_'+stamp);dest.mkdir(parents=True,exist_ok=False)
files={}
def add(p,name):
    p=Path(p);batch.require(p.is_file() and not p.is_symlink() and name not in files,'Unsafe or duplicate member')
    files[name]=dict(path=p,sha256=batch.digest(p),bytes=p.stat().st_size)
rows=[]
inv=batch.read(batch.PREPARED/'inventory_actual140_Full100refs.json');byid={r['id']:r for r in inv['records']}
for identity in ids:
    out=Path(parent['outputs'][identity]);row=complete[identity];accept=batch.read(out/'strict_acceptance.json')
    batch.require({p.name for p in out.iterdir()}=={'receipt.json','bridge_receipt.json','validation_predictions.npz','strict_acceptance.json'},'Unexpected output artifacts')
    receipt=batch.read(out/'receipt.json');bridge=batch.read(out/'bridge_receipt.json')
    batch.require(row['id']==accept['id']==receipt['id']==bridge['id']==identity,'Mixed ID')
    batch.require(row['strict_acceptance_sha256']==batch.digest(out/'strict_acceptance.json'),'Completion acceptance changed')
    batch.require(accept['status']=='MECHANISM_VALID_THREE_VIEWS_ACCEPTED' and accept['native_comparison']['accepted']
        and accept['native_comparison']['max_abs_difference']<=1e-12,'Not strict accepted')
    batch.require(row['checkpoint_sha256']==accept['checkpoint_sha256']==receipt['checkpoint_sha256']==byid[identity]['checkpoint']['sha256'],'Mixed checkpoint')
    batch.require(bridge['source_before']==bridge['source_after'] and bridge['artifact_before']==bridge['artifact_after'],'Source/artifact changed')
    batch.require(batch.digest(out/'receipt.json')==bridge['scientific_body_receipt_sha256'] and batch.digest(out/'bridge_receipt.json')==accept['bridge_receipt_sha256']
        and batch.digest(out/'validation_predictions.npz')==receipt['prediction_arrays_sha256'],'Saved result chain changed')
    batch.require(receipt['valid_n']==19867 and receipt['root_reconstruction']['root_n']==16277,'Wrong actual data denominator')
    for p in sorted(out.iterdir()):add(p,'runs/'+identity+'/'+p.name)
    for folder,suffix in [('logs','.log'),('approvals','.json')]:add(HERE/folder/(identity+suffix),folder+'/'+identity+suffix)
    add(HERE/('completed_'+identity+'.json'),'runtime/completed_'+identity+'.json')
    rows.append(row)
if previous is None:
    for row in batch.read(batch.PREPARED/'FILES_SHA256.json')['members']:add(batch.PREPARED/row['path'],'source/science/'+row['path'])
    add(batch.PREPARED/'FILES_SHA256.json','source/science/FILES_SHA256.json')
    for row in batch.read(HERE/'EXECUTION_SOURCE_SHA256.json')['members']:add(HERE/row['path'],'source/'+row['path'])
    for name in ('EXECUTION_SOURCE_SHA256.json','ROOT_APPROVED.json','EXECUTION_DRAFT.json','APPROVED.json','APPROVED.sha256','preflight.json','start_receipt.json','batch_resource_before.json'):
        add(HERE/name,'source/'+name)
if (HERE/'batch_complete.json').exists():add(HERE/'batch_complete.json','runtime/batch_complete.json')
if (HERE/'batch_failure.json').exists():add(HERE/'batch_failure.json','runtime/batch_failure.json')
add(__file__,'execution/backup_completed.py')
inventory=dict(status='PARTIAL_STRICT_ACCEPTED_REPLAY_INCREMENT' if len(prior|set(ids))<4 else 'ALL4_STRICT_ACCEPTED_REPLAY_INCREMENT',
    utc=datetime.now(timezone.utc).isoformat(),accepted_new_ids=ids,all_accepted_ids=[i for i in batch.SELECTED if i in prior|set(ids)],
    original_inventory_sha256=batch.INVENTORY_SHA,original_prepared_seal_sha256=batch.OLD_SEAL_SHA,
    execution_seal_sha256=batch.digest(HERE/'EXECUTION_SOURCE_SHA256.json'),approval_sha256=parent_sha,
    previous_backup=previous,records=rows,Full_three_views='MISSING_NOT_JOINED_NO_NEW_FULL_INFERENCE',
    models_repacked=0,new_training=0,new_test_inference=0,producer_closed=True,
    members={name:{k:row[k] for k in ('sha256','bytes')} for name,row in files.items()})
payload=(json.dumps(inventory,indent=2,allow_nan=False)+'\n').encode()
archive=dest/'incremental_valid_three_views.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    info=tarfile.TarInfo('backup_inventory.json');info.size=len(payload);t.addfile(info,io.BytesIO(payload))
    for name,row in sorted(files.items()):
        batch.require(batch.digest(row['path'])==row['sha256'] and row['path'].stat().st_size==row['bytes'],'Artifact changed before archive')
        t.add(row['path'],arcname=name,recursive=False)
for row in files.values():batch.require(batch.digest(row['path'])==row['sha256'],'Artifact changed during archive')
receipt=dict(archive_sha256=batch.digest(archive),inventory_sha256=hashlib.sha256(payload).hexdigest(),accepted_new_ids=ids,
    all_accepted_ids=inventory['all_accepted_ids'],members=len(files)+1,source_host=socket.gethostname(),archive=str(archive),
    previous_backup_receipt_sha256=previous.get('receipt_sha256') if previous else None)
batch.save_new(dest/'backup_receipt.json',receipt)
latest=dict(receipt_sha256=batch.digest(dest/'backup_receipt.json'),receipt=str(dest/'backup_receipt.json'),archive_sha256=receipt['archive_sha256'],
    archive=str(archive),all_accepted_ids=inventory['all_accepted_ids'])
tmp=HERE/'BACKUP_LATEST.next.json';batch.save_new(tmp,latest);os.replace(tmp,previous_path)
print(json.dumps(receipt),flush=True)
