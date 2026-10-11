"""Consume an existing original14 comparison; archive once, no CNN/fit/strict rerun."""
import argparse,io,json,os,socket,sys,tarfile,time,importlib.util,hashlib
from pathlib import Path,PurePosixPath
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--comparison-sha256',required=True);p.add_argument('--runtime-manifest',type=Path,required=True);p.add_argument('--runtime-manifest-sha256',required=True);p.add_argument('--destination',type=Path,required=True);p.add_argument('--cpu',type=int,required=True);a=p.parse_args()
pins=json.loads((HERE/'SOURCE_PINS.json').read_bytes());stage=Path(pins['server_stage']);repo=Path(pins['server_repo']);package=Path(pins['server_package'])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sys.platform=='linux' and sha('/etc/vast-agents-guide.md')==pins['guide_sha256']
assert HERE==stage/'closure_source' and os.sched_getaffinity(0)=={a.cpu} and os.getpriority(os.PRIO_PROCESS,0)>=10
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(n)=='1' for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
assert not a.destination.exists() and not a.destination.with_suffix('.failure.json').exists()
assert a.destination.parent==stage and a.destination.name.startswith('offserver_closure_')
assert sha(stage/'scope.json')==pins['scope_sha256']
assert sha(stage/'GATE_COMPARISON.json')==a.comparison_sha256
assert a.runtime_manifest==HERE/'RUNTIME_MEMBERS.json' and sha(a.runtime_manifest)==a.runtime_manifest_sha256
runtime_members=json.loads(a.runtime_manifest.read_bytes())['files']
for name,pin in runtime_members.items():
 pp=PurePosixPath(name);assert name.startswith('runtime/') and '..' not in pp.parts and ':' not in name and '\\' not in name
 src=stage/name;assert src.is_file() and not src.is_symlink() and sha(src)==pin['sha256'] and src.stat().st_size==pin['bytes']
assert not list(stage.glob('failure*.json'))
owners=[];producers=[];broad=0
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name) in (os.getpid(),os.getppid()):continue
 try:
  if (proc/'stat').read_text().rsplit(')',1)[1].split()[0] in ('Z','X'):continue
  argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
  if str(stage) in argv and any(x.endswith('/adapter.py') or x.endswith('/compare_saved.py') for x in argv):producers.append(dict(pid=int(proc.name),argv=argv))
  for task in (proc/'task').iterdir():
   mask=os.sched_getaffinity(int(task.name))
   if a.cpu in mask and len(mask)<=32:owners.append(dict(pid=int(proc.name),tid=int(task.name),cpus=sorted(mask)))
   elif a.cpu in mask:broad+=1
 except (FileNotFoundError,ProcessLookupError):continue
assert not owners and not producers,dict(restricted_owners=owners,producers=producers)
# Original context is server-only metadata binding; it does not run a model.
for rel,digest in pins['sources'].items():
 if rel.endswith(('adapter.py','metadata.py','compare_saved.py','gate.py')):assert sha(repo/rel)==digest
sys.path.insert(0,str(package));import adapter
assert Path(adapter.__file__).resolve()==package/'adapter.py'
gate,scope=adapter.load_context(stage,repo);destination=a.destination
assert scope['jobs']==pins['ordered_jobs'] and len(scope['jobs'])==14
comparison=gate.read(stage/'GATE_COMPARISON.json')
assert comparison['status']=='EXACT14_CPU3_GATE_SAVED_COMPARISON_PASS' and comparison['scope_sha256']==pins['scope_sha256']
assert (comparison['checked_jobs'],comparison['rounds_per_job'],comparison['total_rounds'])==(14,3,42)
assert comparison['scientific_table_records']==0 and comparison['coverage_authorized'] is False
assert comparison['strict_receipts_sha256']=={e['id']:gate.digest(stage/e['output']/'item_receipt.json') for e in scope['jobs']}
for entry in scope['jobs']:
 out=stage/entry['output'];assert not list(out.glob('failure*.json')) and gate.read(out/'acceptance.json')['status']=='PASS'
 assert gate.read(out/'item_receipt.json')['acceptance_sha256']==gate.digest(out/'acceptance.json')
records = []
for entry in scope['jobs']:
    out = gate.HERE / entry['output']
    result = gate.read(out / 'result.json')
    acceptance = gate.read(out / 'acceptance.json')
    audit = gate.read(out / 'gradient_audit.json')
    records.append(dict(id=entry['id'], job_sha256=entry['job_sha256'], method=result['method'],
        distribution=result['distribution'], attack=result['attack'], metrics=result['metrics'],
        checkpoint_sha256=gate.digest(out / 'model.pt'), checkpoint_tensor_sha256=acceptance['checkpoint_tensor_sha256'],
        acceptance_sha256=gate.digest(out / 'acceptance.json'), gradient_audit_sha256=gate.digest(out / 'gradient_audit.json'),
        real_same_point_client_gradients=len(audit), attack_sign_oracle_exact=all(row['attack_sign_oracle_exact'] for row in audit),
        local_optimizer_steps=sum(row['local_optimizer_steps'] for row in audit),
        elapsed_seconds=acceptance['elapsed_seconds']))
destination.mkdir()
delivery = dict(status='EXACT14_EXISTING_ORIGINAL_COMPARISON_ARCHIVE_READY', checked_by_original_gate=True,
    real_image_runs=14, actual_rounds=42, scientific_table_records=0, test_evaluated=False, no_test_image_inference=True,
    records=records, original_scope_sha256=gate.digest(gate.HERE/'scope.json'), comparison_sha256=a.comparison_sha256,
    protected_source_data_after=gate.hashes(gate.REPO,scope['protected_source_hashes']), current_resource=gate.snapshot(),
    closure_preflight=dict(cpu=a.cpu,restricted_owners=owners,producers=producers,broad_mask_threads_containing_cpu=broad,host_exclusive_claim=False),
    complete_unix=time.time(),source_script_sha256=gate.digest(__file__),new_CNN=0,new_fit=0,new_training=0,
    limitation='CPU3 saved-artifact closure only; no GPU equivalence,70round performance or192 authorization')
gate.write(destination/'strict_delivery.json',delivery)
files = {}
def add(path):
    path = Path(path)
    gate.require(path.is_file() and not path.is_symlink(), 'Nonregular backup input')
    name = path.relative_to(gate.HERE).as_posix()
    gate.require(name not in files, 'Duplicate backup member')
    files[name] = dict(path=path, sha256=gate.digest(path), bytes=path.stat().st_size)
for name in scope['local_hashes']:
    add(gate.HERE / name)
for name in runtime_members:
    assert gate.digest(gate.HERE/name)==runtime_members[name]['sha256'] and (gate.HERE/name).stat().st_size==runtime_members[name]['bytes']
    if name not in files:
        add(gate.HERE / name)
for entry in scope['jobs']:
    if entry['job'] not in files:
        add(gate.HERE / entry['job'])
    for path in sorted((gate.HERE / entry['output']).iterdir()):
        add(path)
for name in ('scope.json', 'GATE_COMPARISON.json', 'closure_source/export14.py', 'closure_source/SOURCE_PINS.json', 'closure_source/RUNTIME_MEMBERS.json'):
    if name not in files:add(gate.HERE / name)
add(destination / 'strict_delivery.json')
ids = [entry['id'] for entry in scope['jobs']]
inventory = dict(schema='exact14_exploratory_gradient_canary_backup_v1', accepted_new_ids=ids,
    record_kind='CANARY_NOT_SCIENTIFIC_TABLE', scientific_table_records=0,
    members={name: {key: row[key] for key in ('sha256', 'bytes')} for name, row in files.items()})
data = (json.dumps(inventory, indent=2, allow_nan=False) + '\n').encode()
archive = destination / 'exact14_saved_gradient_canaries.tar.gz'
with tarfile.open(archive, 'w:gz') as handle:
    info = tarfile.TarInfo('backup_inventory.json'); info.size = len(data)
    handle.addfile(info, io.BytesIO(data))
    for name, row in sorted(files.items()):
        gate.require(gate.digest(row['path']) == row['sha256'] and row['path'].stat().st_size == row['bytes'], 'Changed backup input')
        handle.add(row['path'], arcname=name, recursive=False)
gate.write(destination / 'backup_receipt.json', dict(archive_sha256=gate.digest(archive),
    inventory_sha256=__import__('hashlib').sha256(data).hexdigest(), accepted_new_ids=ids,
    members=len(files) + 1, source_host=socket.gethostname(), record_kind='CANARY_NOT_SCIENTIFIC_TABLE',
    scientific_table_records=0, test_evaluated=False, source_script_sha256=gate.digest(__file__), scope_sha256=gate.digest(gate.HERE/'scope.json'), comparison_sha256=a.comparison_sha256, runtime_manifest_sha256=a.runtime_manifest_sha256))
print(json.dumps(dict(status=delivery['status'], records=records, backup=str(archive), members=len(files) + 1)), flush=True)
