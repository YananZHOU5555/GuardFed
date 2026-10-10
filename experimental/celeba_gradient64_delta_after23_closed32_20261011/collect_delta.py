"""Frozen-ID transport loop around the unchanged original gradient strict acceptor."""
from pathlib import Path
import ast, configparser, hashlib, importlib.util, io, json, os, socket, subprocess, sys, tarfile, time, traceback
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
PKG=Path('/workspace/guardfed_checks/celeba_gradient_screen64_v2_20261010')
REPO=Path('/workspace/GuardFed-celeba-expanded')
RUNS=Path('/workspace/celeba_gradient_screen64_v2_results_20261010')
DEST=HERE/'batch'
SEAL='11e2ae63c87e0440669a047c930f5465b13babf559cca17bd8808bf693ce7ced'
GUIDE='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
VERIFIER=Path('/workspace/guardfed_checks/server_reactivation_20261009/evidence_v4.py')
VERIFIER_SHA='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
# Reuse the actual first collector helpers; only CPU ownership metadata is rebound.
p=HERE/'pinned_collect_one.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='985f04d65afd8867cd606ac54b31b081ebeb3626b03b615245abe31a1d5504a7'
text=p.read_text('utf8');tree=ast.parse(text)
for node in tree.body:
 if isinstance(node,ast.FunctionDef) and node.name in ('sha','read','save','require','command','live'):
  body=ast.get_source_segment(text,node)
  if node.name=='live':
   for a,b in [('111 in cpus','110 in cpus'),('CPU111','CPU110'),('containing111','containing110')]:body=body.replace(a,b)
  exec(compile(body,str(p)+'[PINNED_HELPER_RESOURCE_METADATA_ONLY]','exec'),globals())
require(sha('/etc/vast-agents-guide.md')==GUIDE,'guide changed')
require(os.sched_getaffinity(0)=={110} and os.getpriority(os.PRIO_PROCESS,0)==10,'CPU110 nice10 required')
require('idle' in command(['ionice','-p',str(os.getpid())])['stdout'].lower(),'idle IO required')
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CUDA hidden required')
require(not DEST.exists(),'Preserve existing delta');DEST.mkdir()
try:
 auth=read(HERE/'AUTHORIZED_SNAPSHOT.json');ids=auth['authorized_ids'];prior=read(HERE/'PREVIOUS_ROOT_ADOPTION.json');previous=read(HERE/'PREVIOUS_OFFSERVER_ACCEPTANCE.json')
 require(sha(HERE/'PREVIOUS_ROOT_ADOPTION.json')==auth['prior_root_sha256']=='fe35ea95d58697d05c53635a38771c6180ccde23821b37972a330ff7020333d4','Prior root changed')
 require(sha(HERE/'PREVIOUS_OFFSERVER_ACCEPTANCE.json')==auth['prior_offserver_sha256']=='07e225bc72ee1537fc622e9b6d9b260e617bcd3166347e53b6eeee531427fade','Prior offserver changed')
 require(prior['accepted_ids']==previous['accepted_job_ids']==auth['accepted_prior_ids'] and prior['accepted_total']==23,'Actual root-adopted accepted23 binding required')
 require(ids and len(ids)==len(set(ids)) and not set(ids)&set(prior['accepted_ids']),'Exact nonoverlapping delta required')
 before=live();save(DEST/'LIVE_BEFORE.json',before)
 require(not before['restricted_CPU110_owners'],'CPU110 restricted owner')
 require(before['service']['returncode']==0 and 'RUNNING' in before['service']['stdout'],'Queue unhealthy')
 require(not list(RUNS.rglob('*FAILURE*')) and not list(RUNS.rglob('failure.json')),'Preserved queue/job failure')
 require(set(ids)<=set(before['queue']['strict_server_completed_ids']),'Frozen IDs not completed')
 require(sha(PKG/'FILES_SHA256.json')==SEAL,'Frozen package seal drift')
 for n,pin in read(PKG/'FILES_SHA256.json')['files'].items():require(sha(PKG/n)==pin,'Frozen member drift: '+n)
 manifest=read(PKG/'jobs/manifest.json');require(len(manifest['jobs'])==len({x['id'] for x in manifest['jobs']})==64 and sha(PKG/'jobs/manifest.json')==auth['manifest_sha256'],'Exact frozen64 grid changed')
 require([x['id'] for x in manifest['jobs'] if x['id'] in ids]==ids,'Manifest order/foreign ID')
 quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split();require(quota!='max','Explicit quota required')
 memory=int(Path('/sys/fs/cgroup/memory.max').read_text())-int(Path('/sys/fs/cgroup/memory.current').read_text())
 require(int(quota)/int(period)>=24 and memory>=2*1024**3,'CPU/RAM headroom')
 save(DEST/'RESOURCE.json',dict(pid=os.getpid(),cpu_ids=[110],threads=1,nice=10,io='idle',quota_cores=int(quota)/int(period),RAM_headroom_bytes=memory,guide_sha256=GUIDE))
 sys.path.insert(0,str(PKG));import run_queue
 stage=run_queue.STAGE;sys.path.insert(0,str(stage));import worker
 run_queue.bind_shared_inputs(worker,REPO.resolve())
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 require(torch.__version__=='2.11.0+cu128','Original isolated environment required')
 from accept_result import checked_result
 require(sha(stage/'accept_result.py')=='2c5d7699c6e9967d32c9080fb56b4672beb821cee0b20187c68195fb37d204e9','Original scientific acceptor changed')
 files={};records=[]
 def add(name,path):
  path=Path(path);require(path.is_file() and not path.is_symlink() and name not in files,'Invalid archive member');files[name]=dict(path=path,sha256=sha(path),bytes=path.stat().st_size)
 for ID in ids:
  require(not any(ID in x['argv'] or str(RUNS/ID) in x['argv'] for x in before['workers']),'Selected producer alive')
  progress=read(RUNS/ID/'progress.json');require(progress['round']==70 and progress['job_id']==ID and not Path('/proc',str(progress['pid'])).exists(),'Selected producer not quiescent')
  entry=next(r for r in manifest['jobs'] if r['id']==ID);jobpath=PKG/'jobs'/entry['job']
  require(sha(jobpath)==entry['job_sha256'],'job changed');job=read(jobpath)
  worker.verify_hashes(REPO,job['source_hashes'])
  result=checked_result(jobpath,RUNS/ID);require(result is not None,'No original strict result')
  for f in (RUNS/ID).iterdir():add('runs/'+ID+'/'+f.name,f)
  records.append(dict(status='ORIGINAL_CHECKED_RESULT_PASS',id=ID,rounds=result['rounds'],metrics=result['metrics'],checkpoint_sha256=sha(RUNS/ID/'model.pt'),job_sha256=sha(jobpath),validator_sha256=sha(stage/'accept_result.py'),package_seal_sha256=SEAL,source_data_files_rehashed=len(job['source_hashes']),runtime=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,collector_cuda_hidden=True,threads=1),original_training_provenance=result['provenance'],source_host=socket.gethostname(),new_CNN=0,new_training=0,root_adopted=False,method_champion_claim=False))
 for n in read(PKG/'FILES_SHA256.json')['files']:add('source/package/'+n,PKG/n)
 add('source/package/FILES_SHA256.json',PKG/'FILES_SHA256.json')
 # Shared live log is a measured prefix, never a closed per-job log.
 log_receipt=[];conf=Path('/etc/supervisor/conf.d/guardfed_celeba_gradient_screen64_v2a.conf')
 require(conf.is_file(),'Actual service configuration required');add('source/supervisor.conf',conf)
 cfg=configparser.ConfigParser(interpolation=None);cfg.read(conf)
 for key in ('stdout_logfile','stderr_logfile'):
  value=cfg.get('program:guardfed_celeba_gradient_screen64_v2a',key,fallback=None)
  if not value or value.startswith('/dev/'):continue
  path=Path(value)
  if not path.exists():continue
  count=path.stat().st_size;require(count<=8*1024*1024,'Unexpected shared log size')
  data=path.open('rb').read(count);target=DEST/(key+'_prefix.log');target.write_bytes(data);add('logs/'+target.name,target)
  log_receipt.append(dict(source=str(path),prefix_bytes=len(data),sha256=sha(target),complete_per_job_log=False))
 strict=dict(status='ORIGINAL_CHECKED_RESULT_DELTA_PASS',accepted_new_ids=ids,accepted_before=23,strict_cumulative=23+len(ids),records=records,log_prefixes=log_receipt,scientific_acceptor_unchanged=True,source_seal_sha256=SEAL,source_host=socket.gethostname(),new_CNN=0,new_training=0)
 save(DEST/'ORIGINAL_STRICT.json',strict);save(DEST/'LIVE_AFTER.json',live())
 for n in ('ORIGINAL_STRICT.json','LIVE_BEFORE.json','LIVE_AFTER.json','RESOURCE.json'):add('evidence/'+n,DEST/n)
 for n in ('AUTHORIZED_SNAPSHOT.json','PREVIOUS_ROOT_ADOPTION.json','PREVIOUS_OFFSERVER_ACCEPTANCE.json','collect_delta.py','pinned_collect_one.py'):add('evidence/'+n,HERE/n)
 require(sha(VERIFIER)==VERIFIER_SHA,'Original archive verifier changed');add('source/evidence_v4.py',VERIFIER)
 inventory=dict(accepted_new_ids=ids,members={n:{k:r[k] for k in ('sha256','bytes')} for n,r in files.items()},old_models_repacked=0,selected_only=True,scientific_source_seal=SEAL,previous_root_sha256=auth['prior_root_sha256'],previous_offserver_sha256=auth['prior_offserver_sha256'],authorized_snapshot_sha256=sha(HERE/'AUTHORIZED_SNAPSHOT.json'))
 payload=(json.dumps(inventory,indent=2)+'\n').encode();archive=DEST/'gradient64_delta_after23_closed32.tar.gz'
 with tarfile.open(archive,'w:gz') as t:
  info=tarfile.TarInfo('backup_inventory.json');info.size=len(payload);t.addfile(info,io.BytesIO(payload))
  for n,row in sorted(files.items()):
   require(sha(row['path'])==row['sha256'],'Input changed before backup');t.add(row['path'],arcname=n,recursive=False)
 require(all(sha(row['path'])==row['sha256'] for row in files.values()),'Input changed during backup')
 for ID in ids:
  job=read(PKG/'jobs'/next(x['job'] for x in manifest['jobs'] if x['id']==ID));worker.verify_hashes(REPO,job['source_hashes'])
 receipt=dict(archive_sha256=sha(archive),inventory_sha256=hashlib.sha256(payload).hexdigest(),accepted_new_ids=ids,members=len(files)+1,source_host=socket.gethostname(),archive=str(archive),original_strict_sha256=sha(DEST/'ORIGINAL_STRICT.json'),scientific_offserver_accepted=0,previous_root_sha256=auth['prior_root_sha256'],previous_offserver_sha256=auth['prior_offserver_sha256'])
 spec=importlib.util.spec_from_file_location('original_archive_verifier',VERIFIER);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
 receipt['remote_member_verification']=v.verify_archive(archive,receipt);save(DEST/'backup_receipt.json',receipt)
 print(json.dumps(dict(status='REMOTE_STRICT_AND_DELTA_ARCHIVE_PASS',receipt=receipt)),flush=True)
except BaseException as exc:
 save(DEST/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False));raise
