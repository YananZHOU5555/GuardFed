"""Post-exit strict acceptance and incremental gate artifacts; no training/inference."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent.parent
EXPECTED_SEAL='8c08bbabe0321adb4cf0f417a0784af58a7ddab3381a2b6c03e3123f79c3f112'
EXPECTED_APPROVAL='da49e151d2cc2766da90c3559fac972b504c40ef5d64152822706cb43a112c36'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as stream:stream.write(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    assert HERE==Path('/workspace/guardfed_checks/celeba_hybrid_cuda_execution_20261009')
    assert sha(HERE/'FILES_SHA256.json')==EXPECTED_SEAL and sha(HERE/'APPROVED_gate.json')==EXPECTED_APPROVAL
    status=subprocess.run(['supervisorctl','status','guardfed_celeba_hybrid_cuda_four'],capture_output=True,text=True).stdout.strip()
    assert 'EXITED' in status,('Not completed service',status)
    assert not (HERE/'gate_failure.json').exists() and not (HERE/'screen_runs').exists()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            assert str(HERE/'driver.py') not in argv,'Actual CUDA driver still exists'
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    os.sched_setaffinity(0,[104]);os.environ.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(HERE));import body,driver
    scope,approval=driver.approve('gate',HERE/'APPROVED_gate.json',EXPECTED_APPROVAL,dispatch=False)
    scope=dict(scope,runtime_cuda_visible_device=approval['cuda_visible_device'],runtime_gpu_uuid=approval['gpu_uuid'])
    body.verify_scope(scope)
    complete=read(HERE/'gate_complete.json')
    assert complete['status']=='CUDA_GATE_STRICT_ACCEPTED_BACKUP_PENDING' and complete['source_seal_sha256']==EXPECTED_SEAL
    assert complete['accepted_ids']==[e['id'] for e in scope['jobs']]
    _,checked,compare=driver.functions(body,scope)
    records=[]
    for entry in scope['jobs']:
        result=checked(entry,scope);assert result is not None
        output=HERE/entry['output'];prov=read(output/'provenance.json');native=read(output/'native_replay.json');receipt=read(output/'acceptance.json')
        assert prov['gpu_uuid']==approval['gpu_uuid'] and prov['cpu_affinity']==[104] and prov['cpu_threads']==1 and prov['cuda_device_count']==1 and prov['cuda_visible_devices']=='0'
        assert native['metrics']==result['metrics'] and native['prediction_count']==19867
        records.append(dict(id=entry['id'],method=result['method'],distribution=result['distribution'],attack=result['attack'],seed=result['seed'],rounds=len(result['trajectory_metrics']),metrics=result['metrics'],model_sha256=sha(output/'model.pt'),checkpoint_tensor_sha256=receipt['checkpoint_tensor_sha256'],acceptance_sha256=sha(output/'acceptance.json'),native_replay_sha256=sha(output/'native_replay.json'),gpu_uuid=prov['gpu_uuid'],elapsed_seconds=receipt['elapsed_seconds']))
    pairs=compare(scope);assert pairs==complete['pairs'] and len(pairs)==2 and all(r['all_metrics_model_attacks_diagnostics_rng_exact'] for r in pairs)
    after=body.snapshot();before=read(HERE/'gate_dispatch.json')['resources_before']
    prior={r['id']:r['round'] for r in before['active']}
    grew=after['completed']>before['completed'] or any(r['id'] in prior and r['round'] is not None and prior[r['id']] is not None and r['round']>prior[r['id']] for r in after['active'])
    assert grew and not after['failed'],'Protected formal queue did not advance cleanly'
    body.verify_scope(scope)
    output=HERE/'completed_four_incremental_backup';output.mkdir()
    log=Path('/var/log/guardfed_celeba_hybrid_cuda_four.log');(output/'service.log').write_bytes(log.read_bytes())
    delivery=dict(status='FOUR_CUDA_CANARIES_STRICT_ACCEPTED_BACKUP_PENDING',at_unix=time.time(),accepted_ids=complete['accepted_ids'],new_CANARY=4,scientific_table_records=0,formal_multi_seed_records=0,test_evaluated=False,CPU_CUDA_equivalence_claim=False,round70_equivalence_claim=False,core_sha256=driver.CORE_SHA,body_sha256=sha(HERE/'body.py'),adapter_sha256=sha(HERE/'scientific_snapshot/adapters.py'),scientific_worker_sha256=sha(HERE/'scientific_snapshot/worker.py'),writer_policy_sha256=sha(HERE/'writer_policy.py'),source_seal_sha256=EXPECTED_SEAL,approval_sha256=EXPECTED_APPROVAL,all_pairs_exact=True,pairs=pairs,records=records,protected_formal_progress=dict(before=before,after=after,real_growth=True),source_data_before_after_verified=True,screen32_status='PREPARED_NOT_FROZEN',metadata_boundary=scope['metadata_boundary'],service_terminal=status,strict_checker_reused_sha256=sha(HERE/'driver.py'),postexit_collector_sha256=sha(__file__),source_backup_reference=read(HERE/'execution_attachments/startup_backup/backup_receipt.json'))
    save(output/'strict_delivery.json',delivery)
    files=[HERE/entry['output']/name for entry in scope['jobs'] for name in sorted(p.name for p in (HERE/entry['output']).iterdir() if p.is_file())]
    files += [HERE/'gate_complete.json',output/'strict_delivery.json',output/'service.log',Path(__file__)]
    names={str(p.relative_to(HERE)):p for p in files};assert len(names)==len(files)
    inventory=dict(status='CUDA_FOUR_INCREMENTAL_SCIENTIFIC_ARTIFACTS',accepted_new_ids=complete['accepted_ids'],source_host=socket.gethostname(),source_seal_sha256=EXPECTED_SEAL,source_backup_reference=delivery['source_backup_reference'],members={n:dict(sha256=sha(p),bytes=p.stat().st_size) for n,p in sorted(names.items())})
    inventory_path=output/'backup_inventory.json';save(inventory_path,inventory)
    archive=output/'hybrid_cuda_four_artifacts_20261009.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        tar.add(inventory_path,arcname='backup_inventory.json',recursive=False)
        for name,p in sorted(names.items()):tar.add(p,arcname=name,recursive=False)
    receipt=dict(status='ARCHIVED_PENDING_OFFSERVER_VERIFICATION',archive_sha256=sha(archive),inventory_sha256=sha(inventory_path),accepted_new_ids=complete['accepted_ids'],source_host=socket.gethostname(),members=len(names),source_backup_reference=delivery['source_backup_reference'])
    save(output/'backup_receipt.json',receipt)
    print(json.dumps(dict(strict_CANARY_accepted=4,pairs_exact=2,archive=str(archive),archive_sha256=receipt['archive_sha256'],members=len(names),table_records=0)))


if __name__=='__main__':main()
