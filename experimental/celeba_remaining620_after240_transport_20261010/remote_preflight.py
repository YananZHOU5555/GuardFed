"""One read-only exact8/CPU111/source/previous-chain snapshot; no science imports."""
from pathlib import Path
import datetime,hashlib,json,os,sys
Q=Path('/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010')
T=Path('/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert sha(Q/'FILES_SHA256.json')=='a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03'
assert sha(T/'FILES_SHA256.json')=='1a021b707575292c33959c19fcfa2fa1ee8c7f285d20576c562e4843d1488fb3'
for base in (Q,T):
    for n,v in read(base/'FILES_SHA256.json')['files'].items():assert sha(base/n)==v['sha256'] and (base/n).stat().st_size==v['bytes']
assert sha(Q/'ROOT_APPROVED.json')=='55e0c1a5c08fd00a33ff1caaa862b5b1e67c4328559b750a95b6d0a5d1aebb6c'
for dep in read(T/'INPUTS.json')['dependencies'].values():assert sha(dep['remote'])==dep['sha256']
plan=read(Q/'PLAN.json')
candidate_ids=['minus_A_IID_Sp-DFA_seed91001', 'minus_A_IID_Sp-DFA_seed91002', 'minus_A_IID_Sp-DFA_seed91003', 'minus_A_IID_Sp-DFA_seed91004', 'minus_A_IID_Sp-DFA_seed91005', 'minus_A_IID_Sp-DFA_seed91006', 'minus_A_IID_Sp-DFA_seed91007', 'minus_A_IID_Sp-DFA_seed91008', 'minus_A_IID_Sp-DFA_seed91009', 'minus_A_IID_Sp-DFA_seed91010', 'minus_A_non-IID_Benign_seed91001']
ids=list(candidate_ids)
assert len(ids)==11 and ids==[i for i in plan['remaining620_ids'] if i in set(ids)]
latest=read(T/'exports/TRANSPORT_LATEST.json')
assert latest['receipt']=='/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010/exports/A4_20261010T103945406866Z/backup_receipt.json'
assert latest['receipt_sha256']==sha(latest['receipt'])=='5c988da2fe11625f949d4baee5be426b6b184dfbccc78782a2a6abcb236759ac'
assert latest['all_transported_ids']==['minus_C_non-IID_S-DFA_seed91001', 'minus_C_non-IID_S-DFA_seed91002', 'minus_C_non-IID_S-DFA_seed91003', 'minus_C_non-IID_S-DFA_seed91004', 'minus_C_non-IID_S-DFA_seed91005', 'minus_C_non-IID_S-DFA_seed91006', 'minus_C_non-IID_S-DFA_seed91007', 'minus_C_non-IID_S-DFA_seed91008', 'minus_C_non-IID_S-DFA_seed91009', 'minus_C_non-IID_S-DFA_seed91010', 'minus_C_non-IID_Sp-DFA_seed91001', 'minus_C_non-IID_Sp-DFA_seed91002', 'minus_C_non-IID_Sp-DFA_seed91003', 'minus_C_non-IID_Sp-DFA_seed91004', 'minus_C_non-IID_Sp-DFA_seed91005', 'minus_C_non-IID_Sp-DFA_seed91006', 'minus_C_non-IID_Sp-DFA_seed91007', 'minus_C_non-IID_Sp-DFA_seed91008', 'minus_C_non-IID_Sp-DFA_seed91009', 'minus_C_non-IID_Sp-DFA_seed91010', 'minus_A_IID_Benign_seed91001', 'minus_A_IID_Benign_seed91002', 'minus_A_IID_Benign_seed91003', 'minus_A_IID_Benign_seed91004', 'minus_A_IID_Benign_seed91005', 'minus_A_IID_Benign_seed91006', 'minus_A_IID_Benign_seed91007', 'minus_A_IID_Benign_seed91008', 'minus_A_IID_Benign_seed91009', 'minus_A_IID_Benign_seed91010', 'minus_A_IID_F Flip_seed91001', 'minus_A_IID_F Flip_seed91002', 'minus_A_IID_F Flip_seed91003', 'minus_A_IID_F Flip_seed91004', 'minus_A_IID_F Flip_seed91005', 'minus_A_IID_F Flip_seed91006', 'minus_A_IID_F Flip_seed91007', 'minus_A_IID_F Flip_seed91008', 'minus_A_IID_F Flip_seed91009', 'minus_A_IID_F Flip_seed91010', 'minus_A_IID_FedSA_seed91001', 'minus_A_IID_FedSA_seed91002', 'minus_A_IID_FedSA_seed91003', 'minus_A_IID_FedSA_seed91004', 'minus_A_IID_FedSA_seed91005', 'minus_A_IID_FedSA_seed91006', 'minus_A_IID_FedSA_seed91007', 'minus_A_IID_FedSA_seed91008', 'minus_A_IID_FedSA_seed91009', 'minus_A_IID_FedSA_seed91010', 'minus_A_IID_S-DFA_seed91001', 'minus_A_IID_S-DFA_seed91002', 'minus_A_IID_S-DFA_seed91003', 'minus_A_IID_S-DFA_seed91004', 'minus_A_IID_S-DFA_seed91005', 'minus_A_IID_S-DFA_seed91006', 'minus_A_IID_S-DFA_seed91007', 'minus_A_IID_S-DFA_seed91008', 'minus_A_IID_S-DFA_seed91009', 'minus_A_IID_S-DFA_seed91010'] and latest['accepted_offserver']==0
assert os.sched_getaffinity(0)=={111} and os.getpriority(os.PRIO_PROCESS,0)>=10
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
owners=[];active=[];duplicate=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
    try:
        if (proc/'stat').read_text().rsplit(')',1)[1].split()[0] in ('Z','X'):continue
        argv=[s.decode(errors='replace') for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
        if str(T/'transport.py') in argv and 'export' in argv:duplicate.append(int(proc.name))
        if str(Q/'evaluate_remaining.py') in argv and 'worker' in argv and '--id' in argv:active.append(argv[argv.index('--id')+1])
        for thread in (proc/'task').iterdir():
            cpus=os.sched_getaffinity(int(thread.name))
            if len(cpus)<=16 and 111 in cpus:owners.append(dict(pid=int(proc.name),tid=int(thread.name),cpus=sorted(cpus)))
    except (OSError,ValueError):pass
assert not owners and not duplicate,(owners,duplicate,active)
# Read-only closure/resource check; original scientific validator runs in transport unchanged.
allclosed=[p.parent.name for p in (Q/'attempt1/tasks').glob('*/REMOTE_COMPLETE.json')]
ids=[i for i in candidate_ids if i in set(allclosed)]
notready=[i for i in candidate_ids if i not in set(allclosed)]
assert not set(ids)&set(latest['all_transported_ids']) and not set(ids)&set(active)
if not ids:
    print(json.dumps(dict(status='NO_NEW_CLOSED_INTERSECTION_NO_COLLECTION',candidate_ids=candidate_ids,selected_ids=[],notready=notready,actual_remote_closed=len(allclosed),accepted_offserver=0)))
    sys.exit(0)
assert set(latest['all_transported_ids'])<=set(allclosed)
failures=[str(p) for p in (Q/'attempt1').rglob('*FAILURE*.json')]+[str(p) for p in (Q/'attempt1').rglob('failure*.json')]
assert not failures,failures
quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split();assert quota!='max'
nominal=0
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
    try:
        argv=[s.decode(errors='replace') for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
        if argv and 'python' in Path(argv[0]).name and any(a.startswith(('/workspace/guardfed_checks/','/workspace/GuardFed-')) for a in argv[1:]):
            env=dict(x.split('=',1) for x in (proc/'environ').read_text().split('\0') if '=' in x)
            nominal+=max([int(env[k]) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','GUARDFED_CPU_THREADS') if env.get(k,'').isdigit()]+[1])
    except (OSError,ValueError):pass
assert nominal+1<=int(quota)/int(period)
closed=[]
for identity in ids:
    task=Q/'attempt1/tasks'/identity;out=Q/'attempt1/runs'/identity
    assert {p.name for p in out.iterdir()}=={'receipt.json','bridge_receipt.json','validation_predictions.npz','strict_acceptance.json'}
    remote=read(task/'REMOTE_COMPLETE.json');binding=read(task/'binding.json');strict=read(out/'strict_acceptance.json')
    assert remote['status']=='REMOTE_STRICT_CLOSED_PENDING_OFFSERVER' and remote['accepted_offserver']==0
    assert strict['status']=='MECHANISM_VALID_THREE_VIEWS_ACCEPTED' and strict['native_comparison']['max_abs_difference']<=1e-12
    assert remote['id']==binding['id']==strict['id']==identity
    closed.append(dict(id=identity,checkpoint_sha256=binding['checkpoint_sha256'],binding_sha256=sha(task/'binding.json'),strict_sha256=sha(out/'strict_acceptance.json')))
print(json.dumps(dict(status='BOUNDED_CLOSED_INTERSECTION_NO_SELECTED_WORKER_CPU111_AVAILABLE',candidate_ids=candidate_ids,notready=notready,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    selected_ids=ids,closed=closed,restricted_CPU111_thread_owners=owners,other_active_replay_ids=active,duplicate_exporters=duplicate,
    actual_remote_closed=len(allclosed),failure_paths=failures,nominal_threads=nominal,quota_cores=int(quota)/int(period),previous_receipt=latest,source_seal_sha256=sha(Q/'FILES_SHA256.json'),transport_seal_sha256=sha(T/'FILES_SHA256.json'),
    source_members_verified=True,original_archive_verifier_exact=True,accepted_offserver=0,root_adopted=0)))
