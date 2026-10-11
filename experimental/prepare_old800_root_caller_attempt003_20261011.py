"""Derive a source-only recovery caller; preserve prior failures and FedAvg."""
from pathlib import Path
import ast, hashlib, json, sys
R = Path(__file__).resolve().parents[1]
OLD = R / 'tmp/baseline_old800_runtime_root_20261011_attempt002'
D = R / 'tmp/baseline_old800_runtime_root_20261011_attempt003'
D.mkdir(exist_ok=False)
H = lambda b: hashlib.sha256(b).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())
review = R / 'tmp/old800_proc_guard_semantics_review_20261011/SOURCE_REVIEW.json'
assert H(review.read_bytes()) == '561bae41800cb0a19784cdc52f02e5b4236802a4f3a873df7f5ee7ad75d74cc3'
old_guard = (OLD / 'guard.py').read_text(encoding='utf-8')
assert H((OLD / 'guard.py').read_bytes()) == '1309c8697c43593e5ca1bd5ab015a1f96ac409d6b0b219d6fbddb4a2c572c5f2'
old_base = '/workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt002'
new_base = old_base.replace('attempt002', 'attempt003')
span = "process_cwd = str((p / 'cwd').resolve())"
assert old_guard.count(span) == 1
guard = old_guard.replace(old_base, new_base).replace(span, span + " if is_python else ''")
assert guard.replace(new_base, old_base).replace(span + " if is_python else ''", span) == old_guard
(D / 'guard.py').write_bytes(guard.encode())
worker = (OLD / 'worker.py').read_bytes()
assert H(worker) == 'cd15480764def5e85f88700f93212cd7f7adec87431d2e3ccad643670fe5e4ca'
(D / 'worker.py').write_bytes(worker)
old_queue = (OLD / 'run_queue.py').read_text(encoding='utf-8')
assert H((OLD / 'run_queue.py').read_bytes()) == '7036785bd12ecc591174a6c526315073c0828d1bae6399a9cb15c84fc71f9044'
queue = old_queue.replace('import guard\n', 'import guard\nfrom reuse_fedavg import verify_reused\n')
queue = queue.replace("canaries=auth['canary_ids'];selected=canaries if args.phase=='canary' else [i for i in auth['ordered_ids'] if i not in canaries]",
    "canaries=auth['canary_ids'];reused=verify_reused(auth,scope)\n"
    "    selected=[i for i in canaries if i not in reused] if args.phase=='canary' else [i for i in auth['ordered_ids'] if i not in canaries]")
queue = queue.replace("need(item['acceptance']==pin(BASE/'runs'/i/'formal_saved_acceptance.json'),'Canary acceptance changed')",
    "expected=reused[i]['acceptance'] if i in reused else pin(BASE/'runs'/i/'formal_saved_acceptance.json')\n"
    "            need(item['acceptance']==expected,'Canary acceptance changed')")
queue = queue.replace("    completed=[]\n", "    completed=list(reused) if args.phase=='canary' else []\n    newly_completed=[]\n")
queue = queue.replace("            completed.append(identity)\n", "            completed.append(identity);newly_completed.append(identity)\n")
queue = queue.replace("'planned':len(selected),'failed':0,'remaining':len(selected)-len(completed),'root_offserver_adopted':False}",
    "'planned':len(canaries) if args.phase=='canary' else len(selected),\n"
    "                      'new_planned':len(selected),'new_accepted_ids':newly_completed,\n"
    "                      'reused_ids':list(reused) if args.phase=='canary' else [],\n"
    "                      'failed':0,'remaining':len(selected)-len(newly_completed),'root_offserver_adopted':False}")
queue = queue.replace("    save(phase/'COMPLETE.json',{'ids':completed,'cohort':auth['exact_cohort'],'server_saved_accepted':True,'root_offserver_adopted':False})",
    "    save(phase/'COMPLETE.json',{'ids':completed,'new_ids':newly_completed,\n"
    "         'reused_ids':list(reused) if args.phase=='canary' else [],\n"
    "         'cohort':auth['exact_cohort'],'server_saved_accepted':True,'root_offserver_adopted':False})")
assert queue != old_queue
(D / 'run_queue.py').write_bytes(queue.encode())
for name in ('guard.py','worker.py','run_queue.py'):
    compile((D/name).read_bytes(),name,'exec')
assert not ({'guard','worker','run_queue','reuse_fedavg'} & sys.stdlib_module_names)
prepared = read(OLD/'ROOT_PREPARATION.json')
prepared.update(status='ROOT_SOURCE_PREPARATION_PENDING_REUSE_VALIDATOR_NOT_LAUNCHED', namespace=new_base,
    guard_edits=prepared['guard_edits'] + ['Conditional cwd read only for Python; all PID/thread affinity scans retained'],
    proc_guard_source_review={'path':str(review),'sha256':H(review.read_bytes())},
    previous_namespace=old_base, historical_process_117708_type='UNKNOWN',
    recovery_reason='Pre-inference resource-gate PermissionError; no FedAA scientific receipt',
    reused_canary_ids=[prepared['canary_ids'][0]], new_canary_ids=prepared['canary_ids'][1:],
    caller_sources={n:{'sha256':H((D/n).read_bytes()),'bytes':(D/n).stat().st_size} for n in ('guard.py','worker.py','run_queue.py')})
(D/'ROOT_PREPARATION.json').write_bytes((json.dumps(prepared,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(prepared,ensure_ascii=False))
