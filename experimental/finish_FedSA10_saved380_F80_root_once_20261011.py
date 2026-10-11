"""Adopt the fixed actual saved batch and independently verify its F80 table once."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys
R=Path(__file__).resolve().parents[1];H=R/'tmp/mechanism_F_nonIID_FedSA10_three_view_prepare_20261011'
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=Path(p).relative_to(R).as_posix(),sha256=sha(p))
def save(p,v):
 with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def run(label,argv,input=None,stdoutname=None):
 assert not (H/(label+'_COMMAND.json')).exists()
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(argv,input=input,capture_output=True);end=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (H/(stdoutname or label+'.stdout')).write_bytes(p.stdout);(H/(label+'.stderr')).write_bytes(p.stderr)
 save(H/(label+'_COMMAND.json'),dict(argv=argv,start=start,end=end,returncode=p.returncode,automatic_retry=False,input_sha256=hashlib.sha256(input).hexdigest() if input else None));save(H/(label+'_EXIT.json'),dict(returncode=p.returncode,start=start,end=end))
 print(json.dumps(dict(step=label,returncode=p.returncode,stdout_tail=p.stdout.decode(errors='replace')[-1200:],stderr_tail=p.stderr.decode(errors='replace')[-1800:])),flush=True);p.check_returncode()
for n in ['ACTUAL_EXECUTE','ACTUAL_JOIN']:assert read(H/(n+'_EXIT.json'))['returncode']==0
assert not (H/'ACTUAL_HANDOFF.json').exists()
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 taskset -c 111 nice -n 10 ionice -c 3 python -B -']
run('CPU_RELEASE',cmd,(H/'cpu_release_readonly.py').read_bytes(),stdoutname='CPU_RELEASE.json')
raw=read(H/'RAW_STORAGE_INDEX.json')
run('ACTUAL_FINALIZE',[sys.executable,'-B',str(H/'finalize_actual_handoff.py'),'--archive-sha256',raw['archive_sha256'],'--offserver-sha256',raw['offserver_verification_sha256'],'--index-sha256',sha(H/'MECHANISM_INDEX.json'),'--join-sha256',sha(H/'NATIVE_IDENTITY_JOIN.json')])
names=set(read(H/'FILES_SHA256.json')['files'])|{'FILES_SHA256.json','ROOT_SOURCE_REVIEW.json','EXECUTION_INPUTS.json','ACCEPTED_NATIVE_SCOPE.json','MECHANISM_INDEX.json','NATIVE_IDENTITY_JOIN.json','ACTUAL_HANDOFF.json','RAW_STORAGE_INDEX.json','CPU_RELEASE.json','CPU_RELEASE_EXIT.json'}
names.update(p.name for p in H.iterdir() if p.is_file() and (p.name.endswith(('_COMMAND.json','_EXIT.json','.stdout','.stderr','.stderr.txt','.stdout.json')) or p.name in {'PREFLIGHT.stdout.json','PREFLIGHT.stderr.txt'}))
save(H/'ACTUAL_DELIVERY_FILES_SHA256.json',dict(status='ACTUAL_FIXED_FEDSA10_COMPACT_DELIVERY_NOT_ROOT_ADOPTION',files={n:dict(sha256=sha(H/n),bytes=(H/n).stat().st_size) for n in sorted(names)}))
e=read(H/'EXECUTION_INPUTS.json');a=read(H/'ROOT_ACTUAL_PINS.template.json');a.update(status='ROOT_ACTUAL_FIXED_FEDSA10_SOURCE_AND_SAVED_PIN_AUTHORIZATION',native_root_path=e['native_root_path'],native_root_sha256=e['native_root_sha256'],authorized_snapshot_path=e['authorized_snapshot_path'],authorized_snapshot_sha256=e['authorized_snapshot_sha256'],delivery_sha256=sha(H/'ACTUAL_DELIVERY_FILES_SHA256.json'),delivery_members=len(names),handoff_sha256=sha(H/'ACTUAL_HANDOFF.json'),proposed_index_sha256=sha(H/'MECHANISM_INDEX.json'),archive_sha256=raw['archive_sha256'],archive_members=raw['archive_members'],offserver_sha256=raw['offserver_verification_sha256'],source_adopter_sha256=sha(H/'adopt_root.py'),accepted_native_scope_sha256=e['accepted_native_scope_sha256'],execution_authorized=True,root_source_review_sha256=sha(H/'ROOT_SOURCE_REVIEW.json'))
save(H/'ROOT_ACTUAL_PINS.json',a);run('ROOT_ADOPT',[sys.executable,'-B',str(H/'adopt_root.py'),'--actual-pins-sha256',sha(H/'ROOT_ACTUAL_PINS.json')])
adoption=R/'tmp/celeba_mechanism_remaining_F_nonIID_FedSA10_root_adoption_20261011/ROOT_ADOPTION.json';root=read(adoption);index=R/root['records_index_path'];assert root['cumulative_accepted']==380 and root['native_max_abs_difference']==0
b=read(H/'ROOT_BINDING.template.json');b.update(status='ACTUAL_REPLAY380_BOUND_FOR_F_EIGHT_SCENES80',root_adopted=True,adoption=adoption.relative_to(R).as_posix(),adoption_sha256=sha(adoption),index=index.relative_to(R).as_posix(),index_sha256=sha(index),native_root=Path(e['native_root_path']).relative_to(R).as_posix(),native_root_sha256=e['native_root_sha256'],root_status=root['status'],source_seal_sha256=sha(H/'FILES_SHA256.json'))
save(H/'ROOT_BINDING.json',b)
run('ACTUAL_BUILD',[sys.executable,'-B',str(H/'build.py'),'--binding-sha256',sha(H/'ROOT_BINDING.json')])
run('ACTUAL_VERIFY',[sys.executable,'-B',str(H/'verify_saved.py'),'--binding-sha256',sha(H/'ROOT_BINDING.json')],stdoutname='ACTUAL_VERIFY.stdout.json')
run('ROOT_TABLE_ADOPT',[sys.executable,'-B',str(H/'root_adopt_actual.py'),'--binding-sha256',sha(H/'ROOT_BINDING.json'),'--verifier-sha256',sha(H/'ACTUAL_VERIFY.stdout.json'),'--root-source-review-sha256',sha(H/'ROOT_SOURCE_REVIEW.json')])
