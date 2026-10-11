from pathlib import Path
import datetime,hashlib,json,subprocess
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=Path('/workspace/guardfed_checks/celeba_fednga96_bootstrap_20261011')
parent=Path('/workspace/celeba_fednga_fullcoverage_results_20261011');out=parent/'attempt001'
assert parent.resolve()==Path('/workspace/celeba_fednga_fullcoverage_results_20261011') and not parent.exists() and not out.exists()
service='guardfed_celeba_fednga_fullcoverage_v1';status=subprocess.run(['supervisorctl','status',service],capture_output=True,text=True)
assert status.returncode==3 and 'EXITED' in status.stdout
stderr=base/'supervisor.stderr.log';old=stderr.read_text()
assert 'FileNotFoundError' in old and "output.mkdir()" in old and str(out) in old
assert not (base/'actual001/BOOTSTRAP_FAILURE.json').exists()
assert json.loads((base/'actual001/INITIAL_RESOURCE.json').read_bytes())['status']=='ROOT_GRADIENT96_RESOURCE_PREFLIGHT_PASS'
assert sha(base/'root_bootstrap.py')=='2adbf5c5c80edde416ae9bf8eb9ac27532dbbc7b1c2d148c5d458aafd0aca307' and sha(base/'initial_preflight.py')=='f5b8d252df52d9afd276015fa197af7a521280e64f674405b9c8320cd34d9aef'
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
compute=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  argv=p.read_bytes().decode(errors='replace').split('\0')
  if any(a.startswith(str(base)+'/') or a.startswith('/workspace/guardfed_checks/celeba_gradient_fullcoverage_selected_20261011/') for a in argv):compute.append(dict(pid=int(p.parent.name),argv=argv))
 except OSError:pass
assert compute==[],compute
config=Path('/etc/supervisor/conf.d/'+service+'.conf');raw=config.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='1e762065eae5485a00d678c1a865b1b91697d66256cb45ee1b8f704f419bdf0e'
preserve=base/'output_parent_correction001';preserve.mkdir()
for path in (config,base/'supervisor.stdout.log',stderr):
 with (preserve/path.name).open('xb') as f:f.write(path.read_bytes())
changed=raw.replace((str(base)+'/actual001').encode(),(str(base)+'/actual002').encode());assert changed!=raw and raw.count((str(base)+'/actual001').encode())==1
assert not (base/'actual002').exists()
parent.mkdir()
config.write_bytes(changed)
correction=dict(status='ROOT_OUTPUT_PARENT_WIRING_CORRECTION_ONLY_NO_SCIENCE_RETRY',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_status=status.stdout,original_error=old,original_config_sha256=hashlib.sha256(raw).hexdigest(),new_config_sha256=sha(config),parent_created=str(parent),queue_out_absent=True,previous_training_workers=0,original_sources_unchanged=True,changed_only_receipt_dir_to_fresh_actual002=True,initial_resource_probe_rerun_required=True,automatic_retry=False,test=False)
with (preserve/'ROOT_CORRECTION.json').open('x') as f:json.dump(correction,f,indent=2)
calls=[]
for argv in (['supervisorctl','reread'],['supervisorctl','update',service],['supervisorctl','start',service],['supervisorctl','status',service]):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=25);calls.append(dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
 with (preserve/'COMMANDS.json').open('w') as f:json.dump(calls,f,indent=2)
 assert p.returncode==0,p.stdout+p.stderr
print(json.dumps(dict(correction=correction,supervisor=calls)))
