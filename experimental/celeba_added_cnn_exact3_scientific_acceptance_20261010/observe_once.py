import pathlib,subprocess,json,hashlib,datetime
P=pathlib.Path
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
assert sha(P('/etc/vast-agents-guide.md'))=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
root=P('/workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010');attempt=root/'outputs/attempt001';q=subprocess.run(['supervisorctl','status','guardfed_added_cnn_exact3_gate'],capture_output=True,text=True);out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'service':{'code':q.returncode,'stdout':q.stdout,'stderr':q.stderr},'members':{},'json':{},'logs':{},'no_remote_writes':True,'no_CNN_fit_training':True}
if attempt.exists():
 for p in sorted(attempt.rglob('*')):
  if not p.is_file():continue
  out['members'][str(p.relative_to(attempt))]={'sha256':sha(p),'bytes':p.stat().st_size}
  if p.suffix=='.json' and p.stat().st_size<=262144:out['json'][str(p.relative_to(attempt))]=json.loads(p.read_text())
conf=P('/etc/supervisor/conf.d/guardfed_added_cnn_exact3_gate.conf');out['supervisor_config']=conf.read_text()
for line in out['supervisor_config'].splitlines():
 if line.startswith(('stdout_logfile=','stderr_logfile=')):
  path=P(line.split('=',1)[1]);
  if path.is_file():
   with path.open('rb') as f:f.seek(max(0,path.stat().st_size-16384));tail=f.read()
   out['logs'][str(path)]={'bytes':path.stat().st_size,'tail':tail.decode(errors='replace')}
print(json.dumps(out))
