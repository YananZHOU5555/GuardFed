"""Store large observer output only on verified F; preserve all failed attempts."""
import datetime,hashlib,json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent
F=Path('F:/YananResearchStorage/GuardFed/fl_FFlip10_capacity_pool32_20261011/runtime')
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
def run_capture(cmd,payload,name,timeout,*,bulk):
 assert name in ('OBSERVATION','FRESH_OBSERVATION','START_RECEIPT')
 dest=F if bulk else H
 if bulk:
  vol=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command','Get-Volume -DriveLetter F | Select-Object FileSystemLabel,HealthStatus,SizeRemaining | ConvertTo-Json -Compress'],text=True))
  assert vol['FileSystemLabel']=='Yanan 2TB' and vol['HealthStatus']=='Healthy' and vol['SizeRemaining']>1024**3
 dest.mkdir(parents=True,exist_ok=True)
 assert not any((dest/(name+s)).exists() for s in ('.stdout','.stderr','.json','_EXIT.json')) and not (H/(name+'_REF.json')).exists(),'Attempt already exists; no retry'
 start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 try:
  r=subprocess.run(cmd,input=payload,capture_output=True,timeout=timeout);out,err,code=r.stdout,r.stderr,r.returncode;timed=False
 except subprocess.TimeoutExpired as e:out,err,code,timed=e.stdout or b'',e.stderr or b'',None,True
 with (dest/(name+'.stdout')).open('xb') as f:f.write(out)
 with (dest/(name+'.stderr')).open('xb') as f:f.write(err)
 result=dict(started_utc=start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=code,timeout=timed,stdout_sha256=sha(out),stderr_sha256=sha(err),retry_authorized=False)
 save(dest/(name+'_EXIT.json'),result)
 save(H/(name+'_REF.json'),dict(path=str(dest/(name+'.stdout')),sha256=sha(out),bytes=len(out),stderr_path=str(dest/(name+'.stderr')),exit=result))
 assert code==0 and not timed,'Remote command failed/timed out; evidence preserved, stop'
 value=json.loads(out)
 if not bulk:save(H/(name+'.json'),value)
 return dict(value=value,ref=str(H/(name+'_REF.json')))
