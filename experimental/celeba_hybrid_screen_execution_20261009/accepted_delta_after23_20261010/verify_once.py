from pathlib import Path
import subprocess,json,datetime,hashlib
B=Path(__file__).resolve().parent
for name in ['verify_offserver.py','run_record_checks.py']:
 start=datetime.datetime.now(datetime.timezone.utc).isoformat();r=subprocess.run(['python','-B',str(B/name)],capture_output=True)
 stem=Path(name).stem
 for suffix,data in [('STDOUT.txt',r.stdout),('STDERR.txt',r.stderr)]:
  with (B/(stem+'_'+suffix)).open('xb') as f:f.write(data)
 with (B/(stem+'_COMMAND_RECEIPT.json')).open('x') as f:json.dump(dict(start=start,end=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=r.returncode,source_sha256=hashlib.sha256((B/name).read_bytes()).hexdigest(),automatic_retry=False),f,indent=2);f.write('\n')
 print(r.stdout.decode(errors='replace'));print(r.stderr.decode(errors='replace'))
 assert r.returncode==0,'Preserve failure; no automatic retry'
