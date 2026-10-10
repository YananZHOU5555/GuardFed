"""One bounded local metadata check invocation; preserve stdout/stderr and failure."""
from pathlib import Path
import hashlib,json,subprocess,sys
H=Path(__file__).resolve().parent
command=[sys.executable,'-B',str(H/'check_prepared.py')]
with (H/'SELF_CHECK_STDOUT.json').open('xb') as out,(H/'SELF_CHECK_STDERR.txt').open('xb') as err:
 result=subprocess.run(command,stdout=out,stderr=err,check=False)
record={'command':command,'returncode':result.returncode,'CNN':False,'SSH':False,'real_approval_created':False,'checker_sha256':hashlib.sha256((H/'check_prepared.py').read_bytes()).hexdigest()}
with (H/'SELF_CHECK_COMMAND.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.write('\n')
if result.returncode:
 with (H/'SELF_CHECK_FAILURE.json').open('x',encoding='utf-8') as f:json.dump(record|{'stderr':(H/'SELF_CHECK_STDERR.txt').read_text()},f,indent=2);f.write('\n')
 raise SystemExit(result.returncode)
report=json.loads((H/'SELF_CHECK_STDOUT.json').read_bytes())
with (H/'SELF_CHECK.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'status':report['status'],'refusals':report['refusal_count'],'worker_prebind':report['worker_original_pre_bind_path_reached'],'science_seal':report['science_seal'],'execution_seal':report['execution_seal']}))
