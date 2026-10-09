from pathlib import Path
import base64,hashlib,json,subprocess
p=Path('/workspace/guardfed_checks/celeba_flgmm_screen_20261009/accepted_delta_after19_20261009')
files={}
for f in p.iterdir():
 assert f.is_file(),str(f)
 data=f.read_bytes();files[f.name]=dict(sha256=hashlib.sha256(data).hexdigest(),size=len(data),base64=base64.b64encode(data).decode())
r=Path('/workspace/guardfed_checks/celeba_flgmm_screen_20261009/release_v2')
print('FAILURE_PRESERVATION_JSON='+json.dumps(dict(files=files,archive_exists=bool(list(p.glob('*.tar.gz'))),partial_acceptance_exists=(p/'PARTIAL_ACCEPTANCE.json').exists(),source_now={s:hashlib.sha256((r/s).read_bytes()).hexdigest() for s in ['PACKAGE_SHA256.json','source/protocol.json','jobs/manifest.json','source/accept_result.py','screen_common.py','frozen_score.py']},service=subprocess.run(['supervisorctl','status','guardfed_celeba_flgmm_screen'],capture_output=True,text=True).stdout.strip())))
