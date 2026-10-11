"""Authorize one compact descriptive build from four adopted native CUDA cohorts."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys
R=Path(__file__).resolve().parents[1];H=R/'tmp/celeba_uniform_cuda_fourmethod_partial_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def put(p,v):
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2);f.write('\n')
seal='4ffb108466144a3c619b149bf41b414b940a77174a5bc9359b8408d602850b14'
assert sha(H/'FILES_SHA256.json')==seal
for n,v in read(H/'FILES_SHA256.json')['files'].items():
    p=H/n;assert p.resolve().is_relative_to(H.resolve()) and sha(p)==v['sha256'] and p.stat().st_size==v['bytes']
assert read(H/'SOURCE_CHECK.json')['all_root_audit_transport_SHAs_and_native_record_links_verified']
binding=read(H/'ROOT_BINDING.template.json');assert binding['root_execution_authorized'] is False
for pin in binding['roots'].values():
    assert sha(pin['path'])==pin['sha256'];root=read(pin['path']);assert root['accepted']==100 and not root['test']
review=H/'ROOT_SOURCE_REVIEW.json'
put(review,dict(status='ROOT_SOURCE_REVIEW_PARTIAL_FOUR_METHOD_CUDA_NATIVE_TABLE_PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_adoptable=True,source_seal_sha256=seal,compact_JSON_inputs_only=True,native_root_audit_links_checked=True,metadata_only_old1100=True,mean_and_sample_SD_ddof1=True,ten_nine_six_seed_panels=True,cross_scenes_seed_first=True,all_outcomes_retained=True,new_primary_endpoint=False,final_comparison=False,statistics_executed=False))
binding.update(status='ROOT_AUTHORIZED_PARTIAL_FOUR_METHOD_UNIFORM_CUDA_NATIVE_TABLE',root_execution_authorized=True,root_source_review=dict(path=str(review),sha256=sha(review)))
ip=H/'ACTUAL_ROOT_BINDING.json';put(ip,binding)
cmd=[sys.executable,'-B',str(H/'build.py'),'--inputs',str(ip),'--inputs-sha256',sha(ip),'--source-seal-sha256',seal,'--output',str(H/'actual_partial400')]
result=None;error=None
try:
    with (H/'ROOT_BUILD.stdout').open('xb') as out,(H/'ROOT_BUILD.stderr').open('xb') as err:result=subprocess.run(cmd,cwd=R,stdout=out,stderr=err,timeout=120)
except BaseException as exc:error=repr(exc);raise
finally:put(H/'ROOT_BUILD_COMMAND.json',dict(command=cmd,exit_code=None if result is None else result.returncode,error=error,stdout_sha256=sha(H/'ROOT_BUILD.stdout'),stderr_sha256=sha(H/'ROOT_BUILD.stderr'),automatic_retry=False))
assert result.returncode==0,(H/'ROOT_BUILD.stderr').read_text()
print((H/'ROOT_BUILD.stdout').read_text())
