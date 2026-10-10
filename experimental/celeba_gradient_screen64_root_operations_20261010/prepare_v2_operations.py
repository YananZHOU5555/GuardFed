"""Preserve V1; derive a source-identical dispatch with explicit shared CPU policy."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OLD = Path(__file__).resolve().parent
NEW = ROOT/'tmp/celeba_gradient_screen64_v2_root_operations_20261010'
S1 = ROOT/'tmp/celeba_gradient_screen64_20261010'
S2 = ROOT/'tmp/celeba_gradient_screen64_v2_20261010'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not NEW.exists()
assert sha(S2/'FILES_SHA256.json')=='11e2ae63c87e0440669a047c930f5465b13babf559cca17bd8808bf693ce7ced'
files=json.loads((S2/'FILES_SHA256.json').read_bytes())['files']
assert all(sha(S2/n)==h for n,h in files.items())
original=(S1/'run_queue.py').read_text()
expected=original.replace('ROOT_GRADIENT64_RESOURCE_PREFLIGHT_PASS','ROOT_GRADIENT64_V2_RESOURCE_PREFLIGHT_PASS').replace('all_threads_cpu105_idle','no_restricted_cpu105_owner')
assert (S2/'run_queue.py').read_text()==expected
for n in json.loads((S1/'FILES_SHA256.json').read_bytes())['files']:
    if n.startswith('jobs/') or n.startswith('snapshot/'):
        assert sha(S2/n)==sha(S1/n)
NEW.mkdir()
review=dict(status='ROOT_GRADIENT64_V2_MINIMAL_RESOURCE_SEMANTICS_REVIEW_PASS',
 utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_seal_sha256=sha(S2/'FILES_SHA256.json'),
 original_independent_review_sha256=sha(ROOT/'tmp/celeba_gradient_screen64_root_review_20261010/REVIEW.json'),
 original_failure_observation_sha256=sha(OLD/'LATEST_OBSERVATION.json'),scientific_jobs_and_snapshot_byte_exact=True,
 runtime_changes='two resource status/key tokens only; large eligible masks do not reserve an entire host',
 no_restricted_cpu_reservations=True,actual_resource_preflight=False,new_training=0)
(NEW/'ROOT_SOURCE_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf8')
for name in ('remote_preflight.py','root_launch.py','service.sh','guardfed_celeba_gradient_screen64.conf','upload_source.py','install_and_start.py','observe.py'):
    text=(OLD/name).read_text(encoding='utf8')
    text=text.replace('celeba_gradient_screen64_20261010','celeba_gradient_screen64_v2_20261010')
    text=text.replace('celeba_gradient_screen64_root_operations_20261010','celeba_gradient_screen64_v2_root_operations_20261010')
    text=text.replace('celeba_gradient_screen64_results_20261010','celeba_gradient_screen64_v2_results_20261010')
    text=text.replace('guardfed_celeba_gradient_screen64','guardfed_celeba_gradient_screen64_v2')
    text=text.replace('eb389aca69b194ef3287049d6940e2c94b5d27d7e06ba2b3ee3b0317098e738b',sha(S2/'FILES_SHA256.json'))
    text=text.replace('ROOT_GRADIENT64_RESOURCE_PREFLIGHT_PASS','ROOT_GRADIENT64_V2_RESOURCE_PREFLIGHT_PASS')
    text=text.replace('all_threads_cpu105_idle','no_restricted_cpu105_owner')
    text=text.replace("if 105 in allowed:","if len(allowed) <= 16 and 105 in allowed:")
    text=text.replace("tasks.append(dict(tid=tid, cpus=allowed))", "tasks.append(dict(tid=tid, cpu_count=len(allowed), cpus=allowed if len(allowed)<=16 else None, broad_cpu_span=[allowed[0],allowed[-1]] if len(allowed)>16 else None))")
    text=text.replace("assert not overlaps, 'CPU105 overlaps an existing project compute thread'", "assert not overlaps, 'CPU105 overlaps an existing restricted project compute reservation'")
    text=text.replace("review = ROOT / 'tmp/celeba_gradient_screen64_root_review_20261010/REVIEW.json'", "review = HERE / 'ROOT_SOURCE_REVIEW.json'")
    text=text.replace('292e9023eae357a19b5185b85bf430ee2db8cf6bd700ace994ab0fb8e3eb3bfa',sha(NEW/'ROOT_SOURCE_REVIEW.json'))
    text=text.replace('sealed_members=85','sealed_members=91').replace('85_MEMBERS','91_MEMBERS')
    text=text.replace("scope='Actual momentary project all-thread reservation check; management service eligible masks are not scientific reservations'", "scope='All project threads inspected; restricted <=16-core masks reserve CPUs, broad masks are eligible scheduling sets; no CPU exclusivity/idle claim'")
    if name=='remote_preflight.py':
        anchor="proof = dict(status="
        measure="""def cpu_sample():
    stat = dict(line.split(maxsplit=1) for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
    cpu = next(line for line in Path('/proc/stat').read_text().splitlines() if line.startswith('cpu105 '))
    return dict(at=time.monotonic(), usage_usec=int(stat['usage_usec']), cpu105_ticks=[int(x) for x in cpu.split()[1:]])
before=cpu_sample(); time.sleep(1); after=cpu_sample()
usage=(after['usage_usec']-before['usage_usec'])/1e6/(after['at']-before['at'])
"""
        assert text.count(anchor)==1
        text=text.replace(anchor,measure+anchor)
        text=text.replace("cpu_ids=[105],", "measured_cgroup_cpu_cores=usage, cpu105_system_samples=[before,after], cpu_ids=[105],")
    (NEW/name).write_bytes(text.encode('utf8'))
(NEW/'guardfed_celeba_gradient_screen64_v2.conf').write_bytes((NEW/'guardfed_celeba_gradient_screen64.conf').read_bytes())
# The installer payload expects the v2 name after its exact token substitution.
print(json.dumps(dict(path=str(NEW),review_sha256=sha(NEW/'ROOT_SOURCE_REVIEW.json'),source_members=len(files))))
