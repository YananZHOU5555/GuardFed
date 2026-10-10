from pathlib import Path
import json,hashlib,difflib,subprocess,sys,datetime
R=Path.cwd();H=R/'tmp/celeba_flgmm_fullcoverage_delta_after11_20261010';B=R/'tmp/celeba_flgmm_fullcoverage_incremental_20261009';O=R/'tmp/celeba_flgmm_fullcoverage_delta_after9_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes())
latest=read(B/'LATEST_BACKUP.json');assert latest['accepted_total']==11 and latest['next_collector_previous_sha256']=='40e14dc03414462e091f14c7f3d1241fb11ca1766cbb54479f3bbc459f9de7f2'
assert sha(R/latest['next_collector_previous_path'])==latest['next_collector_previous_sha256'];assert sha(R/latest['root_adoption_path'])==latest['root_adoption_sha256']=='6feb41c9f2f06980d29865ca03d59e5d2cffeeb2a6f0d6f0209f065cf80caf80'
(H/'PREVIOUS_LATEST.json').write_bytes((B/'LATEST_BACKUP.json').read_bytes());(H/'PREVIOUS_OFFSERVER_ACCEPTANCE.json').write_bytes((R/latest['next_collector_previous_path']).read_bytes())
s=(B/'collect_delta.py').read_text();marker="    if not wanted:\n"
assert s.count(marker)==1
addition="    # Root-authorized exact1 only; later terminal IDs remain in the full snapshot, unaccepted.\n    authorized_ids=['FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed91003_fullcoverage']\n    wanted=[identity for identity in wanted if identity in authorized_ids]\n"
t=s.replace(marker,addition+marker);(H/'collect_delta.py').write_text(t,encoding='utf8',newline='\n')
(H/'COLLECTOR_SCOPE_DIFF.patch').write_text(''.join(difflib.unified_diff(s.splitlines(True),t.splitlines(True),fromfile='original/collect_delta.py',tofile='exact1/collect_delta.py')),encoding='utf8')
(H/'verify_delta_offserver.py').write_bytes((B/'verify_delta_offserver.py').read_bytes());assert sha(H/'verify_delta_offserver.py')==sha(B/'verify_delta_offserver.py')
assert t.replace(addition,'')==s
(H/'SOURCE_REUSE.json').write_text(json.dumps(dict(original_collector_sha256=sha(B/'collect_delta.py'),actual_collector_sha256=sha(H/'collect_delta.py'),verifier_sha256=sha(H/'verify_delta_offserver.py'),original_strict_and_archive_body_unchanged=True,sole_change='Restrict fresh snapshot terminal difference to root exact one authorized ID; full observed snapshot retained',authorized_ids=['FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed91003_fullcoverage'],previous_root=latest['root_adoption_sha256'],previous_offserver=latest['next_collector_previous_sha256']),indent=2)+'\n',encoding='utf8')
(H/'preflight.py').write_bytes((O/'preflight.py').read_bytes())
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B -']
r=subprocess.run(cmd,input=(H/'preflight.py').read_bytes(),capture_output=True,timeout=60)
(H/'PREFLIGHT.json').write_bytes(r.stdout);(H/'PREFLIGHT_STDERR.txt').write_bytes(r.stderr);(H/'PREFLIGHT_COMMAND.json').write_text(json.dumps(dict(command=cmd,returncode=r.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())),encoding='utf8');r.check_returncode()
p=read(H/'PREFLIGHT.json');print(json.dumps(dict(UTC=p['utc'],CPU106=p['CPU106_free'],scope=p['scope_jobs'],workers=len(p['actual_workers']),service=p['service'],identity=p['source_data_verified'])))
