"""Static byte/AST checks plus synthetic metadata rejection; no Torch, F, SSH or science."""
from pathlib import Path
import ast, copy, difflib, hashlib, importlib.util, json
H=Path(__file__).resolve().parent;R=H.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
adapt=read(H/'SOURCE_ADAPTATIONS.json');diff=[];checks={}
for n,pin in adapt.items():
 parent=R/pin['parent_path'];before=parent.read_bytes();after=(H/n).read_bytes()
 assert sha(parent)==pin['parent_sha256'];compile(after,str(H/n),'exec')
 spans=[dict(a=a,b=b,c=c,d=d,old=before[a:b].hex(),new=after[c:d].hex()) for tag,a,b,c,d in difflib.SequenceMatcher(None,before,after,autojunk=False).get_opcodes() if tag!='equal']
 restored=after
 for e in reversed(spans):assert restored[e['c']:e['d']].hex()==e['new'];restored=restored[:e['c']]+bytes.fromhex(e['old'])+restored[e['d']:]
 assert restored==before
 pin.update(sha256=sha(H/n),raw_inverse_spans=spans);checks[n]=dict(compiled=True,raw_inverse_exact=True,sha256=sha(H/n))
 diff.extend(difflib.unified_diff(before.decode().splitlines(True),after.decode().splitlines(True),fromfile=pin['parent_path'],tofile=str((H/n).relative_to(R))))
def blocks(p,needle):
 s=Path(p).read_bytes().decode();lines=s.splitlines(True)
 found=[n for n in ast.walk(ast.parse(s)) if isinstance(n,(ast.For,ast.With)) and needle in lines[n.lineno-1]]
 return [''.join(lines[n.lineno-1:n.end_lineno]) for n in found]
loopchecks={}
for name,needles in {'collect_delta.py':['for ID in ids:',"with tarfile.open(archive,'w:gz') as t:"], 'verify_delta.py':['for ID in ids:',"with tarfile.open(archive,'r:gz') as t:"]}.items():
 for needle in needles:
  old=blocks(R/adapt[name]['parent_path'],needle);new=blocks(H/name,needle)
  assert old and old==new,(name,needle)
  loopchecks[name+':'+needle]=dict(raw_exact=True,AST_exact=True,blocks=len(old),sha256=[hashlib.sha256(x.encode()).hexdigest() for x in old])
deps=read(H/'DEPENDENCIES.json')
for n,p in deps['files'].items():assert sha(R/n)==p['sha256'] and (R/n).stat().st_size==p['bytes']
contract=read(H/'SCOPE.json');package=R/contract['source_local']
new=(package/'snapshot/gradient_bridge_fullcoverage/accept_result.py').read_bytes()
old=(R/'tmp/celeba_gradient_screen64_v2_20261010/snapshot/gradient_bridge_20261010/accept_result.py').read_bytes()
assert sha(package/'snapshot/gradient_bridge_fullcoverage/accept_result.py')==contract['expected_checker_sha256']
# Preserve the historically frozen coverage-extension diff; this package does not edit that acceptor.
(H/'FROZEN_CHECKER_VS_SCREEN64.patch').write_text(''.join(difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),fromfile='accepted_screen64/accept_result.py',tofile='already_frozen_coverage/accept_result.py')),encoding='utf8',newline='')
spec=importlib.util.spec_from_file_location('pure_NGA_scope_only',H/'scope.py');scope=importlib.util.module_from_spec(spec);spec.loader.exec_module(scope)
auth=read(H/'AUTHORIZED_SNAPSHOT.template.json');tests={}
try:scope.validate(auth,contract,H)
except ValueError:tests['unbound_template_refused']=True
else:raise AssertionError('Unbound authorization accepted')
auth.update(status='ROOT_BOUND_EXACT_NGA96_DELTA_NOT_ACCEPTED',authorized_ids=contract['new_ids'][:2],observed_terminal_ids=contract['new_ids'][:2],observed_failure_paths=[],accepted_prior_ids=[],accepted_prior_count=0,snapshot_path='fixture_only',snapshot_sha256='0'*64,source_approval_path='fixture_only',gate14_adoption_path='fixture_only',gate14_adoption_sha256='0'*64)
for k in ['manifest_sha256','source_approval_sha256','source_seal_sha256','checker_sha256']:auth[k]=contract['expected_'+k]
prior,off=scope.validate(auth,contract,H);assert prior=={'accepted_ids':[],'accepted_total':0} and off['accepted_total']==0
for name,modify in [
 ('foreign_screen_ID',lambda d:d['authorized_ids'].__setitem__(0,contract['reused_screen4'][0]['id'])),
 ('duplicate_new_ID',lambda d:d['authorized_ids'].__setitem__(1,d['authorized_ids'][0])),
 ('not_terminal',lambda d:d['observed_terminal_ids'].pop()),
 ('failed_observation',lambda d:d['observed_failure_paths'].append('fixture_failure')),
 ('source_drift',lambda d:d.update(source_seal_sha256='0'*64)),
 ('wrong_prior_count',lambda d:d.update(accepted_prior_count=1)),
 ('invented_previous_root',lambda d:d.update(prior_root_sha256='0'*64)),
 ('test_requested',lambda d:d.update(test=True))]:
 d=copy.deepcopy(auth);modify(d)
 try:scope.validate(d,contract,H)
 except ValueError:tests[name]=True
 else:raise AssertionError(name)
for n in ['AUTHORIZED_SNAPSHOT.json','ROOT_REVIEW.json','OFFSERVER_ACCEPTANCE.json','DOWNLOAD.json','SAVED_TENSOR_STATE_CHECK.json']:assert not (H/n).exists()
for p in H.glob('*.py'):compile(p.read_bytes(),str(p),'exec')
(H/'SOURCE_ADAPTATIONS.json').write_text(json.dumps(adapt,indent=2)+'\n',encoding='utf8',newline='\n')
(H/'SOURCE_DIFF.patch').write_bytes(''.join(diff).encode())
report=dict(status='SOURCE_ONLY_RAW_INVERSE_ORIGINAL_SCIENTIFIC_LOOPS_AND_SYNTHETIC_SCOPE_PASS_NOT_ACCEPTANCE',files=checks,original_loops=loopchecks,synthetic_refusals=tests,frozen_checker_unmodified=True,new_scientific_results_verified=0,future_terminal_ids=None,future_checker_source_binding=None,future_archive=None,SSH=False,F_access=False,torch_imported=False,CNN=False,fit=False,training=False,STATE=False,Git=False)
(H/'SOURCE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8',newline='\n')
print(json.dumps(dict(status=report['status'],inverse_files=len(checks),original_loop_groups=len(loopchecks),synthetic_refusals=len(tests))))
