from pathlib import Path
import json,hashlib,difflib
R=Path.cwd();H=R/'tmp/FL100_cuda3_saved_closure_prepare_20261011';P=R/'tmp/old800_cuda3_saved_audit_prepare_20261011';frozen=json.loads((H/'SELECTED3_INPUTS.json').read_bytes());IDS=frozen['ids'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def derive(name,fn):
 raw=(P/name).read_bytes();s=raw.decode();spans=[]
 def edit(a,b):
  nonlocal s
  assert a!=b and s.count(a)==1,(name,a[:85],s.count(a));s=s.replace(a,b,1);spans.append(dict(before=a,after=b))
 def between(a,b,v):edit(s[s.index(a):s.index(b,s.index(a))],v)
 fn(edit,between,s);restored=s
 for z in reversed(spans):
  assert restored.count(z['after'])==1,z['after'][:80];restored=restored.replace(z['after'],z['before'],1)
 assert restored.encode()==raw;compile(s,name,'exec');(H/name).write_bytes(s.encode())
 (H/(name+'.inverse.json')).write_text(json.dumps(dict(parent={'path':str((P/name).relative_to(R)),'sha256':sha(P/name)},spans=spans,full_raw_inverse_exact=True,derived_sha256=sha(H/name)),indent=2)+'\n')
 (H/(name+'.patch')).write_text(''.join(difflib.unified_diff(raw.decode().splitlines(True),s.splitlines(True),fromfile='old800/'+name,tofile='FL_fixed3/'+name)),encoding='utf8')
def transport(edit,between,s):
 edit("H=R/'tmp/old800_cuda3_saved_audit_prepare_20261011'","H=R/'tmp/FL100_cuda3_saved_closure_prepare_20261011'")
 edit("inp['base']==template['base']","inp['base']==json.loads((H/'SELECTED3_INPUTS.json').read_bytes())['planned_namespace']")
 edit("F:/YananResearchStorage/GuardFed/uniform_cuda_old800_20261011/canary3_attempt002","F:/YananResearchStorage/GuardFed/uniform_cuda_FL100_20261011/canary3_attempt001")
 between("B=Path('/workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt002')","members={n:{'sha256':sha(p)",(H/'transport_metadata_body.py.txt').read_text('utf-8-sig'))
 edit("code=code.replace('__ROOT_INPUTS__',base64.b64encode(json.dumps(inp,allow_nan=False).encode()).decode())","code=code.replace('__ROOT_INPUTS__',base64.b64encode(json.dumps(inp,allow_nan=False).encode()).decode()).replace('__FROZEN_INPUTS__',base64.b64encode((H/'SELECTED3_INPUTS.json').read_bytes()).decode())")
def audit(edit,between,s):
 edit(next(l for l in s.splitlines() if l.startswith('BASE=')), 'BASE='+repr(frozen['planned_namespace']))
 edit(next(l for l in s.splitlines() if l.startswith('IDS=')), 'IDS='+repr(IDS))
 edit(next(l for l in s.splitlines() if l.startswith('FORMAL_API_SEAL_SHA=')), 'FORMAL_API_SEAL_SHA='+repr(frozen['formal_seal']))
 edit("need(inp['root_authorized_audit'] is True and inp['exact_ids']==IDS,'Root audit binding absent or wrong fixed3')","need(inp['root_authorized_audit'] is True and inp['exact_ids']==IDS and inp['base']==BASE,'Root audit binding absent or wrong fixed3')")
 between('        def remote_pin(pin):',"        metadata=member('source/metadata.npz')", "        from metadata_gate import verify_metadata\n        frozen=read(HERE/'SELECTED3_INPUTS.json')\n        need(sha(HERE/'SELECTED3_INPUTS.json')==read(HERE/'SOURCE_PINS.json')['frozen_selection_sha256'],'Frozen FL selection changed')\n        verified=verify_metadata(inp,x,members,frozen);cohort_sha=frozen['cohort_sha256']\n")
 edit("METHODS={'FedAvg','FedAA','LASA'}","METHODS={'FLGMM'}")
 between("            record=next(r for r in selection['records'] if r['id']==identity)","            fits=saved_fits_for_original_predict", "            item=verified[identity];record=item['record'];r=item['r'];receipt_path=item['receipt_path'];array_path=item['array_path'];accepted_path=item['accepted_path'];run=item['run']\n")
 edit("saved_fits_for_original_predict(r['fits'],ev,record['method'])","saved_fits_for_original_predict(r['fits'],ev)")
 edit("status='OLD800_FIXED3_SAVED_OUTPUTS_AUDIT_NO_REFIT_PASS_NOT_ROOT_ADOPTED'","status='FL100_FIXED3_CUDA_SAVED_OUTPUTS_AUDIT_NO_REFIT_PASS_NOT_ROOT_ADOPTED'")
 edit("uniform800_complete=False","uniform_FL100_complete=False")
 edit("scope='Fixed3 only; original Linux whole plus SHA transport and saved-output consistency. No fit/model/CNN/test/cohort completion.'","scope='FL fixed3 only; original Linux whole plus SHA transport and zero-fit saved-output consistency. No CNN/fit/test/FL100-completion claim.'")
derive('transport.py',transport);derive('audit_saved.py',audit)
print(json.dumps({'transport':sha(H/'transport.py'),'audit':sha(H/'audit_saved.py')}))
