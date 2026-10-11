"""Source-only exact-scene adaptation. No SSH, F drive, inference, fit or acceptance."""
from pathlib import Path
import ast, copy, difflib, hashlib, io, json, re, tokenize

H=Path(__file__).resolve().parent;R=H.parents[1]
P=R/'tmp/mechanism_F_nonIID_FFlip10_saved_binding_prepare_20261011'
T=R/'tmp/mechanism_F_seven_scenes70_table_prepare_20261011'
PR=R/'tmp/celeba_mechanism_remaining_F_nonIID_FFlip10_root_adoption_20261011/ROOT_ADOPTION.json'
OLD=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_seven_scenes70_20261011'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
IDS=[f'minus_F_non-IID_FedSA_seed{s}' for s in range(91001,91011)]

def save(n,v):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def substitute(s,m):
 return re.sub('|'.join(re.escape(k) for k in sorted(m,key=len,reverse=True)),lambda x:m[x.group()],s)

def numbers(s,m):
 lines=s.splitlines(keepends=True);starts=[0]
 for line in lines:starts.append(starts[-1]+len(line))
 edits=[]
 for t in tokenize.generate_tokens(io.StringIO(s).readline):
  if t.type==tokenize.NUMBER and t.string in m:edits.append((starts[t.start[0]-1]+t.start[1],starts[t.end[0]-1]+t.end[1],m[t.string]))
 for a,b,v in reversed(edits):s=s[:a]+v+s[b:]
 return s

adapt={};diff=[]
def emit(n,parent,s):
 raw=parent.read_bytes();new=s.encode();compile(new,str(H/n),'exec')
 spans=[dict(a=a,b=b,c=c,d=d,old=raw[a:b].hex(),new=new[c:d].hex()) for tag,a,b,c,d in difflib.SequenceMatcher(None,raw,new,autojunk=False).get_opcodes() if tag!='equal']
 undo=new
 for e in reversed(spans):assert undo[e['c']:e['d']].hex()==e['new'];undo=undo[:e['c']]+bytes.fromhex(e['old'])+undo[e['d']:]
 assert undo==raw
 with (H/n).open('xb') as f:f.write(new)
 adapt[n]=dict(parent_path=parent.relative_to(R).as_posix(),parent_sha256=sha(parent),sha256=sha(H/n),raw_inverse_spans=spans)
 diff.extend(difflib.unified_diff(raw.decode().splitlines(True),s.splitlines(True),fromfile=str(parent.relative_to(R)),tofile=str((H/n).relative_to(R))))

assert not (H/'PREPARED.json').exists()
root=read(PR);oldproof=read(OLD/'ROOT_VERIFICATION.json');rawindex=read(P/'RAW_STORAGE_INDEX.json')
assert sha(PR)=='a69fd80942b585a89a64d83b306ec121a1d300957c91d321eb3abf1340c20781'
assert root['cumulative_accepted']==370 and oldproof['paired_models']==70 and oldproof['original_fsum_verification']['mean_sd_scalars']==1134
assert oldproof['original_fsum_verification']['display_cells']==567
native=R/root['native_root_path'];assert sha(native)=='4ae0a66ca2cc48289b9a7a18272b83652c73c9da16b2d65989ecd5a1166d002e'
previous_remote=read(P/'EXPORT_COMMAND.json')['tag']
previous_remote='/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010/exports/'+previous_remote+'/backup_receipt.json'
saved_names=['accepted_scope.py','verify_native_inputs.py','bind_native.py','join_saved.py','finalize_actual_handoff.py','adopt_root.py',
             'execute_once.py','export_once.py','download_verify_once.py','remote_preflight.py','remote_cpu111_export.py','cpu_release_readonly.py']
for n in saved_names:
 s=(P/n).read_bytes().decode()
 # All integer substitutions are cohort cardinalities in these non-mathematical orchestration files.
 if n not in ['remote_cpu111_export.py','cpu_release_readonly.py','remote_preflight.py','export_once.py','execute_once.py']:
  s=numbers(s,{'360':'370','370':'380','460':'470','470':'480','50':'51','51':'52','180':'190','190':'200'})
 m={P.name:H.name,'celeba_mechanism_remaining_F_nonIID_FFlip10_root_adoption_20261011':'celeba_mechanism_remaining_F_nonIID_FedSA10_root_adoption_20261011',
    'minus_F_non-IID_F Flip_seed':'minus_F_non-IID_FedSA_seed','F_NONIID_FFLIP':'F_NONIID_FEDSA',
    'native360':'native370','prior360':'prior370','old360':'old370','original360':'original370',
    'old460':'old470','old50':'old51','MECHANISM370':'MECHANISM380','REPLAY370':'REPLAY380','MIN370':'MIN380',
    '>=370':'>=380','three-view370':'three-view380',
    '16299ca4ade394a1b89edb3f9230845cda7ce826c783645b9aada4ea359a5c37':root['records_index_sha256'],
    'ca558d6e558d26afe6b172b65eded83b51ad02eedddc0f9e2d2cbadf7cdf5a11':sha(PR),
    'fa62be8e4d48990f041fdc50942406ace8f343286777bcf1dcd00379ed94c0f1':rawindex['receipt_sha256']}
 if n in ['remote_cpu111_export.py','cpu_release_readonly.py']:m={}
 s=substitute(s,m) if m else s
 if n in ['verify_native_inputs.py','adopt_root.py']:s=s.replace("row['attack']=='F Flip'","row['attack']=='FedSA'")
 if n=='adopt_root.py':
  s=s.replace("['non-IID','F Flip']],partial_F_scenes=","['non-IID','F Flip'],['non-IID','FedSA']],partial_F_scenes=")
  s=s.replace("parent_adopter_path='tmp/celeba_F_nonIID_Benign10_root_adopter_prepare_20261011/adopt_root.py',parent_adopter_sha256=sha(R/'tmp/celeba_F_nonIID_Benign10_root_adopter_prepare_20261011/adopt_root.py')",
              f"parent_adopter_path='tmp/{P.name}/adopt_root.py',parent_adopter_sha256=sha(R/'tmp/{P.name}/adopt_root.py')")
 if n=='download_verify_once.py':s=s.replace('tmp/celeba_mechanism_remaining_F_nonIID_Benign10_prepare_20261011/RAW_STORAGE_INDEX.json',f'tmp/{P.name}/RAW_STORAGE_INDEX.json')
 if n=='remote_preflight.py':
  ls=s.splitlines(keepends=True)
  for i,line in enumerate(ls):
   if line.startswith('candidate_ids='):ls[i]='candidate_ids='+repr(IDS)+'\n'
   if line.startswith("assert latest['receipt']=="):ls[i]="assert latest['receipt']=="+repr(previous_remote)+'\n'
   if line.startswith("assert latest['receipt_sha256']=="):ls[i]="assert latest['receipt_sha256']==sha(latest['receipt'])=="+repr(rawindex['receipt_sha256'])+'\n'
   if line.startswith("assert latest['all_transported_ids']=="):ls[i]="assert latest['all_transported_ids']=="+repr(rawindex['all_transported_ids'])+" and latest['accepted_offserver']==0\n"
  s=''.join(ls)
 if n=='accepted_scope.py':
  s=s.replace("    proof=read(e['native_root_path'])", "    prior=read(p['prior_root_path']);assert sha(p['prior_root_path'])==p['prior_root_sha256']\n    assert prior['accepted_native_scope_sha256']==p['native370_inspection_sha256']\n    assert (Path(p['prior_root_path']).parents[2]/prior['accepted_native_scope_path']).resolve()==Path(p['native370_inspection_path']).resolve()\n    assert sha(p['native370_raw_inspection_path'])==p['native370_raw_inspection_sha256']==prior['native_inspection_sha256']\n    proof=read(e['native_root_path'])")
  # Project root is a CLI-pinned path; do not assume root-adoption directory depth.
  s=s.replace("Path(p['prior_root_path']).parents[2]/prior['accepted_native_scope_path']", "Path(__file__).resolve().parents[2]/prior['accepted_native_scope_path']")
 if n=='verify_native_inputs.py':s=s.replace("'native370_inspection','native370_ledger'", "'native370_inspection','native370_raw_inspection','native370_ledger'")
 if n=='join_saved.py':
  s=s.replace("oldinspection=pin(previousroot.parent/'inspection/inspection.json',previousproof['inspection_sha256']);", "oldrawinspection=pin(previousroot.parent/'inspection/inspection.json',previousproof['inspection_sha256']);oldinspection=pin(P['native370_inspection_path'],P['native370_inspection_sha256']);")
 if n=='adopt_root.py':
  s=s.replace("old_inspection=read(old_native.parent/'inspection/inspection.json');", "old_raw_inspection=read(old_native.parent/'inspection/inspection.json');old_inspection=read(read(S/'PREPARED.json')['native370_inspection_path']);")
 s=s.replace('exact10 F non-IID F Flip','exact10 F non-IID FedSA').replace('exact10 F non-IID Benign export','exact10 F non-IID FedSA export')
 emit(n,P/n,s)

p=read(P/'PREPARED.json')
for suffix in ('root','inspection','ledger'):
 p.pop('native360_'+suffix+'_path');p.pop('native360_'+suffix+'_sha256')
 path=native if suffix=='root' else native.parent/('inspection/inspection.json' if suffix=='inspection' else 'verified_ledger.json')
 p['native370_'+suffix+'_path']=str(path);p['native370_'+suffix+'_sha256']=sha(path)
p['native370_raw_inspection_path']=p['native370_inspection_path'];p['native370_raw_inspection_sha256']=p['native370_inspection_sha256']
p['native370_inspection_path']=str(R/root['accepted_native_scope_path']);p['native370_inspection_sha256']=root['accepted_native_scope_sha256']
assert sha(p['native370_inspection_path'])==p['native370_inspection_sha256']
p.update(status='SOURCE_ONLY_FIXED_F_NONIID_FEDSA10_PENDING_ACTUAL_NATIVE380_AND_SAVED',candidate_ids=IDS,prior_accepted=370,prior_transported=190,
 prior_root_path=str(PR),prior_root_sha256=sha(PR),prior_index_path=str(R/root['records_index_path']),prior_index_sha256=root['records_index_sha256'],
 previous_local_receipt=rawindex['receipt'],previous_remote_receipt=previous_remote,previous_receipt_sha256=rawindex['receipt_sha256'],previous_all_transported_ids=rawindex['all_transported_ids'],
 required_new_native_candidate_ids=IDS,replay_target_total=380,native_target_minimum=380)
assert p['native_binding'] is p['required_native_root_sha256'] is p['required_native_root_path'] is None
save('PREPARED.json',p)

# F80 table preserves all seven accepted scenes and adds only non-IID FedSA10.
table_names=['binding.py','build.py','panels.py','verify_numeric.py','verify_saved.py','root_adopt_actual.py']
old6="('IID','Benign'),('IID','F Flip'),('IID','FedSA'),('IID','S-DFA'),('IID','Sp-DFA'),('non-IID','Benign')"
old7=old6+",('non-IID','F Flip')";new8=old7+",('non-IID','FedSA')"
for n in table_names:
 s=(T/n).read_bytes().decode()
 s=numbers(s,{'120':'140','140':'160','972':'1134','1134':'1296','486':'567','567':'648','1080':'1260','1260':'1440','2880':'3360','3360':'3840','360':'370','370':'380'})
 s=substitute(s,{old7:new8,old6:old7,
  'three_view_F_seven_scenes70_20261011':'three_view_F_eight_scenes80_20261011','three_view_F_six_scenes60_20261011':'three_view_F_seven_scenes70_20261011',
  'tmp/celeba_mechanism_remaining_F_nonIID_Benign10_root_adoption_20261011/ROOT_ADOPTION.json':PR.relative_to(R).as_posix(),
  'tmp/celeba_mechanism_remaining_F_nonIID_Benign10_root_adoption_20261011/MECHANISM360_INDEX.json':root['records_index_path'],
  'ca558d6e558d26afe6b172b65eded83b51ad02eedddc0f9e2d2cbadf7cdf5a11':sha(PR),
  '16299ca4ade394a1b89edb3f9230845cda7ce826c783645b9aada4ea359a5c37':root['records_index_sha256'],
  'cbba6a6cdfb9b861202422f867c847800474961cd02784dd207f87f51c386164':sha(OLD/'ROOT_VERIFICATION.json'),
  '497126f23be22533830d00389799c28238a4456382862d802a48c27cb2307b59':oldproof['files']['tables.json']['sha256'],
  'minus_F_non-IID_F Flip_seed':'minus_F_non-IID_FedSA_seed',
  'ACTUAL_REPLAY370_BOUND_FOR_F_SEVEN_SCENES70':'ACTUAL_REPLAY380_BOUND_FOR_F_EIGHT_SCENES80',
  'SEVEN_F_SCENE':'EIGHT_F_SCENE','SEVEN_SCENES':'EIGHT_SCENES','F70':'F80','F60':'F70',
  'actual_views370':'actual_views380','original360':'original370','parent360':'parent370',
  'old120':'old140','Old120':'Old140','old972':'old1134','Old972':'Old1134','old486':'old567',
  'Exact140':'Exact160','scene120':'scene140','scene140':'scene160',
  'replay370':'replay380','cover370':'cover380','>=370':'>=380',
  'Exact360+10':'Exact370+10','Old360':'Old370','Parent360':'Parent370','parent360':'parent370','Accepted360':'Accepted370','accepted360':'accepted370',
  'old F60':'old F70','Old six-scene':'Old seven-scene','Old six-scene972':'Old seven-scene1134',
  'non-IID F Flip10':'non-IID FedSA10','ten non-IID F Flip':'ten non-IID FedSA','new non-IID F Flip':'new non-IID FedSA',
  'seed/non-IID F Flip':'seed/non-IID FedSA','nonIID_FFlip_native10_rows':'nonIID_FedSA_native10_rows',
  'six F scenes':'seven F scenes','seven F scenes':'eight F scenes',
  'Benign/F Flip':'Benign/F Flip/FedSA','Benign and F Flip only':'Benign, F Flip and FedSA only',
  'The extra observed non-IID FedSA seed91001 is not adopted into this scope. ':'',
  'this seven-scene table':'this eight-scene table','other three non-IID':'other two non-IID','Three non-IID F scenes':'Two non-IID F scenes'})
 # Numeric 70 is also the terminal round: change only explicit table-cardinality fields.
 s=s.replace('complete_scenes=7','complete_scenes=8').replace('paired_models=70','paired_models=80').replace('preserved_pairs=70','preserved_pairs=80').replace('paired=70','paired=80')
 s=s.replace('==(7,70,160,160)','==(8,80,160,160)').replace('nonIID_F_scenes_remaining=3','nonIID_F_scenes_remaining=2')
 s=s.replace('len(rows)==21','len(rows)==24')
 s=s.replace("full['attack']==row['attack']=='F Flip'","full['attack']==row['attack']=='FedSA'")
 s=s.replace("==('non-IID','F Flip')","==('non-IID','FedSA')")
 s=s.replace('Old370 order or exact370','Old370 order or exact380').replace('Old seven-scene120','Old seven-scene140').replace('Exact seven-scene160','Exact eight-scene160').replace('Old1134 six-scene','Old1134 seven-scene').replace('ACTUAL_F80_SEVEN_SCENE','ACTUAL_F80_EIGHT_SCENE').replace('actual views370 root','actual views380 root')
 emit(n,T/n,s)

inputs=read(T/'SOURCE_INPUTS.json');inputs['C_panels']=(H/'panels.py').relative_to(R).as_posix();inputs['C_numeric']=(H/'verify_numeric.py').relative_to(R).as_posix();inputs['old_F70_root']=(OLD/'ROOT_VERIFICATION.json').relative_to(R).as_posix()
for path in [*(T/n for n in table_names),T/'FILES_SHA256.json',PR,R/root['records_index_path'],OLD/'ROOT_VERIFICATION.json',H/'panels.py',H/'verify_numeric.py']:
 inputs['files'][path.relative_to(R).as_posix()]=dict(sha256=sha(path),bytes=path.stat().st_size)
for n in ['records.json','tables.json','TABLES.md']:inputs['files'][(OLD/n).relative_to(R).as_posix()]=oldproof['files'][n]
save('SOURCE_INPUTS.json',inputs)
save('SOURCE_ADAPTATIONS.json',adapt)
with (H/'SOURCE_DIFF.patch').open('x',encoding='utf8',newline='') as f:f.write(''.join(diff))
save('ROOT_BINDING.template.json',dict(status='SOURCE_ONLY_NOT_BOUND',root_adopted=False,required_status='ACTUAL_REPLAY380_BOUND_FOR_F_EIGHT_SCENES80',adoption=None,adoption_sha256=None,index=None,index_sha256=None,native_root=None,native_root_sha256=None,root_status=None,source_seal_sha256=None))
actual=read(P/'ROOT_ACTUAL_PINS.template.json');actual.update(native_total=380,replay_total=380,source_adopter_sha256=sha(H/'adopt_root.py'))
save('ROOT_ACTUAL_PINS.template.json',actual)
save('HANDOFF.json',dict(status='SOURCE_ONLY_NOT_EXECUTED_NOT_ACCEPTED',fixed_ids=IDS,actual_parent_native=370,actual_parent_views=370,actual_parent_F=70,
 proposed_native=380,proposed_views=380,proposed_F=80,required_old_native470_row_tokens_exact=True,required_old_F70_scalars_exact=1134,required_old_F70_display_cells_exact=567,
 future_native_root=None,future_native_root_sha256=None,future_saved_root=None,future_saved_root_sha256=None,future_F80_root=None,
 SSH=False,F_access=False,scientific_execution=False,canonical_STATE_written=False,Git_mutations=0))
print(json.dumps(dict(status='SOURCE_ONLY_DERIVED',saved_sources=len(saved_names),table_sources=len(table_names),future_native380=None,future_saved380=None,future_F80=None)))
