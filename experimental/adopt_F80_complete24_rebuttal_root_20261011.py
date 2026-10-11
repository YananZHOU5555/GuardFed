"""Adopt the complete reply draft after exact source and editorial checks."""
from pathlib import Path
import copy, datetime, hashlib, json, math, re, shutil
R=Path(__file__).resolve().parents[1]
H=R/'tmp/rebuttal_F80_editorial_prepare_20261011'
REV=R/'docs/server_deployment_20260923/revision_20260923'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(H/'FILES_SHA256.json')=='cb7dbdf5b7b33c71407ce1e30f8ac01b3ee5b1fff1f3f914f809cb56a7819083'
for n,pin in read(H/'FILES_SHA256.json')['files'].items():
 p=H/n
 assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],n
for rel,pin in read(H/'SOURCE_PINS.json').items():
 p=R/rel
 assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],rel
parent=REV/'rebuttal_working_F70_FL90_Hybrid20_release_20261011.md'
candidate=H/'rebuttal_working_F80_FL90_Hybrid20_20261011.md'
assert sha(parent)=='0b300bd7b72001f8804d34da4aab824e89ed1285fc78209a2609b9ef03fc1b80'
assert sha(candidate)=='a4fd92d065eb78e4cf325dd87f6fa33168380cb4453ce73a62a9015a4d3f795f'
before=parent.read_bytes().decode('utf8');actual=candidate.read_bytes().decode('utf8')
edits=read(H/'EDITS.json');assert len(edits)==8;s=before
for e in edits:
 assert e['count']==1 and s.count(e['old'])==1
 s=s.replace(e['old'],e['new'])
assert s==actual
for e in reversed(edits):
 assert s.count(e['new'])==1
 s=s.replace(e['new'],e['old'])
assert s==before
quotes=lambda t:[x for x in t.splitlines() if x.startswith('>')]
assert quotes(actual)==quotes(before) and len(quotes(actual))==38
assert len(re.findall(r'(?m)^>(?:[^\n]*)(?:\n>[^\n]*)*',actual))==24
assert [x for x in actual.splitlines() if x.startswith('### R')]==[x for x in before.splitlines() if x.startswith('### R')]
def pointed(row):
 v=read(R/row['source'])
 for k in row['pointer'].strip('/').split('/'):
  k=k.replace('~1','/').replace('~0','~')
  v=v[int(k)] if isinstance(v,list) else v[k]
 return v
bindings=read(H/'NUMERICAL_BINDINGS.json')
assert {k:len(v) for k,v in bindings.items()}=={'display_cells':18,'scalars':36,'all_nine_panel_directions':27,'scope_facts':9}
for row in bindings['scalars']+bindings['scope_facts']:
 v=pointed(row);assert v==row['value'],row
 if isinstance(v,float):assert math.isfinite(v) and v.hex()==row['value'].hex()
for row in bindings['all_nine_panel_directions']:
 v=pointed(row);assert v==row['value'] and (1 if v>0 else -1 if v<0 else 0)==row['sign'],row
for row in bindings['display_cells']:
 v=pointed(row);places=3 if row['unit']=='percentage points' else 5
 display=f"{v['mean']:+.{places}f} ± {v['sample_sd_ddof1']:.{places}f}"
 assert display==row['formatted'] and display in actual,row
prior=read(R/'tmp/rebuttal_FL90_editorial_prepare_20261011/ROOT_EDITORIAL_ADOPTION.json')
cutoff=copy.deepcopy(prior['fixed_scientific_content_cutoff'])
cutoff.update(mechanism_native=380,mechanism_three_view=380,F_pairs=80)
out=REV/'rebuttal_working_F80_FL90_Hybrid20_release_20261011.md'
proof_path=H/'ROOT_EDITORIAL_ADOPTION.json';assert not out.exists() and not proof_path.exists()
shutil.copyfile(candidate,out);assert out.read_bytes()==candidate.read_bytes()
proof=dict(status='ROOT_EDITORIAL_ADOPTED_COMPLETE24_F80_FL90_HYBRID20_AUTHOR_REVIEW_DRAFT',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root_accepted=True,candidate_path=out.relative_to(R).as_posix(),candidate_sha256=sha(out),agent_candidate_sha256=sha(candidate),parent_release_path=parent.relative_to(R).as_posix(),parent_release_sha256=sha(parent),source_only_seal_sha256=sha(H/'FILES_SHA256.json'),editorial_verification_sha256=sha(H/'CHECK_RESULTS.json'),semantic_diff_sha256=sha(H/'EDITS.json'),full_agent_and_root_inverse_exact=True,quote_blocks=24,quote_lines=38,quote_order_exact=True,new_saved_display_cells_checked=bindings['display_cells'],saved_scalars_checked=36,all_panel_direction_pointers_checked=27,new_statistics_computed=0,fixed_scientific_content_cutoff=cutoff,gradient14_completed_as_deployment_only=True,gradient14_scientific_table_records=0,NGA96_started_only=True,NGA96_scientific_accepted=0,NGA96_complete_claimed=False,author_review_draft=True,manuscript_applied=False,primary_endpoint_selected=False,final_test=False,whole_rebuttal_complete=False)
with proof_path.open('x',encoding='utf8',newline='\n') as f:json.dump(proof,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
print(json.dumps(dict(draft=out.relative_to(R).as_posix(),sha256=sha(out),root_sha256=sha(proof_path))))
