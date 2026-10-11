from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,re,shutil
R=Path(__file__).resolve().parents[1]
B=R/'tmp/rebuttal_F60_FL81_gradient64_update_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
check=B/'EDITORIAL_VERIFICATION.json'
assert sha(check)=='72ddcb4aa8f4c723a8079ea66b8bffe46b9e9a624ee14ab271bd6dc8e8161bf1'
v=read(check);parent=R/v['parent_path'];candidate=R/v['candidate_path']
assert sha(parent)==v['parent_sha256']=='879eabe84687f498d56ebd3dabbaed52d7bb5b8aa7eeeda191f46649a952fbef'
assert sha(candidate)==v['candidate_sha256']=='e76df8e53b7379a437bf09426c4b576443022f7766ae77d4ff444c456fd650b6'
assert sha(B/'EDITS.json')=='8bcf497b2c53ae0df178bf58e61e107adb971532da1433e952124e83490df9a0'
old=parent.read_text('utf8');text=old;edits=read(B/'EDITS.json')
assert len(edits)==11
for e in edits:
    assert e['count']==text.count(e['old'])==1
    text=text.replace(e['old'],e['new'],1)
assert text.encode()==candidate.read_bytes()
reverse=text
for e in reversed(edits):
    assert reverse.count(e['new'])==1
    reverse=reverse.replace(e['new'],e['old'],1)
assert reverse.encode()==parent.read_bytes()
blocks=lambda b:re.findall(rb'(?:^>[^\r\n]*(?:\r?\n|$))+',b,re.M)
assert blocks(parent.read_bytes())==blocks(candidate.read_bytes())
assert len(blocks(candidate.read_bytes()))==24
assert sum(len(q.splitlines()) for q in blocks(candidate.read_bytes()))==38
head=lambda b:[line for line in b.splitlines() if line.startswith(b'#')]
assert head(parent.read_bytes())==head(candidate.read_bytes())
for value in v['source_pins'].values():assert sha(R/value['path'])==value['sha256']
tables=read(R/v['source_pins']['F60_tables']['path'])
def pointer(root,path):
    for k in path.strip('/').split('/'):root=root[int(k)] if isinstance(root,list) else root[k]
    return root
for item in v['displayed_new_paired_mean_SD_pairs']:
    row=pointer(tables,item['pointer'])
    assert (row['mean'],row['sample_sd_ddof1'])==(item['mean'],item['sample_sd'])
    decimals=3 if item['pointer'].endswith('/accuracy_pct') else 5
    assert item['display']==f"{row['mean']:+.{decimals}f} ± {row['sample_sd_ddof1']:.{decimals}f}" and item['display'] in text
assert len(v['displayed_new_paired_mean_SD_pairs'])==18
for item in v['nine_actual_panel_direction_bindings']:
    row=pointer(tables,item['pointer'])
    assert (row['distribution'],row['attack'],row['variant'],row['n'])==('non-IID','Benign','minus_F minus Full',item['n'])
    assert [1 if row[m]['mean']>0 else -1 if row[m]['mean']<0 else 0 for m in ('accuracy_pct','aeod','aspd')]==item['minusF_minusFull_signs']
assert v['new_environment_binding']['replay_devices']==tables['replay_devices']
assert v['new_environment_binding']['training_torch']==tables['training_torch']
for name in re.findall(r'\]\((E:/[^)]+)\)',text):assert Path(name).is_file(),name
dest=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_working_F60_FL81_gradient64_20261011.md'
assert not dest.exists() and not (B/'ROOT_EDITORIAL_ADOPTION.json').exists()
shutil.copyfile(candidate,dest);assert sha(dest)==sha(candidate)
proof=dict(status='ROOT_UNIFIED_COMPLETE24_F60_FL81_GRADIENT64_WORKING_REBUTTAL_ADOPTED_FOR_AUTHOR_REVIEW',utc=datetime.now(timezone.utc).isoformat(),draft_path=dest.relative_to(R).as_posix(),draft_sha256=sha(dest),editorial_proof_sha256=sha(check),parent_sha256=sha(parent),source_pins=v['source_pins'],full11_edit_reconstruction_and_inverse_bytes_exact=True,quote_blocks=24,quote_lines=38,headings_order_exact=True,eighteen_mean_SD_pairs_from_saved_JSON_exact=True,nine_panel_directions_retained=True,content_cutoff=v['content_cutoff'],accepted_native=360,accepted_three_view=360,F_complete_scenes=6,Git_snapshot='3edbd954d91bbc5e4993546950202645d82ccbb0',Git_fixed_cutoff=v['Git64_fixed_cutoff'],current_draft_not_in_fixed_commit=True,scientific_checks_recomputed=0,models_arrays_read=0,source_adopter_path=Path(__file__).relative_to(R).as_posix(),source_adopter_sha256=sha(__file__),final_test=False,submitted_manuscript_edited=False,whole_rebuttal_complete=False)
with (B/'ROOT_EDITORIAL_ADOPTION.json').open('x',encoding='utf8',newline='\n') as f:json.dump(proof,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(status=proof['status'],draft_sha256=sha(dest),root_sha256=sha(B/'ROOT_EDITORIAL_ADOPTION.json'))))
