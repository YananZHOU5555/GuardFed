from pathlib import Path
import datetime,hashlib,json,re,shutil
R=Path(__file__).resolve().parents[1];H=R/'tmp/rebuttal_FL90_editorial_prepare_20261011';REV=R/'docs/server_deployment_20260923/revision_20260923'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_bytes())
assert sha(H/'FILES_SHA256.json')=='b655e69bf43fb70cc1866967ebac9746137ae308171bc053355d829c4083e98b'
seal=read(H/'FILES_SHA256.json');pins=seal.get('files',seal)
for n,row in pins.items():
 p=R/n if (R/n).is_file() else H/n
 assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
parent=REV/'rebuttal_working_F70_FL81_Hybrid20_release_20261011.md';candidate=REV/'rebuttal_working_F70_FL90_Hybrid20_candidate_20261011.md'
assert sha(parent)=='9be134d9d8fbaf592274c6c4229783f3fd0dbde058e8b8714a780e0f116c5363' and sha(candidate)=='0b300bd7b72001f8804d34da4aab824e89ed1285fc78209a2609b9ef03fc1b80'
before=parent.read_bytes().decode('utf8');actual=candidate.read_bytes().decode('utf8');s=before;edits=read(H/'SEMANTIC_DIFF.json')
assert len(edits)==10
for e in edits:assert s.count(e['old'])==1;s=s.replace(e['old'],e['new'])
assert s==actual
for e in reversed(edits):assert s.count(e['new'])==1;s=s.replace(e['new'],e['old'])
assert s==before
quotes=lambda text:[l for l in text.splitlines() if l.startswith('>')]
assert quotes(actual)==quotes(before) and len(quotes(actual))==38
assert len(re.findall(r'(?m)^>(?:[^\n]*)(?:\n>[^\n]*)*',actual))==24
assert [l for l in actual.splitlines() if l.startswith('### R')]==[l for l in before.splitlines() if l.startswith('### R')]
V=read(H/'VERIFICATION.json')
for rel,pin in V['inputs'].items():assert sha(R/rel)==pin['sha256'] and (R/rel).stat().st_size==pin['bytes']
tables=read(R/'outputs/guardfed_tables/celeba_flgmm_nine_scenes90_20261011/tables.json')
saved=[]
for view in ('native','shared_calibration'):
 p=next(p for p in tables['panels'] if p['view']==view and p['panel']=='ten');c=next(c for c in p['scenes'] if (c['distribution'],c['attack'])==('non-IID','S-DFA'))
 for metric in ('accuracy','aeod','aspd'):assert c['metrics'][metric]['display'] in actual;saved.append({'view':view,'metric':metric,'display':c['metrics'][metric]['display']})
out=REV/'rebuttal_working_F70_FL90_Hybrid20_release_20261011.md';assert not out.exists();shutil.copyfile(candidate,out);assert out.read_bytes()==candidate.read_bytes()
proof={'status':'ROOT_EDITORIAL_ADOPTED_COMPLETE24_F70_FL90_HYBRID20_AUTHOR_REVIEW_DRAFT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root_accepted':True,'candidate_path':out.relative_to(R).as_posix(),'candidate_sha256':sha(out),'agent_candidate_sha256':sha(candidate),'parent_release_path':parent.relative_to(R).as_posix(),'parent_release_sha256':sha(parent),'source_only_seal_sha256':sha(H/'FILES_SHA256.json'),'editorial_verification_sha256':sha(H/'VERIFICATION.json'),'semantic_diff_sha256':sha(H/'SEMANTIC_DIFF.json'),'full_agent_and_root_inverse_exact':True,'quote_blocks':24,'quote_lines':38,'quote_order_exact':True,'new_saved_display_cells_checked':saved,'new_statistics_computed':0,'fixed_scientific_content_cutoff':{'mechanism_native':370,'mechanism_three_view':370,'F_pairs':70,'FL_native_new':86,'FL_screen_reuse':4,'FL_three_view':90,'FL_complete_scene_records':90,'FL_scenes':9,'Hybrid_three_views':20,'Hybrid_scenes':2,'gradient_search_completed':64},'gradient14_completed_as_deployment_only':True,'gradient14_scientific_table_records':0,'NGA96_started_only':True,'NGA96_complete_claimed':False,'author_review_draft':True,'manuscript_applied':False,'primary_endpoint_selected':False,'final_test':False,'whole_rebuttal_complete':False}
with (H/'ROOT_EDITORIAL_ADOPTION.json').open('x',encoding='utf8') as f:json.dump(proof,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps({'draft':out.relative_to(R).as_posix(),'sha256':sha(out),'root_sha256':sha(H/'ROOT_EDITORIAL_ADOPTION.json')}))
