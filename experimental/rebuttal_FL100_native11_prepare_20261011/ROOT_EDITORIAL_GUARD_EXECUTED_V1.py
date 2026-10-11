from pathlib import Path
import datetime,hashlib,json,re,shutil
R=Path.cwd();H=R/'tmp/rebuttal_FL100_native11_prepare_20261011';V=R/'docs/server_deployment_20260923/revision_20260923'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_bytes());ref=lambda p:dict(path=Path(p).relative_to(R).as_posix(),sha256=sha(p))
e=read(H/'EXACT_EDITS.json');p=R/e['parent'];c=H/'rebuttal_working_F80_FL100_Hybrid20_native11_candidate_20261011.md';assert sha(p)==e['parent_sha256']=='b5d577c9a54d9570d7e39dec373919782d90aa634959843a317912d485dfc98b' and sha(c)==e['candidate_sha256']=='910b263339273c9d973491f1005f35ec4a64ed6a27127616c096ee7f3ffce382'
a=p.read_text(encoding='utf8');b=c.read_text(encoding='utf8');t=a
for edit in e['edits']:assert t.count(edit['before'])==1;t=t.replace(edit['before'],edit['after'])
assert t==b;t=b
for edit in reversed(e['edits']):assert t.count(edit['after'])==1;t=t.replace(edit['after'],edit['before'])
assert t==a
quotes=lambda s:[x for x in s.splitlines() if x.startswith('>')];head=lambda s:[x for x in s.splitlines() if x.startswith('#')];tables=lambda s:[x for x in s.splitlines() if x.startswith('|')]
assert quotes(a)==quotes(b) and len(quotes(b))==38 and head(a)==head(b) and tables(a)==tables(b)
assert len([x for x in head(b) if x.startswith('### ')])==23
for name,pin in read(H/'CHECK.json')['source_pins'].items():assert sha(R/name)==pin,name
for link in re.findall(r'\[[^\]]+\]\((E:/[^)]+)\)',b):assert Path(link.split('#',1)[0]).is_file(),link
out=V/'rebuttal_working_F80_FL100_Hybrid20_native11_20261011.md';assert not out.exists();shutil.copyfile(c,out);assert sha(out)==sha(c)
proof={'status':'ROOT_COMPLETE24_FL100_LEGACY_CPU_NATIVE11_EDITORIAL_UPDATE_ADOPTED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root_adopted':True,'source':ref(__file__),'parent':ref(p),'adopted_draft':ref(out),'exact_edits':ref(H/'EXACT_EDITS.json'),'comment_blocks':24,'reviewer_quote_lines_exact':38,'all_original_headings_and_performance_tables_exact':True,'inverse_reconstructs_parent':True,'new_scientific_results':0,'fixed_scientific_cutoff':{'mechanism_native':380,'mechanism_saved':380,'F_pairs':80,'FL_native':100,'FL_saved_CPU':100,'Hybrid_pairs':20,'NGA_new_accepted_at_cutoff':0,'native_methods':11,'native_evaluation_records':1100},'uniform_CUDA_path_author_approved':True,'uniform_CUDA_results_claimed':False,'final_test':False,'submission_ready':False}
with (H/'ROOT_EDITORIAL_ADOPTION.json').open('x',encoding='utf8',newline='\n') as f:json.dump(proof,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'status':proof['status'],'draft':ref(out),'root':ref(H/'ROOT_EDITORIAL_ADOPTION.json')}))
