from pathlib import Path
import datetime,hashlib,json,re
R=Path(__file__).resolve().parents[1];H=R/'tmp/rebuttal_F70_Hybrid20_editorial_prepare_20261011';REV=R/'docs/server_deployment_20260923/revision_20260923'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(n,v):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
assert __debug__ and not (H/'ROOT_EDITORIAL_ADOPTION.json').exists()
assert sha(H/'FILES.json')=='4da8a12a480cc3d4afdec4be5ff98f90422d5317d8cc7db574f1a659ee4b1571'
for rel,v in read(H/'FILES.json').items():assert sha(R/rel)==v['sha256'] and (R/rel).stat().st_size==v['bytes']
for v in read(H/'SOURCE_PINS.json'):assert sha(v['path'])==v['sha256'] and Path(v['path']).stat().st_size==v['bytes']
candidate=REV/'rebuttal_working_F70_FL81_Hybrid20_20261011.md';parent=REV/'rebuttal_working_F60_FL81_gradient64_20261011.md'
assert sha(candidate)=='14fcaa9d60c4ab9ca17b9964a5d1dc4f178ad5537f7ff3c1c23b06586f2dfb19' and sha(parent)=='e76df8e53b7379a437bf09426c4b576443022f7766ae77d4ff444c456fd650b6'
before=parent.read_bytes().decode();s=before
for e in read(H/'EDITS.json'):assert s.count(e['old'])==e['count'];s=s.replace(e['old'],e['new'])
assert s.encode()==candidate.read_bytes()
undo=s
for e in reversed(read(H/'EDITS.json')):assert undo.count(e['new'])==e['count'];undo=undo.replace(e['new'],e['old'])
assert undo.encode()==parent.read_bytes()
assert len([l for l in s.splitlines() if l.startswith('>')])==38
assert [l for l in s.splitlines() if l.startswith('>')]==[l for l in before.splitlines() if l.startswith('>')]
assert [l for l in s.splitlines() if l.startswith('### R')]==[l for l in before.splitlines() if l.startswith('### R')]
numeric=read(H/'NUMERICAL_BINDINGS.json')
for v in numeric['formatted_pairs']:
 p=read(v['source'])
 for key in v['pointer'].strip('/').split('/'):p=p[int(key)] if isinstance(p,list) else p[key]
 assert p==v['value'] and v['formatted'] in s
assert len(numeric['formatted_pairs'])==24
# Avoid a self-expiring "not yet published" sentence in the publishable draft.
old='**Response.** A verified revision snapshot is available at [GuardFed, commit 3edbd954](https://github.com/YananZHOU5555/GuardFed/tree/3edbd954d91bbc5e4993546950202645d82ccbb0). This fixed release includes the accepted F50 evidence, FLGMM\'s 81 three-view records and eight-complete-scene table of 80 records, and the completed 64-run Fed-NGA/Huber validation search.'
new='**Response.** A verified revision snapshot is available at [GuardFed, commit 8515952b](https://github.com/YananZHOU5555/GuardFed/tree/8515952b668c3a22427a8d0db5cc0a2d4c7cf6e3). This fixed release includes the accepted F60 evidence, FLGMM\'s 81 three-view records and eight-complete-scene table of 80 records, the ten-checkpoint IID Benign hybrid table, and the completed 64-run Fed-NGA/Huber validation search.'
old2='That fixed Git64 release stops at F50; it does not include the subsequent non-IID F-deletion scenes, the two-scene hybrid extension or this F70/Hybrid20 prose update. This new author-review draft has not yet been published.'
new2='That fixed release stops at F60 and the Benign hybrid scene; its cutoff does not include the subsequent non-IID F Flip deletion scene, the two-scene hybrid extension or this F70/Hybrid20 prose update. These later additions have their own pinned acceptance records and release history.'
release_edits=[{'old':old,'new':new},{'old':old2,'new':new2}]
final=s
for e in release_edits:assert final.count(e['old'])==1;final=final.replace(e['old'],e['new'])
inverse=final
for e in reversed(release_edits):assert inverse.count(e['new'])==1;inverse=inverse.replace(e['new'],e['old'])
assert inverse==s
publication=R/'docs/server_deployment_20260923/training_20260923/publication_closed_increment65_verified_20261011.json'
assert sha(publication)=='844ffa5ee7a42a1de54d8465edec0150dd9c9dc42677e67657aebba5e2aa671d'
out=REV/'rebuttal_working_F70_FL81_Hybrid20_release_20261011.md'
with out.open('x',encoding='utf8',newline='\n') as f:f.write(final)
assert [l for l in final.splitlines() if l.startswith('>')]==[l for l in before.splitlines() if l.startswith('>')]
save('ROOT_RELEASE_EDITS.json',release_edits)
save('ROOT_EDITORIAL_ADOPTION.json',{'status':'ROOT_EDITORIAL_ADOPTED_COMPLETE24_F70_FL81_HYBRID20_AUTHOR_REVIEW_DRAFT','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root_accepted':True,'source':str(Path(__file__).relative_to(R)),'source_sha256':sha(__file__),'candidate_path':str(out.relative_to(R)),'candidate_sha256':sha(out),'agent_candidate_sha256':sha(candidate),'old_F60_sha256':sha(parent),'source_only_seal_sha256':sha(H/'FILES.json'),'editorial_verification_sha256':sha(H/'EDITORIAL_VERIFICATION.json'),'root_release_edits_sha256':sha(H/'ROOT_RELEASE_EDITS.json'),'full_agent_and_root_inverse_exact':True,'quote_blocks':24,'quote_lines':38,'quote_order_exact':True,'new_saved_mean_SD_pairs_checked':24,'new_statistics_computed':0,'F_pair_count':70,'F_scenes':7,'Hybrid_checkpoints':20,'Hybrid_scenes':2,'FL_native_new':86,'FL_screen_reuse':4,'FL_three_view_total':81,'gradient_search_completed':64,'gradient14_complete_claimed':False,'publication65_fixed_receipt_sha256':sha(publication),'author_review_draft':True,'manuscript_applied':False,'primary_endpoint_selected':False,'final_test':False,'whole_rebuttal_complete':False})
print(json.dumps({'draft':str(out.relative_to(R)),'draft_sha256':sha(out),'root_adoption_sha256':sha(H/'ROOT_EDITORIAL_ADOPTION.json'),'quotes':24,'saved_mean_SD_pairs_checked':24}))
