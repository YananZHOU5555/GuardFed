from pathlib import Path
import copy,datetime,hashlib,json
R=Path.cwd();T=R/'docs/server_deployment_20260923/training_20260923';H=R/'tmp/root_CUDA7_resume1_Hybrid29_20261011';p=T/'TRAINING_STATE.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(p)=='d37d0382b291f15f0dcb4a3b9bd896bc6ad80b2bb448ed851030c9cc5109d806'
before=p.read_bytes();s=json.loads(before);original={k:copy.deepcopy(s[k]) for k in ['latest_working_rebuttal_draft','latest_rebuttal_draft']}
draft=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_working_F80_FL100_Hybrid20_native11_20261011.md';root=R/'tmp/rebuttal_FL100_native11_prepare_20261011/ROOT_EDITORIAL_ADOPTION.json'
assert sha(draft)=='910b263339273c9d973491f1005f35ec4a64ed6a27127616c096ee7f3ffce382' and sha(root)=='0f6b6342cd01eba3e7495040433e491f920078b41f650bb9d4e0ddd9a7440fff'
for key in original:
    v=s[key];assert v['content_cutoff']['FL_three_view']==90
    v.update(status='ROOT_COMPLETE24_NATIVE11_F80_FL100_HYBRID20_CURRENT',entry=draft.relative_to(R).as_posix(),sha256=sha(draft),root_proof=dict(path=root.relative_to(R).as_posix(),sha256=sha(root)),FL100_incorporated=True,uniform_CUDA_complete_claim=False,published_in_Git69=False)
    v['content_cutoff'].update(FL_three_view=100,FL_complete_scene_records=100,FL_scenes=10)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();s['current_entry_writer']=dict(path=Path(__file__).relative_to(R).as_posix(),sha256=sha(__file__),utc=now);s['local_work_verified_utc']=now;s['updated_unix']=datetime.datetime.now(datetime.timezone.utc).timestamp()
(H/'PRIOR_STATE_BEFORE_ALIAS_FIX.json').open('xb').write(before)
p.write_text(json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
proof=dict(status='STALE_LATEST_ALIASES_UPDATED_TO_ROOT_ACCEPTED_FL100_DRAFT',utc=now,prior_aliases=original,state_before_sha256=hashlib.sha256(before).hexdigest(),state_after_sha256=sha(p),new_science=0,draft_content_changed=False,five_entry_historical_tails_untouched=True)
with (H/'ROOT_ALIAS_FIX.json').open('x',encoding='utf8',newline='\n') as f:json.dump(proof,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'state_sha256':sha(p),'proof_sha256':sha(H/'ROOT_ALIAS_FIX.json')}))
