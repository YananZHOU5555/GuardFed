"""Review and stage a pinned, disjoint closed-evidence increment; never push here."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'tmp/revision-publish-20260928'
TRAIN=Path('docs/server_deployment_20260923/training_20260923');CHECKS=TRAIN/'server_reactivation_20261009'
parser=argparse.ArgumentParser();parser.add_argument('--increment',type=int,choices=(17,18,19),default=17)
args=parser.parse_args()
profiles={
    17:dict(previous='795f4b09c60c4a81de3d4aa67dad5beac5075827',start=10,end=18,prior_n=570,
        prior_sha='d80b4bea9692144e28a78e3a0871ef7301beea762574f680254d25aaa13b681c',baseline=658,native=68,views=28,
        native_tag='root_delta_20261009T142459Z',native_delta=8,handoffs=['BOUNDED_DELTA_010_017_HANDOFF.json'],
        live_tags=['20261009T142327Z'],formal_tag='20261009T143236Z'),
    18:dict(previous='1ac345c0dda16dcdfdedfcc0b020f58a24092aef',start=18,end=24,prior_n=658,
        prior_sha='43c5a3f0c13de9870485a1a68fbbe32755f6786feedc95ab8108947d2ed6f6a7',baseline=724,native=71,views=60,
        native_tag='root_delta_20261009T144558Z',native_delta=3,
        handoffs=['BOUNDED_DELTA_018_021_HANDOFF.json','BOUNDED_DELTA_022_023_HANDOFF.json'],
        live_tags=['20261009T143614Z','20261009T144231Z'],formal_tag='20261009T150054Z'),
    19:dict(previous='59c6e47b00e4875767dd1814fa09382cfc2b4e1c',start=24,end=24,prior_n=724,
        prior_sha='456ceac1a9b149fea39ef4c91e7910d679a91123a02058106de105e0303b23c2',baseline=724,native=71,views=60,
        native_tag=None,native_delta=0,handoffs=[],live_tags=[],formal_tag='20261009T151358Z')}
profile=profiles[args.increment];PREVIOUS=profile['previous'];MAPPING={}
def git(*args,**kwargs):return subprocess.check_output(['git','-c','core.longpaths=true',*args],cwd=REPO,**kwargs)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def copy(src,dst,expected=None):
    dst=Path(dst);target=REPO/dst;digest=sha(src)
    assert target.resolve().is_relative_to(REPO.resolve()) and src.stat().st_size<100_000_000
    assert expected is None or digest==expected
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
    assert sha(target)==digest and MAPPING.setdefault(dst.as_posix(),digest)==digest
assert git('rev-parse','HEAD',text=True).strip()==PREVIOUS and not git('status','--porcelain',text=True).strip()
state=read(ROOT/TRAIN/'TRAINING_STATE.json');main=state['celeba_mechanism_v1']
assert main['scientific_results_offserver_verified']==profile['native'] and main['three_view_new_models_offserver_verified']==profile['views']
assert state['final_evaluator_runtime_20261009']['actual_native_valid_image_replays_accepted']==profile['baseline']
evidence=ROOT/'tmp/celeba_valid_gpu_remaining440_v2_evidence_20261009'
prior=evidence/f"chunk_{profile['start']-1:03d}/cumulative_{profile['prior_n']}_accepted.json";prior_data=read(prior);reviews=[]
assert sha(prior)==profile['prior_sha']
for index in range(profile['start'],profile['end']):
    folder=evidence/f'chunk_{index:03d}';proof_path=folder/'ROOT_OFFSERVER_VERIFICATION.json';proof=read(proof_path)
    collector_path=folder/f'cumulative_{460+11*(index+1)}_accepted.json';collector=read(collector_path)
    assert proof['status']=='ROOT_GPU440_RESOURCE_GUARD_V2_CHUNK_OFFSERVER_SAVED_ARRAY_PASS'
    assert proof['chunk_index']==index and proof['native_max_abs_difference']==0
    assert proof['saved_metrics_verified']==99 and proof['saved_confusion_counts_verified']==264 and proof['saved_prediction_rules_verified']==33
    assert proof['new_CNN_inference']==0 and not proof['final_test']
    assert sha(folder/'chunk_evidence.tar.gz')==proof['archive_sha256']
    assert collector['previous_collector_sha256']==sha(prior) and collector['new_proof_sha256']==sha(proof_path)
    assert collector['accepted_n']==len(set(collector['accepted_ids']))==prior_data['accepted_n']+11
    assert set(collector['accepted_ids'])-set(prior_data['accepted_ids'])==set(proof['accepted_new_ids'])
    assert not set(proof['accepted_new_ids']).intersection(prior_data['accepted_ids'])
    reviews.append(dict(chunk=index,archive_sha256=proof['archive_sha256'],proof_sha256=sha(proof_path),collector_sha256=sha(collector_path)))
    for p in folder.iterdir():
        if p.is_file():copy(p,Path('experimental')/evidence.name/folder.name/p.name)
    prior,prior_data=collector_path,collector
assert prior_data['accepted_n']==profile['baseline']
for name in profile['handoffs']:
    handoff=evidence/name;copy(handoff,Path('experimental')/evidence.name/handoff.name)
backup=ROOT/CHECKS/'mechanism_science_backups_20261009';tag=profile['native_tag']
if tag is not None:
    proof=read(backup/tag/'ROOT_DELTA_VERIFICATION.json')
    assert proof['total_new_strict_and_offserver']==profile['native'] and len(proof['new_ids'])==profile['native_delta']
    assert proof['oldFull_models_repacked']==0 and not proof['test']
    for name in (tag+'.tar.gz',tag+'.tar.gz.receipt.json',tag+'_offserver_verification.json','verified_ledger.json'):
        copy(backup/name,CHECKS/'mechanism_science_backups_20261009'/name)
    for folder in (backup/tag,backup/('mechanism_inspection_v4_'+tag)):
        for p in folder.rglob('*'):
            if p.is_file():copy(p,CHECKS/'mechanism_science_backups_20261009'/p.relative_to(backup))
runtime=ROOT/'tmp/celeba_valid_gpu_remaining440_resource_gate_v2_execution_20261009'
for live_tag in profile['live_tags']:
    for name in (f'live_{live_tag}.json',f'live_{live_tag}.ROOT.json'):
        copy(runtime/name,Path('experimental')/runtime.name/name)
for name in ('RUNNING.md','TRAINING_STATE.json','REBUTTAL_COMPLETION_20261009.md',
             'celeba_mechanism_v1/EXECUTION.md',f'publication_closed_increment{args.increment-1}_verified_20261009.json'):
    copy(ROOT/TRAIN/name,TRAIN/name)
for name in ('MONITOR_HANDOFF.md','latest_formal_live.json',f"root_live_{profile['formal_tag']}.json"):
    copy(ROOT/CHECKS/name,CHECKS/name)
for name in ('update_reactivation_state_20261009.py','update_completion_current_20261009.py',
             'verify_publication_increment16_20261009.py',Path(__file__).name):
    copy(ROOT/'tmp'/name,Path('experimental/server_reactivation_20261009')/name)
if args.increment==18:
    prep=ROOT/'tmp/celeba_mechanism_valid_incremental_next37_20261009';execution=prep/'execution_candidate'
    delta=execution/'backups/incremental_20261009T144055Z';adopt=read(delta/'ROOT_ADOPTION_REVIEW.json')
    assert sha(delta/'ROOT_ADOPTION_REVIEW.json')=='a7a563390de455914b33cf62064d674347e0c26754a778b9a344ac99bdc4fac3'
    assert adopt['accepted_new']==32 and adopt['cumulative_three_view_models']==60
    assert sha(delta/'incremental_valid_three_views.tar.gz')==adopt['archive_sha256']
    assert adopt['all_native_differences_zero'] and adopt['new_Full_inference']==0 and not adopt['test_inference']
    for p in delta.iterdir():
        if p.is_file():copy(p,Path('experimental')/p.relative_to(ROOT/'tmp'))
    for name in ('ROOT_PROGRESS_20261009T143713Z.json','ROOT_PROGRESS_20261009T143713Z.RAW.json'):
        copy(execution/name,Path('experimental')/prep.name/'execution_candidate'/name)
    for name in ('adopt_mechanism_next37_remaining32_root_20261009.py','observe_mechanism_next37_root_20261009.py'):
        copy(ROOT/'tmp'/name,Path('experimental/server_reactivation_20261009')/name)
    table=ROOT/TRAIN/'celeba_mechanism_v1/interim_tables_20261009T144558Z'
    for p in table.iterdir():
        if p.is_file():copy(p,p.relative_to(ROOT))
    prep11=ROOT/'tmp/celeba_mechanism_valid_incremental_next11_20261009'
    seal=read(prep11/'FILES_SHA256.json')
    assert sha(prep11/'FILES_SHA256.json')=='65f706e8c7c7d7e18e76c8a300dd845297c97b3bd5c6c182bbbb8aad0103b5ff'
    for row in seal['members']:copy(prep11/row['path'],Path('experimental')/prep11.name/row['path'],row['sha256'])
    for name in ('FILES_SHA256.json','root_source_review/ROOT_REVIEW.json','root_review_20261009T145600Z/selfcheck.json'):
        copy(prep11/name,Path('experimental')/prep11.name/name)
    # The separately reviewed paired-table artifact is optional until its actual root proof exists.
    paired_table=ROOT/TRAIN/'celeba_mechanism_v1/three_view_interim_20261009T145900Z'
    if paired_table.exists():
        root_review=read(paired_table/'ROOT_REVIEW.json')
        assert root_review['status']=='ROOT_SIX_SCENE_THREE_VIEW_PAIRED_SAVED_RECEIPTS_AND_STATISTICS_PASS'
        assert root_review['paired_checkpoints']==60 and root_review['new_inference']==0 and not root_review['final_test']
        for p in paired_table.iterdir():
            if p.is_file():copy(p,p.relative_to(ROOT))
        src=ROOT/'tmp/celeba_mechanism_three_view_paired_interim_20261009'
        for p in src.iterdir():
            if p.is_file():copy(p,Path('experimental')/src.name/p.name)
if args.increment==19:
    execution=ROOT/'tmp/celeba_mechanism_valid_incremental_next11_20261009/execution_candidate'
    startup=read(execution/'ROOT_STARTUP_OBSERVATION.json')
    assert startup['status']=='ROOT_NEXT11_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS'
    assert startup['scientific_offserver_new_accepted']==0 and startup['original60_not_rerun']
    assert startup['deployment_receipt_sha256']==sha(execution/'deployment_receipt.json')
    assert sha(execution/'EXECUTION_SOURCE_SHA256.json')=='9f1252dd7c11abfe7cee297b2028ca9c58f179d964efb39ee6008ea860508b15'
    for row in read(execution/'EXECUTION_SOURCE_SHA256.json')['members']:
        assert sha(execution/row['path'])==row['sha256']
    for p in execution.iterdir():
        if p.is_file():copy(p,Path('experimental')/p.relative_to(ROOT/'tmp'))
    for name in ('review_mechanism_next37_execution_root_20261009.py','deploy_mechanism_next37_root_20261009.py','observe_mechanism_next37_root_20261009.py'):
        copy(ROOT/'tmp'/name,Path('experimental/server_reactivation_20261009')/name)
receipt_rel=TRAIN/f'publication_closed_increment{args.increment}_20261009.json'
attributes=REPO/'.gitattributes';content=attributes.read_text();patterns=[receipt_rel.as_posix()+' -text']
if args.increment==18:
    patterns += ['experimental/celeba_mechanism_valid_incremental_next11_20261009/** -text',
        'experimental/celeba_mechanism_three_view_paired_interim_20261009/** -text']
for pattern in patterns:
    if pattern not in content:content+='\n'+pattern+'\n'
attributes.write_text(content,newline='\n')
receipt=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_commit=PREVIOUS,
    copied_sha256=MAPPING.copy(),new_GPU_chunks_root_reviewed=reviews,baseline_valid_replays_accepted=profile['baseline'],
    mechanism_offserver_verified=profile['native'],mechanism_three_view_offserver_verified=profile['views'],FLGMM_offserver_verified=13,
    Hybrid_offserver_verified=4,native_tolerance=1e-12,duplicated_old_models=0,test_started=False,scientific_goal_complete=False)
if args.increment==19:receipt['next11_evaluation_started_with_no_new_offserver_acceptance']=True
with (ROOT/receipt_rel).open('x',encoding='utf8',newline='\n') as stream:
    json.dump(receipt,stream,ensure_ascii=False,indent=2);stream.write('\n')
copy(ROOT/receipt_rel,receipt_rel);names=[*MAPPING,'.gitattributes']
for start in range(0,len(names),25):
    git('add','-f','--',*names[start:start+25]);git('add','--renormalize','--',*names[start:start+25])
changed=git('diff','--cached','--name-only','-z').decode().split('\0')[:-1];assert set(changed)<=set(names)
payload=git('cat-file','--batch',input=''.join(':'+n+'\n' for n in MAPPING).encode());position=0
for name,digest in MAPPING.items():
    end=payload.index(b'\n',position);header=payload[position:end].split();assert header[1]==b'blob';size=int(header[2])
    assert hashlib.sha256(payload[end+1:end+1+size]).hexdigest()==digest,name;position=end+size+2
assert position==len(payload)
print(json.dumps(dict(status='ROOT_CLOSED_INCREMENT_AND_INDEX_BLOB_SHA_PASS_NO_COMMIT_OR_PUSH',
    changed_files=len(changed),byte_verified_files=len(MAPPING),baseline_accepted=profile['baseline'],mechanism_native_accepted=profile['native'])))
