"""Review and stage a pinned, disjoint closed-evidence increment; never push here."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'tmp/revision-publish-20260928'
TRAIN=Path('docs/server_deployment_20260923/training_20260923');CHECKS=TRAIN/'server_reactivation_20261009'
parser=argparse.ArgumentParser();parser.add_argument('--increment',type=int,choices=(17,18,19,20,21,22,23,24,25,26,27),default=17)
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
        native_tag=None,native_delta=0,handoffs=[],live_tags=[],formal_tag='20261009T151358Z'),
    20:dict(previous='de40f774f8ea180440dc111ad45a608a666ac168',start=24,end=40,prior_n=724,
        prior_sha='456ceac1a9b149fea39ef4c91e7910d679a91123a02058106de105e0303b23c2',baseline=900,native=82,views=71,
        native_tag='root_delta_20261009T154227Z',native_delta=11,handoffs=['BOUNDED_DELTA_024_036_HANDOFF.json','BOUNDED_DELTA_037_039_HANDOFF.json'],
        live_tags=['20261009T152426Z'],formal_tag='20261009T154505Z'),
    21:dict(previous='1b16f4753852f994171329baf2e79fdb6f90281a',start=40,end=40,prior_n=900,
        prior_sha='00e0cc89784832f8fc8293ce39e2c8ec6247f0e5d0fc37288cd3032133a9e6a3',baseline=900,native=82,views=82,
        native_tag=None,native_delta=0,handoffs=[],live_tags=[],formal_tag='20261009T162838Z'),
    22:dict(previous='d9130d3987ce38ed5a7a3a2083221b8f1bee95da',start=40,end=40,prior_n=900,
        prior_sha='00e0cc89784832f8fc8293ce39e2c8ec6247f0e5d0fc37288cd3032133a9e6a3',baseline=900,native=82,views=82,
        native_tag=None,native_delta=0,handoffs=[],live_tags=[],formal_tag='20261009T170115Z'),
    23:dict(previous='8780a3f8435dcbc5717cecb6237d79151f88b002',start=40,end=40,prior_n=900,
        prior_sha='00e0cc89784832f8fc8293ce39e2c8ec6247f0e5d0fc37288cd3032133a9e6a3',baseline=900,native=92,views=92,
        native_tag='root_delta_20261009T171247Z',native_delta=10,handoffs=[],live_tags=[],formal_tag='20261009T174808Z'),
    24:dict(previous='5f4784f59e0300159f17dd65e6cc5a3c1614b41f',start=40,end=40,prior_n=900,
        prior_sha='00e0cc89784832f8fc8293ce39e2c8ec6247f0e5d0fc37288cd3032133a9e6a3',baseline=900,native=104,views=100,
        native_tag='root_delta_20261009T181013Z',native_delta=12,handoffs=[],live_tags=[],formal_tag='20261009T182951Z'),
    25:dict(previous='1899126e405770a46a6936ee25483b60b28cd84a',start=40,end=40,prior_n=900,
        prior_sha='00e0cc89784832f8fc8293ce39e2c8ec6247f0e5d0fc37288cd3032133a9e6a3',baseline=900,native=112,views=101,
        native_tag='root_delta_20261009T190056Z',native_delta=8,handoffs=[],live_tags=[],formal_tag='20261009T190051Z'),
    26:dict(previous='b55159fc951c8d3bfd41b5c039cb1239fbdb2219',start=40,end=40,prior_n=900,
        prior_sha='00e0cc89784832f8fc8293ce39e2c8ec6247f0e5d0fc37288cd3032133a9e6a3',baseline=900,native=112,views=112,
        native_tag=None,native_delta=0,handoffs=[],live_tags=[],formal_tag='20261009T193513Z'),
    27:dict(previous='9ed3ff9d5fa1eba3a42d858275c7e9994b028776',start=40,end=40,prior_n=900,
        prior_sha='00e0cc89784832f8fc8293ce39e2c8ec6247f0e5d0fc37288cd3032133a9e6a3',baseline=900,native=112,views=112,
        native_tag=None,native_delta=0,handoffs=[],live_tags=[],formal_tag='20261009T195725Z')}
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
assert git('rev-parse','HEAD',text=True).strip()==PREVIOUS
if git('status','--porcelain',text=True).strip():
    assert args.increment==23
    attempt=ROOT/CHECKS/'publication23_stage_attempt1.json'
    assert sha(attempt)=='26e89edefc54e09b421c6623f4220869d6a888b618d4dc944438795bd8272dd6'
    prior_copy=read(attempt)
    assert prior_copy['previous_commit']==PREVIOUS and not prior_copy['commit_created'] and not prior_copy['pushed']
    current={row[3:].decode('utf8'):row[:2].decode() for row in git('status','--porcelain=v1','-z','--untracked-files=all').split(b'\0') if row}
    assert current=={row['path']:row['status'] for row in prior_copy['files']}
    for row in prior_copy['files']:assert sha(REPO/row['path'])==row['sha256']

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
if args.increment==23:
    for path,digest in (
            (backup/(tag+'.tar.gz'),'839eb1f3b0bb6fa272f869c190e810b8cad33d565c6f9a2cf55b995e68411778'),
            (backup/(tag+'.tar.gz.receipt.json'),'fdb90bb38b80dba4c61e12a62882510747ef3380110e64455086c3699f130481'),
            (backup/(tag+'_offserver_verification.json'),'1f793c5975e6af2fb087304986e7cb2dd0af80d1b7a330aa4704b49aae1ef470'),
            (backup/tag/'ROOT_DELTA_VERIFICATION.json','a4a8c506668db73b16931a37fa3fe843c5590b0bf5049846967bb9a86be6bd56'),
            (backup/'verified_ledger.json','dda48b06f25f529bb8688acc5f6979d5bd27adf3964068e204451f43d43072a2'),
            (ROOT/CHECKS/f"root_live_{profile['formal_tag']}.json",'e220eb91200d601ec4b4060058b4fc6c718a9ad55f74d2ae566bbee16c025da1')):
        assert sha(path)==digest
if args.increment==24:
    for path,digest in (
            (backup/(tag+'.tar.gz'),'1f8fed22d210028b18aa4682bacd80ec24c4c3ef1f9d461b26e108bb06a20f37'),
            (backup/(tag+'.tar.gz.receipt.json'),'90694d60ded4afed5f59569a55311893cbdf99115d0b571c75f1f47520a86bbc'),
            (backup/(tag+'_offserver_verification.json'),'14336440162dbf72dd28e81abc6dfee00a8da40a2830f591a4cf5876ddf64ee6'),
            (backup/tag/'ROOT_DELTA_VERIFICATION.json','12da8eda9e62483c3886f17fa69b7f7a58da3e32cf4fc85bc480fcbfced4e6c1'),
            (backup/tag/'ROOT_INDEPENDENT_REVIEW.json','3fe664eafb9ac0cf1d52aaca956073fb2f685d82bb0fea62aaac71780e09c18d'),
            (backup/'verified_ledger.json','8fc6b8b1ea9fdb4591c3da5e83ebe51284803720479c3d436f3a8c737430999d'),
            (ROOT/CHECKS/f"root_live_{profile['formal_tag']}.json",'b561d77507105a927b5d58c4770a2d55621d569aa7fd5620498b91f0a91c3b8d')):
        assert sha(path)==digest
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
if args.increment==20:
    prep=ROOT/'tmp/celeba_mechanism_valid_incremental_next11_20261009';execution=prep/'execution_candidate'
    delta=execution/'backups/incremental_20261009T152532Z';adopt=read(delta/'ROOT_ADOPTION_REVIEW.json')
    assert sha(delta/'ROOT_ADOPTION_REVIEW.json')=='692ecd168ecab0b9c960965decb68424ce2ad80a5cf7ca452d2739da6b0a768a'
    assert adopt['accepted_new']==11 and adopt['cumulative_three_view_models']==71
    assert sha(delta/'incremental_valid_three_views.tar.gz')==adopt['archive_sha256']
    assert adopt['all_native_differences_zero'] and adopt['new_Full_inference']==0 and not adopt['test_inference']
    for p in delta.iterdir():
        if p.is_file():copy(p,Path('experimental')/p.relative_to(ROOT/'tmp'))
    for p in execution.glob('ROOT_PROGRESS_*'):
        if sha(p)==adopt['remote_terminal_proof_sha256'] or p.name==read(ROOT/TRAIN/'TRAINING_STATE.json')['celeba_mechanism_v1']['next11_valid_replay']['latest_progress_path'].split('/')[-1].replace('.json','.RAW.json'):
            copy(p,Path('experimental')/p.relative_to(ROOT/'tmp'))
    copy(execution/'ROOT_BACKUP_COMMAND_STDOUT.json',Path('experimental')/prep.name/'execution_candidate/ROOT_BACKUP_COMMAND_STDOUT.json')
    for name in ('backup_mechanism_next11_root_20261009.py','adopt_mechanism_next11_root_20261009.py','review_paired71_table_root_20261009.py'):
        copy(ROOT/'tmp'/name,Path('experimental/server_reactivation_20261009')/name)
    copy(ROOT/CHECKS/'auxiliary_screens_20261009T153214Z.json',CHECKS/'auxiliary_screens_20261009T153214Z.json',
        '5bfe255142c9a6798f82a1331d83aaa45917b5e27781c4dac965028552257b84')
    copy(ROOT/'docs/返修实验总览.md',Path('docs/返修实验总览.md'))
    writing=ROOT/'tmp/celeba_mechanism_rebuttal_increment_20261009'
    assert sha(writing/'FILES_SHA256.json')=='07112ccc2c14a3fdbee0eb5fb1d9a9a5cb817fe99b099614b8726cf33251634f'
    for row in read(writing/'source_map.json')['files']:
        assert sha(Path(row['path']))==row['sha256']
    for row in read(writing/'FILES_SHA256.json')['members']:
        copy(writing/row['path'],Path('experimental')/writing.name/row['path'],row['sha256'])
    copy(writing/'FILES_SHA256.json',Path('experimental')/writing.name/'FILES_SHA256.json')
    paired71=ROOT/TRAIN/'celeba_mechanism_v1/three_view_interim71_20261009'
    if paired71.exists():
        accepted=read(paired71/'ROOT_REVIEW.json')
        assert accepted['complete_paired_scenes']==7 and accepted['new_inference']==0 and not accepted['final_test']
        for p in paired71.iterdir():
            if p.is_file():copy(p,p.relative_to(ROOT))
        source=ROOT/'tmp/celeba_mechanism_three_view_paired71_20261009'
        for row in read(source/'FILES_SHA256.json')['members']:
            copy(source/row['path'],Path('experimental')/source.name/row['path'],row['sha256'])
        copy(source/'FILES_SHA256.json',Path('experimental')/source.name/'FILES_SHA256.json')
    integrated=ROOT/'tmp/guardfed_rebuttal_integrated71_20261009'
    if (integrated/'ROOT_REVIEW.json').exists():
        assert sha(integrated/'FILES_SHA256.json')=='823af5315a174b7dfbe6ad629d480c92426cde6b6ebfff716296d3464e64951a'
        assert read(integrated/'ROOT_REVIEW.json')['status']=='ROOT_SOURCE_BOUND_SEVEN_SCENE_COMPLETE_DRAFT_REVIEW_PASS_PENDING_FULL_COHORT'
        for row in read(integrated/'SOURCE_MAP.json')['sources']:
            assert sha(Path(row['path']))==row['sha256']
        for row in read(integrated/'FILES_SHA256.json')['members']:
            copy(integrated/row['path'],Path('experimental')/integrated.name/row['path'],row['sha256'])
        for name in ('FILES_SHA256.json','ROOT_REVIEW.json'):
            copy(integrated/name,Path('experimental')/integrated.name/name)
        canonical=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated71_20261009'
        for row in read(integrated/'FILES_SHA256.json')['members']:
            copy(canonical/row['path'],canonical.relative_to(ROOT)/row['path'],row['sha256'])
        for name in ('FILES_SHA256.json','ROOT_REVIEW.json'):
            copy(canonical/name,canonical.relative_to(ROOT)/name,sha(integrated/name))
        # Publish only sealed prior inputs; never include its retained nonsealed cache.
        old=ROOT/'tmp/guardfed_rebuttal_integrated_20261009'
        assert sha(old/'FILES_SHA256.json')=='ee8147a759bdccf548094b49b52a13570ae142df3ca5abdda6ae8e07fe321881'
        for row in read(old/'FILES_SHA256.json')['members']:
            copy(old/row['path'],Path('experimental')/old.name/row['path'],row['sha256'])
        copy(old/'FILES_SHA256.json',Path('experimental')/old.name/'FILES_SHA256.json')
    baseline_tables=ROOT/'tmp/celeba_nine_method_three_view_tables_20261009'
    if (baseline_tables/'ROOT_REVIEW.json').exists():
        assert sha(baseline_tables/'FILES_SHA256.json')=='1bc00ab3e2f5b69f915730dc1b94fb9da49cc8cc69f9756f8b2ea092b3e33c8e'
        assert read(baseline_tables/'ROOT_REVIEW.json')['status']=='ROOT_NINE_METHOD900_THREE_VIEW_RECEIPTS_COUNTS_AND_TABLE_STATISTICS_PASS'
        table_seal=read(baseline_tables/'FILES_SHA256.json')
        for name,row in table_seal.items():
            copy(baseline_tables/name,Path('experimental')/baseline_tables.name/name,row['sha256'])
        for name in ('FILES_SHA256.json','HANDOFF.json','ROOT_REVIEW.json'):
            copy(baseline_tables/name,Path('experimental')/baseline_tables.name/name)
        canonical=ROOT/'outputs/guardfed_tables/celeba_nine_method_three_view_20261009'
        for name in [*table_seal,'FILES_SHA256.json','HANDOFF.json','ROOT_REVIEW.json']:
            copy(canonical/name,canonical.relative_to(ROOT)/name,sha(baseline_tables/name))
        copy(ROOT/'tmp/review_nine_method_three_view_tables_root_20261009.py',Path('experimental/server_reactivation_20261009/review_nine_method_three_view_tables_root_20261009.py'))
if args.increment==21:
    for base_name,delta_name,total in (
            ('celeba_flgmm_screen_20261009_v2_dispatch','accepted_delta_after13_20261009',19),
            ('celeba_hybrid_screen_execution_20261009','accepted_delta_after4_20261009',6)):
        base=ROOT/'tmp'/base_name;delta=base/delta_name;latest=read(base/'LATEST_BACKUP.json')
        assert latest['accepted']==total and sha(base/latest['chain_file'])==latest['chain_sha256']
        proof=read(delta/'ROOT_ADOPTION_REVIEW.json')
        assert proof['status']=='ROOT_LOCKED_AUXILIARY_DELTA_ARCHIVE_MEMBER_AND_ORIGINAL_RECORD_ACCEPTANCE_PASS'
        assert proof['accepted_total']==total and proof['new_inference']==0 and not proof['final_test']
        for row in read(delta/'DELIVERY_FILES_SHA256.json')['members']:
            copy(delta/row['path'],Path('experimental')/base_name/delta_name/row['path'],row['sha256'])
        for name in ('DELIVERY_FILES_SHA256.json','ROOT_ADOPTION_REVIEW.json',read(base/latest['chain_file'])['archive']):
            copy(delta/name,Path('experimental')/base_name/delta_name/name)
        for name in ('LATEST_BACKUP.json',latest['chain_file']):copy(base/name,Path('experimental')/base_name/name)
    copy(ROOT/'tmp/adopt_auxiliary_deltas_root_20261009.py',Path('experimental/server_reactivation_20261009/adopt_auxiliary_deltas_root_20261009.py'))
    copy(ROOT/'docs/返修实验总览.md',Path('docs/返修实验总览.md'))
    copy(ROOT/CHECKS/'resource_boundary_20261009T1608Z.json',CHECKS/'resource_boundary_20261009T1608Z.json')
    copy(ROOT/CHECKS/'auxiliary_screens_20261009T163446Z.json',CHECKS/'auxiliary_screens_20261009T163446Z.json',
        'ee6737a73d28c2d8fc58d0c1e39f71631e56011fb8d7789f627fc4ed95e6491a')
    variant_source=ROOT/'tmp/celeba_mechanism_remaining_variants_source_plan_20261009'
    assert sha(variant_source/'FILES_SHA256.json')=='baee87c13c06debd1cca7b93ea4174a87d7c688ef8e76627bdfe0e17ea20b5c9'
    variant_proof=read(variant_source/'ROOT_SOURCE_REVIEW.json')
    assert variant_proof['status']=='ROOT_SOURCE_ONLY_REMAINING_VARIANT_MAPPING_AND_RECIPE_REVIEW_PASS_NOT_DISPATCH'
    assert variant_proof['planned_job_bytes_and_paired_recipe_verified']==800 and not variant_proof['dispatch_authorized']
    for name,row in read(variant_source/'FILES_SHA256.json')['files'].items():
        copy(variant_source/name,Path('experimental')/variant_source.name/name,row['sha256'])
    for name in ('FILES_SHA256.json','ROOT_SOURCE_REVIEW.json'):
        copy(variant_source/name,Path('experimental')/variant_source.name/name)
    after71=ROOT/'tmp/celeba_mechanism_valid_incremental_after71_20261009'
    after71_execution=after71/'execution_candidate'
    if (after71_execution/'ROOT_STARTUP_OBSERVATION.json').exists():
        startup=read(after71_execution/'ROOT_STARTUP_OBSERVATION.json')
        assert startup['status']=='ROOT_AFTER71_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS'
        assert startup['original71_not_rerun'] and startup['scientific_offserver_new_accepted']==0 and not startup['test_inference']
        assert sha(after71/'PACKAGE_SHA256.json')=='64224d899d591b07de068b48c59c477032e0f898c869abb2a34070b89d2e5b6b'
        for name,row in read(after71/'PACKAGE_SHA256.json')['artifacts'].items():
            copy(after71/name,Path('experimental')/after71.name/name,row['sha256'])
        for name in ('PACKAGE_SHA256.json','HANDOFF.json'):
            copy(after71/name,Path('experimental')/after71.name/name)
        for p in (after71/'root_independent_review').iterdir():
            if p.is_file():copy(p,Path('experimental')/p.relative_to(ROOT/'tmp'))
        for name in ('ROOT_APPROVED.json','EXECUTION_DRAFT.json','deployment_receipt.json','ROOT_STARTUP_OBSERVATION.json',
                'ROOT_STARTUP_OBSERVATION.RAW.json','preflight.json','start_receipt.json','APPROVED.json','APPROVED.sha256','root_deployment_source.tar.gz'):
            copy(after71_execution/name,Path('experimental')/after71.name/'execution_candidate'/name)
        for name in ('deploy_mechanism_after71_root_20261009.py','observe_mechanism_next37_root_20261009.py'):
            copy(ROOT/'tmp'/name,Path('experimental/server_reactivation_20261009')/name)
        for p in after71_execution.glob('ROOT_PROGRESS_*.json'):
            copy(p,Path('experimental')/after71.name/'execution_candidate'/p.name)
        closure_paths=list((after71_execution/'backups').glob('*/ROOT_ADOPTION_REVIEW.json'))
        assert len(closure_paths)==1
        closure=read(closure_paths[0]);delta=closure_paths[0].parent
        assert closure['status']=='ROOT_AFTER71_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
        assert closure['accepted_new']==11 and closure['cumulative_three_view_models']==82 and closure['original71_unchanged']
        assert sha(delta/'incremental_valid_three_views.tar.gz')==closure['archive_sha256']
        for p in delta.iterdir():
            if p.is_file():copy(p,Path('experimental')/p.relative_to(ROOT/'tmp'))
        for name in ('ROOT_BACKUP_ATTEMPT.json','ROOT_BACKUP_COMMAND_STDOUT.json','ROOT_TRANSPORT_RECOVERY_REVIEW.json',
                'ROOT_BACKUP_TRANSPORT_ATTEMPT.json','ROOT_BACKUP_TRANSPORT_COMMAND_RESULT.json'):
            copy(after71_execution/name,Path('experimental')/after71.name/'execution_candidate'/name)
        for name in ('backup_mechanism_after71_root_20261009.py','adopt_mechanism_after71_root_20261009.py',
                'backup_mechanism_after71_transport_v2_root_20261009.py','adopt_mechanism_after71_transport_v2_root_20261009.py'):
            copy(ROOT/'tmp'/name,Path('experimental/server_reactivation_20261009')/name)
        for review_name in ('celeba_mechanism_after71_closure_source_review_20261009','celeba_mechanism_after71_transport_v2_source_review_20261009'):
            review_dir=ROOT/'tmp'/review_name
            for row in read(review_dir/'FILES_SHA256.json')['members']:
                copy(review_dir/row['path'],Path('experimental')/review_name/row['path'],row['sha256'])
            for p in review_dir.iterdir():
                if p.name in ('FILES_SHA256.json','ROOT_OPERATION_REVIEW.json','ROOT_SOURCE_OPERATION_REVIEW.json'):
                    copy(p,Path('experimental')/review_name/p.name)
    reply_v2=ROOT/'tmp/guardfed_rebuttal_integrated71_v2_20261009'
    reply_v2_canonical=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated71_v2_20261009'
    assert sha(reply_v2/'FILES_SHA256.json')=='44cbeba04d1835017d8336c4dde6109cd50553bfa8a6623dffe5e6b3f0c6c580'
    reply_v2_proof=read(reply_v2/'ROOT_REVIEW.json')
    assert reply_v2_proof['status']=='ROOT_MINIMAL_V2_SEVEN_SCENE_PROSE_CORRECTION_AND_ACCEPTED900_SOURCE_PASS'
    assert reply_v2_proof['all_allowed_substitutions_reversed_to_exact_originals'] and not reply_v2_proof['new_performance_claims']
    for name in [*[r['path'] for r in read(reply_v2/'FILES_SHA256.json')['members']],'FILES_SHA256.json','ROOT_REVIEW.json']:
        copy(reply_v2/name,Path('experimental')/reply_v2.name/name)
        copy(reply_v2_canonical/name,reply_v2_canonical.relative_to(ROOT)/name,sha(reply_v2/name))
if args.increment==22:
    copy(ROOT/CHECKS/'auxiliary_screens_20261009T170231Z.json',CHECKS/'auxiliary_screens_20261009T170231Z.json',
        '15dd20dff83dfb39f9695577c376d446124314fde6b28e8b5d467d946cfa7424')
    # Canonical accepted artifacts only: descriptive analysis, candidate figure and display PDF.
    for relative,seal_sha,root_sha,files_key in (
            ('outputs/guardfed_tables/celeba_nine_method_view_attribution_20261009',
             '93f405b0df99166dc523ec5126a49cabafa4e2bcc5065966046e15ff05c4b5fb',
             '81dce12481dfec62bf62555acedcd9c2735756665594896faf27ed48ac1d2082','files'),
            ('outputs/guardfed_figures/synthetic_terminal_candidate_20261009',
             '5004d983fb32f7534d447659999e84c2925f9d0fb20ef79df75be4df7fc95504',
             '837412a3cbb6d2f74b5e560b884f704bf2ca2877ebfa11828075900880d283d3','files'),
            ('outputs/guardfed_tables/celeba_nine_method_three_view_pdf_20261009',
             'ab2923437aa0286c572b5a173821f6c256559a4f4f27e4caf82317691adcd9e2',
             'b96103d3b588d8fc969d598c4b2aa2ca0991dbb96af6c4be07cf42282eb29f16',None)):
        canonical=ROOT/relative
        assert sha(canonical/'FILES_SHA256.json')==seal_sha
        assert sha(canonical/'ROOT_REVIEW.json')==root_sha
        assert read(canonical/'ROOT_REVIEW.json')['source_seal_sha256']==seal_sha
        seal=read(canonical/'FILES_SHA256.json');members=seal[files_key] if files_key else seal
        for name,row in members.items():
            copy(canonical/name,Path(relative)/name,row['sha256'])
        copy(canonical/'FILES_SHA256.json',Path(relative)/'FILES_SHA256.json',seal_sha)
        copy(canonical/'ROOT_REVIEW.json',Path(relative)/'ROOT_REVIEW.json',root_sha)
    locator=TRAIN/'manuscript_source_locator_20261009'
    locator_root_sha='5fc32efc81e7931c7d7b70b8d542824d16a729b73808f553069aac5b726a9218'
    assert sha(ROOT/locator/'ROOT_REVIEW.json')==locator_root_sha
    locator_proof=read(ROOT/locator/'ROOT_REVIEW.json')
    copy(ROOT/locator/'REPORT.md',locator/'REPORT.md',locator_proof['report_sha256'])
    copy(ROOT/locator/'EVIDENCE.json',locator/'EVIDENCE.json',locator_proof['evidence_sha256'])
    copy(ROOT/locator/'ROOT_REVIEW.json',locator/'ROOT_REVIEW.json',locator_root_sha)
    for relative in ('docs/server_deployment_20260923/revision_20260923/rebuttal_validation900_addendum_20261009.md',
            'docs/返修实验总览.md'):
        copy(ROOT/relative,Path(relative))
if args.increment==23:
    copy(ROOT/CHECKS/'auxiliary_screens_20261009T170231Z.json',CHECKS/'auxiliary_screens_20261009T170231Z.json',
        '15dd20dff83dfb39f9695577c376d446124314fde6b28e8b5d467d946cfa7424')
    table=TRAIN/'celeba_mechanism_v1/native_interim92_20261009'
    seal_sha='fd89108efec447d058f04f45bed96222b762eaf4c0d4d2df1c4b055f534b47dc'
    root_sha='ef7ff011b68eb67216e0d97e289e74e4334f0d492b12e773dc62bad19917794e'
    assert sha(ROOT/table/'FILES_SHA256.json')==seal_sha and sha(ROOT/table/'ROOT_REVIEW.json')==root_sha
    table_proof=read(ROOT/table/'ROOT_REVIEW.json')
    assert table_proof['source_seal_sha256']==seal_sha and table_proof['accepted_native92']==profile['native']
    assert table_proof['complete_scenes']==9 and table_proof['new_three_view_acceptance']==0 and not table_proof['test']
    for name,row in read(ROOT/table/'FILES_SHA256.json')['files'].items():
        copy(ROOT/table/name,table/name,row['sha256'])
    copy(ROOT/table/'FILES_SHA256.json',table/'FILES_SHA256.json',seal_sha)
    copy(ROOT/table/'ROOT_REVIEW.json',table/'ROOT_REVIEW.json',root_sha)
    for relative in ('docs/返修实验总览.md',
            'docs/server_deployment_20260923/revision_20260923/rebuttal_validation900_addendum_20261009.md'):
        copy(ROOT/relative,Path(relative))
    for base_name,package_sha in (
            ('celeba_mechanism_valid_incremental_after82_20261009','5dc4cb3f5c6d824e886b7f570b0b8c948407a6fbfddbca650a4530de66d636a7'),
            ('celeba_mechanism_valid_incremental_after82_v2_20261009','4f2232933fcf474548dba28724afb8dbb8b39c12ae42169c91e1440f51934d28')):
        base=ROOT/'tmp'/base_name;execution=base/'execution_candidate'
        assert sha(base/'PACKAGE_SHA256.json')==package_sha
        for name,row in read(base/'PACKAGE_SHA256.json')['artifacts'].items():
            copy(base/name,Path('experimental')/base_name/name,row['sha256'])
        copy(base/'PACKAGE_SHA256.json',Path('experimental')/base_name/'PACKAGE_SHA256.json',package_sha)
        for path in (base/'root_independent_review').iterdir():
            if path.is_file():copy(path,Path('experimental')/base_name/'root_independent_review'/path.name)
        for path in execution.iterdir():
            if path.is_file() and (path.name.startswith('ROOT_') or path.name in
                    ('EXECUTION_DRAFT.json','deployment_receipt.json','APPROVED.json','APPROVED.sha256','preflight.json','start_receipt.json')):
                copy(path,Path('experimental')/base_name/'execution_candidate'/path.name)
    repaired=ROOT/'tmp/celeba_mechanism_valid_incremental_after82_v2_20261009/execution_candidate'
    delta=repaired/'backups/incremental_20261009T174922Z'
    assert sha(delta/'ROOT_ADOPTION_REVIEW.json')=='b9e40d1ca565c0bcf146058433ff3e037ab4e824aa6972d1a3f3f47a088e8683'
    adoption=read(delta/'ROOT_ADOPTION_REVIEW.json')
    assert (adoption['prior_three_view_models'],adoption['accepted_new'],adoption['cumulative_three_view_models'])==(82,10,92)
    assert adoption['all_native_differences_zero'] and adoption['original82_unchanged'] and not adoption['test_inference']
    for key,name in (('archive_sha256','incremental_valid_three_views.tar.gz'),
            ('offserver_verification_sha256','OFFSERVER_VERIFICATION.json'),('backup_receipt_sha256','backup_receipt.json')):
        assert adoption[key]==sha(delta/name)
    for path in delta.iterdir():
        if path.is_file():copy(path,Path('experimental')/repaired.parent.name/'execution_candidate/backups'/delta.name/path.name)
    for name in ('prepare_after82_root_operations_20261009.py','deploy_mechanism_after82_root_20261009.py',
            'observe_mechanism_after82_root_20261009.py','prepare_after82_v2_root_operations_20261009.py',
            'deploy_mechanism_after82_v2_root_20261009.py','observe_mechanism_after82_v2_root_20261009.py',
            'prepare_after82_v2_backup_helpers_20261009.py','backup_mechanism_after82_v2_root_20261009.py',
            'adopt_mechanism_after82_v2_root_20261009.py','update_current_delivery_entries_20261009.py'):
        copy(ROOT/'tmp'/name,Path('experimental/server_reactivation_20261009')/name)
    copy(ROOT/CHECKS/'publication23_stage_attempt1.json',CHECKS/'publication23_stage_attempt1.json','26e89edefc54e09b421c6623f4220869d6a888b618d4dc944438795bd8272dd6')
    paired=TRAIN/'celeba_mechanism_v1/three_view_interim92_20261009'
    paired_seal='4ae84bc75ac08902ea6787bd525270950945f6e5166887010b41b5d40df61c7d'
    paired_root='7f432387cdc8afc97cc1972d528b3b141ec6316eddec523d6134ba0f20563546'
    assert sha(ROOT/paired/'FILES_SHA256.json')==paired_seal and sha(ROOT/paired/'ROOT_REVIEW.json')==paired_root
    accepted=read(ROOT/paired/'ROOT_REVIEW.json')
    assert accepted['complete_scenes']==9 and accepted['independent_mean_sd_scalars']==1458
    assert accepted['accepted_three_view92']==92 and accepted['new_inference']==0 and not accepted['test']
    for name,row in read(ROOT/paired/'FILES_SHA256.json')['files'].items():
        copy(ROOT/paired/name,paired/name,row['sha256'])
    copy(ROOT/paired/'FILES_SHA256.json',paired/'FILES_SHA256.json',paired_seal)
    copy(ROOT/paired/'ROOT_REVIEW.json',paired/'ROOT_REVIEW.json',paired_root)
    for name in ('review_mechanism_three_view92_root_20261009.py','adopt_auxiliary_after19_after6_root_20261009.py'):
        copy(ROOT/'tmp'/name,Path('experimental/server_reactivation_20261009')/name)
    for base_name,delta_name,total,seal_name,seal_sha,root_sha in (
            ('celeba_flgmm_screen_20261009_v2_dispatch','accepted_delta_after19_v2_20261009',26,
             'FILES_SHA256.json','e18cd7c016af1e0799e22a9dd06f69576934ac9ab831b93401f77e3aa9915c26',
             '93a5ab127fb357be93cf7b07bda6d09fab8839f73bca87406222b3f270fd1134'),
            ('celeba_hybrid_screen_execution_20261009','accepted_delta_after6_20261009',10,
             'DELIVERY_FILES_SHA256.json','6b28b22bdab9767bffc04be90a26b19214092d8bf74858ba71d4c523f2bb3988',
             'd09bd7e2181ed709d04f95a16c4ca99ecd179ab2ca7b978dfa7dd018296e8f91')):
        base=ROOT/'tmp'/base_name;delta=base/delta_name;latest=read(base/'LATEST_BACKUP.json')
        assert latest['accepted']==total and sha(base/latest['chain_file'])==latest['chain_sha256']
        assert sha(delta/seal_name)==seal_sha and sha(delta/'ROOT_ADOPTION_REVIEW.json')==root_sha
        proof=read(delta/'ROOT_ADOPTION_REVIEW.json')
        assert proof['accepted_total']==total and proof['new_inference']==0 and not proof['final_test']
        sealed=read(delta/seal_name)
        rows=sealed['members'] if seal_name.startswith('DELIVERY') else [dict(path=name,**row) for name,row in sealed['artifacts'].items()]
        for row in rows:
            copy(delta/row['path'],Path('experimental')/base_name/delta_name/row['path'],row['sha256'])
        for name in (seal_name,'ROOT_ADOPTION_REVIEW.json',read(base/latest['chain_file'])['archive']):
            copy(delta/name,Path('experimental')/base_name/delta_name/name)
        for name in ('LATEST_BACKUP.json',latest['chain_file']):
            copy(base/name,Path('experimental')/base_name/name)
    failure=ROOT/'tmp/celeba_flgmm_screen_20261009_v2_dispatch/accepted_delta_after19_20261009'
    failure_seal='0504e8b086ba6b67227b23b08f3b794c5f6637178d33249f2c26284e4c49f82e'
    assert sha(failure/'FAILURE_FILES_SHA256.json')==failure_seal
    failed=read(failure/'FAILURE_FILES_SHA256.json')
    assert failed['accepted_new']==0 and failed['archive'] is False
    for name,row in failed['artifacts'].items():
        copy(failure/name,Path('experimental')/failure.parent.name/failure.name/name,row['sha256'])
    copy(failure/'FAILURE_FILES_SHA256.json',Path('experimental')/failure.parent.name/failure.name/'FAILURE_FILES_SHA256.json',failure_seal)
if args.increment==24:
    native=ROOT/TRAIN/'celeba_mechanism_v1/native_interim100_20261009'
    paired=ROOT/TRAIN/'celeba_mechanism_v1/three_view_interim100_20261009'
    for directory,seal_name,seal_pin,root_pin in (
            (native,'FILES_SHA256.json','f42edf527aa9bf6ed94933d508c13c351b62c5fe6f25f2ea3399f82f7fc5385b','22972d2bb879680d334324c6796cdd7c044c85af98153a3262f1b8c9a494fa4a'),
            (paired,'FINAL_FILES_SHA256.json','766c08939e7f1d9fa6ab46e233ba7d2859bb0d3b9a0f418d73f5942b8eec0e88','20d1031448701938063a398fc5416e404b1e8f0549807c85dab0c0204618a2a0')):
        assert sha(directory/seal_name)==seal_pin and sha(directory/'ROOT_REVIEW.json')==root_pin
        root_review=read(directory/'ROOT_REVIEW.json')
        assert root_review['source_seal_sha256']==seal_pin and not root_review['test'] and root_review['new_inference']==0
        for name,row in read(directory/seal_name)['files'].items():copy(directory/name,(directory/name).relative_to(ROOT),row['sha256'])
        for name in (seal_name,'ROOT_REVIEW.json'):copy(directory/name,(directory/name).relative_to(ROOT))
    assert read(native/'ROOT_REVIEW.json')['scalar_checks']==540
    assert read(paired/'ROOT_REVIEW.json')['independent_mean_sd_scalars']==1620
    assert read(paired/'ROOT_REVIEW.json')['display_cells_verified']==810
    prep=ROOT/'tmp/celeba_mechanism_valid_incremental_after92_20261009';execution=prep/'execution_candidate'
    assert sha(prep/'PACKAGE_RECEIPT.json')=='a3983358664a0556677e920fcea9e2b8c3556de67f11d1d304f67559255e6a94'
    package=read(prep/'PACKAGE_RECEIPT.json')
    assert len(package['members'])==package['archive_members']==30
    assert sha(prep/'prepared_source.tar.gz')==package['source_archive_sha256']=='91a659f7876e2f7d5497f9fef4254ba5551458f2b8262797ee625906c712b9b9'
    for name,row in package['members'].items():copy(prep/name,Path('experimental')/prep.name/name,row['sha256'])
    for name in ('PACKAGE_RECEIPT.json','prepared_source.tar.gz','root_independent_review/ROOT_READY_REVIEW.json'):
        copy(prep/name,Path('experimental')/prep.name/name)
    delta=execution/'backups/incremental_20261009T183102Z';adopt=read(delta/'ROOT_ADOPTION_REVIEW.json')
    assert sha(delta/'ROOT_ADOPTION_REVIEW.json')=='9050eb059a797c70f0ca977294989b5ae5757286dbc85b36d529012cb5ab72ee'
    assert (adopt['prior_three_view_models'],adopt['accepted_new'],adopt['cumulative_three_view_models'])==(92,8,100)
    assert adopt['original92_unchanged'] and adopt['all_native_differences_zero'] and adopt['new_Full_inference']==0 and not adopt['test_inference']
    for key,name in [('archive_sha256','incremental_valid_three_views.tar.gz'),('backup_receipt_sha256','backup_receipt.json'),('offserver_verification_sha256','OFFSERVER_VERIFICATION.json')]:
        assert sha(delta/name)==adopt[key]
    for p in delta.iterdir():
        if p.is_file():copy(p,Path('experimental')/p.relative_to(ROOT/'tmp'))
    for p in execution.iterdir():
        if p.is_file():copy(p,Path('experimental')/p.relative_to(ROOT/'tmp'))
    for name in ('prepare_after92_root_operations_20261009.py','deploy_mechanism_after92_root_20261009.py',
            'observe_mechanism_after92_root_20261009.py','prepare_after92_backup_helpers_root_20261009.py',
            'backup_mechanism_after92_root_20261009.py','adopt_mechanism_after92_root_20261009.py',
            'review_native100_tables_root_20261009.py','review_mechanism_three_view100_root_20261009.py',
            'update_overview_closure100_root_20261009.py'):
        copy(ROOT/'tmp'/name,Path('experimental/server_reactivation_20261009')/name)
    for base_name,seal_name in [('celeba_mechanism_native100_tables_20261009','FILES_SHA256.json'),
            ('celeba_mechanism_three_view100_tables_20261009','FINAL_FILES_SHA256.json')]:
        directory=ROOT/'tmp'/base_name
        for name,row in read(directory/seal_name)['files'].items():copy(directory/name,Path('experimental')/base_name/name,row['sha256'])
        copy(directory/seal_name,Path('experimental')/base_name/seal_name)
    copy(ROOT/'docs/返修实验总览.md',Path('docs/返修实验总览.md'))
    addendum=Path('docs/server_deployment_20260923/revision_20260923/rebuttal_validation900_addendum_20261009.md')
    copy(ROOT/addendum,addendum)
if args.increment==25:
    review=read(backup/tag/'ROOT_INDEPENDENT_REVIEW.json')
    assert review['native_accepted']==112 and review['original204_records_exact']
    assert review['minus_U_complete_100'] and review['minus_C_partial_12']
    prep=ROOT/'tmp/celeba_mechanism_valid_C1_gate_20261009';execution=prep/'execution_candidate'
    assert sha(prep/'PACKAGE_RECEIPT.json')=='17d2eb8f11e28df3bb1a8287d1d0227103f12adf37d3b84c3f9fe3bfb5d86c30'
    for name,pin in read(prep/'PACKAGE_RECEIPT.json')['members'].items():assert sha(prep/name)==pin['sha256']
    delta=execution/'backups/incremental_20261009T185829Z'
    assert sha(delta/'ROOT_ADOPTION_REVIEW.json')=='d045665b066dafc25f9970adfdffef9c9a8a388575ec87b9b54d5dcabfa65cab'
    C1=read(delta/'ROOT_ADOPTION_REVIEW.json')
    assert (C1['prior_three_view_models'],C1['accepted_new'],C1['cumulative_three_view_models'])==(100,1,101)
    assert C1['all_native_differences_zero'] and C1['original100_unchanged'] and not C1['test_inference']
    for key,name in [('archive_sha256','incremental_valid_three_views.tar.gz'),('backup_receipt_sha256','backup_receipt.json'),('offserver_verification_sha256','OFFSERVER_VERIFICATION.json')]:
        assert C1[key]==sha(delta/name)
    for path in prep.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            copy(path,Path('experimental')/path.relative_to(ROOT/'tmp'))
    writing=ROOT/'tmp/guardfed_rebuttal_integrated100_20261009'
    canonical=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated100_20261009'
    assert sha(writing/'FINAL_FILES_SHA256.json')=='250cae33a02ba400225939e258586ba57009a812749696785190df29eabb983b'
    assert sha(writing/'ROOT_REVIEW.json')==sha(canonical/'ROOT_REVIEW.json')=='6dfb210c5d53c2badbe4fb53220008e784430fa33f9e68f0ebb81968ce18c391'
    for name,pin in read(writing/'FINAL_FILES_SHA256.json')['files'].items():
        copy(writing/name,Path('experimental')/writing.name/name,pin['sha256'])
        copy(canonical/name,(canonical/name).relative_to(ROOT),pin['sha256'])
    for name in ('FINAL_FILES_SHA256.json','ROOT_REVIEW.json'):
        copy(writing/name,Path('experimental')/writing.name/name)
        copy(canonical/name,(canonical/name).relative_to(ROOT))
    copy(ROOT/'docs/返修实验总览.md',Path('docs/返修实验总览.md'))
    for name in ('auxiliary_screens_20261009T190050Z.json','auxiliary_screens_20261009T190050Z.RAW.json'):
        copy(ROOT/CHECKS/name,CHECKS/name)
    for name in ('prepare_C1_gate_root_operations_20261009.py','deploy_mechanism_C1_gate_root_20261009.py',
            'observe_mechanism_C1_gate_root_20261009.py','prepare_C1_backup_helpers_root_20261009.py',
            'backup_mechanism_C1_gate_root_20261009.py','adopt_mechanism_C1_gate_root_20261009.py',
            'adopt_rebuttal100_root_20261009.py','capture_authorized_screens_root_20261009.py',
            'update_overview_closure100_root_20261009.py'):
        copy(ROOT/'tmp'/name,Path('experimental/server_reactivation_20261009')/name)
if args.increment==26:
    prep=ROOT/'tmp/celeba_mechanism_valid_C_after1_20261009'
    delta=prep/'execution_candidate/backups/incremental_20261009T193419Z'
    assert sha(delta/'ROOT_ADOPTION_REVIEW.json')=='8d064687ad7841e1050120a12ada9e10458fea5bb6a4d9da5aeed77584437fc5'
    adopted=read(delta/'ROOT_ADOPTION_REVIEW.json')
    assert (adopted['prior_three_view_models'],adopted['accepted_new'],adopted['cumulative_three_view_models'])==(101,11,112)
    assert adopted['all_native_differences_zero'] and adopted['original101_unchanged'] and not adopted['test_inference']
    for key,name in [('archive_sha256','incremental_valid_three_views.tar.gz'),('backup_receipt_sha256','backup_receipt.json'),('offserver_verification_sha256','OFFSERVER_VERIFICATION.json')]:
        assert adopted[key]==sha(delta/name)
    for path in prep.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            copy(path,Path('experimental')/path.relative_to(ROOT/'tmp'))
    for base_name,seal_name in [('celeba_mechanism_native_C_Benign10_20261009','FILES_SHA256.json'),
            ('celeba_mechanism_C_three_view_table_prepare_20261009','FINAL_FILES_SHA256.json')]:
        source=ROOT/'tmp'/base_name
        for item in read(source/seal_name)['members']:
            copy(source/item['path'],Path('experimental')/base_name/item['path'],item['sha256'])
        copy(source/seal_name,Path('experimental')/base_name/seal_name)
    for leaf in ('native_C_Benign10_20261009','three_view_C_Benign10_20261009'):
        canonical=ROOT/TRAIN/'celeba_mechanism_v1'/leaf
        root_review=read(canonical/'ROOT_VERIFICATION.json')
        assert root_review['mean_SD_scalars']==(54 if leaf.startswith('native_') else 162)
        assert not root_review['test'] and root_review['original_U100_unchanged']
        for path in canonical.rglob('*'):
            if path.is_file():copy(path,path.relative_to(ROOT))
    extra=ROOT/'tmp/guardfed_remaining_three_baseline_spec_decision_20261009'
    assert sha(extra/'FILES_SHA256.json')=='0e955b461a0543c615fb5307f693901a310c0479e24a433e4c6856cc8cdf427f'
    for name,pin in read(extra/'FILES_SHA256.json')['files'].items():
        copy(extra/name,Path('experimental')/extra.name/name,pin['sha256'])
    copy(extra/'FILES_SHA256.json',Path('experimental')/extra.name/'FILES_SHA256.json')
    for name in ('prepare_C_after1_root_operations_20261009.py','deploy_mechanism_C_after1_root_20261009.py',
            'observe_mechanism_C_after1_root_20261009.py','prepare_C_after1_backup_helpers_root_20261009.py',
            'backup_mechanism_C_after1_root_20261009.py','adopt_mechanism_C_after1_root_20261009.py',
            'adopt_native_C_Benign10_table_root_20261009.py','adopt_C_three_view_table_root_20261009.py',
            'update_overview_closure100_root_20261009.py'):
        copy(ROOT/'tmp'/name,Path('experimental/server_reactivation_20261009')/name)
    copy(ROOT/'docs/返修实验总览.md',Path('docs/返修实验总览.md'))
if args.increment==27:
    final=ROOT/'tmp/celeba_flgmm_final6_closure_20261009'
    assert sha(final/'ROOT_SUMMARY_ADOPTION.json')=='e602761016e199da157862da3f24c9f9d0f191cfde10fad49074540c672b4a7f'
    decision=read(final/'ROOT_SUMMARY_ADOPTION.json')
    assert decision['accepted_records']==32 and decision['new_inference']==0 and not decision['formal100_binding_or_execution']
    assert sha(final/'summary32_final/SUMMARY32.json')==decision['summary_sha256']
    assert sha(final/'FILES_SHA256.json')=='1d7cc11f95727a57478dd8575f65170024b36df98187030ac2de05e1e08cf6b9'
    for name,pin in read(final/'FILES_SHA256.json')['files'].items():assert sha(final/name)==pin['sha256']
    actual=final/'actual_20261009T194424Z';proof=read(actual/'ROOT_ADOPTION_REVIEW.json')
    assert sha(actual/'ROOT_ADOPTION_REVIEW.json')==decision['final32_root_proof_sha256']
    assert (proof['accepted_before'],proof['accepted_new'],proof['accepted_total'])==(26,6,32)
    assert sha(actual/'accepted_final6_delta.tar.gz')==proof['archive_sha256']
    for directory in (final,ROOT/'tmp/celeba_flgmm_summary32_root_independent_20261009'):
        for path in directory.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:copy(path,Path('experimental')/path.relative_to(ROOT/'tmp'))
    parent=ROOT/'tmp/celeba_flgmm_screen_20261009_v2_dispatch';latest=read(parent/'LATEST_BACKUP.json')
    assert latest['accepted']==32 and sha(parent/latest['chain_file'])==latest['chain_sha256']
    for name in ('LATEST_BACKUP.json',latest['chain_file']):copy(parent/name,Path('experimental')/parent.name/name)
    for name in ('execute_flgmm_final6_root_20261009.py','adopt_flgmm_final6_root_20261009.py',
            'capture_flgmm_original_snapshot_root_20261009.py','run_flgmm_summary32_legacy2_root_20261009.py',
            'adopt_flgmm_summary32_root_20261009.py','update_overview_closure100_root_20261009.py'):
        copy(ROOT/'tmp'/name,Path('experimental/server_reactivation_20261009')/name)
    for name in ('auxiliary_screens_20261009T195440Z.json','auxiliary_screens_20261009T195440Z.RAW.json'):
        copy(ROOT/CHECKS/name,CHECKS/name)
    copy(ROOT/'docs/返修实验总览.md',Path('docs/返修实验总览.md'))
receipt_rel=TRAIN/f'publication_closed_increment{args.increment}_20261009.json'
attributes=REPO/'.gitattributes';content=attributes.read_text();patterns=[receipt_rel.as_posix()+' -text']
if args.increment==18:
    patterns += ['experimental/celeba_mechanism_valid_incremental_next11_20261009/** -text',
        'experimental/celeba_mechanism_three_view_paired_interim_20261009/** -text']
if args.increment==20:
    patterns += ['docs/返修实验总览.md -text',
        'experimental/celeba_mechanism_rebuttal_increment_20261009/** -text',
        'experimental/celeba_mechanism_three_view_paired71_20261009/** -text',
        'experimental/guardfed_rebuttal_integrated71_20261009/** -text',
        'experimental/guardfed_rebuttal_integrated_20261009/** -text',
        'experimental/celeba_nine_method_three_view_tables_20261009/** -text',
        'outputs/guardfed_tables/celeba_nine_method_three_view_20261009/** -text',
        'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated71_20261009/** -text',
        'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim71_20261009/** -text']
if args.increment==21:
    patterns += ['experimental/celeba_flgmm_screen_20261009_v2_dispatch/accepted_delta_after13_20261009/** -text',
        'experimental/celeba_hybrid_screen_execution_20261009/accepted_delta_after4_20261009/** -text',
        'experimental/celeba_mechanism_valid_incremental_after71_20261009/** -text',
        'experimental/celeba_mechanism_remaining_variants_source_plan_20261009/** -text',
        'experimental/celeba_mechanism_after71_closure_source_review_20261009/** -text',
        'experimental/celeba_mechanism_after71_transport_v2_source_review_20261009/** -text',
        'experimental/guardfed_rebuttal_integrated71_v2_20261009/** -text',
        'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated71_v2_20261009/** -text']
if args.increment==22:
    patterns += ['outputs/guardfed_tables/celeba_nine_method_view_attribution_20261009/** -text',
        'outputs/guardfed_figures/synthetic_terminal_candidate_20261009/** -text',
        'outputs/guardfed_tables/celeba_nine_method_three_view_pdf_20261009/** -text',
        'docs/server_deployment_20260923/training_20260923/manuscript_source_locator_20261009/** -text',
        'docs/server_deployment_20260923/revision_20260923/rebuttal_validation900_addendum_20261009.md -text',
        'docs/返修实验总览.md -text']
if args.increment==23:
    patterns += ['docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_interim92_20261009/** -text',
        'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim92_20261009/** -text',
        'experimental/celeba_mechanism_valid_incremental_after82_20261009/** -text',
        'experimental/celeba_mechanism_valid_incremental_after82_v2_20261009/** -text',
        'experimental/celeba_flgmm_screen_20261009_v2_dispatch/accepted_delta_after19_20261009/** -text',
        'experimental/celeba_flgmm_screen_20261009_v2_dispatch/accepted_delta_after19_v2_20261009/** -text',
        'experimental/celeba_hybrid_screen_execution_20261009/accepted_delta_after6_20261009/** -text']
if args.increment==24:
    patterns += ['docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_interim100_20261009/** -text',
        'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim100_20261009/** -text',
        'experimental/celeba_mechanism_valid_incremental_after92_20261009/** -text',
        'experimental/celeba_mechanism_native100_tables_20261009/** -text',
        'experimental/celeba_mechanism_three_view100_tables_20261009/** -text']
if args.increment==25:
    patterns += ['experimental/celeba_mechanism_valid_C1_gate_20261009/** -text',
        'experimental/guardfed_rebuttal_integrated100_20261009/** -text',
        'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated100_20261009/** -text']
if args.increment==26:
    patterns += ['experimental/celeba_mechanism_valid_C_after1_20261009/** -text',
        'experimental/celeba_mechanism_native_C_Benign10_20261009/** -text',
        'experimental/celeba_mechanism_C_three_view_table_prepare_20261009/** -text',
        'experimental/guardfed_remaining_three_baseline_spec_decision_20261009/** -text',
        'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_C_Benign10_20261009/** -text',
        'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_Benign10_20261009/** -text']
if args.increment==27:
    patterns += ['experimental/celeba_flgmm_final6_closure_20261009/** -text',
        'experimental/celeba_flgmm_summary32_root_independent_20261009/** -text']
for pattern in patterns:
    if pattern not in content:content+='\n'+pattern+'\n'
attributes.write_text(content,newline='\n')
receipt=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_commit=PREVIOUS,
    copied_sha256=MAPPING.copy(),new_GPU_chunks_root_reviewed=reviews,baseline_valid_replays_accepted=profile['baseline'],
    mechanism_offserver_verified=profile['native'],mechanism_three_view_offserver_verified=profile['views'],
    FLGMM_offserver_verified=state['flgmm_screen32_20261009']['offserver_accepted70round_jobs'],
    Hybrid_offserver_verified=state['hybrid_screen32_20261009']['offserver_accepted70round_jobs'],
    native_tolerance=1e-12,duplicated_old_models=0,test_started=False,scientific_goal_complete=False)
if args.increment==19:receipt['next11_evaluation_started_with_no_new_offserver_acceptance']=True
if args.increment==22:
    receipt.update(increment_scope='accepted_evidence_descriptive_analysis_display_and_source_location_only',
        new_scientific_experiments=0,synthetic_figure_candidate_adopted=False,main_manuscript_edited=False,
        primary_endpoint_selected=False)
if args.increment==23:
    receipt.update(increment_scope='accepted_native_and_repaired_three_view_deltas_with_nine_scene_tables',
        new_mechanism_native_accepted=profile['native_delta'],new_mechanism_three_view_accepted=10,
        original_after82_pre_CNN_failed_attempt_preserved=True,
        auxiliary_observations_promoted_to_accepted=False,new_FLGMM_strict_offserver=7,new_Hybrid_strict_offserver=4,
        original_FLGMM_prior_schema_collector_failure_preserved=True,nine_scene_paired_table_independent_checks=1458,
        main_manuscript_edited=False,
        primary_endpoint_selected=False)
if args.increment==24:
    receipt.update(increment_scope='accepted_native104_and_complete_U100_three_view_ten_scene_tables',
        new_mechanism_native_accepted=12,new_mechanism_three_view_accepted=8,minus_U_complete_native=100,
        minus_U_complete_three_view=100,minus_C_native_partial=4,ten_scene_paired_statistic_scalars_checked=1620,
        ten_scene_paired_display_cells_checked=810,old_nine_scene_records_preserved=184,
        main_manuscript_edited=False,primary_endpoint_selected=False)
if args.increment==25:
    receipt.update(increment_scope='native112_C_single_interface_gate_and_complete_U100_24_comment_writing',
        new_mechanism_native_accepted=8,new_mechanism_three_view_accepted=1,
        minus_U_complete_native=100,minus_C_native_partial=12,minus_U_complete_three_view=100,minus_C_three_view=1,
        C_single_gate_is_not_complete_C_scene=True,complete_rebuttal_comments_verbatim=24,
        writing_numeric_pointer_checks=37,writing_author_review_only=True,main_manuscript_edited=False,
        primary_endpoint_selected=False)
if args.increment==26:
    receipt.update(increment_scope='accepted_C11_increment_and_C_IID_Benign_native_and_three_view_paired_tables',
        new_mechanism_native_accepted=0,new_mechanism_three_view_accepted=11,
        minus_U_complete_three_view=100,minus_C_three_view=12,complete_C_scenes=1,
        C_three_view_mean_SD_scalars=162,C_three_view_display_cells=81,C_native_mean_SD_scalars=54,
        other_C_scenes_complete=False,main_manuscript_edited=False,primary_endpoint_selected=False)
if args.increment==27:
    receipt.update(increment_scope='FLGMM_final6_acceptance_and_frozen32_record_only_recipe_selection',
        new_mechanism_native_accepted=0,new_mechanism_three_view_accepted=0,new_FLGMM_strict_offserver=6,
        FLGMM_summary_candidates=8,FLGMM_selected_recipe=decision['selected_recipe'],FLGMM_search_seed_n=1,
        FLGMM_summary_root_sha256=sha(final/'ROOT_SUMMARY_ADOPTION.json'),
        original_schema_failure_preserved=True,missing_legacy_host_flag_fabricated=False,
        formal100_started=False,main_manuscript_edited=False,primary_endpoint_selected=False)
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
