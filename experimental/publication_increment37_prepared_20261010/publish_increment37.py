"""Prepared closed increment37 stage entry. No commit/push or experiment execution."""
from pathlib import Path
import argparse, datetime, hashlib, importlib.util, json, re, shutil, sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
REPO=ROOT/'tmp/revision-publish-20260928'
TRAIN=Path('docs/server_deployment_20260923/training_20260923');CHECKS=TRAIN/'server_reactivation_20261009'
NATIVE=CHECKS/'mechanism_science_backups_20261009';TAG='root_delta_20261010T003657Z'
C4=Path('tmp/celeba_mechanism_valid_C_after56_20261010')
C60=TRAIN/'celeba_mechanism_v1/three_view_C_six_scenes_20261010'
PARENT='277be091be761249372c1da17c97d2bf83ef62ef';BRANCH='codex/revision-evidence-baselines-20260928'
IDS=[f'minus_C_non-IID_Benign_seed{n}' for n in range(91007,91011)]
COUNTS=dict(native=160,three_view=160,FL_new=12,Hybrid=19,baseline_valid=900)
EXCLUDED={'__pycache__','verified_extract','verified','restored','.git'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
base_path=ROOT/'tmp/publication_increment36_prepared_20261010/publish_increment36.py'
assert sha(base_path)=='7db4f0a6dfb536ff1742957ae29d1ef6183d2bfa18f1fffe408bae813caa72d3', 'Original byte-verification source changed'
spec=importlib.util.spec_from_file_location('original_increment36_byte_transport',base_path)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
git=base.git;verify_blobs=base.verify_blobs

def closed_guard(closed):
    assert not sys.flags.optimize
    assert closed['status']=='ROOT_CLOSED_INCREMENT37_INPUTS' and closed['parent_commit']==PARENT
    assert closed['counts']==COUNTS
    for key in ['C4_adoption','C60_root','C60_seal','state','formal_live','previous_publication']:
        row=closed['closure_pins'][key]
        assert isinstance(row['path'],str) and re.fullmatch('[0-9a-f]{64}',row['sha256'])

def closure_guard(proof,scope,science,execution):
    assert proof['status']=='ROOT_C_AFTER56_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
    assert (proof['prior_three_view_models'],proof['accepted_new'],proof['cumulative_three_view_models'])==(156,4,160)
    assert proof['accepted_new_ids']==scope['selected_ids']==IDS
    assert len(set(scope['excluded_prior_ids']))==len(scope['excluded_prior_ids'])==156 and not set(IDS)&set(scope['excluded_prior_ids'])
    for key in ['original156_unchanged','all_native_differences_zero','server_strict_bound_in_saved_receipts','negative_results_preserved','source_scope_complete']:assert proof[key] is True
    assert proof['new_training']==proof['new_Full_inference']==proof['new_CNN_inference_for_root_review']==0 and proof['test_inference'] is False
    assert proof['science_seal_sha256']==science and proof['execution_seal_sha256']==execution
    assert proof['prior156_root_adoption_sha256']=='a7231a724ea3a4d3443bbe196137b0c72db025cbc98d3ac66842334ab0fe9080'

def table_guard(root,tables,verification,binding,adoption_sha):
    assert root['status']=='ROOT_C60_SIX_SCENE_THREE_VIEW_TABLES_ADOPTED'
    assert (tables['unique_records'],tables['paired_models'],tables['complete_scenes'])==(120,60,6)
    assert tables['nonIID_complete_scenes']==['Benign'] and tables['full_nonIID_coverage'] is False
    assert tables['primary_endpoint_selected'] is tables['final_test'] is False and tables['new_inference']==0
    assert (verification['mean_sd_scalars'],verification['display_mean_sd_cells'],verification['receipt_metrics_from_group_counts'],verification['base_confusion_counts_structurally_checked'])==(972,486,1080,2880)
    for key in ['old100_records_bytes_exact','old810_statistics_exact','old405_cells_exact','old162_IID_seed_first_bytes_exact']:assert verification[key] is True
    assert binding['actual_C4_binding']['adoption_sha256']==adoption_sha and binding['new_inference']==0

def plan(args):
    prepared=read(HERE/'PREPARED_INPUTS.json')
    for rel,digest in prepared['reference_pins'].items():assert sha(ROOT/rel)==digest,rel
    assert sha(args.closed_inputs)==args.closed_inputs_sha256
    closed=read(args.closed_inputs);closed_guard(closed)
    def pin(key):
        row=closed['closure_pins'][key];p=(ROOT/row['path']).resolve();p.relative_to(ROOT.resolve())
        assert sha(p)==row['sha256'],key
        return p,read(p)
    adoption,proof=pin('C4_adoption');table_root,tr=pin('C60_root');table_seal,ts=pin('C60_seal')
    state_path,state=pin('state');live,health=pin('formal_live');previous_path,previous=pin('previous_publication')
    assert adoption.name=='ROOT_ADOPTION_REVIEW.json' and adoption.parent.parent==(ROOT/C4/'execution_candidate/backups').resolve()
    science=sha(ROOT/C4/'FILES_SHA256.json');execution=sha(ROOT/C4/'execution_candidate/EXECUTION_SOURCE_SHA256.json')
    scope=read(ROOT/C4/'SCOPE.json');closure_guard(proof,scope,science,execution)
    for name,key in [('incremental_valid_three_views.tar.gz','archive_sha256'),('backup_receipt.json','backup_receipt_sha256'),('OFFSERVER_VERIFICATION.json','offserver_verification_sha256')]:assert sha(adoption.parent/name)==proof[key]
    off=read(adoption.parent/'OFFSERVER_VERIFICATION.json')
    assert off['status']=='INCREMENTAL_INDEPENDENT_SAVED_ARRAYS_THREE_VIEWS_PASS' and off['accepted_new_ids']==off['all_accepted_ids']==IDS
    assert (off['independent_metric_checks'],off['independent_confusion_count_checks'],off['prediction_rule_checks'])==(36,96,12)
    assert table_root==(ROOT/C60/'ROOT_VERIFICATION.json').resolve() and table_seal.parent==(ROOT/C60).resolve()
    table_guard(tr,read(ROOT/C60/'snapshot/tables.json'),read(ROOT/C60/'snapshot/verification.json'),read(ROOT/C60/'snapshot/SOURCE_BINDINGS.json'),sha(adoption))
    assert state_path==(ROOT/TRAIN/'TRAINING_STATE.json').resolve()
    main=state['celeba_mechanism_v1']
    assert main['scientific_results_offserver_verified']==main['three_view_new_models_offserver_verified']==160
    assert set(main['three_view_accepted_ids'])==set(scope['excluded_prior_ids'])|set(IDS) and main['test_started'] is False
    assert state['final_evaluator_runtime_20261009']['actual_native_valid_image_replays_accepted']==900
    assert state['flgmm_fullcoverage_v2_20261009']['new_accepted']==12 and state['flgmm_fullcoverage_v2_20261009']['final_test'] is False
    hy=state['hybrid_screen32_20261009'];assert hy['offserver_accepted70round_jobs']==19 and hy['selected_recipe'] is None and hy['final_test'] is False
    assert live.parent==(ROOT/CHECKS).resolve() and re.fullmatch(r'root_live_\d{8}T\d{6}Z.json',live.name) and not health['failed'] and not health['failure_files']
    assert previous_path==(ROOT/TRAIN/'publication_closed_increment36_verified_20261010.json').resolve()
    assert previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and previous['commit']==PARENT and previous['branch']==BRANCH
    assert [previous[k] for k in ['mechanism_offserver_verified','mechanism_three_view_offserver_verified','FLGMM_fullcoverage_new_offserver_verified','Hybrid_offserver_verified']]==[156,156,11,19]
    prior_receipt=ROOT/TRAIN/'publication_closed_increment36_20261010.json';assert sha(prior_receipt)==previous['publication_receipt_sha256']
    hybrid=ROOT/prepared['unchanged_Hybrid_root_path']
    base.hybrid_unchanged_guard(read(hybrid),sha(hybrid),read(prior_receipt),'experimental/'+hybrid.relative_to(ROOT/'tmp').as_posix(),prepared['unchanged_Hybrid_root_sha256'])
    fl=read(ROOT/prepared['FL_root_path']);assert (fl['accepted_before'],fl['accepted_new'],fl['accepted_total'])==(11,1,12)
    assert fl['accepted_new_ids']==['FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed91003_fullcoverage'] and len(set(fl['accepted_job_ids']))==12
    assert fl['previous_root_adoption_sha256']=='6feb41c9f2f06980d29865ca03d59e5d2cffeeb2a6f0d6f0209f065cf80caf80'
    assert fl['old_models_repacked']==fl['root_new_CNN']==0 and fl['final_test'] is False and fl['negative_results_preserved'] is True
    latest=read(ROOT/'tmp/celeba_flgmm_fullcoverage_incremental_20261009/LATEST_BACKUP.json')
    assert latest['accepted_total']==12 and latest['root_adoption_sha256']==sha(ROOT/prepared['FL_root_path'])
    assert set(prepared['required_extra_paths'])==set(closed['extra_pins'])
    assert closed['extra_pins'][(CHECKS/'latest_formal_live.json').as_posix()]==sha(live)
    ex=(ROOT/C4/'execution_candidate').resolve();runtime=closed['C4_runtime_pins']
    assert set(prepared['required_C4_runtime_names'])<={Path(n).name for n in runtime}
    assert any(re.fullmatch(r'ROOT_PROGRESS_\d{8}T\d{6}Z.json',Path(n).name) and d==proof['remote_terminal_proof_sha256'] for n,d in runtime.items())
    assert all((ROOT/n).resolve().parent==ex for n in runtime)
    reader_failure=read(ROOT/'tmp/celeba_mechanism_C_after56_source_review_20261010/ROOT_TRANSPORT_READER_FAILURE.json')
    transport_review=read(ROOT/'tmp/celeba_mechanism_C_after56_source_review_20261010/ROOT_TRANSPORT_REVIEW.json')
    assert reader_failure['scientific_acceptance_granted_by_this_record'] is False and reader_failure['actual_deployment_receipt_sha256']==sha(ex/'deployment_receipt.json')
    assert transport_review['actual_deployment_receipt_already_exists'] is True and transport_review['local_seal_reader_failure_preserved'] is True
    assert transport_review['review_timing'].startswith('Supplemental transport fixtures ran after actual deployment;')
    assert git('rev-parse','HEAD')==PARENT and git('branch','--show-current')==BRANCH
    assert not git('status','--porcelain','--untracked-files=all'),'Publish worktree must be clean'
    tracked=set(git('ls-tree','-r','--name-only',PARENT).splitlines());sources={};mapping={}
    def add(path,destination=None,expected=None):
        path=Path(path).resolve();rel=path.relative_to(ROOT.resolve());assert not EXCLUDED&set(rel.parts)
        assert path.is_file() and path.suffix.lower() not in {'.pt','.pth','.pem','.key','.npz'} and path.name!='.env' and path.stat().st_size<100_000_000
        assert 'celeba_hybrid_screen_execution_20261009' not in rel.parts and 'rebuttal_integrated_C50_20261010' not in rel.parts
        recovery=ROOT/'tmp/celeba_baselines/source_recovery_novel_20261010'
        if path.is_relative_to(recovery):assert path.parent==recovery and path.name in prepared['source_recovery_published_subset'],'Third-party source body remains local'
        digest=sha(path);assert expected is None or digest==expected,rel
        dest=Path(destination) if destination else (Path('experimental')/rel.relative_to('tmp') if rel.parts[0]=='tmp' else rel)
        name=dest.as_posix();assert not dest.is_absolute() and '..' not in dest.parts and not any(c in name for c in '\n\r"')
        if path.suffix.lower() in {'.json','.py','.md','.txt','.log','.sh','.patch','.template'}:assert not re.search(rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|sk-proj-[A-Za-z0-9_-]{30,}',path.read_bytes()),rel
        if name.endswith(('.tar.gz','.tar')):assert name not in tracked,'Old archive must not be republished: '+name
        assert name not in mapping or mapping[name]==digest
        sources[name]=path;mapping[name]=digest
    def sealed(folder,seal):
        data=read(folder/seal);rows=data.get('members') or [{'path':n,**(v if isinstance(v,dict) else {'sha256':v})} for n,v in data['files'].items()]
        for row in rows:
            p=(folder/row['path']).resolve();assert p.is_relative_to(folder.resolve());add(p,expected=row['sha256'])
        add(folder/seal)
        return {row['path'] for row in rows}
    for row in prepared['ready_manifest']:add(ROOT/row['source'],row['destination'],row['sha256'])
    for p in sorted(adoption.parent.iterdir()):
        if p.is_file():add(p)
    for rel,digest in runtime.items():assert re.fullmatch('[0-9a-f]{64}',digest);add(ROOT/rel,expected=digest)
    table_members=sealed(table_seal.parent,table_seal.name)
    assert {'snapshot/records.json','snapshot/tables.json','snapshot/verification.json','snapshot/TABLES.md','snapshot/SOURCE_BINDINGS.json'}<=table_members
    add(table_root)
    for p in [state_path,live,previous_path,args.closed_inputs.resolve()]:add(p)
    for rel,digest in closed['extra_pins'].items():assert re.fullmatch('[0-9a-f]{64}',digest);add(ROOT/rel,expected=digest)
    for rel,digest in closed.get('root_helper_pins',{}).items():
        p=(ROOT/rel).resolve();assert p.parent==(ROOT/'tmp').resolve() and p.suffix in {'.py','.json'} and re.search(r'C60|C_after56|publication37',p.name)
        assert re.fullmatch('[0-9a-f]{64}',digest);add(p,expected=digest)
    sealed(HERE,'FILES_SHA256.json')
    archives=[d for n,d in mapping.items() if n.endswith(('.tar.gz','.tar'))]
    assert len(archives)==len(set(archives)) and sum(p.stat().st_size for p in sources.values())<100_000_000
    args.C4_adoption=adoption;args.formal_live=live;args.C60_root=table_root
    return sources,mapping,proof,health

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closed-inputs',type=Path,required=True);parser.add_argument('--closed-inputs-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute-stage',action='store_true')
    args=parser.parse_args();assert args.execute_stage,'Prepared only: explicit --execute-stage required'
    output=args.output.resolve();assert output.parent==HERE and not output.exists()
    sources,mapping,proof,health=plan(args)
    output.mkdir()
    receipt=dict(status='INCREMENT37_SOURCE_BYTES_PREPARED_FOR_INDEX',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_commit=PARENT,copied_sha256=mapping.copy(),C4_root_adoption_path=args.C4_adoption.relative_to(ROOT).as_posix(),C4_root_adoption_sha256=sha(args.C4_adoption),C60_root_path=args.C60_root.relative_to(ROOT).as_posix(),C60_root_sha256=sha(args.C60_root),baseline_valid_replays_accepted=900,mechanism_offserver_verified=160,mechanism_three_view_offserver_verified=160,FLGMM_offserver_verified=32,FLGMM_fullcoverage_new_offserver_verified=12,Hybrid_offserver_verified=19,formal_live_path=args.formal_live.relative_to(ROOT).as_posix(),formal_live_sha256=sha(args.formal_live),observed_main_terminal=health['queue_completed'],observed_main_active=len(health['active']),observed_terminal_is_not_acceptance=True,closed_inputs_sha256=args.closed_inputs_sha256,C60_table_included=True,C50_full_reply_republished=False,source_recovery_subset=read(HERE/'PREPARED_INPUTS.json')['source_recovery_published_subset'],third_party_source_bodies_redistributed=False,transport_reader_failure_preserved=True,supplemental_transport_review_post_deployment=True,new_GPU_chunks_root_reviewed=[],duplicated_old_models=0,test_started=False,scientific_goal_complete=False,scope='Closed validation evidence only; no recipe/endpoint choice, complete mechanism claim, manuscript application or final test.')
    receipt_path=output/'publication_closed_increment37_20261010.json';receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    receipt_rel=(TRAIN/receipt_path.name).as_posix();sources[receipt_rel]=receipt_path;mapping[receipt_rel]=sha(receipt_path)
    try:
        for name,source in sources.items():
            assert sha(source)==mapping[name],name
            destination=REPO/name;assert destination.resolve().is_relative_to(REPO.resolve())
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
            assert sha(destination)==mapping[name],name
        attributes=REPO/'.gitattributes';original=attributes.read_bytes() if attributes.exists() else b''
        attributes.write_bytes(original+(b'' if not original or original.endswith(b'\n') else b'\n')+''.join('"'+n+'" -text\n' for n in mapping).encode('utf-8'))
        mapping['.gitattributes']=sha(attributes);names=list(mapping)
        for start in range(0,len(names),25):git('add','-f','--',*names[start:start+25])
        for start in range(0,len(names),25):git('add','--renormalize','--',*names[start:start+25])
        verify_blobs(':',mapping)
        changed=set(filter(None,git('diff','--cached','--name-only','-z',binary=True).decode().split('\0')))
        assert changed<=set(mapping) and git('rev-parse','HEAD')==PARENT
        result=dict(status='INDEX_BLOB_SHA_PASS_NOT_COMMITTED_OR_PUSHED',staged_paths=len(changed),blob_paths_verified=len(mapping),receipt_sha256=sha(receipt_path),index_sha256=mapping)
        (output/'INDEX_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
    except Exception as error:
        (output/'FAILURE.json').write_text(json.dumps(dict(status='FAILED_PRESERVE_WORKTREE_AND_INDEX_NO_AUTOMATIC_RETRY',error=repr(error)),indent=2)+'\n',encoding='utf-8');raise

if __name__=='__main__':main()
