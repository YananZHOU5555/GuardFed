"""Stage exact adopted C10 closure/C70 table bytes; root must explicitly execute. Never commit/push."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,re,shutil,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REPO=ROOT/'tmp/revision-publish-20260928'
TRAIN=Path('docs/server_deployment_20260923/training_20260923');CHECKS=TRAIN/'server_reactivation_20261009'
C10=Path('tmp/celeba_mechanism_valid_C_after60_20261010');BACKUP=C10/'execution_candidate/backups/incremental_20261010T015546Z'
C70=TRAIN/'celeba_mechanism_v1/three_view_C_seven_scenes_20261010'
PARENT='3601c9dfca63dc1c6203aceb2c7fc066faa630fa';BRANCH='codex/revision-evidence-baselines-20260928'
COUNTS=dict(native=170,three_view=170,FL_new=16,Hybrid=22,baseline_valid=900)
IDS=[f'minus_C_non-IID_F Flip_seed{s}' for s in range(91001,91011)]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
base_path=ROOT/'tmp/publication39_preparation_20261010/publish_increment39.py'
assert sha(base_path)=='44d408568c013adc1f183965d263ef3c0c576f0046630b197bb3fa6d08f9f36b'
spec=importlib.util.spec_from_file_location('pub39_byte_transport',base_path);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
git=base.git;verify_blobs=base.verify_blobs

def closed_guard(c):
    assert not sys.flags.optimize
    assert c['status']=='ROOT_CLOSED_INCREMENT40_INPUTS' and c['parent_commit']==PARENT and c['counts']==COUNTS
    assert set(c['closure_pins'])=={'C10_root','C70_root','state','formal_live','previous_publication'}
    for row in c['closure_pins'].values():assert isinstance(row['path'],str) and re.fullmatch('[0-9a-f]{64}',row['sha256'])

def closure_guard(r):
    assert r['status']=='ROOT_C_AFTER60_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
    assert (r['prior_three_view_models'],r['accepted_new'],r['cumulative_three_view_models'])==(160,10,170)
    assert r['accepted_new_ids']==IDS and r['archive_members_verified']==103 and r['content_members_verified']==102
    assert r['original160_unchanged'] and r['all_native_differences_zero'] and r['negative_results_preserved'] and r['source_scope_complete']
    assert r['new_training']==r['new_Full_inference']==r['new_CNN_inference_for_root_review']==0 and not r['test_inference']
    assert r['science_seal_sha256']=='ed8ecc84781b799e205c9e139dce8e753cb5545139b7e48ef23d5f8d07a50a77'
    assert r['execution_seal_sha256']=='12c1b612d9c9697f6a537344ebace5aaa7c1bf5d7bbe760a3866168adca89896'

def table_guard(r):
    assert r['status']=='ROOT_C70_SEVEN_SCENE_THREE_VIEW_TABLES_ADOPTED'
    assert (r['unique_records'],r['paired_models'],r['complete_scenes'],r['mean_SD_scalars_recomputed'],r['display_cells'])==(140,70,7,1134,567)
    assert r['nonIID_complete_scenes']==['Benign','F Flip'] and r['IID_complete_scenes']==5 and r['seed_panels']==[10,9,6]
    assert r['all_negative_results_retained'] and r['old120_records_preserved'] and r['old162_IID_aggregate_bytes_exact']
    assert not r['full_nonIID_coverage'] and not r['whole_rebuttal_complete'] and not r['incorporated_into_full_rebuttal'] and not r['test']
    assert r['new_CNN']==r['new_training']==r['new_Full_inference']==0

def plan(args):
    prep=read(HERE/'PREPARED_INPUTS.json')
    for rel,digest in prep['reference_pins'].items():assert sha(ROOT/rel)==digest
    assert sha(args.closed_inputs)==args.closed_inputs_sha256
    c=read(args.closed_inputs);closed_guard(c)
    def pin(key,expected):
        row=c['closure_pins'][key];p=(ROOT/row['path']).resolve();assert p==expected.resolve() and sha(p)==row['sha256'];return p,read(p)
    adoption,r=pin('C10_root',ROOT/BACKUP/'ROOT_ADOPTION_REVIEW.json');closure_guard(r)
    assert sha(adoption)=='7e687f5a050aa0496b9a8c2b3606bd8787c138de6679e436dee3eb5b60df2dc8'
    for name,key in [('incremental_valid_three_views.tar.gz','archive_sha256'),('backup_receipt.json','backup_receipt_sha256'),('OFFSERVER_VERIFICATION.json','offserver_verification_sha256')]:assert sha(adoption.parent/name)==r[key]
    off=read(adoption.parent/'OFFSERVER_VERIFICATION.json')
    assert off['status']=='INCREMENTAL_INDEPENDENT_SAVED_ARRAYS_THREE_VIEWS_PASS' and off['accepted_new_ids']==IDS
    assert (off['independent_metric_checks'],off['independent_confusion_count_checks'],off['prediction_rule_checks'])==(90,240,30)
    table_root,tr=pin('C70_root',ROOT/C70/'ROOT_VERIFICATION.json')
    # Actual root adoption schema and hashes are pinned in PREPARED_INPUTS after adoption exists.
    assert sha(table_root)==prep['C70_root_sha256'];table_guard(tr)
    assert tr['C10_root_adoption_sha256']==sha(adoption) and sha(ROOT/prep['C70_review_path'])==tr['independent_review_sha256']
    seal=read(ROOT/C70/'ACTUAL_FILES_SHA256.json');assert sha(ROOT/C70/'ACTUAL_FILES_SHA256.json')==prep['C70_seal_sha256'] and len(seal['files'])==35
    for rel,row in seal['files'].items():assert sha(ROOT/C70/rel)==row['sha256']
    review=read(ROOT/prep['C70_review_path'])
    assert review['status']=='INDEPENDENT_C70_SEVEN_SCENE_ARITHMETIC_AND_PROVENANCE_PASS_NO_ADOPTION'
    assert (review['unique_records'],review['paired_models'],review['complete_scenes'],review['mean_SD_scalars_recomputed'],review['display_cells'],review['count_metrics_recomputed'],review['confusion_count_checks'])==(140,70,7,1134,567,1260,3360)
    assert review['actual_C10_root_adoption_sha256']==sha(adoption) and review['old120_record_JSON_bytes_and_order_exact'] and review['old972_scalars_exact'] and review['old486_display_cells_exact'] and review['old162_IID_aggregate_bytes_exact']
    assert not review['whole_mechanism_complete'] and not review['whole_rebuttal_complete'] and not review['test']
    state_path,state=pin('state',ROOT/TRAIN/'TRAINING_STATE.json');main=state['celeba_mechanism_v1']
    assert main['scientific_results_offserver_verified']==main['three_view_new_models_offserver_verified']==170 and not main['test_started']
    scope=read(ROOT/C10/'SCOPE.json');assert scope['selected_ids']==IDS and len(scope['excluded_prior_ids'])==160
    assert set(main['three_view_accepted_ids'])==set(scope['excluded_prior_ids'])|set(IDS)
    assert state['flgmm_fullcoverage_v2_20261009']['new_accepted']==16 and state['hybrid_screen32_20261009']['offserver_accepted70round_jobs']==22
    assert state['final_evaluator_runtime_20261009']['actual_native_valid_image_replays_accepted']==900
    assert state['latest_rebuttal_draft']['complete_C_scenes']==6 and state['latest_rebuttal_draft']['root_proof_sha256']=='b2e3958ed4000fe0365929c94408c411cd47c70e637c5151bb40603ad097098a'
    lp=c['closure_pins']['formal_live'];live=(ROOT/lp['path']).resolve();assert live.parent==(ROOT/CHECKS).resolve() and re.fullmatch(r'root_live_\d{8}T\d{6}Z.json',live.name) and sha(live)==lp['sha256'];health=read(live)
    assert not health['failed'] and not health['failure_files']
    previous_path,previous=pin('previous_publication',ROOT/TRAIN/'publication_closed_increment39_verified_20261010.json')
    assert previous['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and previous['commit']==PARENT and previous['branch']==BRANCH
    assert [previous[k] for k in ['mechanism_offserver_verified','mechanism_three_view_offserver_verified','FLGMM_fullcoverage_new_offserver_verified','Hybrid_offserver_verified']]==[170,160,16,22]
    assert sha(ROOT/TRAIN/'publication_closed_increment39_20261010.json')==previous['publication_receipt_sha256']
    assert set(c['extra_pins'])==set(prep['required_extra_paths']) and c['extra_pins'][(CHECKS/'latest_formal_live.json').as_posix()]==sha(live)
    assert git('rev-parse','HEAD')==PARENT and git('branch','--show-current')==BRANCH and not git('status','--porcelain','--untracked-files=all')
    verify_blobs(PARENT+':',prep['parent_recovery_blobs'])
    tracked=set(git('ls-tree','-r','--name-only',PARENT).splitlines());sources={};mapping={}
    def add(path,expected=None):
        path=Path(path);assert not path.is_symlink();path=path.resolve();rel=path.relative_to(ROOT.resolve())
        assert not {'__pycache__','verified_extract','restored','.git'}&set(rel.parts) and path.is_file() and path.suffix.lower() not in {'.pt','.pth','.pem','.key','.npz'} and path.name!='.env' and path.stat().st_size<100_000_000
        assert 'rebuttal_integrated_C60_20261010' not in rel.parts and 'three_view_C_six_scenes_20261010' not in rel.parts
        digest=sha(path);assert expected is None or digest==expected
        dest=Path('experimental')/rel.relative_to('tmp') if rel.parts[0]=='tmp' else rel;name=dest.as_posix();assert not dest.is_absolute() and '..' not in dest.parts and not any(x in name for x in '\n\r"')
        if path.suffix.lower() in {'.json','.py','.md','.txt','.log','.sh','.patch'}:assert not re.search(rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|sk-proj-[A-Za-z0-9_-]{30,}',path.read_bytes())
        if name.endswith(('.tar.gz','.tar')):assert name not in tracked
        assert name not in mapping or mapping[name]==digest;sources[name]=path;mapping[name]=digest
    for row in prep['ready_manifest']:add(ROOT/row['path'],row['sha256'])
    for p in [adoption,table_root,state_path,live,previous_path,ROOT/TRAIN/'publication_closed_increment39_20261010.json',args.closed_inputs.resolve()]:add(p)
    for rel,digest in c['extra_pins'].items():add(ROOT/rel,digest)
    seal=read(HERE/'FILES_SHA256.json')
    for name,row in seal['files'].items():add(HERE/name,row['sha256'])
    add(HERE/'FILES_SHA256.json')
    archives=[d for n,d in mapping.items() if n.endswith(('.tar','.tar.gz'))];assert len(archives)==len(set(archives))==1 and sum(p.stat().st_size for p in sources.values())<100_000_000
    args.C10_root=adoption;args.C70_root=table_root;args.formal_live=live
    return sources,mapping,health


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closed-inputs',type=Path,required=True);parser.add_argument('--closed-inputs-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute-stage',action='store_true')
    args=parser.parse_args();assert args.execute_stage,'Prepared only: explicit --execute-stage required'
    output=args.output.resolve();assert output.parent==HERE and not output.exists()
    sources,mapping,health=plan(args)
    output.mkdir()
    receipt=dict(status='INCREMENT40_SOURCE_BYTES_PREPARED_FOR_INDEX',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_commit=PARENT,copied_sha256=mapping.copy(),
        C10_root_adoption_path=args.C10_root.relative_to(ROOT).as_posix(),C10_root_adoption_sha256=sha(args.C10_root),C70_root_path=args.C70_root.relative_to(ROOT).as_posix(),C70_root_sha256=sha(args.C70_root),
        baseline_valid_replays_accepted=900,mechanism_offserver_verified=170,mechanism_three_view_offserver_verified=170,FLGMM_offserver_verified=32,FLGMM_fullcoverage_new_offserver_verified=16,Hybrid_offserver_verified=22,
        formal_live_path=args.formal_live.relative_to(ROOT).as_posix(),formal_live_sha256=sha(args.formal_live),observed_main_terminal=health['queue_completed'],observed_main_active=len(health['active']),observed_terminal_is_not_acceptance=True,
        closed_inputs_sha256=args.closed_inputs_sha256,C70_table_included=True,C70_in_full_rebuttal=False,C60_full_reply_republished=False,third_party_source_bodies_redistributed=False,
        duplicated_old_models=0,test_started=False,scientific_goal_complete=False,manuscript_applied=False,scope='Actual C10 three-view closure and C70 seven-scene table only; C60 complete author-review prose unchanged. Three non-IID C scenarios and six other controls remain incomplete; no final test.')
    receipt_path=output/'publication_closed_increment40_20261010.json';receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
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
