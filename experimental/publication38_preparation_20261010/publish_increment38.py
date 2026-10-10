"""Stage only the adopted increment38 delta. Explicit execution; never commit/push."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,re,shutil,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REPO=ROOT/'tmp/revision-publish-20260928'
TRAIN=Path('docs/server_deployment_20260923/training_20260923');CHECKS=TRAIN/'server_reactivation_20261009'
REPLY=Path('docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C60_20261010')
FL=Path('tmp/celeba_flgmm_fullcoverage_delta_after12_20261010')
HY=Path('tmp/celeba_hybrid_screen_execution_20261009/accepted_delta_after19_20261010')
PARENT='8c71a8f76ac69fbbf9aa84d77a6264d294509daf';BRANCH='codex/revision-evidence-baselines-20260928'
COUNTS=dict(native=160,three_view=160,FL_new=14,Hybrid=21,baseline_valid=900)
EXCLUDED={'__pycache__','verified_extract','verified','restored','.git'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
base_path=ROOT/'tmp/publication_increment37_prepared_20261010/publish_increment37.py'
assert sha(base_path)=='fe4294384a92357b74dcbbc0e8b65f86afbe9c63463668da1c1f06445495d8c9'
spec=importlib.util.spec_from_file_location('original_increment37_byte_transport',base_path)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
git=base.git;verify_blobs=base.verify_blobs

def closed_guard(c):
    assert not sys.flags.optimize
    assert c['status']=='ROOT_CLOSED_INCREMENT38_INPUTS' and c['parent_commit']==PARENT and c['counts']==COUNTS
    assert set(c['closure_pins'])=={'C60_reply_root','FL_root','Hybrid_root','state','formal_live','previous_publication'}
    for row in c['closure_pins'].values():assert isinstance(row['path'],str) and re.fullmatch('[0-9a-f]{64}',row['sha256'])

def reply_guard(r):
    assert r['status']=='ROOT_C60_COMPLETE_AUTHOR_REVIEW_TEXT_REVERSIBLE_DELTA_AND_SOURCE_POINTERS_PASS'
    assert r['root_C60_table_sha256']=='f2465c5e81291deca92535adcd0b0a29c49b3f76c2330927aa3db50925a9c742'
    assert r['prior_C50_reply_root_sha256']=='64b083fb61cb8bff17304031af01a0d692d03fdc16e956de0dcd154f5a524691'
    assert tuple(r[k] for k in ('original_comments_verbatim','changed_spans','C_scalar_pointer_checks','new_numeric_cells','direction_checks','links_checked','complete_C_scenes'))==(24,10,36,18,27,50,6)
    assert r['author_review_only'] and r['original_numeric_displays_preserved'] and r['all_negative_results_retained']
    assert not r['manuscript_applied'] and not r['final_test'] and not r['whole_rebuttal_complete']

def delta_guard(fl,hy):
    assert fl['status']=='ROOT_FL96_LINKED_DELTA_ARCHIVE_SOURCE_CHECKPOINT_AND_ORIGINAL_STRICT_BINDING_PASS'
    assert tuple(fl[k] for k in ('accepted_before','accepted_new','accepted_total'))==(12,2,14)
    assert len(set(fl['accepted_job_ids']))==14 and len(set(fl['accepted_new_ids']))==2
    assert fl['old_models_repacked']==fl['root_new_CNN']==0 and not fl['final_test'] and fl['negative_results_preserved']
    assert fl['planned_new']==96 and fl['reused_separately']==4
    assert hy['status']=='ROOT_LOCKED_AUXILIARY_DELTA_ARCHIVE_MEMBER_AND_ORIGINAL_RECORD_ACCEPTANCE_PASS'
    assert tuple(hy[k] for k in ('accepted_before','accepted_new','accepted_total'))==(19,2,21)
    assert hy['new_inference']==0 and not hy['selection_performed'] and not hy['scientific_changes'] and not hy['final_test'] and not hy['formal100_started']

def plan(args):
    prep=read(HERE/'PREPARED_INPUTS.json')
    for rel,pin in prep['reference_pins'].items():assert sha(ROOT/rel)==pin
    assert sha(args.closed_inputs)==args.closed_inputs_sha256
    c=read(args.closed_inputs);closed_guard(c)
    def pin(key,expected):
        row=c['closure_pins'][key];p=(ROOT/row['path']).resolve();assert p==expected.resolve() and sha(p)==row['sha256']
        return p,read(p)
    reply,r=pin('C60_reply_root',ROOT/REPLY/'ROOT_REVIEW.json');reply_guard(r)
    assert sha(reply)=='b2e3958ed4000fe0365929c94408c411cd47c70e637c5151bb40603ad097098a'
    flp,fl=pin('FL_root',ROOT/FL/'ROOT_ADOPTION_REVIEW.json');hyp,hy=pin('Hybrid_root',ROOT/HY/'ROOT_ADOPTION_REVIEW.json');delta_guard(fl,hy)
    assert fl['previous_root_adoption_sha256']==prep['FL_prior_root_sha256']
    flprev=read(ROOT/FL/'PREVIOUS_OFFSERVER_ACCEPTANCE.json')
    assert fl['accepted_job_ids'][:12]==flprev['accepted_job_ids'] and not set(flprev['accepted_job_ids'])&set(fl['accepted_new_ids'])
    assert fl['accepted_job_ids'][12:]==fl['accepted_new_ids']
    assert fl['source_package_sha256']=='6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230'
    for name,key in [('accepted_delta.tar.gz','archive_sha256'),('OFFSERVER_ACCEPTANCE.json','offserver_acceptance_sha256'),('BACKUP_SHA256.json','server_receipt_sha256')]:assert sha(ROOT/FL/'batch'/name)==fl[key]
    for name,key in [('hybrid_after19_delta.tar.gz','archive_sha256'),('OFFSERVER_MEMBER_TENSOR_PROOF.json','offserver_proof_sha256'),('PARTIAL_ACCEPTANCE.json','strict_receipt_sha256'),('ROOT_READY_CHAIN_LINK.json','ready_link_sha256')]:assert sha(ROOT/HY/name)==hy[key]
    link=read(ROOT/HY/'ROOT_READY_CHAIN_LINK.json');assert (link['previous_accepted'],link['accepted_new'],link['accepted_total_if_root_adopts'])==(19,2,21)
    assert link['previous_root_adoption_sha256']==prep['Hybrid_prior_root_sha256'] and link['no_old_models_repacked'] and not link['recipe_selection']
    assert len(set(link['accepted_job_ids']))==21 and len(set(link['accepted_new_ids']))==2 and link['accepted_job_ids'][19:]==link['accepted_new_ids']
    previous_hy=read(ROOT/HY/'PREVIOUS_CHAIN.json');assert link['accepted_job_ids'][:19]==previous_hy['accepted_job_ids']
    state_path,state=pin('state',ROOT/TRAIN/'TRAINING_STATE.json');m=state['celeba_mechanism_v1']
    assert m['scientific_results_offserver_verified']==m['three_view_new_models_offserver_verified']==160 and not m['test_started']
    assert state['latest_rebuttal_draft']['root_proof_sha256']==sha(reply) and state['latest_rebuttal_draft']['complete_C_scenes']==6
    assert state['flgmm_fullcoverage_v2_20261009']['new_accepted']==14
    assert state['hybrid_screen32_20261009']['offserver_accepted70round_jobs']==21 and state['hybrid_screen32_20261009']['selected_recipe'] is None
    assert state['final_evaluator_runtime_20261009']['actual_native_valid_image_replays_accepted']==900
    lp=c['closure_pins']['formal_live'];live=(ROOT/lp['path']).resolve();assert live.parent==(ROOT/CHECKS).resolve() and re.fullmatch(r'root_live_\d{8}T\d{6}Z.json',live.name) and sha(live)==lp['sha256'];health=read(live)
    assert not health['failed'] and not health['failure_files']
    prevp,prev=pin('previous_publication',ROOT/TRAIN/'publication_closed_increment37_verified_20261010.json')
    assert prev['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and prev['commit']==PARENT and prev['branch']==BRANCH
    assert sha(ROOT/TRAIN/'publication_closed_increment37_20261010.json')==prev['publication_receipt_sha256']
    latest_fl=read(ROOT/'tmp/celeba_flgmm_fullcoverage_incremental_20261009/LATEST_BACKUP.json');assert latest_fl['accepted_total']==14 and latest_fl['root_adoption_sha256']==sha(flp)
    latest_hy=read(ROOT/HY.parent/'LATEST_BACKUP.json');chain=ROOT/HY.parent/'BACKUP_CHAIN_accepted_delta_after19_20261010.json'
    assert latest_hy['accepted']==21 and latest_hy['chain_file']==chain.name and latest_hy['chain_sha256']==sha(chain) and read(chain)['accepted_job_ids']==link['accepted_job_ids']
    assert set(c['extra_pins'])==set(prep['required_extra_paths']) and c['extra_pins'][(CHECKS/'latest_formal_live.json').as_posix()]==sha(live)
    assert git('rev-parse','HEAD')==PARENT and git('branch','--show-current')==BRANCH and not git('status','--porcelain','--untracked-files=all')
    tracked=set(git('ls-tree','-r','--name-only',PARENT).splitlines());sources={};mapping={}
    def add(path,expected=None):
        path=Path(path);assert not path.is_symlink();path=path.resolve();rel=path.relative_to(ROOT.resolve())
        assert not EXCLUDED&set(rel.parts) and path.is_file() and path.suffix.lower() not in {'.pt','.pth','.pem','.key','.npz'} and path.name!='.env' and path.stat().st_size<100_000_000
        assert 'three_view_C_six_scenes_20261010' not in rel.parts and 'source_recovery_novel_20261010' not in rel.parts
        digest=sha(path);assert expected is None or digest==expected
        dest=Path('experimental')/rel.relative_to('tmp') if rel.parts[0]=='tmp' else rel;name=dest.as_posix()
        assert not dest.is_absolute() and '..' not in dest.parts and not any(x in name for x in '\n\r"')
        if path.suffix.lower() in {'.json','.py','.md','.txt','.log','.sh','.patch'}:assert not re.search(rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|sk-proj-[A-Za-z0-9_-]{30,}',path.read_bytes())
        if name.endswith(('.tar.gz','.tar')):assert name not in tracked
        assert name not in mapping or mapping[name]==digest;sources[name]=path;mapping[name]=digest
    def sealed(folder,seal,expected,omit=()):
        assert sha(folder/seal)==expected;obj=read(folder/seal);rows=obj.get('members') or [{'path':n,**v} for n,v in obj['files'].items()]
        for row in rows:
            p=(folder/row['path']).resolve();assert p.is_relative_to(folder.resolve()) and sha(p)==row['sha256']
            if row['path'] not in omit:add(p,row['sha256'])
        add(folder/seal);return rows
    rows=sealed(ROOT/REPLY,'FILES_SHA256.json',r['source_seal_sha256']);assert len(rows)==11
    assert sha(ROOT/REPLY/'rebuttal_integrated_20261009.md')==r['rebuttal_sha256'] and sha(ROOT/REPLY/'manuscript_insertions_integrated_20261009.md')==r['insertions_sha256']
    sealed(ROOT/FL,'DELIVERY_FILES_SHA256.json',fl['delivery_seal_sha256'])
    # The identical scientific body is already published; retain an explicit recovery reference.
    omit='local_record_bridge_v2/checked_record_body.py';assert sha(ROOT/HY/omit)==prep['Hybrid_omitted_body']['sha256']==sha(ROOT/prep['Hybrid_omitted_body']['existing_source'])
    verify_blobs(PARENT+':',{prep['Hybrid_omitted_body']['published_source']:prep['Hybrid_omitted_body']['sha256']})
    sealed(ROOT/HY,'DELIVERY_FILES_SHA256.json',hy['delivery_seal_sha256'],[omit]);add(ROOT/HY/'ROOT_DELIVERY_COPY.json')
    for p in (reply,flp,hyp,state_path,live,prevp,ROOT/TRAIN/'publication_closed_increment37_20261010.json',args.closed_inputs.resolve()):add(p)
    for rel,digest in c['extra_pins'].items():assert re.fullmatch('[0-9a-f]{64}',digest);add(ROOT/rel,digest)
    sealed(HERE,'FILES_SHA256.json',sha(HERE/'FILES_SHA256.json'))
    archives=[d for n,d in mapping.items() if n.endswith(('.tar.gz','.tar'))];assert len(archives)==len(set(archives)) and sum(p.stat().st_size for p in sources.values())<100_000_000
    args.formal_live=live;args.C60_reply_root=reply;args.FL_root=flp;args.Hybrid_root=hyp
    return sources,mapping,health


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closed-inputs',type=Path,required=True);parser.add_argument('--closed-inputs-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute-stage',action='store_true')
    args=parser.parse_args();assert args.execute_stage,'Prepared only: explicit --execute-stage required'
    output=args.output.resolve();assert output.parent==HERE and not output.exists()
    sources,mapping,health=plan(args)
    output.mkdir()
    receipt=dict(status='INCREMENT38_SOURCE_BYTES_PREPARED_FOR_INDEX',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_commit=PARENT,copied_sha256=mapping.copy(),
        C60_reply_root_path=args.C60_reply_root.relative_to(ROOT).as_posix(),C60_reply_root_sha256=sha(args.C60_reply_root),FL_root_path=args.FL_root.relative_to(ROOT).as_posix(),FL_root_sha256=sha(args.FL_root),Hybrid_root_path=args.Hybrid_root.relative_to(ROOT).as_posix(),Hybrid_root_sha256=sha(args.Hybrid_root),
        baseline_valid_replays_accepted=900,mechanism_offserver_verified=160,mechanism_three_view_offserver_verified=160,FLGMM_offserver_verified=32,FLGMM_fullcoverage_new_offserver_verified=14,Hybrid_offserver_verified=21,
        formal_live_path=args.formal_live.relative_to(ROOT).as_posix(),formal_live_sha256=sha(args.formal_live),observed_main_terminal=health['queue_completed'],observed_main_active=len(health['active']),observed_terminal_is_not_acceptance=True,
        closed_inputs_sha256=args.closed_inputs_sha256,C60_full_reply_included=True,C60_table_republished=False,third_party_source_bodies_redistributed=False,Hybrid_scientific_body_republished=False,
        omitted_scientific_body_reference=read(HERE/'PREPARED_INPUTS.json')['Hybrid_omitted_body'],duplicated_old_models=0,test_started=False,scientific_goal_complete=False,manuscript_applied=False,scope='Adopted C60 author-review text and exact auxiliary deltas only; no new endpoint, manuscript application or final test.')
    receipt_path=output/'publication_closed_increment38_20261010.json';receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
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
