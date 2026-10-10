"""Explicit file-manifest increment. Reuse sealed40 byte transport; never commit/push."""
from pathlib import Path
import argparse,ast,datetime,hashlib,importlib.util,json,re,shutil,sys,textwrap
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REPO=ROOT/'tmp/revision-publish-20260928'
TRAIN=Path('docs/server_deployment_20260923/training_20260923');PARENT='59ec6455c1402ff3bfdac454cbf8de9daf0d216d';BRANCH='codex/revision-evidence-baselines-20260928'
COUNTS=dict(native=180,three_view=170,FL_new=18,Hybrid=23,baseline_valid=900)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
BASE=ROOT/'tmp/publication40_preparation_20261010/publish_increment40.py'
# Exact source pin is filled from the already published40 file before sealing.
BASE_SHA='392ce239dbbfafd0de08c60122dbeaa18e8b6abefb3b5f18dce46b84d889afb4'
assert sha(BASE)==BASE_SHA
spec=importlib.util.spec_from_file_location('sealed40_transport',BASE);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
git=base.git;verify_blobs=base.verify_blobs

def closed_guard(c):
    assert not sys.flags.optimize and c['status']=='ROOT_CLOSED_INCREMENT41_INPUTS'
    assert c['parent_commit']==PARENT and c['counts']==COUNTS
    assert set(c['roots'])=={'native','FL','Hybrid','state','live','previous_publication'}

def evidence_guard(n,f,h,state):
    assert n['status']=='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS' and n['total_new_strict_and_offserver']==180
    assert n['new_ids']==[f'minus_C_non-IID_FedSA_seed{s}' for s in range(91001,91011)] and n['oldFull_models_repacked']==0 and not n['test']
    assert (f['accepted_before'],f['accepted_new'],f['accepted_total'],f['planned_new'],f['reused_separately'])==(16,2,18,96,4)
    assert (h['accepted_before'],h['accepted_new'],h['accepted_total'])==(22,1,23)
    assert f['old_models_repacked']==f['root_new_CNN']==h['new_inference']==0
    assert not f['final_test'] and not h['final_test'] and not h['selection_performed'] and not h['formal100_started']
    m=state['celeba_mechanism_v1'];assert m['scientific_results_offserver_verified']==180 and m['three_view_new_models_offserver_verified']==170 and not m['test_started']
    assert state['flgmm_fullcoverage_v2_20261009']['new_accepted']==18 and state['hybrid_screen32_20261009']['offserver_accepted70round_jobs']==23
    assert state['final_evaluator_runtime_20261009']['actual_native_valid_image_replays_accepted']==900
    assert state['latest_rebuttal_draft']['complete_C_scenes']==6 and state['latest_rebuttal_draft']['root_proof_sha256']=='b2e3958ed4000fe0365929c94408c411cd47c70e637c5151bb40603ad097098a'

def plan(args):
    assert sha(args.closed_inputs)==args.closed_inputs_sha256;c=read(args.closed_inputs);closed_guard(c)
    def pinned(row):
        p=(ROOT/row['path']).resolve();assert p.is_relative_to(ROOT) and p.is_file() and sha(p)==row['sha256'];return p
    roots={k:pinned(v) for k,v in c['roots'].items()};d={k:read(p) for k,p in roots.items()}
    evidence_guard(d['native'],d['FL'],d['Hybrid'],d['state'])
    assert c['frozen_snapshot_destinations'][c['roots']['state']['path']]==(TRAIN/'TRAINING_STATE.json').as_posix()
    for row in c['artifact_bindings']:pinned(row)
    chain=c['native_chain'];ledger=read(pinned(chain['current']));prior=read(pinned(chain['previous']))
    assert len(ledger['entries'])==27 and ledger['entries'][:-1]==prior['entries'] and len(prior['entries'])==26
    assert ledger['entries'][-1]['receipt_sha256']==d['native']['receipt_sha256'] and sha(pinned(chain['current']))==d['native']['ledger_sha256']
    native_receipt=read(pinned(chain['receipt']));native_off=read(pinned(chain['offserver']))
    assert native_receipt['accepted_new_ids']==native_off['accepted_new_ids']==d['native']['new_ids'] and native_off['pass'] and native_off['members_verified']==110
    assert native_receipt['previous_receipt_sha256']==prior['entries'][-1]['receipt_sha256'] and native_receipt['reused_full_weights_repacked']==0
    independent=read(pinned(c['native_independent']))
    assert independent['native_accepted']==180 and independent['added_n']==10 and independent['source_root_proof_sha256']==sha(roots['native'])
    assert independent['original270_raw_json_record_bytes_exact'] and independent['ledger_previous26_entries_exact'] and not independent['three_view_acceptance_changed']
    prev=d['previous_publication'];assert prev['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and prev['commit']==PARENT and prev['branch']==BRANCH
    assert not d['live']['failed'] and not d['live']['failure_files']
    assert roots['live'].parent==ROOT/TRAIN/'server_reactivation_20261009' and re.fullmatch(r'root_live_\d{8}T\d{6}Z.json',roots['live'].name)
    assert git('rev-parse','HEAD')==PARENT and git('branch','--show-current')==BRANCH and not git('status','--porcelain','--untracked-files=all')
    verify_blobs(PARENT+':',c['parent_recovery_blobs'])
    for sealrow in c['sealed_deliveries']:
        p=pinned(sealrow);seal=read(p);rows=seal.get('members') or [dict(path=n,**v) for n,v in seal['files'].items()]
        assert len(rows)==sealrow['members']
        for row in rows:
            q=(p.parent/row['path']).resolve();assert q.is_relative_to(p.parent) and sha(q)==row['sha256']
    sources={};mapping={};tracked=set(git('ls-tree','-r','--name-only',PARENT).splitlines())
    def add(path,digest,destination=None):
        p=Path(path);assert not p.is_symlink() and sha(p)==digest;p=p.resolve();rel=p.relative_to(ROOT)
        assert not {'__pycache__','restored','verified_extract','.git'}&set(rel.parts) and p.is_file() and p.suffix.lower() not in {'.pt','.pth','.npz','.key','.pem'} and p.name!='.env' and p.stat().st_size<100_000_000
        assert not any('three_view_C_' in x or 'rebuttal_integrated_C' in x for x in rel.parts)
        dest=destination or (Path('experimental')/rel.relative_to('tmp') if rel.parts[0]=='tmp' else rel).as_posix();assert not any(x in dest for x in '\n\r"') and not Path(dest).is_absolute() and '..' not in Path(dest).parts
        if destination:assert c['frozen_snapshot_destinations'].get(rel.as_posix())==destination
        if dest.endswith(('.tar','.tar.gz')):assert dest not in tracked
        if p.suffix.lower() in {'.json','.py','.md','.txt','.log','.sh','.patch'}:assert not re.search(rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|sk-proj-[A-Za-z0-9_-]{30,}',p.read_bytes())
        assert dest not in mapping or mapping[dest]==digest;sources[dest]=p;mapping[dest]=digest
    for row in c['files']:add(ROOT/row['path'],row['sha256'],c['frozen_snapshot_destinations'].get(row['path']))
    for p in roots.values():add(p,sha(p),c['frozen_snapshot_destinations'].get(p.relative_to(ROOT).as_posix()))
    add(args.closed_inputs.resolve(),args.closed_inputs_sha256)
    for n,row in read(HERE/'FILES_SHA256.json')['files'].items():add(HERE/n,row['sha256'])
    add(HERE/'FILES_SHA256.json',sha(HERE/'FILES_SHA256.json'))
    archives={n:v for n,v in mapping.items() if n.endswith(('.tar','.tar.gz'))}
    assert archives==c['new_archives'] and len(set(archives.values()))==3
    assert sum(p.stat().st_size for p in sources.values())<100_000_000
    return sources,mapping,d

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--closed-inputs',type=Path,required=True);ap.add_argument('--closed-inputs-sha256',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--execute-stage',action='store_true');args=ap.parse_args();assert args.execute_stage
    output=args.output.resolve();assert output.parent==HERE and not output.exists()
    sources,mapping,d=plan(args);output.mkdir()
    receipt=dict(status='INCREMENT41_SOURCE_BYTES_PREPARED_FOR_INDEX',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_commit=PARENT,copied_sha256=mapping.copy(),baseline_valid_replays_accepted=900,mechanism_offserver_verified=180,mechanism_three_view_offserver_verified=170,FLGMM_offserver_verified=32,FLGMM_fullcoverage_new_offserver_verified=18,Hybrid_offserver_verified=23,observed_main_terminal=d['live']['queue_completed'],observed_main_active=len(d['live']['active']),observed_terminal_is_not_acceptance=True,closed_inputs_sha256=args.closed_inputs_sha256,C70_table_included=False,C70_in_full_rebuttal=False,C60_full_reply_republished=False,third_party_source_bodies_redistributed=False,duplicated_old_models=0,test_started=False,scientific_goal_complete=False,manuscript_applied=False)
    receipt_path=output/'publication_closed_increment41_20261010.json';receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n');receipt_rel=(TRAIN/receipt_path.name).as_posix();sources[receipt_rel]=receipt_path;mapping[receipt_rel]=sha(receipt_path)
    # Execute the exact existing40 copy/-f/-text/index/failure block, not a reimplementation.
    source=BASE.read_text('utf8');mainnode=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='main');node=next(n for n in mainnode.body if isinstance(n,ast.Try))
    block=textwrap.dedent('\n'.join(source.splitlines()[node.lineno-1:node.end_lineno]))
    exec(compile(block,str(BASE)+':byte_transport','exec'),dict(globals(),sources=sources,mapping=mapping,output=output,receipt_path=receipt_path))

if __name__=='__main__':main()
