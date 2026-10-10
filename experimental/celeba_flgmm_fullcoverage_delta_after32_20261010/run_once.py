"""Exact six frozen FL96 records; prepared only until root source review."""
from pathlib import Path
import argparse, ast, difflib, hashlib, json, runpy, sys, traceback
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
R = H.parents[1]
PRIOR = R/'tmp/celeba_flgmm_fullcoverage_delta_after28_20261010'
ORIGINAL = R/'tmp/celeba_flgmm_fullcoverage_delta_after22_20261010'
OBS = R/'tmp/celeba_aux_live_followup_20261010_afterHybrid32/attempt_v2'
S = Path('F:/YananResearchStorage/GuardFed')/H.name/'batch'
PREVIOUS = '254f5b844f5f302a320c5eaaa33efe790e53337c69ee31bae0f237361b5472b6'
ROOTPROOF = 'ae1e65bf764f3ed3e6657e95fa3798617788c8659ac30eb542035a0031663cf3'
SNAPSHOT = '821aef3b5b8593b96f17566c30c2a33618d3e049c9baa77a42058b9c81bf6932'
IDS = [f'FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed{s}_fullcoverage' for s in range(91005,91011)]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())

def save(name, value):
    with (H/name).open('x', encoding='utf8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')

def reused():
    seal = read(PRIOR/'DELIVERY_FILES_SHA256.json')
    assert sha(PRIOR/'run_once.py') == seal['files']['run_once.py']['sha256']
    ns = runpy.run_path(str(PRIOR/'run_once.py'), run_name='sealed_after28_transport')
    g = ns['reused']()
    original_cpu106 = g['cpu106']
    def cpu110(text):
        text = original_cpu106(text)
        for a,b in [('CPU106','CPU110'),('{106}','{110}'),('106 in a','110 in a'),
                    ('106 not in aff','110 not in aff'),('106 in aff','110 in aff'),
                    ('CPU=106','CPU=110'),('-c 106 ','-c 110 '),("'106'","'110'"),
                    ('(106,1,10)','(110,1,10)'),('helper_CPU=106','helper_CPU=110')]:
            text = text.replace(a,b)
        return text
    g.update(H=H, S=S, PREVIOUS=PREVIOUS, ROOTPROOF=ROOTPROOF, cpu106=cpu110)
    source = (ORIGINAL/'run_once.py').read_text('utf8'); tree = ast.parse(source)
    changes = {
        'parent':[('{22+len(ids)}','{32+len(ids)}',1)],
        'verify':[('{22+len(ids)}','{32+len(ids)}',1)],
        'finalize':[
            ('total=22+n','total=32+n',1),
            ("['INITIAL_GUIDE','INITIAL_OWNER','SNAPSHOT','GUIDE','OWNER','PREFLIGHT','COLLECT','SERVER_SHA','SCP','VERIFY','CLOSED']", "['SNAPSHOT','GUIDE','OWNER','PREFLIGHT','COLLECT','SERVER_SHA','SCP','VERIFY','CLOSED']",1),
            ("prior['accepted_total']==22","prior['accepted_total']==32",1),
            ("p['accepted_job_ids'][:22]","p['accepted_job_ids'][:32]",1),
            ('prior_accepted_new=22','prior_accepted_new=32',1),
            ('old22_ordered_prefix_exact','old32_ordered_prefix_exact',2),
            ('accepted_before=22','accepted_before=32',1),
            ('# Actual FL96 after22:','# Actual FL96 after32:',1),
            ('ordered prior22','ordered prior32',1)]}
    for name,rebindings in changes.items():
        node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        body = ast.get_source_segment(source,node)
        for before,after,count in rebindings:
            assert body.count(before)==count,(name,before)
            body = body.replace(before,after)
        exec(compile(cpu110(body),str(ORIGINAL/'run_once.py')+'[COUNT_CPU110_PATH_ONLY]', 'exec'),g)
    return g

def prepare(g):
    assert not S.exists() and not S.is_symlink(), 'Existing raw evidence must not be overwritten'
    assert sha(OBS/'SNAPSHOT.json')==SNAPSHOT and read(OBS/'COMMAND.json')['returncode']==0
    snapshot=read(OBS/'SNAPSHOT.json');f=snapshot['FLGMM']
    assert not f['queue_failed'] and not f['failure_paths'] and not f['source']['changed_members']
    assert not snapshot['source_data_binding']['changed_paths']
    assert not snapshot['resources']['restricted_thread_owners']['110']
    latest=read(g['B']/'LATEST_BACKUP.json');priorpath=Path(latest['next_collector_previous_path'])
    assert latest['accepted_total']==32 and latest['root_adoption_sha256']==ROOTPROOF
    assert sha(R/latest['root_adoption_path'])==ROOTPROOF and sha(priorpath)==latest['next_collector_previous_sha256']==PREVIOUS
    prior=read(priorpath);terminal=f['terminal_ids']
    assert len(prior['accepted_job_ids'])==len(set(prior['accepted_job_ids']))==32
    assert set(prior['accepted_job_ids'])<=set(terminal)
    assert [i for i in terminal if i not in prior['accepted_job_ids']]==IDS and len(terminal)==38
    for name,path in [('PREVIOUS_LATEST.json',g['B']/'LATEST_BACKUP.json'),('PREVIOUS_OFFSERVER_ACCEPTANCE.json',priorpath)]:
        with (H/name).open('xb') as out:out.write(path.read_bytes())
    command=read(OBS/'COMMAND.json')
    save('SNAPSHOT_COMMAND.json',dict(exit_code=command['returncode'],original_command_path=(OBS/'COMMAND.json').relative_to(R).as_posix(),original_command_sha256=sha(OBS/'COMMAND.json'),original_start=command['start'],original_end=command['end'],executed_again=False))
    save('AUTHORIZED_SNAPSHOT.json',dict(status='PREPARED_FIXED_ONE_SNAPSHOT_EXACT6_NOT_ACCEPTANCE',snapshot_path=(OBS/'SNAPSHOT.json').relative_to(R).as_posix(),snapshot_sha256=SNAPSHOT,snapshot_utc=snapshot['utc'],prior_count=32,terminal_count=38,authorized_ids=IDS,source_hashes={'PACKAGE_SHA256.json':f['source']['sha256'],'manifest.json':f['manifest_sha256']},prior_root_path=latest['root_adoption_path'],prior_root_sha256=ROOTPROOF,prior_offserver_sha256=PREVIOUS,no_future_terminal_ids_allowed=True,root_adoption_required=True,new_accepted=0))
    old=g['O']/'collect_delta.py';assert sha(old)=='575ecbd7f948ad66d58d28cb336100da1efc965559ee05ba500f3ef259d98c20'
    original=old.read_text('utf8')
    before='    authorized_ids='+repr(['FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed'+str(s)+'_fullcoverage' for s in (91006,91007)])+'\n'
    edits=[(before,'    authorized_ids='+repr(IDS)+'\n'),
           ("    queue_snapshot=read(RELEASE/'queue_progress.json')\n","    queue_snapshot=read(RELEASE/'queue_progress.json')\n    assert not queue_snapshot['failed'], 'Frozen queue failure forbids collection'\n"),
           ('    wanted=[identity for identity in wanted if identity in authorized_ids]\n','    wanted=[identity for identity in wanted if identity in authorized_ids]\n    assert wanted==authorized_ids, \'All six authorized outputs must be terminal, unique and ordered\'\n'),
           ("    queue=read(RELEASE/'queue_progress.json')\n","    queue=read(RELEASE/'queue_progress.json')\n    assert not queue['failed'], 'Queue failure forbids collection'\n")]
    effective=original
    for a,b in edits:
        assert effective.count(a)==1;effective=effective.replace(a,b)
    def loop(body):
        node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name=='execute')
        return ast.get_source_segment(body,next(n for n in node.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='identity'))
    assert loop(effective)==loop(original)
    assert effective[effective.index('    before=repo_identity'):]==original[original.index('    before=repo_identity'):]
    wrapper="from pathlib import Path\nimport hashlib\np=Path(%r)\ns=p.read_text('utf8')\nassert hashlib.sha256(p.read_bytes()).hexdigest()==%r\nfor a,b in %r:\n    assert s.count(a)==1\n    s=s.replace(a,b)\nexec(compile(s,__file__,'exec'),globals())\n"%('/workspace/guardfed_checks/celeba_flgmm_fullcoverage_delta_after14_20261010/collect_delta.py',sha(old),edits)
    with (H/'collect_delta.py').open('x',encoding='utf8',newline='\n') as out:out.write(wrapper)
    save('SOURCE_REUSE.json',dict(parent_collector_path=old.relative_to(R).as_posix(),parent_collector_sha256=sha(old),thin_entry_sha256=sha(H/'collect_delta.py'),effective_collector_sha256=hashlib.sha256(effective.encode()).hexdigest(),sole_effective_change='Exact6 authorized IDs and pre-science exact-completeness/queue.failed refusal guards',per_ID_scientific_loop_bytes_exact=True,archive_and_strict_body_bytes_exact=True,original_verifier_path=(g['B']/'verify_delta_offserver.py').relative_to(R).as_posix(),original_verifier_sha256=sha(g['B']/'verify_delta_offserver.py'),source_seal_sha256=sha(g['B']/'FILES_SHA256.json'),scientific_body_unchanged=True,parent_transport_sha256='5b00a63088fea7294faa12dafae656f1ff4312e9b151a14bd7cfe1eb806136db',reused_after22_transport_sha256=sha(ORIGINAL/'run_once.py'),reused_after28_transport_sha256=sha(PRIOR/'run_once.py'),resource_metadata_change='CPU106 -> CPU110; one thread/nice10/idleIO/CUDA-hidden unchanged'))
    with (H/'SOURCE_DIFF.patch').open('x',encoding='utf8',newline='\n') as out:
        out.write(''.join(difflib.unified_diff(original.splitlines(True),effective.splitlines(True),fromfile='original_after14_collector',tofile='effective_exact6_collector')))
    # Pure metadata guard checks; no scientific imports, SSH or models.
    assert IDS==list(IDS) and len(IDS)==len(set(IDS))==6 and not set(IDS)&set(prior['accepted_job_ids'])
    guards=ast.parse("assert wanted==authorized_ids\nassert not queue_snapshot['failed']")
    code=compile(guards,'EXACT6_METADATA_GUARDS','exec')
    exec(code,dict(wanted=IDS,authorized_ids=IDS,queue_snapshot={'failed':False}))
    for wanted,failed in ((IDS[:-1],False),(IDS+[IDS[0]],False),(list(reversed(IDS)),False),(IDS,True)):
        try:exec(code,dict(wanted=wanted,authorized_ids=IDS,queue_snapshot={'failed':failed}))
        except AssertionError:pass
        else:raise AssertionError('Metadata refusal guard failed')
    save('PREPARE_CHECKS.json',dict(status='PREPARED_SOURCE_AND_METADATA_ONLY_PASS_NOT_EXECUTED',exact_ids=IDS,old32_ids_excluded=True,positive_exact6=True,missing_duplicate_wrong_order_refused=True,per_ID_scientific_loop_bytes_exact=True,archive_and_strict_body_bytes_exact=True,original_verifier_bytes_exact=True,CPU_metadata110_only=True,no_SSH_or_Torch_or_scientific_execution=True,volume=g['storage'](0)))

def gate(g,path,expected):
    assert path and expected and sha(path)==expected
    review=read(path)
    assert review['source_adoptable'] is True and review['exact_selected_ids']==IDS
    assert review['prepared_seal_sha256']==sha(H/'PREPARED_FILES_SHA256.json')
    for name,pin in read(H/'PREPARED_FILES_SHA256.json')['files'].items():
        assert sha(H/name)==pin['sha256'] and (H/name).stat().st_size==pin['bytes']
    assert sha(g['B']/'LATEST_BACKUP.json')==sha(H/'PREVIOUS_LATEST.json')
    assert sha(Path(read(H/'PREVIOUS_LATEST.json')['next_collector_previous_path']))==PREVIOUS
    save('ROOT_SOURCE_REVIEW_BINDING.json',dict(review_path=path.as_posix(),review_sha256=expected,prepared_seal_sha256=review['prepared_seal_sha256'],exact_selected_ids=IDS,source_review_only=True))

if __name__=='__main__':
    try:
        parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','collect','verify','finalize'))
        parser.add_argument('--review',type=Path);parser.add_argument('--review-sha256');args=parser.parse_args();g=reused()
        if args.phase=='prepare': prepare(g)
        elif args.phase=='collect': gate(g,args.review,args.review_sha256);g['parent'](IDS).collect()
        elif args.phase=='verify': g['verify']()
        else:g['finalize']()
    except BaseException as error:
        if not isinstance(error,SystemExit):save('FAILURE_'+str(__import__('time').time_ns())+'.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise
