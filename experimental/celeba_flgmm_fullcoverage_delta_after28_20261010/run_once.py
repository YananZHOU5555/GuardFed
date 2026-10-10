"""One frozen FL96 delta; reuse original scientific collector/verifier unchanged."""
from pathlib import Path
import ast,datetime,difflib,hashlib,json,runpy,sys,traceback
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[1]
ORIGINAL=R/'tmp/celeba_flgmm_fullcoverage_delta_after22_20261010'
OBS=R/'tmp/celeba_aux_live_followup_20261010T065319Z'
S=Path('F:/YananResearchStorage/GuardFed')/H.name/'batch'
PREVIOUS='bf84b750013d143dacc16b93fb2094339c79483e8951be7b07a8f9a317915491'
ROOTPROOF='e51007c549f8cc95970ce06cb61b4a477c11eff5d00e05a9aeac9ffb30dd449c'
SNAPSHOT='5836e593ccf980a8e6bd9116ab2dbe26ee2b54fb4dd8f025cb81ed3d933f271c'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())

def save(name,value):
    with (H/name).open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def reused():
    path=ORIGINAL/'run_once.py'
    seal=read(ORIGINAL/'DELIVERY_FILES_SHA256.json')
    assert sha(path)==seal['files']['run_once.py']['sha256']
    ns=runpy.run_path(str(path),run_name='sealed_prior_transport')
    g=ns['parent'].__globals__
    g.update(H=H,S=S,PREVIOUS=PREVIOUS,ROOTPROOF=ROOTPROOF)
    source=path.read_text('utf8');tree=ast.parse(source)
    changes={
        'parent':[('{22+len(ids)}','{28+len(ids)}',1)],
        'verify':[('{22+len(ids)}','{28+len(ids)}',1)],
        'finalize':[
            ('total=22+n','total=28+n',1),
            ("['INITIAL_GUIDE','INITIAL_OWNER','SNAPSHOT','GUIDE','OWNER','PREFLIGHT','COLLECT','SERVER_SHA','SCP','VERIFY','CLOSED']","['SNAPSHOT','GUIDE','OWNER','PREFLIGHT','COLLECT','SERVER_SHA','SCP','VERIFY','CLOSED']",1),
            ("prior['accepted_total']==22","prior['accepted_total']==28",1),
            ("p['accepted_job_ids'][:22]","p['accepted_job_ids'][:28]",1),
            ('prior_accepted_new=22','prior_accepted_new=28',1),
            ('old22_ordered_prefix_exact','old28_ordered_prefix_exact',2),
            ('accepted_before=22','accepted_before=28',1),
            ('# Actual FL96 after22:','# Actual FL96 after28:',1),
            ('ordered prior22','ordered prior28',1)
        ]
    }
    for name,rebindings in changes.items():
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        body=ast.get_source_segment(source,node)
        for before,after,count in rebindings:
            assert body.count(before)==count,(name,before)
            body=body.replace(before,after)
        exec(compile(body,str(path)+'[COUNT_PATH_ONLY_REUSE]', 'exec'),g)
    assert sha(R/'tmp/guardfed_local_storage.py')=='2482f2f46243abafd33ea2fadc94d555c0d80327fefe19caea3eb8f3f6bb3d63'
    g['source_check']()
    return g

def prepare(g):
    assert not S.exists() and not S.is_symlink()
    assert sha(OBS/'SNAPSHOT.json')==SNAPSHOT and read(OBS/'COMMAND.json')['exit_code']==0
    snapshot=read(OBS/'SNAPSHOT.json');f=snapshot['FLGMM'];findings=read(OBS/'FINDINGS.json')
    assert findings['source_and_failure_clean'] and findings['CPU106_free'] and not findings['existing_collectors']
    latest=read(g['B']/'LATEST_BACKUP.json');priorpath=Path(latest['next_collector_previous_path'])
    assert latest['accepted_total']==28 and latest['root_adoption_sha256']==ROOTPROOF
    assert sha(R/latest['root_adoption_path'])==ROOTPROOF and sha(priorpath)==latest['next_collector_previous_sha256']==PREVIOUS
    prior=read(priorpath);terminal=[r['id'] for r in f['rows'] if r['observed_terminal']]
    assert len(prior['accepted_job_ids'])==len(set(prior['accepted_job_ids']))==28
    assert set(prior['accepted_job_ids'])<=set(terminal)
    ids=[i for i in terminal if i not in prior['accepted_job_ids']]
    assert ids==findings['FL_new_ids'] and len(ids)==4 and len(terminal)==32
    assert ids==['FLGMM_Tg20_L2.0_lr0.001_IID_FedSA_seed91010_fullcoverage']+['FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed'+str(s)+'_fullcoverage' for s in (91002,91003,91004)]
    for name,path in [('PREVIOUS_LATEST.json',g['B']/'LATEST_BACKUP.json'),('PREVIOUS_OFFSERVER_ACCEPTANCE.json',priorpath),('SNAPSHOT.json',OBS/'SNAPSHOT.json'),('SNAPSHOT_COMMAND.json',OBS/'COMMAND.json')]:
        with (H/name).open('xb') as out:out.write(path.read_bytes())
    save('AUTHORIZED_SNAPSHOT.json',dict(status='FIXED_ONE_SNAPSHOT_EXACT_DELTA_NOT_ACCEPTANCE',snapshot_sha256=SNAPSHOT,snapshot_utc=snapshot['utc'],prior_count=28,terminal_count=32,authorized_ids=ids,source_hashes={'PACKAGE_SHA256.json':f['source_package_sha256'],'manifest.json':f['manifest_sha256']},prior_root_path=latest['root_adoption_path'],prior_root_sha256=ROOTPROOF,prior_offserver_sha256=PREVIOUS,no_future_terminal_ids_allowed=True,root_adoption_required=True))
    old=g['O']/'collect_delta.py';assert sha(old)=='575ecbd7f948ad66d58d28cb336100da1efc965559ee05ba500f3ef259d98c20'
    text=old.read_text('utf8')
    before='    authorized_ids='+repr(['FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed'+str(s)+'_fullcoverage' for s in (91006,91007)])+'\n'
    after='    authorized_ids='+repr(ids)+'\n';assert text.count(before)==1
    effective=text.replace(before,after)
    def scientific_loop(body):
        node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name=='execute')
        loop=next(n for n in node.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='identity')
        return ast.get_source_segment(body,loop)
    assert scientific_loop(effective)==scientific_loop(text)
    wrapper="from pathlib import Path\nimport hashlib\np=Path(%r)\ns=p.read_text('utf8')\nassert hashlib.sha256(p.read_bytes()).hexdigest()==%r\na=%r\nb=%r\nassert s.count(a)==1\nexec(compile(s.replace(a,b),__file__,'exec'),globals())\n"%('/workspace/guardfed_checks/celeba_flgmm_fullcoverage_delta_after14_20261010/collect_delta.py',sha(old),before,after)
    with (H/'collect_delta.py').open('x',encoding='utf8',newline='\n') as out:out.write(wrapper)
    save('SOURCE_REUSE.json',dict(parent_collector_path=old.relative_to(R).as_posix(),parent_collector_sha256=sha(old),thin_entry_sha256=sha(H/'collect_delta.py'),effective_collector_sha256=hashlib.sha256(effective.encode()).hexdigest(),sole_effective_change='authorized_ids bound to the one 06:53:40UTC snapshot delta',per_ID_scientific_loop_bytes_exact=True,original_verifier_path=(g['B']/'verify_delta_offserver.py').relative_to(R).as_posix(),original_verifier_sha256=sha(g['B']/'verify_delta_offserver.py'),source_seal_sha256=sha(g['B']/'FILES_SHA256.json'),scientific_body_unchanged=True,parent_transport_sha256='5b00a63088fea7294faa12dafae656f1ff4312e9b151a14bd7cfe1eb806136db',reused_after22_transport_sha256=sha(ORIGINAL/'run_once.py'),resource_metadata_change='Original CPU106 one thread/nice10/idleIO/CUDA-hidden unchanged'))
    with (H/'SOURCE_DIFF.md').open('x',encoding='utf8',newline='\n') as out:
        out.write('# Exact4 metadata rebinding\n\nOriginal after14 scientific collector body and per-ID loop are byte exact; sole effective collector change is the frozen authorized ID literal. Original after22 transport functions are reused in memory, changing only prior22/counts to prior28, independent output namespace and actual previous/root hashes. The SNAPSHOT command is the one already successful source/guide/CPU observation; redundant INITIAL_GUIDE/INITIAL_OWNER/SNAPSHOT SSH calls are omitted. collect retains its own actual guide/owner/preflight before one strict call. No original verifier, source/data/method/seed/rule/service is modified.\n\n')
        out.write('```diff\n'+''.join(difflib.unified_diff(text.splitlines(True),effective.splitlines(True),fromfile='original_after14_collector',tofile='effective_exact4_collector'))+'```\n')
    print(json.dumps(dict(prepared=True,prior28_exact=True,authorized_ids=ids,scientific_loop_exact=True)),flush=True)

if __name__=='__main__':
    try:
        phase=sys.argv[1];g=reused()
        if phase=='prepare':prepare(g)
        elif phase=='collect':g['parent'](read(H/'AUTHORIZED_SNAPSHOT.json')['authorized_ids']).collect()
        elif phase=='verify':g['verify']()
        elif phase=='finalize':g['finalize']()
        else:raise ValueError('Unknown bounded phase')
    except BaseException as error:
        save('FAILURE_'+str(__import__('time').time_ns())+'.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise
