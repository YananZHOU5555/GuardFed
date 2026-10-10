"""Flatten already reviewed readers into one private exact10 batch; source only."""
from pathlib import Path
import ast, datetime, difflib, hashlib, json

H=Path(__file__).resolve().parent
R=H.parents[1]
P=R/'tmp/celeba_flgmm_fullcoverage_delta_after38_20261010'
O=R/'tmp/celeba_flgmm_fullcoverage_delta_after14_20261010'
B=R/'tmp/celeba_flgmm_fullcoverage_incremental_20261009'
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,text):
    with (H/name).open('x',encoding='utf-8',newline='\n') as f:f.write(text)
def save(name,value):write(name,json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def replace(text,before,after,count=1):
    assert text.count(before)==count,(before,text.count(before),count)
    return text.replace(before,after)
def function(text,name):
    n=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(text,n)
def cpu110(text):
    for a,b in [('CPU107','CPU110'),('{107}','{110}'),('107 in a','110 in a'),
                ('107 not in aff','110 not in aff'),('107 in aff','110 in aff'),
                ('CPU=107','CPU=110'),('-c 107 ','-c 110 '),("'107'","'110'"),
                ('CPU106','CPU110'),('(106,1,10)','(110,1,10)'),('helper_CPU=106','helper_CPU=110')]:
        text=text.replace(a,b)
    return text

pinpath=R/'tmp/five_queue_review_20261011/fl_next.json'
assert sha(pinpath)=='7ca65a27cdcb9dfae4475ec9f9d96d5550541892b037eb18121e6855445397a3'
pins=read(pinpath);IDS=pins['exact_new_ids']
assert len(IDS)==len(set(IDS))==10
for rel,digest in pins['source_files'].items():assert sha(R/rel)==digest
latest=B/'LATEST_BACKUP.json'; prior=Path(pins['prior_offserver_path'])
assert sha(latest)==pins['latest_sha256'] and sha(prior)==pins['prior_offserver_sha256']
assert sha(R/pins['prior_root'])==pins['prior_root_sha256']
snapshot=R/pins['snapshot']['path'];assert sha(snapshot)==pins['snapshot']['sha256']
current=read(snapshot)
assert IDS==[r['id'] for r in current['FLGMM']['rows'] if r['observed_terminal'] and r['id'] not in read(prior)['accepted_job_ids']]
for name,path in [('PREVIOUS_LATEST.json',latest),('PREVIOUS_OFFSERVER_ACCEPTANCE.json',prior)]:
    with (H/name).open('xb') as f:f.write(path.read_bytes())
save('AUTHORIZED_SNAPSHOT.json',dict(status='ROOT_AUTHORIZED_FIXED_EXACT10_AFTER_NATIVE44_SOURCE_PREPARATION_NOT_ACCEPTANCE',
    snapshot_path=pins['snapshot']['path'],snapshot_sha256=sha(snapshot),snapshot_utc=current['utc'],
    prior_count=44,terminal_count=54,authorized_ids=IDS,prior_root_path=pins['prior_root'],
    prior_root_sha256=pins['prior_root_sha256'],prior_offserver_sha256=pins['prior_offserver_sha256'],
    no_future_terminal_ids_allowed=True,root_adoption_required=True,new_accepted=0))

# Materialize the accepted after38 effective collector, then change only exact IDs/message.
original=(O/'collect_delta.py').read_text('utf-8')
wrapper=(P/'collect_delta.py').read_text('utf-8')
edits=ast.literal_eval(next(n for n in ast.parse(wrapper).body if isinstance(n,ast.For)).iter)
old_effective=original
for before,after in edits:old_effective=replace(old_effective,before,after)
assert hashlib.sha256(old_effective.encode()).hexdigest()==read(P/'SOURCE_REUSE.json')['effective_collector_sha256']
prior6=read(P/'AUTHORIZED_SNAPSHOT.json')['authorized_ids']
collector=replace(old_effective,'    authorized_ids='+repr(prior6)+'\n','    authorized_ids='+repr(IDS)+'\n')
collector=replace(collector,'All six authorized outputs','All ten authorized outputs')
# Historical comment only; scientific operations below before=repo_identity stay exact.
collector=replace(collector,'fixed snapshot exact2 only','fixed snapshot exact10 only')
assert collector[collector.index('    before=repo_identity'):]==original[original.index('    before=repo_identity'):]
write('collect_delta.py',collector)

src16=(R/'tmp/celeba_flgmm_fullcoverage_delta_after16_20261010/run_once.py').read_text('utf-8')
assert hashlib.sha256((R/'tmp/celeba_flgmm_fullcoverage_delta_after16_20261010/run_once.py').read_bytes()).hexdigest()=='5b00a63088fea7294faa12dafae656f1ff4312e9b151a14bd7cfe1eb806136db'
src22=(R/'tmp/celeba_flgmm_fullcoverage_delta_after22_20261010/run_once.py').read_text('utf-8')
preflight=cpu110((O/'preflight.py').read_text('utf-8'))
preflight=replace(preflight,"ids=['FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed'+str(x)+'_fullcoverage' for x in (91006,91007)]",'ids='+repr(IDS))
write('preflight.py',preflight)
close=cpu110((O/'close_observation.py').read_text('utf-8')).replace('celeba_flgmm_fullcoverage_delta_after14_20261010/collect_delta.py',H.name+'/collect_delta.py')
write('close_observation.py',close)
collect=cpu110(function(src16,'collect'))
collect=replace(collect,"preflight=(O/'preflight.py').read_text('utf8');assert sha(O/'preflight.py')=='7fc3d7a78ad71a3c9b48522020ad5d773aa9303791fc6ff9fa080a25648340d2'", "preflight=checked_source(H/'preflight.py',read(H/'PREPARED_FILES_SHA256.json')['files']['preflight.py']['sha256'])")
collect=replace(collect," assert preflight.count('(91006,91007)')==1;preflight=preflight.replace('(91006,91007)','(91008,91009)')\n",'')
collect=replace(collect,"receipt['accepted_total']==18 and receipt['accepted_new']==2","receipt['accepted_total']==54 and receipt['accepted_new']==10")
verify=function(src22,'verify')
verify=replace(verify,"ids = read(H/'AUTHORIZED_SNAPSHOT.json')['authorized_ids']; m = parent(ids)","ids = read(H/'AUTHORIZED_SNAPSHOT.json')['authorized_ids']; m = types.SimpleNamespace(run=run,REMOTE=REMOTE,SSH=SSH)")
verify=replace(verify,'{22+len(ids)}','{44+len(ids)}')
verify=replace(verify,"closed = cpu106((O/'close_observation.py').read_text('utf8')).replace('celeba_flgmm_fullcoverage_delta_after14_20261010/collect_delta.py', H.name+'/collect_delta.py')", "closed = checked_source(H/'close_observation.py',read(H/'PREPARED_FILES_SHA256.json')['files']['close_observation.py']['sha256'])")
finalize=cpu110(function(src22,'finalize'))
for before,after,count in [
    ('total=22+n','total=44+n',1),
    ("['INITIAL_GUIDE','INITIAL_OWNER','SNAPSHOT','GUIDE','OWNER','PREFLIGHT','COLLECT','SERVER_SHA','SCP','VERIFY','CLOSED']","['GUIDE','OWNER','PREFLIGHT','COLLECT','SERVER_SHA','SCP','VERIFY','CLOSED']",1),
    ("prior['accepted_total']==22","prior['accepted_total']==44",1),("p['accepted_job_ids'][:22]","p['accepted_job_ids'][:44]",1),
    ('prior_accepted_new=22','prior_accepted_new=44',1),('old22_ordered_prefix_exact','old44_ordered_prefix_exact',2),
    ('accepted_before=22','accepted_before=44',1),('# Actual FL96 after22:','# Actual FL96 after44:',1),('ordered prior22','ordered prior44',1),
    ('Original after14 collector is SHA-bound through a tiny entry changing only its authorized-ID literal;', 'Original after14 collector is directly materialized with the accepted exact-completeness guards and frozen ten-ID literal;',1)]:
    finalize=replace(finalize,before,after,count)
header='''"""Direct reuse of original collector/transport/verifier; exact10 after44 only."""
from pathlib import Path
import argparse,ast,base64,datetime,hashlib,json,subprocess,sys,traceback,types
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[1]
B=R/'tmp/celeba_flgmm_fullcoverage_incremental_20261009'
O=R/'tmp/celeba_flgmm_fullcoverage_delta_after14_20261010'
S=Path('F:/YananResearchStorage/GuardFed')/H.name/'batch'
REMOTE='/workspace/guardfed_checks/'+H.name
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55']
PREFIX='env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 taskset -c 110 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B -'
PACKAGE='6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
'''
header+='PREVIOUS='+repr(pins['prior_offserver_sha256'])+'\nROOTPROOF='+repr(pins['prior_root_sha256'])+'\nIDS='+repr(IDS)+'\n'
functions=[function(src22,n) for n in ('save','replace_exact','checked_source','source_check','storage')]
functions.extend([function(src16,'run'),collect,verify,finalize])
tail='''
def pinned_scope(review):
    assert review['source_adoptable'] is True and review['exact_selected_ids']==IDS
    assert review['prepared_seal_sha256']==sha(H/'PREPARED_FILES_SHA256.json')
    for n,pin in read(H/'PREPARED_FILES_SHA256.json')['files'].items():
        assert sha(H/n)==pin['sha256'] and (H/n).stat().st_size==pin['bytes']
    a=read(H/'AUTHORIZED_SNAPSHOT.json')
    assert a['authorized_ids']==IDS and a['prior_count']==44 and a['terminal_count']==54
    assert sha(R/a['snapshot_path'])==a['snapshot_sha256']
    assert sha(B/'LATEST_BACKUP.json')==sha(H/'PREVIOUS_LATEST.json')
    assert sha(R/a['prior_root_path'])==a['prior_root_sha256']==ROOTPROOF
    assert sha(H/'PREVIOUS_OFFSERVER_ACCEPTANCE.json')==a['prior_offserver_sha256']==PREVIOUS
    assert sha(R/'tmp/guardfed_local_storage.py')=='2482f2f46243abafd33ea2fadc94d555c0d80327fefe19caea3eb8f3f6bb3d63'
    source_check()

if __name__=='__main__':
    try:
        p=argparse.ArgumentParser();p.add_argument('phase',choices=['collect','verify','finalize'])
        p.add_argument('--review',type=Path,required=True);p.add_argument('--review-sha256',required=True);a=p.parse_args()
        assert __debug__ and sha(a.review)==a.review_sha256
        pinned_scope(read(a.review))
        if a.phase=='collect':
            save('F_VOLUME_BEFORE_COLLECT.json',storage(0));collect()
        elif a.phase=='verify':verify()
        else:finalize()
    except BaseException as error:
        if not isinstance(error,SystemExit):save('FAILURE_'+str(__import__('time').time_ns())+'.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise
'''
run=header+'\n\n'.join(functions)+'\n'+tail
compile(run,str(H/'run_once.py'),'exec');write('run_once.py',run)
reuse=read(P/'SOURCE_REUSE.json')
reuse.update(thin_entry_sha256=sha(H/'collect_delta.py'),effective_collector_sha256=sha(H/'collect_delta.py'),
    sole_effective_change='Exact10 after accepted44; direct materialized source, no wrapper execution chain',
    predecessor_root_sha256=pins['prior_root_sha256'],predecessor_offserver_sha256=pins['prior_offserver_sha256'],
    transport_flattened=True,source_snapshot_sha256=sha(snapshot),no_new_scientific_acceptances=True)
save('SOURCE_REUSE.json',reuse)
diff=''.join(difflib.unified_diff(old_effective.splitlines(True),collector.splitlines(True),fromfile='accepted_after38_effective_collector',tofile='direct_exact10_collector'))
for old,new,name in [(function(src16,'collect'),collect,'collect'),(function(src22,'verify'),verify,'verify'),(function(src22,'finalize'),finalize,'finalize')]:
    diff+=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='original_'+name,tofile='direct_'+name))
write('SOURCE_DIFF.patch',diff)
save('SOURCE_PARENT_PINS.json',dict(files={p.relative_to(R).as_posix():sha(p) for p in [
    O/'collect_delta.py',O/'preflight.py',O/'close_observation.py',O/'check_saved_tensors.py',
    P/'collect_delta.py',P/'SOURCE_REUSE.json',P/'run_once.py',
    R/'tmp/celeba_flgmm_fullcoverage_delta_after16_20261010/run_once.py',
    R/'tmp/celeba_flgmm_fullcoverage_delta_after22_20261010/run_once.py',B/'verify_delta_offserver.py',
    B/'FILES_SHA256.json',R/'tmp/guardfed_local_storage.py',pinpath]},
    scientific_per_ID_and_archive_tail_bytes_exact=True,old44_metadata_bytes_copied_exact=True,
    direct_flattening_no_runtime_parent_wrapper_chain=True,actual_collect_verify_finalize_executed=False))
print(json.dumps({'prepared_path':H.relative_to(R).as_posix(),'exact10':IDS,'collect_sha256':sha(H/'collect_delta.py'),'run_once_sha256':sha(H/'run_once.py')}))
