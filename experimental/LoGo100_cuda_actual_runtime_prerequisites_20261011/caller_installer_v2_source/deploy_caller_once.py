"""Metadata-only deployment proposal; no service start, scientific import or evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,base64,hashlib,json,shlex,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
BASE='/workspace/guardfed_checks/uniform_cuda_LoGo100_20261011_attempt001'
ART='/workspace/guardfed_checks/uniform_cuda_LoGo100_artifacts_20261011_attempt001'
CACHE='/workspace/guardfed_checks/FedAvg100_cuda_adopted_cache_index_20261011_actual002'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def need(ok,msg):
    if not ok:raise ValueError(msg)
def blob(v):return (json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def pin(path,h):return {'path':str(path),'sha256':h}
def prepare(inputs,inputs_sha,inputs_bytes):
    need(inputs['execution_authorized'] is True and inputs['root_restoration_authorized'] is True,'Root install/restore decision required')
    need(type(inputs['authorize_fixed2_queue']) is bool,'Explicit queue authorization required')
    need(inputs['installer_source_sha256']==sha(__file__),'Installer source changed')
    for role in ['installer_review','fresh_resource']:
        p=inputs[role];need(sha(p['path'])==p['sha256'],'Missing actual '+role)
    review=read(inputs['installer_review']['path'])
    need(review.get('source_adoptable') is True and review.get('installer_source_sha256')==sha(__file__),'Independent review does not cover this installer')
    need(hashlib.sha256(inputs_bytes).hexdigest()==inputs_sha,'Original root input bytes changed')
    fresh=read(inputs['fresh_resource']['path']);cpu=fresh['cpu113']
    need(fresh['guide_sha256']=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
         and cpu['candidate_available_observation_only'] is True and cpu['siblings']==[113,369]
         and not cpu['unreadable_live_processes'] and not cpu['conflicting_restricted_tasks'],'Fresh independent CPU113 gate failed')
    need(0<=(datetime.now(timezone.utc)-datetime.fromisoformat(fresh['utc'])).total_seconds()<300,'Fresh resource snapshot expired')
    need(int(fresh['memory']['max'])-fresh['memory']['current']>8*1024**3 and fresh['workspace_disk_free']>10*1024**3,'Fresh RAM/disk gate failed')
    files={};code={};caller=ROOT/'tmp/LoGo100_cuda_cache_runtime_prepare_20261011';adapter=ROOT/'tmp/LoGo100_uniform_cuda_cache_adapter_prepare_20261011'
    def add(name,b):need(name not in files,'Duplicate deploy member');files[name]=base64.b64encode(b).decode()
    for directory,prefix,expected in [(caller,'','78558bd5d32f6830bd4547cece83eaf55afdef8607d66c9d0aa120a72f5af7f0'),(adapter,'adapter/','64e00351224554df99e82478e43b5c4353c1dcf1152564a371ca858a5195b35b')]:
        seal=directory/'FILES_SHA256.json';need(sha(seal)==expected,'Original source seal changed')
        for n,row in read(seal)['files'].items():need(sha(directory/n)==row['sha256'],'Changed original code');add(prefix+n,(directory/n).read_bytes())
        add(prefix+'FILES_SHA256.json',seal.read_bytes())
    reviews=[('caller','tmp/LoGo100_cuda_cache_runtime_independent_review_20261011/SOURCE_REVIEW.json','b828824c784ff575108ed6fcbfc8edfb2320cfe2456ee517c17eb39e24731bc9'),('adapter','tmp/LoGo100_uniform_cuda_cache_independent_review_20261011/REVIEW.json','ad20203e701fffc3ecd77657de1699f053aa259b8a7ea36b728dc28f59d62ba8'),('cache','tmp/FedAvg_cuda100_saved_cache_closure_independent_review_20261011/REVIEW.json','7cf772e2753453e5cd5c4a582743b93e4e5a5a3f156bfa24bee6d1fb7f7f8a69')]
    for name,p,h in reviews:need(sha(ROOT/p)==h,'Original source review changed');add('reviews/'+name+'.json',(ROOT/p).read_bytes())
    author=ROOT/'docs/server_deployment_20260923/training_20260923/celeba_uniform_cuda_evaluation_v1/AUTHOR_DECISION_20261011.json';need(sha(author)=='a1f1be152a0aa5668b91d73d7297ea3e9275efaac8d82c3b22cfbe97b7b08fce','Author decision changed');add('AUTHOR_DECISION.json',author.read_bytes())
    inv=read(ROOT/'tmp/LoGo100_cuda_actual_prerequisites_20261011/ACTUAL_ORIGINAL_INVENTORY.json');scope=read(adapter/'EXACT100.json')
    need([r['post_id'] for r in inv['records']]==[r['post_id'] for r in scope['records']],'Changed original100 order')
    def actual(m):return pin(ART+'/'+m['relative_destination'],m['sha256'])
    cachepin=pin(CACHE+'/cache_verifier.py','0105fe99a1c3dde354e5796c67a343da70bf69ff3246ab2803efac42f449f486')
    corepin=pin('/workspace/GuardFed-celeba-expanded/scripts/reproduce_paper_tables.py','cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed')
    rows=[]
    for row in inv['records']:
        original=row['source_members'];rows.append({'post_id':row['post_id'],'artifacts':{k:actual(v) for k,v in row['artifacts'].items()},'runtime_dependencies':{'bridge':actual(original['logofair_bridge_20261010/bridge.py']),'core':corepin,'cache_verifier':cachepin,'resource_gate':pin(BASE+'/guard.py',sha(caller/'guard.py'))},'original_source_members':{k:actual(original[k]) for k in ['logofair/adapter.py','logofair/upstream/fedlearn/models/FedFairPostClient.py']}})
    restore={'status':'ORIGINAL100_RESTORED_BYTE_IDENTITIES_PENDING_ROOT_REGISTRATION','cohort_sha256':sha(adapter/'EXACT100.json'),'records':rows,'transport':pin(ART+'/TRANSPORT_VERIFICATION.json','b86b959f8737479ce3917921dfc9aa59f50890ea19051d5dfe73a407a0060712')};restorebytes=blob(restore);add('RESTORE_INDEX.json',restorebytes)
    restorepin=pin(BASE+'/RESTORE_INDEX.json',hashlib.sha256(restorebytes).hexdigest())
    deps={m['original_path']:m for m in inv['members']};external={ART+'/'+v['relative_destination']:v['sha256'] for v in deps.values()}
    handoff=read(ROOT/'tmp/FedAvg100_cuda_linux_index_actual002_20261011/HANDOFF.json')
    for row in handoff['linux_pins'].values():external[row['path']]=row['sha256']
    indexpath=ROOT/'tmp/FedAvg100_cuda_linux_index_actual002_20261011/ROOT_CACHE_INDEX.json';need(sha(indexpath)=='6a21265f08fb26da130d95c67bf4130039b25916f44a543cd695e15665a25b64','Actual cache index changed');index=read(indexpath)
    for k in ['root','audit','transport','selection','metadata']:external[index[k]['path']]=index[k]['sha256']
    for row in index['records']:
        for item in list(row['members'].values())+[row['metadata']]:external[item['path']]=item['sha256']
    external[corepin['path']]=corepin['sha256'];external['/workspace/GuardFed-celeba-expanded/src/data_loader.py']='41723cc5dd57d7d7c4878bb93989426569279d608c0357cf0b6429c8c4a2a091'
    external[restore['transport']['path']]=restore['transport']['sha256']
    env=read(ROOT/'tmp/LoGo100_cuda_actual_runtime_prerequisites_20261011/ENVIRONMENT_CLOSURE.json');need(sha(ROOT/'tmp/LoGo100_cuda_actual_runtime_prerequisites_20261011/ENVIRONMENT_CLOSURE.json')=='308a1d3cb847e36fb57e2e82b298503c4b31618cc3786dc564ffe0bc229c34d5','Actual CPU environment closure changed')
    python=pin(env['executable']['path'],env['executable']['sha256']);external[python['path']]=python['sha256'];external[env['pyvenv_cfg']['path']]=env['pyvenv_cfg']['sha256']
    for d in env['distributions']:
        for item in d['metadata_files'].values():external[item['path']]=item['sha256']
    for path,h in external.items():
        need(path.startswith('/workspace/') and not any(k in Path(path).suffix.lower() for k in ['.pt','.pth','.safetensors','.jpg','.png']), 'No model/image hashing in post-only deployment')
        if path.endswith(('.py','.py.txt')):code[path]=h
    services=inputs['protected_services'];names={'guardfed_celeba_fednga_fullcoverage_v1','guardfed_celeba_hybrid_fullcoverage','guardfed_celeba_mechanism_formal','guardfed_uniform_cuda500_resume2_valid','guardfed_uniform_cuda800_remaining_v3_valid','guardfed_uniform_cuda_FL100_remaining_valid','sglang'}
    observed={line.split()[0]:line.split()[1] for line in fresh['services']['stdout'].splitlines() if len(line.split())>1}
    need(set(services)==names and services.get('sglang')==['STOPPED'] and all(services[n]==['RUNNING','EXITED'] for n in names-{'sglang'}) and all(observed[n] in services[n] for n in names),'Six project lifecycles and strict sglang policy required')
    need(inputs['protected_services_observed']=={n:observed[n] for n in names},'Actual fresh service observation must be preserved separately')
    q=read(caller/'ROOT_QUEUE_AUTH.template.json');a=q['approval_base'];a.update(source_seal=pin(BASE+'/adapter/FILES_SHA256.json',sha(adapter/'FILES_SHA256.json')),author_decision=pin(BASE+'/AUTHOR_DECISION.json',sha(author)),repo='/workspace/GuardFed-celeba-expanded')
    q.update(status='ROOT_AUTHORIZED_LOGO100_CALLER_V1' if inputs['authorize_fixed2_queue'] else 'DEPLOYED_NOT_AUTHORIZED',execution_authorized=inputs['authorize_fixed2_queue'],python=python,repo='/workspace/GuardFed-celeba-expanded',exact_cohort=pin(BASE+'/adapter/EXACT100.json',sha(adapter/'EXACT100.json')),deadline_unix=inputs['deadline_unix'],adapter=pin(BASE+'/adapter/adapter.py',sha(adapter/'adapter.py')),cache_verifier=cachepin,resource_gate_source=pin(BASE+'/guard.py',sha(caller/'guard.py')),runtime_source_review=pin(BASE+'/reviews/caller.json',reviews[0][2]),adapter_source_review=pin(BASE+'/reviews/adapter.json',reviews[1][2]),cache_source_review=pin(BASE+'/reviews/cache.json',reviews[2][2]),restore_index=restorepin,protected_services=services,protected_services_observed=inputs['protected_services_observed'],FedAvg_CUDA_root=index['root'],FedAvg_CUDA_index=pin(CACHE+'/ROOT_CACHE_INDEX.json',sha(indexpath)))
    add('ROOT_INSTALL_INPUTS.json',inputs_bytes)
    return {'base':BASE,'files':files,'external_pins':external,'external_code_pins':code,'restore_index':restorepin,'transport':restore['transport'],'queue':q,'root_install_inputs_sha256':inputs_sha,'root_restoration_authorized':True}

REMOTE=r'''
from pathlib import Path
import base64,hashlib,json,os,sys,time
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def need(ok,msg):
 if not ok:raise ValueError(msg)
def write(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
need(sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa','Guide changed')
b=json.load(sys.stdin);base=Path(b['base']);need(str(base)=='/workspace/guardfed_checks/uniform_cuda_LoGo100_20261011_attempt001' and base.resolve()==base and not base.exists(),'Fresh caller namespace required')
need(b['root_restoration_authorized'] is True,'No root restoration decision')
fingerprint={}
for p,h in b['external_pins'].items():
 path=Path(p);need(path.resolve()==path and not path.is_symlink() and sha(path)==h,'Changed needed original/cache/environment dependency: '+p)
 st=path.stat();fingerprint[p]={'sha256':h,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'ino':st.st_ino}
need(time.time()<b['queue']['deadline_unix'],'Root deadline expired')
base.mkdir();deployed={}
for name,data in b['files'].items():
 p=Path(name);need(not p.is_absolute() and '..' not in p.parts,'Unsafe source member')
 raw=base64.b64decode(data,validate=True);dest=base/p;dest.parent.mkdir(parents=True,exist_ok=True)
 if name.endswith('.py'):compile(raw,str(dest),'exec')
 with dest.open('xb') as f:f.write(raw)
 deployed[name]={'sha256':sha(dest),'bytes':len(raw)}
root={'status':'ROOT_LOGO100_ORIGINAL_ARTIFACT_RESTORE_ADOPTED','index':b['restore_index'],'all_members_sha256_verified':True,'transport':b['transport'],'root_install_inputs_sha256':b['root_install_inputs_sha256'],'science_executed':False}
write(base/'RESTORE_ROOT.json',root);q=b['queue'];q['restore_root']={'path':str(base/'RESTORE_ROOT.json'),'sha256':sha(base/'RESTORE_ROOT.json')}
write(base/'ROOT_QUEUE_AUTH.json',q);write(base/'SOURCE_DATA_FINGERPRINT.json',{'files':fingerprint})
for name in ['RESTORE_ROOT.json','ROOT_QUEUE_AUTH.json','SOURCE_DATA_FINGERPRINT.json']:deployed[name]={'sha256':sha(base/name),'bytes':(base/name).stat().st_size}
write(base/'DEPLOYMENT_SEAL.json',{'files':deployed,'external_code_pins':b['external_code_pins'],'source_only_deploy':True,'no_science_start':True})
print(json.dumps({'status':'LOGO_CALLER_SOURCE_AND_NEEDED_DEPENDENCIES_DEPLOYED_NO_START','base':str(base),'source_members':len(deployed),'hashed_needed_dependencies':len(fingerprint),'restore_root_sha256':sha(base/'RESTORE_ROOT.json'),'queue_auth_sha256':sha(base/'ROOT_QUEUE_AUTH.json'),'deployment_seal_sha256':sha(base/'DEPLOYMENT_SEAL.json'),'science_started':False}))
'''
def main():
    p=argparse.ArgumentParser();p.add_argument('--root-inputs',type=Path,required=True);p.add_argument('--root-inputs-sha256',required=True);p.add_argument('--execute',action='store_true');args=p.parse_args()
    need(sha(args.root_inputs)==args.root_inputs_sha256,'Actual root inputs SHA required');bundle=prepare(read(args.root_inputs),args.root_inputs_sha256,args.root_inputs.read_bytes())
    with (HERE/'ACTUAL_DEPLOY_BUNDLE.json').open('xb') as f:f.write(blob(bundle))
    if not args.execute:print('Prepared only; no SSH or deployment');return
    command='env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 10 ionice -c 3 python3 -B -c '+shlex.quote(REMOTE)
    argv=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=15','root@89.22.197.55',command]
    rc=None;error=None
    try:
        with (HERE/'DEPLOY.stdout').open('xb') as out,(HERE/'DEPLOY.stderr').open('xb') as err:r=subprocess.run(argv,input=blob(bundle),stdout=out,stderr=err,timeout=600)
        rc=r.returncode
    except BaseException as exc:error=repr(exc);raise
    finally:
        with (HERE/'DEPLOY_COMMAND.json').open('x',encoding='utf8') as f:json.dump({'argv':argv,'source_sha256':sha(__file__),'root_inputs_sha256':args.root_inputs_sha256,'returncode':rc,'exception':error,'automatic_retry':False},f,indent=2)
    need(rc==0,'Preserve failed deployment; no automatic retry')
if __name__=='__main__':main()
