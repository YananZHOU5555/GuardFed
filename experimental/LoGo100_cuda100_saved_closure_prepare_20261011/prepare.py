"""Local source/metadata preparation only; never opens F or invokes remote/science."""
from pathlib import Path
import hashlib,json,difflib
H=Path(__file__).resolve().parent;R=H.parents[1];P=R/'tmp/LoGo_cuda_fixed2_saved_audit_prepare_v2_20261011'
assert not (H/'FILES_SHA256.json').exists(),'Sealed preparation is immutable'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def put(n,v):
 target=json.dumps(v,ensure_ascii=False,indent=2)+'\n'
 with (H/n).open('w',encoding='utf8',newline='\n') as f:f.write(target)
assert sha(P/'FILES_SHA256.json')=='eb08186667ef3ca700cd853aa4dfc919181c86f080b89e5b2b1484203051e3a3'
for n,pin in read(P/'FILES_SHA256.json')['files'].items():assert sha(P/n)==pin['sha256']
cohort=R/'tmp/LoGo100_uniform_cuda_cache_adapter_prepare_20261011/EXACT100.json'
assert sha(cohort)=='52d6a15e077eb50b1d81d2aa0e481c6e2e5fb5dfeea24e84ba6036535dffed9b'
rows=read(cohort)['records'];ids=[r['post_id'] for r in rows]
canaries=read(P/'ROOT_INPUTS.template.json')['exact_ids'];remaining=[i for i in ids if i not in canaries]
assert len(ids)==len(set(ids))==100 and len(remaining)==98
assert all(next(r for r in rows if r['post_id']==v['post_id'])==v for v in read(P/'EXACT2_REFS.json')['records'])
if (H/'EXACT100_REFS.json').exists():assert (H/'EXACT100_REFS.json').read_bytes()==cohort.read_bytes()
else:
 with (H/'EXACT100_REFS.json').open('xb') as f:f.write(cohort.read_bytes())
pins={}
for k,n,expected in [('root','root_adoption/ROOT_SCIENTIFIC_ADOPTION.json','cdf9a578fd26a9a85ae674aff7e55932e9cd6e75b895b8e3b39adc1ad12753bc'),('transport','LOGO_FIXED2_TRANSPORT_VERIFICATION.json','ebb9b246b205057ca7f756fae727093e375fa1b33a702d7c7379adf6b783fb6d'),('audit','actual/SAVED_AUDIT.json','41364281d5735fe0b10bb50ab4b9309766a82420c0f8c53aeda63f26a7d134e6')]:
 p=P/n;assert sha(p)==expected;pins[k]={'path':str(p),'sha256':expected}
pins['ids']=canaries;pins['members']=read(P/'LOGO_FIXED2_TRANSPORT_VERIFICATION.json')['members']
put('PRIOR2_PINS.json',pins)
for n in ['original_metric_functions.py.txt','original_saved_decision_block.py.txt','original_prefix_functions.py.txt','SOURCE_REUSE.json']:
 if (H/n).exists():assert (H/n).read_bytes()==(P/n).read_bytes()
 else:
  with (H/n).open('xb') as f:f.write((P/n).read_bytes())
def derive(name,changes):
 original=(P/name).read_bytes();s=original.decode('utf8');spans=[];nl='\r\n' if b'\r\n' in original else '\n'
 for before,after in changes:
  before=before.replace('\r\n','\n').replace('\n',nl);after=after.replace('\r\n','\n').replace('\n',nl)
  count=s.count(before);assert count>0,(name,before[:160]);s=s.replace(before,after);spans.append({'before':before,'after':after,'count':count})
 b=s.encode();restored=s
 for x in reversed(spans):assert restored.count(x['after'])==x['count'];restored=restored.replace(x['after'],x['before'])
 assert restored.encode()==original
 with (H/name).open('wb') as f:f.write(b)
 put(name+'.INVERSE.json',dict(parent=str(P/name),parent_sha256=sha(P/name),derived_sha256=hashlib.sha256(b).hexdigest(),spans=spans,complete_raw_inverse=True))
 diff=''.join(difflib.unified_diff(original.decode().splitlines(True),s.splitlines(True),fromfile='v2/'+name,tofile='all100/'+name))
 with (H/(name+'.diff')).open('w',encoding='utf8',newline='') as f:f.write(diff)
oldids="IDS=['LoGoFair-DP_07_FedAvg_IID_Benign_seed91001','LoGoFair-DP_07_FedAvg_IID_F Flip_seed91001']"
newids='IDS='+repr(ids)+'\nCANARY_IDS='+repr(canaries)+'\nREMAINING_IDS=[i for i in IDS if i not in CANARY_IDS]\nPRIOR_MEMBERS='+repr(pins['members'])
derive('metadata_gate.py',[
 (oldids,newids),
 ("auth['canary_ids']==IDS","auth['canary_ids']==CANARY_IDS"),
 ("job['seed']==91001","job['seed']==int(row['post_id'].rsplit('_seed',1)[1])"),
 ("Only original fixed2","Only frozen original100"),
 ("def input_gate(x):\n", "def input_gate(x):\n    need(len(IDS)==len(set(IDS))==100 and len(REMAINING_IDS)==98,'Bad frozen100 scope')\n    need(x['prior2_root_sha256']=='cdf9a578fd26a9a85ae674aff7e55932e9cd6e75b895b8e3b39adc1ad12753bc','Unbound original adopted2')\n    need(isinstance(x['remaining_service'],str) and x['remaining_service'].startswith('guardfed_') and all(c.isalnum() or c=='_' for c in x['remaining_service']),'Unbound remaining service')\n    for p in [x['canary_complete'],x['remaining_complete'],x['root_canary_gate']]:\n        need(set(p)=={'path','sha256'} and isinstance(p['path'],str) and isinstance(p['sha256'],str) and len(p['sha256'])==64,'Unbound completion proof')\n    for role,rel in [('canary_complete','canary/COMPLETE.json'),('remaining_complete','remaining/COMPLETE.json'),('root_canary_gate','ROOT_CANARY_ACCEPTANCE.json')]:need(x[role]['path']==x['server_base']+'/'+rel,'Mixed completion namespace')\n")])
register_old="        g.need(name not in files or files[name]==p,'Archive member collision');files[name]=p;paths[str(p)]=name;return p"
register_new="""        g.need(name not in files or files[name]==p,'Archive member collision')
        if name in g.PRIOR_MEMBERS:
            g.need(g.PRIOR_MEMBERS[name]=={'sha256':pin['sha256'],'bytes':p.stat().st_size},'Changed adopted2 live member')
            reused[name]=g.PRIOR_MEMBERS[name];protected[str(p)]=pin['sha256']
        else:files[name]=p
        paths[str(p)]=name;return p"""
lifecycle_old="""    g.need(not (B/'canary/QUEUE_FAILURE.json').exists(),'Failed canary queue')
    complete=g.read(B/'canary/COMPLETE.json');g.need(complete['ids']==g.IDS and complete['server_saved_accepted'] is True,'Incomplete fixed2 Linux canary')
    register(g.pin(B/'canary/COMPLETE.json'),'base/CANARY_COMPLETE.json')
    service=subprocess.run(['supervisorctl','status',x['service']],capture_output=True,text=True,timeout=30)
    g.need(service.returncode in (0,3) and len(service.stdout.split())>=2 and service.stdout.split()[0]==x['service'] and service.stdout.split()[1]=='EXITED','Canary service not normally exited')"""
lifecycle_new="""    service_status={}
    for phase,ids,pin,service_name in [('canary',g.CANARY_IDS,x['canary_complete'],x['service']),('remaining',g.REMAINING_IDS,x['remaining_complete'],x['remaining_service'])]:
        g.need(not (B/phase/'QUEUE_FAILURE.json').exists(),'Failed queue')
        complete=g.read(register(pin,'base/'+phase.upper()+'_COMPLETE.json'))
        g.need(complete['ids']==ids and complete['cohort']==auth['exact_cohort'] and complete['server_saved_accepted'] is True,'Incomplete/disordered Linux phase')
        progress=g.read(B/phase/'progress.json');g.need(progress['accepted_ids']==ids and progress['planned']==len(ids) and progress['failed']==0,'Incomplete progress')
        service=subprocess.run(['supervisorctl','status',service_name],capture_output=True,text=True,timeout=30)
        g.need(service.returncode in (0,3) and len(service.stdout.split())>=2 and service.stdout.split()[0]==service_name and service.stdout.split()[1]=='EXITED','Service not normally exited')
        service_status[phase]=service.stdout.strip()
    gate=g.read(register(x['root_canary_gate'],'base/ROOT_CANARY_ACCEPTANCE.json'))
    g.need(gate['root_accepted'] is True and gate['ids']==g.CANARY_IDS and gate['cohort']==auth['exact_cohort'] and gate['FedAvg_CUDA_root']==x['FedAvg_CUDA_root'],'Wrong prior2 root gate')
    for i in g.CANARY_IDS:g.need(gate['records'][i]['task_acceptance']==x['task_accepted'][i],'Changed adopted2 task')"""
derive('remote_export.py',[
 ('files={};paths={};protected={}','files={};paths={};protected={};reused={}'),
 (register_old,register_new),
 ("auth=g.read(files['base/ROOT_QUEUE_AUTH.json'])","auth=g.read(x['root_queue_auth']['path'])"),
 ("dep=g.read(files['base/DEPLOYMENT_SEAL.json'])","dep=g.read(x['deployment_seal']['path'])"),
 (lifecycle_old,lifecycle_new),
 ('LoGo_fixed2_saved_attempt001','LoGo100_delta98_saved_attempt001'),
 ("for i in g.IDS}\n    g.need(all", "for i in g.REMAINING_IDS}\n    g.need(set(reused)==set(g.PRIOR_MEMBERS) and reused==g.PRIOR_MEMBERS,'Incomplete retained2 identities')\n    g.need({n for n in files if n.endswith('.npz')}==allowed_npz,'Only exact new98 scores may be exported')\n    g.need(all"),
 ("'server_paths':paths", "'server_paths':{p:n for p,n in paths.items() if n in files},'reused_server_paths':{p:n for p,n in paths.items() if n in reused},'reused_members':reused"),
 ('LOGO_FIXED2.tar.gz','LOGO100_DELTA98.tar.gz'),
 ("service=service.stdout.strip(),no_pickle_download=True", "service=service_status,no_pickle_download=True,new_arrays=98,reused_arrays=2,reused_member_pins=reused,reused_server_paths={p:n for p,n in paths.items() if n in reused}")])
derive('transport.py',[
 ('fixed2_saved_attempt001','all100_delta98_saved_attempt001'),
 ('EXACT2_REFS.json','EXACT100_REFS.json'),
 ('LOGO_FIXED2','LOGO100_DELTA98')])
derive('audit_saved.py',[
 ('EXACT2_REFS.json','EXACT100_REFS.json'),
 ("    archive_folder=folder\n", "    from prior2_overlay import join_prior\n    paths,member_roots,members=join_prior(x,folder,index,t)\n    archive_folder=folder\n"),
 ("p=archive_folder/paths[pin['path']]", "p=member_roots[paths[pin['path']]]/paths[pin['path']]"),
 ("    def load(pin):return g.read(localpin(pin))\n", "    def load(pin):return g.read(localpin(pin))\n    for pin,ids in [(x['canary_complete'],g.CANARY_IDS),(x['remaining_complete'],g.REMAINING_IDS)]:\n        c=load(pin);g.need(c['ids']==ids and c['server_saved_accepted'] is True and c['cohort']==load(x['root_queue_auth'])['exact_cohort'],'Different transported COMPLETE')\n    gate=load(x['root_canary_gate']);g.need(gate['root_accepted'] is True and gate['ids']==g.CANARY_IDS and gate['cohort']==load(x['root_queue_auth'])['exact_cohort'] and gate['FedAvg_CUDA_root']==x['FedAvg_CUDA_root'],'Different transported prior2 root gate')\n    for i in g.CANARY_IDS:g.need(gate['records'][i]['task_acceptance']==x['task_accepted'][i],'Transported prior2 task differs')\n"),
 ("g.sha(Path(t['verified_extract'])/rel)","g.sha(member_roots[rel]/rel)"),
 ("status='LOGO_FIXED2_SAVED_PREDICTIONS_NO_REFIT_PASS_NOT_ROOT_ADOPTED'","status='LOGO100_ALL100_CUDA_SAVED_PREDICTIONS_NO_REFIT_PASS_NOT_ROOT_ADOPTED'"),
 ('prediction_items_checked=39734,metric_values_checked=6,integer_confusion_counts_recomputed=16','prediction_items_checked=1986700,metric_values_checked=300,integer_confusion_counts_recomputed=800'),
 ('records=2,metrics=6,counts=16','records=100,metrics=300,counts=800')])
derive('check_metadata_chain.py',[
 ("x.update(server_base=B,service='guardfed_SYNTHETIC_ONLY',", "x.update(remaining_service='guardfed_SYNTHETIC_REMAINING',canary_complete=p(B+'/canary/COMPLETE.json'),remaining_complete=p(B+'/remaining/COMPLETE.json'),root_canary_gate=p(B+'/ROOT_CANARY_ACCEPTANCE.json'),server_base=B,service='guardfed_SYNTHETIC_ONLY',"),
 ("row=g.read(H/'EXACT2_REFS.json')['records'][0]", "row=next(r for r in g.read(H/'EXACT100_REFS.json')['records'] if r['post_id'].endswith('_seed91010'))"),
 ('canary_ids=g.IDS','canary_ids=g.CANARY_IDS'),
 ('seed=91001','seed=91010'),
 ("'c1dfa5b5881407c16b0c7c37cfa17cb27c62f9d9fe125a51bdca0c77a0865e81'", "'0527f26e9c22da94f35d647d96ab06b227568c9febd9850c6ac5f251a408c16d'"),
 ("reject=0\n", "reject=0\nwrong_seed=copy.deepcopy(docs);wrong_seed[original['job']['path']]['seed']=91001\ntry:g.verify_record(x,row,lambda pin:wrong_seed[pin['path']])\nexcept ValueError:reject+=1\nelse:raise AssertionError('Mismatched per-row seed accepted')\n")])
x=read(P/'ROOT_INPUTS.template.json');x.update(status='UNBOUND_NOT_AUTHORIZATION_ALL100_DELTA98',exact_ids=ids,task_accepted={i:{'path':None,'sha256':None} for i in ids},remaining_service=None,canary_complete={'path':None,'sha256':None},remaining_complete={'path':None,'sha256':None},root_canary_gate={'path':None,'sha256':None},prior2_root_sha256=pins['root']['sha256'])
put('ROOT_INPUTS.template.json',x)
put('SCOPE.json',dict(status='SOURCE_ONLY_FUTURE100_NOT_COMPLETED',ids=ids,canary_ids=canaries,remaining_ids=remaining,source_cohort={'path':str(cohort),'sha256':sha(cohort)},prior2_root=pins['root'],records=100,new_scores=98,reused_scores=2,predictions=1986700,metrics=300,primitive_confusion_counts=800,actual_transport=None,actual_audit=None,actual_root=None))
print('Prepared exact100 source from sealed fixed2; no remote, F or scientific execution.')
