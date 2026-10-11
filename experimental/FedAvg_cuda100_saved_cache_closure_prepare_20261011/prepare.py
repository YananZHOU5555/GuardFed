from pathlib import Path
import ast,json,hashlib,difflib
R=Path.cwd();H=R/'tmp/FedAvg_cuda100_saved_cache_closure_prepare_20261011';P=R/'tmp/old800_cuda3_saved_audit_prepare_20261011/mixed_producer_v2';O=P.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
put=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
v=json.loads((R/'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json').read_bytes());rows=[r for r in v['records'] if r['method']=='FedAvg'];IDS=[r['id'] for r in rows];assert len(IDS)==len(set(IDS))==100
MAP={i:('attempt002' if i=='FedAvg_IID_Benign_seed91001' else 'attempt003') for i in IDS}
can=lambda v:hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
put(H/'EXACT100_REFS.json',dict(status='FROZEN_INPUT_REFS_NOT_CUDA_RESULTS',inventory={'path':'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json','sha256':sha(R/'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json')},ordered_ids=IDS,records=[dict(id=r['id'],inventory_index=v['records'].index(r),canonical_sha256=can(r)) for r in rows],record_producers=MAP))
def derive(name,fn):
 raw=(P/name).read_bytes();s=raw.decode();spans=[]
 def edit(a,b):
  nonlocal s
  assert a!=b and s.count(a)==1,(name,a[:100],s.count(a));s=s.replace(a,b,1);spans.append({'before':a,'after':b})
 fn(edit,s)
 restored=s
 for z in reversed(spans):
  assert restored.count(z['after'])==1,z['after'][:90]
  restored=restored.replace(z['after'],z['before'],1)
 assert restored.encode()==raw
 compile(s,name,'exec');(H/name).write_bytes(s.encode());put(H/(name+'.inverse.json'),dict(parent={'path':(P/name).relative_to(R).as_posix(),'sha256':sha(P/name)},spans=spans,whole_raw_inverse_exact=True,derived_sha256=sha(H/name)))
 (H/(name+'.patch')).write_text(''.join(difflib.unified_diff(raw.decode().splitlines(True),s.splitlines(True),fromfile='mixed_v2/'+name,tofile='FedAvg100/'+name)),encoding='utf8')
def audit(edit,s):
 edit('HERE=Path(__file__).resolve().parent;ORIGINALS=HERE.parent;ROOT=HERE.parents[2]',"HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];ORIGINALS=ROOT/'tmp/old800_cuda3_saved_audit_prepare_20261011'")
 for prefix,val in [('IDS=',IDS),('RECORD_PRODUCERS=',MAP)]:
  line=next(l for l in s.splitlines() if l.startswith(prefix));edit(line,prefix+repr(val))
 edit("selection['scope']=='ORIGINAL800_FIXED3_MIXED_PRODUCER_SELECTION_V2'","selection['scope']=='ORIGINAL800_FEDAVG100_MIXED_PRODUCER_SELECTION_V1'")
 edit("{r['method'] for r in selection['records']}=={'FedAvg','FedAA','LASA'}","{r['method'] for r in selection['records']}=={'FedAvg'}")
 edit("set(qauth['canary_ids'])","set(qauth['ordered_ids'])")
 edit("failure_record['id']==IDS[1]","failure_record['id']=='FedAA-DDPG_policy0.001_keep16_local0.001_IID_Benign_seed91001'")
 edit("metric_values_checked=27,integer_base_counts_checked=72,prediction_rules_checked=9","metric_values_checked=900,integer_base_counts_checked=2400,prediction_rules_checked=300")
 edit("records=3,sha256=", "records=100,sha256=")
 edit("status='OLD800_FIXED3_SAVED_OUTPUTS_AUDIT_NO_REFIT_PASS_NOT_ROOT_ADOPTED'","status='FEDAVG100_SAVED_OUTPUTS_AUDIT_NO_REFIT_PASS_NOT_ROOT_ADOPTED'")
 edit("scope='Fixed3 only; original Linux whole plus SHA transport and saved-output consistency. No fit/model/CNN/test/cohort completion.'","scope='FedAvg100 only; original Linux whole plus SHA transport and saved-output consistency. No fit/model/CNN/test/full800 completion.'")
 # Added metadata gate lies outside the original per-record scientific loop.
 edit("        metadata=member('source/metadata.npz');need(sha(metadata)==METADATA_SHA,'Original metadata SHA differs')","        for record in selection['records']:\n            need(sha(member('original_results/'+record['id']+'.json'))==record['result']['sha256'],'Original result member changed')\n        metadata=member('source/metadata.npz');need(sha(metadata)==METADATA_SHA,'Original metadata SHA differs')")
def transport(edit,s):
 edit("H=R/'tmp/old800_cuda3_saved_audit_prepare_20261011/mixed_producer_v2'","H=R/'tmp/FedAvg_cuda100_saved_cache_closure_prepare_20261011'")
 edit("check_bulk_storage(50*1024**2);F=Path('F:/YananResearchStorage/GuardFed/uniform_cuda_old800_20261011/canary3_mixed_attempt003')","check_bulk_storage(250*1024**2);F=Path('F:/YananResearchStorage/GuardFed/uniform_cuda_old800_20261011/FedAvg100_mixed_attempt003')")
 line=next(l for l in s.splitlines() if l.startswith("ids=inp['exact_ids'];assert ids=="));edit(line,"ids=inp['exact_ids'];assert ids=="+repr(IDS))
 edit("set(auth['canary_ids'])","set(auth['ordered_ids'])")
 edit("   assert not (str(B/'worker.py') in argv or (str(B/'run_queue.py') in argv and 'canary' in argv)),('Selected producer still live',producer,proc.name)","   if str(B/'worker.py') in argv:\n    assert '--approval' in argv and argv.index('--approval')+1<len(argv),('Unidentified producer worker',proc.name)\n    approval_path=Path(argv[argv.index('--approval')+1]);worker_approval=read(approval_path)\n    assert worker_approval['id'] not in ids,('Selected FedAvg producer still live',producer,proc.name)")
 edit("read(failure)['id']==ids[1]","read(failure)['id']=='FedAA-DDPG_policy0.001_keep16_local0.001_IID_Benign_seed91001'")
 edit("export=B/'exports/old800_fixed3_mixed_saved_audit_attempt001'","export=B/'exports/FedAvg100_mixed_saved_audit_attempt001'")
 edit("scope='ORIGINAL800_FIXED3_MIXED_PRODUCER_SELECTION_V2'","scope='ORIGINAL800_FEDAVG100_MIXED_PRODUCER_SELECTION_V1'")
 edit("selected=[next(r for r in cohort['records'] if r['id']==i) for i in ids]","selected=[r for r in cohort['records'] if r['method']=='FedAvg'];assert [r['id'] for r in selected]==ids and len(selected)==100")
 edit(" assert not (out.parent/(identity+'.formal_failure.json')).exists()"," assert not (out.parent/(identity+'.formal_failure.json')).exists()\n record=next(r for r in selected if r['id']==identity)\n originals=[Path(path) for path,meta in binding['artifact_before'].items() if meta['sha256']==record['result']['sha256']]\n assert len(originals)==1 and sha(originals[0])==record['result']['sha256'] and originals[0].stat().st_size==record['result']['bytes']\n files['original_results/'+identity+'.json']=originals[0]")
 # Result metadata exported, never checkpoint/full inventory.
 # Longer bounded archive/hash operation, no retry added.
 edit('capture_output=True,timeout=90)','capture_output=True,timeout=300)')
derive('audit_saved.py',audit);derive('transport.py',transport)
pins=json.loads((P/'SOURCE_PINS.json').read_bytes());put(H/'SOURCE_PINS.json',pins)
t=json.loads((P/'ROOT_INPUTS.template.json').read_bytes());t.update(exact_ids=IDS,record_producers=MAP,accepted_sha256={i:None for i in IDS});put(H/'ROOT_INPUTS.template.json',t)
# Original read_prefix and array_sha are copied raw, not executed.
s=(O/'original_replay.py.txt').read_text();fs=[ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in {'read_prefix','array_sha'}];assert len(fs)==2
(H/'original_prefix_functions.py.txt').write_text('\n\n'.join(fs)+'\n',encoding='utf8')
print(json.dumps({'derived':2,'ids':len(IDS),'original_prefix_sha256':sha(H/'original_prefix_functions.py.txt')}))
