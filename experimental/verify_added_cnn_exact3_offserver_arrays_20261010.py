"""Original saved-array/fit block on F; Windows audit difference stays excluded."""
import ast,datetime,hashlib,importlib.util,json,sys,types
from pathlib import Path
R=Path(__file__).resolve().parents[1];S=R/'tmp/celeba_added_cnn_exact3_scientific_acceptance_20261010';O=R/'tmp/celeba_added_cnn_exact3_root_execution_20261010'
BASE=Path('F:/YananResearchStorage/GuardFed/added_cnn_exact3_valid_20261010/attempt001/verified_extract')
sys.path.insert(0,str(S))
spec=importlib.util.spec_from_file_location('sealed_offserver_array_setup',S/'verify_offserver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
text=(S/'verify_offserver.py').read_text(encoding='utf-8');assert hashlib.sha256(text.encode()).hexdigest()=='f8760d536837f68529d10dc6cf7985f2cddc1a4a75faa4ef8e4aaa0f944426e2'
node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='main');prefix=[]
for statement in node.body:
    if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='results' for t in statement.targets):break
    prefix.append(statement)
sys.argv=[str(S/'verify_offserver.py'),'--bundle',str(BASE/'bundle'),'--metadata-npz',str(BASE/'metadata.npz'),'--output',str(O/'ARRAY_SETUP_UNUSED_OUTPUT.json'),'--allow-original-cached-root-refit']
ns=dict(m.__dict__);exec(compile(ast.Module(body=prefix,type_ignores=[]),'<sealed offserver setup only>','exec'),ns)
source=R/'tmp/celeba_valid_gpu_remaining440_v2_evidence_20261009/chunk_039/verified_extract/sourcefreeze/062_saved_science.py'
source_text=source.read_text(encoding='utf-8');assert hashlib.sha256(source.read_bytes()).hexdigest()=='d512e5b2b6614b762d921dd94b2b5162687c0bbddde4caaf8b584b3b22dba745'
fn=next(n for n in ast.parse(source_text).body if isinstance(n,ast.FunctionDef) and n.name=='check_saved')
block=next(n for n in fn.body if isinstance(n,ast.With));assert ast.get_source_segment(source_text,block).startswith("with v2.np.load(path / 'validation_predictions.npz'")
diagnostic=ns['b'].read(O/'OFFSERVER_ROOT_DIAGNOSTIC.json');assert len(diagnostic['records'])==3
assert diagnostic['records'][0]['root_receipt_differences'][0]['path']=='/server_sampling_audit/group_kl'
assert len(diagnostic['records'][0]['root_receipt_differences'])==1 and all(not r['root_receipt_differences'] for r in diagnostic['records'][1:])
rows=[]
for row,diagnosis in zip(ns['rows'],diagnostic['records']):
    b,ev,core=ns['b'],ns['ev'],ns['core'];record=b.runtime_record(row);path=BASE/'bundle'/row['id'];receipt=b.read(path/'receipt.json')
    assert diagnosis['id']==row['id'] and diagnosis['root_id_and_partition_original_assertions_pass'] and diagnosis['weights_before_after_saved_exact']
    assert ns['bridge'].identity_record(row['method'],row['id'])==row['identity']
    assert receipt['external_identity_record_sha256']==b.canonical(record) and receipt==ns['gate']['receipts'][len(rows)]
    assert receipt['weights_before']==receipt['weights_after']
    paths={k:Path(record[k]['path']) for k in b.KINDS};before=b.measure(paths,record)
    cfg=core.ExperimentConfig(**record['config']);root_ids,root_y,root_s,actual_root=ev.rebuild_root(core,cfg,record,ns['ids'],ns['y'],ns['sensitive'])
    v2=types.SimpleNamespace(np=ns['np'],require=b.need,digest=b.sha,canonical=b.canonical,VIEWS=ev.VIEWS,check_native=ev.check_native)
    scope=dict(v2=v2,path=path,r=receipt,root_ids=root_ids,root_y=root_y,root_s=root_s,ids=ns['ids'],y=ns['y'],s=ns['sensitive'],record=record,cfg=cfg,evaluator=ev,core=core,original_result=b.read(paths['result']))
    exec(compile(ast.Module(body=[block],type_ignores=[]),'<exact original check_saved saved-array block>','exec'),scope)
    assert b.measure(paths,record)==before
    rows.append({'id':row['id'],'array_sha256':b.sha(path/'validation_predictions.npz'),'receipt_sha256':b.sha(path/'receipt.json'),'native_comparison':scope['comparison'],'root_ID_partition_original_assertions_pass':True,'root_fit_parameters_and_diagnostics_exact':True,'all_saved_predictions_and_counts_and_metrics_exact':True,'local_full_root_receipt_exact':not diagnosis['root_receipt_differences'],'preserved_root_audit_differences':diagnosis['root_receipt_differences']})
report={'status':'OFFSERVER_ORIGINAL_ARRAY_FIT_METRICS_BLOCK_PASS_FULL_ROOT_AUDIT_REMAINS_LINUX_ONLY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':rows,'original_saved_check_file_sha256':b.sha(source),'exact_saved_array_AST_block_sha256':hashlib.sha256(ast.dump(block,include_attributes=False).encode()).hexdigest(),'runtime':{'python':sys.version,'numpy':ns['np'].__version__,'pandas':ns['pd'].__version__,'torch':ns['torch'].__version__},'cached_root_refits':3,'new_CNN':0,'new_training':0,'test':False,'native_tolerance_unchanged':1e-12,'Windows_full_check_failure_preserved':True,'no_root_adoption':True}
with (O/'OFFSERVER_ARRAY_REFIT_CHECK.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':report['status'],'records':3,'native_differences':[r['native_comparison']['max_abs_difference'] for r in rows],'proof_sha256':b.sha(O/'OFFSERVER_ARRAY_REFIT_CHECK.json')}))
