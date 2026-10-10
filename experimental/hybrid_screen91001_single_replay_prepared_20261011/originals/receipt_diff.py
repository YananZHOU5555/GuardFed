"""One bounded root reconstruction diagnostic; no fitting, scoring or acceptance."""
import ast,datetime,hashlib,importlib.util,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
S=R/'tmp/celeba_added_cnn_exact3_scientific_acceptance_20261010'
O=R/'tmp/celeba_added_cnn_exact3_root_execution_20261010'
BASE=Path('F:/YananResearchStorage/GuardFed/added_cnn_exact3_valid_20261010/attempt001/verified_extract')
sys.path.insert(0,str(S))
spec=importlib.util.spec_from_file_location('sealed_offserver_setup',S/'verify_offserver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
text=(S/'verify_offserver.py').read_text(encoding='utf-8')
assert hashlib.sha256((S/'verify_offserver.py').read_bytes()).hexdigest()=='f8760d536837f68529d10dc6cf7985f2cddc1a4a75faa4ef8e4aaa0f944426e2'
node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='main')
prefix=[]
for statement in node.body:
    if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='results' for t in statement.targets):break
    prefix.append(statement)
sys.argv=[str(S/'verify_offserver.py'),'--bundle',str(BASE/'bundle'),'--metadata-npz',str(BASE/'metadata.npz'),'--output',str(O/'DIAGNOSTIC_UNUSED_OUTPUT.json'),'--allow-original-cached-root-refit']
ns=dict(m.__dict__);exec(compile(ast.Module(body=prefix,type_ignores=[]),'<sealed verifier setup only; no results/fit loop>','exec'),ns)
def diff(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        output=[]
        for k in sorted(set(a)|set(b)):
            if k not in a or k not in b:output.append({'path':path+'/'+str(k),'local':a.get(k),'saved':b.get(k),'kind':'missing'})
            else:output.extend(diff(a[k],b[k],path+'/'+str(k)))
        return output
    if isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):return [{'path':path,'kind':'length','local':len(a),'saved':len(b)}]
        return sum((diff(x,y,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))),[])
    if a==b:return []
    out={'path':path,'local':a,'saved':b,'kind':'scalar'}
    if isinstance(a,float) and isinstance(b,float):out.update(local_hex=a.hex(),saved_hex=b.hex(),difference=a-b)
    return [out]
rows=[]
for row in ns['rows']:
    record=ns['b'].runtime_record(row);saved=ns['b'].read(BASE/'bundle'/row['id']/'receipt.json')
    cfg=ns['core'].ExperimentConfig(**record['config'])
    root_ids,ry,rs,actual=ns['ev'].rebuild_root(ns['core'],cfg,record,ns['ids'],ns['y'],ns['sensitive'])
    rows.append({'id':row['id'],'root_id_and_partition_original_assertions_pass':True,'root_receipt_differences':diff(actual,saved['root_reconstruction']),'weights_before_after_saved_exact':saved['weights_before']==saved['weights_after'],'root_n':len(root_ids)})
report={'status':'ROOT_RECONSTRUCTION_DIAGNOSTIC_ONLY_NOT_ACCEPTANCE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.version,'numpy':ns['np'].__version__,'pandas':ns['pd'].__version__,'torch':ns['torch'].__version__,'records':rows,'new_fit':0,'new_CNN':0,'new_training':0,'test':False,'original_functions_unchanged':True,'scientific_tolerance_changed':False}
with (O/'OFFSERVER_ROOT_DIAGNOSTIC.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps(report,ensure_ascii=False))
