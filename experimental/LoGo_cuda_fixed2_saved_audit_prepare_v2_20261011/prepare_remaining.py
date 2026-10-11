"""Source-only extractor. Do not rerun after sealing."""
from pathlib import Path
import ast,hashlib,json,difflib
H=Path(__file__).resolve().parent;R=H.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def raw_function(p,name):
    lines=p.read_bytes().splitlines(keepends=True);tree=ast.parse(p.read_text('utf8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    return b''.join(lines[node.lineno-1:node.end_lineno])
summary=R/'tmp/celeba_logofair100_summary_20261010/summary100.py';core=R/'tmp/celeba_added_cnn_three_view_gate_preparation_20261010/originals/core.py';ev=R/'tmp/old800_cuda3_saved_audit_prepare_20261011/original_evaluator.py.txt'
assert sha(core)=='cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed'
functions=[(summary,'saved_metric_error'),(core,'compute_metrics'),(ev,'check_binary'),(ev,'group_metrics')]
chunks=[raw_function(p,n) for p,n in functions];(H/'original_metric_functions.py.txt').write_bytes(b'\n\n'.join(chunks)+b'\n')
raw=summary.read_bytes();start=raw.index(b"            with old_bridge.np.load(folder / 'scores_predictions.npz'");end=raw.index(b'            records.append(',start);block=raw[start:end]
(H/'original_saved_decision_block.py.txt').write_bytes(block)
p=H/'audit_saved.py';body=p.read_bytes();assert body.count(b'            # ORIGINAL_SAVED_BLOCK\n')==1;p.write_bytes(body.replace(b'            # ORIGINAL_SAVED_BLOCK\n',block))
prefix=R/'tmp/FedAvg_cuda100_saved_cache_closure_prepare_20261011/original_prefix_functions.py.txt';(H/prefix.name).write_bytes(prefix.read_bytes())
cohort=R/'tmp/LoGo100_uniform_cuda_cache_adapter_prepare_20261011/EXACT100.json';scope=json.loads(cohort.read_bytes());ids=['LoGoFair-DP_07_FedAvg_IID_Benign_seed91001','LoGoFair-DP_07_FedAvg_IID_F Flip_seed91001'];refs=dict(source_cohort={'path':str(cohort.relative_to(R)),'sha256':sha(cohort)},records=[next(r for r in scope['records'] if r['post_id']==i) for i in ids]);save('EXACT2_REFS.json',refs)
parent=R/'tmp/FedAvg_cuda100_saved_cache_closure_prepare_20261011/transport.py';text=parent.read_text('utf8');tail=text[text.index("cmd=['ssh','-p','60350'"):]
tail=tail.replace('OLD800_FIXED3_ATTEMPT1','LOGO_FIXED2')
header='''from pathlib import Path,PurePosixPath
import argparse,base64,datetime,hashlib,json,subprocess,sys,tarfile
import metadata_gate as g
R=Path.cwd();H=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'tmp'))
from guardfed_local_storage import check_bulk_storage
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):
 with Path(p).open('x',encoding='utf8',newline='\\n') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\\n')
p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--inputs-sha256',required=True);p.add_argument('--source-seal-sha256',required=True);a=p.parse_args()
assert __debug__ and sha(a.inputs)==a.inputs_sha256
inp=json.loads(a.inputs.read_bytes());g.input_gate(inp);assert inp['root_authorized_transport'] is True
assert sha(H/'FILES_SHA256.json')==a.source_seal_sha256
for n,item in json.loads((H/'FILES_SHA256.json').read_bytes())['files'].items():assert Path(n).name==n and sha(H/n)==item['sha256']
review=Path(inp['source_review']['path']);assert sha(review)==inp['source_review']['sha256'] and json.loads(review.read_bytes())['source_adoptable'] is True and json.loads(review.read_bytes())['source_seal_sha256']==a.source_seal_sha256
assert sha(R/'tmp/guardfed_local_storage.py')=='2482f2f46243abafd33ea2fadc94d555c0d80327fefe19caea3eb8f3f6bb3d63'
volume=check_bulk_storage(250*1024**2);F=Path('F:/YananResearchStorage/GuardFed/uniform_cuda_LoGo100_20261011/fixed2_saved_attempt001');assert not F.exists();F.mkdir(parents=True)
payload={'inputs':inp,'refs':json.loads((H/'EXACT2_REFS.json').read_bytes()),'gate':base64.b64encode((H/'metadata_gate.py').read_bytes()).decode(),'remote':base64.b64encode((H/'remote_export.py').read_bytes()).decode()}
code="import base64,json,sys,types\\nx=json.loads(base64.b64decode("+repr(base64.b64encode(json.dumps(payload).encode()).decode())+"))\\ng=types.ModuleType('metadata_gate');g.__file__='/source-only-bound/metadata_gate.py';sys.modules['metadata_gate']=g\\nexec(compile(base64.b64decode(x['gate']),g.__file__,'exec'),g.__dict__)\\nns={'__file__':'/source-only-bound/remote_export.py'};exec(compile(base64.b64decode(x['remote']),ns['__file__'],'exec'),ns);ns['main'](x['inputs'],x['refs'])\\n"
'''
(H/'transport.py').write_text(header+tail,encoding='utf8',newline='\n')
save('TRANSPORT_INVERSE.json',{'parent':str(parent.relative_to(R)),'parent_sha256':sha(parent),'start_marker':"cmd=['ssh','-p','60350'",'change':{'before':'OLD800_FIXED3_ATTEMPT1','after':'LOGO_FIXED2'},'parent_tail_sha256':hashlib.sha256(text[text.index("cmd=['ssh','-p','60350'"):].encode()).hexdigest(),'raw_physical_newline_note':'Parent CRLF normalized by read_text; suffix normalized text/AST exact under single filename rename.'})
pins={str(p.relative_to(R)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in [summary,core,ev,prefix,parent,cohort]}
save('SOURCE_REUSE.json',{'source_pins':pins,'functions':[{'name':n,'source':str(p.relative_to(R)),'raw_sha256':hashlib.sha256(c).hexdigest()} for (p,n),c in zip(functions,chunks)],'saved_decision_block_raw_sha256':hashlib.sha256(block).hexdigest(),'saved_decision_block_unchanged':True,'metric_functions_raw_unchanged':True,'transport_suffix_inverse':'TRANSPORT_INVERSE.json','no_scientific_calls':True})
def nullpin():return dict(path=None,sha256=None)
save('ROOT_INPUTS.template.json',dict(status='UNBOUND_NOT_AUTHORIZATION',root_authorized_transport=False,root_authorized_audit=False,exact_ids=ids,server_base=None,service=None,root_queue_auth=nullpin(),deployment_seal=nullpin(),task_accepted={i:nullpin() for i in ids},source_review=nullpin(),FedAvg_CUDA_root={'path':'/workspace/guardfed_checks/FedAvg100_cuda_adopted_cache_index_20261011_actual002/ROOT_SCIENTIFIC_ADOPTION.json','sha256':'83a047fdd4d9c997c3ca31ff713de134575f1bcc6c75f6da579031f604efb083'},FedAvg_CUDA_index={'path':'/workspace/guardfed_checks/FedAvg100_cuda_adopted_cache_index_20261011_actual002/ROOT_CACHE_INDEX.json','sha256':'6a21265f08fb26da130d95c67bf4130039b25916f44a543cd695e15665a25b64'},local_FedAvg_root={'path':str((R/'tmp/FedAvg_cuda100_saved_cache_closure_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json').resolve()),'sha256':'83a047fdd4d9c997c3ca31ff713de134575f1bcc6c75f6da579031f604efb083'},local_FedAvg_index={'path':str((R/'tmp/FedAvg100_cuda_linux_index_actual002_20261011/ROOT_CACHE_INDEX.json').resolve()),'sha256':'6a21265f08fb26da130d95c67bf4130039b25916f44a543cd695e15665a25b64'},local_FedAvg_transport={'path':str((R/'tmp/FedAvg_cuda100_saved_cache_closure_prepare_20261011/OLD800_FIXED3_ATTEMPT1_TRANSPORT_VERIFICATION.json').resolve()),'sha256':'e02c2efb1b1ed84952eeb87ae298690a64b130ad25be61c590d3bc709fb30c74'},transport=nullpin(),archive_sha256=None))
print('Source prepared; no scientific functions imported or executed')
