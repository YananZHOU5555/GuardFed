"""Read-only original five-result/two-pair checks. CPU tensor reads only; never inference."""
from pathlib import Path
import argparse,ast,hashlib,json,sys
if sys.flags.optimize:raise RuntimeError('Optimized Python forbidden')
PACKAGE='6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230'
CANARIES='5052d4f31fe2fb63ac16706fb287b8ce64efe61b8974f83d5d091b475216cee7'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def comparison_body(source):
 tree=ast.parse(source);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 block=next(n for n in main.body if isinstance(n,ast.Try)).body
 start=next(i for i,n in enumerate(block) if isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names))
 selected=block[start:start+3]
 assert isinstance(selected[1],ast.Assign) and isinstance(selected[2],ast.For)
 assert ast.unparse(selected[1])=='pairs = []'
 return ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[]))
def verify(stage):
 stage=Path(stage).resolve();assert sha(stage/'PACKAGE_SHA256.json')==PACKAGE and sha(stage/'run_canaries.py')==CANARIES
 assert not list((stage/'preflight').rglob('failure*.json'))
 sys.path.insert(0,str(stage))
 import screen_common as sc
 assert Path(sc.__file__).resolve()==stage/'screen_common.py'
 protocol,manifest=sc.local_identity()
 gate=sc.read(stage/'GATE_ACCEPTANCE.json')
 assert (gate['status'],gate['package_sha256'],gate['accepted_new_canaries'],gate['same_horizon_pairs'],gate['horizon'],gate['formal_table_samples'])==('PASS',PACKAGE,5,2,3,0)
 actual={p.relative_to(stage).as_posix():sha(p) for p in (stage/'preflight').rglob('*') if p.is_file()}
 assert actual==gate['artifact_hashes'],'Exact final artifact set/hash mismatch'
 import run_canaries as original
 assert Path(original.__file__).resolve()==stage/'run_canaries.py'
 code=comparison_body((stage/'run_canaries.py').read_text())
 ns=dict(vars(original),manifest=manifest)
 exec(compile(code,str(stage/'run_canaries.py')+'[UNCHANGED_SAVED_COMPARISON]','exec'),ns)
 assert len(ns['pairs'])==2 and ns['pairs']==gate['pairs']
 assert len(manifest['preflight_jobs'])==5
 assert len(list((stage/'preflight/references').iterdir()))==2
 import torch,numpy
 return {'status':'PASS_SAVED_ORIGINAL_COMPARISON','package_sha256':PACKAGE,'original_compare_source_sha256':CANARIES,
  'gate_sha256':sha(stage/'GATE_ACCEPTANCE.json'),'accepted_new':5,'same_horizon_pairs':2,'total_runs':7,'rounds':3,
  'pair_ids':ns['pairs'],'artifact_count':len(actual),'formal_table_samples':0,'CNN_calls':0,
  'local_verification_runtime':{'python':sys.version,'torch':torch.__version__,'numpy':numpy.__version__},
  'runtime_statement':'Saved tensor/record comparison only. This verifier runtime is not the training runtime and asserts no cross-device training equivalence.'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 result=verify(a.stage)
 with a.out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
