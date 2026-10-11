"""Bind actual root-adopted fixed2 evidence to the unchanged remaining queue contract."""
from pathlib import Path
import argparse,base64,hashlib,json
HERE=Path(__file__).resolve().parent
CANARIES=['LoGoFair-DP_07_FedAvg_IID_Benign_seed91001','LoGoFair-DP_07_FedAvg_IID_F Flip_seed91001']
BASE='/workspace/guardfed_checks/uniform_cuda_LoGo100_20261011_attempt001'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(inputs,rootpin):
 need(inputs['execution_authorized'] is True,'Actual root remaining authorization required')
 rp=Path(rootpin['path']);need(sha(rp)==rootpin['sha256'],'Changed actual root adoption');raw=rp.read_bytes();r=json.loads(raw)
 need(r['root_accepted'] is True and r['ids']==CANARIES and set(r['records'])==set(CANARIES),'Actual adopted fixed2 required')
 need(r['cohort']=={'path':BASE+'/adapter/EXACT100.json','sha256':'52d6a15e077eb50b1d81d2aa0e481c6e2e5fb5dfeea24e84ba6036535dffed9b'},'Foreign cohort')
 need(r['FedAvg_CUDA_root']=={'path':'/workspace/guardfed_checks/FedAvg100_cuda_adopted_cache_index_20261011_actual002/ROOT_SCIENTIFIC_ADOPTION.json','sha256':'83a047fdd4d9c997c3ca31ff713de134575f1bcc6c75f6da579031f604efb083'},'Foreign adopted CUDA cache')
 for identity in CANARIES:
  for role,name in [('acceptance','formal_saved_acceptance.json'),('binding','formal_binding.json')]:
   pin=r['records'][identity][role];need(set(pin)=={'path','sha256'} and pin['path']==BASE+'/runs/'+identity+'/'+name and len(pin['sha256'])==64 and all(c in '0123456789abcdef' for c in pin['sha256']),'Invalid fixed2 original pin')
 audit=Path(r['offserver_audit']['path']);need(sha(audit)==r['offserver_audit']['sha256'],'Missing or changed actual offserver audit')
 json.loads(audit.read_bytes())
 gate={k:r[k] for k in ['root_accepted','ids','cohort','FedAvg_CUDA_root','records']}
 gate.update(root_adoption=rootpin,offserver_audit=r['offserver_audit'])
 return dict(inputs,canary_gate=gate,root_adoption_bytes=base64.b64encode(raw).decode(),audit_bytes=base64.b64encode(audit.read_bytes()).decode())
def main():
 p=argparse.ArgumentParser();p.add_argument('--root-inputs',type=Path,required=True);p.add_argument('--root-inputs-sha256',required=True);p.add_argument('--root-adoption',type=Path,required=True);p.add_argument('--root-adoption-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 need(sha(a.root_inputs)==a.root_inputs_sha256,'Changed root start inputs')
 out=bind(json.loads(a.root_inputs.read_text()),{'path':str(a.root_adoption.resolve()),'sha256':a.root_adoption_sha256})
 need(out['start_source_sha256']==sha(HERE/'start_remote.py') and out['launcher_sha256']==sha(HERE/'run_root.py'),'Changed starter/launcher')
 with a.output.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({'bound_input_sha256':sha(a.output),'path':str(a.output),'scientific_execution':False}))
if __name__=='__main__':main()
