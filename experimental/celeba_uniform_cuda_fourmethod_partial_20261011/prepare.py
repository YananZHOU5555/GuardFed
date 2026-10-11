"""Copy prior formatting/statistical source and bind existing roots; never compute statistics."""
from pathlib import Path
import json,hashlib,ast,difflib
H=Path(__file__).resolve().parent;R=H.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def put(n,v):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
parent=R/'tmp/celeba_eleven_method_native_table_prepare_20261011/build.py';text=parent.read_text('utf8');tree=ast.parse(text)
def rawfunc(name):
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 return ''.join(text.splitlines(True)[node.lineno-1:node.end_lineno])
original=rawfunc('summaries');new=original.replace('def summaries(rows, old_summary):','def summaries(rows):').replace('panels = {}; aggregates = {}; old_checks = 0','panels = {}; aggregates = {}').replace("        need(panel_rows[:100] == old_summary[panel], 'Original ten-method native statistics/IDs/display changed')\n",'').replace('        old_checks += 100*3\n','').replace('return panels,aggregates,old_checks','return panels,aggregates')
changes=[('def summaries(rows, old_summary):','def summaries(rows):'),('panels = {}; aggregates = {}; old_checks = 0','panels = {}; aggregates = {}'),("        need(panel_rows[:100] == old_summary[panel], 'Original ten-method native statistics/IDs/display changed')\n",''),('        old_checks += 100*3\n',''),('return panels,aggregates,old_checks','return panels,aggregates')]
put('STATS_SOURCE.json',dict(parent={'path':str(parent),'sha256':sha(parent)},original_summaries=original,changes=[dict(before=a,after=b) for a,b in changes],derived_summaries_sha256=hashlib.sha256(new.encode()).hexdigest(),math_changed=False,original_display_function=rawfunc('original_values_for')))
header="""from pathlib import Path
from statistics import mean,stdev
from collections import defaultdict
import ast,hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
METHODS=['FedAvg','GuardFed-AD2+','FLGMM','LoGoFair-DP']
DISTS=['IID','non-IID']
ATTACKS=['Benign','F Flip','FedSA','S-DFA','Sp-DFA']
METRICS=[('accuracy','ACC (%) ↑',100,2),('aeod','AEOD ↓',1,4),('aspd','ASPD ↓',1,4)]
SEEDS={'ten':list(range(91001,91011)),'nonselection_nine':list(range(91002,91011)),'matching_six':list(range(91005,91011))}
def need(ok,message):
 if not ok:raise ValueError(message)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
"""
notes="Partial four-method uniformly-CUDA validation; not the final17-method comparison or a newly selected manuscript endpoint. Native predictions only; ACC in %, AEOD/ASPD on [0,1]. AEOD is the implemented absolute TPR gap, not full equalized odds. Mean ± sample SD(ddof1). Original70-round CNN checkpoints are reused; CNN/root+valid forward evaluation uses CUDA FP32, while each frozen method retains its original postprocessing implementation/device. LoGoFair-DP applies its unchanged fitted recipe07/fitseed1719 state to adopted FedAvg CUDA margins, with netcal1.3.6 CPU postprocessing and20 image-ID virtual cohorts, not real training clients; DP means demographic parity. No refit or new CNN is performed for LoGo. Original cu128/cu130 training histories and adaptation boundaries remain; CUDA evaluation is not a claim of training-runtime equivalence. Seed91001 participated in validation selection; n9/n6 are sensitivity subsets, not untouched test cohorts. All fixed seeds, negative outcomes and constant predictors are retained. No score, best-seed selection, significance or final-test claim. Each native pipeline includes its own postprocessing and does not isolate aggregation causality."
markdown=rawfunc('markdown').replace('CelebA native view — eleven-method author-review candidate','partial four-method uniformly-CUDA validation')
source=header+'\n'+rawfunc('original_values_for')+'\n\n'+new+'\n\nNOTES='+repr(notes)+'\n\n'+markdown+'\n'
with (H/'statistics_original.py').open('x',encoding='utf8',newline='\n') as f:f.write(source)
roots={
'FedAvg':('tmp/FedAvg_cuda100_saved_cache_closure_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json','83a047fdd4d9c997c3ca31ff713de134575f1bcc6c75f6da579031f604efb083'),
'GuardFed-AD2+':('tmp/Full100_cuda_saved_closure_prepare_v2_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json','0d33ecb624323f0f334debcb12f888886f85410cd9b0d5607c687dbb6575cc08'),
'FLGMM':('tmp/FL100_cuda100_saved_closure_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json','ea935ead7da50ba7eff0dd7e4e4b289ca9365427096ad8431808ae4d76acb751'),
'LoGoFair-DP':('tmp/LoGo100_cuda100_saved_closure_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json','c926f7ee3cd6dea31f178d3a6eab3661826f9ecd4e7c90e54d3027e6b94c1317')}
bound={};statuses={}
for m,(rel,h) in roots.items():
 p=R/rel;assert sha(p)==h;bound[m]=dict(path=str(p),sha256=h);statuses[m]=json.loads(p.read_bytes())['status']
old=R/'outputs/guardfed_tables/celeba_eleven_method_native_20261011/records_native_1100.json';proof=old.parent/'ROOT_REVIEW.json';renderer=R/'outputs/guardfed_tables/celeba_nine_method_final_20261004/build_tables.py'
put('INPUT_PINS.json',dict(roots=bound,root_statuses=statuses,metadata={'path':str(old),'sha256':sha(old)},metadata_root={'path':str(proof),'sha256':sha(proof)},renderer=dict(path=str(renderer.relative_to(R)).replace('\\','/'),sha256=sha(renderer)),original_builder=dict(path=str(parent),sha256=sha(parent))))
put('ROOT_BINDING.template.json',dict(status='ACTUAL_FOUR_ROOTS_BOUND_STATISTICS_NOT_AUTHORIZED',root_execution_authorized=False,view='native',final_test=False,new_fit=0,new_inference=0,roots=bound,root_source_review={'path':None,'sha256':None}))
print('Bound actual four100 roots; statistical functions prepared, not executed.')
