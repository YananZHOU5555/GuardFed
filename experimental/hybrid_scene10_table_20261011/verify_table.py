"""Independent stdlib fsum/ddof1/table/source audit;no builder/statistics import."""
from pathlib import Path
import hashlib,json,math
R=Path(__file__).resolve().parents[2];O=R/'outputs/guardfed_tables/celeba_hybrid_IID_Benign10_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
inputs=read(O/'INPUTS.json')
for p,v in inputs['files'].items():assert sha(R/p)==v
assert sha(R/inputs['record8_path'])==inputs['record8_sha256'] and sha(R/inputs['builder_path'])==inputs['builder_sha256']
rootpath='tmp/celeba_hybrid_native9_root_adoption_20261011/ROOT_ADOPTION.json';root=read(R/rootpath);prior=read(R/'tmp/celeba_hybrid_first1_root_adoption_20261010/ROOT_ADOPTION.json');old=read(R/'tmp/hybrid_native_after1_20261011/IID_BENIGN10_NATIVE.json')
assert inputs['files'][rootpath]=='1434b40de5116bf3d53bc5a6ae2bd3b90f4e54222f23ad099ff72b1fd3ef1775' and root['cumulative_accepted']==9
records=read(O/'records.json')['records'];tables=read(O/'tables.json');coverage=read(O/'COVERAGE.json');md=(O/'TABLES.md').read_text('utf8')
assert [r['seed'] for r in records]==list(range(91001,91011)) and len({r['id'] for r in records})==len({r['checkpoint_sha256'] for r in records})==10
def pointer(v,path):
 for part in path.strip('/').split('/') if path.strip('/') else []:v=v[int(part)] if isinstance(v,list) else v[part]
 return v
for a,b in zip(records,old['records']):
 assert all(a[k]==b[k] for k in ['id','seed','metrics','checkpoint_sha256']) and (a['rounds'],a['evaluation_split'],a['n_eval'])==(70,'valid',19867)
 assert inputs['files'][a['source_path']]==a['source_sha256'];v=pointer(read(R/a['source_path']),a['source_pointer'])
 if a['seed']==91002:assert v['accepted_new_ids']==[a['id']] and v['metrics']==a['metrics'] and v['checkpoint_sha256']==a['checkpoint_sha256']
 else:assert all(v[k]==a[k] for k in ['id','metrics','checkpoint_sha256','acceptance_sha256'])
 assert set(a['metrics'])=={'accuracy','aeod','aspd'} and all(math.isfinite(x) for x in a['metrics'].values())
assert coverage['unique_records']==coverage['unique_checkpoints']==10 and coverage['complete_scenes']==1 and not coverage['three_view_table'] and not coverage['whole100_complete']
expected=[('all10',list(range(91001,91011))),('exclude_selection_seed9',list(range(91002,91011))),('fixed_last6',list(range(91005,91011)))];differences=[];cells=[]
for p,(name,seeds) in zip(tables['panels'],expected):
 selected=[x for x in records if x['seed'] in seeds];assert p['name']==name and p['seeds']==seeds and p['n']==len(seeds)
 assert p['ids']==[x['id'] for x in selected] and p['checkpoint_sha256']==[x['checkpoint_sha256'] for x in selected]
 for metric in ['accuracy','aeod','aspd']:
  x=[r['metrics'][metric] for r in selected];mean=math.fsum(x)/len(x);sd=math.sqrt(math.fsum((v-mean)**2 for v in x)/(len(x)-1));saved=p['statistics'][metric]
  assert (saved['n'],saved['ddof'])==(len(x),1)
  for label,value in [('mean',mean),('sample_SD',sd)]:differences.append(abs(value-saved[label]));assert abs(value-saved[label])<=1e-12
  factor=100 if metric=='accuracy' else 1;digits=6 if metric=='accuracy' else 8;text=f'{mean*factor:.{digits}f} ± {sd*factor:.{digits}f}'
  assert p['display'][metric]==text and text in md;cells.append(text)
assert len(tables['panels'])==3 and len(differences)==18 and len(cells)==9
assert root['all_metrics_same_terminal_checkpoint'] and root['source_data_before_after_exact'] and not root['final_test']
for target in ['records.json','COVERAGE.json','ENVIRONMENT_SCOPE.json','INPUTS.json']:assert (O/target).is_file() and f']({target})' in md
proof=dict(status='PASS_INDEPENDENT_BOUNDED_NATIVE_SCENE_STATISTICS_SOURCE_AND_DISPLAY',unique_records=10,unique_checkpoints=10,complete_scenes=1,panels=3,seed_panels=[p['seeds'] for p in tables['panels']],mean_SD_scalars_recomputed=18,display_cells_verified=9,max_abs_difference=max(differences),tolerance=1e-12,ddof=1,all_ten_original_metrics_and_checkpoint_ids_exact=True,root_native9_sha256=inputs['files'][rootpath],no_CNN_or_fit_or_training=True,no_arrays_or_Torch=True,whole100_complete=False,three_view=False,new_final_test=False,checker_path=Path(__file__).relative_to(R).as_posix(),checker_sha256=sha(Path(__file__)),table_sha256=sha(O/'tables.json'),records_sha256=sha(O/'records.json'),markdown_sha256=sha(O/'TABLES.md'))
with (O/'VERIFICATION.json').open('x',encoding='utf8',newline='\n') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(proof,indent=2))
