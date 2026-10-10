"""Bind this short writing update to accepted JSON; never calculate new statistics."""
from pathlib import Path
import hashlib,json,re,sys
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent;R=D.parents[1]
B='docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_five_scenes_20261010/'
C='tmp/rebuttal_C40_addendum_prepared_20261010/'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
paths=[B+'ROOT_VERIFICATION.json',B+'snapshot/tables.json',B+'snapshot/TABLES.md',B+'snapshot/cross_scene_seed_first.json',B+'snapshot/coverage.json',B+'snapshot/records.json',B+'snapshot/SOURCE_BINDINGS.json',B+'snapshot/verification.json','tmp/celeba_mechanism_C50_root_arithmetic_review_20261010/ROOT_ARITHMETIC_REVIEW.json',C+'C40_REVIEWER_ADDENDUM.md',C+'C40_MANUSCRIPT_INSERTIONS.md','docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C20_20261009/rebuttal_integrated_20261009.md']
pins={p:dict(sha256=sha(R/p),bytes=(R/p).stat().st_size) for p in paths}
assert pins[B+'ROOT_VERIFICATION.json']['sha256']=='811ad551c1398f6e57681f59046c316b05c609ef30d4897158460b8121b1a28d'
assert pins[paths[8]]['sha256']=='8903af110c3bf5727d7dbe5687711b0b2ee30c23ce0ef893c62c370ef2da69dc'
t=read(R/(B+'snapshot/tables.json'));cross=read(R/(B+'snapshot/cross_scene_seed_first.json'))
docs={n:(D/n).read_text('utf8') for n in ['C50_REVIEWER_ADDENDUM.md','C50_MANUSCRIPT_INSERTIONS.md']}
cells=[]
for i,n,metrics in [(0,10,['accuracy_pct','aeod','aspd']),(3,10,['accuracy_pct','aeod','aspd']),(1,9,['accuracy_pct']),(2,6,['accuracy_pct'])]:
 for m in metrics:
  value=t['panels'][i]['rows'][14][m];d=3 if m=='accuracy_pct' else 5
  display=f"{value['mean']:+.{d}f} ± {value['sample_sd_ddof1']:.{d}f}"
  occurrences={name:[j for j,line in enumerate(text.splitlines(),1) if display in line] for name,text in docs.items()}
  assert all(occurrences.values()),display
  cells.append(dict(source_path=B+'snapshot/tables.json',pointer=f'/panels/{i}/rows/14/{m}',value=value,view=t['panels'][i]['view'],scene='Sp-DFA',panel_n=n,metric=m,decimal_places=d,display=display,document_lines=occurrences))
comments=[]
for i,line in enumerate((R/(C+'C40_REVIEWER_ADDENDUM.md')).read_text('utf8').splitlines(),1):
 if line.startswith('> 2.') or line.startswith('> 7.'):
  assert line in docs['C50_REVIEWER_ADDENDUM.md']
  comments.append(dict(source_path=C+'C40_REVIEWER_ADDENDUM.md',source_line=i,text=line,document='C50_REVIEWER_ADDENDUM.md'))
facts=[]
def fact(label,path,pointer,value):facts.append(dict(label=label,source_path=path,pointer=pointer,value=value))
root=read(R/(B+'ROOT_VERIFICATION.json'))
for key in ['checked_utc','unique_records','paired_models','complete_scenes','replay_devices','training_torch','seed_panels','test','primary_endpoint','other_C_scenes_complete','whole_rebuttal_complete','original_four_scene80_records_exact','original_four_scene648_statistics_exact','original_four_scene324_cells_preserved']:
 fact(key,B+'ROOT_VERIFICATION.json','/'+key,root[key])
for i in [0,1,2]:fact('fixed_seed_panel',B+'snapshot/tables.json',f'/panels/{i}/seeds',t['panels'][i]['seeds'])
fact('valid_rows',B+'snapshot/records.json','/records/0/data_contract/actual_evaluation_rows',19867)
fact('seed_first_weighting',B+'snapshot/cross_scene_seed_first.json','/scene_weighting',cross['scene_weighting'])
directions=[]
def signs(path,value,pointer,expected,label):
 assert [1 if value[m]['mean']>0 else -1 if value[m]['mean']<0 else 0 for m in ['accuracy_pct','aeod','aspd']]==expected
 directions.append(dict(label=label,source_path=path,pointer=pointer,signs=expected))
for i in range(9):
 signs(B+'snapshot/tables.json',t['panels'][i]['rows'][14],f'/panels/{i}/rows/14',[-1,-1,1] if i in [2,8] else [1,1,1] if i in [3,4,5] else [1,-1,1],'Sp-DFA fixed panel')
for i in [0,3,6]:signs(B+'snapshot/cross_scene_seed_first.json',cross['panels'][i]['rows'][2],f'/panels/{i}/rows/2',[1,1,1] if i==3 else [1,-1,1],'ten-seed cross-scene trade-off')
for i,j,expected in [(0,5,[1,1,1]),(1,5,[1,1,1]),(2,5,[1,-1,1]),(0,2,[-1,1,-1]),(0,8,[1,-1,1]),(1,8,[1,1,-1]),(2,8,[1,1,-1]),(4,8,[1,-1,-1]),(5,8,[1,1,1]),(0,11,[-1,1,1]),(1,11,[-1,1,1]),(2,11,[1,1,1])]:
 signs(B+'snapshot/tables.json',t['panels'][i]['rows'][j],f'/panels/{i}/rows/{j}',expected,'retained earlier counterexample/subset direction')
metadata=dict(status='ACTUAL_ROOT_ADOPTED_C50_SHORT_WRITING_SOURCE_POINTERS',root_adoption_sha256=pins[B+'ROOT_VERIFICATION.json']['sha256'],source_pins=pins,quoted_mean_SD_cells=cells,verbatim_comment_bindings=comments,fact_bindings=facts,direction_bindings=directions,textual_source_bindings=[dict(source_path=B+'snapshot/TABLES.md',source_line=3,contains='round70'),dict(source_path=C+'C40_REVIEWER_ADDENDUM.md',contains='six other image-control variants'),dict(source_path=C+'C40_REVIEWER_ADDENDUM.md',contains='official-test exposure'),dict(source_path=C+'C40_REVIEWER_ADDENDUM.md',contains='historical/current driver equality was not established')],new_statistics=False,manuscript_applied=False)
with (D/'SOURCE_POINTERS.json').open('x',encoding='utf8',newline='\n') as f:json.dump(metadata,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(source_pins=len(pins),quoted_mean_SD_cells=len(cells),scalar_pointers=2*len(cells),original_comments=len(comments),facts=len(facts),directions=len(directions))))
