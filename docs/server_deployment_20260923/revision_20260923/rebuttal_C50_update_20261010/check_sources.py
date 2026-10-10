"""Check quoted accepted numbers, original comments, scope and links; no statistics."""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,re,sys
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent;R=D.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def pointer(value,path):
 for part in path.lstrip('/').split('/') if path else []:
  part=part.replace('~1','/').replace('~0','~');value=value[int(part)] if isinstance(value,list) else value[part]
 return value
def main():
 a=argparse.ArgumentParser();a.add_argument('--report',type=Path);args=a.parse_args()
 s=read(D/'SOURCE_POINTERS.json');data={}
 for path,pin in s['source_pins'].items():
  p=R/path;assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],('source',path)
  if p.suffix=='.json':data[path]=read(p)
 docs={n:(D/n).read_text('utf8') for n in ['C50_REVIEWER_ADDENDUM.md','C50_MANUSCRIPT_INSERTIONS.md']}
 mapped=set()
 for b in s['quoted_mean_SD_cells']:
  v=pointer(data[b['source_path']],b['pointer']);d=b['decimal_places']
  assert v==b['value'] and f"{v['mean']:+.{d}f} ± {v['sample_sd_ddof1']:.{d}f}"==b['display']
  for name,lines in b['document_lines'].items():
   for line in lines:assert b['display'] in docs[name].splitlines()[line-1];mapped.add((name,line,b['display']))
 for name,text in docs.items():
  for i,line in enumerate(text.splitlines(),1):
   for cell in re.findall(r'[+\-]?\d+(?:\.\d+)? ± \d+(?:\.\d+)?',line):assert (name,i,cell) in mapped,('unmapped cell',name,i,cell)
 for b in s['fact_bindings']:assert pointer(data[b['source_path']],b['pointer'])==b['value'],b['label']
 for b in s['verbatim_comment_bindings']:
  line=(R/b['source_path']).read_text('utf8').splitlines()[b['source_line']-1]
  assert line==b['text'] and line in docs[b['document']]
 for b in s['textual_source_bindings']:
  text=(R/b['source_path']).read_text('utf8')
  assert b['contains'] in (text.splitlines()[b['source_line']-1] if 'source_line' in b else text)
 for b in s['direction_bindings']:
  row=pointer(data[b['source_path']],b['pointer'])
  assert [1 if row[m]['mean']>0 else -1 if row[m]['mean']<0 else 0 for m in ['accuracy_pct','aeod','aspd']]==b['signs']
 root=next(v for p,v in data.items() if p.endswith('/ROOT_VERIFICATION.json'))
 assert root['status']=='ROOT_C50_FIVE_SCENE_THREE_VIEW_TABLES_ADOPTED'
 assert root['unique_records']==100 and root['paired_models']==50 and root['complete_scenes']==5
 assert root['test'] is False and root['primary_endpoint']=='PENDING_AUTHOR' and root['whole_rebuttal_complete'] is False
 assert s['root_adoption_sha256']=='811ad551c1398f6e57681f59046c316b05c609ef30d4897158460b8121b1a28d'
 records=next(v['records'] for p,v in data.items() if p.endswith('/snapshot/records.json'))
 assert len(records)==len({r['id'] for r in records})==100
 by={}
 for r in records:
  key=(r['variant'],r['attack'],r['seed']);assert key not in by;by[key]=r
  assert r['distribution']=='IID' and set(r['views'])=={'native','raw','shared_calibration'}
  assert r['data_contract']['evaluation_split']=='valid' and r['data_contract']['actual_evaluation_rows']==19867
  assert re.fullmatch('[0-9a-f]{64}',r['checkpoint_sha256']) and all(v['prediction_count']==19867 for v in r['views'].values())
  for key in ['accuracy','aeod','aspd','group_confusion_counts']:assert r['views']['native'][key]==r['views']['shared_calibration'][key]
  assert r['fits']['raw']['rule']=='argmax_margin_strictly_positive'
  for view in ['native','shared_calibration']:assert r['fits'][view]['fit_data']=='clean_train_root_only'
 for scene in ['Benign','F Flip','FedSA','S-DFA','Sp-DFA']:
  for seed in range(91001,91011):
   f=by['Full',scene,seed];c=by['minus_C',scene,seed]
   for key in ['root_image_ids_sha256','evaluation_image_ids_sha256','train_image_ids_sha256']:assert f['data_contract'][key]==c['data_contract'][key]
 for v in ['Full','minus_C']:
  assert dict(collections.Counter(r['replay_runtime']['device'] for r in records if r['variant']==v))==root['replay_devices'][v]
  assert dict(collections.Counter(r['training_torch'] for r in records if r['variant']==v))==root['training_torch'][v]
 coverage=next(v for p,v in data.items() if p.endswith('/snapshot/coverage.json'))
 for view,rows in coverage.items():
  assert len(rows)==10 and sum(r['complete'] for r in rows)==5
  assert all(r['distribution']=='IID' and r['n']==10 if r['complete'] else r['distribution']=='non-IID' and r['n']==0 for r in rows)
 table=next(v for p,v in data.items() if p.endswith('/snapshot/tables.json'))
 seeds={10:list(range(91001,91011)),9:list(range(91002,91011)),6:list(range(91005,91011))}
 for p in table['panels']:assert p['seeds']==seeds[len(p['seeds'])] and all(r['seeds']==p['seeds'] and r['complete'] for r in p['rows'])
 links=0
 for name,text in docs.items():
  for required in ['AUTHOR_REVIEW / DO_NOT_SUBMIT_BEFORE_FULL_COHORT','not applied','not full equalized odds','official-test exposure','final test under a frozen final protocol has not been run','six other image-control variants','historical/current driver equality was not established']:
   assert required in text,(name,required)
  for target in re.findall(r'\]\(([^)]+)\)',text):assert Path(target).is_absolute() and Path(target).is_file(),target;links+=1
 # Exactly two proposed body paragraphs; headings/status are not manuscript prose.
 body=[line for line in docs['C50_MANUSCRIPT_INSERTIONS.md'].splitlines() if line.startswith('**Individual') or line.startswith('**Interpretation')]
 assert len(body)==2
 for path,pin in s['source_pins'].items():assert sha(R/path)==pin['sha256']
 result=dict(status='PASS_C50_SHORT_WRITING_SOURCES_NUMBERS_COMMENTS_SCOPE_LINKS',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_pins=len(s['source_pins']),quoted_mean_SD_cells=len(s['quoted_mean_SD_cells']),quoted_scalar_pointers=2*len(s['quoted_mean_SD_cells']),original_comments_exact=len(s['verbatim_comment_bindings']),fact_bindings=len(s['fact_bindings']),direction_checks=len(s['direction_bindings']),links_checked=links,unique_records=100,matched_pairs=50,complete_IID_scenes=5,native_shared_equal_metrics_and_counts=100,manuscript_candidate_paragraphs=2,root_adoption_sha256=s['root_adoption_sha256'],source_pointers_sha256=sha(D/'SOURCE_POINTERS.json'),documents_sha256={n:sha(D/n) for n in docs},new_statistics=False,new_CNN=0,SSH=False,Git_or_STATE_modified=False,manuscript_applied=False,source_guard_scope='Quoted display/source/identity facts only; accepted arithmetic and archive checks referenced, not repeated.')
 if args.report:
  with args.report.open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
