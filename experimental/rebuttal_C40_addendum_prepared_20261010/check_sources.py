"""Check only C40 author-review display/source bindings; no inference or new analysis."""
from pathlib import Path
import collections,datetime,hashlib,json,re,sys
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent;R=D.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def pointer(value,path):
 for part in path.lstrip('/').split('/') if path else []:
  part=part.replace('~1','/').replace('~0','~')
  value=value[int(part)] if isinstance(value,list) else value[part]
 return value
actual_inputs=read(D/'ACTUAL_INPUTS.json');assert actual_inputs['status']=='ACTUAL_CANONICAL_C40_ROOT_TABLE_BOUND'
s=read(D/'SOURCE_POINTERS.json');assert s['status']=='ACTUAL_C40_WRITING_SOURCE_POINTERS';data={}
for path,pin in s['source_pins'].items():
 p=R/path;assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],('source pin',path)
 if p.suffix=='.json':data[path]=read(p)
docs={p.name:p.read_text('utf8') for p in [D/'C40_REVIEWER_ADDENDUM.md',D/'C40_MANUSCRIPT_INSERTIONS.md']}
scalarchecks=0;cellchecks=0;factchecks=0
for b in s['quoted_scalar_bindings']:
 assert b['metric'] in ['accuracy_pct','aeod','aspd'] and b['statistic'] in ['mean','sample_sd_ddof1']
 assert b['decimal_places']==(3 if b['metric']=='accuracy_pct' else 5)
 actual=pointer(data[b['source_path']],b['pointer'])
 assert type(actual)==type(b['value']) and actual==b['value'],b
 for line in b['document_lines']:assert b['paired_cell'] in docs[b['document']].splitlines()[line-1]
 scalarchecks+=1
mapped=set()
for b in s['quoted_mean_SD_cells']:
 actual=pointer(data[b['source_path']],b['base_pointer']);d=b['decimal_places']
 display=f"{actual['mean']:+.{d}f} ± {actual['sample_sd_ddof1']:.{d}f}"
 assert display==b['display']
 for line in b['document_lines']:
  assert display in docs[b['document']].splitlines()[line-1]
  mapped.add((b['document'],line,display))
 cellchecks+=1
unmapped=[]
for name,text in docs.items():
 for i,line in enumerate(text.splitlines(),1):
  for cell in re.findall(r'[+\-]?\d+(?:\.\d+)? ± \d+(?:\.\d+)?',line):
   if (name,i,cell) not in mapped:unmapped.append((name,i,cell))
assert not unmapped,unmapped
for b in s['fact_bindings']:
 actual=pointer(data[b['source_path']],b['pointer'])
 assert type(actual)==type(b['value']) and actual==b['value'],b
 for name,phrase in b['document_phrases'].items():assert phrase in docs[name],('fact prose',b['label'],name,phrase)
 factchecks+=1
for b in s['verbatim_comment_bindings']:
 line=(R/b['source_path']).read_text('utf-8-sig').splitlines()[b['source_line']-1]
 assert line==b['text'] and line in docs[b['document']]
for b in s['textual_source_bindings']:
 source=(R/b['source_path']).read_text('utf-8-sig').splitlines()
 for line,excerpt in b['source_excerpts'].items():assert source[int(line)-1]==excerpt
assert actual_inputs['root_sha256']==s['root_adoption_sha256']
for path,pin in actual_inputs['source_pins'].items():assert sha(R/path)==pin['sha256'] and (R/path).stat().st_size==pin['bytes']
for name,h in actual_inputs['prepared_files_unchanged'].items():assert sha(D/name)==h
table=next(d for p,d in data.items() if p.endswith('/snapshot/tables.json'))
root=next(d for p,d in data.items() if p.endswith('/ROOT_VERIFICATION.json'))
records=next(d['records'] for p,d in data.items() if p.endswith('/snapshot/records.json'))
coverage=data[s['derived_scope_checks']['coverage_source']]
assert root['status']=='ROOT_C40_FOUR_SCENE_THREE_VIEW_TABLES_ADOPTED'
assert sha(R/next(p for p in data if p.endswith('/ROOT_VERIFICATION.json')))==s['root_adoption_sha256']
assert root['mean_SD_scalars_recomputed']==648 and root['display_cells']==324 and root['count_metrics_recomputed']==720
assert root['original_three_scene60_records_exact'] and root['original_three_scene486_statistics_exact'] and root['original_three_scene243_cells_preserved'] and root['prior_S_DFA_six_original_record_bytes_preserved']
assert root['unique_records']==len(records)==80 and root['paired_models']==40
assert root['test'] is False and root['primary_endpoint']=='PENDING_AUTHOR' and root['whole_rebuttal_complete'] is False
assert actual_inputs['tables_sha256']==root['tables_sha256'] and actual_inputs['independent_arithmetic_review_sha256']==root['independent_review_sha256']
assert table['new_inference']==table['new_training']==0
assert len({r['id'] for r in records})==80
by={}
for r in records:
 key=(r['variant'],r['distribution'],r['attack'],r['seed']);assert key not in by;by[key]=r
 assert r['distribution']=='IID' and r['attack'] in ['Benign','F Flip','FedSA','S-DFA']
 assert set(r['views'])=={'native','raw','shared_calibration'}
 assert re.fullmatch('[0-9a-f]{64}',r['checkpoint_sha256'])
 assert r['data_contract']['evaluation_split']=='valid' and r['data_contract']['actual_evaluation_rows']==19867
 for v in r['views'].values():assert v['prediction_count']==19867
 for k in ['accuracy','aeod','aspd','group_confusion_counts']:assert r['views']['native'][k]==r['views']['shared_calibration'][k]
for scene in ['Benign','F Flip','FedSA','S-DFA']:
 for seed in range(91001,91011):
  full=by['Full','IID',scene,seed];control=by['minus_C','IID',scene,seed]
  assert full['data_contract']['evaluation_image_ids_sha256']==control['data_contract']['evaluation_image_ids_sha256']
  assert full['data_contract']['root_image_ids_sha256']==control['data_contract']['root_image_ids_sha256']
devices={variant:dict(collections.Counter(r['replay_runtime']['device'] for r in records if r['variant']==variant)) for variant in ['Full','minus_C']}
training={variant:dict(collections.Counter(r['training_torch'] for r in records if r['variant']==variant)) for variant in ['Full','minus_C']}
assert devices==root['replay_devices']==table['replay_devices'] and training==root['training_torch']==table['training_torch']
seeds={10:list(range(91001,91011)),9:list(range(91002,91011)),6:list(range(91005,91011))}
for p in table['panels']:
 assert p['seeds']==seeds[len(p['seeds'])]
 for row in p['rows']:assert row['seeds']==p['seeds'] and row['n']==row['expected_n']==len(p['seeds']) and row['complete']
for view,rows in coverage.items():
 assert len(rows)==10 and sum(r['complete'] for r in rows)==4
 assert {(r['distribution'],r['attack']) for r in rows if r['complete']}=={('IID','Benign'),('IID','F Flip'),('IID','FedSA'),('IID','S-DFA')}
 assert sum(not r['complete'] for r in rows)==6
assert s['derived_scope_checks']['complete_scene_cells']==4 and s['derived_scope_checks']['remaining_scene_cells']==6
assert s['derived_scope_checks']['planned_image_control_variants']==8 and s['derived_scope_checks']['other_incomplete_variants']==6
def paired(view,n,scene):
 panel=next(p for p in table['panels'] if p['view']==view and len(p['seeds'])==n)
 return next(r for r in panel['rows'] if r['attack']==scene and r['variant']=='minus_C minus Full')
for b in s['negative_direction_checks']:
 row=paired(b['view'],b['n'],b['scene'])
 if 'signs' in b:
  assert [1 if row[m]['mean']>0 else -1 if row[m]['mean']<0 else 0 for m in ['accuracy_pct','aeod','aspd']]==b['signs']
 else:assert (1 if row[b['metric']]['mean']>0 else -1)==b['sign']
original_response=next((R/p).read_text('utf-8-sig') for p in s['source_pins'] if p.endswith('/rebuttal_integrated_20261009.md'))
assert 'one of the eight image controls' in original_response and 'six other image controls' in original_response
links=[]
for name,text in docs.items():
 assert 'AUTHOR_REVIEW / DO_NOT_SUBMIT' in text and 'not applied' in text
 assert 'ACC is percent and ΔACC is in percentage points (pp)' in text and 'AEOD/ASPD are on [0,1]' in text
 assert 'final test under a frozen final protocol has not been run' in text
 for required in ['pending author decisions' if name=='C40_REVIEWER_ADDENDUM.md' else 'remain author decisions','not full equalized odds','selection seed 91001','six other C scenes','six other image-control variants']:
  assert required in text,('boundary text',name,required)
 for target in re.findall(r'\]\(([^)]+)\)',text):
  p=Path(target);assert p.is_absolute() and p.exists(),('link',name,target)
  links.append(dict(document=name,target=target))
for path,pin in s['source_pins'].items():assert sha(R/path)==pin['sha256']
report=dict(status='PASS_C40_AUTHOR_REVIEW_WRITING_NUMERIC_POINTERS_SCOPE_AND_LINKS',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_pins_verified=len(s['source_pins']),scalar_pointer_entries_checked=scalarchecks,display_mean_SD_cells_checked=cellchecks,scope_fact_bindings_checked=factchecks,verbatim_original_comment_excerpts_exact=2,old_complete_response_source_sha_unchanged=True,old_manuscript_insertions_source_sha_unchanged=True,links_checked=len(links),unique_records_scope_checked=80,paired_scene_seed_identities_checked=40,native_shared_equal_records=80,actual_devices_from_records=devices,actual_training_from_records=training,fixed_seed_panels=[10,9,6],excluded_partial_C_records=0,previously_partial_S_DFA_six_now_included=6,other_C_scenes_incomplete=6,negative_direction_checks_passed=len(s['negative_direction_checks']),accepted_full_arithmetic_root_sha256=s['root_adoption_sha256'],source_pointers_sha256=sha(D/'SOURCE_POINTERS.json'),checker_sha256=sha(Path(__file__)),document_sha256={n:sha(D/n) for n in docs},scientific_audit_reexecuted=False,new_statistics_or_selection=False,new_CNN=0,SSH=False,STATE_or_Git_modified=False,manuscript_applied=False,author_review_only=True,scope='Local quoted display/source/scope check only. The previously accepted independent648-scalar audit is referenced, not rerun.')
out=D/'CHECK_RESULTS.json';assert not out.exists()
with out.open('x',encoding='utf8',newline='\n') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(report,ensure_ascii=False))
