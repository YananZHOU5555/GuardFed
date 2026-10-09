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
assert (D/'ACTUAL_INPUTS.json').is_file(), 'Prepared only: actual canonical C40 inputs have not been bound'
actual_inputs=read(D/'ACTUAL_INPUTS.json')
assert actual_inputs['status']=='ACTUAL_CANONICAL_C40_ROOT_TABLE_BOUND'
s=read(D/'SOURCE_POINTERS.json');data={}
for path,pin in s['source_pins'].items():
 p=R/path;assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],('source pin',path)
 if p.suffix=='.json':data[path]=read(p)
docs={p.name:p.read_text('utf8') for p in [D/'C40_REVIEWER_ADDENDUM.md',D/'C40_MANUSCRIPT_INSERTIONS.md']}
scalarchecks=0;cellchecks=0;factchecks=0
for b in s['quoted_scalar_bindings']:
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
table=next(d for p,d in data.items() if p.endswith('/snapshot/tables.json'))
root=next(d for p,d in data.items() if p.endswith('/ROOT_VERIFICATION.json'))
records=next(d['records'] for p,d in data.items() if p.endswith('/snapshot/records.json'))
coverage=data[s['derived_scope_checks']['coverage_source']]
excluded=data[s['derived_scope_checks']['excluded_source']]['records']
assert root['status']=='ROOT_C40_THREE_SCENE_THREE_VIEW_TABLES_ADOPTED'
assert sha(R/next(p for p in data if p.endswith('/ROOT_VERIFICATION.json')))==s['root_adoption_sha256']
assert root['mean_SD_scalars_recomputed']==486 and root['display_cells']==243 and root['count_metrics_recomputed']==540
assert root['original_two_scene40_records_exact'] and root['original_two_scene324_statistics_exact'] and root['original_two_scene162_cells_preserved']
assert root['unique_records']==len(records)==60 and root['paired_models']==30
assert root['test'] is False and root['primary_endpoint']=='PENDING_AUTHOR' and root['whole_rebuttal_complete'] is False
assert table['new_inference']==table['new_training']==0
assert len({r['id'] for r in records})==60
by={}
for r in records:
 key=(r['variant'],r['distribution'],r['attack'],r['seed']);assert key not in by;by[key]=r
 assert r['distribution']=='IID' and r['attack'] in ['Benign','F Flip','FedSA']
 assert set(r['views'])=={'native','raw','shared_calibration'}
 assert re.fullmatch('[0-9a-f]{64}',r['checkpoint_sha256'])
 assert r['data_contract']['evaluation_split']=='valid' and r['data_contract']['actual_evaluation_rows']==19867
 for v in r['views'].values():assert v['prediction_count']==19867
 for k in ['accuracy','aeod','aspd','group_confusion_counts']:assert r['views']['native'][k]==r['views']['shared_calibration'][k]
for scene in ['Benign','F Flip','FedSA']:
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
 assert len(rows)==10 and sum(r['complete'] for r in rows)==3
 assert {(r['distribution'],r['attack']) for r in rows if r['complete']}=={('IID','Benign'),('IID','F Flip'),('IID','FedSA')}
 assert sum(not r['complete'] for r in rows)==7
assert [r['id'] for r in excluded]==s['derived_scope_checks']['excluded_IDs']
assert not {r['id'] for r in excluded}&{r['id'] for r in records}
def paired(view,n,scene):
 panel=next(p for p in table['panels'] if p['view']==view and len(p['seeds'])==n)
 return next(r for r in panel['rows'] if r['attack']==scene and r['variant']=='minus_C minus Full')
for b in s['negative_direction_checks']:
 row=paired(b['view'],b['n'],b['scene'])
 if 'signs' in b:
  assert [1 if row[m]['mean']>0 else -1 if row[m]['mean']<0 else 0 for m in ['accuracy_pct','aeod','aspd']]==b['signs']
 else:assert (1 if row[b['metric']]['mean']>0 else -1)==b['sign']
links=[]
for name,text in docs.items():
 assert 'AUTHOR_REVIEW / DO_NOT_SUBMIT' in text and 'not applied' in text
 for required in ['pending author decisions' if name=='C40_REVIEWER_ADDENDUM.md' else 'remain author decisions','not full equalized odds','selection seed 91001','seven C','six other image-control variants']:
  assert required in text,('boundary text',name,required)
 for target in re.findall(r'\]\(([^)]+)\)',text):
  p=Path(target);assert p.is_absolute() and p.exists(),('link',name,target)
  links.append(dict(document=name,target=target))
for path,pin in s['source_pins'].items():assert sha(R/path)==pin['sha256']
report=dict(status='PASS_C40_AUTHOR_REVIEW_WRITING_NUMERIC_POINTERS_SCOPE_AND_LINKS',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_pins_verified=len(s['source_pins']),scalar_pointer_entries_checked=scalarchecks,display_mean_SD_cells_checked=cellchecks,scope_fact_bindings_checked=factchecks,verbatim_original_comment_excerpts_exact=2,old_complete_response_source_sha_unchanged=True,old_manuscript_insertions_source_sha_unchanged=True,links_checked=len(links),unique_records_scope_checked=60,paired_scene_seed_identities_checked=30,native_shared_equal_records=60,actual_devices_from_records=devices,actual_training_from_records=training,fixed_seed_panels=[10,9,6],excluded_S_DFA_preserved=6,other_C_scenes_incomplete=7,negative_direction_checks_passed=len(s['negative_direction_checks']),accepted_full_arithmetic_root_sha256=s['root_adoption_sha256'],source_pointers_sha256=sha(D/'SOURCE_POINTERS.json'),checker_sha256=sha(Path(__file__)),document_sha256={n:sha(D/n) for n in docs},scientific_audit_reexecuted=False,new_statistics_or_selection=False,new_CNN=0,SSH=False,STATE_or_Git_modified=False,manuscript_applied=False,author_review_only=True,scope='Local quoted display/source/scope check only. The previously accepted independent486-scalar audit is referenced, not rerun.')
out=D/'CHECK_RESULTS.json';assert not out.exists()
with out.open('x',encoding='utf8',newline='\n') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(report,ensure_ascii=False))
