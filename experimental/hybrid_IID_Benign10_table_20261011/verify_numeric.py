"""Independent fsum/ddof1 and original count-loop verification; no arrays/fit."""
from pathlib import Path
import ast,hashlib,json,math
H=Path(__file__).resolve().parent;R=H.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert __debug__ and not (H/'VERIFICATION.json').exists()
 records=json.loads((H/'records.json').read_bytes())['records'];tables=json.loads((H/'tables.json').read_bytes());panels=tables['panels']
 assert len(records)==len({r['id'] for r in records})==10 and [r['seed'] for r in records]==list(range(91001,91011))
 assert sum(r['selection_seed'] for r in records)==1 and records[0]['phase']=='screen' and all(r['phase']=='fullcoverage' for r in records[1:])
 byseed={r['seed']:r for r in records};errors=[];strings=[]
 assert len(panels)==9 and len({(p['view'],tuple(p['seeds'])) for p in panels})==9
 for p in panels:
  assert p['view'] in ['native','raw','shared_calibration'] and p['seeds'] in [list(range(91001,91011)),list(range(91002,91011)),list(range(91005,91011))]
  assert p['ids']==[byseed[s]['id'] for s in p['seeds']] and p['checkpoint_sha256']==[byseed[s]['checkpoint_sha256'] for s in p['seeds']]
  assert len(p['rows'])==1;row=p['rows'][0];assert row['n']==row['expected_n']==len(p['seeds']) and row['complete'] and row['seeds']==p['seeds']
  for metric in ['accuracy_pct','aeod','aspd']:
   xs=[100*byseed[s]['views'][p['view']]['accuracy'] if metric=='accuracy_pct' else byseed[s]['views'][p['view']][metric] for s in p['seeds']]
   mean=math.fsum(xs)/len(xs);sd=math.sqrt(math.fsum((x-mean)**2 for x in xs)/(len(xs)-1))
   errors.extend([abs(mean-row[metric]['mean']),abs(sd-row[metric]['sample_sd_ddof1'])]);value=f"{row[metric]['mean']:.{3 if metric=='accuracy_pct' else 5}f} ± {row[metric]['sample_sd_ddof1']:.{3 if metric=='accuracy_pct' else 5}f}";assert value==row['display'][metric];strings.append(value)
 assert len(errors)==54 and max(errors)<=1e-12
 lines=[l for l in (H/'TABLES.md').read_text(encoding='utf8').splitlines() if l.startswith('| ')][1:];assert len(lines)==9
 for p,line in zip(panels,lines):
  cols=[x.strip() for x in line.strip('|').split('|')];row=p['rows'][0];assert cols==[p['view'],p['label'],str(row['n']),*[row['display'][m] for m in ['accuracy_pct','aeod','aspd']]]
 # Reuse the original F20 independent per-record group-count arithmetic AST.
 source=R/'tmp/celeba_F_IID_two_scenes20_table_20261011/verify_numeric.py';assert sha(source)=='30a1d9df0387e398566b0e676763fc8896b090d98839be58ff1457e57acdf7ca'
 original=ast.parse(source.read_bytes());verify=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='verify');loop=next(n for n in verify.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='record')
 ns={'records':records,'metric_checks':0,'count_checks':0};exec(compile(ast.Module(body=[loop],type_ignores=[]),str(source),'exec'),ns)
 assert ns['metric_checks']==90 and ns['count_checks']==240
 assert all(r['weights_before']==r['weights_after'] and r['valid_n']==19867 and r['native_comparison']['max_abs_difference']<=1e-12 for r in records)
 old=json.loads((R/'outputs/guardfed_tables/celeba_hybrid_IID_Benign10_20261011/records.json').read_bytes())['records'];assert all(a['id']==b['id'] and a['checkpoint_sha256']==b['checkpoint_sha256'] and {m:a['views']['native'][m] for m in ['accuracy','aeod','aspd']}==b['metrics'] for a,b in zip(records,old))
 result={'status':'HYBRID_BENIGN10_SAVED_RECEIPT_TABLE_ARITHMETIC_PASS_NOT_ROOT_TABLE_ADOPTION','unique_checkpoints':10,'same_checkpoint_three_views':True,'seed_panels':[10,9,6],'mean_SD_scalars_recomputed':54,'display_cells_checked':27,'max_abs_difference':max(errors),'metric_values_recomputed_from_group_counts':ns['metric_checks'],'base_count_checks':ns['count_checks'],'original_count_loop_AST_unmodified':True,'original_count_checker_source_sha256':sha(source),'ddof':1,'old_native_30_checkpoint_metrics_exact':True,'original_screen_seed91001_preserved':True,'no_result_filtering':True,'fit_CNN_or_array_operations':0,'final_test':False,'source_files_sha256':{p.name:sha(p) for p in [H/'build.py',H/'verify_numeric.py']}}
 with (H/'VERIFICATION.json').open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
