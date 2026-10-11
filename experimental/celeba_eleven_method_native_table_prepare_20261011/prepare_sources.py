"""Source-only derivation. Does not import/run builders or calculate statistics."""
import ast, difflib, hashlib, json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[1]
P=R/'tmp/celeba_ten_method_native_table_20261010'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def replace_function(text,name,replacement):
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    old=ast.get_source_segment(text,node)
    return old,replacement.strip()

approval='''def approval_gate(binding, pins):
    need(binding['status']=='ROOT_REVIEWED_ACTUAL_FLGMM96_PLUS4_NATIVE_TABLE_READY'
         and binding['root_execution_authorized'] is True and binding['view']=='native'
         and binding['final_test'] is False, 'Explicit actual root binding required')
    ref=binding['flgmm96_root'];proof=pinned(location(ref['path']),ref['sha256'])
    need(proof['status']=='ROOT_FL96_LINKED_DELTA_ARCHIVE_SOURCE_CHECKPOINT_AND_ORIGINAL_STRICT_BINDING_PASS'
         and proof['accepted_total']==96 and proof['planned_new']==96 and proof['reused_separately']==4
         and proof['final_test'] is False, 'Actual96 strict/offserver root required, not running/partial')
    manifest=read(ROOT/pins['manifest_path'])
    jobs={j['id']:j for j in manifest['jobs']}
    need(len(jobs)==96 and len(proof['accepted_job_ids'])==96
         and set(proof['accepted_job_ids'])==set(jobs), 'Root96/manifest identity mismatch')
    need(binding['reused_screen_ids']==pins['reused_screen_ids'] and binding['new_expected']==96
         and binding['reused_expected']==4 and binding['new_fit']==binding['new_inference']==0, 'Fixed96+4 only')
    env=binding['environment_evidence'];pinned(location(env['path']),env['sha256'])
    need(isinstance(binding['environment_note'],str) and binding['environment_note'].strip(), 'Actual reviewed environment disclosure required')
    return manifest,proof
'''
source='''def source_inputs():
    need(not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE'), 'Optimized Python refused')
    pins=read(HERE/'INPUT_PINS.json')
    for name,expected in pins['files'].items(): need(digest(ROOT/name)==expected,'Pinned source changed: '+name)
    proof=read(OLD/'ROOT_REVIEW.json')
    need(proof['status']=='ROOT_TEN_METHOD_NATIVE1000_DESCRIPTIVE_TABLE_ADOPTED'
         and proof['records']==1000 and proof['methods']==10 and proof['final_test'] is False,'Actual original1000 required')
    need(METHODS[:-1]==pins['native_methods'], 'Original method display order changed')
    return pins
'''
join='''def join_records(old, binding, pins, manifest):
    expected={(m,d,a,s) for m in METHODS for d in DISTS for a in ATTACKS for s in SEEDS['ten']}
    need(len(old)==1000 and len({r['id'] for r in old})==1000, 'Actual original1000 required')
    jobs={j['id']:j for j in manifest['jobs']}; accepted={}; origins={}
    protocol=read(ROOT/pins['protocol_path'])
    need(isinstance(binding['native96_sources'],list) and binding['native96_sources'], 'Actual native96 source references required')
    for source in binding['native96_sources']:
        document=pinned(location(source['path']),source['sha256'])
        selected=source['record_ids'];need(len(selected)==len(set(selected)), 'Duplicate source selection')
        entries=document['records']; byid={r['id']:(i,r) for i,r in enumerate(entries)}
        need(len(byid)==len(entries) and set(selected)<=set(byid), 'Original record source IDs differ')
        for rid in selected:
            need(rid in jobs and rid not in accepted, 'Unexpected or repeated native ID')
            i,row=byid[rid]; accepted[rid]=row
            origins[rid]=dict(path=source['path'],sha256=source['sha256'],pointer='/records/'+str(i),reused_screen=False)
    need(set(accepted)==set(jobs), 'Exactly all96 original native records required')
    for j in manifest['reused_jobs']:
        need(j['id'] not in accepted,'Screen reused twice')
        accepted[j['id']]=j['accepted_record']
        origins[j['id']]=dict(path=pins['manifest_path'],sha256=pins['files'][pins['manifest_path']],
             pointer='/reused_jobs/'+str(manifest['reused_jobs'].index(j))+'/accepted_record',reused_screen=True)
        jobs[j['id']]=dict(j,seed=j['accepted_record']['seed'])
    need(len(accepted)==100, 'All100 FLGMM native records required')
    rows=list(old)  # Preserve every original1000 record object and its order.
    for d in DISTS:
        for a in ATTACKS:
            for seed in SEEDS['ten']:
                matches=[r for r in accepted.values() if (r['distribution'],r['attack'],r['seed'])==(d,a,seed)]
                need(len(matches)==1,'Missing/duplicate FLGMM cell');r=matches[0];j=jobs[r['id']]
                need(r['candidate']==pins['selected_recipe']['id'] and r['rounds']==70
                     and r['evaluation_stats']['prediction_count']==19867, 'Recipe/terminal/validation denominator differs')
                need(r['job_sha256']==j['job_sha256'] and (j['distribution'],j['attack'],j['seed'])==(d,a,seed),'Frozen job identity differs')
                need(set(r['metrics'])=={k for k,_,_,_ in METRICS}, 'Original native metric schema required; views cannot be substituted')
                for key in ('checkpoint_sha256','original_acceptance_sha256','screen_identity_sha256'):
                    need(isinstance(r[key],str) and len(r[key])==64,'Missing original checkpoint/acceptance identity')
                need(r['source_hashes']==protocol['source_hashes'] and bool(r['adapter_source_hashes']), 'Original frozen source/data identity differs')
                reference=manifest['reused_jobs'][0]['accepted_record']['adapter_source_hashes']
                for key in ('flgmm_adapter.py','sources/flgmm_pinned.py','sources/flgmm_fedavg.py'):
                    need(r['adapter_source_hashes'][key]==reference[key], 'Frozen FLGMM aggregation source differs')
                if not origins[r['id']]['reused_screen']:
                    job=pinned((ROOT/pins['manifest_path']).parent/'jobs'/j['job'],j['job_sha256']);cfg=job['config']
                    need(cfg['rounds']==70 and cfg['celeba_evaluation_split']=='valid' and cfg['seed']==seed
                         and cfg['client_alpha']=={'IID':5000.,'non-IID':5.}[d]
                         and cfg['celeba_train_limit']==cfg['celeba_eval_limit']==0,'Frozen job round/split/seed/alpha/data scope differs')
                rows.append(dict(id=r['id'],method='FLGMM',distribution=d,attack=a,seed=seed,
                    checkpoint_sha256=r['checkpoint_sha256'],metrics=r['metrics'],source_record=origins[r['id']],
                    original_native_record=r,native_definition='Original accepted FLGMM author-code native margin>0 output; no shared calibration',
                    terminal_round=70,evaluation_split='valid',test_evaluation_performed=False))
    need(rows[:1000]==old,'Original1000 objects changed')
    need(len(rows)==len({r['id'] for r in rows})==1100 and {(r['method'],r['distribution'],r['attack'],r['seed']) for r in rows}==expected,'1100 unique complete cells required')
    for r in rows:need(all(math.isfinite(r['metrics'][k]) and 0<=r['metrics'][k]<=1 for k,_,_,_ in METRICS),'Invalid metric')
    return rows
'''
run='''def run(args):
    global NOTES
    pins=source_inputs();binding=pinned(args.root_binding,args.root_binding_sha256)
    manifest,proof=approval_gate(binding,pins)
    old=read(OLD/'records_native_1000.json');need(old['view']=='native','Original view changed')
    rows=join_records(old['records'],binding,pins,manifest)
    panels,aggregates,n=summaries(rows,read(OLD/'summary_statistics.json'))
    old_aggregates=read(OLD/'seed_first_aggregates.json')
    for panel in SEEDS:need(aggregates[panel][:30]==old_aggregates[panel],'Original10 seed-first statistics changed')
    NOTES += ' FLGMM follows the author-code largest-cluster/full-model adaptation with declared zero-standard-deviation protection; native predictions are uncalibrated margin>0, not shared group thresholds. '
    NOTES += 'Its fixed recipe uses warmup20/control-width2.0/local-lr0.001 selected with seed91001 validation across IID/non-IID Benign/S-DFA. All four selected screen checkpoints are explicitly reused; the other96 are newly trained records. '
    NOTES += binding['environment_note']
    text=markdown(panels,aggregates)
    original_display=[line for line in (OLD/'TABLES.md').read_text(encoding='utf8').splitlines() if line.startswith('| ')]
    preserved_display=[line for line in text.splitlines() if line.startswith('| ') and not line.startswith('| FLGMM |')
         and not any(line.startswith('| '+panel+' | '+scope+' | FLGMM |') for panel in SEEDS for scope in ('IID','non-IID','balanced_all10'))]
    need(preserved_display==original_display,'Original table display/header/order changed')
    need(args.out.resolve().is_relative_to(HERE) and not args.out.exists(),'Fresh owned snapshot required')
    args.out.mkdir(parents=True)
    values={'records_native_1100.json':dict(records=rows,view='native',source1000_sha256=pins['old_records_sha256']),
        'summary_statistics.json':panels,'seed_first_aggregates.json':aggregates,
        'SOURCE_BINDINGS.json':dict(fixed_inputs=pins,actual_root_binding=dict(path=str(args.root_binding.resolve()),sha256=args.root_binding_sha256),
             actual_flgmm96_root=binding['flgmm96_root'],native96_sources=binding['native96_sources'],reused_screen_ids=pins['reused_screen_ids'],
             original1000_objects_exact=True,old_native_summary_metric_objects_exact=n,original_seed_first_aggregates_exact=True,original_display_lines_exact=True,
             environment_note=binding['environment_note'],environment_evidence=binding['environment_evidence'],
             models=1100,methods=11,scenarios_per_method=10,final_test=False,new_inference=0,new_fit=0,main_endpoint_selected=False,full17_complete=False)}
    for name,value in values.items():(args.out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\\n',encoding='utf8')
    (args.out/'TABLES.md').write_text(text,encoding='utf8')
    print(json.dumps(dict(status='ACTUAL_NATIVE1100_TABLE_CANDIDATE_NOT_ROOT_ADOPTED',output=str(args.out),old_exact_metric_objects=n)))
'''

old=(P/'build.py').read_text('utf8');changes=[];new=old
def change(before,after):
    global new
    assert before!=after and before in new
    new=new.replace(before,after);changes.append({'before':before,'after':after})
change('requires actual root adoption of LoGoFair100','requires actual root adoption of FLGMM96 plus four explicit screen records')
change("OLD = ROOT / 'outputs/guardfed_tables/celeba_nine_method_three_view_20261009'","OLD = ROOT / 'outputs/guardfed_tables/celeba_ten_method_native_20261010'")
change("'GuardFed-AD2+', 'LoGoFair-DP']","'GuardFed-AD2+', 'LoGoFair-DP', 'FLGMM']")
change("LOGO_RECORDS_SHA = 'fdc7c4f2402e26fdaa7b34bbfa792ceccafc5e3d32759d77940def2ed1fbc98d'","# Actual FLGMM evidence is late-bound; no future hash is predicted.")
for name,replacement in [('approval_gate',approval),('source_inputs',source),('join_records',join),('run',run)]:change(*replace_function(new,name,replacement))
change('panel_rows[:90]', 'panel_rows[:100]')
change('Original nine-method native statistics/IDs/display changed','Original ten-method native statistics/IDs/display changed')
change('old_checks += 90*3','old_checks += 100*3')
change('ten-method author-review candidate','eleven-method author-review candidate')
change('ten methods are not the full17','eleven methods are not the full17')
change("p.add_argument('--logofair-root',type=Path,required=True);p.add_argument('--logofair-root-sha256',required=True)","p.add_argument('--root-binding',type=Path,required=True);p.add_argument('--root-binding-sha256',required=True)")

all_changes={'build.py':changes};derived={'build.py':new}
vold=(P/'verify_numeric.py').read_text('utf8');vnew=vold;vc=[]
for a,b in [('records_native_1000','records_native_1100'),('==1000','==1100'),('==100 and len(aggregates[panel])==30','==110 and len(aggregates[panel])==33'),('(1800,900,540)','(1980,990,594)'),('NATIVE1000','NATIVE1100'),('records=1000','records=1100')]:
    assert a in vnew;vnew=vnew.replace(a,b);vc.append({'before':a,'after':b})
all_changes['verify_numeric.py']=vc;derived['verify_numeric.py']=vnew
for name,text in derived.items():
    original=(P/name).read_bytes();restored=text.encode()
    for c in reversed(all_changes[name]):restored=restored.replace(c['after'].encode(),c['before'].encode())
    assert restored==original,name
    compile(text,str(H/name),'exec')
    (H/name).write_text(text,encoding='utf8',newline='\n')
    (H/(name+'.diff')).write_text(''.join(difflib.unified_diff(original.decode().splitlines(True),text.splitlines(True),fromfile=str(P/name),tofile=str(H/name))),encoding='utf8')
(H/'SOURCE_INVERSE.json').write_text(json.dumps({'parents':{n:{'path':str((P/n).relative_to(R)),'sha256':sha(P/n)} for n in derived},'changes':all_changes,'complete_raw_byte_inverse_exact':True},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def fn(text,name):return ast.get_source_segment(text,next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name))
assert fn(old,'original_values_for')==fn(new,'original_values_for')
summ=fn(new,'summaries').replace('panel_rows[:100]','panel_rows[:90]').replace('Original ten-method native statistics/IDs/display changed','Original nine-method native statistics/IDs/display changed').replace('old_checks += 100*3','old_checks += 90*3')
assert summ==fn(old,'summaries')
assert fn(vold,'calculate')==fn(vnew,'calculate')
assert fn(old,'markdown').replace('ten-method author-review candidate','eleven-method author-review candidate')==fn(new,'markdown')
(H/'SOURCE_CHECK.json').write_text(json.dumps({'status':'SOURCE_ONLY_PREPARED_NOT_EXECUTED','complete_inverse_exact':True,'compile_only':True,'original_values_for_loader_byte_exact':True,'summaries_only_old_prefix_cardinality_metadata_changed':True,'mean_sampleSD_and_seed_first_loops_unchanged':True,'independent_fsum_calculate_byte_exact':True,'numeric_tolerance_unchanged':'1e-12','render_loop_only_title_changed':True,'original1000_metadata_hashes_checked':True,'future_root_binding_null':True,'expected_native_records':1100,'expected_scene_scalars':1980,'expected_scene_display_cells':990,'expected_aggregate_scalars':594,'statistics_executed':False,'SSH':False,'F_access':False,'existing_outputs_changed':False},indent=2)+'\n')
print('Derived builder/verifier; complete byte inverse and original scientific loops checked; no statistics executed.')
