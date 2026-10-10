"""Bounded C100 independent stdlib arithmetic/provenance review; no builder or inference."""
import ast, collections, datetime, hashlib, itertools, json, math, sys, traceback
from pathlib import Path
sys.dont_write_bytecode = True
assert not sys.flags.optimize
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
BASE=ROOT/'tmp/celeba_mechanism_C100_table_20261010'; SNAP=BASE/'snapshot'
OLD=ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_eight_scenes_20261010'
CHAIN=ROOT/'tmp/celeba_mechanism_remaining620_C100_root_adoption_20261010'
FULL=ROOT/'outputs/guardfed_tables/celeba_nine_method_three_view_20261009/records_three_views_900.json'
ACTUAL='7dc195a38f0062f910bb01e32708fb8446d963e01612d3262b69fba9cfd9f6cf'
HANDOFF='97419752aeab767f68fdc394a1e85724c061a688f7bd9ce6dfea6cf8e5a4afe9'
SOURCE='15dcead777b13a07896e08fe2fba434754f7099838c3d3d65ce07ba51bc544db'
SNAPSHOT='00a6a877e861391d587f360ac4c7252a51f22d00520784f52631266507a9da8b'
ROOT200='2ac1d2f200d9671de5271f24b0cbb3a0772afb88c16ae6ccf71805ed1588ee46'
INDEX200='0940513f702c42ce9451d42ba6d66cbc9ab70868c90ca2137612b8555ca5dc96'
FULL900='983bca43dff7e94dd79a312f273c65ed3ea1d146fff3ebfbf3bd2a854e124529'
VIEWS=('native','raw','shared_calibration'); METRICS=('accuracy_pct','aeod','aspd')
ATTACKS=('Benign','F Flip','FedSA','S-DFA','Sp-DFA')
SCENES=list(itertools.product(('IID','non-IID'),ATTACKS))
VARIANTS=('Full','minus_C','minus_C minus Full')
SEEDS=[list(range(91001,91011)),list(range(91002,91011)),list(range(91005,91011))]
EXPECTED=[f'minus_C_non-IID_{a}_seed{s}' for a in ('S-DFA','Sp-DFA') for s in SEEDS[0]]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
canonical=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def pinned(path,digest):
    path=Path(path);assert sha(path)==digest,str(path)
    return read(path)

def files(base,members):
    for name,pin in members.items():
        path=base/name;assert path.resolve().is_relative_to(base.resolve())
        assert sha(path)==pin['sha256'] and path.stat().st_size==pin['bytes'],name

def function(source,name):
    return next(ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)

# Reuse only the previously independently accepted pure mean/ddof1 and raw-JSON span readers.
PRIOR=ROOT/'tmp/celeba_mechanism_C70_independent_review_20261010/review.py'
assert sha(PRIOR)=='fbe27a120011d1266d23dd3bcdff93d9de0a5df746f2a571215c1ec09b138fb4'
prior_text=PRIOR.read_text('utf8')
exec(compile(ast.Module(body=[n for n in ast.parse(prior_text).body if isinstance(n,ast.FunctionDef) and n.name in ('mean_sd','raw_records')],type_ignores=[]),str(PRIOR),'exec'))

def source_review(bindings):
    pins=read(BASE/'INPUT_PINS.json')['files'];files(ROOT,pins)
    assert bindings['prepared_input_pins']==pins
    parent=read(OLD/'source/REBINDS.json');checks={}
    for kind,name in [('panels','aggregate_panels'),('verify_numeric','verify_aggregate')]:
        spec=parent[kind];p=ROOT/spec['source'];assert sha(p)==spec['sha256']
        text=p.read_text('utf8')
        for before,after,count in spec['replacements']:
            assert text.count(before)==count;text=text.replace(before,after)
        assert hashlib.sha256(text.encode()).hexdigest()==spec['effective_sha256']
        now=(BASE/(kind+'.py')).read_text('utf8')
        assert function(text,name)==function(now,name)
        checks[name]=hashlib.sha256(function(now,name).encode()).hexdigest()
    original_functions={}
    for path,names in [('tmp/celeba_mechanism_three_view_paired71_20261009/inputs.py',['receipt_identity','normalized']),('tmp/celeba_mechanism_valid_C_after70_20261010/bridge.py',['canonical']),('tmp/celeba_mechanism_three_view100_tables_20261009/build.py',['full_record'])]:
        text=(ROOT/path).read_text('utf8')
        for name in names:original_functions[name]=hashlib.sha256(function(text,name).encode()).hexdigest()
    assert original_functions==bindings['source_functions']
    return dict(input_pins_verified=len(pins),original_identity_normalizer_function_hashes=original_functions,
        original_five_scene_aggregate_and_checker_source_exact=checks,original_statistics_source_sha256=sha(ROOT/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'),
        candidate_builder_called=False,candidate_checker_called=False)

def provenance(records,bindings):
    storage_path=ROOT/'tmp/guardfed_local_storage.py';ns={'__file__':str(storage_path)}
    exec(compile(storage_path.read_bytes(),str(storage_path),'exec'),ns);volume=ns['check_bulk_storage'](0)
    root=pinned(CHAIN/'ROOT_ADOPTION.json',ROOT200);index=pinned(CHAIN/'MECHANISM200_INDEX.json',INDEX200)
    assert root['records_index_sha256']==INDEX200 and root['cumulative_accepted']==200 and root['new_accepted']==19
    assert index['all_ids']==read(ROOT/index['first181_index'])['all_ids']+EXPECTED[1:]
    assert len(index['all_ids'])==len(set(index['all_ids']))==200
    assert bindings['actual_root200_sha256']==ROOT200 and bindings['index200_sha256']==INDEX200
    assert bindings['original_C80_root_sha256']==sha(OLD/'ROOT_VERIFICATION.json')=='71105f39c6345efb3706fe538a51686f80ad9e8a89f5601a04b349ebcc76e487'
    assert bindings['new_inference']==0 and bindings['no_model_or_array_reads'] is True
    first=pinned(ROOT/index['first_adoption'],index['first_adoption_sha256'])
    first_index=pinned(ROOT/index['first181_index'],index['first181_index_sha256'])
    assert first['new_accepted']==1 and first['cumulative_accepted']==181
    assert index['first_record']==first_index['new_records'][0]
    arc=index['new_archive'];assert arc['accepted_new_ids']==EXPECTED[1:]
    assert arc['previous_receipt_sha256']==first['receipt_sha256']
    receipt=pinned(arc['receipt'],arc['receipt_sha256']);offserver=pinned(arc['offserver_verification'],arc['offserver_verification_sha256'])
    pinned(arc['previous_local_receipt'],arc['previous_receipt_sha256'])
    assert arc['receipt_sha256']==root['receipt_sha256'] and arc['offserver_verification_sha256']==root['offserver_proof_sha256']
    assert root['native_max_abs_difference']==0 and root['original180_unchanged'] and root['first181_unchanged']
    assert root['new_CNN']==root['new_training']==root['Full_inference']==0 and root['test'] is False
    native=pinned(ROOT/index['new_native_inspection'],index['new_native_inspection_sha256'])
    assert native['new_count']==200 and len(native['records'])==300
    native_by={r['id']:r for r in native['records']};byid={r['id']:r for r in records}
    saved=[index['first_record']]+index['new_records'];assert [r['id'] for r in saved]==EXPECTED
    details=[]
    for item in saved:
        identity=item['id'];bpin=index['new_binding_files'][identity]
        binding=pinned(bpin['path'],bpin['sha256']);assert binding==index['new_bindings'][identity]
        rec=binding['record'];art=index['new_artifacts'][identity]
        artifacts={k:pinned(v['path'],v['sha256']) for k,v in art.items()}
        sr=artifacts['scientific_receipt'];strict=artifacts['strict_json'];bridge=artifacts['bridge_receipt'];out=byid[identity]
        assert binding['status']=='IMMUTABLE_TERMINAL_ONCE_BOUND' and binding['id']==rec['id']==identity
        assert rec['accepted_v4_row']==native_by[identity]
        assert rec['terminal_round']==70 and rec['original_split']=='valid' and rec['original_n_eval']==19867
        assert rec['actual_alpha']==5.0 and rec['config']['client_alpha']==5.0 and rec['config']['ablation_component']=='C'
        assert rec['config_canonical_sha256']==canonical(rec['config'])
        assert sr['model_inventory_record_sha256']==canonical(rec)
        assert all(sr[k]==rec[k]==out[k] for k in ('id','method','distribution','attack','seed'))
        assert sr['config_canonical_sha256']==out['config_sha256']==rec['config_canonical_sha256']
        assert out['checkpoint_sha256']==sr['checkpoint_sha256']==strict['checkpoint_sha256']==item['checkpoint_sha256']==rec['checkpoint']['sha256']
        assert bridge['artifact_before']==bridge['artifact_after'] and bridge['source_before']==bridge['source_after']
        model_pins=[p['sha256'] for name,p in bridge['artifact_before'].items() if name.endswith('/model.pt')]
        assert model_pins==[rec['checkpoint']['sha256']]
        assert bridge['scientific_body_receipt_sha256']==art['scientific_receipt']['sha256'] and bridge['native_tolerance']==1e-12
        assert strict['status']=='MECHANISM_VALID_THREE_VIEWS_ACCEPTED' and bridge['status']=='MECHANISM_NATIVE_VALID_REPLAY_PASS'
        assert strict['views']==sr['views']==item['views']==out['views'] and out['fits']==sr['fits'] and out['replay_runtime']==sr['runtime']
        assert out['data_contract']==rec['data_contract'] and out['training_torch']==rec['training_torch']
        assert sr['original_result_sha256']==rec['result']['sha256'] and sr['original_job_sha256']==rec['raw_job']['sha256']
        assert sr['valid_n']==19867 and sr['root_reconstruction']['root_n']==16277
        assert sr['valid_image_ids_sha256']==rec['data_contract']['evaluation_image_ids_sha256']
        assert sr['root_reconstruction']['root_image_ids_sha256']==rec['data_contract']['root_image_ids_sha256']
        assert sr['weights_before']==sr['weights_after'] and not sr['optimizer_created'] and not sr['gradients_created']
        assert sr['native_comparison']['accepted'] and sr['native_comparison']['max_abs_difference']<=1e-12
        assert strict['bridge_receipt_sha256']==art['bridge_receipt']['sha256']
        assert strict['inventory_sha256']==bridge['inventory_sha256']==art['bound_inventory']['sha256']
        assert bridge['approval_sha256']==art['delegated_approval']['sha256']
        assert strict['paired_full_reference']==rec['paired_full']
        assert out['provenance']['binding']==bpin and out['provenance']['artifacts']==art and out['provenance']['saved_array_record']==item
        for view in VIEWS:assert out['fits'][view]['fit_data']==('none' if view=='raw' else 'clean_train_root_only')
        details.append(dict(id=identity,checkpoint_sha256=out['checkpoint_sha256'],scientific_receipt_sha256=art['scientific_receipt']['sha256'],native_max_abs_difference=sr['native_comparison']['max_abs_difference']))
    full900=pinned(FULL,FULL900);full_by={r['id']:r for r in full900['records']};assert len(full_by)==900
    refs={r['id']:r for r in read(ROOT/'tmp/celeba_mechanism_valid_C_after70_20261010/inventory_actual180_Full100refs.json')['full_references']}
    for r in records:
        if r['variant']=='Full':
            src=full_by[r['id']];ref=refs[r['id']]
            assert src['checkpoint_sha256']==ref['checkpoint_sha256']==r['checkpoint_sha256']
            assert src['model_inventory_record_sha256']==ref['baseline_record_canonical_sha256']
            assert src['same_checkpoint_all_views'] and not src['test_evaluation_performed']
            assert all(r[k]==src[k] for k in ('id','method','distribution','attack','seed','training_torch','views','fits'))
            assert r['config_sha256']==src['config_canonical_sha256'] and r['data_contract']==src['original_inventory_record']['data_contract'] and r['replay_runtime']==src['runtime']
            assert r['provenance']['accepted900_sha256']==FULL900 and r['provenance']['source_binding']==src['source_binding']
        else:
            nr=native_by[r['id']];assert nr['checkpoint_sha256']==r['checkpoint_sha256']
            for m in METRICS:assert abs(r['views']['native']['accuracy' if m=='accuracy_pct' else m]*(100 if m=='accuracy_pct' else 1)-nr[m])<=1e-12
    return dict(actual_root200_sha256=ROOT200,index200_sha256=INDEX200,native200_inspection_sha256=index['new_native_inspection_sha256'],new20_receipt_source_native_checkpoint_joins=details,
        all100_C_native_metric_checkpoint_joins=True,Full100_actual900_reference_source_joins=True,Full900_sha256=FULL900,
        first_receipt_sha256=arc['previous_receipt_sha256'],new19_receipt_sha256=arc['receipt_sha256'],new19_offserver_sha256=arc['offserver_verification_sha256'],
        archive_and_tensor_hash_proofs_reused_from_root_adoption=True,fresh_F_volume=volume,no_archive_extraction=True)

def run():
    assert sha(BASE/'ACTUAL_FILES_SHA256.json')==ACTUAL and sha(BASE/'ACTUAL_HANDOFF.json')==HANDOFF
    af=read(BASE/'ACTUAL_FILES_SHA256.json')['files'];assert len(af)==27;files(BASE,af)
    assert sha(BASE/'SOURCE_FILES_SHA256.json')==SOURCE and sha(SNAP/'FILES_SHA256.json')==SNAPSHOT
    sf=read(BASE/'SOURCE_FILES_SHA256.json')['files'];files(BASE,sf)
    snapfiles=read(SNAP/'FILES_SHA256.json')['files'];files(SNAP,snapfiles)
    records=read(SNAP/'records.json')['records'];tables=read(SNAP/'tables.json');bindings=read(SNAP/'SOURCE_BINDINGS.json')
    sr=source_review(bindings);prov=provenance(records,bindings)
    by={(r['variant'],r['distribution'],r['attack'],r['seed']):r for r in records}
    assert len(records)==len({r['id'] for r in records})==len(by)==200
    assert set(by)=={(v,d,a,s) for v in VARIANTS[:2] for d,a in SCENES for s in SEEDS[0]}
    assert raw_records(SNAP/'records.json')[:160]==raw_records(OLD/'snapshot/records.json')
    assert records[:160]==read(OLD/'snapshot/records.json')['records']
    count_errors=[];metric_checks=count_checks=0
    for r in records:
        assert r['data_contract']['evaluation_split']=='valid' and r['data_contract']['actual_train_rows']==162770 and r['data_contract']['actual_evaluation_rows']==19867
        assert set(r['views'])==set(r['fits'])==set(VIEWS)
        counterpart=by['minus_C' if r['variant']=='Full' else 'Full',r['distribution'],r['attack'],r['seed']]
        assert r['data_contract']==counterpart['data_contract']
        for v in r['views'].values():
            a,b=v['group_confusion_counts']['0'],v['group_confusion_counts']['1'];n=a['n']+b['n']
            assert (a['n'],a['positives'],a['negatives'],b['n'],b['positives'],b['negatives'])==(11409,6157,5252,8458,3445,5013)
            assert n==v['prediction_count']==19867
            for g in (a,b):
                assert all(type(g[k]) is int and g[k]>=0 for k in ('tp','fp','tn','fn'))
                assert g['tp']+g['fn']==g['positives'] and g['fp']+g['tn']==g['negatives'] and g['positives']+g['negatives']==g['n'];count_checks+=4
            calc=dict(accuracy=(a['tp']+a['tn']+b['tp']+b['tn'])/n,aeod=abs(a['tp']/(a['tp']+a['fn'])-b['tp']/(b['tp']+b['fn'])),aspd=abs((a['tp']+a['fp'])/a['n']-(b['tp']+b['fp'])/b['n']))
            for k,x in calc.items():count_errors.append(abs(x-v[k]));metric_checks+=1
    assert metric_checks==1800 and count_checks==4800 and max(count_errors)<=1e-12
    def value(v,d,a,s,w,m):return by[v,d,a,s]['views'][w]['accuracy' if m=='accuracy_pct' else m]*(100 if m=='accuracy_pct' else 1)
    def xs(v,scenes,seeds,w,m):
        def within(variant,s):return math.fsum(value(variant,d,a,s,w,m) for d,a in scenes)/len(scenes)
        return [within('minus_C',s)-within('Full',s) if v==VARIANTS[2] else within(v,s) for s in seeds]
    expected_panels=set(itertools.product(VIEWS,map(tuple,SEEDS)))
    def check_panels(panels,scenes,scene_rows):
        assert len(panels)==9 and {(p['view'],tuple(p['seeds'])) for p in panels}==expected_panels
        errors=[];computed=[];directions=[]
        for p in panels:
            expect={(d,a,v) for d,a in scenes for v in VARIANTS} if scene_rows else {('aggregate','aggregate',v) for v in VARIANTS}
            got={(r['distribution'],r['attack'],r['variant']) for r in p['rows']} if scene_rows else {('aggregate','aggregate',r['variant']) for r in p['rows']}
            assert got==expect and len(p['rows'])==len(expect)
            for row in p['rows']:
                assert row['complete'] and row['n']==row['expected_n']==len(p['seeds']) and row['seeds']==p['seeds']
                selected=[(row['distribution'],row['attack'])] if scene_rows else scenes;nums={}
                for m in METRICS:
                    mu,sd=mean_sd(xs(row['variant'],selected,p['seeds'],p['view'],m));nums[m]=(mu,sd)
                    errors.extend((abs(mu-row[m]['mean']),abs(sd-row[m]['sample_sd_ddof1'])))
                computed.append(nums)
                if row['variant']==VARIANTS[2]:directions.append(dict(view=p['view'],n=len(p['seeds']),distribution=row['distribution'],attack=row['attack'],means={m:nums[m][0] for m in METRICS}))
        assert max(errors)<=1e-12
        return errors,computed,directions
    panels=tables['panels'];errors,computed,directions=check_panels(panels,SCENES,True);assert len(errors)==1620
    lines=[x for x in (SNAP/'TABLES.md').read_text('utf8').splitlines() if x.startswith('| ') and ' ± ' in x];assert len(lines)==len(computed)==270
    for line,nums in zip(lines,computed):
        cells=line.strip('| ').split(' | ')
        for i,m in enumerate(METRICS,3):
            mu,sd=nums[m];prec=3 if m=='accuracy_pct' else 5;assert cells[i]==f'{mu:.{prec}f} ± {sd:.{prec}f}'
    addedscenes={('non-IID','S-DFA'),('non-IID','Sp-DFA')}
    kept=[dict(p,rows=[r for r in p['rows'] if (r['distribution'],r['attack']) not in addedscenes]) for p in panels]
    assert kept==read(OLD/'snapshot/tables.json')['panels']
    assert [x for x in lines if not any(x.startswith('| '+d+' '+a+' |') for d,a in addedscenes)]==[x for x in (OLD/'snapshot/TABLES.md').read_text('utf8').splitlines() if x.startswith('| ') and ' ± ' in x]
    paired=read(SNAP/'paired_per_seed.json');pairchecks=0
    for w in VIEWS:
        assert len(paired[w])==100 and {(p['distribution'],p['attack'],p['seed']) for p in paired[w]}=={(d,a,s) for d,a in SCENES for s in SEEDS[0]}
        for p in paired[w]:
            assert p['variant']=='minus_C'
            for m in METRICS:assert abs(p[m]-xs(VARIANTS[2],[(p['distribution'],p['attack'])],[p['seed']],w,m)[0])<=1e-12;pairchecks+=1
    assert pairchecks==900
    coverage=read(SNAP/'coverage.json')
    for w in VIEWS:
        assert len(coverage[w])==10 and {(r['distribution'],r['attack']) for r in coverage[w]}==set(SCENES)
        assert all(r['variant']=='minus_C' and r['complete'] and r['n']==r['expected_n']==10 and r['seeds']==SEEDS[0] for r in coverage[w])
    assert (SNAP/'cross_scene_seed_first.json').read_bytes()==(OLD/'snapshot/cross_scene_seed_first.json').read_bytes()
    oldagg=read(SNAP/'cross_scene_seed_first.json');addagg=read(SNAP/'cross_scene_additional.json')
    assert addagg['n_is_seed_count'] and addagg['scene_mean_before_seed_statistics'] and oldagg['new_endpoint_selected'] is False
    aggregates={};aggregate_directions={}
    for label,ps,scenes in [('IID',oldagg['panels'],[('IID',a) for a in ATTACKS]),('nonIID',addagg['nonIID_five_scene_panels'],[('non-IID',a) for a in ATTACKS]),('balanced',addagg['balanced_ten_scene_panels'],SCENES)]:
        ae,_,ad=check_panels(ps,scenes,False);assert len(ae)==162
        assert all(row['distribution']=={'IID':'IID','nonIID':'non-IID','balanced':'Balanced IID/non-IID'}[label] for p in ps for row in p['rows'])
        aggregates[label]=dict(scalars=162,max_abs_difference=max(ae),scenes_per_seed=len(scenes),n_is_seed_count=True);aggregate_directions[label]=ad
    assert (tables['unique_records'],tables['paired_models'],tables['complete_scenes'])==(200,100,10)
    assert tables['full_nonIID_coverage'] and tables['primary_endpoint_selected'] is False and tables['final_test'] is False and tables['new_inference']==0
    runtime={v:dict(collections.Counter(r['replay_runtime']['device'] for r in records if r['variant']==v)) for v in VARIANTS[:2]}
    training={v:dict(collections.Counter(r['training_torch'] for r in records if r['variant']==v)) for v in VARIANTS[:2]}
    assert runtime=={'Full':{'cpu':5,'cuda:0':95},'minus_C':{'cpu':100}}
    assert training['Full']=={'2.11.0+cu128':98,'2.11.0+cu130':2} and training['minus_C']=={'2.11.0+cu128':100}
    doc=(SNAP/'TABLES.md').read_text('utf8')+(BASE/'REPORT.md').read_text('utf8')
    for term in ['sampleSD(ddof1)','absolute TPR gap','not full equalized odds','validation exposure','official-test exposure','other six control','necessity','statistical significance']:assert term in doc,term
    negatives=[d for d in directions if (d['distribution'],d['attack']) in addedscenes]
    assert any(d['means']['accuracy_pct']>0 and d['means']['aeod']<0 for d in negatives)
    files(BASE,af);assert sha(BASE/'ACTUAL_FILES_SHA256.json')==ACTUAL
    return dict(status='INDEPENDENT_C100_TEN_SCENE_ARITHMETIC_AND_PROVENANCE_PASS_NO_ADOPTION',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_handoff_sha256=HANDOFF,actual_delivery_seal_sha256=ACTUAL,actual_delivery_members_verified=len(af),source_seal_sha256=SOURCE,source_members_verified=len(sf),snapshot_seal_sha256=SNAPSHOT,snapshot_members_verified=len(snapfiles),
        unique_records=200,paired_models=100,complete_scenes=10,mean_SD_scalars_recomputed=len(errors),display_cells=810,count_metrics_recomputed=metric_checks,confusion_count_checks=count_checks,paired_seed_metric_checks=pairchecks,
        max_abs_difference=max(errors),count_metric_max_abs_difference=max(count_errors),aggregates=aggregates,additional_seed_first_scalars_recomputed=324,total_seed_first_scalars_recomputed=486,
        old160_record_JSON_bytes_and_order_exact=True,old1296_scalars_exact=True,old648_display_cells_exact=True,old162_IID_aggregate_bytes_exact=True,
        same_checkpoint_all_views_and_same_seed_all_metrics=True,native_shared_identical_records=sum(r['views']['native']==r['views']['shared_calibration'] for r in records),
        source_review=sr,source_connections=prov,new_S_DFA_Sp_DFA_all_panel_paired_directions=negatives,seed_first_paired_directions=aggregate_directions,
        replay_devices=runtime,training_torch=training,AEOD_definition='absolute TPR gap; not full equalized odds',primary_endpoint='PENDING_AUTHOR',findings=[],candidate_adoptable=True,
        reviewer_authored_C100_table_builder=False,reviewer_authored_C100_numeric_verifier=False,reviewer_authored_reused_C10_evaluator_parent_source=True,reviewer_authored_C19_transport_packaging=True,
        auxiliary_schema_failure_preserved='ROOT_ARITHMETIC_FAILURE.json',auxiliary_schema_correction='Bridge checkpoint binding is the model.pt entry in exact artifact_before/after, not a top-level checkpoint_sha256; scientific_body_receipt_sha256 is also checked. Failure occurred before arithmetic; candidate unchanged.',
        canonical_modified=False,adoption_performed=False,STATE_modified=False,Git_used=False,new_CNN=0,new_training=0,new_Full_inference=0,test=False,
        limitations=['Seed panels10/9/6 are fixed descriptive subsets; selection seed91001, validation use and historical official-test exposure remain.',
        'Full mixed replay CPU5/GPU95 and historical training cu12898/cu1302; C CPU100, cu128100. Device/environment differences are not removed.',
        'Saved confusion counts and recorded metrics are recomputed; no prediction arrays, checkpoints or calibration fits are regenerated.',
        'Root-adopted archive/member/tensor proofs reused; only the accepted JSON/receipt/source/native joins are reread. No old archive extraction.',
        'All ten C scenes complete; six other image controls and overall revision remain incomplete. No causal necessity, significance, universal-win or final-test conclusion.',
        'Reviewer did not author C100 builder or checker; authored reused older evaluator parent and C19 transport packaging. Root independently adopted that transport.'])

if __name__=='__main__':
    try:
        output=HERE/'ROOT_ARITHMETIC_REVIEW.json';assert not output.exists();proof=run()
        with output.open('x',encoding='utf8',newline='\n') as f:json.dump(proof,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps(dict(status=proof['status'],sha256=sha(output),scalars=proof['mean_SD_scalars_recomputed'],additional=proof['additional_seed_first_scalars_recomputed'],max_abs_difference=proof['max_abs_difference'])))
    except BaseException as e:
        failure=HERE/'ROOT_ARITHMETIC_FAILURE.json';number=2
        while failure.exists():failure=HERE/f'ROOT_ARITHMETIC_FAILURE_{number}.json';number+=1
        with failure.open('x',encoding='utf8',newline='\n') as f:json.dump(dict(error=repr(e),traceback=traceback.format_exc(),candidate_modified=False,no_automatic_retry=True),f,indent=2);f.write('\n')
        raise
