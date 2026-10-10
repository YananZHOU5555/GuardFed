"""Independent fixed-recipe LoGoFair100 provenance/hash/arithmetic review; never fit."""
from pathlib import Path
import argparse, datetime, hashlib, json, math, os, sys, traceback

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[1]
SOURCE = ROOT/'tmp/celeba_logofair_fullcoverage_20261010'
SUMMARY_SOURCE = ROOT/'tmp/celeba_logofair100_summary_20261010'
SCREEN = ROOT/'tmp/celeba_logofair_screen32_20261010'
ADOPTION = ROOT/'tmp/celeba_logofair32_root_adoption_20261010/ROOT_ADOPTION.json'
SCENES = [(d,a) for d in ('IID','non-IID') for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')]
SEEDS = list(range(91001,91011))
PANELS = {'ten_seed':SEEDS,'exclude_selection':SEEDS[1:],'matching_six':SEEDS[4:]}
METRICS = ('accuracy_pct','aeod','aspd')
PINS = {
    'summary_source':'af28607a600ce4ccb03d2ef88bc76e5b83bf3c0415040344950f2f684716a1a1',
    'source':'89f3e4b4fd02352fa3c0f5080ce00ab6854eaabb4e3803a2635a1d3ba51196c3',
    'inventory':'21c534337bc35e4a53b904bc103431acc4d2672bc54f3f8f23c87eb79e743d0f',
    'screen32_adoption':'145f30628270b457d84d47aeab60825c9d4fdb23a0ae27fcd44a35a19e62ae90',
    'index':'1d2cbb94e47716dd21854c704f5d78506cf6f075774d38e06d89ec71d3be8ba7',
    'ACCEPTANCE100.json':'7f0a509c715d3f762f4184bc4f0d82d3e99ef497f11d74e04b4fb2cb8c36f8e3',
    'records100.json':'fdc7c4f2402e26fdaa7b34bbfa792ceccafc5e3d32759d77940def2ed1fbc98d',
    'SUMMARY100.json':'e09cf1139f63661528bfaf7ee730ec558dec72107982bbee07d74d654955acb1',
    'TABLES.md':'1cd87594eb25f953581a4153a64b4888e4e10c332f972988667810ac3c1c91a8',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_bytes())


def pinned(path, digest):
    assert isinstance(digest,str) and len(digest)==64 and sha(path)==digest, str(path)
    return read(path)


def sealed(directory, digest):
    seal=pinned(directory/'FILES_SHA256.json',digest)
    for name,pin in seal['files'].items():
        path=(directory/name).resolve()
        assert path.is_relative_to(directory.resolve()), 'source seal path escape'
        wanted=pin if isinstance(pin,str) else pin['sha256']
        assert sha(path)==wanted, name
        if isinstance(pin,dict): assert path.stat().st_size==pin['bytes'], name


def fpath(path):
    p=Path(path).resolve()
    assert p.drive.upper()=='F:', 'Existing bulk must remain on F; no fallback'
    return p


def statistics_independent(xs):
    assert len(xs)>1 and all(math.isfinite(x) for x in xs)
    mean=math.fsum(xs)/len(xs)
    sd=math.sqrt(math.fsum((x-mean)**2 for x in xs)/(len(xs)-1))
    return mean,sd


def review(args):
    sealed(OWN,args.source_seal)
    folder,stage,index_path,inputs_path=map(fpath,(args.summary_dir,args.stage,args.index,args.inputs))
    for name in ('ACCEPTANCE100.json','records100.json','SUMMARY100.json','TABLES.md'):
        assert sha(folder/name)==PINS[name], name
    a=read(folder/'ACCEPTANCE100.json'); records=read(folder/'records100.json'); summary=read(folder/'SUMMARY100.json')
    assert a['status']=='ORIGINAL_STRICT100_AND_SAVED_PREDICTION_SUMMARY_PASS_ROOT_REVIEW_PENDING'
    assert (a['accepted_n'],a['original_new96'],a['original_reused4'],a['root_adopted'])==(100,96,4,0)
    assert a['new_fits']==a['new_CNN']==0 and a['final_test'] is False and a['tolerance']==1e-12
    assert a['saved_prediction_items']==1986700 and a['saved_metric_checks']==300 and a['saved_metric_max_difference']<=1e-12
    assert a['source_seal_sha256']==PINS['summary_source'] and a['screen32_root_adoption_sha256']==PINS['screen32_adoption']
    sealed(SUMMARY_SOURCE,PINS['summary_source']); sealed(SOURCE,PINS['source'])
    for name,digest in read(SUMMARY_SOURCE/'INPUT_PINS.json').items(): assert sha(ROOT/name)==digest, name
    inventory=pinned(SOURCE/'CACHE_IDENTITIES100.json',PINS['inventory'])['references']
    adoption=pinned(ADOPTION,PINS['screen32_adoption']); candidate=adoption['selected_recipe']
    assert adoption['status']=='ROOT_LOGOFAIR_SCREEN32_ADOPTED' and adoption['accepted_count']==32 and adoption['virtual_cohorts']==20
    assert candidate['id']=='LoGoFair-DP_07' and candidate['settings']['post_rounds']==30
    assert sha(adoption['summary_path'])==adoption['summary_sha256'] and sha(adoption['strict_index_path'])==adoption['strict_index_sha256']
    old_index=read(adoption['strict_index_path'])
    manifest=pinned(stage/'manifest.json',a['manifest_sha256']); stage_seal=pinned(stage/'SOURCE_SHA256.json',a['stage_source_sha256'])
    index=pinned(index_path,PINS['index']); inputs=pinned(inputs_path,a['root_inputs_sha256'])
    assert a['index_sha256']==PINS['index']==sha(index_path) and index['manifest_sha256']==a['manifest_sha256']
    assert not (index_path.parent/'QUEUE_FAILURE.json').exists() and not (folder/'SUMMARY_FAILURE.json').exists()
    assert manifest['candidate']==candidate and manifest['adoption_sha256']==PINS['screen32_adoption']
    assert manifest['summary_sha256']==adoption['summary_sha256'] and manifest['inputs_sha256']==a['root_inputs_sha256']
    assert manifest['scientific_stage']=='fixed_recipe_validation_postprocessing100' and not manifest['final_test'] and manifest['new_CNN']==0
    assert inputs['status']=='ROOT_ACCEPTED_EXISTING_FEDAVG100_AND_FIXED_COHORT_MAPPINGS' and inputs['cache_identity_sha256']==PINS['inventory']
    approval=pinned(inputs_path.parent/'BIND_APPROVAL.json',manifest['bind_approval_sha256'])
    assert approval['summary_adoption_sha256']==PINS['screen32_adoption'] and approval['inputs_sha256']==a['root_inputs_sha256']
    assert approval['prepared_source_seal_sha256']==PINS['source'] and (approval['new_fits'],approval['reused'],approval['test'])==(96,4,False)
    expected={candidate['id']+'_'+r['cell_id']:r for r in inventory}
    assert len(expected)==len(inventory)==100 and {(r['distribution'],r['attack'],r['seed']) for r in inventory}=={(d,t,s) for d,t in SCENES for s in SEEDS}
    old_ids={i for i,r in expected.items() if r['seed']==91001 and r['attack'] in ('Benign','S-DFA')}
    jobs,reused=manifest['jobs'],manifest['reused_jobs']
    assert len(jobs)==96 and len(reused)==len(old_ids)==4 and not {r['id'] for r in jobs}&old_ids
    assert {r['id'] for r in jobs}|{r['id'] for r in reused}==set(expected) and {r['id'] for r in reused}==old_ids
    assert index['status']=='LOCAL_STRICT96_PLUS4_ROOT_REVIEW_PENDING' and index['reused_jobs']==reused
    assert [r['id'] for r in index['records']]==[r['id'] for r in jobs]
    assert [r['id'] for r in records]==[r['id'] for r in jobs+reused] and len(records)==100
    assert len({r['id'] for r in records})==100 and {(r['distribution'],r['attack'],r['seed']) for r in records}=={(d,t,s) for d,t in SCENES for s in SEEDS}
    locations={r['id']:r for r in inputs['references']}; rows={r['id']:r for r in index['records']+reused}
    assert len(locations)==100 and set(locations)=={r['id'] for r in inventory}
    old_jobs={r['id']:r for r in read(SCREEN/'jobs/manifest.json')['jobs']}
    assert set(inputs['mappings'])=={str(s) for s in SEEDS}
    artifacts={}
    def add(path,digest):
        key=str(Path(path).resolve())
        assert key not in artifacts or artifacts[key]==digest, 'Conflicting artifact aliases'
        artifacts[key]=digest
    for name,digest in stage_seal.items():
        path=(stage/name).resolve(); assert path.is_relative_to(stage), 'Stage path escape'
        add(path,digest)
    for r,entry in zip(records,jobs+reused):
        identity=r['id']; row=expected[identity]; pin=rows[identity]; is_old=identity in old_ids
        assert (r['baseline_id'],r['distribution'],r['attack'],r['seed'],r['fit_seed'],r['reused_screen_record'])==(row['id'],row['distribution'],row['attack'],row['seed'],1719,is_old)
        assert pin['cell_id']==entry['cell_id']==row['cell_id']
        job_path=SCREEN/'jobs'/old_jobs[identity]['job'] if is_old else stage/entry['job']
        job_digest=old_jobs[identity]['job_sha256'] if is_old else entry['job_sha256']; job=pinned(job_path,job_digest); add(job_path,job_digest)
        if is_old: assert pin==dict(next(x for x in old_index['records'] if x['id']==identity),cell_id=row['cell_id'])
        assert (job['id'],job['baseline_id'],job['candidate'],job['seed'],job['fit_seed'],job['evaluation_split'])==(identity,row['id'],candidate['id'],row['seed'],1719,'valid')
        assert job['settings']==candidate['settings'] and job['settings']['post_rounds']==30
        result=pinned(pin['result'],pin['result_sha256']); acceptance=pinned(pin['acceptance'],pin['acceptance_sha256'])
        out=fpath(pin['result']).parent
        assert fpath(pin['acceptance'])==out/'acceptance.json'
        assert is_old or out==index_path.parent/identity
        assert r['result_path']==pin['result'] and r['result_sha256']==pin['result_sha256'] and r['acceptance_path']==pin['acceptance'] and r['acceptance_sha256']==pin['acceptance_sha256']
        assert result['status']=='complete' and result['job']==job and result['settings']==job['settings'] and result['fit_seed']==1719
        assert [x['round'] for x in result['history']]==list(range(1,31)) and result['metrics']==r['metrics']
        assert result['environment']==r['environment'] and result['metrics']['prediction_count']==19867
        assert acceptance['status']=='PASS' and acceptance['job_sha256']==job_digest
        reference=fpath(locations[row['id']]['path']); location=locations[row['id']]
        assert (location['cell_id'],location['distribution'],location['attack'],location['seed'])==(row['cell_id'],row['distribution'],row['attack'],row['seed'])
        assert all(location[k] for k in ('source_hashes_exact','original70rounds_exact','root_valid_disjoint_and_IDs_exact'))
        if not is_old: assert fpath(entry['reference'])==reference and entry['mapping']==inputs['mappings'][str(row['seed'])]
        mapping=inputs['mappings'][str(row['seed'])]; meta=pinned(fpath(mapping['metadata']),mapping['metadata_sha256'])
        assert job['mapping_sha256']==r['mapping_sha256']==result['mapping_sha256']==mapping['sha256']==meta['mapping_sha256']
        assert job['mapping_metadata_sha256']==result['mapping_metadata_sha256']==mapping['metadata_sha256']
        assert meta['status']=='FROZEN' and meta['approved'] is True and meta['cohorts']==20 and meta['true_training_client_identity'] is False
        assert meta['input_fields_for_mapping']==['image_id'] and meta['semantics']=='declared_virtual_partition'
        assert result['root_image_ids_sha256']==row['root_image_ids_sha256']==meta['root_image_ids_sha256']
        assert result['valid_image_ids_sha256']==row['valid_image_ids_sha256']==meta['valid_image_ids_sha256']
        assert r['checkpoint_sha256']==result['checkpoint_sha256']==row['checkpoint']['sha256']==location['checkpoint_sha256']
        assert r['cache_sha256']==result['accepted_margin_cache_sha256']==row['cache']['sha256']==location['cache_sha256']
        assert result['original_result_sha256']==row['result']['sha256']==location['result_sha256']
        assert row['source_job']['sha256']==location['source_job_sha256']
        native_job=read(reference/'result.json')['revision_job']
        assert r['pretrained_source_runtime']=={k:native_job.get(k) for k in ('python_version','torch_version','visible_gpu','cpu_threads')}
        for name,digest in acceptance['artifact_hashes'].items():
            path=(out/name).resolve(); assert path.is_relative_to(out), 'Artifact path escape'
            add(path,digest)
        for name,key in (('model.pt','checkpoint'),('result.json','result'),('source_job.json','source_job'),('margins.npz','cache')): add(reference/name,row[key]['sha256'])
        add(fpath(mapping['path']),mapping['sha256']); add(fpath(mapping['metadata']),mapping['metadata_sha256'])
        assert all(math.isfinite(r['metrics'][k]) and 0<=r['metrics'][k]<=1 for k in ('accuracy','aeod','aspd'))
        constant=r['constant_prediction']
        assert constant in (None,0,1) and (constant is None or r['metrics']['positive_rate']==constant)
    declared={str(Path(p).resolve()):h for p,h in a['artifact_hashes'].items()}
    assert len(a['artifact_hashes'])==len(declared)==len(artifacts)==1027 and declared==artifacts
    for path,digest in artifacts.items(): assert sha(path)==digest, 'Artifact drift: '+path
    constants=[r['id'] for r in records if r['constant_prediction'] is not None]
    assert constants==a['constant_prediction_ids']==summary['constant_prediction_ids'] and len(constants)==1
    assert summary['all_negative_and_constant_records_retained'] and not summary['recipe_search_or_score_ranking_performed']
    assert summary['fit_seed']==1719 and summary['model_seed_panels']==PANELS and set(summary['panels'])==set(PANELS)
    assert summary['units']=={'accuracy_pct':'percent','aeod':'absolute TPR gap','aspd':'absolute positive-rate gap'} and summary['sample_SD']=='sample SD, ddof1'
    cells={(r['distribution'],r['attack'],r['seed']):r for r in records}
    errors=[]; rows_for_display=[]; seed_first_checks=0
    for name,seeds in PANELS.items():
        panel=summary['panels'][name]
        assert [(r['distribution'],r['attack']) for r in panel['per_scene']]==SCENES
        assert [r['seed'] for r in panel['cross_scene_per_seed']]==seeds
        independent_seed_first={}
        for s,stored in zip(seeds,panel['cross_scene_per_seed']):
            assert stored['n_scenarios']==10
            independent_seed_first[s]={}
            for metric in METRICS:
                xs=[100*cells[d,t,s]['metrics']['accuracy'] if metric=='accuracy_pct' else cells[d,t,s]['metrics'][metric] for d,t in SCENES]
                value=math.fsum(xs)/10
                assert abs(value-stored[metric])<=1e-12
                independent_seed_first[s][metric]=value; seed_first_checks+=1
        display_rows=panel['per_scene']+[dict(distribution='Seed-first',attack='All10 scenes',**panel['cross_scene'])]
        for row in display_rows:
            assert row['n']==row['expected_n']==len(seeds) and row['seeds']==seeds and row['complete']
            for metric in METRICS:
                xs=[independent_seed_first[s][metric] for s in seeds] if row['distribution']=='Seed-first' else [100*cells[row['distribution'],row['attack'],s]['metrics']['accuracy'] if metric=='accuracy_pct' else cells[row['distribution'],row['attack'],s]['metrics'][metric] for s in seeds]
                mean,sd=statistics_independent(xs)
                errors.extend((abs(mean-row[metric]['mean']),abs(sd-row[metric]['sample_sd_ddof1'])))
            rows_for_display.append(row)
    lines=[x for x in (folder/'TABLES.md').read_text(encoding='utf8').splitlines() if x.startswith('| ') and ' ± ' in x]
    assert len(lines)==len(rows_for_display)==33
    displays=0
    for line,row in zip(lines,rows_for_display):
        parts=line.strip('| ').split(' | ')
        assert parts[:3]==[row['distribution'],row['attack'],str(row['n'])]
        for offset,metric in enumerate(METRICS,3):
            digits=3 if metric=='accuracy_pct' else 5
            assert parts[offset]==f"{row[metric]['mean']:.{digits}f} ± {row[metric]['sample_sd_ddof1']:.{digits}f}"; displays+=1
    assert len(errors)==198 and max(errors)<=1e-12 and displays==99 and seed_first_checks==75
    for name in ('ACCEPTANCE100.json','records100.json','SUMMARY100.json','TABLES.md'): assert sha(folder/name)==PINS[name]
    assert sha(index_path)==PINS['index'] and sha(inputs_path)==a['root_inputs_sha256']
    return dict(status='INDEPENDENT_LOGOFAIR100_HASH_PROVENANCE_AND_ARITHMETIC_PASS_NO_ADOPTION',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_records=100,new96=96,reused4=4,complete_scenes=10,
        artifact_hashes_recomputed=1027,original_acceptance_JSON_hashes_checked=100,mean_SD_scalars_recomputed=198,
        display_cells=99,rows=33,cross_scene_seed_first_metric_checks=75,max_abs_statistical_difference=max(errors),
        fit_seed=1719,recipe='LoGoFair-DP_07',virtual_cohorts=20,constant_predictions=1,
        constant_prediction_ids=constants,original_strict_reexecuted=False,arrays_loaded=False,Torch_loaded=False,
        summary_files_sha256={n:PINS[n] for n in ('ACCEPTANCE100.json','records100.json','SUMMARY100.json','TABLES.md')},
        original_source_seal_sha256=PINS['source'],summary_source_seal_sha256=PINS['summary_source'],
        stage_source_sha256=a['stage_source_sha256'],manifest_sha256=a['manifest_sha256'],index_sha256=PINS['index'],
        root_inputs_sha256=a['root_inputs_sha256'],screen32_root_adoption_sha256=PINS['screen32_adoption'],
        review_source_seal_sha256=args.source_seal,root_adopted=False,canonical_written=False,new_CNN=0,new_fits=0,test=False,final_test=False,
        limitations='Fixed recipe07; virtual cohorts are not training clients. AEOD is absolute TPR gap. Validation selection seed91001/other exposed validation seeds, mixed pretrained runtimes, float32-margin adaptation and historical test metadata exposure remain. No new search, significance, final-test or primary-endpoint claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('summary-dir','stage','index','inputs'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-seal',required=True); args=p.parse_args()
    if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'): raise RuntimeError('Unoptimized Python required')
    assert not (OWN/'ROOT_REVIEW.json').exists() and not (OWN/'REVIEW_FAILURE.json').exists(), 'One-shot review output exists'
    try:
        proof=review(args)
        with (OWN/'ROOT_REVIEW.json').open('x',encoding='utf8') as f: json.dump(proof,f,indent=2); f.write('\n')
        print(json.dumps({'status':proof['status'],'proof_sha256':sha(OWN/'ROOT_REVIEW.json'),'artifact_hashes':1027,'scalars':198,'cells':99}))
    except Exception as exc:
        with (OWN/'REVIEW_FAILURE.json').open('x',encoding='utf8') as f:
            json.dump({'status':'REVIEW_FAILED_NO_ADOPTION','command':sys.argv,'error':repr(exc),'traceback':traceback.format_exc(),'automatic_retry':False},f,indent=2); f.write('\n')
        raise


if __name__=='__main__': main()
