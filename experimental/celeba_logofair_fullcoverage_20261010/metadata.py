"""Bind a complete, root-adopted screen to 96 new score fits and four references."""
import argparse, copy, hashlib, importlib.util, json, shutil, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
SCREEN=HERE.parent/'celeba_logofair_screen32_20261010'
REUSE=HERE.parent/'celeba_baselines/logofair_bridge_20261009/reuse_manifest.json'
SCREEN_SEAL='accd5cb8582a344f870188f1e70661b6dc9dc948cc88c9e6f6651e451607bc49'
METHOD='LoGoFair-DP-official-adapted'
SEEDS=list(range(91001,91011))
sys.path.insert(0,str(SCREEN))
from stage_inputs import digest, read, require, bulk_path, write
from summarize import summarize

def pinned(path, wanted):
    require(isinstance(wanted,str) and len(wanted)==64 and digest(path)==wanted,'Missing/changed external SHA: '+str(path))
    return read(path)

def verify_sources():
    require(not sys.flags.optimize,'Optimized Python forbidden')
    require(digest(SCREEN/'FILES_SHA256.json')==SCREEN_SEAL,'Original screen source changed')
    for name,want in read(SCREEN/'FILES_SHA256.json')['files'].items():require(digest(SCREEN/name)==want,'Screen member changed')
    for name,want in read(HERE/'INPUT_PINS.json').items():require(digest(name)==want,'Actual accepted metadata changed')
    for name,want in read(HERE/'FILES_SHA256.json')['files'].items():require(digest(HERE/name)==want,'Prepared adapter changed')

def selection(summary, adoption, summary_sha, index_sha):
    require(adoption['status']=='ROOT_LOGOFAIR_SCREEN32_ADOPTED' and adoption['accepted_count']==32
        and adoption['summary_sha256']==summary_sha and adoption['strict_index_sha256']==index_sha
        and adoption['source_seal_sha256']==SCREEN_SEAL and adoption['test_evaluated'] is False,
        'Complete32 independent root adoption required')
    require(len(adoption['independent_acceptance_sha256'])==64,'Independent strict/offserver proof binding required')
    expected={e['id'] for e in read(SCREEN/'jobs/manifest.json')['jobs']}
    require({r['id'] for r in summary['records']}==expected,'Foreign or missing screen jobs')
    require(all(r['seed']==91001 and r['fit_seed']==1719 for r in summary['records']),'Screen seed changed')
    computed=summarize(summary['records'])
    require(all(summary[k]==v for k,v in computed.items()),'Original four-condition score/tie summary differs')
    require(summary['strict_index_sha256']==index_sha,'Strict index differs')
    chosen=computed['selected_per_method'][METHOD]['candidate']
    return next(c for c in read(SCREEN/'snapshot/logofair_bridge_20261010/protocol.json')['candidates'] if c['id']==chosen)

def verify_index(index, summary):
    manifest={e['id']:e for e in read(SCREEN/'jobs/manifest.json')['jobs']}
    refs={e['id']:e for e in read(SCREEN/'snapshot/logofair_bridge_20261010/reuse_manifest.json')['entries']}
    require(index['status']=='LOCAL_ORIGINAL_STRICT32_COMPLETE_ROOT_REVIEW_PENDING'
        and index['source_seal_sha256']==SCREEN_SEAL and len(index['records'])==32
        and {r['id'] for r in index['records']}==set(manifest),'Exact original strict32 index required')
    wanted={r['id']:r for r in summary['records']}
    for r in index['records']:
        result=pinned(r['result'],r['result_sha256']);accept=pinned(r['acceptance'],r['acceptance_sha256'])
        job=read(SCREEN/'jobs'/manifest[r['id']]['job'])
        source=refs[job['baseline_id']]['source_job'];reported=wanted[r['id']]
        require((reported['candidate'],reported['distribution'],reported['attack'],reported['seed'],reported['fit_seed'])
            ==(job['candidate'],source['distribution'],source['attack'],91001,1719),'Summary scientific cell changed')
        require(accept['status']=='PASS' and accept['job_sha256']==manifest[r['id']]['job_sha256']
            and result['job']==job and result['metrics']==wanted[r['id']]['metrics']
            and result['checkpoint_sha256']==wanted[r['id']]['checkpoint_sha256'],'Strict result/summary mismatch')
        for name,want in accept['artifact_hashes'].items():require(digest(Path(r['result']).parent/name)==want,'Strict artifact changed')

def bridge_source():
    source=(SCREEN/'snapshot/logofair_bridge_20261010/bridge.py').read_text(encoding='utf8')
    old='job["seed"] != 91001'
    require(source.count(old)==1,'Unexpected original identity predicate')
    return source.replace(old,'job["seed"] not in protocol["seeds"]')

def bind(summary_path,summary_sha,adoption_path,adoption_sha,index_path,index_sha,inputs_path,approval_path,approval_sha,out):
    verify_sources()
    summary=pinned(summary_path,summary_sha);adoption=pinned(adoption_path,adoption_sha)
    candidate=selection(summary,adoption,summary_sha,index_sha)
    index=pinned(index_path,index_sha);verify_index(index,summary)
    approval=pinned(approval_path,approval_sha);inputs=read(inputs_path)
    require(approval['status']=='ROOT_LOGOFAIR_FULLCOVERAGE_BIND_APPROVED'
        and approval['summary_adoption_sha256']==adoption_sha and approval['inputs_sha256']==digest(inputs_path)
        and approval['prepared_source_seal_sha256']==digest(HERE/'FILES_SHA256.json')
        and approval['new_fits']==96 and approval['reused']==4 and approval['test'] is False,'Exact external scope approval required')
    require(inputs['status']=='ROOT_ACCEPTED_EXISTING_FEDAVG100_AND_FIXED_COHORT_MAPPINGS'
        and inputs['cache_identity_sha256']==digest(HERE/'CACHE_IDENTITIES100.json'),'Root-bound cached inputs required')
    refs={e['id']:e for e in read(REUSE)['entries']};locations={r['id']:r for r in inputs['references']}
    identities=read(HERE/'CACHE_IDENTITIES100.json')['references']
    require(len(locations)==100 and set(locations)==set(refs),'All100 exact FedAvg references required')
    require(set(inputs['mappings'])==set(map(str,SEEDS)),'All10 actual seed populations required')
    oldmap=read(SCREEN/'mapping_metadata.json')
    checked_mappings=set()
    for row in identities:
        location=locations[row['id']];folder,_=bulk_path(location['path'],0)
        for name,key in (('model.pt','checkpoint'),('result.json','result'),('margins.npz','cache')):
            require(digest(folder/name)==row[key]['sha256'],'Accepted reference changed: '+row['id']+'/'+name)
        mapping=inputs['mappings'][str(row['seed'])]
        mp,_=bulk_path(mapping['path'],0);meta_path,_=bulk_path(mapping['metadata'],0)
        require(digest(mp)==mapping['sha256'],'Actual cohort artifact changed')
        meta=pinned(meta_path,mapping['metadata_sha256'])
        require(meta['approved'] is True and meta['status']=='FROZEN' and meta['semantics']=='declared_virtual_partition'
            and meta['true_training_client_identity'] is False and meta['cohorts']==20
            and meta['domain_hex']==oldmap['domain_hex'] and meta['rule']==oldmap['rule']
            and meta['author_decision_sha256']==oldmap['author_decision_sha256']
            and meta['root_image_ids_sha256']==row['root_image_ids_sha256']
            and meta['valid_image_ids_sha256']==row['valid_image_ids_sha256'],'Original domain and accepted root/valid order required')
        if row['seed'] not in checked_mappings:
            from population import verify_mapping
            verify_mapping(mp,row['root_image_ids_sha256'],row['valid_image_ids_sha256'])
            checked_mappings.add(row['seed'])
        if row['seed']==91001:
            require(mapping['sha256']==oldmap['mapping_sha256'] and mapping['metadata_sha256']==digest(SCREEN/'mapping_metadata.json'),'Old4 mapping byte identity required')
    out,_=bulk_path(out,16*1024*1024);require(not out.exists(),'No overwrite/partial retry')
    out.mkdir(parents=True);shutil.copytree(SCREEN/'snapshot',out/'snapshot')
    stage=out/'snapshot/logofair_bridge_20261010'
    (stage/'bridge.py').write_text(bridge_source(),encoding='utf8')
    protocol=copy.deepcopy(read(stage/'protocol.json'))
    protocol.update(version='logofair_fixed_recipe_fullcoverage100_v1',seeds=SEEDS,candidates=[candidate],
        selection_summary_sha256=summary_sha,selection_root_adoption_sha256=adoption_sha,
        status='FROZEN',execution_started=False,fullcoverage_approval_sha256=approval_sha)
    protocol['limitations'] += ['96 new postprocessing fits plus4 original screen records; no CNN inference.',
        '10 pretrained-model seeds; fit_seed1719 remains fixed, not10 independently chosen post-fit seeds.',
        'Legacy per-result validation_postprocessing_screen label retained; manifest is fullcoverage identity.']
    write(stage/'protocol.json',protocol);write(stage/'reuse_manifest.json',read(REUSE))
    hashes={n:digest(stage/n) for n in ('bridge.py','prepare_reuse.py','protocol.json','reuse_manifest.json')}
    jobs=[];reused=[];(out/'jobs').mkdir()
    screen_records={r['id']:r for r in index['records']}
    for row in identities:
        identity=candidate['id']+'_'+row['cell_id'];m=inputs['mappings'][str(row['seed'])]
        if row['seed']==91001 and row['attack'] in ('Benign','S-DFA'):
            r=screen_records[identity];require(read(r['result'])['baseline_id']==row['id'],'Reuse baseline identity mismatch')
            reused.append(dict(cell_id=row['cell_id'],**r));continue
        job=dict(id=identity,method=METHOD,baseline_id=row['id'],candidate=candidate['id'],settings=candidate['settings'],
            seed=row['seed'],fit_seed=1719,evaluation_split='valid',mapping_sha256=m['sha256'],mapping_metadata_sha256=m['metadata_sha256'],local_hashes=hashes)
        path=out/'jobs'/(identity+'.json');write(path,job)
        jobs.append(dict(id=identity,cell_id=row['cell_id'],job='jobs/'+path.name,job_sha256=digest(path),reference=locations[row['id']]['path'],mapping=m))
    require(len(jobs)==96 and len(reused)==4,'Exact96/4 partition required')
    write(out/'manifest.json',dict(status='BOUND_NOT_EXECUTION_AUTHORIZED',jobs=jobs,reused_jobs=reused,
        candidate=candidate,summary_sha256=summary_sha,adoption_sha256=adoption_sha,inputs_sha256=digest(inputs_path),
        bind_approval_sha256=approval_sha,new_CNN=0,final_test=False,scientific_stage='fixed_recipe_validation_postprocessing100'))
    write(out/'SOURCE_SHA256.json',{p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('summary','summary-sha','adoption','adoption-sha','index','index-sha','inputs','approval','approval-sha','out'):p.add_argument('--'+n,required=True)
    a=p.parse_args();bind(a.summary,a.summary_sha,a.adoption,a.adoption_sha,a.index,a.index_sha,a.inputs,a.approval,a.approval_sha,a.out)
