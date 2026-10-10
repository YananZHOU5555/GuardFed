"""One explicit metadata-schema finalization of preserved restored bytes; never extract again."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, os, sys, tarfile, time, traceback
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent/'v2'
SOURCE=ROOT/'tmp/celeba_logofair_fullcoverage_20261010'
SOURCE_SHA='89f3e4b4fd02352fa3c0f5080ce00ab6854eaabb4e3803a2635a1d3ba51196c3'
INVENTORY_SHA='21c534337bc35e4a53b904bc103431acc4d2672bc54f3f8f23c87eb79e743d0f'
DEST=Path('F:/YananResearchStorage/GuardFed/logofair_fullcoverage_20261010/inputs001')
FINISH=DEST/'finalizer_v2'
PRESERVED_FAILURE_SHA='d593b04fed4c7eeced25ed6dd8300466b47005b3f0fcbe3384c440cce6b63e4b'
DIAGNOSIS_SHA='a1fe89c758c497fec129248f8bbe9713ede65dc1d23d927f2f3667ccb8bb50f0'
OLD4_REVIEW_SHA='8c1f58c7dc9925d3d0044ecf5a43bbb0edb43800f6269158219ba08a8146b098'
EXTRAS={'cpu_threads','finished_unix','python_version','torch_version','visible_gpu'}
ENRICHED_IDS={'FedAvg_lr0.002_Benign_seed91001','FedAvg_lr0.002_S-DFA_seed91001'}
KINDS={'checkpoint':'model.pt','result':'result.json','source_job':'source_job.json','cache':'margins.npz'}
sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(ROOT/'tmp/celeba_logofair_screen32_20261010'))
from stage_inputs import digest, bulk_path, read, require


def save(path,value):
    with Path(path).open('x',encoding='utf8') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    import population
    population.verify_sources()
    require(digest(SOURCE/'FILES_SHA256.json')==SOURCE_SHA and digest(SOURCE/'CACHE_IDENTITIES100.json')==INVENTORY_SHA,'Wrong prepared source/inventory')
    require(DEST.is_dir() and digest(HERE.parent/'STAGING_FAILURE.json')==PRESERVED_FAILURE_SHA and digest(DEST/'STAGING_FAILURE.json')==PRESERVED_FAILURE_SHA,'Original failed stage evidence changed')
    require(digest(HERE.parent/'STAGING_FAILURE_DIAGNOSIS.json')==DIAGNOSIS_SHA and digest(HERE.parent/'OLD4_IDENTITY_REVIEW.json')==OLD4_REVIEW_SHA,'Schema/source diagnosis changed')
    require(not HERE.exists() and not FINISH.exists(),'Single finalizer only; no overwrite or retry')
    rows=read(SOURCE/'CACHE_IDENTITIES100.json')['references'];require(len(rows)==len({r['id'] for r in rows})==100,'Exact100 identity required')
    expected_cells={(d,a,s) for d in ('IID','non-IID') for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA') for s in range(91001,91011)}
    require({(r['distribution'],r['attack'],r['seed']) for r in rows}==expected_cells,'Wrong100 cell scope')
    refs={e['id']:e for e in read(ROOT/'tmp/celeba_baselines/logofair_bridge_20261009/reuse_manifest.json')['entries']}
    require(set(refs)=={r['id'] for r in rows},'Original accepted reference set differs')
    old=read(SOURCE/'EXISTING_F_PATHS.json');require(digest(old['input_receipt'])==old['input_receipt_sha256'],'Prior F receipt changed')
    existing={Path(p).name:Path(p) for p in old['existing_four_reference_directories']}
    locations={r['id']:existing.get(r['id'],DEST/r['id']) for r in rows}
    cache_members=read(ROOT/'docs/server_deployment_20260923/training_20260923/celeba_shared_calibration_v1/backup_inventory.json')
    grouped={};ledger=[];required=9*4*1024**2
    for row in rows:
        require(Path(row['id']).name==row['id'],'Unsafe frozen reference ID')
        for kind,name in KINDS.items():
            pin=row[kind];size=pin.get('bytes',cache_members[pin['member']]['bytes'] if kind=='cache' else 0)
            if kind=='cache':require(cache_members[pin['member']]['sha256']==pin['sha256'],'Accepted cache member inventory drift')
            entry=dict(id=row['id'],kind=kind,name=name,path=str(locations[row['id']]/name),sha256=pin['sha256'],bytes=size,
                archive=str(Path(pin['archive']).resolve()),archive_sha256=pin['archive_sha256'],member=pin['member'])
            if row['id'] in existing:
                require(digest(entry['path'])==pin['sha256'] and Path(entry['path']).stat().st_size==size,'Prior F reference changed')
                entry['operation']='REUSE_EXISTING_F_BYTES_NO_COPY'
            else:
                required+=size;entry['operation']='VERIFY_ALREADY_RESTORED_EXACT_ARCHIVE_MEMBER_NO_COPY'
                grouped.setdefault((entry['archive'],entry['archive_sha256']),[]).append(entry)
            ledger.append(entry)
    id_rows=[r for r in rows if r['seed']!=91001 and r['distribution']=='IID' and r['attack']=='Benign'];require(len(id_rows)==9,'Exact9 ID populations')
    id_entries=[]
    for row in id_rows:
        pin=row['accepted_ID_arrays'];e=dict(id=row['id'],seed=row['seed'],kind='accepted_ID_arrays',name='validation_predictions.npz',
            path=str(DEST/'accepted_id_arrays'/str(row['seed'])/'validation_predictions.npz'),sha256=pin['sha256'],
            archive=str(Path(pin['archive']['path']).resolve()),archive_sha256=pin['archive']['sha256'],member=pin['member'],operation='VERIFY_ALREADY_RESTORED_EXACT_ARCHIVE_MEMBER_NO_COPY')
        grouped.setdefault((e['archive'],e['archive_sha256']),[]).append(e);id_entries.append(e)
    guarded,storage=bulk_path(FINISH,9*4*1024**2);require(guarded==FINISH.resolve(),'Wrong F finalization destination')
    HERE.mkdir(exist_ok=False);FINISH.mkdir(parents=True,exist_ok=False)
    before=read(SOURCE/'FILES_SHA256.json')['files'];archives=[];start=time.monotonic()
    try:
        for (archive,wanted),entries in grouped.items():
            require(digest(archive)==wanted,'Accepted source archive changed: '+archive)
            for e in entries:
                target=Path(e['path']);require(target.is_file() and digest(target)==e['sha256'],'Preserved restored member differs')
                require('bytes' not in e or target.stat().st_size==e['bytes'],'Preserved restored member size differs')
                e['bytes']=target.stat().st_size
            archives.append(dict(path=archive,sha256=wanted,exact_members_previously_restored=len(entries),reextracted_in_finalizer=0))
        identities=[]
        for row in rows:
            folder=locations[row['id']];job=read(folder/'source_job.json');result=read(folder/'result.json');ref=refs[row['id']]
            original=dict(ref['source_job']);original.pop('checkpoint_sha256',None)
            expected_extra=EXTRAS if row['id'] in ENRICHED_IDS else set()
            require(set(original)-set(job)==expected_extra and set(job)<=set(original) and all(original[k]==v for k,v in job.items())
                and all(result['revision_job'].get(k)==v for k,v in original.items()),'Raw/enriched reference/result schema differs: '+row['id'])
            require(all(result['revision_job'][k]==job[k] for k in job),'Original result/job differs')
            require((result['dataset'],result['method'],result['distribution'],result['attack'],result['seed'],result['rounds'])
                ==('celeba','FedAvg',row['distribution'],row['attack'],row['seed'],70),'Scientific reference identity differs')
            require(result['config']==job['config'] and result['alpha']==row['alpha'] and [r['round'] for r in result['round_summaries']]==list(range(1,71)),'Recipe/alpha/terminal horizon differs')
            contract=result['data_contract']['image_data_contract']
            require(contract['actual_train_rows']==162770 and contract['actual_evaluation_rows']==19867 and contract['evaluation_split']=='valid'
                and contract['train_eval_disjoint'] and contract['root_client_disjoint'] and contract['root_image_ids_sha256']==row['root_image_ids_sha256']
                and contract['evaluation_image_ids_sha256']==row['valid_image_ids_sha256'],'Source/data/root/valid reference differs')
            acc=ref['accepted_record'];require(acc['checkpoint_sha256']==row['checkpoint']['sha256'] and acc['original_result_sha256']==row['result']['sha256']
                and acc['cache_sha256']==row['cache']['sha256'] and all(abs(result['metrics'][k]-acc['native'][k])<=1e-12 for k in ('accuracy','aeod','aspd')),'Accepted checkpoint/cache/native identity differs')
            identities.append(dict(id=row['id'],cell_id=row['cell_id'],seed=row['seed'],distribution=row['distribution'],attack=row['attack'],
                path=str(folder),checkpoint_sha256=row['checkpoint']['sha256'],result_sha256=row['result']['sha256'],source_job_sha256=row['source_job']['sha256'],cache_sha256=row['cache']['sha256'],
                source_hashes_exact=True,original70rounds_exact=True,root_valid_disjoint_and_IDs_exact=True))
        original_meta=population.read(population.SCREEN/'mapping_metadata.json')
        old_mapping=Path(old['original_mapping']);old_meta=old_mapping.parent/'mapping_metadata.json'
        require(digest(old_mapping)==original_meta['mapping_sha256'] and digest(old_meta)==digest(population.SCREEN/'mapping_metadata.json'),'Original91001 mapping changed')
        mappings={'91001':dict(path=str(old_mapping),sha256=digest(old_mapping),metadata=str(old_meta),metadata_sha256=digest(old_meta),status='ORIGINAL_FROZEN_SCREEN32_MAPPING_REUSED_NO_COPY')}
        for e in id_entries:
            out=FINISH/'populations'/str(e['seed']);population.prepare(e['seed'],e['path'],out)
            meta=read(out/'mapping_metadata.json');require(meta['status']=='PREPARED_NOT_APPROVED' and not meta['approved'] and not meta['execution_authorized'],'New mapping must stay unapproved')
            mappings[str(e['seed'])]=dict(path=str(out/'mapping.npz'),sha256=digest(out/'mapping.npz'),metadata=str(out/'mapping_metadata.json'),metadata_sha256=digest(out/'mapping_metadata.json'),status=meta['status'],accepted_ID_arrays=e)
        population.verify_sources();require(before==read(SOURCE/'FILES_SHA256.json')['files'],'Prepared source changed')
        for e in ledger+id_entries:require(digest(e['path'])==e['sha256'],'Input changed during stage')
        receipt=dict(status='EXISTING_FEDAVG100_BYTES_STAGED_NEW9_MAPPINGS_PREPARED_NOT_ROOT_APPROVED',
            source_seal_sha256=SOURCE_SHA,cache_identity_sha256=INVENTORY_SHA,references=identities,mappings=mappings,
            original_four_reference_directories_reused=True,existing_files_reused=16,reference_files_restored=0,accepted_ID_array_files_restored=0,reference_files_previously_restored=384,accepted_ID_array_files_previously_restored=9,
            files=ledger+id_entries,archives=archives,storage_preflight=storage,elapsed_seconds=time.monotonic()-start,
            recovery_basis=dict(original_stage_failure_sha256=PRESERVED_FAILURE_SHA,diagnosis_sha256=DIAGNOSIS_SHA,old4_identity_review_sha256=OLD4_REVIEW_SHA,archive_reextraction=False,raw_jobs_modified=False,exact_extra_fields=sorted(EXTRAS),exact_enriched_ids=sorted(ENRICHED_IDS)),
            new_CNN_calls=0,new_training=0,new_postprocessing_fits=0,new_score_caches=0,mapping_arrays_generated=9,
            recipe_selected=False,root_approved=False,original91001_mapping_unchanged=True)
        save(FINISH/'INPUT_RECEIPT.json',receipt)
        save(HERE/'STAGING_RESULT.json',dict(status=receipt['status'],F_receipt=str(FINISH/'INPUT_RECEIPT.json'),F_receipt_sha256=digest(FINISH/'INPUT_RECEIPT.json'),
            source_seal_sha256=SOURCE_SHA,inventory_sha256=INVENTORY_SHA,references=100,new_mappings=9,existing_files_reused=16,restored_reference_files=0,restored_ID_arrays=0,previously_restored_reference_files=384,previously_restored_ID_arrays=9,
            newly_written_reference_or_ID_bytes=0,previously_restored_bytes=sum(e['bytes'] for e in ledger+id_entries if e['operation']=='VERIFY_ALREADY_RESTORED_EXACT_ARCHIVE_MEMBER_NO_COPY'),elapsed_seconds=receipt['elapsed_seconds'],root_approved=False,CNN=0,fits=0))
        print(json.dumps(read(HERE/'STAGING_RESULT.json')))
    except BaseException as error:
        failure=dict(error=repr(error),traceback=traceback.format_exc(),F_destination=str(FINISH),automatic_retry=False)
        save(FINISH/'STAGING_FAILURE.json',failure);save(HERE/'STAGING_FAILURE.json',failure);raise


if __name__=='__main__':main()
