"""Read-only Hybrid32 delivery review; stdlib only, no original strict execution."""
from pathlib import Path
import ast, datetime, hashlib, json, math, tarfile, traceback

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
B = ROOT / 'tmp/celeba_hybrid32_final_collection_20261010'
H = ROOT / 'tmp/celeba_hybrid_screen_execution_20261009'
F = Path('F:/YananResearchStorage/GuardFed/celeba_hybrid32_final_collection_20261010')
read = lambda p: json.loads(p.read_bytes())
def sha(p):
    digest = hashlib.sha256()
    with Path(p).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()
def save(name, value):
    with (OUT / name).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False); handle.write('\n')

def main():
    assert sha(B/'FILES_SHA256.json') == '4370962e33b24e7fe9c5ffb461b1a99adc39621bc46efaaf25deb28c64b26e59'
    delivery = read(B/'FILES_SHA256.json')['members']
    assert len(delivery) == len({x['path'] for x in delivery}) == 82
    for item in delivery:
        p = B/item['path']; assert p.resolve().is_relative_to(B.resolve())
        assert sha(p) == item['sha256'] and p.stat().st_size == item['size']
    handoff, link, summary = [read(B/n) for n in ['HANDOFF.json','ROOT_READY_CHAIN_LINK.json','SUMMARY32.json']]
    assert sha(B/'HANDOFF.json') == '65cf0df579719b461db97efe91968eb67df326c4af605f30aedac61b0ecab445'
    assert sha(B/'ROOT_READY_CHAIN_LINK.json') == handoff['ready_sha256']
    assert sha(B/'SUMMARY32.json') == handoff['summary32_sha256']
    previous = read(B/'PREVIOUS_CHAIN.json'); latest = read(B/'PREVIOUS_LATEST.json')
    assert sha(H/'LATEST_BACKUP.json') == sha(B/'PREVIOUS_LATEST.json') == link['previous_latest_sha256']
    assert sha(H/latest['chain_file']) == sha(B/'PREVIOUS_CHAIN.json') == link['previous_chain_sha256']
    assert previous['accepted_total'] == latest['accepted'] == 27
    source = read(H/'FILES_SHA256.json')
    assert sha(H/'FILES_SHA256.json') == link['source_seal_sha256'] == '2c496ae11369465d27ed223f8552321ec8cd424e5fd87e9f2d34e5ea8531e06f'
    for name, digest in source['files'].items(): assert sha(H/name) == digest
    first = (B/'collect_once.py').read_text(encoding='utf-8')
    fixed = (B/'attempt_v2/collect_once.py').read_text(encoding='utf-8')
    old_service = " service=subprocess.check_output(['supervisorctl','status','guardfed_celeba_hybrid_screen32'],text=True).strip()\n"
    new_service = " service_result=subprocess.run(['supervisorctl','status','guardfed_celeba_hybrid_screen32'],text=True,capture_output=True)\n service=service_result.stdout.strip();service_fields=service.split()\n assert len(service_fields)>=2 and service_fields[0]=='guardfed_celeba_hybrid_screen32' and not service_result.stderr.strip()\n assert (service_fields[1],service_result.returncode) in [('EXITED',3),('RUNNING',0)],'Unexpected service state/return code'\n"
    assert first.count(old_service) == 1 and fixed == first.replace(old_service,new_service).replace("files['hybrid32_final_collection_20261010/", "files['hybrid32_final_collection_v2_20261010/")
    meta = read(B/'SOURCE_RECEIPT.json')['collector']; reconstructed = first
    for a,b in reversed(meta['replacements']): assert reconstructed.count(b) >= 1; reconstructed = reconstructed.replace(b,a)
    parent = ROOT/meta['parent']; assert sha(parent) == meta['parent_sha256']
    assert reconstructed == parent.read_text(encoding='utf-8')
    science = lambda text: text[text.index(' import torch\n'):text.index(' # The original queue')]
    assert science(fixed) == science(parent.read_text(encoding='utf-8'))
    assert ast.dump(ast.parse(science(fixed).replace('\n ', '\n')[1:])) == ast.dump(ast.parse(science(first).replace('\n ', '\n')[1:]))
    failure = read(B/'FAILURE.json'); assert 'CalledProcessError(3' in failure['error'] and failure['automatic_retry'] is False
    assert read(B/'COLLECT_TRANSPORT_RECEIPT.json')['exit_code'] == 3
    assert sha(B/'PARTIAL_ACCEPTANCE.json') == sha(B/'attempt_v2/PARTIAL_ACCEPTANCE.json')
    assert sha(B/'BACKUP_SHA256.json') == sha(B/'attempt_v2/BACKUP_SHA256.json')
    authorized = read(B/'AUTHORIZED_SNAPSHOT.json'); snapshot = read(B/'attempt_v2/live_snapshot.json')
    assert authorized['service']['returncode'] == 3 and ' EXITED ' in authorized['service']['stdout']
    assert not authorized['screen_failure'] and ' EXITED ' in snapshot['service']
    assert len(snapshot['rows']) == 32 and all(r['terminal'] and r['result_exists'] and r['round']==70 and not r['failures'] for r in snapshot['rows'])
    assert not any('driver.py' in ' '.join(p['argv']) and '--kind screen' in ' '.join(p['argv']) for p in snapshot['processes'])
    raw = read(B/'RAW_STORAGE_INDEX.json'); assert sha(B/'RAW_STORAGE_INDEX.json') == handoff['raw_storage_index_sha256']
    for item in raw['files'].values():
        p=Path(item['path']); assert p.resolve().is_relative_to(F.resolve())
        assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes']
    archive=Path(handoff['archive_path']); members=read(F/'MEMBERS.json')['members']
    assert sha(archive)==handoff['archive_sha256'] and archive.stat().st_size==handoff['archive_bytes']
    assert sha(F/'MEMBERS.json')==handoff['inventory_sha256']
    with tarfile.open(archive) as tar:
        rows=tar.getmembers(); assert len(rows)==71 and len({m.name for m in rows})==71
        assert {m.name for m in rows} == set(members)|{'MEMBERS.json'} and all(m.isfile() for m in rows)
        for m in rows:
            value=tar.extractfile(m).read(); expected=members.get(m.name,dict(sha256=sha(F/'MEMBERS.json'),size=(F/'MEMBERS.json').stat().st_size))
            assert len(value)==expected['size'] and hashlib.sha256(value).hexdigest()==expected['sha256']
    strict, off, record = [read(F/n) for n in ['PARTIAL_ACCEPTANCE.json','OFFSERVER_MEMBER_TENSOR_PROOF.json','LOCAL_RECORD_CHECKS.json']]
    for name,key in [('PARTIAL_ACCEPTANCE.json','server_strict_sha256'),('OFFSERVER_MEMBER_TENSOR_PROOF.json','offserver_sha256'),('LOCAL_RECORD_CHECKS.json','record_sha256')]: assert sha(F/name)==handoff[key]
    new = handoff['accepted_new_ids']; assert len(new)==len(set(new))==5 and not set(new)&set(previous['accepted_job_ids'])
    assert strict['accepted_new_ids']==off['accepted_new_ids']==new and {r['id'] for r in record['records']}==set(new)
    assert strict['source_data_verified_before_after'] and strict['source_seal_sha256']==link['source_seal_sha256']
    assert strict['collector_sha256']==sha(B/'attempt_v2/collect_once.py') and strict['runtime']['CPU']==[107] and strict['runtime']['threads']==1
    assert off['status']=='ORIGINAL_SERVER_STRICT_PLUS_OFFSERVER_ALL_MEMBERS_AND_CPU_TENSORS_VERIFIED'
    assert record['status']=='RECORD_BOUND_ORIGINAL_SCIENTIFIC_AND_WRITER_CHECKS_PASS' and record['server_check_receipt_sha256']==sha(F/'PARTIAL_ACCEPTANCE.json')
    old_records={}; chain=H/latest['chain_file']; expected=latest['chain_sha256']; chain_pins=[]
    while True:
        assert sha(chain)==expected; c=read(chain); d=H/c['delta_dir']; receipt=d/'PARTIAL_ACCEPTANCE.json'
        assert sha(receipt)==c['server_strict_sha256']; accepted=read(receipt)
        assert accepted['source_seal_sha256']==link['source_seal_sha256']
        root=ROOT/c['root_adoption_path'] if 'root_adoption_path' in c else d/'ROOT_RECORD_REVIEW.json'
        assert sha(root)==c.get('root_adoption_sha256',c.get('root_record_review_sha256'))
        assert sha(d/'OFFSERVER_MEMBER_TENSOR_PROOF.json')==c.get('offserver_proof_sha256',c.get('offserver_tensor_sha256'))
        assert {r['id'] for r in accepted['records']}==set(c['accepted_new_ids'])
        for row in accepted['records']: assert row['id'] not in old_records; old_records[row['id']]=row
        chain_pins.append(dict(path=chain.relative_to(ROOT).as_posix(),sha256=sha(chain)))
        if not c.get('previous_chain_file'): break
        chain=H/c['previous_chain_file']; expected=c['previous_chain_sha256']
    assert set(old_records)==set(previous['accepted_job_ids']) and len(old_records)==27
    records=dict(old_records,**{r['id']:r for r in strict['records']}); scope=read(H/'screen_scope.json'); protocol=read(H/'runtime_protocol.json')
    assert len(records)==32 and link['accepted_job_ids']==previous['accepted_job_ids']+new
    assert set(records)=={e['id'] for e in scope['jobs']}==set(link['accepted_job_ids'])
    presented={r['id']:r for c in summary['all_candidates'] for r in c['records']}; assert len(presented)==32
    constants=[]
    for entry in scope['jobs']:
        job=read(H/entry['job']); r=records[entry['id']]; p=r['original_provenance']
        assert sha(H/entry['job'])==entry['job_sha256']==p['job_sha256'] and p['scope_sha256']==sha(H/'screen_scope.json')
        assert p['source_hashes']==scope['protected_source_hashes'] and p['local_hashes']==scope['local_hashes']
        assert r['rounds']==70 and r['seed']==91001 and r['evaluation_stats']['prediction_count']==19867
        assert (r['distribution'],r['attack'],r['alpha'])==(job['distribution'],job['attack'],job['config']['client_alpha'])
        assert job['config']['rounds']==70 and job['config']['celeba_evaluation_split']=='valid' and job['runtime_protocol_sha256']==sha(H/'runtime_protocol.json')
        assert presented[r['id']]['metrics']==r['metrics'] and presented[r['id']]['checkpoint_sha256']==r['model_sha256'] and presented[r['id']]['acceptance_sha256']==r['acceptance_sha256']
        assert presented[r['id']]['candidate']==job['tuning_candidate'] and presented[r['id']]['distribution']==job['distribution'] and presented[r['id']]['attack']==job['attack']
        if r['evaluation_stats']['positive_rate'] in [0,1]: constants.append(r['id'])
        if r['id'] in new:
            out=F/'restored'/entry['output']; result=read(out/'result.json'); contract=result['data_contract']; image=contract['image_data_contract']
            assert result['metrics']==r['metrics'] and sha(out/'model.pt')==r['model_sha256'] and sha(out/'acceptance.json')==r['acceptance_sha256']
            assert result['config']==job['config'] and result['rounds']==70 and len(result['round_summaries'])==70
            assert contract['train_rows']==162770 and contract['test_rows']==19867 and contract['root_clean_rows']==16277
            assert image['evaluation_split']=='valid' and image['train_eval_disjoint'] and image['root_client_disjoint']
    assert sha(H/'summarize.py')==summary['original_summarize_sha256']=='47a83a21b35983b6fab1d499784df15e01f81792a795b2ed3eb06361eec4403f'
    assert protocol['score']=='accuracy - .35*(.45*aeod + .45*aspd + .10*max(aeod,aspd)) - .10*max(0,max(aeod,aspd)-.06)'
    aggregates=[]; error=0.0
    for candidate in protocol['candidates']:
        rs=[r for r in presented.values() if r['candidate']==candidate['id']]
        assert len(rs)==4 and {(r['distribution'],r['attack']) for r in rs}=={(d,a) for d in ['IID','non-IID'] for a in ['Benign','S-DFA']}
        means={k:math.fsum(r['metrics'][k] for r in rs)/4 for k in ['accuracy','aeod','aspd']}
        scores=[]
        for r in rs:
            m=r['metrics']; gap=max(m['aeod'],m['aspd'])
            scores.append(m['accuracy']-.35*math.fsum([.45*m['aeod'],.45*m['aspd'],.10*gap])-.10*max(0,gap-.06))
        value=math.fsum(scores)/4; declared=next(r for r in summary['all_candidates'] if r['candidate']==candidate['id'])
        error=max(error,abs(value-declared['mean_four_condition_score']),*(abs(means[k]-declared['mean_four_condition_metrics'][k]) for k in means))
        aggregates.append(dict(candidate=candidate['id'],score=value,metrics=means))
    assert error<=1e-15
    ordered=sorted(aggregates,key=lambda c:(-c['score'],c['candidate'])); accuracy=sorted(aggregates,key=lambda c:(-c['metrics']['accuracy'],c['candidate']))[0]['candidate']
    def dominates(a,b):
        x,y=a['metrics'],b['metrics']; return x['accuracy']>=y['accuracy'] and x['aeod']<=y['aeod'] and x['aspd']<=y['aspd'] and x!=y
    pareto=sorted(c['candidate'] for c in aggregates if not any(dominates(d,c) for d in aggregates))
    assert [c['candidate'] for c in ordered]==[c['candidate'] for c in summary['all_candidates']]
    assert summary['selected_recipe']==ordered[0]['candidate'] and summary['accuracy_champion']==accuracy and summary['three_metric_Pareto']==pareto
    assert summary['seed_n']==1 and summary['no_sample_std_or_significance'] and summary['four_conditions_not_four_seeds'] and summary['all_negative_results_preserved']
    assert summary['new_root_adopted']==0 and not summary['final_test_evaluated'] and summary['formal_multi_seed_records']==0
    save('ROOT_INDEPENDENT_REVIEW.json',dict(status='PASS_ACTUAL_HYBRID32_FROZEN_SCREEN_ROOT_ADOPTABLE_NO_ADOPTION',blocking_findings=[],source_delivery_seal_sha256=sha(B/'FILES_SHA256.json'),handoff_sha256=sha(B/'HANDOFF.json'),ready_link_sha256=sha(B/'ROOT_READY_CHAIN_LINK.json'),summary_sha256=sha(B/'SUMMARY32.json'),raw_storage_index_sha256=sha(B/'RAW_STORAGE_INDEX.json'),archive_sha256=sha(archive),archive_members=71,delivery_members_verified=82,F_files_hashed=len(raw['files']),source_members_verified=len(source['files']),previous_accepted=27,new_strict_offserver_verified=5,total_complete=32,exact_new_ids=new,original_grid_exact=True,old_chain_pins=chain_pins,original_scientific_loop_bytes_exact=True,repair_only_EXITED_rc3_status_and_independent_namespace=True,first_failure_preserved_sha256=sha(B/'FAILURE.json'),mean_and_score_scalars_recomputed=32,maximum_independent_fsum_difference=error,selected_recipe=ordered[0]['candidate'],accuracy_champion=accuracy,three_metric_Pareto=pareto,score_gap_to_runner_up=ordered[0]['score']-ordered[1]['score'],all_candidates=aggregates,constant_terminal_count=len(constants),constant_terminal_ids=constants,negative_candidates_preserved=8,seed_n=1,no_sample_SD_or_significance=True,scenario_count_not_independent_seed_n=True,original_strict_or_Torch_executed=False,prediction_arrays_loaded=False,training_or_CNN_or_fit=False,SSH=False,source_or_canonical_STATE_Git_changed=False,root_adopted=False,formal100=False,test=False,reviewed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))

if __name__=='__main__':
    try: main()
    except BaseException as error:
        save('REVIEW_V3_FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False)); raise
