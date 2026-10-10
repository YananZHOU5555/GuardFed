"""Root-only adoption of the fixed saved-array 240+11 batch; no inference or fitting."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, shutil, tarfile
from guardfed_local_storage import check_bulk_storage

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'tmp/celeba_remaining620_after240_transport_20261010'
DEST = ROOT / 'tmp/celeba_mechanism_remaining620_after240_root_adoption_20261010'
read = lambda p: json.loads(Path(p).read_bytes())
canonical = lambda x: hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    assert not DEST.exists()
    storage=check_bulk_storage()
    seal=SRC/'DELIVERY_FILES_SHA256.json'
    assert sha(seal)=='f24092a17fda3a8b706a94289d8dae8dcecb7020ae31f7ea03f4096a561d7d85'
    assert len(read(seal)['files'])==44
    for name,pin in read(seal)['files'].items():
        p=SRC/name
        assert p.resolve().is_relative_to(SRC.resolve()) and not p.is_symlink()
        assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
    hp=SRC/'HANDOFF.json';h=read(hp)
    assert sha(hp)=='037e65396c7a3abf26f772c74d6eeeb8c5a9bf15230caabc9e9fd3efade509a1'
    assert h['status']=='EXACT11_ORIGINAL_SAVED_ARRAY_TRANSPORT_AND_NATIVE251_JOIN_PASS_PENDING_ROOT'
    assert h['accepted_offserver']==h['root_adopted']==0
    assert (h['prior_root_accepted_replays'],h['new_transported'],h['replay_total_if_root_adopts'])==(240,11,251)
    assert (h['prior_transported'],h['total_transported'],h['metrics'],h['counts'],h['rules'],h['native_max_abs_difference'])==(60,71,99,264,33,0)
    expected=[f'minus_A_IID_Sp-DFA_seed{s}' for s in range(91001,91011)]+['minus_A_non-IID_Benign_seed91001']
    assert h['selected_ids']==expected
    join=read(SRC/'NATIVE_IDENTITY_JOIN.json')
    assert sha(SRC/'NATIVE_IDENTITY_JOIN.json')==h['native_identity_join_sha256']=='3aa076255e31211ca965a992f0f291deda6bfec9c97e0a5853e7a4db0f8f4d6d'
    assert join['new_ids']==expected and join['new_source_config_data_checkpoint_receipt_identities_exact']==join['Full_reference_joins']==11
    assert join['old343_native_records_exact'] and join['old36_native_ledger_entries_exact'] and join['prior240_objects_and_order_unchanged']
    p=SRC/'MECHANISM_INDEX.json';index=read(p)
    assert sha(p)==h['proposed_index_sha256']=='fd1531be09ccaa90190e174dc46296632023db38fdfc11ed249c1e188310c5e6'
    prior=ROOT/index['prior_index_path'];prior_proof=ROOT/index['prior_adoption_path']
    assert sha(prior)==index['prior_index_sha256']=='aa0df30db62549f1be808f188fa9a0f9479b9daa812406cd98ff438b2068f8cc'
    assert sha(prior_proof)==index['prior_adoption_sha256']=='8e58b5b568ccd1ccf9a9e6a58978a1882533f83a5034d8d086e96924e40e778e'
    assert read(prior_proof)['cumulative_accepted']==240
    assert index['all_ids']==read(prior)['all_ids']+expected and len(set(index['all_ids']))==251 and index['new_ids']==expected
    for field,pinfield in [('archive_path','archive_sha256'),('receipt_path','receipt_sha256'),('offserver_path','offserver_sha256')]:assert sha(h[field])==h[pinfield]
    receipt=read(h['receipt_path']);off=read(h['offserver_path'])
    assert receipt['accepted_new_ids']==expected and receipt['all_transported_ids']==index['new_archive']['all_transported_ids']
    assert len(receipt['all_transported_ids'])==71 and receipt['previous_backup_receipt_sha256']==h['prior_receipt_sha256']=='5c988da2fe11625f949d4baee5be426b6b184dfbccc78782a2a6abcb236759ac'
    assert receipt['source_seal_sha256']==off['source_seal_sha256']=='a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03'
    assert receipt['transport_source_seal_sha256']=='1a021b707575292c33959c19fcfa2fa1ee8c7f285d20576c562e4843d1488fb3'
    original=ROOT/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'
    assert sha(original)=='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
    spec=importlib.util.spec_from_file_location('original_archive_checker_after240',original)
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    archive_check=verifier.verify_archive(Path(h['archive_path']),receipt)
    assert archive_check['pass'] and archive_check['members_verified']==101 and archive_check['different_host_observed']
    saved=off['original_saved_array_verification'];records=saved['records']
    assert off['receipt_sha256']==h['receipt_sha256'] and off['actual_transport_archive_sha256']==h['archive_sha256']
    assert off['accepted_offserver']==0 and off['root_adoption_pending'] and records==index['new_records']
    assert [r['id'] for r in records]==expected and saved['accepted_n']==11
    assert (saved['independent_metric_checks'],saved['independent_confusion_count_checks'],saved['prediction_rule_checks'])==(99,264,33)
    native_root=Path(h['native251_root']);nativeproof=read(native_root)
    assert sha(native_root)==h['native251_root_sha256']=='19312cdfefec81001248507923392e1b48558977cba1b0cbb37c15f1a999dccb'
    assert nativeproof['root_adopted'] and nativeproof['total_new_strict_and_offserver']==251
    ip=ROOT/index['native_inspection_path'];inspection=read(ip)
    assert sha(ip)==index['native_inspection_sha256']==nativeproof['inspection_sha256']
    ledger=read(native_root.parent/'verified_ledger.json')
    assert sha(native_root.parent/'verified_ledger.json')==nativeproof['ledger_sha256']==h['native251_ledger_sha256']
    previous_native=ROOT/read(prior_proof)['native_root_path'];old_inspection=read(previous_native.parent/'inspection/inspection.json');old_ledger=read(previous_native.parent/'verified_ledger.json')
    assert sha(previous_native)==read(prior_proof)['native_root_sha256']
    assert sha(previous_native.parent/'inspection/inspection.json')==read(previous_native)['inspection_sha256']
    assert sha(previous_native.parent/'verified_ledger.json')==read(previous_native)['ledger_sha256']
    assert len(old_inspection['records'])==343 and len(inspection['records'])==351
    assert [r for r in inspection['records'] if r['id'] not in nativeproof['new_ids']]==old_inspection['records']
    assert len(ledger['entries'])==37 and ledger['entries'][:-1]==old_ledger['entries']
    native_rows={r['id']:r for r in inspection['records']};native_checks=[]
    for native_archive in index['native_archives']:
        ap=Path(native_archive['path']);rp=ROOT/native_archive['root_verification_path']
        assert sha(ap)==native_archive['sha256']==read(rp)['archive_sha256'] and sha(rp)==native_archive['root_verification_sha256']
        with tarfile.open(ap,'r:gz') as archive:
            inv=json.load(archive.extractfile('backup_inventory.json'));assert inv['accepted_new_ids']==read(rp)['new_ids']
            for identity in set(expected)&set(inv['accepted_new_ids']):
                binding=index['new_bindings'][identity];r=binding['record'];row=native_rows[identity]
                assert r['accepted_v4_row']==binding['native_acceptance']['row']==row
                assert (r['variant'],r['distribution'],r['attack'],r['actual_alpha'],r['terminal_round'],r['original_split'],r['original_n_eval'])==('minus_A',row['distribution'],row['attack'],5000 if row['distribution']=='IID' else 5,70,'valid',19867)
                assert r['seed']==row['seed']==int(identity[-5:]) and r['config']['ablation_component']=='A' and canonical(r['config'])==r['config_canonical_sha256']
                assert r['checkpoint']['sha256']==binding['checkpoint_sha256']==row['checkpoint_sha256']
                for name,item in [('model.pt',r['checkpoint']),('result.json',r['result'])]:
                    member='runs/'+identity+'/'+name;data=archive.extractfile(member).read();actual=hashlib.sha256(data).hexdigest()
                    assert actual==item['sha256']==inv['members'][member]['sha256'] and len(data)==inv['members'][member]['bytes']
                    native_checks.append(dict(id=identity,member=member,sha256=actual,bytes=len(data)))
                    if name=='result.json':
                        result=json.loads(data)
                        assert result['config']==r['config'] and result['rounds']==70 and result['seed']==r['seed']
                        assert result['revision_job']['source_hashes']==r['source_hashes'] and result['data_contract']['image_data_contract']==r['data_contract']
                assert inv['members']['jobs/'+identity+'.json']['sha256']==r['raw_job']['sha256']
                assert all(item['sha256'] in row['files'].values() for item in (r['checkpoint'],r['result'],r['raw_job']))
    assert len(native_checks)==22 and {x['id'] for x in native_checks}==set(expected)
    for rec in records:
        identity=rec['id'];binding=index['new_bindings'][identity];r=binding['record']
        bf=index['new_binding_files'][identity];assert sha(bf['path'])==bf['sha256'] and read(bf['path'])==binding
        arts=index['new_artifacts'][identity]
        for a in arts.values():assert sha(a['path'])==a['sha256']
        strict=read(arts['strict_json']['path']);sci=read(arts['scientific_receipt']['path']);bridge=read(arts['bridge_receipt']['path'])
        assert strict['status']=='MECHANISM_VALID_THREE_VIEWS_ACCEPTED' and strict['views']==sci['views']==rec['views']
        assert rec['checkpoint_sha256']==strict['checkpoint_sha256']==sci['checkpoint_sha256']==r['checkpoint']['sha256']
        assert strict['native_comparison']['accepted'] and strict['native_comparison']['tolerance']==bridge['native_tolerance']==1e-12
        assert strict['native_comparison']['max_abs_difference']==rec['native_max_abs_difference']==0
        assert bridge['source_before']==bridge['source_after'] and bridge['artifact_before']==bridge['artifact_after']
        assert sci['weights_before']==sci['weights_after'] and not sci['optimizer_created'] and not sci['gradients_created']
        assert (rec['independent_metric_checks'],rec['independent_confusion_count_checks'],rec['prediction_rule_checks'])==(9,24,3)
    assert all(h[k]==0 for k in ('new_CNN','new_fit','new_training','Full_inference','test_inference','models_repacked'))
    release=read(SRC/'CPU_RELEASE.json');assert not release['CPU111_restricted_owners'] and not release['exporters']
    assert all(len([i for i in index['all_ids'] if i.startswith('minus_A_IID_'+a+'_seed')])==10 for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA'))
    DEST.mkdir();target=DEST/'MECHANISM251_INDEX.json';shutil.copyfile(p,target);assert sha(target)==sha(p)
    proof=dict(status='ROOT_AFTER240_EXACT11_SAVED_ARRAYS_NATIVE251_ADOPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_new_ids=expected,new_accepted=11,prior_accepted=240,cumulative_accepted=251,remaining620_new_accepted=71,records_index_path=target.relative_to(ROOT).as_posix(),records_index_sha256=sha(target),root_ready_path=hp.relative_to(ROOT).as_posix(),root_ready_sha256=sha(hp),delivery_seal_sha256=sha(seal),archive_path=h['archive_path'],archive_sha256=h['archive_sha256'],archive_members=101,archive_check=archive_check,offserver_proof_path=h['offserver_path'],offserver_proof_sha256=h['offserver_sha256'],native_root_path=native_root.relative_to(ROOT).as_posix(),native_root_sha256=sha(native_root),native_archives=index['native_archives'],native_inspection_path=index['native_inspection_path'],native_inspection_sha256=sha(ip),native_ledger_sha256=h['native251_ledger_sha256'],native_members_rehashed=22,native_member_checks=native_checks,exact_native_records_checked=11,independent_metrics=99,independent_counts=264,prediction_rules=33,native_max_abs_difference=0,original240_unchanged=True,source_seal_sha256='a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03',complete_A_scenes=[['IID',a] for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')],partial_A_scenes=[dict(distribution='non-IID',attack='Benign',seeds=[91001],n=1,target_n=10)],new_scene_table_created=False,Full_inference=0,new_CNN=0,new_fit=0,new_training=0,test=False,whole_rebuttal_complete=False,fresh_F_volume=storage,source_adopter_path=Path(__file__).relative_to(ROOT).as_posix(),source_adopter_sha256=sha(__file__),parent_adopter_path='tmp/adopt_A40_replays_root_20261010.py',parent_adopter_sha256=sha(ROOT/'tmp/adopt_A40_replays_root_20261010.py'))
    with (DEST/'ROOT_ADOPTION.json').open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
    print(json.dumps(dict(accepted=251,A=51,complete_IID_scenes=5,root_sha256=sha(DEST/'ROOT_ADOPTION.json'))))

if __name__=='__main__':main()
