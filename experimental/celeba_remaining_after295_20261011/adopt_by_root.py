"""Prepared adopter for five saved-array records, retaining the prior295 evidence chain; not executed."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,shutil,tarfile
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from guardfed_local_storage import check_bulk_storage
R=Path(__file__).resolve().parents[2]
S=R/'tmp/celeba_remaining_after295_20261011'
D=R/'tmp/celeba_mechanism_remaining_after295_root_adoption_20261011'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
canon=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--actual-pins-sha256',required=True);args=parser.parse_args()
assert sha(S/'ROOT_ACTUAL_PINS.json')==args.actual_pins_sha256
actual=read(S/'ROOT_ACTUAL_PINS.json')
assert actual['native_total']==300 and actual['replay_total']==300 and actual['new_replays']==5
assert not D.exists()
from verify_native_inputs import verify_native_inputs
execution=verify_native_inputs()
assert execution['native_root_sha256']==actual['native_root_sha256'] and execution['root_native_total']==actual['native_total']
storage=check_bulk_storage()
seal=S/'ACTUAL_DELIVERY_FILES_SHA256.json'
assert sha(seal)==actual['delivery_sha256']
files=read(seal)['files'];assert len(files)==actual['delivery_members']
for name,pin in files.items():
    p=S/name
    assert p.resolve().is_relative_to(S.resolve()) and not p.is_symlink()
    assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
hp=S/'ACTUAL_HANDOFF.json';h=read(hp)
assert sha(hp)==actual['handoff_sha256']
expected=[f'minus_A_non-IID_Sp-DFA_seed{s}' for s in range(91006,91011)]
assert h['selected_ids']==expected and h['accepted_offserver']==h['root_adopted']==0
assert (h['prior_root_accepted_replays'],h['new_transported'],h['replay_total_if_root_adopts'],h['prior_transported'],h['total_transported'])==(295,5,300,115,120)
assert (h['metrics'],h['counts'],h['rules'],h['native_max_abs_difference'])==(45,120,15,0)
assert all(h[k]==0 for k in ('new_CNN','new_fit','new_training','Full_inference','Full_weights_repacked','test','canonical_STATE_written','canonical_ledger_written','Git_mutations'))
p=S/'MECHANISM_INDEX.json';index=read(p)
assert sha(p)==h['proposed_index_sha256']==actual['proposed_index_sha256']
prior=R/index['prior_index_path'];prior_root=R/index['prior_adoption_path']
assert sha(prior)==index['prior_index_sha256']=='7c46fcb20c15c377b1df17378f14b6282ee394bdcaacecd8d4f92be344777bef'
assert sha(prior_root)==index['prior_adoption_sha256']=='70689e63467d8866caa3beb06d2be5f911defd0a4d5f7f061ab005d20bd1c1bb'
assert read(prior_root)['cumulative_accepted']==295
assert index['all_ids']==read(prior)['all_ids']+expected and len(set(index['all_ids']))==300 and index['new_ids']==expected
join=read(S/'NATIVE_IDENTITY_JOIN.json')
assert sha(S/'NATIVE_IDENTITY_JOIN.json')==h['native_identity_join_sha256']
assert join['new_ids']==expected and join['prior295_objects_and_order_unchanged'] and join['old395_native_records_exact'] and join['old_native_ledger_entries_exact']
assert join['new_source_config_data_checkpoint_receipt_identities_exact']==join['Full_reference_joins']==5
for a,b in [('archive_path','archive_sha256'),('receipt_path','receipt_sha256'),('offserver_path','offserver_sha256')]:assert sha(h[a])==h[b]
receipt=read(h['receipt_path']);off=read(h['offserver_path'])
assert receipt['accepted_new_ids']==expected and receipt['all_transported_ids']==index['new_archive']['all_transported_ids']
assert len(receipt['all_transported_ids'])==120 and receipt['previous_backup_receipt_sha256']==h['prior_receipt_sha256']=='37424c0d0bdb2e4b82b0d3b61570e0debb4405677262e86a94c480b0d32ac657'
assert receipt['source_seal_sha256']==off['source_seal_sha256']=='a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03'
assert receipt['transport_source_seal_sha256']=='1a021b707575292c33959c19fcfa2fa1ee8c7f285d20576c562e4843d1488fb3'
original=R/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'
assert sha(original)=='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
spec=importlib.util.spec_from_file_location('original_archive_checker_after295',original)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
archive_check=v.verify_archive(Path(h['archive_path']),receipt)
assert archive_check['pass'] and archive_check['members_verified']==47 and archive_check['different_host_observed']
saved=off['original_saved_array_verification'];records=saved['records']
assert off['receipt_sha256']==h['receipt_sha256'] and off['actual_transport_archive_sha256']==h['archive_sha256']
assert off['accepted_offserver']==0 and off['root_adoption_pending'] and records==index['new_records']
assert [r['id'] for r in records]==expected and saved['accepted_n']==5
assert (saved['independent_metric_checks'],saved['independent_confusion_count_checks'],saved['prediction_rule_checks'])==(45,120,15)
native_root=Path(h['native_root']);np=read(native_root)
assert sha(native_root)==h['native_root_sha256']==actual['native_root_sha256']
assert np['root_adopted'] and np['total_new_strict_and_offserver']==actual['native_total']
ip=R/index['native_inspection_path'];inspection=read(ip);ledger=read(native_root.parent/'verified_ledger.json')
assert sha(ip)==index['native_inspection_sha256']==np['inspection_sha256']
assert sha(native_root.parent/'verified_ledger.json')==np['ledger_sha256']==h['native_ledger_sha256']
old_native=Path(read(S/'PREPARED.json')['native295_root_path']);old_inspection=read(old_native.parent/'inspection/inspection.json');old_ledger=read(old_native.parent/'verified_ledger.json')
assert len(old_inspection['records'])==395 and len(inspection['records'])==100+actual['native_total']
assert [r for r in inspection['records'] if r['id'] not in np['new_ids']]==old_inspection['records']
assert len(ledger['entries'])==43 and ledger['entries'][:-1]==old_ledger['entries']
rows={r['id']:r for r in inspection['records']};native_checks=[]
for na in index['native_archives']:
    ap=Path(na['path']);rp=R/na['root_verification_path']
    assert sha(ap)==na['sha256']==read(rp)['archive_sha256'] and sha(rp)==na['root_verification_sha256']
    with tarfile.open(ap,'r:gz') as archive:
        inv=json.load(archive.extractfile('backup_inventory.json'))
        assert inv['accepted_new_ids']==read(rp)['new_ids']
        for identity in set(expected)&set(inv['accepted_new_ids']):
            binding=index['new_bindings'][identity];r=binding['record'];row=rows[identity]
            assert r['accepted_v4_row']==binding['native_acceptance']['row']==row
            assert (r['variant'],r['distribution'],r['attack'],r['actual_alpha'],r['terminal_round'],r['original_split'],r['original_n_eval'])==('minus_A','non-IID',row['attack'],5,70,'valid',19867) and row['attack'] in ('S-DFA','Sp-DFA')
            assert r['seed']==row['seed']==int(identity[-5:]) and r['config']['ablation_component']=='A' and canon(r['config'])==r['config_canonical_sha256']
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
assert len(native_checks)==10 and {x['id'] for x in native_checks}==set(expected)
for rec in records:
    identity=rec['id'];binding=index['new_bindings'][identity];r=binding['record'];bf=index['new_binding_files'][identity]
    assert sha(bf['path'])==bf['sha256'] and read(bf['path'])==binding
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
release=read(S/'CPU_RELEASE.json');assert not release['CPU111_restricted_owners'] and not release['exporters']
for attack in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA'):
    assert {int(i[-5:]) for i in index['all_ids'] if i.startswith('minus_A_non-IID_'+attack+'_seed')}==set(range(91001,91011))
D.mkdir();target=D/'MECHANISM300_INDEX.json';shutil.copyfile(p,target);assert sha(target)==sha(p)
proof=dict(status='ROOT_AFTER295_EXACT5_SAVED_ARRAYS_REPLAY300_ADOPTED',utc=datetime.now(timezone.utc).isoformat(),accepted_new_ids=expected,new_accepted=5,prior_accepted=295,cumulative_accepted=300,remaining620_new_accepted=120,records_index_path=target.relative_to(R).as_posix(),records_index_sha256=sha(target),root_ready_path=hp.relative_to(R).as_posix(),root_ready_sha256=sha(hp),delivery_seal_sha256=sha(seal),archive_path=h['archive_path'],archive_sha256=h['archive_sha256'],archive_members=47,archive_check=archive_check,offserver_proof_path=h['offserver_path'],offserver_proof_sha256=h['offserver_sha256'],native_root_path=native_root.relative_to(R).as_posix(),native_root_sha256=sha(native_root),native_archives=index['native_archives'],native_inspection_path=index['native_inspection_path'],native_inspection_sha256=sha(ip),native_ledger_sha256=h['native_ledger_sha256'],native_members_rehashed=10,native_member_checks=native_checks,exact_native_records_checked=5,independent_metrics=45,independent_counts=120,prediction_rules=15,native_max_abs_difference=0,original295_unchanged=True,complete_A_scenes=[['IID',a] for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')]+[['non-IID',a] for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA')],partial_A_scenes=[],partial_A_seed_ids=[],new_scene_table_created=False,Full_inference=0,new_CNN=0,new_fit=0,new_training=0,test=False,whole_rebuttal_complete=False,fresh_F_volume=storage,source_adopter_path=Path(__file__).relative_to(R).as_posix(),source_adopter_sha256=sha(__file__),parent_adopter_path='tmp/celeba_remaining_after288_20261011/adopt_by_root.py',parent_adopter_sha256=sha(R/'tmp/celeba_remaining_after288_20261011/adopt_by_root.py'))
with (D/'ROOT_ADOPTION.json').open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(dict(status=proof['status'],accepted=300,A=100,root_sha256=sha(D/'ROOT_ADOPTION.json'),index_sha256=sha(target))))
