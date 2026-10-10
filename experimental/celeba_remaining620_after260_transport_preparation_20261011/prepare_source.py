"""One source-only metadata rebinding of adopted after251 transport; never executes it."""
from pathlib import Path
import ast,difflib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
OLD=R/'tmp/celeba_remaining620_after251_transport_20261010'
BASE=R/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name,value):
    with (H/name).open('x',encoding='utf8',newline='\n') as f:
        f.write(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    source=R/'tmp/celeba_mechanism_remaining_evaluation_v2_20261010'
    plan=read(source/'PLAN.json')
    assert sha(source/'FILES_SHA256.json')=='a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03'
    parent=R/'tmp/celeba_mechanism_remaining620_after251_root_adoption_20261010/ROOT_ADOPTION.json'
    index=parent.parent/'MECHANISM260_INDEX.json'
    assert sha(parent)=='e0a7ebef21f5cf9dad24fbb92ee4a874efc10fbb7f2ed7f507d9b4dfaef85200'
    assert sha(index)=='aff72cb0b530622e4fb3d2e5f7a1ee89cc741d02f6cf27665d2f3c6f77e28d1c'
    prior=read(index);raw=read(OLD/'RAW_STORAGE_INDEX.json')
    assert len(prior['all_ids'])==260 and len(raw['all_transported_ids'])==80
    assert sha(raw['receipt'])==raw['receipt_sha256']=='4ab3d3e4c3d94795fbe20b2056edc001aa6908321fd8b3ba72b7178a55156266'
    ids=[i for i in plan['remaining620_ids'] if i not in set(prior['all_ids'])][:20]
    assert ids==[f'minus_A_non-IID_{attack}_seed{seed}' for attack in ['F Flip','FedSA'] for seed in range(91001,91011)]
    oldnative=BASE/'root_delta_20261010T140311Z/ROOT_DELTA_VERIFICATION.json'
    assert sha(oldnative)=='e96b1bb6da76016eaf247061c098902d1292c3bf093dc77e63fdddb41d563c66'
    oldinspection=oldnative.parent/'inspection/inspection.json';oldledger=oldnative.parent/'verified_ledger.json'
    assert sha(oldinspection)==read(oldnative)['inspection_sha256'] and sha(oldledger)==read(oldnative)['ledger_sha256']
    oldids={r['id'] for r in read(oldinspection)['records']}
    missing=[i for i in ids if i not in oldids];assert len(missing)==8
    previous_remote='/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010/exports/after251_20261010T130333282821Z/backup_receipt.json'
    native_roots=[BASE/'root_delta_20261010T125824Z/ROOT_DELTA_VERIFICATION.json',oldnative]
    p=dict(status='PREPARED_EXACT20_WAITING_ACTUAL_NATIVE280_NO_EXECUTION',candidate_ids=ids,prior_accepted=260,prior_transported=80,
        prior_root_path=str(parent),prior_root_sha256=sha(parent),prior_index_path=str(index),prior_index_sha256=sha(index),
        previous_local_receipt=raw['receipt'],previous_remote_receipt=previous_remote,previous_receipt_sha256=raw['receipt_sha256'],
        previous_all_transported_ids=raw['all_transported_ids'],native272_root_path=str(oldnative),native272_root_sha256=sha(oldnative),
        native272_inspection_path=str(oldinspection),native272_inspection_sha256=sha(oldinspection),
        native272_ledger_path=str(oldledger),native272_ledger_sha256=sha(oldledger),
        known_native272_candidate_ids=[i for i in ids if i in oldids],pending_native280_ids=missing,
        known_native_archive_roots=[dict(path=str(q),sha256=sha(q)) for q in native_roots],
        future_native_root_path=None,future_native_root_sha256=None,execution_performed=False)
    save('PREPARED.json',p);changes={};patch=[]
    oldseal=read(OLD/'SOURCE_FILES_SHA256.json')['files']
    def change(name,replacements):
        assert sha(OLD/name)==oldseal[name]['sha256']
        old=(OLD/name).read_text('utf8');text=old
        for before,after in replacements:
            assert text.count(before)==1,(name,before,text.count(before))
            text=text.replace(before,after,1)
        compile(text,name,'exec');save(name,text)
        changes[name]=dict(parent_sha256=sha(OLD/name),candidate_sha256=sha(H/name),replacements=[dict(before=a,after=b) for a,b in replacements])
        patch.extend(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='after251/'+name,tofile='after260/'+name))
    pre=(OLD/'remote_preflight.py').read_text('utf8')
    idline=next(line for line in pre.splitlines() if line.startswith('candidate_ids='))
    latestline=next(line for line in pre.splitlines() if line.startswith("assert latest['all_transported_ids']"))
    change('remote_preflight.py',[(idline,'candidate_ids='+repr(ids)),('assert len(ids)==9','assert len(ids)==20'),
        (repr(read(OLD/'PREPARED.json')['previous_remote_receipt']),repr(previous_remote)),
        (repr(read(OLD/'PREPARED.json')['previous_receipt_sha256']),repr(raw['receipt_sha256'])),
        (latestline,"assert latest['all_transported_ids']=="+repr(raw['all_transported_ids'])+" and latest['accepted_offserver']==0"),
        ("'Exact9 batch is not entirely closed; do not export a partial batch'","'Exact20 batch is not entirely closed; do not export a partial batch'")])
    change('execute_once.py',[])
    change('export_once.py',[('len(pre[\'selected_ids\'])==9','len(pre[\'selected_ids\'])==20'),("tag='after251_'","tag='after260_'")])
    change('download_verify_once.py',[("N=len(receipt['accepted_new_ids']);assert N==9","N=len(receipt['accepted_new_ids']);assert N==20"),
        ("'remaining620_after251_transport_20261010'","'rem620_a260'"),
        ("'tmp/celeba_remaining620_after240_transport_20261010/RAW_STORAGE_INDEX.json'","'tmp/celeba_remaining620_after251_transport_20261010/RAW_STORAGE_INDEX.json'"),
        (repr(read(OLD/'PREPARED.json')['previous_receipt_sha256']),repr(raw['receipt_sha256'])),('cumulative=71+N','cumulative=80+N')])
    change('remote_cpu111_export.py',[])
    change('join_saved.py',[
        ('assert N==9','assert N==20'),("==len(prior['all_ids'])==251","==len(prior['all_ids'])==260"),
        ("currentproof['total_new_strict_and_offserver']>=260","currentproof['total_new_strict_and_offserver']==280"),
        ("previousroot=R/prior_proof['native_root_path'];previousproof=pin(previousroot,prior_proof['native_root_sha256'])","previousroot=Path(P['native272_root_path']);previousproof=pin(previousroot,P['native272_root_sha256'])"),
        ("len(oldinspection['records'])==351","len(oldinspection['records'])==372"),
        ("for proof,path in ((currentproof,Path(E['native_root_path'])),):","for root_pin in E['native_archive_roots']:\n   path=Path(root_pin['path']);proof=pin(path,root_pin['sha256'])"),
        ("==251+N and allids[:251]","==260+N and allids[:260]"),
        ('SAVED_ARRAY_EXACT9_NATIVE_IDENTITY_JOIN_PASS_PENDING_ROOT','SAVED_ARRAY_EXACT20_NATIVE_IDENTITY_JOIN_PASS_PENDING_ROOT'),
        ('prior251_objects_and_order_unchanged','prior260_objects_and_order_unchanged'),
        ('old351_native_records_exact','old372_native_records_exact'),('cumulative=251+N','cumulative=260+N')])
    before=ast.parse((OLD/'join_saved.py').read_text('utf8'));after=ast.parse((H/'join_saved.py').read_text('utf8'))
    recordloop=lambda tree:next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='rec')
    assert ast.dump(recordloop(before),include_attributes=False)==ast.dump(recordloop(after),include_attributes=False)
    transport=R/'tmp/celeba_mechanism_remaining_evaluation_transport_20261010/transport.py'
    assert sha(transport)=='f71e6e4152625a5a0582a61ff9b3e55e4ccee65dfd70a4e657851c0247f9c9d7'
    save('SOURCE_DIFF.patch',''.join(patch))
    save('SOURCE_REUSE.json',dict(files=changes,unchanged_transport_sha256=sha(transport),
        original_join_per_record_AST_exact=True,export_CPU111_bridge_byte_exact=True,
        scientific_functions_changed=False,new_CNN=0,new_fit=0,new_training=0,execution_performed=False,
        source_data_pins=read(R/'tmp/celeba_mechanism_remaining_evaluation_transport_20261010/INPUTS.json')))
    print(json.dumps(dict(status='SOURCE_ONLY_PREPARED_NOT_BOUND_NOT_EXECUTED',candidate_ids=ids,native272_known=12,native280_dependency_unbound=8)))


if __name__=='__main__':main()
