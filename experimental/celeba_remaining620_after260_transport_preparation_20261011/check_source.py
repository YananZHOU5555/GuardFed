"""Focused stdlib metadata/source checks; no transport, science imports, or F bulk reads."""
from pathlib import Path
import ast,hashlib,json,sys
from verify_native_inputs import validate
H=Path(__file__).resolve().parent;R=H.parents[1]
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=read(H/'PREPARED.json');ids=p['candidate_ids']
    plan=read(R/'tmp/celeba_mechanism_remaining_evaluation_v2_20261010/PLAN.json')
    prior=read(p['prior_index_path'])
    assert ids==[i for i in plan['remaining620_ids'] if i not in set(prior['all_ids'])][:20]
    assert len(ids)==len(set(ids))==20 and not set(ids)&set(prior['all_ids'])
    assert len(p['known_native272_candidate_ids'])==12 and len(p['pending_native280_ids'])==8
    root=R/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T150639Z/ROOT_DELTA_VERIFICATION.json'
    assert sha(root)=='6cc2c1513f48897d17a28b240a6d512b8caaf2ed9fb329d4d00240336335a6c4'
    proof=read(root)
    e=dict(native_root_verified=True,exact_candidate_ids=ids,native_root_path=str(root),native_root_sha256=sha(root),
        native_inspection_path=str(root.parent/'inspection/inspection.json'),native_inspection_sha256=proof['inspection_sha256'],
        native_ledger_path=str(root.parent/'verified_ledger.json'),native_ledger_sha256=proof['ledger_sha256'],
        native_archive_roots=p['known_native_archive_roots']+[dict(path=str(root),sha256=sha(root))])
    validate(e,p)
    transport=R/'tmp/celeba_mechanism_remaining_evaluation_transport_20261010/transport.py'
    assert sha(transport)=='f71e6e4152625a5a0582a61ff9b3e55e4ccee65dfd70a4e657851c0247f9c9d7'
    tree=ast.parse(transport.read_text('utf8'))
    ns={}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'require','select_delta'}]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(transport),'exec'),ns)
    selected=ns['select_delta'](plan,ids,p['previous_all_transported_ids'])
    assert len(selected)==100 and set(selected)==set(p['previous_all_transported_ids'])|set(ids)
    refusals=[]
    for label,requested in [('duplicate',ids+[ids[0]]),('out_of_order',list(reversed(ids))),
        ('old_transported',[p['previous_all_transported_ids'][0]]),('Full',['GuardFed-AD2+_non-IID_FedSA_seed91001'])]:
        try:ns['select_delta'](plan,requested,p['previous_all_transported_ids'])
        except ValueError:refusals.append(label)
        else:raise AssertionError('Unexpected original delta acceptance: '+label)
    old=R/'tmp/celeba_remaining620_after251_transport_20261010'
    reuse=read(H/'SOURCE_REUSE.json')
    for name,item in reuse['files'].items():
        text=(H/name).read_text('utf8')
        for c in reversed(item['replacements']):
            assert text.count(c['after'])==1
            text=text.replace(c['after'],c['before'],1)
        assert text==(old/name).read_text('utf8')
    recordloop=lambda tree:next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='rec')
    assert ast.dump(recordloop(ast.parse((old/'join_saved.py').read_text('utf8'))),include_attributes=False)==ast.dump(recordloop(ast.parse((H/'join_saved.py').read_text('utf8'))),include_attributes=False)
    assert sha(H/'remote_cpu111_export.py')==sha(old/'remote_cpu111_export.py')
    assert sha(H/'execute_once.py')==sha(old/'execute_once.py')
    assert not (H/'EXECUTION_INPUTS.json').exists() and not (H/'PREFLIGHT_COMMAND.json').exists()
    compiled=[]
    for q in sorted(H.glob('*.py')):compile(q.read_text('utf8'),str(q),'exec');compiled.append(q.name)
    assert 'torch' not in sys.modules and not list(H.glob('__pycache__'))
    result=dict(status='SOURCE_AND_ACTUAL_NATIVE280_METADATA_BINDING_CHECK_PASS_NOT_EXECUTED',
        candidate_ids=ids,prior_replay_count=260,prior_transport_count=80,planned_transport_new=20,
        actual_native_total=280,actual_native_records=380,old372_native_records_exact=True,
        actual_native_ledger_entries=40,old39_native_ledger_entries_exact=True,
        original_transport_source_unchanged=True,original_join_per_record_AST_exact=True,
        metadata_diff_inverse_exact=True,execute_once_and_CPU111_bridge_byte_exact=True,
        original_delta_selection_positive100=True,focused_original_delta_refusals=refusals,
        native_root_sha256=sha(root),compiled=compiled,new_CNN=0,new_fit=0,new_training=0,
        new_transport=0,accepted_offserver=0,root_adopted=0,SSH=0,F_bulk_reads=0,F_writes=0,Git_mutations=0,test=False)
    with (H/'SOURCE_CHECK.json').open('x',encoding='utf8',newline='\n') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
