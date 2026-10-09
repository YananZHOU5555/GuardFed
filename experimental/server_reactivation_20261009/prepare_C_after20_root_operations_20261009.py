"""Rebind the accepted C_after12 root transport to the reviewed exact5 scope."""
from pathlib import Path
import ast, hashlib, json

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'tmp/celeba_mechanism_C_after12_root_operations_20261009'
NEW = ROOT / 'tmp/celeba_mechanism_C_after20_root_operations_20261009'
NEW.mkdir(exist_ok=True)
replacements = {
    'C_after12': 'C_after20', 'C_AFTER12': 'C_AFTER20',
    'EXACT8': 'EXACT5', 'exact8': 'exact5',
    '4e8b7bde2f3cb30c850623c0153ecba01ba1135185f7e2375b4b052301224f99': 'df9d23010d0d0acc3d81b8f53b061c69e8eca8228febc3a1a2fc35b0cc1c3739',
    '0d81aa3f26490dd26c9f2254fa89cde5bc23e3e402c6333d089da22cc167efae': 'd359805ca590958da37a02735fdd2daf1efe615332d91c37118835b97e24b260',
    'a95a92487ac3d68411b4c0137b19d130332068b572569cf5642c4af8487009c9': '414816e646a0ba979f878946a83e3e0d1636b5b62ca4e50707f6eeb96771a5ae',
    '2e7a3c7a87b635f1954fbb633035baed462aa192828e025d118497e38ad76c8a': '4dcffa2615b679f217ac953456b7cafc91f90f6ff3abdc670f95b74a2635016d',
    '7e0e0a3d1e32efaf7a02e57327cc2b255b9616eef25eb0d5a45bd830d52e4e73': '88c1272b423d91908dbd60b8095c5a4434c2b339807f85ead179afdda802c679',
    'inventory_actual120_Full100refs.json': 'inventory_actual125_Full100refs.json',
    "review['native_accepted_snapshot'] == 120": "review['native_accepted_snapshot'] == 125",
    'old112_records_exact': 'old120_records_exact',
    "[f'minus_C_IID_F Flip_seed{s}' for s in range(91003, 91011)]": "[f'minus_C_IID_FedSA_seed{s}' for s in (91001, 91003, 91005, 91006, 91008)]",
    '== 112': '== 120', '==112': '==120',
    '== 8': '== 5', '==8': '==5',
    '(72,192,24)': '(45,120,15)',
    'prior_three_view_models=112,accepted_new=8,cumulative_three_view_models=120': 'prior_three_view_models=120,accepted_new=5,cumulative_three_view_models=125',
    'original112_unchanged': 'original120_unchanged', 'prior112_root_adoption_sha256': 'prior120_root_adoption_sha256',
    'original112_not_rerun': 'original120_not_rerun',
    'C_after1_20261009/execution_candidate/backups/incremental_20261009T193419Z': 'C_after12_20261009/execution_candidate/backups/incremental_20261009T202623Z',
    '8d064687ad7841e1050120a12ada9e10458fea5bb6a4d9da5aeed77584437fc5': '817d5f8ebebb566ee4b851fd600edcddaf07a410d29e618d1e5a821c5748b775',
}
rows = []
for name in ('deploy.py', 'observe.py', 'backup.py', 'adopt.py'):
    old = (OLD / name).read_text(encoding='utf-8-sig')
    text = old
    for before, after in replacements.items():
        text = text.replace(before, after)
    if name == 'deploy.py':
        text = text.replace("('guardfed_celeba_mechanism_valid_C_after1','EXITED')", "('guardfed_celeba_mechanism_valid_C_after12','EXITED')")
        text = text.replace('PRIOR_NEXT11_EXITED', 'PRIOR_C_AFTER12_EXITED')
        text = text.replace("cmd=['ionice','-c','3'", "cmd=['taskset','-c','112-119','ionice','-c','3'")
        text = text.replace("root_remote_stdout_sha256=", "new_independent_source_review_path=str(review_path), new_independent_source_review_sha256=args.review_sha256, root_remote_stdout_sha256=")
    ast.parse(text)
    assert text != old and 'EXACT8' not in text and 'exact8' not in text
    target = NEW / name
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    rows.append(dict(path=name, sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        accepted_transport_source_sha256=hashlib.sha256((OLD/name).read_bytes()).hexdigest()))
with (NEW/'TRANSPORT_REBIND.json').open('x', encoding='utf-8') as stream:
    json.dump(dict(source_only=True, SSH=False, original_science_changed=False,
        exact5=True, prior120_not_replayed=True, members=rows), stream, indent=2)
print(json.dumps(rows))
