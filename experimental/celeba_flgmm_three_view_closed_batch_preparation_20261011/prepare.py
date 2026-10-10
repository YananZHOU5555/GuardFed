"""One-time compact metadata freeze; never executes replay, fitting or transport."""
from pathlib import Path
import ast
import copy
import datetime
import difflib
import hashlib
import json
import math
import shutil
import subprocess
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OLD = ROOT / 'tmp/celeba_added_cnn_three_view_gate_preparation_20261010'
BRIDGE = ROOT / 'tmp/celeba_added_cnn_three_view_bridge_20261010'
ROOT44 = ROOT / 'tmp/celeba_flgmm_fullcoverage_delta_after38_20261010/ROOT_ADOPTION_REVIEW.json'
ROOT44_SHA = 'f61a7fa480a62a9394d67a64533568b43e6491ee01d8e1f02005f35a42b6a510'
EXACT3 = ROOT / 'tmp/celeba_added_cnn_exact3_root_execution_20261010/ROOT_SCIENTIFIC_ADOPTION.json'
EXACT3_SHA = '631d7ee2acf1cbe3453d849523e262f29ff5b354b5ddb56c60e5779e94364456'
PREFIX = 'FLGMM_Tg20_L2.0_lr0.001'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def save(name, value):
    with (HERE / name).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def pin(p, expected=None):
    p = Path(p).resolve()
    h = sha(p)
    assert expected is None or h == expected, ('Changed pinned compact input', str(p))
    return {'path': p.as_posix(), 'sha256': h, 'bytes': p.stat().st_size}


def main():
    assert not (HERE / 'MANIFEST.json').exists(), 'Freeze once; preserve prior attempt'
    volume = json.loads(subprocess.check_output(['powershell', '-NoProfile', '-Command',
        'Get-Volume -DriveLetter F | Select-Object DriveLetter,FileSystemLabel,HealthStatus,SizeRemaining,Size | ConvertTo-Json -Compress'], text=True))
    assert volume['FileSystemLabel'] == 'Yanan 2TB' and volume['HealthStatus'] == 'Healthy'
    assert volume['SizeRemaining'] >= 1024 ** 3
    sources, mapping = {}, {}
    (HERE / 'originals').mkdir(exist_ok=True)
    (HERE / 'proofs').mkdir(exist_ok=True)

    def remember(p, expected=None):
        q = pin(p, expected)
        sources[q['path']] = {k: q[k] for k in ('sha256', 'bytes')}
        return q

    def compact(p, relative, expected=None):
        q = remember(p, expected)
        assert q['bytes'] < 1024 ** 2, 'Only compact source/proof copies'
        dst = HERE / relative
        if not dst.exists():
            shutil.copyfile(q['path'], dst)
        assert sha(dst) == q['sha256']
        mapping[q['path']] = dict(q, kind='package', relative=relative)
        return q

    oldseal = read(OLD / 'FILES_SHA256.json')
    assert sha(OLD / 'FILES_SHA256.json') == '49c42b90214eae57d456f8ded1d2a4ff2de2762a78d6bcf5c4ae1a5766de9434'
    remember(OLD / 'FILES_SHA256.json')
    for name in ('bridge.py', 'core.py', 'evaluator.py', 'replay.py', 'celeba_data.py', 'SOURCE_REUSE.json'):
        rel = 'originals/' + name
        compact(OLD / rel, rel, oldseal['files'][rel]['sha256'])
    reuse = read(HERE / 'originals/SOURCE_REUSE.json')
    for e, name in zip(reuse['function_sources'], ('evaluator.py', 'replay.py')):
        mapping[e['file']['path']] = dict(e['file'], kind='package', relative='originals/' + name)
    mapping[reuse['original_core']['path']] = dict(reuse['original_core'], kind='package', relative='originals/core.py')
    # Original validation and identity constructor, compiled without NumPy/Torch imports.
    original = (BRIDGE / 'bridge.py').read_text(encoding='utf-8')
    remember(BRIDGE / 'bridge.py', 'b15ef1332ff0c8077e148707aa49594cbe218681ab3d9192976974cc8de6e1ba')
    tree = ast.parse(original)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'require', 'validate_metadata'}]
    ns = {'METHODS': frozenset({'FLGMM'}), 'np': types.SimpleNamespace(isfinite=math.isfinite)}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<original-metadata-validation>', 'exec'), ns)
    identity_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'identity_record')
    constructor = ast.FunctionDef(name='construct', args=ast.parse('def f(job_id, method, job, result, prov, evidence, chain): pass').body[0].args,
        body=[copy.deepcopy(identity_node.body[-1])], decorator_list=[])
    constructor = ast.fix_missing_locations(ast.Module(body=[constructor], type_ignores=[]))
    cn = {'copy': copy}
    exec(compile(constructor, '<unchanged-original-identity-return>', 'exec'), cn)
    checker = ROOT / 'tmp/celeba_flgmm_fullcoverage_root_operations_20261009/attempt_20261009T200518912319Z/verified_manual_v2/stage/source/accept_result.py'
    checker_pin = compact(checker, 'originals/accept_result.py', 'a1b39bc0334c81aeba0c7cd9cec73e40c0a9eaf47b66132e614f36c7db935cd0')
    shared = read(OLD / 'MANIFEST.json')['runtime_repo_hashes']
    records, chains, inventory, roots = {}, {}, {}, []

    def register(rootpath, batch, ids, number, *, screen=False, root_adoption=None):
        rp = compact(rootpath, f'proofs/{number}_root.json')
        root = read(rootpath)
        offp = compact(batch / 'OFFSERVER_ACCEPTANCE.json', f'proofs/{number}_offserver.json')
        sp = compact(batch / 'PARTIAL_ACCEPTANCE.json', f'proofs/{number}_strict.json')
        mp = compact(batch / 'MEMBERS.json', f'proofs/{number}_members.json')
        bp = compact(batch / 'BACKUP_SHA256.json', f'proofs/{number}_receipt.json')
        off, strict, members, receipt = (read(batch / name) for name in
            ('OFFSERVER_ACCEPTANCE.json', 'PARTIAL_ACCEPTANCE.json', 'MEMBERS.json', 'BACKUP_SHA256.json'))
        offkey = 'offserver_proof_sha256' if screen else 'offserver_acceptance_sha256'
        assert root[offkey] == offp['sha256']
        assert receipt['acceptance_sha256'] == sp['sha256'] and receipt['inventory_sha256'] == mp['sha256'] == off['inventory_sha256']
        assert root['archive_sha256'] == receipt['archive_sha256'] == off['archive_sha256']
        assert root['server_strict_sha256' if screen else 'server_receipt_sha256'] == (sp['sha256'] if screen else bp['sha256'])
        assert off['original_checked_result_replayed_locally'] is True and off['final_test'] is False
        actual_checker = checker_pin
        if screen:
            actual_checker = compact(ROOT / 'tmp/celeba_flgmm_screen_20261009_v2/source/accept_result.py',
                'originals/screen_accept_result.py', '000b2711673431bd83487eeee0d79913e71805d221bb25829646538f630a06c5')
        assert strict['original_checker_sha256'] == actual_checker['sha256']
        assert strict['before_source_data'] == strict['after_source_data']
        adoption_pin = None
        if root_adoption:
            adoption_pin = compact(root_adoption, f'proofs/{number}_adoption.json', root['root_adoption_sha256'])
            ad = read(root_adoption)
            assert ad['strict_receipt_sha256'] == sp['sha256'] and ad['offserver_proof_sha256'] == offp['sha256']
            assert ad['archive_sha256'] == receipt['archive_sha256']
        chain = {'root': dict(rp, status=root['status']), 'offserver': dict(offp, status=off['status']),
            'strict': dict(sp, status=strict['status']), 'raw_index': mp, 'index_binding': offp,
            'root_offserver_key': offkey, 'index_binding_key': 'inventory_sha256', 'index_files_key': 'members',
            'checker': actual_checker, 'strict_receipt': bp, 'records': {}}
        if adoption_pin:
            chain['root_adoption'] = adoption_pin
        assert set(ids) <= set(root['accepted_new_ids'])
        smap, omap = ({x['id']: x for x in doc['records']} for doc in (strict, off))
        for rid in ids:
            assert rid not in records and rid.startswith(PREFIX + '_')
            sr, ore = smap[rid], omap[rid]
            if screen:
                assert set(ore) == {'id', 'rounds', 'seed', 'distribution', 'attack', 'metrics', 'evaluation_stats', 'checkpoint_sha256'}
                assert all(sr[k] == v for k, v in ore.items())
            else:
                assert sr == ore
            evidence = {}
            out = batch / 'restored/runs' / rid
            server_out = ('/workspace/guardfed_checks/celeba_flgmm_screen_20261009/release_v2/runs/' if screen else
                '/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/stage/runs/') + rid
            artifacts = {}
            for name in ('job.json', 'result.json', 'provenance.json', 'acceptance.json', 'model.pt'):
                k = name.split('.')[0]
                key = 'runs/' + rid + '/' + name
                info = members['members'][key]
                p = out / name
                ep = {'path': p.resolve().as_posix(), 'sha256': info['sha256'], 'bytes': info.get('bytes', info.get('size')), 'index_key': key}
                if name != 'model.pt':
                    remember(p, ep['sha256'])
                    assert p.stat().st_size == ep['bytes']
                evidence[k] = ep
                artifacts[k] = dict(ep, server_path=server_out + '/' + name)
                mapping[ep['path']] = dict(ep, kind='server', server_path=server_out + '/' + name)
            job, result, prov, acceptance = (read(out / name) for name in ('job.json', 'result.json', 'provenance.json', 'acceptance.json'))
            ns['validate_metadata']('FLGMM', job, result, prov, acceptance, sr, ore)
            assert prov['job_sha256'] == sr['job_sha256']
            if screen:
                stored_bytes = (out / 'job.json').read_bytes()
                assert b'\r' not in stored_bytes
                original_bytes = stored_bytes.replace(b'\n', b'\r\n')
                assert hashlib.sha256(original_bytes).hexdigest() == prov['job_sha256']
                relative = f'proofs/{number}_original_job_{rid}.json'
                original_job_path = HERE / relative
                if not original_job_path.exists():
                    with original_job_path.open('xb') as f:
                        f.write(original_bytes)
                assert original_job_path.read_bytes() == original_bytes
                original_job_pin = pin(original_job_path, prov['job_sha256'])
                chain.setdefault('original_job_by_id', {})[rid] = original_job_pin
                mapping[original_job_pin['path']] = dict(original_job_pin, kind='package', relative=relative)
                artifacts['original_job'] = dict(original_job_pin, server_path='/workspace/guardfed_checks/celeba_flgmm_screen_20261009/release_v2/jobs/' + rid + '.json')
            else:
                assert prov['job_sha256'] == evidence['job']['sha256']
            assert sr['original_acceptance_sha256'] == evidence['acceptance']['sha256']
            assert evidence['model']['sha256'] == sr['checkpoint_sha256']
            assert job['tuning_candidate'] == PREFIX
            for rel, h in shared.items():
                assert prov['source_hashes'][rel] == h
            for name, h in acceptance['artifact_hashes'].items():
                key = 'runs/' + rid + '/' + name
                info = members['members'][key]
                assert info['sha256'] == h
                pp = out / name
                ap = {'path': pp.resolve().as_posix(), 'sha256': h, 'bytes': info.get('bytes', info.get('size')), 'server_path': server_out + '/' + name}
                artifacts.setdefault(name, ap)
                mapping[ap['path']] = dict(ap, kind='server')
            chain['records'][rid] = evidence
            ident = cn['construct'](rid, 'FLGMM', job, result, prov, evidence, chain)
            env = prov.get('environment', prov)
            records[rid] = {'id': rid, 'method': 'FLGMM', 'distribution': job['distribution'], 'attack': job['attack'],
                'seed': job['config']['seed'], 'terminal_round': 70, 'split': 'valid', 'n_eval': 19867,
                'actual_alpha': result['alpha'], 'identity': ident, 'runtime_output': server_out,
                'runtime_artifacts': artifacts, 'original_training_device': env['device'], 'original_training_torch': env['torch'],
                'external_proofs': {k: chain[k] for k in ('root', 'offserver', 'strict')}, 'original_checkpoint_reused_from_screen': screen}
            inventory[rid] = dict(sr, root_pin=rp, original_screen_reuse=screen)
            chains[rid] = chain
        roots.append({'root': rp, 'accepted_new_ids': ids, 'strict': sp, 'offserver': offp,
                      'members': mp, 'backup_receipt': bp, 'archive_sha256': receipt['archive_sha256']})

    remember(ROOT44, ROOT44_SHA)
    latest = read(ROOT44)
    assert latest['accepted_total'] == 44 and latest['reused_separately'] == 4
    accepted44 = latest['accepted_job_ids']
    assert len(accepted44) == len(set(accepted44)) == 44
    chain_roots = []
    p = ROOT44
    while True:
        j = read(p)
        chain_roots.append(p)
        q = j.get('previous_root_adoption_path')
        if not q:
            break
        p = ROOT / q
        assert sha(p) == j['previous_root_adoption_sha256']
    for number, p in enumerate(reversed(chain_roots)):
        j = read(p)
        index = p.parent / 'RAW_STORAGE_INDEX.json'
        batch = Path(read(index)['raw_storage_root']) if index.exists() else p.parent / 'batch'
        if index.exists():
            remember(index, j.get('raw_storage_index_sha256'))
        register(p, batch, j['accepted_new_ids'], number)
    assert set(records) == set(accepted44)
    stage = ROOT / 'tmp/celeba_flgmm_fullcoverage_root_operations_20261009/attempt_20261009T200518912319Z/verified_manual_v2/stage'
    adoption = read(ROOT / 'tmp/celeba_flgmm_fullcoverage_binding_20261009/ROOT_BOUND_ADOPTION.json')
    remember(ROOT / 'tmp/celeba_flgmm_fullcoverage_binding_20261009/ROOT_BOUND_ADOPTION.json', 'fcecfc0a3582695edfd54c70db38e7dafdd5bf46dcdff8212b9dfc03fc7506fc')
    bound_pin = remember(stage / 'manifest.json', adoption['manifest_sha256'])
    bound = read(stage / 'manifest.json')
    reuse4 = bound['reused_jobs']
    assert len(reuse4) == 4
    screen_chain = ROOT / 'tmp/celeba_flgmm_screen_20261009_v2_dispatch/BACKUP_CHAIN_accepted_delta_after19_v2_20261009.json'
    sc = read(screen_chain)
    screen_root = ROOT / sc['root_adoption_path']
    remember(screen_chain, '3844e24261b7e3a89a550b0fcb0016bbf2fb1b8a579f6e51b6a3e5afff23032b')
    ids4 = [x['id'] for x in reuse4]
    register(screen_chain, screen_root.parent, ids4, len(chain_roots), screen=True, root_adoption=screen_root)
    for x in reuse4:
        assert all(inventory[x['id']][k] == v for k, v in x['accepted_record'].items() if k not in ('accuracy', 'aeod', 'aspd', 'score'))
        assert inventory[x['id']]['job_sha256'] == x['job_sha256']
    assert len(records) == len(inventory) == 48
    models = [r['identity']['checkpoint']['sha256'] for r in records.values()]
    equal_weight_groups = {}
    for rid, r in records.items():
        equal_weight_groups.setdefault(r['identity']['checkpoint']['sha256'], []).append(rid)
    # Check actual accepted checkpoint identities, never ID substring aliases.
    exactpin = remember(EXACT3, EXACT3_SHA)
    exact = read(EXACT3)
    assert exact['root_adoption'] is True and exact['Linux_whole_original_saved_check_pass'] is True
    old900 = ROOT / 'tmp/celeba_nine_method_three_view_tables_20261009/records_three_views_900.json'
    n900pin = remember(old900)
    old900doc = read(old900)
    assert old900doc['final_collector_sha256'] == '00e0cc89784832f8fc8293ce39e2c8ec6247f0e5d0fc37288cd3032133a9e6a3'
    n900 = old900doc['records']
    assert len(n900) == 900 and all(x['method'] != 'FLGMM' for x in n900)
    accepted_views = {(x['id'], x['checkpoint_sha256']): {'id': x['id'], 'source': n900pin} for x in n900}
    for x in exact['records']:
        accepted_views[(x['id'], x['checkpoint_sha256'])] = {'id': x['id'], 'source': exactpin, 'actual_record': x}
    skips = []
    ordered = accepted44 + ids4
    for rid in ordered:
        r = records[rid]
        h = r['identity']['checkpoint']['sha256']
        if (rid, h) in accepted_views:
            proof = accepted_views[(rid, h)]
            assert proof['id'] == rid and proof['source']['sha256'] == EXACT3_SHA
            skips.append({'id': rid, 'checkpoint_sha256': h, 'reason': 'ACTUAL_ROOT_ADOPTED_EXACT3_THREE_VIEW_INTERFACE', 'accepted_proof': proof})
    assert len(skips) == 1 and skips[0]['id'] == PREFIX + '_IID_S-DFA_seed91005_fullcoverage'
    selected = [rid for rid in ordered if rid not in {x['id'] for x in skips}]
    assert len(selected) == 47 and all(rid in selected for rid in ids4)
    # Registry contains only this finite pending batch. Original checker is unchanged.
    registry = {'status': 'FROZEN_EXTERNAL_ROOT_STRICT_OFFSERVER_IDENTITIES_NO_NEW_REPLAY',
        'chains_by_id': {rid: chains[rid] for rid in selected}, 'chains': {'FLGMM': {'registered_ids': selected}}}
    save('PROOF_PINS.json', registry)
    manifest = {'scope': 'FLGMM_CLOSED44_PLUS4_MINUS_ADOPTED1_EXACT47_THREE_VIEW_CANDIDATE',
        'status': 'PREPARED_NOT_EXECUTED_NOT_DISPATCH_AUTHORIZED', 'bridge_root_review_sha256': 'be5e6c949254960f55b765ec976f1e681a26450416f156ca237965dedc87fe69',
        'server_repo': '/workspace/GuardFed-celeba-expanded', 'server_python': '/workspace/guardfed_envs/celeba-cu128-20261009/bin/python',
        'views': ['native', 'raw', 'shared_calibration'], 'native_tolerance': 1e-12, 'cpu_affinity': list(range(120,128)), 'threads': 8,
        'new_scientific_acceptances': 0, 'dispatch_authorized': False, 'root_accepted_new_training_records': 44,
        'separately_reused_screen_checkpoints': 4, 'previous_three_view_accepted_in_scope': 1, 'pending_three_view_records': 47,
        'exact_ids': selected, 'records': [records[rid] for rid in selected], 'path_map': mapping,
        'runtime_repo_hashes': shared, 'proof_registry_sha256': sha(HERE / 'PROOF_PINS.json'), 'FL44_root_pin': pin(ROOT44),
        'authorizations_and_runtime_facts_pending': ['independent/root source review and explicit exact47 authorization',
            'fresh Linux CPU120-127 eligible/all-thread freedom and quota/memory/GPU/storage preflight',
            'actual original server artifact/source/data SHA and selected-producer quiescence checks',
            'actual replay receipts and unchanged Linux whole saved checker; F offserver arrays/members and root adoption'],
        'no_final_primary_endpoint_choice': True, 'test': False, 'new_bulk_copies': 0}
    save('MANIFEST.json', manifest)
    save('SKIP_AND_REUSE.json', {'strict_training_scope': 48, 'root44': pin(ROOT44), 'bound_reuse_manifest': bound_pin,
        'screen_reuse_ids': ids4, 'screen_reuse_three_view_already_accepted': [], 'screen_reuse_to_evaluate': ids4,
        'skip': skips, 'pending_exact_ids': selected, 'old900_methods': sorted(set(x['method'] for x in n900)),
        'old900_record_checkpoint_intersection': [], 'observed_but_unaccepted_training_ids_included': 0,
        'distinct_checkpoint_bytes': len(set(models)), 'equal_weight_distinct_scenario_records_retained': {h: ids for h, ids in equal_weight_groups.items() if len(ids)>1},
        'other_three_view_scope': 'Mechanism accepted indexes are Full/minus-control identities; root confirms no other added-CNN acceptance beyond exact3.',
        'future_coverage_or_recipe_selection': False})
    save('ACCEPTED_NATIVE_IDENTITIES.json', {'records': [inventory[rid] for rid in ordered], 'root_chain': roots,
        'identity_only_not_three_view_acceptance': True, 'checkpoint_hashes_read_from_original_root_bound_member_indexes': True,
        'large_model_or_archive_rehashes_performed': 0})
    save('SOURCE_PINS.json', {'files': sources, 'runtime_source_data_hashes': shared,
        'FL_fullcoverage_package_sha256': latest['source_package_sha256'], 'screen_reuse_package_sha256': read(screen_root.parent/'PARTIAL_ACCEPTANCE.json')['package_sha256']})
    save('STORAGE_CHECK.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'volume': volume,
        'operation': 'Read existing compact JSON/source; write compact preparation only on E', 'new_bulk_writes': 0,
        'future_bulk_location': 'Original Linux server or F:/YananResearchStorage/GuardFed/ after fresh volume/capacity guard'})
    print(json.dumps({'strict_scope': 48, 'skipped': 1, 'pending_exact': 47, 'reused_screen_pending': 4,
                      'root_chunks': len(roots), 'manifest_sha256': sha(HERE/'MANIFEST.json'), 'new_CNN': 0, 'new_fit': 0}))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        failure = HERE / ('PREPARATION_FAILURE.txt' if not (HERE / 'PREPARATION_FAILURE.txt').exists() else 'PREPARATION_UNEXPECTED_FAILURE.txt')
        with failure.open('x', encoding='utf-8') as f:
            f.write(traceback.format_exc())
        raise
