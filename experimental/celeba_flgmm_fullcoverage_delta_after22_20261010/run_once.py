"""One original FL96 strict delta after22; raw evidence goes directly to F."""
from pathlib import Path
import ast, datetime, hashlib, json, sys, traceback, types
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
R = H.parents[1]
P = R/'tmp/celeba_flgmm_fullcoverage_delta_after16_20261010'
B = R/'tmp/celeba_flgmm_fullcoverage_incremental_20261009'
O = R/'tmp/celeba_flgmm_fullcoverage_delta_after14_20261010'
S = Path('F:/YananResearchStorage/GuardFed')/H.name/'batch'
PREVIOUS = '15db13ee3c35f49d9fc11b8eaf37d5ef7f360e335359bf6362a388cd496de341'
ROOTPROOF = '31a1e0d16f14a1855acad3654d656ccb84377b7e6bd8fc3d9c10bf560658c8bc'
PACKAGE = '6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())

def save(n, v):
    with (H/n).open('x', encoding='utf8', newline='\n') as f:
        json.dump(v, f, ensure_ascii=False, indent=2); f.write('\n')

def replace_exact(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)

def checked_source(path, expected):
    assert sha(path) == expected, str(path)
    return path.read_text('utf8')

def cpu106(text):
    # Resource metadata only; never replace arbitrary digits inside source SHA pins.
    for a, b in [('CPU107', 'CPU106'), ('{107}', '{106}'), ('107 in a', '106 in a'),
                 ('107 not in aff', '106 not in aff'), ('107 in aff', '106 in aff'),
                 ('CPU=107', 'CPU=106'), ('-c 107 ', '-c 106 '), ("'107'", "'106'")]:
        text = text.replace(a, b)
    return text

def rebind_preflight(text, ids):
    old = "ids=['FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed'+str(x)+'_fullcoverage' for x in (91006,91007)]"
    return cpu106(replace_exact(text, old, 'ids='+repr(ids)))

def parent(ids=None):
    text = checked_source(P/'run_once.py', '5b00a63088fea7294faa12dafae656f1ff4312e9b151a14bd7cfe1eb806136db')
    text = cpu106(text)
    if ids is not None:
        text = replace_exact(text, "IDS=[f'FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed{s}_fullcoverage' for s in (91008,91009)]", 'IDS='+repr(ids))
        text = replace_exact(text, "PREVIOUS='8ca5e7afa10527fd01607082b0a461f0a226a4941091f182ba8fd966cd09197d'", 'PREVIOUS='+repr(PREVIOUS))
        text = replace_exact(text, "assert preflight.count('(91006,91007)')==1;preflight=preflight.replace('(91006,91007)','(91008,91009)')", 'preflight=rebind_preflight(preflight,IDS)')
        text = replace_exact(text, "receipt['accepted_total']==18 and receipt['accepted_new']==2", f"receipt['accepted_total']=={22+len(ids)} and receipt['accepted_new']=={len(ids)}")
    m = types.ModuleType('pinned_original_transport')
    m.__file__ = str(H/'transport_reused.py')
    m.rebind_preflight = rebind_preflight
    exec(compile(text, m.__file__, 'exec'), m.__dict__)
    return m

def snapshot():
    path = R/'tmp/celeba_flgmm_fullcoverage_delta_after18_20261010/run_once.py'
    text = checked_source(path, 'b2faa6f1e1cc344bc44dfa9ef38f3243e038fda1325b9bdb4fdc29749ab6a5f8')
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'snapshot')
    text = ast.get_source_segment(text, node)
    for a, b in [("latest['accepted_total']==18", "latest['accepted_total']==22"),
                 ("len(set(previous['accepted_job_ids']))==18", "len(set(previous['accepted_job_ids']))==22"),
                 ('prior_count=18', 'prior_count=22'), ('accepted_before=18', 'accepted_before=22')]:
        text = replace_exact(text, a, b)
    ns = dict(globals())
    exec(compile(text, str(path)+'[PINNED_SNAPSHOT_METADATA_REBIND]', 'exec'), ns)
    ns['snapshot']()

def source_check():
    assert sha(B/'FILES_SHA256.json') == 'f6de59a56de25a8d316cf7e05c44eafc9751041757e4dc61c88f7bccfd988472'
    for n, pin in read(B/'FILES_SHA256.json')['files'].items(): assert sha(B/n) == pin['sha256']
    assert sha(B/'verify_delta_offserver.py') == 'ecb6627ab90aafa667795d2b3146f6b67abb1924462534640fc31b5dc7b7edfe'

def storage(required):
    path = R/'tmp/guardfed_local_storage.py'
    ns = {'__file__': str(path), '__name__': 'storage_guard'}
    exec(compile(path.read_text('utf8'), str(path), 'exec'), ns)
    return ns['check_bulk_storage'](required)

def verify():
    ids = read(H/'AUTHORIZED_SNAPSHOT.json')['authorized_ids']; m = parent(ids)
    names = ['accepted_delta.tar.gz', 'BACKUP_SHA256.json', 'MEMBERS.json', 'PARTIAL_ACCEPTANCE.json']
    code = "from pathlib import Path\nimport hashlib,json\nb=Path(%r)\nprint(json.dumps({n:dict(sha256=hashlib.sha256((b/n).read_bytes()).hexdigest(),size=(b/n).stat().st_size) for n in %r}))\n" % (m.REMOTE+'/batch', names)
    pins = json.loads(m.run('SERVER_SHA', m.SSH+['python -B -'], code, 30)); save('SERVER_TRANSFER_SHA256.json', pins)
    save('F_VOLUME_BEFORE_TRANSFER.json', storage(sum(x['size'] for x in pins.values())))
    assert not S.exists() and not S.is_symlink() and S.resolve().drive.upper() == 'F:'
    S.mkdir(parents=True)
    m.run('SCP', ['scp','-q','-P','60350','-o','BatchMode=yes','-o','ConnectTimeout=15']+['root@89.22.197.55:'+m.REMOTE+'/batch/'+n for n in names]+[str(S)], timeout=180)
    for n, pin in pins.items(): assert sha(S/n) == pin['sha256'] and (S/n).stat().st_size == pin['size']
    # Original verifier alone performs safe extraction, every member SHA and strict record replay.
    save('F_VOLUME_BEFORE_RESTORE.json', storage(sum(x['size'] for x in read(S/'MEMBERS.json')['members'].values())))
    out = m.run('VERIFY', [sys.executable,'-B',str(B/'verify_delta_offserver.py'),'--batch',str(S),'--release',str(R/'tmp/celeba_flgmm_fullcoverage_root_operations_20261009/attempt_20261009T200518912319Z/verified_manual_v2/stage'),'--receipt-sha256',sha(S/'BACKUP_SHA256.json')], timeout=180)
    print(out.decode(), flush=True)
    tensor = checked_source(O/'check_saved_tensors.py', '877ab0396a9814fd8c7cb41f1fce03ebdf9b84a04b410edeb5b8663e882e3b8d')
    tensor = replace_exact(tensor, "proof['accepted_new']==2 and proof['accepted_total']==16", f"proof['accepted_new']=={len(ids)} and proof['accepted_total']=={22+len(ids)}")
    tensor = tensor.replace("H/'batch/", "S/'")
    exec(compile(tensor, str(O/'check_saved_tensors.py')+'[COUNT_AND_F_PATH_ONLY]', 'exec'), dict(S=S, __file__=str(H/'check_saved_tensors_reused.py'), __name__='__main__'))
    closed = cpu106((O/'close_observation.py').read_text('utf8')).replace('celeba_flgmm_fullcoverage_delta_after14_20261010/collect_delta.py', H.name+'/collect_delta.py')
    save('COLLECTOR_CLOSED.json', json.loads(m.run('CLOSED', m.SSH+['python -B -'], closed, 30)))
    save('RAW_STORAGE_INDEX.json', dict(status='F_ONLY_ACTUAL_RAW_ARCHIVE_AND_RESTORE_SHA_INDEX', raw_storage_root=S.as_posix(), internal_drive_fallback=False, files={p.relative_to(S).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(S.rglob('*')) if p.is_file()},volume_final=storage(0)))

def finalize():
    a=read(H/'AUTHORIZED_SNAPSHOT.json'); prior=read(H/'PREVIOUS_OFFSERVER_ACCEPTANCE.json'); latest=read(H/'PREVIOUS_LATEST.json')
    p=read(S/'OFFSERVER_ACCEPTANCE.json'); receipt=read(S/'BACKUP_SHA256.json'); server=read(S/'PARTIAL_ACCEPTANCE.json')
    tensor=read(H/'SAVED_TENSOR_STATE_CHECK.json'); close=read(H/'COLLECTOR_CLOSED.json'); reuse=read(H/'SOURCE_REUSE.json')
    ids=a['authorized_ids']; n=len(ids); total=22+n
    assert all(read(H/(x+'_COMMAND.json'))['exit_code']==0 for x in ['INITIAL_GUIDE','INITIAL_OWNER','SNAPSHOT','GUIDE','OWNER','PREFLIGHT','COLLECT','SERVER_SHA','SCP','VERIFY','CLOSED'])
    assert sha(B/'LATEST_BACKUP.json')==sha(H/'PREVIOUS_LATEST.json') and sha(R/latest['root_adoption_path'])==ROOTPROOF
    assert sha(H/'PREVIOUS_OFFSERVER_ACCEPTANCE.json')==receipt['previous_chain_sha256']==PREVIOUS
    assert prior['accepted_total']==22 and p['accepted_total']==total and p['accepted_new']==n
    assert p['accepted_job_ids'][:22]==prior['accepted_job_ids'] and len(p['accepted_job_ids'])==len(set(p['accepted_job_ids']))==total
    assert p['new_ids']==receipt['accepted_new_ids']==ids and p['package_sha256']==PACKAGE
    assert server['before_source_data']==server['after_source_data'] and (server['helper_cpu'],server['helper_threads'],server['helper_nice'])==(106,1,10)
    assert sha(S/'accepted_delta.tar.gz')==receipt['archive_sha256'] and sha(S/'MEMBERS.json')==receipt['inventory_sha256']
    assert tensor['CNN_forward_calls']==tensor['optimizer_calls']==tensor['data_loads']==0 and [x['id'] for x in tensor['records']]==ids
    assert close['CPU106_released'] and not close['queue']['failed'] and reuse['per_ID_scientific_loop_bytes_exact'] and reuse['scientific_body_unchanged']
    assert sha(H/'collect_delta.py')==reuse['thin_entry_sha256']==server['collector_sha256']
    ready=dict(status='ROOT_READY_FL96_INCREMENT_STRICT_OFFSERVER_PASS_NOT_ADOPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_new_ids=ids,prior_accepted_new=22,accepted_new=n,accepted_new_cumulative=total,planned_new=96,separately_reused70round=4,planned_total=100,single_fixed_authorization_snapshot=True,authorization_sha256=sha(H/'AUTHORIZED_SNAPSHOT.json'),source_package_sha256=PACKAGE,collector_sha256=sha(H/'collect_delta.py'),thin_entry_parent_collector_sha256=reuse['parent_collector_sha256'],effective_collector_sha256=reuse['effective_collector_sha256'],original_strict_and_archive_body_unchanged=True,verifier_sha256=reuse['original_verifier_sha256'],source_reuse_sha256=sha(H/'SOURCE_REUSE.json'),raw_storage_root=S.as_posix(),raw_storage_index_sha256=sha(H/'RAW_STORAGE_INDEX.json'),offserver_acceptance_path=(S/'OFFSERVER_ACCEPTANCE.json').as_posix(),offserver_acceptance_sha256=sha(S/'OFFSERVER_ACCEPTANCE.json'),archive_path=(S/'accepted_delta.tar.gz').as_posix(),archive_sha256=receipt['archive_sha256'],archive_bytes=receipt['archive_size'],archive_members=receipt['archived_member_count'],member_manifest_path=(S/'MEMBERS.json').as_posix(),inventory_sha256=receipt['inventory_sha256'],server_backup_receipt_sha256=sha(S/'BACKUP_SHA256.json'),server_strict_sha256=receipt['acceptance_sha256'],previous_root_path=latest['root_adoption_path'],previous_root_sha256=ROOTPROOF,previous_actual_offserver_sha256=PREVIOUS,previous_latest_sha256=sha(H/'PREVIOUS_LATEST.json'),old22_ordered_prefix_exact=True,full_saved_tensor_check_sha256=sha(H/'SAVED_TENSOR_STATE_CHECK.json'),tensor_count=sum(x['tensor_count'] for x in tensor['records']),tensor_elements=sum(x['elements'] for x in tensor['records']),records=p['records'],source_data_before_after_exact=True,helper_CPU=106,helper_threads=1,nice=10,IO='idle',CUDA_VISIBLE_DEVICES='',server_verification_runtime=server['acceptance_runtime'],local_verification_runtime=p['verification_runtime'],resource_preflight_sha256=sha(H/'PREFLIGHT.json'),resource_after_sha256=sha(H/'COLLECTOR_CLOSED.json'),CPU106_released=True,no_old_models_repacked=True,CNN_forward_calls=0,training_calls=0,final_test=False,recipe_selection_performed=False,root_adoption_required=True,LATEST_STATE_Git_unchanged=True,scientific_complete=False,prediction_arrays_recomputed=False)
    save('ROOT_READY_HANDOFF.json',ready)
    save('ROOT_READY_CHAIN_LINK.json',dict(status='ROOT_REVIEW_PENDING_CHAIN_LINK_NO_CANONICAL_WRITE',previous_latest_path=B.relative_to(R).as_posix()+'/LATEST_BACKUP.json',previous_latest_sha256=sha(H/'PREVIOUS_LATEST.json'),previous_root_path=latest['root_adoption_path'],previous_root_sha256=ROOTPROOF,previous_offserver_path=latest['next_collector_previous_path'],previous_offserver_sha256=PREVIOUS,proposed_next_collector_previous_path=ready['offserver_acceptance_path'],proposed_next_collector_previous_sha256=ready['offserver_acceptance_sha256'],accepted_before=22,accepted_new=n,accepted_total=total,accepted_job_ids=p['accepted_job_ids'],new_ids=ids,archive_sha256=receipt['archive_sha256'],member_manifest_sha256=receipt['inventory_sha256'],source_package_sha256=PACKAGE,handoff_sha256=sha(H/'ROOT_READY_HANDOFF.json'),old22_ordered_prefix_exact=True,reused4_repacked=False,root_adoption_required=True))
    text=f'''# Actual FL96 after22: root adoption pending\n\nOne frozen snapshot {a['snapshot_utc']} selected exactly {n} new terminal IDs, cumulative {total}/96 plus four separate reused references. Original server strict/member/archive and unchanged offserver scientific record checks passed; complete70-round valid-only identities, source/data/checkpoints and ordered prior22 are bound in ROOT_READY_HANDOFF.json. Later completions are excluded.\n\nOriginal after14 collector is SHA-bound through a tiny entry changing only its authorized-ID literal; original per-job science bytes remain exact. Original transport, preflight, tensor and close readers are reused with counts/namespace/CPU106 metadata only. CPU106 one thread, nice10, idleIO, CUDA hidden was verified free and released. The saved tensor check constructs the original CNN state layout without any forward/data/optimizer calls. Server cu128 and local CPU Torch runtimes differ and are recorded. Prediction arrays are absent and were not regenerated. All negative results and failures are preserved. No recipe selection, final test, training or inference was performed.\n\nArchive/models/raw/logprefix/restored files were written directly to a freshly label/health/capacity-checked F:/YananResearchStorage/GuardFed/, with no E fallback. RAW_STORAGE_INDEX.json binds every actual F member. E contains only small code, index, receipts and reports. Shared LATEST/STATE/Git/queue/source/data were not changed. Root must independently review and adopt.\n'''
    with (H/'README.md').open('x',encoding='utf8',newline='\n') as f:f.write(text)
    files={q.relative_to(H).as_posix():dict(sha256=sha(q),bytes=q.stat().st_size) for q in sorted(H.rglob('*')) if q.is_file() and '__pycache__' not in q.parts}
    save('DELIVERY_FILES_SHA256.json',dict(status='ACTUAL_INCREMENT_OFFSERVER_PASS_ROOT_PENDING',files=files,raw_storage_index_sha256=sha(H/'RAW_STORAGE_INDEX.json'),root_adoption_not_performed=True))
    for name,pin in files.items(): assert sha(H/name)==pin['sha256'] and (H/name).stat().st_size==pin['bytes']
    print(json.dumps(dict(accepted_new=n,total_new=total,archive_sha256=receipt['archive_sha256'],members=receipt['archived_member_count'],handoff_sha256=sha(H/'ROOT_READY_HANDOFF.json'),chain_link_sha256=sha(H/'ROOT_READY_CHAIN_LINK.json'),delivery_seal_sha256=sha(H/'DELIVERY_FILES_SHA256.json'),files=len(files))))

if __name__=='__main__':
    try:
        phase=sys.argv[1];source_check()
        if phase=='snapshot':snapshot()
        elif phase=='collect':
            ids=read(H/'AUTHORIZED_SNAPSHOT.json')['authorized_ids'];assert ids
            parent(ids).collect()
        elif phase=='verify':verify()
        elif phase=='finalize':finalize()
        else:raise ValueError('Unknown bounded phase')
    except BaseException as error:
        save('FAILURE_'+str(__import__('time').time_ns())+'.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False));raise
