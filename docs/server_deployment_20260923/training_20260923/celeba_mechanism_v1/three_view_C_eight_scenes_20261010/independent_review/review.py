"""C80 independent stdlib review, reusing the pinned independent C70 reader in memory."""
import ast, datetime, hashlib, json, re, sys, traceback, types
from pathlib import Path
sys.dont_write_bytecode = True
if sys.flags.optimize: raise RuntimeError('Optimized Python is forbidden')
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
BASE=ROOT/'tmp/celeba_mechanism_C_eight_scenes_20261010'
PRIOR=ROOT/'tmp/celeba_mechanism_C70_independent_review_20261010'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
ACTUAL='42ae68d84e4f0231fe8a9804a84f5a62a72b796d4fa6d06bd4093f21fd70bfae'
HANDOFF='45c799bfa184a85380180f0c41e06d16d82613830a74479cc9e62a85ef853717'
SOURCE='54d6f53696587eeb0d29a195e73ca86dd57c341d46ef210a3500c5be4f8a7fbc'
SNAPSHOT='b22d017bff4e13f1acc21a97205914b454d80b2c495b2b1798ede8bcc24185bc'
ADOPTION='3fc1e49e927a971a577d648dd9a7ff44ec7ac81552ea250026349d4f2e06d615'
SCIENCE='e08b8617bb30847b380372d8a3bdc3ac5b518136833295f998dc78603e6ad399'
EXECUTION='c93a5c605f7553bb336ba566a18bd87050f3d1e32d2b6320ee9ae15a51681224'
INVENTORY='b4ffdf5f3dc549e397f82db49290cdceb1ee024478cdc4e58365f5c537d144a9'
PRIOR_ROOT='abab0188adfa2d857316b23a282fd104c50dd282ca00ffed628b78b2accaea70'
BINDING='6bb73000047cc2659e63ab54cb0733e00e739cbf5afaed78011452724b263f93'

def rebound(path,pin,pairs):
    assert sha(path)==pin, str(path)
    text=path.read_text('utf8')
    assert len(dict(pairs))==len(pairs)
    for old,new in pairs: assert old in text, old
    substitutions=dict(pairs)
    pattern='|'.join(re.escape(x) for x in sorted(substitutions,key=len,reverse=True))
    result=re.sub(pattern,lambda match:substitutions[match.group()],text)
    compile(result,str(path),'exec')
    return result

def verify_seal(base,files):
    for name,pin in files.items():
        path=base/name
        assert path.resolve().is_relative_to(base.resolve())
        assert sha(path)==pin['sha256'] and path.stat().st_size==pin['bytes'], name

def source_changes():
    changes=read(BASE/'REBINDS.json'); functions={}; effective_pins={}
    for kind,spec in changes.items():
        path=ROOT/spec['source']; assert sha(path)==spec['sha256']
        original=path.read_text('utf8'); effective=original
        for before,after,count in spec['replacements']:
            assert effective.count(before)==count
            effective=effective.replace(before,after)
        assert hashlib.sha256(effective.encode()).hexdigest()==spec['effective_sha256']
        old={n.name:ast.get_source_segment(original,n) for n in ast.parse(original).body if isinstance(n,ast.FunctionDef)}
        new={n.name:ast.get_source_segment(effective,n) for n in ast.parse(effective).body if isinstance(n,ast.FunctionDef)}
        for name in {'build':['module','verify_inputs'],'panels':['need','aggregate_panels'],'verify_numeric':['verify_aggregate']}[kind]:
            assert old[name]==new[name], (kind,name)
            functions[kind+'.'+name]=hashlib.sha256(new[name].encode()).hexdigest()
        if kind=='build':
            assert old['displayed_cells'].replace('189','216')==new['displayed_cells']
            functions['build.displayed_cells_only_row_count189_to216']=hashlib.sha256(new['displayed_cells'].encode()).hexdigest()
        effective_pins[kind]=spec['effective_sha256']
    old=ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_seven_scenes_20261010/snapshot/SOURCE_BINDINGS.json'
    assert read(BASE/'snapshot/SOURCE_BINDINGS.json')['source_functions']==read(old)['source_functions']
    return dict(effective_source_sha256=effective_pins,unchanged_function_sha256=functions,strict_join_Full_normalizer_source_bindings_exact=True,builder_or_author_checker_called=False)

def run():
    assert sha(BASE/'ACTUAL_FILES_SHA256.json')==ACTUAL and sha(BASE/'ACTUAL_HANDOFF.json')==HANDOFF
    allfiles=read(BASE/'ACTUAL_FILES_SHA256.json')['files']; assert len(allfiles)==39
    verify_seal(BASE,allfiles)
    source_review=source_changes()
    connection_pairs=[
        ('celeba_mechanism_C60_root_arithmetic_review_20261010','celeba_mechanism_C70_independent_review_20261010'),
        ('7273c2f3340a1c62c3bb9874117b5bff0bbe447751e0621d2c737de8ff9189ec','dcb2f2cbb0e19490f07f7f9c41556feb0995074f7067c42f637dc37ea8c7371d'),
        ('INDEPENDENT_C60_SIX_SCENE','INDEPENDENT_C70_SEVEN_SCENE'),
        ('celeba_mechanism_valid_C_after60_20261010/inventory_actual170_','celeba_mechanism_valid_C_after70_20261010/inventory_actual180_'),
        ('celeba_mechanism_valid_C_after56_20261010/inventory_actual160_','celeba_mechanism_valid_C_after60_20261010/inventory_actual170_'),
        ('c28047e56778d6285140dcb22386410a0fc3dd3a803a9bd9aba182be373c1cef',INVENTORY),
        ('302dd45e9f05c646671d31d26775607af7a4fe70876fa1e60643939f972742f4','c28047e56778d6285140dcb22386410a0fc3dd3a803a9bd9aba182be373c1cef'),
        ('len(native)==170 and len(before)==160','len(native)==180 and len(before)==170'),
        ('celeba_mechanism_valid_C_after56_20261010/execution_candidate/backups/incremental_20261010T005530Z','celeba_mechanism_valid_C_after60_20261010/execution_candidate/backups/incremental_20261010T015546Z'),
        ('21b7f5beac762bf808415685817a4027e9da66640a7ef0e4c7d4ac74eeb1f05e','7e687f5a050aa0496b9a8c2b3606bd8787c138de6679e436dee3eb5b60df2dc8'),
        ('prior160_root_adoption_sha256','prior170_root_adoption_sha256'),
        ('minus_C_non-IID_F Flip_seed','minus_C_non-IID_FedSA_seed'),
        ("'F Flip'","'FedSA'"),('(160,10,170)','(170,10,180)'),
        ('prior_C60_proof_sha256','prior_C70_proof_sha256'),('prior120_record_provenance_reused','prior140_record_provenance_reused'),
        ('nonIID_F_Flip10_chain','nonIID_FedSA10_chain')]
    connections=rebound(PRIOR/'source_connections.py','9d191a725abf4dc613eb5fdb3568a6b66a354b9feaabc8bd50792f422c143aea',connection_pairs)
    cm=types.ModuleType('source_connections');cm.__file__=str(HERE/'source_connections.py')
    exec(compile(connections,cm.__file__,'exec'),cm.__dict__);sys.modules['source_connections']=cm
    pairs=[
        ('celeba_mechanism_C70_table_preparation_20261010','celeba_mechanism_C_eight_scenes_20261010'),
        ('three_view_C_six_scenes_20261010','three_view_C_seven_scenes_20261010'),
        ('758d49ac894a423f151f64cb0619f630bb6fdb697d9e8d16949b163ba09f336e',SOURCE),
        ('f2465c5e81291deca92535adcd0b0a29c49b3f76c2330927aa3db50925a9c742',PRIOR_ROOT),
        ("('non-IID','Benign'),('non-IID','F Flip')]","('non-IID','Benign'),('non-IID','F Flip'),('non-IID','FedSA')]"),
        ('cf81e43fd957b098ce8f72821b2530c4db6854cca8726c03d0f8e4c1a8084d2e',SNAPSHOT),
        ('7e687f5a050aa0496b9a8c2b3606bd8787c138de6679e436dee3eb5b60df2dc8',ADOPTION),
        ('len(source_files) == 12','len(source_files) == 13'),('original_C60_root_sha256','original_C70_root_sha256'),
        ('tmp/celeba_mechanism_C70_root_operations_20261010/C10_BINDING.json','tmp/celeba_mechanism_C_eight_scenes_20261010/C10_BINDING.json'),
        ('a50061f0f70102babe08ccac3151da5c308c20a04e79762110bf42dcb60ed9e1',BINDING),
        ('celeba_mechanism_valid_C_after60_20261010','celeba_mechanism_valid_C_after70_20261010'),
        ('ed8ecc84781b799e205c9e139dce8e753cb5545139b7e48ef23d5f8d07a50a77',SCIENCE),
        ('12c1b612d9c9697f6a537344ebace5aaa7c1bf5d7bbe760a3866168adca89896',EXECUTION),
        ('inventory_actual170_','inventory_actual180_'),
        ('c28047e56778d6285140dcb22386410a0fc3dd3a803a9bd9aba182be373c1cef',INVENTORY),
        ('ROOT_C_AFTER60_INCREMENT_','ROOT_C_AFTER70_INCREMENT_'),('(160,10,170)','(170,10,180)'),
        ('minus_C_non-IID_F Flip_seed','minus_C_non-IID_FedSA_seed'),('original160_unchanged','original170_unchanged'),
        ('== 140','== 160'),('== 120','== 140'),('[:120]','[:140]'),
        ('metric_checks == 1260 and count_checks == 3360','metric_checks == 1440 and count_checks == 3840'),
        ("len(panel['rows']) == 21","len(panel['rows']) == 24"),('len(errors) == 1134','len(errors) == 1296'),
        ('len(recomputed) == 189','len(recomputed) == 216'),
        ("!=('non-IID','F Flip')","!=('non-IID','FedSA')"),
        ("not x.startswith('| non-IID F Flip |')","not x.startswith('| non-IID FedSA |')"),
        ('len(old_lines)*3 == 486','len(old_lines)*3 == 567'),("len(paired[view]) == 70","len(paired[view]) == 80"),
        ('INDEPENDENT_C70_SEVEN_SCENE','INDEPENDENT_C80_EIGHT_SCENE'),
        ('unique_records=140,paired_models=70,complete_scenes=7','unique_records=160,paired_models=80,complete_scenes=8'),
        ('mean_SD_scalars_recomputed=1134,display_cells=567,count_metrics_recomputed=1260,confusion_count_checks=3360','mean_SD_scalars_recomputed=1296,display_cells=648,count_metrics_recomputed=1440,confusion_count_checks=3840'),
        ('old120_record_JSON_bytes_and_order_exact','old140_record_JSON_bytes_and_order_exact'),('old972_scalars_exact','old1134_scalars_exact'),('old486_display_cells_exact','old567_display_cells_exact'),
        ('other_three_nonIID_C_scenes_complete','other_two_nonIID_C_scenes_complete'),
        ('reviewer_authored_C70_table_builder','reviewer_authored_C80_table_builder'),('reviewer_authored_C70_numeric_verifier','reviewer_authored_C80_numeric_verifier'),
        ("source_review_sha256=sha(HERE/'SOURCE_REVIEW.json'),auxiliary_fixture_target_correction_sha256=sha(HERE/'SOURCE_BOUNDARY_CORRECTION.json'),","reviewer_prior_after70_source_review_sha256='bc73777fdba77ac5fba53203bf20993fef3087edc261a6c80f5e1d83f21f43e5',"),
        ("prior_independent_C60_reader_sha256='0a6ba5e3a033006a5bf1b7054bb3b4bfc8ad3bc1a084f8f5a0190d1eb5fa1979'","prior_independent_C70_reader_sha256='fbe27a120011d1266d23dd3bcdff93d9de0a5df746f2a571215c1ec09b138fb4'"),
        ('Seven-scene rows remain separate','Eight-scene rows remain separate'),
        ('Reviewer authored the reused exact10 evaluator source; reviewer did not author this C70 table builder/numeric verifier.','Reviewer reviewed the after70 evaluator source and authored its reused after60 parent; reviewer did not author this C80 table builder/numeric verifier.')]
    source=rebound(PRIOR/'review.py','fbe27a120011d1266d23dd3bcdff93d9de0a5df746f2a571215c1ec09b138fb4',pairs)
    rm=types.ModuleType('independent_C80_math');rm.__file__=str(HERE/'review.py')
    exec(compile(source,rm.__file__,'exec'),rm.__dict__)
    proof=rm.verify(SNAPSHOT,ADOPTION)
    snapshot=BASE/'snapshot'; records=read(snapshot/'records.json')['records']; coverage=read(snapshot/'coverage.json')
    for view,rows in coverage.items():
        assert view in rm.VIEWS and len(rows)==10
        for row in rows:
            expected=(row['distribution'],row['attack']) in rm.CELLS
            assert row['variant']=='minus_C' and row['complete']==expected and row['expected_n']==10
            assert row['n']==(10 if expected else 0) and row['seeds']==(list(range(91001,91011)) if expected else [])
    for record in records:
        assert len({record['checkpoint_sha256']})==1 and set(record['views'])==set(rm.VIEWS)
    handoff=read(BASE/'ACTUAL_HANDOFF.json'); tables=read(snapshot/'tables.json')
    assert (tables['unique_records'],tables['paired_models'],tables['complete_scenes'])==(160,80,8)
    assert tables['nonIID_complete_scenes']==['Benign','F Flip','FedSA'] and tables['full_nonIID_coverage'] is False
    assert handoff['eight_scene_aggregate'] is False and handoff['IID_seed_first_scene_count']==5
    assert handoff['negative_results_preserved'] and handoff['primary_endpoint']=='PENDING_AUTHOR'
    assert handoff['replay_devices']==proof['replay_devices'] and handoff['training_torch']==proof['training_torch']
    document=(BASE/'README.md').read_text('utf8')+(snapshot/'TABLES.md').read_text('utf8')
    for term in ['sampleSD(ddof1)','91001','validation','test','CPU/GPU','AEOD','six other controls','not compute an imbalanced eight-scene aggregate']:
        assert term in document,term
    assert proof['native_shared_identical_records']==160 and proof['paired_seed_metric_checks']==720
    assert sha(ROOT/'tmp/celeba_mechanism_C_after70_source_review_20261010/ROOT_INDEPENDENT_REVIEW.json')==proof['reviewer_prior_after70_source_review_sha256']
    proof.update(actual_handoff_sha256=HANDOFF,actual_delivery_seal_sha256=ACTUAL,actual_delivery_members_verified=39,
        minimal_source_difference_review=source_review,reviewer_authored_current_after70_evaluator_source=False,
        reviewer_reviewed_after70_evaluator_source=True,reviewer_authored_reused_after60_parent_source=True,
        independent_reader_effective_sha256=hashlib.sha256(source.encode()).hexdigest(),
        independent_connections_effective_sha256=hashlib.sha256(connections.encode()).hexdigest(),
        eight_scene_aggregate_present=False,coverage_complete_C_scenes=8,coverage_pending_C_scenes=2,findings=[],
        same_checkpoint_all_views_source_receipts_verified=True,
        auxiliary_review_failure_preserved='ROOT_ARITHMETIC_FAILURE.json',
        auxiliary_review_failure_scope='Before arithmetic: reviewer incorrectly required displayed_cells source bytes unchanged; legitimate row-count189-to216 metadata change now normalized explicitly. Candidate unchanged.')
    proof['reviewer_authored_C10_evaluator_source']=False
    verify_seal(BASE,allfiles)
    assert sha(BASE/'ACTUAL_FILES_SHA256.json')==ACTUAL
    return proof

if __name__=='__main__':
    try:
        output=HERE/'ROOT_ARITHMETIC_REVIEW.json';assert not output.exists()
        proof=run()
        with output.open('x',encoding='utf8',newline='\n') as f:json.dump(proof,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps(dict(status=proof['status'],path=str(output),sha256=sha(output),scalars=proof['mean_SD_scalars_recomputed'],cells=proof['display_cells'],maximum_error=proof['max_abs_difference'])))
    except BaseException as error:
        failure=HERE/'ROOT_ARITHMETIC_FAILURE.json'; number=2
        while failure.exists():
            failure=HERE/f'ROOT_ARITHMETIC_FAILURE_{number}.json';number+=1
        with failure.open('x',encoding='utf8',newline='\n') as f:json.dump(dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,author_source_modified=False),f,indent=2);f.write('\n')
        raise
