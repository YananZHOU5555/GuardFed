"""One-time scope-only derivation from the actually used finite FLGMM pipeline."""
from pathlib import Path
import ast, hashlib, json, subprocess, sys, difflib

H = Path(__file__).resolve().parent
R = H.parents[1]
O = R / 'tmp/celeba_flgmm_three_view_closed_batch_preparation_20261011'
NATIVE = 'tmp/fl_native_after54_20261011/ROOT_ADOPTION_REVIEW.json'
NATIVE_SHA = '66f8b578393b76fb8402a2f3c3d29ecbaf3f31aea4459371876c0f9f07e363ee'
PRIOR = 'tmp/celeba_flgmm_closed47_root_execution_20261011/ROOT_SCIENTIFIC_ADOPTION.json'
PRIOR_SHA = '22fc113add73814ae40accf563ce5f63bd63d50bcf53b7262bdb2b996b016dfe'

def write(name, text):
    with (H / name).open('x', encoding='utf8', newline='\n') as f:
        f.write(text)

def main():
    assert not (H / 'BOUND_EXACT13.json').exists()
    cmd = [sys.executable, '-B', str(H/'bind_exact13.py'), '--native-root', NATIVE, '--native-root-sha256', NATIVE_SHA]
    actual = subprocess.run(cmd, capture_output=True, text=True)
    write('BIND_COMMAND.json', json.dumps({'argv':cmd,'exit_code':actual.returncode},indent=2)+'\n')
    write('BIND.stderr', actual.stderr)
    assert actual.returncode == 0, actual.stderr
    bound = json.loads(actual.stdout)
    write('BOUND_EXACT13.json', json.dumps(bound,indent=2)+'\n')
    source = (O/'prepare.py').read_text(encoding='utf8')
    assert hashlib.sha256((O/'prepare.py').read_bytes()).hexdigest() == '67bc9d8fecba611d5dabda50b358c51d34ae1384086900cac3f03cf34c5f3096'
    a = source.index('    remember(ROOT44, ROOT44_SHA)')
    b = source.index('    # Registry contains only this finite pending batch.', a)
    phase = '''    bound = read(HERE / 'BOUND_EXACT13.json')
    assert bound['prior_three_view_root_sha256'] == PRIOR48_SHA
    remember(PRIOR48, PRIOR48_SHA)
    prior = read(PRIOR48)
    prior_ids = [x['id'] for x in prior['records']] + [x['id'] for x in prior['prior_interface_explicitly_reused']]
    assert len(prior_ids) == len(set(prior_ids)) == 48 and prior['root_adoption'] is True
    remember(ROOT57, ROOT57_SHA)
    latest = read(ROOT57)
    assert latest['accepted_total'] == 57 and latest['reused_separately'] == 4
    selected = bound['exact_ids']
    expected = [PREFIX + '_IID_Sp-DFA_seed' + str(s) + '_fullcoverage' for s in range(91007,91011)]
    expected += [PREFIX + '_non-IID_Benign_seed' + str(s) + '_fullcoverage' for s in range(91002,91011)]
    assert selected == expected and not set(selected).intersection(prior_ids)
    assert [rid for rid in latest['accepted_job_ids'] if rid not in set(prior_ids)] == selected
    for number, entry in enumerate(bound['sources']):
        p = ROOT / entry['root']
        remember(p, entry['root_sha256'])
        j = read(p)
        index = p.parent / 'RAW_STORAGE_INDEX.json'
        remember(index, entry['raw_index_sha256'])
        assert j['raw_storage_index_sha256'] == entry['raw_index_sha256']
        batch = Path(read(index)['raw_storage_root'])
        register(p, batch, j['accepted_new_ids'], number)
    assert list(records) == selected and len(records) == 13
    for x in bound['records']:
        assert records[x['id']]['identity']['checkpoint']['sha256'] == x['checkpoint_sha256']
    ordered = selected
'''
    source = source[:a]+phase+source[b:]
    source = source.replace("ROOT44 = ROOT / 'tmp/celeba_flgmm_fullcoverage_delta_after38_20261010/ROOT_ADOPTION_REVIEW.json'", "ROOT57 = ROOT / '"+NATIVE+"'")
    source = source.replace("ROOT44_SHA = 'f61a7fa480a62a9394d67a64533568b43e6491ee01d8e1f02005f35a42b6a510'", "ROOT57_SHA = '"+NATIVE_SHA+"'")
    source = source.replace("EXACT3 = ROOT / 'tmp/celeba_added_cnn_exact3_root_execution_20261010/ROOT_SCIENTIFIC_ADOPTION.json'", "PRIOR48 = ROOT / '"+PRIOR+"'")
    source = source.replace("EXACT3_SHA = '631d7ee2acf1cbe3453d849523e262f29ff5b354b5ddb56c60e5779e94364456'", "PRIOR48_SHA = '"+PRIOR_SHA+"'")
    source = source.replace('FLGMM_CLOSED44_PLUS4_MINUS_ADOPTED1_EXACT47_THREE_VIEW_CANDIDATE','FLGMM_NATIVE57_PLUS4_MINUS_ADOPTED48_EXACT13_THREE_VIEW_CANDIDATE')
    source = source.replace("'root_accepted_new_training_records': 44", "'root_accepted_new_training_records': 57").replace("'previous_three_view_accepted_in_scope': 1", "'previous_three_view_accepted_in_scope': 48").replace("'pending_three_view_records': 47", "'pending_three_view_records': 13")
    source = source.replace("'FL44_root_pin': pin(ROOT44)", "'FL57_root_pin': pin(ROOT57), 'prior48_root_pin': pin(PRIOR48)")
    source = source.replace('exact47 authorization', 'exact13 authorization')
    a = source.index("    save('SKIP_AND_REUSE.json'")
    b = source.index("    save('ACCEPTED_NATIVE_IDENTITIES.json'",a)
    source = source[:a]+'''    save('SKIP_AND_REUSE.json', {'strict_training_scope': 61, 'root57': pin(ROOT57),
        'prior48_root': pin(PRIOR48), 'prior48_ids': prior_ids, 'pending_exact_ids': selected,
        'prior48_unchanged': True, 'observed_but_unaccepted_training_ids_included': 0,
        'screen_reuse_pending': 0, 'future_coverage_or_recipe_selection': False})
'''+source[b:]
    source = source.replace(", 'screen_reuse_package_sha256': read(screen_root.parent/'PARTIAL_ACCEPTANCE.json')['package_sha256']", '')
    source = source.replace("'strict_scope': 48, 'skipped': 1, 'pending_exact': 47, 'reused_screen_pending': 4", "'strict_scope': 61, 'prior_accepted': 48, 'pending_exact': 13, 'reused_screen_pending': 0")
    ast.parse(source)
    oldregister = next(n for n in ast.walk(ast.parse((O/'prepare.py').read_text())) if isinstance(n,ast.FunctionDef) and n.name=='register')
    newregister = next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name=='register')
    assert ast.dump(oldregister)==ast.dump(newregister)
    write('prepare.py',source)
    original=(O/'candidate.py').read_text(encoding='utf8')
    assert hashlib.sha256((O/'candidate.py').read_bytes()).hexdigest() == 'd8412c4aa782e767afd174d92b7568f0153afb5215f8f65fbff178d84960d1be'
    candidate=original.replace('exact47','exact13').replace('Exact47','Exact13').replace('Exactly47','Exactly13').replace('EXACT47','EXACT13')
    candidate=candidate.replace('FLGMM_CLOSED44_PLUS4_MINUS_ADOPTED1_EXACT13_THREE_VIEW_CANDIDATE','FLGMM_NATIVE57_PLUS4_MINUS_ADOPTED48_EXACT13_THREE_VIEW_CANDIDATE')
    candidate=candidate.replace("== 47", "== 13").replace("['root_accepted_new_training_records'] == 44", "['root_accepted_new_training_records'] == 57").replace("['previous_three_view_accepted_in_scope'] == 1", "['previous_three_view_accepted_in_scope'] == 48")
    candidate=candidate.replace('FL44_root_pin','FL57_root_pin').replace('f61a7fa480a62a9394d67a64533568b43e6491ee01d8e1f02005f35a42b6a510',NATIVE_SHA).replace('Wrong root44','Wrong root57')
    candidate=candidate.replace("    require('FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91005_fullcoverage' not in m['exact_ids'], 'Existing three-view checkpoint refused')", "    require(m['prior48_root_pin']['sha256'] == '"+PRIOR_SHA+"', 'Wrong prior48')\n    expected_ids = ['FLGMM_Tg20_L2.0_lr0.001_IID_Sp-DFA_seed'+str(s)+'_fullcoverage' for s in range(91007,91011)] + ['FLGMM_Tg20_L2.0_lr0.001_non-IID_Benign_seed'+str(s)+'_fullcoverage' for s in range(91002,91011)]\n    require(m['exact_ids'] == expected_ids, 'Exact13 ID whitelist changed')")
    candidate=candidate.replace('celeba_flgmm_three_view_closed_batch_20261011/outputs','fl_three_view_after48_20261011/outputs').replace('guardfed_flgmm_closed_exact13_valid.lock','guardfed_flgmm_after48_exact13_valid.lock')
    ast.parse(candidate)
    write('candidate.py',candidate)
    write('SCOPE_DIFF.patch', ''.join(difflib.unified_diff((O/'prepare.py').read_text().splitlines(True),source.splitlines(True),fromfile='original47/prepare.py',tofile='after48/prepare.py'))+''.join(difflib.unified_diff(original.splitlines(True),candidate.splitlines(True),fromfile='original47/candidate.py',tofile='after48/candidate.py')))
    print(json.dumps({'status':'SCOPE_DERIVED_NO_SCIENCE','register_AST_exact':True,'pending':13}))

if __name__=='__main__': main()
