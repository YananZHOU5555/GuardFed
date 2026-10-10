"""One local source preparation; no Git, F writes, network or science execution."""
from pathlib import Path
import difflib, hashlib, json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'tmp/publication50_source_preparation_20261010'
SCOPE = 'CLOSED251_COMPACT_EXACT3_INTERFACE3_WITH_PRESERVED_CROSS_PLATFORM_AUDIT_FAILURE'
sha = lambda b: hashlib.sha256(b).hexdigest()
read = lambda p: json.loads(p.read_bytes())


def save(name, d):
    p = HERE / name
    assert not p.exists()
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')


def replace(s, a, b):
    assert s.count(a) == 1, a
    return s.replace(a, b)


def main():
    assert sha((OLD/'FILES_SHA256.json').read_bytes()) == '1e62b8afc414e1b893e5d40a068f6d037d127190a52dab35180769d90f0b1a44'
    seal = read(OLD/'FILES_SHA256.json')
    for name, pin in seal['files'].items():
        b = (OLD/name).read_bytes()
        assert sha(b) == pin['sha256'] and len(b) == pin['bytes']
    previous = read(OLD/'PREPARED_MANIFEST.json')
    groups = read(OLD/'EXTRA_CANDIDATES.json')['adopted_extras']
    assert len(groups) == 3 and sum(len(g['files']) for g in groups) == 37
    execution = 'tmp/celeba_added_cnn_exact3_root_execution_20261010/'
    scientific = 'tmp/celeba_added_cnn_exact3_scientific_acceptance_20261010/'
    names = [execution+n for n in (
        'ROOT_SCIENTIFIC_ADOPTION.json','ROOT_OFFSERVER_SOURCE_REVIEW.json','TRANSPORT_VERIFICATION.json',
        'SCIENCE_COMMAND.json','SCIENCE_EXIT.json','OFFSERVER_ROOT_DIAGNOSTIC.json','LINUX_EXACT_ROOT_CHECK.json',
        'LINUX_EXACT_ROOT_EXIT.json','OFFSERVER_ARRAY_REFIT_CHECK.json','PRESERVED_WINDOWS_FAILURE.json',
        'linux_saved_root_remote.py','transport_remote.py')]
    names += ['tmp/'+n for n in (
        'adopt_added_cnn_exact3_root_20261010.py','verify_added_cnn_exact3_root_20261010.py',
        'verify_added_cnn_exact3_linux_saved_root_20261010.py','verify_added_cnn_exact3_offserver_arrays_20261010.py',
        'diagnose_added_cnn_exact3_offserver_root_20261010.py','transport_added_cnn_exact3_root_20261010.py')]
    names += [scientific+n for n in (
        'FILES_SHA256.json','HANDOFF.json','SOURCE_PINS.json','saved_binding.py','verify_offserver.py',
        'OBSERVED_METADATA_CHECKS.json','REPORT.md','observe_once.py','COMMAND.json','OBSERVATION.json',
        'FINAL_COMMAND.json','FINAL_OBSERVATION.json')]
    assert len(names) == len(set(names)) == 30
    added = []
    for name in names:
        p = ROOT/name
        b = p.read_bytes()
        assert len(b) < 1_000_000 and p.suffix in {'.json','.py','.md'}
        added.append(dict(source=name,destination='experimental/'+name[4:],sha256=sha(b),bytes=len(b)))
    proof_path = execution+'ROOT_SCIENTIFIC_ADOPTION.json'
    proof_sha = '631d7ee2acf1cbe3453d849523e262f29ff5b354b5ddb56c60e5779e94364456'
    assert sha((ROOT/proof_path).read_bytes()) == proof_sha
    facts = {'/interface_records_accepted':3,'/mechanism_three_view_cutoff_unchanged':251,
        '/final_test':False,'/cross_platform_audit_failure_preserved':True,'/root_adoption':True,
        '/Linux_whole_original_saved_check_pass':True,'/Windows_whole_saved_check_pass':False,
        '/Windows_original_array_refit_block_pass':True}
    proof = read(ROOT/proof_path)
    assert all(proof[k[1:]] == v for k,v in facts.items())
    failed = read(ROOT/(execution+'PRESERVED_WINDOWS_FAILURE.json'))
    trace = failed['original_stderr_text'].encode('utf-8')
    original_stderr = (ROOT/(execution+'SCIENCE_STDERR.txt')).read_bytes()
    assert len(original_stderr) == 1287 and sha(original_stderr) == failed['original_stderr_sha256']
    assert trace == original_stderr.replace(b'\r\n',b'\n') and len(trace) == 1275
    closed = dict(status='EXACT_ROOT_NAMED_COMPACT_EXTRAS_CLOSED_NO_WILDCARDS',
        inherited_three_groups=groups, inherited_file_count=37,
        scientific_root_binding=dict(path=proof_path,sha256=proof_sha,expect=facts),
        scientific_files=added, scientific_file_count=30,
        omitted='Saved arrays, archives and receipts stay on F/server; source scientific/preflight seals are compact subsets, not full Git mirrors.')
    save('EXACT_EXTRA_FILES.json',closed)
    d = previous
    d['scope'] = SCOPE
    d['exact3_interface_records_separate'] = 3
    d['bindings']['exact3_scientific'] = closed['scientific_root_binding']
    entries = {e['source']:e for e in d['files']}
    refs = {e['source']:e for e in d['parent_recovery_references']}
    for e in [e for g in groups for e in g['files']] + added:
        if e['source'] in entries: assert entries[e['source']] == e
        entries[e['source']] = e
        refs.pop(e['source'],None)
    # Every role is carried as a file, even if a byte-identical parent blob exists.
    for pin in d['bindings'].values():
        if pin and pin['path'] in refs:
            e = refs.pop(pin['path']); e.pop('parent_commit')
            entries[e['source']] = e
    d['files'] = sorted(entries.values(),key=lambda e:e['source'])
    d['parent_recovery_references'] = sorted(refs.values(),key=lambda e:e['source'])
    d['allowed_paths'] = sorted(set(entries)|set(refs)|set(d['pending_mutable']))
    d['extra_files_policy'] = 'Closed exact 37 inherited + 30 root-named scientific files; no arbitrary extras input'
    save('PREPARED_MANIFEST.json',d)
    pins = read(OLD/'SOURCE_PINS.json')
    pins['previous50_seal'] = dict(path=(OLD/'FILES_SHA256.json').relative_to(ROOT).as_posix(),sha256=sha((OLD/'FILES_SHA256.json').read_bytes()))
    pins['previous50_manifest'] = dict(path=(OLD/'PREPARED_MANIFEST.json').relative_to(ROOT).as_posix(),sha256=sha((OLD/'PREPARED_MANIFEST.json').read_bytes()))
    save('SOURCE_PINS.json',pins)
    template = read(OLD/'ROOT_INPUTS_TEMPLATE.json')
    template.pop('adopted_extras');template.pop('gate_final_receipt')
    template['status'] = 'TEMPLATE_NOT_ACTUAL_ROOT_INPUTS'
    save('ROOT_INPUTS_TEMPLATE.json',template)

    original = (OLD/'publish_increment50.py').read_text(encoding='utf-8')
    code = replace(original,'CLOSED251_COMPACT_NO_BULK_SOURCE_ONLY_EXACT3',SCOPE)
    code = replace(code,"'exact3_metadata','exact3_independent','current_state'}", "'exact3_metadata','exact3_independent','current_state','exact3_scientific'}")
    code = replace(code,"'current_state':{'/celeba_mechanism_v1/scientific_results_strictly_accepted':251,", "'exact3_scientific':"+repr(facts)+",\n 'current_state':{'/celeba_mechanism_v1/scientific_results_strictly_accepted':251,")
    code = replace(code,'def plan(d):', '''def closed_scope(d):
    prepared=read(HERE/'PREPARED_MANIFEST.json')
    packet=read(HERE/'FILES_SHA256.json')['files']
    own=HERE.relative_to(ROOT).as_posix()+'/'
    fixed={e['source']:e for e in prepared['files']+prepared['parent_recovery_references']}
    for name,pin in packet.items():
        fixed[own+name]=dict(source=own+name,destination=destination(own+name),**pin)
    seal=HERE/'FILES_SHA256.json'
    fixed[own+seal.name]=dict(source=own+seal.name,destination=destination(own+seal.name),
                            sha256=sha(seal.read_bytes()),bytes=seal.stat().st_size)
    allowed=set(fixed)|set(prepared['pending_mutable'])
    files={e['source']:e for e in d['files']};refs={e['source']:e for e in d['parent_recovery_references']}
    assert not set(files)&set(refs) and set(files)|set(refs)==allowed
    assert len(files)==len(d['files']) and len(refs)==len(d['parent_recovery_references'])
    assert set(d['allowed_paths'])==allowed and len(d['allowed_paths'])==len(allowed)
    for name,e in fixed.items():
        actual=(files|refs)[name]
        assert all(actual[k]==e[k] for k in ('source','destination','sha256','bytes')), 'Closed file differs: '+name
    assert set(d['bindings'])==ROLES
    for role,pin in d['bindings'].items():
        assert pin and pin['path'] in files, 'Every role proof must be a real snapshot file: '+role
        assert files[pin['path']]['sha256']==pin['sha256']
        if role!='current_state':assert pin==prepared['bindings'][role], 'Closed role binding differs: '+role
    assert d['exact3_interface_records_separate']==3


def plan(d):''')
    code = replace(code,"    available={e['source']:e for e in entries+refs}","    closed_scope(d)\n    available={e['source']:e for e in entries}")
    code = replace(code,"        scope=d['scope'],accepted=ACCEPTED,test_started=False", "        scope=d['scope'],exact3_interface_records_separate=3,accepted=ACCEPTED,test_started=False")
    code = replace(code,"    ns['save'](out/'SOURCE_SNAPSHOT.json'", "    assert all((snapshot/pin['path']).is_file() for pin in d['bindings'].values())\n    ns['save'](out/'SOURCE_SNAPSHOT.json'")
    code = replace(code,"    d=read(path); assert sha(path.read_bytes())==expected", "    d=read(path); assert sha(path.read_bytes())==expected\n    closed_scope(d)")
    code = replace(code,"            _,b=frozen_source(pin['path'],d['allowed_paths'])", "            assert (root/pin['path']).is_file(), 'Role must exist in F source snapshot: '+role\n            _,b=frozen_source(pin['path'],d['allowed_paths'])")
    assert 'import publication50' not in code and 'source_preparation_20261010' not in code
    (HERE/'publish_increment50.py').write_text(code,encoding='utf-8',newline='\n')
    (HERE/'verify_increment50.py').write_bytes((OLD/'verify_increment50.py').read_bytes())
    diff = ''.join(difflib.unified_diff(original.splitlines(True),code.splitlines(True),fromfile='sealed50/publish_increment50.py',tofile='actual_scope50/publish_increment50.py'))
    (HERE/'SOURCE_DIFF.patch').write_text(diff,encoding='utf-8',newline='\n')
    print(json.dumps(dict(status='SOURCE_PREPARED_NOT_FINALIZED_OR_PUBLISHED',files=len(entries),bytes=sum(e['bytes'] for e in entries.values()),roles=len(d['bindings']),inherited_extra_files=37,new_scientific_compact_files=30,mutable_pending=8)))


if __name__=='__main__':main()
