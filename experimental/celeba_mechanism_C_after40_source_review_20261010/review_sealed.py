"""One bounded offline evidence pass; final source adoption requires manual diff review."""
from pathlib import Path, PurePosixPath
import argparse, ast, hashlib, json, subprocess, sys, tarfile

sys.dont_write_bytecode = True
if sys.flags.optimize:
    raise RuntimeError('Optimized review is forbidden')
H = Path(__file__).resolve().parent
R = H.parents[1]
B = R/'tmp/celeba_mechanism_valid_C_after40_20261010'
P = R/'tmp/celeba_mechanism_valid_C_after36_20261010'
IDS = [f'minus_C_IID_Sp-DFA_seed{s}' for s in range(91001, 91008)]
FUNCTIONS = ['require','digest','canonical','read','save_new','load','cell',
             'full_reference','validate_variant_metadata','bind_runtime','reference_baseline_full']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())

def save(name, value):
    with (H/name).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')

def pins(root, rows):
    if isinstance(rows, dict):
        rows = [dict(v, path=k) for k,v in rows.items()]
    for row in rows:
        path = root/row['path']
        assert path.is_file() and sha(path) == row['sha256'], str(path)
        assert path.stat().st_size == row.get('size', row.get('bytes')), str(path)
    return len(rows)

def functions(path):
    source = path.read_text('utf8')
    return {n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}

def record_bytes(path):
    source = path.read_text('utf-8-sig')
    cursor = source.index('[',source.index('"records"'))+1
    result = []; decoder = json.JSONDecoder()
    while True:
        while source[cursor].isspace() or source[cursor] == ',': cursor += 1
        if source[cursor] == ']': return result
        value,end = decoder.raw_decode(source,cursor)
        result.append((value['id'],source[cursor:end])); cursor = end

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['package','science','execution','inventory']:
        parser.add_argument('--'+name+'-sha256',required=True)
    args = parser.parse_args()
    assert not (H/'SEALED_EVIDENCE.json').exists(), 'Single review invocation only'
    prep = read(H/'REVIEW_PREPARATION.json')
    pins(R,prep['parent_pins'])
    paths = {'package':B/'PACKAGE_SHA256.json','science':B/'FILES_SHA256.json',
             'execution':B/'execution_candidate/EXECUTION_SOURCE_SHA256.json',
             'inventory':B/'inventory_actual147_Full100refs.json'}
    for name,path in paths.items(): assert sha(path)==getattr(args,name+'_sha256')
    counts = {}
    for name in ['package','science','execution']:
        seal = read(paths[name]); counts[name]=pins(paths[name].parent,seal.get('members',seal.get('files')))
    counts['input_pins']=pins(R,read(B/'INPUT_PINS.json'))
    receipt=read(B/'PACKAGE_RECEIPT.json'); archive=B/receipt['source_archive']
    assert sha(archive)==receipt['source_archive_sha256']
    with tarfile.open(archive) as bundle:
        members=bundle.getmembers()
        assert len(members)==len(set(x.name for x in members))==receipt['archive_members']
        assert {x.name for x in members}==set(receipt['members'])
        for member in members:
            rel=PurePosixPath(member.name)
            assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
            data=bundle.extractfile(member).read(); pin=receipt['members'][member.name]
            assert len(data)==pin['bytes'] and hashlib.sha256(data).hexdigest()==pin['sha256']
            assert data==(B/member.name).read_bytes()
    before=read(P/'inventory_actual140_Full100refs.json'); inv=read(paths['inventory'])
    prior={r['id']:r for r in before['records']}; current={r['id']:r for r in inv['records']}
    assert len(prior)==140 and len(current)==len(inv['records'])==147
    assert set(current)-set(prior)==set(IDS) and inv['selected_replay_ids']==IDS
    assert {k:current[k] for k in prior}==prior
    assert [x for x in record_bytes(paths['inventory']) if x[0] in prior]==record_bytes(P/'inventory_actual140_Full100refs.json')
    assert inv['full_references']==before['full_references'] and len(inv['full_references'])==100
    assert inv['native_tolerance']==1e-12 and inv['views']==before['views']
    assert set(inv['excluded_prior_replay_ids'])==set(prior)
    for identity in IDS:
        row=current[identity]
        assert (row['variant'],row['distribution'],row['attack'],row['seed'])==('minus_C','IID','Sp-DFA',int(identity[-5:]))
        assert row['config']['ablation_component']=='C' and row['actual_alpha']==5000
        assert row['terminal_round']==70 and row['original_split']=='valid' and row['original_n_eval']==19867
        assert row['data_contract']['actual_train_rows']==162770
    oldfn,newfn=functions(P/'bridge.py'),functions(B/'bridge.py')
    assert all(oldfn[name]==newfn[name] for name in FUNCTIONS)
    assert (P/'execution_candidate/resource_extra.py').read_bytes()==(B/'execution_candidate/resource_extra.py').read_bytes()
    # Review only; original metadata fixture stops at bind_runtime, never imports scientific torch.
    command=[sys.executable,'-B',str(B/'check_prepared.py')]
    completed=subprocess.run(command,capture_output=True,timeout=180)
    (H/'METADATA_STDOUT.json').write_bytes(completed.stdout)
    (H/'METADATA_STDERR.log').write_bytes(completed.stderr)
    assert completed.returncode==0, completed.stderr.decode(errors='replace')
    metadata=json.loads(completed.stdout)
    assert metadata==read(B/'SELF_CHECK.json')
    assert metadata['worker_original_pre_bind_path_reached']==IDS
    assert metadata['refusal_count']>=69 and metadata['per_child_exact7_science_approval_positive']==7
    assert not metadata['CNN'] and not metadata['torch_imported'] and not metadata['numpy_imported']
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    save('SEALED_EVIDENCE.json',dict(status='AUTOMATED_SOURCE_EVIDENCE_PASS_PENDING_MANUAL_DIFF_NATIVE_CONNECTION_REVIEW',
        pins={k:sha(v) for k,v in paths.items()},counts=counts,source_tar_members=len(members),
        exact_selected_ids=IDS,old140_records_exact=True,old140_record_bytes_exact=True,
        Full100_references_exact=True,scientific_functions_byteexact=FUNCTIONS,
        metadata_check=metadata,actual_dispatch_authorized_by_this_review=False,
        native_archive_source_review_still_required=True,source_adoptable=False))

if __name__=='__main__':
    try: main()
    except BaseException as exc:
        import traceback
        if not isinstance(exc,SystemExit):
            save('REVIEW_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),retry=False))
        raise
