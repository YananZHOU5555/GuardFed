"""Read source/compact evidence pins only; never execute scientific table code."""
import ast
import hashlib
import json
from pathlib import Path
import py_compile
import runpy
import tempfile

H = Path(__file__).resolve().parent
R = H.parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    import source_adapter as adapter
    pins = json.loads((H / 'SOURCE_PINS.json').read_bytes())
    for rel, pin in pins.items():
        p = R / rel
        assert p.stat().st_size == pin['bytes'] and sha(p) == pin['sha256'], rel
    original = runpy.run_path(str(adapter.A80 / 'source_adapter.py'))
    contracts = json.loads((H / 'SOURCE_ADAPTATIONS.json').read_bytes())
    report = {}
    with tempfile.TemporaryDirectory(prefix='A90_source_compile_') as tmp:
        for name in ('build.py', 'finish.py', 'verify_saved.py'):
            before = original['mapped'](name)
            after = adapter.mapped(name)  # Includes exact reversible byte check.
            compile(after, name + ' [scope only]', 'exec')
            py_compile.compile(str(H / name), cfile=str(Path(tmp) / (name + 'c')), doraise=True)
            old_ast, new_ast = ast.parse(before), ast.parse(after)
            old_functions = {n.name: n for n in old_ast.body if isinstance(n, ast.FunctionDef)}
            new_functions = {n.name: n for n in new_ast.body if isinstance(n, ast.FunctionDef)}
            assert old_functions.keys() == new_functions.keys()
            unchanged = []
            for function in old_functions.keys() - {'main'}:
                assert ast.dump(old_functions[function]) == ast.dump(new_functions[function]), function
                unchanged.append(function)
            # Same arithmetic entry points; only scope/count/source metadata are adapted.
            scientific_calls = {'panels', 'verify', 'aggregate_panels', 'verify_aggregate'}
            calls = lambda tree: [ast.dump(n) for n in ast.walk(tree)
                                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                                  and n.func.attr in scientific_calls]
            assert calls(old_ast) == calls(new_ast)
            if name == 'build.py':
                main_before, main_after = old_functions['main'], new_functions['main']
                source_loop = lambda n: next(x for x in n.body if isinstance(x, ast.For)
                                             and isinstance(x.target, ast.Tuple)
                                             and ast.unparse(x.target) == '(index, ids, adoption_path, index_path, batch_root)')
                assert ast.dump(source_loop(main_before)) == ast.dump(source_loop(main_after))
                assert "if r['distribution']=='IID' or (r['distribution']=='non-IID' and r['attack'] in ('Benign','F Flip','FedSA'))" in after
            if name == 'finish.py':
                assert 'aggregate_panels(iid_records, evidence)' in after
                assert "f.write((base['PREV']/'IID_SEED_FIRST.json').read_bytes())" in after
            report[name] = dict(mapped_source_sha256=hashlib.sha256(after.encode()).hexdigest(),
                                reversible_replacements=len(contracts[name]['replacements']),
                                unchanged_helper_AST=sorted(unchanged), original_science_calls_AST_exact=True)
        py_compile.compile(str(H / 'source_adapter.py'), cfile=str(Path(tmp) / 'adapter.pyc'), doraise=True)
    assert 'old_cells==648' in adapter.mapped('build.py')
    assert 'old1296_scalars_and648_cells_exact' in adapter.mapped('verify_saved.py')
    assert 'assert cells==810' in adapter.mapped('verify_saved.py')
    assert not (H / 'ROOT_BINDING.json').exists(), 'Source-only preparation must remain unbound'
    try:
        adapter.preflight('build.py')
    except ValueError as error:
        assert 'not bound' in str(error)
    else:
        raise AssertionError('Missing actual295 proof did not fail closed')
    assert not any((H / n).exists() for n in ('records.json', 'tables.json', 'TABLES.md', 'IID_SEED_FIRST.json'))
    print(json.dumps(dict(status='SOURCE_ONLY_PASS_NO_SCIENTIFIC_BUILD', sources=report,
                          dependency_pins_verified=len(pins), root295_bound=False,
                          missing_root295_rejected=True, new_inference=0, new_fit=0,
                          table_generation_executed=False), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
