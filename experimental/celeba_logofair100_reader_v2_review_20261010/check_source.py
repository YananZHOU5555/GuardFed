"""Pure source/path review: never import the binder or query a volume/process."""
import ast, hashlib, json, ntpath, types
from pathlib import Path, PureWindowsPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'tmp/celeba_logofair100_reader_v2_20261010/bind_with_read_guard.py'
OPS = ROOT / 'tmp/celeba_logofair100_root_operations_20261010'
META = ROOT / 'tmp/celeba_logofair_fullcoverage_20261010/metadata.py'
EXPECTED = 'd8158c1ab988e10f650aab994a2815aee15e74d9af8fdbef5e4c14e28702ad1e'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def require(ok, message):
    if not ok:
        raise AssertionError(message)

class WindowsPath(PureWindowsPath):
    def resolve(self):
        return type(self)(ntpath.normpath(str(self)))

def main():
    require(sha(SOURCE) == EXPECTED, 'Reader source drift')
    require(sha(OPS/'FILES_SHA256.json') == '1f27eafe56693216ca30f6bd03e92f4c9e3c9e0b14d0248a28e6b29db9a7514f', 'Old operations seal drift')
    seal = json.loads((OPS/'FILES_SHA256.json').read_bytes())
    for name, pin in seal['files'].items():
        p = OPS/name
        require(sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes'], 'Old member drift: '+name)
    tree = ast.parse(SOURCE.read_text(encoding='utf8'))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    guard = next(n for n in run.body if isinstance(n, ast.FunctionDef) and n.name == 'read_guard')
    calls = []
    def original_bulk(path, required):
        calls.append((str(path), required))
        require(required >= 0, 'Original negative-size refusal')
        return ('original-path', {'fresh': True, 'required': required})
    counts = {'read_checks': 0, 'fresh_positive_write_checks': 0}
    ns = dict(Path=WindowsPath, storage_root=WindowsPath('F:/YananResearchStorage/GuardFed'),
        module=types.SimpleNamespace(require=require), original_bulk=original_bulk,
        counts=counts, volume={'initial': True})
    exec(compile(ast.Module(body=[guard], type_ignores=[]), str(SOURCE), 'exec'), ns)
    read_guard = ns['read_guard']
    p, proof = read_guard('F:/YananResearchStorage/GuardFed/inputs/model.pt', 0)
    require(str(p).startswith('F:') and proof is ns['volume'] and not calls, 'Read path must not query volume')
    refused = []
    for path in ('E:/GuardFed/a', 'C:/a', 'F:/other/a', 'F:/YananResearchStorage/GuardFed',
                 'F:/YananResearchStorage/GuardFed/../GuardFed2/a', '//host/share/a'):
        try:
            read_guard(path, 0)
        except AssertionError:
            refused.append(path)
        else:
            raise AssertionError('Unsafe read accepted: '+path)
    for amount in (1, 1024*1024, 16*1024*1024):
        result = read_guard('F:/YananResearchStorage/GuardFed/stage001', amount)
        require(result[1] == {'fresh': True, 'required': amount}, 'Original write proof lost')
    try:
        read_guard('F:/YananResearchStorage/GuardFed/stage001', -1)
    except AssertionError:
        pass
    else:
        raise AssertionError('Negative required bytes bypassed original guard')
    require([c[1] for c in calls] == [1, 1024*1024, 16*1024*1024, -1], 'Write arguments changed')
    metadata_tree = ast.parse(META.read_text(encoding='utf8'))
    bind_node = next(n for n in metadata_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'bind')
    original_ns = {'bulk_path': original_bulk, 'sentinel': object()}
    exec(compile(ast.Module(body=[bind_node], type_ignores=[]), str(META), 'exec'), original_ns)
    original = original_ns['bind']
    clone_globals = dict(original.__globals__, bulk_path=read_guard)
    clone = types.FunctionType(original.__code__, clone_globals, original.__name__, original.__defaults__, original.__closure__)
    require(clone.__code__ is original.__code__, 'Bind bytecode changed')
    require({k for k in original.__globals__ if original.__globals__[k] is not clone.__globals__[k]} == {'bulk_path'}, 'Other bind global changed')
    require('module.run(a)' not in ast.unparse(run), 'Original root mutation flow was re-entered')
    require('module.bulk_path =' not in ast.unparse(run), 'Original module storage helper changed')
    require('original_bulk(path, required)' in ast.unparse(guard), 'Positive write delegation absent')
    require("bulk_path(out, 16 * 1024 * 1024)" in ast.unparse(bind_node), 'Original bind write boundary absent')
    report = dict(status='PASS_SOURCE_READY_ONLY_ROOT_CANCELLATION_AND_LIVE_EMPTY_STAGE_STILL_REQUIRED',
        source_sha256=sha(SOURCE), input_pins={str(p.relative_to(ROOT)).replace('\\','/'): sha(p)
            for p in (SOURCE, OPS/'FILES_SHA256.json', OPS/'prepare_inputs_and_bind_approval.py', OPS/'common.py', META)},
        original_operations_members_verified=len(seal['files']), positive_read_path_checks=1,
        unsafe_read_path_refusals=len(refused), original_positive_write_delegations=3,
        negative_size_original_refusal=True, original_bind_same_code_object=True,
        only_bind_global_changed='bulk_path', original_root_run_not_invoked=True,
        original_common_save_globals_untouched=True,
        findings=[],
        boundaries=['This is source/path-only review, not actual cancellation, volume health, bind or100-record acceptance.',
            'Root must independently verify PID133008 stopped and stage001 absent, and supply externally SHA-pinned actual cancellation/inputs/approval.',
            'Read-only requests reuse initial volume proof; actual byte digests/mapping verification still fail on missing/detached/mismatched files. The cached proof is not a live mount claim.',
            'All original positive write boundaries remain fresh. Original metadata.bind guards its16MiB stage write once before the write batch; this is not a new per-file volume check.',
            'The continuation leaves prior approved inputs/attempt001 unchanged; partial new-stage failure remains non-resumable without separate review.',
            'New continuation has no structured exception handler; root command stderr/returncode and partial directory must be preserved on error.'],
        executions=dict(bind=False, cancellation=False, volume_query=False, process_query=False,
            original_script_import=False, CNN=False, fit=False, network=False, Git=False))
    target = HERE/'REVIEW.json'
    with target.open('x', encoding='utf8', newline='\n') as f:
        json.dump(report, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps(dict(report=str(target), report_sha256=sha(target), checker_sha256=sha(Path(__file__)), source_sha256=sha(SOURCE))))

if __name__ == '__main__':
    main()
