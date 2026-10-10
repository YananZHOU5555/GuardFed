"""Pure source checks for the final bootstrap/failure-evidence changes; no Torch."""
import ast
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from bridge_adapter import digest, read, require
from evaluate_remaining import save_new

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    bindings = read(HERE / 'SOURCE_BINDINGS.json')
    for relative, pin in bindings['input_pins'].items():
        path = ROOT / relative
        require(path.stat().st_size == pin['bytes'] and digest(path) == pin['sha256'], 'Original input changed: ' + relative)
    parsed = {p.name: ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in HERE.glob('*.py')}
    original = (ROOT / 'tmp/celeba_mechanism_valid_incremental_next11_20261009/prepare.py').read_text(encoding='utf-8')
    assignment = [n for n in ast.walk(ast.parse(original)) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'record' for t in n.targets)]
    require(len(assignment) == 1 and (HERE / 'record_constructor.py').read_text(encoding='utf-8') == ast.get_source_segment(original, assignment[0]) + '\n', 'Original record constructor source changed')
    functions = {n.name: n for n in parsed['evaluate_remaining.py'].body if isinstance(n, ast.FunctionDef)}
    worker = functions['worker']; text = ast.unparse(worker)
    calls = [n for n in ast.walk(worker) if isinstance(n, ast.Call)]
    def line(name, argument=None):
        found = [n.lineno for n in calls if ast.unparse(n.func) == name
                 and (argument is None or n.args and ast.unparse(n.args[0]) == argument)]
        require(len(found) == 1, 'Unique source call expected: ' + name + str(argument))
        return found[0]
    require(line('load', "'remaining620_original_v3_bootstrap'") < line('evidence.validators') < line('evidence.accept_new')
            < line('save_new', "task / 'binding.json'") < line("runtime['replay_one']"), 'Bootstrap/strict/once-binding/inference order changed')
    require('set_num_interop_threads' not in text and 'set_num_threads' not in text, 'Do not initialize thread pools a second time')
    require("torch.get_num_threads() == 8" in text and "torch.get_num_interop_threads() == 1" in text, 'Original8/1 CPU bootstrap not checked')
    top = parsed['evaluate_remaining.py'].body[-1]
    require(isinstance(top, ast.If) and "locals().get('a')" in ast.unparse(top)
            and "RUNTIME.is_dir() and" not in ast.unparse(top), 'Approved pre-attempt failures must retain structured evidence')
    require('torch' not in sys.modules, 'Pure source check imported Torch')
    save_new(HERE / 'SOURCE_CHECK.json', dict(status='PURE_SOURCE_BOOTSTRAP_AND_EVIDENCE_CHECK_PASS',
        parsed_python_files=list(parsed), original_input_pins_unchanged=len(bindings['input_pins']), original_constructor_source_exact=True,
        original_bootstrap_precedes_strict_tensor_checks=True, interop_initialization_delegated_once_to_original_v2=True,
        original_strict_precedes_atomic_binding_and_CNN=True, approved_pre_attempt_failures_preserved=True,
        source_checked_sha256=digest(HERE / 'evaluate_remaining.py'), CNN=0, Torch_imported=False))
    print(json.dumps({'status': 'PASS', 'source_only': True, 'CNN': 0}))


if __name__ == '__main__':
    main()
