"""Future root call only: select exact10; preserve original v4 backup science."""
from pathlib import Path
import argparse, ast, hashlib, importlib.util, json, sys

TOOL = Path('/workspace/guardfed_checks/server_reactivation_20261009/evidence_v4.py')
TOOL_SHA = '3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
FIXED_IDS = [f'minus_F_non-IID_F Flip_seed{s}' for s in range(91001, 91011)]
OLD = '    delta = sorted(accepted - already)'
NEW = "    require(len(FIXED_IDS) == len(set(FIXED_IDS)) == 10 and FIXED_IDS == sorted(FIXED_IDS), 'Fixed exact10 order required')\n    require(set(FIXED_IDS) <= accepted - already, 'Missing, unfinished or previously backed-up fixed ID')\n    delta = list(FIXED_IDS)"


def derive_backup(text):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'backup')
    original = ast.get_source_segment(text, node)
    assert original.count(OLD) == 1
    adapted = original.replace(OLD, NEW)
    assert adapted.replace(NEW, OLD) == original
    compile(adapted, '<original backup; fixed-ID metadata selector only>', 'exec')
    return original, adapted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('inspection', 'ledger', 'archive'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    assert __debug__ and sys.platform == 'linux'
    raw = TOOL.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == TOOL_SHA
    spec = importlib.util.spec_from_file_location('_original_native_v4_fixed_scope', TOOL)
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    before, adapted = derive_backup(raw.decode('utf8'))
    namespace = dict(original.__dict__, FIXED_IDS=FIXED_IDS)
    exec(compile(adapted, str(TOOL) + ':FIXED_IDS_METADATA_ONLY', 'exec'), namespace)
    result = namespace['backup'](a.inspection.resolve(), a.ledger.resolve(), a.archive.resolve())
    assert result['accepted_new_ids'] == FIXED_IDS
    result.update(fixed_selected_ids=FIXED_IDS,
                  original_backup_function_sha256=hashlib.sha256(before.encode()).hexdigest(),
                  effective_backup_function_sha256=hashlib.sha256(adapted.encode()).hexdigest())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
