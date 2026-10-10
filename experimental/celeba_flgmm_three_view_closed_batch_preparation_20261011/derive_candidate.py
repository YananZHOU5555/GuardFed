"""Derive the original successful runtime with finite-scope metadata bindings only."""
from pathlib import Path
import difflib
import hashlib

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'celeba_added_cnn_three_view_gate_preparation_20261010/candidate.py'
EXPECTED = '3f98138ffe25b6800fd99d2eabc85402308bd3e21da275341a6b77367b7dc190'


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:100]
    return text.replace(old, new)


def main():
    before = OLD.read_text(encoding='utf-8')
    assert hashlib.sha256(OLD.read_bytes()).hexdigest() == EXPECTED
    after = replace_once(before, 'Exact-three valid interface candidate.', 'Exact47 already-accepted FLGMM valid three-view candidate.')
    start = after.index('    expected = [', after.index('def validate_manifest'))
    end = after.index("    require(m['views']", start)
    after = after[:start] + """    expected = [('FLGMM', rid) for rid in m['exact_ids']]
    require(len(expected) == len(set(m['exact_ids'])) == 47, 'Frozen exact47 only')
    require([(r['method'], r['id']) for r in m['records']] == expected, 'Exact47 order drift')
    require(m['scope'] == 'FLGMM_CLOSED44_PLUS4_MINUS_ADOPTED1_EXACT47_THREE_VIEW_CANDIDATE', 'Wrong scope')
    require(m['root_accepted_new_training_records'] == 44 and m['separately_reused_screen_checkpoints'] == 4
            and m['previous_three_view_accepted_in_scope'] == 1 and m['pending_three_view_records'] == 47, 'Wrong closed scope')
    require(m['FL44_root_pin']['sha256'] == 'f61a7fa480a62a9394d67a64533568b43e6491ee01d8e1f02005f35a42b6a510', 'Wrong root44')
    require('FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91005_fullcoverage' not in m['exact_ids'], 'Existing three-view checkpoint refused')
    require(m['proof_registry_sha256'] == sha(HERE / 'PROOF_PINS.json'), 'Registry changed')
    registry = read(HERE / 'PROOF_PINS.json')['chains_by_id']
    require(set(registry) == set(m['exact_ids']), 'Only source-bound accepted exact IDs')
""" + after[end:]
    after = replace_once(after,
        "        require(r['runtime_artifacts']['model']['sha256'] == r['identity']['checkpoint']['sha256'], 'Checkpoint identity drift')",
        "        require(r['runtime_artifacts']['model']['sha256'] == r['identity']['checkpoint']['sha256'], 'Checkpoint identity drift')\n        require(registry[r['id']]['records'][r['id']] == r['identity']['original_artifact_pins'], 'Wrong strict chunk or artifact registration')")
    after = replace_once(after,
        "str(bridge.HERE / 'PROOF_PINS.json'): HERE / 'originals/PROOF_PINS.json'",
        "str(bridge.HERE / 'PROOF_PINS.json'): HERE / 'PROOF_PINS.json'")
    after = replace_once(after,
        '    bridge.digest, bridge.read_pin = digest, read_pin',
        """    # Root-bound exact chunks, including legacy screen4, remain separate.
    registry = read(HERE / 'PROOF_PINS.json')
    bridge.PROOFS_SHA256 = m['proof_registry_sha256']
    identity_source = (HERE / 'originals/bridge.py').read_text(encoding='utf-8')
    identity_node = next(n for n in ast.parse(identity_source).body
                         if isinstance(n, ast.FunctionDef) and n.name == 'identity_record')
    changed = {'chain': 0, 'job_sha': 0}
    class Register(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == 'chain':
                    require(ast.unparse(node.value) == "manifest['chains'][method]", 'Unexpected original registry selection')
                    changed['chain'] += 1
                    node.value = ast.parse("manifest['chains_by_id'][job_id]", mode='eval').body
                elif node.targets[0].id == 'job_sha':
                    require(ast.unparse(node.value) == "evidence['job']['sha256']", 'Unexpected original job binding')
                    changed['job_sha'] += 1
                    node.value = ast.parse('original_job_binding(job_id, chain, evidence, job)', mode='eval').body
            return node
    rebound = ast.fix_missing_locations(Register().visit(ast.Module(body=[identity_node], type_ignores=[])))
    require(changed == {'chain': 1, 'job_sha': 1}, 'Only two metadata registration assignments')
    def original_job_binding(rid, chain, evidence, job):
        pin = chain.get('original_job_by_id', {}).get(rid)
        if pin is None:
            return evidence['job']['sha256']
        # Original source job was CRLF; archived runtime copy is LF, values exact.
        require(read_pin(pin) == job, 'Original/stored screen job value drift')
        stored = path_for(evidence['job']['path']).read_bytes()
        original = path_for(pin['path']).read_bytes()
        require(b'\\r' not in stored and stored.replace(b'\\n', b'\\r\\n') == original, 'Not the exact accepted CRLF/LF pair')
        return pin['sha256']
    bridge.original_job_binding = original_job_binding
    exec(compile(rebound, '<original-bridge-two-metadata-bindings>', 'exec'), bridge.__dict__)
    original_identity = bridge.identity_record
    def checked_identity(method, rid):
        require(method == 'FLGMM' and rid in m['exact_ids'], 'Unregistered batch ID')
        chain = registry['chains_by_id'][rid]
        receipt = read_pin(chain['strict_receipt'])
        require(receipt['acceptance_sha256'] == chain['strict']['sha256']
                and receipt['inventory_sha256'] == chain['raw_index']['sha256'], 'Strict/member receipt link changed')
        if 'root_adoption' in chain:
            root_index = read_pin(chain['root'])
            adopted = read_pin(chain['root_adoption'])
            require(root_index['root_adoption_sha256'] == chain['root_adoption']['sha256']
                    and adopted['strict_receipt_sha256'] == chain['strict']['sha256']
                    and adopted['offserver_proof_sha256'] == chain['offserver']['sha256'], 'Legacy screen root adoption link changed')
        return original_identity(method, rid)
    bridge.identity_record = checked_identity
    bridge.digest, bridge.read_pin = digest, read_pin""")
    for old, new in [
        ('Exactly three added-CNN valid interface gates; not full100, method ranking, final primary endpoint, test, or CUDA equivalence',
         'Exactly47 accepted FLGMM checkpoints; finite partial coverage, not full100, method ranking, final primary endpoint, test, or CUDA equivalence'),
        ('ROOT_AUTHORIZED_EXACT3_VALID_IMAGE_INTERFACE_GATE', 'ROOT_AUTHORIZED_FLGMM_CLOSED_EXACT47_THREE_VIEW'),
        ('ROOT_LINUX_EXACT3_PREFLIGHT_PASS', 'ROOT_LINUX_FLGMM_EXACT47_PREFLIGHT_PASS'),
        ('/workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010/outputs',
         '/workspace/guardfed_checks/celeba_flgmm_three_view_closed_batch_20261011/outputs'),
        ('/tmp/guardfed_added_cnn_exact3_valid_gate.lock', '/tmp/guardfed_flgmm_closed_exact47_valid.lock'),
        ('EXACT3_VALID_INTERFACE_PASS_NOT_ROOT_ADOPTED', 'FLGMM_EXACT47_THREE_VIEW_PASS_NOT_ROOT_ADOPTED')]:
        after = replace_once(after, old, new)
    compile(after, 'candidate.py', 'exec')
    with (HERE / 'candidate.py').open('x', encoding='utf-8', newline='\n') as f:
        f.write(after)
    with (HERE / 'SOURCE_DIFF.patch').open('x', encoding='utf-8', newline='\n') as f:
        f.write(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
            fromfile=str(OLD), tofile='candidate.py')))


if __name__ == '__main__':
    main()
