"""Metadata/chain/source fixtures only; no archive, arrays, model, NumPy or CNN."""
import ast
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
import transport as t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    q, plan, pins = t.source(ROOT / 'tmp/celeba_mechanism_remaining_evaluation_v2_20261010', 'a3461e20592cd2bda3d53c7215fed377bbaa693360ba1abe02aa87fe8aa6fc03')
    ids = plan['remaining620_ids'][:2]; refusals = {}
    def refuse(label, call):
        try: call()
        except (ValueError, FileNotFoundError) as exc: refusals[label] = str(exc)
        else: raise AssertionError('Expected refusal: ' + label)
    assert t.select_delta(plan, ids, []) == ids
    for label, selected, prior in [('duplicate_ID', [ids[0], ids[0]], []), ('already_transported', [ids[0]], [ids[0]]),
                                   ('old180', [plan['excluded180_ids'][0]], []), ('Full', [plan['full_references'][0]['id']], []),
                                   ('foreign', ['FOREIGN'], []), ('reordered', list(reversed(ids)), [])]:
        refuse(label, lambda selected=selected, prior=prior: t.select_delta(plan, selected, prior))
    fixture = HERE / 'metadata_only_fixtures'; fixture.mkdir()
    chain = fixture / 'chain'; chain.mkdir()
    assert t.previous(chain, None, None, plan) == ([], None)
    batch = chain / 'batch0'; batch.mkdir()
    receipt = batch / 'backup_receipt.json'
    t.save(receipt, dict(TEST_ONLY_NO_ARCHIVE_EXISTS=True, source_seal_sha256=pins['remaining620_v2_seal_sha256'],
        accepted_offserver=0, accepted_new_ids=[ids[0]], all_transported_ids=[ids[0]], previous_backup_receipt_sha256=None, previous_receipt=None))
    latest = dict(receipt=str(receipt), receipt_sha256=t.sha(receipt), all_transported_ids=[ids[0]], accepted_offserver=0)
    t.save(chain / 'TRANSPORT_LATEST.json', latest)
    assert t.previous(chain, receipt, t.sha(receipt), plan)[0] == [ids[0]]
    refuse('wrong_previous_SHA', lambda: t.previous(chain, receipt, '0' * 64, plan))
    (chain / 'orphan_partial').mkdir()
    refuse('orphan_partial_export', lambda: t.previous(chain, receipt, t.sha(receipt), plan))
    unbound = fixture / 'unbound'; (unbound / 'runs' / ids[0]).mkdir(parents=True)
    refuse('future_unbound_output', lambda: t.closed(q, plan, unbound, ids[0], 'TEST_ONLY', {}, None))
    fake = {n: {'id': ids[0]} for n in ('REMOTE_COMPLETE.json', 'binding.json', 'strict_acceptance.json', 'receipt.json', 'bridge_receipt.json')}
    fake.update({'inventory.json': {}, 'APPROVED.json': {}})
    fake['REMOTE_COMPLETE.json'].update(status='REMOTE_STRICT_CLOSED_PENDING_OFFSERVER', accepted_offserver=1)
    paths = [Path(n) for n in ('receipt.json', 'bridge_receipt.json', 'validation_predictions.npz', 'strict_acceptance.json')]
    with patch.object(Path, 'iterdir', return_value=iter(paths)), patch.object(t, 'read', side_effect=lambda p: fake[Path(p).name]):
        refuse('offserver_acceptance_forged', lambda: t.closed(q, plan, unbound, ids[0], 'TEST_ONLY', {}, None))
    pin = pins['dependencies']['writer']; body = t.writer_body(ROOT / pin['local_relative'], pin['sha256'])
    compile(body, '<original writer fixture>', 'exec')
    refuse('writer_source_drift', lambda: t.writer_body(ROOT / pin['local_relative'], '0' * 64))
    pin = pins['dependencies']['saved_verifier']; body = t.saved_verifier_body(ROOT / pin['local_relative'], pin['sha256'])
    original = (ROOT / pin['local_relative']).read_text(encoding='utf-8')
    original = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == 'verify')
    projected = ast.parse(body).body[0]
    original_normalized, projected_normalized = copy.deepcopy(original), copy.deepcopy(projected)
    # Only the inventory SHA expression and descriptive subset string differ.
    projected_normalized.body[0] = original_normalized.body[0]
    for n in ast.walk(projected_normalized):
        if isinstance(n, ast.Constant) and n.value == 'This explicit remaining620 transport subset of validation terminal replays; not final test or complete900 mechanism evidence.':
            n.value = 'This exact10-scope subset of validation terminal replays; not final test or complete900 mechanism evidence.'
    assert ast.dump(original_normalized, include_attributes=False) == ast.dump(projected_normalized, include_attributes=False)
    refuse('saved_verifier_source_drift', lambda: t.saved_verifier_body(ROOT / pin['local_relative'], '0' * 64))
    refuse('local_bulk_on_E', lambda: t.F_guard(HERE, 0))
    volume = t.F_guard(Path('F:/YananResearchStorage/GuardFed'), 0)
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    for p in HERE.glob('*.py'): ast.parse(p.read_text(encoding='utf-8'))
    t.save(HERE / 'SELF_CHECK.json', dict(status='METADATA_CHAIN_AND_ORIGINAL_BODY_SOURCE_CHECK_PASS_NO_EXECUTION',
        frozen_v2_source_seal_sha256=pins['remaining620_v2_seal_sha256'], exact620_metadata_positive=True,
        explicit_prior_chain_positive=True, refusals=refusals, archive_writer_original_source_extracted=True,
        original_saved_checker_AST_exact_except_inventory_SHA_and_label=True, actual_fresh_F_volume=volume,
        fixture_receipts_are_TEST_ONLY_not_real_transport=True, real_exports=0, archives_written=0, arrays_written=0,
        models_written=0, accepted_offserver=0, NumPy_imported=False, Torch_imported=False, CNN=0, SSH=0))
    print(json.dumps({'status': 'PASS', 'refusals': len(refusals), 'real_exports': 0, 'CNN': 0}))


if __name__ == '__main__': main()
