"""Metadata/source guards only; never creates a future adoption or computes tables."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
import build as b


def main():
    for name in ['build.py','panels.py','verify_numeric.py','check_prepared.py']:
        ast.parse((b.H/name).read_text('utf8'))
    inputs = b.read(b.H/'INPUTS.json')
    for name,pin in inputs['files'].items():
        assert b.sha(b.R/name) == pin['sha256'] and (b.R/name).stat().st_size == pin['bytes']
    assert b.sha(b.OLD/'ROOT_VERIFICATION.json') == '0d047461186184718f58c5423c546137930b287d3bd7cff8e06eacd6e28ad73f'
    assert b.sha(b.OLD/'FINAL_FILES_SHA256.json') == '68ee93369bb5d45bcb2063fd4bb1b1995e2a7b05abee4e78b9cc8a174a801f31'
    assert b.sha(b.OLD/'snapshot/records.json') == '6cbd01fe8abdc304a388b25d14c0e972e07bd80d195b7ff0b8e57a42c81aeaf8'
    for row in b.read(b.OLD/'FINAL_FILES_SHA256.json')['members']:
        assert b.sha(b.OLD/row['path']) == row['sha256']
    prior = b.read(b.C11/'inventory_actual112_Full100refs.json')
    actual = b.read(b.C8/'inventory_actual120_Full100refs.json')
    assert len(b.scope(prior,actual)) == 20
    refuses = []
    def reject(name,call):
        try: call()
        except (ValueError,KeyError,FileNotFoundError): refuses.append(name); return
        raise AssertionError('Not refused: '+name)
    for name,change in [
        ('selected7',lambda x:x.update(selected_replay_ids=b.EXPECTED[:-1])),
        ('old112_mutation',lambda x:x['records'][0].update(terminal_round=69)),
        ('Full_reference_mutation',lambda x:x['full_references'][0].update(checkpoint_sha256='0'*64)),
        ('boundary_mutation',lambda x:x.update(excluded_prior_replay_ids=[])),
        ('native_tolerance_mutation',lambda x:x.update(native_tolerance=1e-6))]:
        bad = copy.deepcopy(actual); change(bad)
        reject(name,lambda:b.scope(prior,bad))
    reject('old_C11_adoption_not_C8',lambda:b.adoption_gate(b.PRIOR,b.sha(b.PRIOR)))
    reject('missing_C8_adoption',lambda:b.adoption_gate(b.C8/'execution_candidate/backups/NOT_CLOSED/ROOT_ADOPTION_REVIEW.json','0'*64))
    result = subprocess.run([sys.executable,'-B',str(b.H/'build.py'),'--output',str(b.H/'NOT_GENERATED')],capture_output=True,text=True)
    assert result.returncode == 2 and not (b.H/'NOT_GENERATED').exists()
    refuses.append('CLI_requires_actual_adoption_and_SHA')
    old_panel = (b.OLD/'panels.py').read_text('utf8')
    expected = old_panel.replace("expected=[('IID','Benign')]","expected=[('IID','Benign'),('IID','F Flip')]").replace('Only the exact C IID Benign ten-shared-seed scene is publishable','Only exact C IID Benign10 and F Flip10 scenes are publishable')
    assert expected == (b.H/'panels.py').read_text('utf8')
    old_verify = (b.OLD/'verify_numeric.py').read_text('utf8')
    new_verify = (b.H/'verify_numeric.py').read_text('utf8')
    # Arithmetic body is copied unchanged; only its row-count assertion changes.
    old_body = old_verify.split('    errors=[]',1)[1].split('    assert len(errors)',1)[0]
    new_body = new_verify.split('    errors=[]',1)[1].split('    assert len(errors)',1)[0]
    assert old_body.replace('assert len(rows)==3','assert len(rows)==6') == new_body
    old_counts = old_verify.split('    metric_checks=0;count_checks=0',1)[1].split('    assert metric_checks',1)[0]
    new_counts = new_verify.split('    metric_checks=0;count_checks=0',1)[1].split('    assert metric_checks',1)[0]
    assert old_counts == new_counts
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    assert not (b.H/'snapshot').exists()
    print(json.dumps(dict(status='PREPARED_SOURCE_METADATA_CHECKS_PASS_NO_TABLES_OR_FUTURE_PROOF',source_pins=len(inputs['files']),actual_native_controls=20,exact_future_C8_ids=b.EXPECTED,refusals=refuses,arithmetic_body_unchanged=True,count_metric_formula_unchanged=True,planned_unique_records=40,planned_statistics=324,planned_display_cells=162,planned_count_metrics=360,old_Benign216_regression_not_new_evidence=True,actual_future_C8_proof_created=False,new_result_generated=False,torch_imported=False,numpy_imported=False),indent=2))


if __name__ == '__main__': main()
