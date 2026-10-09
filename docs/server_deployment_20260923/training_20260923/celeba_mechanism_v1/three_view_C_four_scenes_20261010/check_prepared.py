"""Source/metadata checks only: no statistics, future proof or inference output."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
sys.dont_write_bytecode = True
import build as b


def main():
    for name in ['build.py','panels.py','verify_numeric.py','check_prepared.py']:
        ast.parse((b.H/name).read_text('utf8'))
    inputs = b.read(b.H/'INPUTS.json')
    for name,pin in inputs['files'].items():
        assert b.sha(b.R/name)==pin['sha256'] and (b.R/name).stat().st_size==pin['bytes']
    root = b.read(b.OLD/'ROOT_VERIFICATION.json')
    assert root['status']=='ROOT_C30_THREE_SCENE_THREE_VIEW_TABLES_ADOPTED'
    assert root['unique_records']==60 and root['mean_SD_scalars_recomputed']==486
    old_text=(b.OLD/'snapshot/records.json').read_text('utf8')
    old_spans=b.record_spans(old_text)
    assert len(old_spans)==60 and b.record_spans(json.dumps(b.read(b.OLD/'snapshot/records.json'),ensure_ascii=False,indent=2,allow_nan=False)+'\n')==old_spans
    for name,pin in b.read(b.OLD/'ACTUAL_FILES_SHA256.json')['files'].items():
        assert b.sha(b.OLD/name)==pin['sha256']
    prior=b.read(b.C28/'inventory_actual136_Full100refs.json')
    current=b.read(b.C4/'inventory_actual140_Full100refs.json')
    controls=b.scope(prior,current)
    assert len(controls)==40 and len([r for r in controls.values() if r['attack']=='S-DFA'])==10
    refusals=[]
    def reject(name,call):
        try:call()
        except (ValueError,KeyError,FileNotFoundError):refusals.append(name);return
        raise AssertionError('Not rejected: '+name)
    for name,change in [
        ('missing_fourth',lambda x:x.update(selected_replay_ids=b.EXPECTED[:-1])),
        ('duplicate_fourth',lambda x:x.update(selected_replay_ids=b.EXPECTED+[b.EXPECTED[0]])),
        ('old136_mutation',lambda x:x['records'][0].update(terminal_round=69)),
        ('Full_reference_mutation',lambda x:x['full_references'][0].update(checkpoint_sha256='0'*64)),
        ('prior_boundary_mutation',lambda x:x.update(excluded_prior_replay_ids=[])),
        ('tolerance_mutation',lambda x:x.update(native_tolerance=1e-6)),
        ('wrong_scene',lambda x:next(r for r in x['records'] if r['id']==b.EXPECTED[-1]).update(attack='FedSA')),
        ('terminal69',lambda x:next(r for r in x['records'] if r['id']==b.EXPECTED[0]).update(terminal_round=69)),
        ('test_split',lambda x:next(r for r in x['records'] if r['id']==b.EXPECTED[0]).update(original_split='test'))]:
        bad=copy.deepcopy(current);change(bad);reject(name,lambda:b.scope(prior,bad))
    reject('old_after28_adoption',lambda:b.adoption_gate(b.PRIOR,b.sha(b.PRIOR)))
    missing=b.C4/'execution_candidate/backups/NOT_CLOSED/ROOT_ADOPTION_REVIEW.json'
    reject('missing_actual_after36_adoption',lambda:b.adoption_gate(missing,'0'*64))
    # Explicit process-memory fixture checks only the future gate shape, never saved.
    fixture=dict(status='ROOT_C_AFTER36_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS',
        prior_three_view_models=136,accepted_new=4,cumulative_three_view_models=140,
        accepted_new_ids=b.EXPECTED,original136_unchanged=True,source_scope_complete=True,
        negative_results_preserved=True,prior136_root_adoption_sha256=b.sha(b.PRIOR),
        new_training=0,new_Full_inference=0,test_inference=False)
    actual_read,actual_sha=b.read,b.sha
    with patch.object(b,'read',lambda p:fixture if Path(p)==missing else actual_read(p)),patch.object(b,'sha',lambda p:'1'*64 if Path(p)==missing else actual_sha(p)):
        assert b.adoption_gate(missing,'1'*64)==fixture
        for key,value in [('status','PREPARED_NOT_APPROVED'),('accepted_new',2),('accepted_new_ids',b.EXPECTED[:2]),('original136_unchanged',False),('source_scope_complete',False),('negative_results_preserved',False),('prior136_root_adoption_sha256','0'*64),('new_Full_inference',1),('test_inference',True)]:
            saved=fixture[key];fixture[key]=value
            reject('adoption_'+key,lambda:b.adoption_gate(missing,'1'*64));fixture[key]=saved
        reject('wrong_actual_adoption_sha',lambda:b.adoption_gate(missing,'2'*64))
    run=subprocess.run([sys.executable,'-B',str(b.H/'build.py'),'--output',str(b.H/'NOT_GENERATED')],capture_output=True,text=True)
    assert run.returncode==2 and not (b.H/'NOT_GENERATED').exists()
    refusals.append('CLI_requires_actual_adoption_and_SHA')
    old_panel=(b.OLD/'panels.py').read_text('utf8')
    expected=old_panel.replace("expected=[('IID','Benign'),('IID','F Flip'),('IID','FedSA')]","expected=[('IID','Benign'),('IID','F Flip'),('IID','FedSA'),('IID','S-DFA')]").replace('Only exact C IID Benign10, F Flip10 and FedSA10 scenes are publishable','Only exact C IID Benign10, F Flip10, FedSA10 and S-DFA10 scenes are publishable')
    assert expected==(b.H/'panels.py').read_text('utf8')
    old=(b.OLD/'verify_numeric.py').read_text('utf8');new=(b.H/'verify_numeric.py').read_text('utf8')
    body=lambda s:s.split('    errors=[]',1)[1].split('    assert len(errors)',1)[0]
    assert body(old).replace('len(rows)==9','len(rows)==12')==body(new)
    counts=lambda s:s.split('    metric_checks=0;count_checks=0',1)[1].split('    assert metric_checks',1)[0]
    assert counts(old)==counts(new)
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules and not (b.H/'snapshot').exists()
    assert not missing.exists()
    print(json.dumps(dict(status='PREPARED_METADATA_SOURCE_PASS_NO_TABLES_NO_FUTURE_PROOF',
        pins=len(inputs['files']),actual_native_controls=40,prior_three_view_accepted=136,
        exact_pending_replay_ids=b.EXPECTED,positive_future_gate_shape_fixture_in_memory_only=True,
        refusals=refusals,refusal_count=len(refusals),arithmetic_body_unchanged=True,
        confusion_formula_unchanged=True,planned_unique_table_records=4*10*2,
        planned_mean_sd_scalars=4*3*3*3*3*2,planned_display_cells=4*3*3*3*3,
        planned_receipt_metrics=4*10*2*3*3,planned_confusion_checks=4*10*2*3*2*4,
        previously_partial_S_DFA6_requires_actual_C4=True,old60_records_and486_scalars_and243_cells_regression_required=True,
        actual_future_root_proof_created=False,new_statistical_result_generated=False,
        Torch=False,CNN=False),indent=2))


if __name__=='__main__':main()
