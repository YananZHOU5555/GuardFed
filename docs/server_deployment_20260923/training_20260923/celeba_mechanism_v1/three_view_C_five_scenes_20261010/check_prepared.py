"""Metadata fixtures only. Reads sealed native150 metadata, never calculates C50 tables."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
sys.dont_write_bytecode=True
import build as b


def main():
    for name in ['build.py','panels.py','verify_numeric.py','check_prepared.py']:
        ast.parse((b.H/name).read_text('utf8'))
    inputs=b.read(b.H/'INPUTS.json')
    for name,pin in inputs['files'].items():
        assert b.sha(b.R/name)==pin['sha256'] and (b.R/name).stat().st_size==pin['bytes']
    root=b.read(b.OLD/'ROOT_VERIFICATION.json')
    assert root['status']=='ROOT_C40_FOUR_SCENE_THREE_VIEW_TABLES_ADOPTED'
    assert root['unique_records']==80 and root['mean_SD_scalars_recomputed']==648 and root['display_cells']==324
    spans=b.record_spans((b.OLD/'snapshot/records.json').read_text('utf8'))
    assert len(spans)==80 and spans==b.record_spans(json.dumps(b.read(b.OLD/'snapshot/records.json'),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    # Actual sealed native metadata; no future replay acceptance assumed.
    prior=b.read(b.C40/'inventory_actual147_Full100refs.json'); fixture=b.read(b.C3/'inventory_actual150_Full100refs.json')
    assert len(prior['records'])==147 and len(fixture['records'])==150
    assert len(b.scope(prior,fixture))==50
    refusals=[]
    def reject(label,fn):
        try:fn()
        except (ValueError,KeyError,FileNotFoundError):refusals.append(label);return
        raise AssertionError('Not rejected: '+label)
    for label,change in [
        ('missing_C3',lambda x:x.update(selected_replay_ids=b.EXPECTED[:-1])),
        ('duplicate_C3',lambda x:x.update(selected_replay_ids=b.EXPECTED+[b.EXPECTED[0]])),
        ('old147_changed',lambda x:x['records'][0].update(terminal_round=69)),
        ('Full_reference_changed',lambda x:x['full_references'][0].update(checkpoint_sha256='0'*64)),
        ('native_tolerance_changed',lambda x:x.update(native_tolerance=1e-6)),
        ('wrong_variant',lambda x:next(r for r in x['records'] if r['id']==b.EXPECTED[-1]).update(variant='minus_U')),
        ('wrong_attack',lambda x:next(r for r in x['records'] if r['id']==b.EXPECTED[-1]).update(attack='S-DFA')),
        ('test_split',lambda x:next(r for r in x['records'] if r['id']==b.EXPECTED[-1]).update(original_split='test')),
        ('round69',lambda x:next(r for r in x['records'] if r['id']==b.EXPECTED[-1]).update(terminal_round=69)),
        ('excluded_prior_changed',lambda x:x.update(excluded_prior_replay_ids=[]))]:
        bad=copy.deepcopy(fixture);change(bad);reject(label,lambda:b.scope(prior,bad))
    reject('old_C7_adoption_wrong_namespace',lambda:b.adoption_gate(b.PRIOR,b.sha(b.PRIOR)))
    missing=b.C3/'execution_candidate/backups/NOT_ACTUAL/ROOT_ADOPTION_REVIEW.json'
    reject('missing_actual_C3_adoption',lambda:b.adoption_gate(missing,'0'*64))
    # Explicit in-memory schema fixture, not an approval/acceptance document.
    proof=dict(status='ROOT_C_AFTER47_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS',prior_three_view_models=147,accepted_new=3,cumulative_three_view_models=150,accepted_new_ids=b.EXPECTED,original147_unchanged=True,source_scope_complete=True,negative_results_preserved=True,prior147_root_adoption_sha256=b.sha(b.PRIOR),science_seal_sha256='1'*64,execution_seal_sha256='1'*64,all_native_differences_zero=True,server_strict_bound_in_saved_receipts=True,new_training=0,new_Full_inference=0,test_inference=False)
    real_read,real_sha=b.read,b.sha
    def fake_read(path):
        if path==missing:return proof
        if path==b.C3/'FILES_SHA256.json':return {'members':[]}
        return real_read(path)
    def fake_sha(path):
        return '1'*64 if Path(path).is_relative_to(b.C3) else real_sha(path)
    with patch.object(b,'read',fake_read),patch.object(b,'sha',fake_sha):
        assert b.adoption_gate(missing,'1'*64)==proof
        for key,value in [('status','PREPARED_NOT_APPROVED'),('accepted_new',7),('accepted_new_ids',b.EXPECTED[:2]),('original147_unchanged',False),('science_seal_sha256','2'*64),('execution_seal_sha256','2'*64),('all_native_differences_zero',False),('server_strict_bound_in_saved_receipts',False),('source_scope_complete',False),('negative_results_preserved',False),('prior147_root_adoption_sha256','2'*64),('new_Full_inference',1),('test_inference',True)]:
            saved=proof[key];proof[key]=value;reject('adoption_'+key,lambda:b.adoption_gate(missing,'1'*64));proof[key]=saved
        reject('external_adoption_SHA_mismatch',lambda:b.adoption_gate(missing,'2'*64))
    run=subprocess.run([sys.executable,'-B',str(b.H/'build.py'),'--output',str(b.H/'NOT_GENERATED')],capture_output=True,text=True)
    assert run.returncode==2 and not (b.H/'NOT_GENERATED').exists();refusals.append('CLI_actual_proof_required')
    old=(b.OLD/'verify_numeric.py').read_text('utf8');new=(b.H/'verify_numeric.py').read_text('utf8')
    body=lambda s:s.split('    errors=[]',1)[1].split('    assert len(errors)',1)[0]
    assert body(old).replace('len(rows)==12','len(rows)==15')==body(new)
    counts=lambda s:s.split('    metric_checks=0;count_checks=0',1)[1].split('    assert metric_checks',1)[0]
    assert counts(old)==counts(new)
    oldpanel=(b.R/'tmp/celeba_mechanism_three_view_C_four_scenes_prepared_20261010/panels.py').read_text('utf8')
    transformed=oldpanel.replace("('IID','S-DFA')]","('IID','S-DFA'),('IID','Sp-DFA')]").replace('and S-DFA10 scenes','S-DFA10 and Sp-DFA10 scenes')
    assert (b.H/'panels.py').read_text('utf8').split('\n\ndef aggregate_panels',1)[0]==transformed
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules and not (b.H/'snapshot').exists() and not missing.exists()
    print(json.dumps(dict(status='PREPARED_METADATA_SOURCE_PASS_NO_C50_STATISTICS',pins=len(inputs['files']),actual_old_records=80,exact_future_ids=b.EXPECTED,refusals=refusals,refusal_count=len(refusals),future_fixture_in_memory_only=True,sealed_native150_input_read=True,actual_C3_replay_accepted=False,scene_arithmetic_and_confusion_body_unchanged=True,planned_records=100,planned_scene_scalars=810,planned_scene_cells=405,planned_receipt_metrics=900,planned_group_count_checks=2400,planned_separate_seed_first_scalars=162,old80_648_324_regression_required=True,new_statistical_results=0,CNN=0),indent=2))


if __name__=='__main__':main()
