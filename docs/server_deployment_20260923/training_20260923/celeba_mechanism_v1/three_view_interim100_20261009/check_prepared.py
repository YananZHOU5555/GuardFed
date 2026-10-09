"""Actual inventory/source guard checks only; never generate unaccepted tables."""
import ast
import copy
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
import build as b

def main():
    pins=b.read(b.H/'INPUTS.json')['files']
    for name,pin in pins.items():
        p=b.R/name;b.need(b.sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],'Pinned input drift '+name)
    i92=b.read(b.P92/'inventory_actual92_Full100refs.json');i100=b.read(b.P100/'inventory_actual100_Full100refs.json')
    actual=b.scope(i92,i100);refusals=[]
    mutations={
        'missing_seed':lambda j:j['records'].pop(),
        'duplicate_id':lambda j:j['records'].__setitem__(-1,copy.deepcopy(j['records'][-2])),
        'wrong_selected_id':lambda j:j['selected_replay_ids'].__setitem__(0,'minus_C_IID_Benign_seed91001'),
        'already_closed_id':lambda j:j['selected_replay_ids'].__setitem__(0,'minus_U_IID_Benign_seed91001'),
        'Full_as_new':lambda j:j['selected_replay_ids'].__setitem__(0,j['full_references'][0]['id']),
        'changed_prior_root':lambda j:j['records'][0]['data_contract'].__setitem__('root_image_ids_sha256','0'*64),
        'changed_Full':lambda j:j['full_references'][0].__setitem__('checkpoint_sha256','0'*64),
        'tolerance_changed':lambda j:j.__setitem__('native_tolerance',1e-6),
        'test_split':lambda j:j['records'][-1].__setitem__('original_split','test'),
        'short_round':lambda j:j['records'][-1].__setitem__('terminal_round',69),
    }
    for name,mutate in mutations.items():
        bad=copy.deepcopy(i100);mutate(bad)
        try:b.scope(i92,bad)
        except ValueError as error:refusals.append({'case':name,'error':str(error)})
        else:raise AssertionError('Guard accepted '+name)
    try:b.adoption_gate({},b.H/'UNAPPROVED.json','0'*64)
    except ValueError as error:refusals.append({'case':'unreviewed_proof','error':str(error)})
    else:raise AssertionError('Unreviewed proof accepted')
    old=b.module('old_join_source',b.OLD/'build_final.py')
    original_inputs=b.R/'tmp/celeba_mechanism_three_view_paired71_20261009/inputs.py'
    ns={'need':b.need,'VIEWS':b.VIEWS,'hashlib':b.hashlib,'json':json}
    extracted=old.funcs(original_inputs,['receipt_identity','normalized'],ns)
    extracted.update(old.funcs(b.P100/'bridge.py',['canonical'],ns))
    for name in ('build.py','panels.py','verify_numeric.py','check_prepared.py'):
        ast.parse((b.H/name).read_text(encoding='utf-8'))
    proof=dict(status='PREPARED_SOURCE_AND_EXACT_SCOPE_GUARDS_PASS_NO_NEW8_ACCEPTANCE',input_pins=len(pins),exact_new_ids=b.EXPECTED,prior_records_unchanged=92,Full100_references_unchanged=True,refusals=refusals,extracted_unchanged_scientific_functions=extracted,new100_tables_generated=False,actual_after92_adoption_provided=False,torch_imported='torch' in sys.modules,new_CNN=False,new_fit=False)
    b.write(b.H/'PREPARED_CHECKS.json',proof);print(json.dumps(dict(status=proof['status'],refusals=len(refusals),torch_imported=proof['torch_imported'])))

if __name__=='__main__':main()
