"""Exact new-scene metadata contract; no numeric statistics or scientific execution."""
import re
from pathlib import PurePosixPath

IDS=['FLGMM_Tg20_L2.0_lr0.001_non-IID_S-DFA_seed'+str(s)+'_fullcoverage' for s in range(91002,91011)]


def require(ok,message):
    if not ok:raise ValueError(message)

def sha(value):
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None,'Missing/invalid actual SHA')

def relative(value):
    require(isinstance(value,str) and '\\' not in value and ':' not in value and not value.startswith('/'),'Repository-relative path required')
    require('..' not in PurePosixPath(value).parts and PurePosixPath(value).as_posix()==value,'Noncanonical path refused')

def validate_binding(binding):
    require(binding.get('root_adopted') is True,'Actual root90 adoption must precede table construction')
    for key in ('root90','candidate','transport'):relative(binding.get(key))
    for key in ('root90_sha256','candidate_seal_sha256','transport_sha256','native86_root_sha256'):sha(binding.get(key))
    require(binding['root90'].endswith('/ROOT_SCIENTIFIC_ADOPTION.json'),'Actual root adoption proof required')
    require(binding['transport'].endswith('/TRANSPORT_VERIFICATION.json'),'Actual verified F transport proof required')


def validate_root(root,old,binding):
    validate_binding(binding)
    require(root['root_adoption'] is True and root['FLGMM_total_three_view_records']==90 and root['new_three_view_records_accepted']==9,'Only actual90 accepted scope')
    require(root['native_training_records']==86 and root['original_screen_reuse_separate']==4,'Native86 plus four screen reuse required')
    require(root['prior_root_sha256']=='adc7726ca05561c6903ebb153e34fb3dc69a1df0118d87e6bf7dfa3e31655799','Actual parent81 required')
    require(root['native_root_sha256']==binding['native86_root_sha256'],'Native86 adopted source differs')
    validate_new_rows(root['new_records'])
    require(root['records']==old['records']+root['new_records'] and root['prior_interface_explicitly_reused']==old['prior_interface_explicitly_reused'],'Prior81 objects/order or exact new9 differ')
    require(root['Linux_whole_original_saved_check_pass'] is True and root['Linux_root_fit_verified'] is True and root['Windows_saved_outputs_audit_pass'] is True and root['Windows_saved_outputs_audit_fit_calls']==0,'Complementary evidence incomplete')
    require(root['Windows_whole_saved_check_pass'] is False and root['Windows_original47_exact_refit_pass'] is False and root['cross_platform_bitwise_recalibration_claimed'] is False and root['test'] is False,'Preserved failure or test boundary differs')
    require(root['original_Windows47_failure_sha256']==old['original_Windows47_failure_sha256'],'Original Windows failure pin changed')
    for role,path,pin in [('candidate_seal',binding['candidate']+'/FILES_SHA256.json',binding['candidate_seal_sha256']),('F_transport',binding['transport'],binding['transport_sha256'])]:
        require(root['proof_files'][role]['path']==path and root['proof_files'][role]['sha256']==pin,'Actual root/source/transport binding differs')


def validate_new_rows(rows):
    require(isinstance(rows,list) and len(rows)==9,'Exactly nine new rows required')
    require([r.get('id') for r in rows]==IDS,'Exact new10 IDs and order required')
    for seed,r in zip(range(91002,91011),rows):
        require((r.get('method'),r.get('distribution'),r.get('attack'),r.get('seed'))==('FLGMM','non-IID','S-DFA',seed),'Only non-IID S-DFA nine-new-seed scope allowed')
        for key in ('checkpoint_sha256','array_sha256','receipt_sha256'):sha(r.get(key))
