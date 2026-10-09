"""Materialize only the exact native140 / prior views136 metadata rebind."""
import ast,hashlib,json,re
from pathlib import Path
H=Path(__file__).resolve().parent; R=H.parents[1]
OLD=H.with_name('celeba_mechanism_valid_C_after28_20261009')
B=R/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009'
TAG='root_delta_20261009T223148Z'; IDS=[f'minus_C_IID_S-DFA_seed{s}' for s in range(91007,91011)]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,data):
 p=H/name
 with p.open('x',encoding='utf-8',newline='\n') as f:f.write(data if isinstance(data,str) else json.dumps(data,indent=2)+'\n')
def replace(source,changes):
 return re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda m:changes[m.group(0)],source)
def assignment(source,name,value):
 n=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
 return source.replace(ast.get_source_segment(source,n),name+'='+repr(value),1)
pins={
 'archive':B/(TAG+'.tar.gz'),'receipt':B/(TAG+'.tar.gz.receipt.json'),'proof':B/(TAG+'_offserver_verification.json'),
 'ledger':B/TAG/'verified_ledger.json','inspection':B/('mechanism_inspection_v4_'+TAG)/'inspection.json',
 'root_delta':B/TAG/'ROOT_DELTA_VERIFICATION.json','root_review':H/'ROOT_NATIVE140_INDEPENDENT_REVIEW.json'}
verify=(OLD/'verify_native_snapshot.py').read_text('utf-8-sig')
verify=replace(verify,{
 'root_delta_20261009T214755Z':TAG,'root_delta_20261009T212311Z':'root_delta_20261009T214755Z',
 '413cf036dcc306cd4276448d839dbe6c542fcca49dcd7b54a5cf2a4dccec5345':'74426180abdf1fccb40af004df06ecae2fc1c6316166ffce63c44d160c077c2a',
 '42c3b021b27ba53eedb2e2f0d684eab9a0c8aa5af8b3b916591cd297399a3794':'1cc05e3c627653d9b37cc0b4eb6081454fd7aa2d0bb5d5ce587e4a140444336e',
 'f9e3aac3667fbf706c6e08d4d5c02818209820753b85473b75162435024d4212':'42c3b021b27ba53eedb2e2f0d684eab9a0c8aa5af8b3b916591cd297399a3794',
 'de6a24a40e454ee3d3fc833ef07df2950db262b74f8a508b43b5d61de1c6d0db':'cc541f4e628f70888346684ae43b3ae949ff3bd03af967fa80c7c3f73a98d175',
 'native136':'native140','NATIVE136':'NATIVE140','original228':'original236','OLD228':'OLD236','added8':'added4','EXACT8_C_FEDSA_SDFA':'EXACT4_C_SDFA',
 'ledger_previous19':'ledger_previous20','minus_C_partial_36':'minus_C_partial_40','minus_C_IID_SDFA_partial_6':'minus_C_IID_SDFA_complete_10'})
verify=re.sub(r'\b(136|236|228|94|20|19|36)\b',lambda m:{'136':'140','236':'240','228':'236','94':'62','20':'21','19':'20','36':'40'}[m.group()],verify)
verify=assignment(verify,'IDS',IDS).replace("target=B/TAG/'ROOT_INDEPENDENT_REVIEW.json'","target=H/'ROOT_NATIVE140_INDEPENDENT_REVIEW.json'")
verify=verify.replace("'new8':IDS","'new4':IDS")
write('verify_native_snapshot.py',verify)
source=(OLD/'prepare.py').read_text('utf-8-sig')
source=replace(source,{
 'celeba_mechanism_valid_C_after25_20261009':OLD.name,'celeba_mechanism_valid_C_after28_20261009':H.name,
 'C_AFTER25':'C_AFTER28','C_AFTER28':'C_AFTER36','valid_C_after20':'valid_C_after25','valid_C_after25':'valid_C_after28','valid_C_after28':'valid_C_after36',
 'sealed_C_after25':'sealed_C_after28','C_after28/':'C_after36/','closed_C_after25':'closed_C_after28','C_after28_metadata':'C_after36_metadata',
 '383d53753e907d4810048fbd6fc77ccafbcd4bd3067cfe1fc3de3ce4b6390a87':'1d06eca4b5eccf94fa5c510ca867d4281e0a8fe3bce5a2b17b18864e43bdcc67',
 'adbad62e9a7fa3936790255e40bd29262441798d17e104a8cda26c1d9b722560':'c823dce69ef511ff15faf1b964805184a6a8c10f271214de1cc6706446a49ba2',
 'fe2f65d5a0353949b2d15cb5c04828d8fdd7e7574d7788f00d7b096ece3489de':'7cf324b92be73420b7f4497673519c779666664d13a98ad93f4207434c1de635',
 'incremental_20261009T213828Z':'incremental_20261009T221150Z',
 '0d1661bdc21025b957fa4e6ac39d4c8924cb50ab1b91920268119941312a615a':'cd197e57b42dda87bc239736e401cb5accd029313546533f6a48caf27950e1e0',
 'ba3edd176219106c781e8b437017ff7444cee8b3e47073b25932928682661ef3':'0d1661bdc21025b957fa4e6ac39d4c8924cb50ab1b91920268119941312a615a',
 'inventory_actual128':'inventory_actual136','inventory_actual136':'inventory_actual140',
 'NATIVE136':'NATIVE140','native136':'native140','native_v4_metadata_only':'native_v4_C_after36_metadata_only',
 'original228':'original236','added8':'added4','actual128':'actual136','accepted128':'accepted136','actual136':'actual140','accepted136':'accepted140',
 'original125':'original128','original128':'original136','prior125':'prior128','prior128':'prior136','PRIOR128':'PRIOR136',
 'closed128':'closed136','closed125':'closed128','Prior125':'Prior128','Prior128':'Prior136','Excluded-prior125/selected3':'Excluded-prior128/selected8','Excluded-prior128/selected8':'Excluded-prior136/selected4',
 'selected3':'selected8','selected8':'selected4','SELECTED_8':'SELECTED_4','exact3':'exact8','exact8':'exact4','Only3':'Only8','Only8':'Only4','prepared3':'prepared8','prepared8':'prepared4','reviewed3':'reviewed8','reviewed8':'reviewed4','ALL3':'ALL8','ALL8':'ALL4',
 'U100+C25':'U100+C28','U100+C28':'U100+C36','exact three C':'exact eight C','exact eight C':'exact four C','C8;':'C4;',
 "scope['excluded_prior_ids']) == 125":"scope['excluded_prior_ids']) == 128",
 })
source=re.sub(r'\b(136|128|125|664|672|94|8|3)\b',lambda m:{'136':'140','128':'136','125':'128','664':'660','672':'664','94':'62','8':'4','3':'8'}[m.group()],source)
source=assignment(source,'SELECTED',IDS)
write('prepare.py',source)
check=(OLD/'check_prepared.py').read_text('utf-8-sig')
check=replace(check,{'celeba_mechanism_valid_C_after25_20261009':OLD.name,'C_AFTER28':'C_AFTER36','inventory_actual136':'inventory_actual140','prior128':'prior136','closed128':'closed136','exact8':'exact4','exact8-terminal':'exact4-terminal',
 'per_child_exact8':'per_child_exact4','bounded C8':'bounded C4','0d1661bdc21025b957fa4e6ac39d4c8924cb50ab1b91920268119941312a615a':'cd197e57b42dda87bc239736e401cb5accd029313546533f6a48caf27950e1e0',
 "service = 'guardfed_celeba_mechanism_valid_C_after25'":"service = 'guardfed_celeba_mechanism_valid_C_after28'"})
check=re.sub(r'\b(136|128|664)\b',lambda m:{'136':'140','128':'136','664':'660'}[m.group()],check)
check=check.replace('for n in (0,1,2,7,9,11):','for n in (0,1,3,5,8,11):')
check=check.replace("assert ids==[f'minus_C_IID_FedSA_seed{s}' for s in (91009,91010)]+[f'minus_C_IID_S-DFA_seed{s}' for s in range(91001,91007)]","assert ids==[f'minus_C_IID_S-DFA_seed{s}' for s in range(91007,91011)]")
check=check.replace("'minus_C_IID_S-DFA_seed91007'","'minus_C_IID_S-DFA_seed91006'").replace("'len(chosen)==8'","'len(chosen)==4'").replace("'per_child_exact4_science_approval_positive':8","'per_child_exact4_science_approval_positive':4")
write('check_prepared.py',check)
write('REBIND_MATERIALIZATION.json',{'status':'METADATA_SOURCE_MATERIALIZED_NOT_RUN','selected_ids':IDS,'parent_prepare_sha256':sha(OLD/'prepare.py'),'parent_verifier_sha256':sha(OLD/'verify_native_snapshot.py'),'parent_checker_sha256':sha(OLD/'check_prepared.py')})
print(json.dumps({'status':'MATERIALIZED','selected':IDS,'verifier_sha256':sha(H/'verify_native_snapshot.py')}))
