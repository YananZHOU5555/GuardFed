"""Rebind prior packaging/commands only; scientific source is already sealed."""
from pathlib import Path
import ast,json,re
H=Path(__file__).resolve().parent;O=H.with_name('celeba_mechanism_valid_C_after40_20261010')
s=(O/'seal_delivery.py').read_text(encoding='utf-8-sig')
changes={
 'celeba_mechanism_valid_C_after36_20261010':'celeba_mechanism_valid_C_after40_20261010',
 'inventory_actual147_Full100refs.json':'inventory_actual150_Full100refs.json',
 'inventory_actual140_Full100refs.json':'inventory_actual147_Full100refs.json',
 'old140_records_exact':'old147_records_exact','old140_raw_json_bytes_and_order_exact':'old147_raw_json_bytes_and_order_exact',
 'actual_native147_root_review_sha256':'actual_native150_root_review_sha256',
 'prior140_root_adoption_sha256':'prior147_root_adoption_sha256',
 'eb1f6c3120febb138e32af25484ac790cf962cd15d5ea9921140119bc5a3317a':'64732337f35f81bed49e40c60fa1d5229fb6a7557c45272505fee488c5ad20ea',
 'native_global=147':'native_global=150','inventory_records=147':'inventory_records=150',
 'prior_three_views_excluded=140':'prior_three_views_excluded=147',
 'PREPARED_ONLY_EXACT7_SOURCE_AND_METADATA_GATES_PASS_NO_EXECUTION':'PREPARED_ONLY_EXACT3_SOURCE_AND_METADATA_GATES_PASS_NO_EXECUTION',
 'native_accepted_snapshot=147':'native_accepted_snapshot=150','three_view_accepted_unchanged=140':'three_view_accepted_unchanged=147',
 'C_IID_SpDFA_native_n=7':'C_IID_SpDFA_native_n=10','C_IID_SpDFA_scene_complete=False':'C_IID_SpDFA_scene_complete=True',
 "service='guardfed_celeba_mechanism_valid_C_after40'":"service='guardfed_celeba_mechanism_valid_C_after47'",
 "prior_service_required_exited='guardfed_celeba_mechanism_valid_C_after36'":"prior_service_required_exited='guardfed_celeba_mechanism_valid_C_after40'",
}
s=re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda m:changes[m.group(0)],s)
ast.parse(s)
with (H/'seal_delivery.py').open('x',encoding='utf-8',newline='\n') as f:f.write(s)
c=(O/'COMMANDS.md').read_text(encoding='utf-8-sig')
changes={
 'celeba_mechanism_valid_C_after40_20261010':'celeba_mechanism_valid_C_after47_20261010',
 'guardfed_celeba_mechanism_valid_C_after40':'guardfed_celeba_mechanism_valid_C_after47',
 'prior C_after36':'prior C_after40','C_AFTER40':'C_AFTER47','prior140':'prior147',
 'closed140_must_not_replay':'closed147_must_not_replay','exact7':'exact3','all7':'all3',
 'expects82 total/81content and63 metrics/168 counts/21 prediction rules':'expects54 total/53content and27 metrics/72 counts/9 prediction rules',
}
c=re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda m:changes[m.group(0)],c)
with (H/'COMMANDS.md').open('x',encoding='utf-8',newline='\n') as f:f.write(c)
record={'status':'ACTUAL_LOCAL_NATIVE_INCREMENT_REVIEW_EXIT0','command':['python','-B','tmp/celeba_mechanism_valid_C_after47_20261010/verify_native_snapshot.py','--root-delta','docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261009T233607Z/ROOT_DELTA_VERIFICATION.json','--root-delta-sha256','744c17ea5ef63184fdbbc66d7a95f8277dc3497fd82f6ba76aaa19e6bbb34b65'],'returncode':0,'root_review_sha256':'c0e78899ce232b81c7c2849ed9d05283c3c740bc962176074ad7281c72c24f4a','SSH':False,'CNN':False}
with (H/'NATIVE_REVIEW_COMMAND.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.write('\n')
