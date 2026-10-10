"""Consume the root-run saved-table check without repeating it, then adopt compact outputs."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys
R=Path(__file__).resolve().parents[2];H=R/'tmp/celeba_F_IID_two_scenes20_table_20261011'
D=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_two_scenes20_20261011'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(p,v):
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
assert not D.exists()
assert sha(H/'FILES_SHA256.json')=='aef78ed959b26e614cff5186b1f1178da49ef086b31701bbd96b72cf87608f19'
assert sha(H/'ROOT_SOURCE_REVIEW.json')=='d37ec6b0bd400e5fdc2ef3c2450f635f010f152d43b78261e2ca7594d599ed45'
assert sha(H/'ROOT_BINDING.json')=='9f4d59ca70e5770e9514c0c7007b5ef030c96ac3ba2c70032cec0e91b69b05ac'
assert sha(H/'OUTPUTS_SHA256.json')=='6106d8f2d6cf4cfe0178f48f9ebf87134ff6ba4a6def8ba602c23e0c6c90403e'
assert sha(H/'verify_saved.py')=='ca35d6f5463be3578cba57b2645b5c4c48c882ffd081c578fcbc164147c999fe'
for n,pin in read(H/'OUTPUTS_SHA256.json')['files'].items():assert sha(H/n)==pin['sha256'] and (H/n).stat().st_size==pin['bytes']
import argparse
parser=argparse.ArgumentParser()
for name in ('command','stdout','stderr','exit'):parser.add_argument('--root-verify-'+name+'-sha256',required=True)
args=parser.parse_args()
for name,file in [('command','ROOT_VERIFY_COMMAND.json'),('stdout','ROOT_VERIFY.stdout'),('stderr','ROOT_VERIFY.stderr'),('exit','ROOT_VERIFY_EXIT.json')]:
    assert sha(H/file)==getattr(args,'root_verify_'+name+'_sha256')
assert read(H/'ROOT_VERIFY_EXIT.json')['exit_code']==0,'Preserve root failure; no retry or adoption'
command=read(H/'ROOT_VERIFY_COMMAND.json');argv=command['argv'] if isinstance(command,dict) else command
assert Path(argv[2]).resolve()==(H/'verify_saved.py').resolve() and argv[-2:]==['--binding-sha256',sha(H/'ROOT_BINDING.json')]
checked=read(H/'ROOT_VERIFY.stdout');table=read(H/'tables.json');binding=read(H/'ROOT_BINDING.json')
assert checked['status']=='PASS_ORIGINAL_FSUM_COUNTS_AND_SAVED_DISPLAY_SCOPE'
assert (checked['mean_sd_scalars'],checked['display_cells'],checked['receipt_metrics_from_group_counts'],checked['base_confusion_counts_structurally_checked'])==(324,162,360,960)
old=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_Benign10_20261011'
old_pins={n:pin for n,pin in read(H/'SOURCE_INPUTS.json')['files'].items() if n.startswith(old.relative_to(R).as_posix()+'/')}
assert old_pins and all(sha(R/n)==pin['sha256'] for n,pin in old_pins.items())
checks=read(H/'checks.json')
assert checks['old20_record_JSON_bytes_and_order_exact'] and checks['old162_scalars_exact'] and checks['old81_cells_preserved']
assert (table['complete_scenes'],table['paired_models'],table['preserved_records'])==(2,20,40)
D.mkdir(parents=True)
names=list(read(H/'OUTPUTS_SHA256.json')['files'])+['ROOT_BINDING.json','ROOT_SOURCE_REVIEW.json','FILES_SHA256.json','ACTUAL_HANDOFF.json','OUTPUTS_SHA256.json','ROOT_VERIFY_COMMAND.json','ROOT_VERIFY.stdout','ROOT_VERIFY.stderr','ROOT_VERIFY_EXIT.json']
for n in names:shutil.copyfile(H/n,D/n);assert sha(H/n)==sha(D/n)
proof=dict(status='ROOT_F20_TWO_COMPLETE_SCENE_THREE_VIEW_TABLE_ADOPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root_adoption=True,paired_models=20,complete_scenes=2,preserved_records=40,partial_other_scene_pairs=0,seed_panels=[10,9,6],mean_SD_scalars_recomputed=324,display_cells=162,metrics_from_group_counts=360,base_integer_confusion_counts_checked=960,max_abs_difference=checked['max_abs_difference'],original_Full20_records_exact=True,original_F20_saved_views_exact=True,old20_record_JSON_bytes_and_order_exact=True,old162_scalars_exact=True,old81_cells_preserved=True,complete_scene_keys=[["IID","Benign"],["IID","F Flip"]],replay_devices=table['replay_devices'],training_torch=table['training_torch'],native_shared_metrics_and_counts_exact=read(H/'checks.json')['native_shared_metrics_and_counts_exact'],source_acceptance_path=binding['adoption'],source_acceptance_sha256=binding['adoption_sha256'],source_native_root_sha256=binding['native_root_sha256'],source_binding_sha256=sha(H/'ROOT_BINDING.json'),source_seal_sha256=sha(H/'FILES_SHA256.json'),output_seal_sha256=sha(H/'OUTPUTS_SHA256.json'),files_sha256={n:sha(D/n) for n in names},canonical_table=(D/'TABLES.md').relative_to(R).as_posix(),source_candidate_path=H.relative_to(R).as_posix(),independent_review_path=(H/'ROOT_VERIFY.stdout').relative_to(R).as_posix(),independent_review_sha256=sha(H/'ROOT_VERIFY.stdout'),root_adopter_path=Path(__file__).relative_to(R).as_posix(),root_adopter_sha256=sha(Path(__file__)),actual_root_command_exit=0,new_CNN=0,new_fits=0,new_training=0,test=False,whole_rebuttal_complete=False,limitation='Two IID scenes, Benign and F Flip only; native/shared coincide, not independent evidence; all fixed10/9/6 panels and negative paired differences retained. Full2CPU18GPU versus F20CPU and seed91001 validation selection disclosed; no cross-scene aggregate, F100, necessity, significance or final-test claim.')
assert all(sha(R/n)==pin['sha256'] for n,pin in old_pins.items())
save(D/'ROOT_VERIFICATION.json',proof)
print(json.dumps(dict(status=proof['status'],root_path=(D/'ROOT_VERIFICATION.json').relative_to(R).as_posix(),root_sha256=sha(D/'ROOT_VERIFICATION.json'),stats=324,cells=162)))
