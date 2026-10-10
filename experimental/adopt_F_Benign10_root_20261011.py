"""Run the reviewed original saved-table check once, then adopt compact outputs."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys
R=Path(__file__).resolve().parents[1];H=R/'tmp/celeba_F_IID_Benign10_table_20261011'
D=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_Benign10_20261011'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(p,v):
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
assert not D.exists()
assert sha(H/'FILES_SHA256.json')=='b36f28bb0501453ffd2cf021bd3f9dd493474a110365847e98292d31d0ab5b1a'
assert sha(H/'ROOT_SOURCE_REVIEW.json')=='2cd5a7b78ee3826d72e8cc24fac773e7fb3e71958d19a72f7a857d7919ae1d77'
assert sha(H/'ROOT_BINDING.json')=='be6fb6c77cdbf3f734fc8d84c7c847f31d1d4bb22b4d1ad4547a250d24e45306'
assert sha(H/'OUTPUTS_SHA256.json')=='eef79c4965b90bc07132f06564dc6af3147b9599a9db7233b87da100440cd59b'
assert sha(H/'verify_saved.py')=='c7c71f2ba47a401920ab12d2b52f795c1f13cb569681406c805a54f82c39d161'
for n,pin in read(H/'OUTPUTS_SHA256.json')['files'].items():assert sha(H/n)==pin['sha256'] and (H/n).stat().st_size==pin['bytes']
argv=[sys.executable,'-B',str(H/'verify_saved.py'),'--binding-sha256',sha(H/'ROOT_BINDING.json')]
save(H/'ROOT_VERIFY_COMMAND.json',argv)
with (H/'ROOT_VERIFY.stdout.json').open('xb') as out,(H/'ROOT_VERIFY.stderr.txt').open('xb') as err:run=subprocess.run(argv,stdout=out,stderr=err)
save(H/'ROOT_VERIFY_EXIT.json',dict(returncode=run.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
assert run.returncode==0,'Preserve failure; no blind retry or adoption'
checked=read(H/'ROOT_VERIFY.stdout.json');table=read(H/'tables.json');binding=read(H/'ROOT_BINDING.json')
assert checked['status']=='PASS_ORIGINAL_FSUM_COUNTS_AND_SAVED_DISPLAY_SCOPE'
assert (checked['mean_sd_scalars'],checked['display_cells'],checked['receipt_metrics_from_group_counts'],checked['base_confusion_counts_structurally_checked'])==(162,81,180,480)
D.mkdir(parents=True)
names=list(read(H/'OUTPUTS_SHA256.json')['files'])+['ROOT_BINDING.json','ROOT_SOURCE_REVIEW.json','FILES_SHA256.json','ACTUAL_HANDOFF.json','OUTPUTS_SHA256.json','ROOT_VERIFY_COMMAND.json','ROOT_VERIFY.stdout.json','ROOT_VERIFY.stderr.txt','ROOT_VERIFY_EXIT.json']
for n in names:shutil.copyfile(H/n,D/n);assert sha(H/n)==sha(D/n)
proof=dict(status='ROOT_F10_SINGLE_COMPLETE_SCENE_THREE_VIEW_TABLE_ADOPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root_adoption=True,paired_models=10,complete_scenes=1,preserved_records=20,partial_other_scene_pairs=0,seed_panels=[10,9,6],mean_SD_scalars_recomputed=162,display_cells=81,metrics_from_group_counts=180,base_integer_confusion_counts_checked=480,max_abs_difference=checked['max_abs_difference'],original_Full10_records_exact=True,original_F10_saved_views_exact=True,replay_devices=table['replay_devices'],training_torch=table['training_torch'],native_shared_metrics_and_counts_exact=read(H/'checks.json')['native_shared_metrics_and_counts_exact'],source_acceptance_path=binding['adoption'],source_acceptance_sha256=binding['adoption_sha256'],source_native_root_sha256=binding['native_root_sha256'],source_binding_sha256=sha(H/'ROOT_BINDING.json'),source_seal_sha256=sha(H/'FILES_SHA256.json'),output_seal_sha256=sha(H/'OUTPUTS_SHA256.json'),files_sha256={n:sha(D/n) for n in names},canonical_table=(D/'TABLES.md').relative_to(R).as_posix(),source_candidate_path=H.relative_to(R).as_posix(),independent_review_path=(H/'ROOT_VERIFY.stdout.json').relative_to(R).as_posix(),independent_review_sha256=sha(H/'ROOT_VERIFY.stdout.json'),root_adopter_path=Path(__file__).relative_to(R).as_posix(),root_adopter_sha256=sha(Path(__file__)),actual_root_command_exit=0,new_CNN=0,new_fits=0,new_training=0,test=False,whole_rebuttal_complete=False,limitation='One IID Benign scene only; native/shared coincide, not independent evidence; all fixed10/9/6 panels and negative paired differences retained. Full2CPU8GPU versus F10CPU and seed91001 validation selection disclosed; no F100, necessity, significance or final-test claim.')
save(D/'ROOT_VERIFICATION.json',proof)
print(json.dumps(dict(status=proof['status'],root_path=(D/'ROOT_VERIFICATION.json').relative_to(R).as_posix(),root_sha256=sha(D/'ROOT_VERIFICATION.json'),stats=162,cells=81)))
