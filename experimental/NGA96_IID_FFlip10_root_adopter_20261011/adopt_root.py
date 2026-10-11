"""Root adoption of fixed10 F Flip after actual prior9; never collects or reruns science."""
from pathlib import Path
import argparse,datetime,hashlib,json,tarfile
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
H=R/'tmp/NGA96_IID_FFlip10_delta_prepare_20261011'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=Path(p).relative_to(R).as_posix(),sha256=sha(p))
def check_inputs(pins):
 assert pins['status']=='ROOT_BOUND_NGA_FFLIP10_AFTER_PRIOR9_NOT_ADOPTED'
 for key in ('source_adopter_sha256','root_review_sha256','authorization_sha256','offserver_sha256','download_sha256','release_sha256','tensor_sha256','raw_storage_index_sha256','archive_sha256'):
  value=pins[key];assert isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value) and value!='0'*64,key
 assert isinstance(pins['archive_members'],int) and not isinstance(pins['archive_members'],bool) and pins['archive_members']>0
 assert set(pins['phase_command_sha256'])=={'COLLECT','REMOTE_ARTIFACTS','SCP','VERIFY','RELEASE'}
 for value in pins['phase_command_sha256'].values():assert isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value) and value!='0'*64
parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--inputs-sha256',required=True);args=parser.parse_args()
assert __debug__ and sha(args.inputs)==args.inputs_sha256
inputs=read(args.inputs);check_inputs(inputs);assert sha(Path(__file__))==inputs['source_adopter_sha256']
assert not (O/'ROOT_ADOPTION.json').exists()
previous_root=R/'tmp/NGA96_IID_Benign9_root_adopter_prepare_20261011/ROOT_ADOPTION.json'
previous_offserver=R/'tmp/NGA96_IID_Benign9_after1_prepare_20261011/OFFSERVER_ACCEPTANCE.json'
assert sha(previous_root)=='a71dfbf295d8563c69ba07aa4b2fac2b7f0d9dddf6117d3a5198554fd7f177c2'
assert sha(previous_offserver)=='1ae8fead36bb832ee10f913b777fb807edab125e417405cf8d302a5e7b5fcd92'
assert (H/'PREVIOUS_ROOT_ADOPTION.json').read_bytes()==previous_root.read_bytes()
assert (H/'PREVIOUS_OFFSERVER_ACCEPTANCE.json').read_bytes()==previous_offserver.read_bytes()
previous=read(previous_root);old=previous['accepted_ids']
assert previous['root_adopted'] is True and previous['test'] is False and previous['accepted_total']==len(old)==9
assert old==['FedNGA_eta0.001_IID_Benign_seed91002_coverage', 'FedNGA_eta0.001_IID_Benign_seed91003_coverage', 'FedNGA_eta0.001_IID_Benign_seed91004_coverage', 'FedNGA_eta0.001_IID_Benign_seed91005_coverage', 'FedNGA_eta0.001_IID_Benign_seed91006_coverage', 'FedNGA_eta0.001_IID_Benign_seed91007_coverage', 'FedNGA_eta0.001_IID_Benign_seed91008_coverage', 'FedNGA_eta0.001_IID_Benign_seed91009_coverage', 'FedNGA_eta0.001_IID_Benign_seed91010_coverage'] and previous['records'][0]['constant_negative_retained'] is True
assert sha(H/'SOURCE_FILES_SHA256.json')=='c05bda43552adf4432d3ad2c2df05c23e3d805d85489ac056f9c3bca057c9fbd'
for name,pin in read(H/'SOURCE_FILES_SHA256.json')['files'].items():assert sha(H/name)==pin['sha256'] and (H/name).stat().st_size==pin['bytes']
for key,name in [('authorization','AUTHORIZED_SNAPSHOT.json'),('download','DOWNLOAD.json'),('release','COLLECTOR_RELEASE.json'),('tensor','SAVED_TENSOR_STATE_CHECK.json'),('raw_storage_index','RAW_STORAGE_INDEX.json')]:assert sha(H/name)==inputs[key+'_sha256']
for phase,digest in inputs['phase_command_sha256'].items():assert sha(H/(phase+'_COMMAND.json'))==digest
expected_members=inputs['archive_members']

auth=read(H/'AUTHORIZED_SNAPSHOT.json');off=read(H/'OFFSERVER_ACCEPTANCE.json');download=read(H/'DOWNLOAD.json');release=read(H/'COLLECTOR_RELEASE.json')
assert sha(H/'ROOT_REVIEW.json')==inputs['root_review_sha256']
review=read(H/'ROOT_REVIEW.json');assert review['status']=='PARENT_AUTHORIZED_FIXED_NGA96_INCREMENTAL_DELTA' and review['root_adoption_performed'] is False
assert review['source_files_sha256']==sha(H/'SOURCE_FILES_SHA256.json') and review['authorization_sha256']==sha(H/'AUTHORIZED_SNAPSHOT.json')
assert sha(H/'OFFSERVER_ACCEPTANCE.json')==inputs['offserver_sha256']
ids=auth['authorized_ids'];assert ids==['FedNGA_eta0.001_IID_F Flip_seed91001_coverage', 'FedNGA_eta0.001_IID_F Flip_seed91002_coverage', 'FedNGA_eta0.001_IID_F Flip_seed91003_coverage', 'FedNGA_eta0.001_IID_F Flip_seed91004_coverage', 'FedNGA_eta0.001_IID_F Flip_seed91005_coverage', 'FedNGA_eta0.001_IID_F Flip_seed91006_coverage', 'FedNGA_eta0.001_IID_F Flip_seed91007_coverage', 'FedNGA_eta0.001_IID_F Flip_seed91008_coverage', 'FedNGA_eta0.001_IID_F Flip_seed91009_coverage', 'FedNGA_eta0.001_IID_F Flip_seed91010_coverage'] and auth['accepted_prior_ids']==old and auth['accepted_prior_count']==9
assert review['exact_new_ids']==ids and auth['prior_root_sha256']==sha(previous_root) and auth['prior_offserver_sha256']==sha(previous_offserver)
assert off['previous_root_sha256']==sha(previous_root) and off['previous_offserver_sha256']==sha(previous_offserver)
assert off['authorized_snapshot_sha256']==sha(H/'AUTHORIZED_SNAPSHOT.json')
assert off['status']=='OFFSERVER_ORIGINAL_STRICT_DELTA_PASS' and off['accepted_new_ids']==ids and off['accepted_job_ids']==old+ids and off['accepted_before']==9 and off['accepted_total']==19
assert len(set(off['accepted_job_ids']))==19 and off['previous_accepted_ordered_prefix_exact'] is True
assert off['root_adopted'] is False and off['coverage100_completed_claim'] is False and off['all_negative_results_retained'] is True
assert off['new_inference']==off['new_training']==off['old_models_repacked']==off['reused_screen4_repacked']==0 and off['reused_screen4_count']==4
assert off['scientific_package_sha256']=='06e3d638aea70123d7fcf73e5a219a0c0d3165c8cf60acb9dee4eace5faad09b' and off['original_validator_sha256']=='d00dd2f3c0d8486f3e32044c3553972d1ff98515b54834a0a8b51048d2a5555e'
assert off['archive_verification']['pass'] and off['archive_verification']['different_host_observed'] and release['CPU110_released'] and not release['collector_processes']
for phase in ['COLLECT','REMOTE_ARTIFACTS','SCP','VERIFY','RELEASE']:assert read(H/(phase+'_COMMAND.json'))['exit_code']==0,phase
archive=Path(off['archive']);assert archive.drive.upper()=='F:' and sha(archive)==off['archive_sha256']==download['archive_sha256']==inputs['archive_sha256']
assert sha(H/'SAVED_TENSOR_STATE_CHECK.json')==off['tensor_proof_sha256']
tensor=read(H/'SAVED_TENSOR_STATE_CHECK.json');assert tensor['models']==10 and tensor['CNN_forward_calls']==tensor['optimizer_calls']==tensor['data_loads']==0
assert [r['id'] for r in tensor['records']]==ids and all(r['tensor_count']==8 for r in tensor['records'])
with tarfile.open(archive,'r:gz') as t:
 inventory_raw=t.extractfile('backup_inventory.json').read();inventory=json.loads(inventory_raw)
 assert hashlib.sha256(inventory_raw).hexdigest()==off['inventory_sha256'] and inventory['accepted_new_ids']==ids
 assert len(t.getmembers())==len(inventory['members'])+1==off['members_verified']==expected_members
 for name,pin in inventory['members'].items():
  m=t.getmember(name);assert m.isfile() and m.size==pin['bytes'];assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==pin['sha256'],name
 assert t.extractfile('evidence/PREVIOUS_ROOT_ADOPTION.json').read()==previous_root.read_bytes()
 assert t.extractfile('evidence/PREVIOUS_OFFSERVER_ACCEPTANCE.json').read()==previous_offserver.read_bytes()
assert [r['id'] for r in off['records']]==ids and all(r['rounds']==70 and isinstance(r['constant_negative_retained'],bool) for r in off['records'])
assert all(r['checkpoint_sha256']==tensor['records'][i]['checkpoint_sha256'] for i,r in enumerate(off['records']))
records=previous['records']+off['records'];assert records[:9]==previous['records'] and len(records)==19
proof=dict(status='ROOT_NGA96_IID_FFLIP_FIXED10_AFTER_PRIOR9_CUMULATIVE19_STRICT_OFFSERVER_FULLSTATE_PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root_adopted=True,test=False,accepted_ids=old+ids,accepted_total=19,accepted_new_ids=ids,accepted_before=9,scientific_package_sha256=off['scientific_package_sha256'],original_validator_sha256=off['original_validator_sha256'],offserver=ref(H/'OFFSERVER_ACCEPTANCE.json'),authorization=ref(H/'AUTHORIZED_SNAPSHOT.json'),root_review=ref(H/'ROOT_REVIEW.json'),collector_release=ref(H/'COLLECTOR_RELEASE.json'),raw_storage_index=ref(H/'RAW_STORAGE_INDEX.json'),archive_path=archive.as_posix(),archive_sha256=sha(archive),root_rehashed_members=expected_members,saved_tensor_state_proof=ref(H/'SAVED_TENSOR_STATE_CHECK.json'),records=records,prior_root_sha256=sha(previous_root),prior_offserver_sha256=sha(previous_offserver),prior_root=ref(previous_root),prior_offserver=ref(previous_offserver),root_inputs=ref(args.inputs.resolve()),source_adopter_sha256=sha(Path(__file__)),old9_ordered_record_prefix_exact=True,reused_screen4_count_separate=4,reused_screen4_repacked=0,coverage100_completed_claim=False,primary_endpoint_selected=False,final_evaluation=False,all_negative_results_retained=True,constant_predictor_not_fairness_success=True,whole_rebuttal_complete=False,new_inference=0,new_training=0)
with (O/'ROOT_ADOPTION.json').open('x',encoding='utf8',newline='\n') as f:json.dump(proof,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
print(json.dumps(dict(root_sha256=sha(O/'ROOT_ADOPTION.json'),accepted_new=10,accepted_total=19,coverage96_remaining=77,reused4_separate=True)))
