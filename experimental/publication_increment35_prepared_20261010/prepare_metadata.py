"""Write source-only contracts from closed, immutable parent inputs; no Git calls."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
def write(n,v):
 with (H/n).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2);f.write('\n')
base='docs/server_deployment_20260923/training_20260923/';native=base+'server_reactivation_20261009/mechanism_science_backups_20261009/';tag='root_delta_20261009T233607Z'
ready={
 'tmp/publication_increment34_prepared_v2_20261010/publish_increment34.py':'59309c9ae03971efdddee3027e0a045248bfb92f3fb5f9673127568e07af2953',
 'tmp/publication_increment34_prepared_v2_20261010/verify_increment34.py':'347d8817a6342e6cef325d5d5a6e1a10ea73c5ce7d82f9466d373f54216c19d5',
 native+tag+'/ROOT_INDEPENDENT_REVIEW.json':'c0e78899ce232b81c7c2849ed9d05283c3c740bc962176074ad7281c72c24f4a',
 native+tag+'/ROOT_DELTA_VERIFICATION.json':'744c17ea5ef63184fdbbc66d7a95f8277dc3497fd82f6ba76aaa19e6bbb34b65',
 native+tag+'/verified_ledger.json':'7e04be828e3b6824dbb13e40b0cc8d3a9435f48541a0fafebc2748f259691353',
 native+'mechanism_inspection_v4_'+tag+'/inspection.json':'5ad6f7753271c882b20a8ecc975a3fbdc9228e8aaa894222e63f80f118adb439',
 native+tag+'.tar.gz':'11ae8eb7f7cae59e10cd5df79f9681b13ebbdec2c4f460968d93e9fb82e98866',
 native+tag+'.tar.gz.receipt.json':'6af158d14a6f13844d4509c28c3e66c1b5583a1fc239aef7ec7f6af0f6f13951',
 native+tag+'_offserver_verification.json':'1a552aa963a6c8497d71165e69a19b74672f7a469adb219c491dc39e070c5a7e',
 'tmp/celeba_mechanism_valid_C_after47_20261010/PACKAGE_SHA256.json':'4298e771c432184f4987869863f15658860102b3d73c8a5c809e6e8838a451d1',
 'tmp/celeba_mechanism_valid_C_after47_20261010/FILES_SHA256.json':'aa5073948bd1c0854b3a4d760ee58b892909f702c31950e5ee80f0cf83b2efc0',
 'tmp/celeba_mechanism_valid_C_after47_20261010/execution_candidate/EXECUTION_SOURCE_SHA256.json':'b647c5a6759e709ab4c4fdd9d7ba74361901f98973d67f69176569b1a14c8ef5'}
for p,s in ready.items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==s
required=[base+n for n in ['RUNNING.md','TRAINING_STATE.json','REBUTTAL_COMPLETION_20261009.md','celeba_mechanism_v1/EXECUTION.md','server_reactivation_20261009/MONITOR_HANDOFF.md','server_reactivation_20261009/latest_formal_live.json']]+['docs/返修实验总览.md','tmp/celeba_flgmm_fullcoverage_incremental_20261009/LATEST_BACKUP.json','tmp/celeba_hybrid_screen_execution_20261009/LATEST_BACKUP.json','tmp/celeba_hybrid_screen_execution_20261009/BACKUP_CHAIN_accepted_delta_after18_20261010.json']
write('PREPARED_INPUTS.json',{'status':'SOURCE_ONLY_REQUIRES_ACTUAL_C3_CLOSURE_AND_ROOT_INJECTED_INPUTS','parent_commit':'ae94e17a7b1c3b3f6298eaa9d7f09bbf31cca17d','branch':'codex/revision-evidence-baselines-20260928','ready_sha256':ready,'required_extra_paths':required,'actual_future_C3_root_adoption_sha256':None,'actual_optional_C50_table_sha256':None,'source_preparation_is_publication':False})
write('ROOT_CLOSED_INPUTS_TEMPLATE.json',{'status':'PREPARED_NOT_CLOSED_ROOT_MUST_FILL_ACTUAL_PATHS_AND_SHA','parent_commit':'ae94e17a7b1c3b3f6298eaa9d7f09bbf31cca17d','counts':None,'required_closed_counts':{'native':150,'three_view':150,'FL_new':9,'Hybrid':19,'baseline_valid':900},'closure_pins':{k:{'path':None,'sha256':None} for k in ['C3_adoption','C3_source_review','FL_adoption','Hybrid_adoption','state','formal_live','previous_publication']},'extra_pins':{p:None for p in required},'C50_table':None,'optional_C50_table_shape':{'directory':None,'root_proof':{'path':None,'sha256':None},'seal':{'filename':None,'sha256':None},'C3_adoption_field':'ACTUAL_EXISTING_ROOT_PROOF_FIELD_NAME'},'note':'Root adds the exact actual after47 root operations/source review seal/helpers, auxiliary linked chain/observations and root updater script SHA to extra_pins. No nonexistent fields or future hashes are assumed.'})
write('READ_PROBE_NOT_READY.json',{'status':'PRESERVED_SOURCE_PREPARATION_READ_PROBE_ONLY','path':'tmp/celeba_mechanism_C_after47_root_operations_20261010/ROOT_SOURCE_REVIEW.json','error':'Get-Content: path does not exist','command_exit_code':1,'boundary':'Independent completed-source query ended at a not-yet-created future review path. No source, index, approval or acceptance write occurred. Future review is required through the root-injected actual closure pin.'})
