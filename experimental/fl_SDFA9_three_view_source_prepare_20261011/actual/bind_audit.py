"""Late-bind actual Pool32 exact9 proofs before the original zero-fit audit; no fit."""
from pathlib import Path
import argparse,ast,hashlib,json,sys
H=Path(__file__).resolve().parent;C=H.parent;S=C/'saved_v2';R=C.parents[1];A=H/'saved001'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
T=R/'tmp/fl_three_view_after48_20261011/saved/AUDIT_TEST_PINS.json'
PINNED={
 C/'FILES_SHA256.json':'98e76bade35c1f9070d3b2682c08812ec9b5c948c09b6e2d9d3bf12c3a4fcd7a',
 S/'STATIC_SOURCE_SHA256.json':'d39a1886fe45cf45dc9cf877dc4a698e1e43e5ec186efcc39b0b8b2c9868c1f2',
 T:'7bca1d2631e2d3d59a974e5eb3eeee0ec37e394878650283749d6aa332065a66',
 R/'tmp/fl_native_after77_SDFA9_prepared_20261011/ROOT_ADOPTION_REVIEW.json':'904e3ee036dc4df66459f804db1903211b873ffb5de714ccb97f3900e39c824a',
 R/'tmp/fl_FedSA10_three_view_prepared_20261011/ROOT_SCIENTIFIC_ADOPTION.json':'adc7726ca05561c6903ebb153e34fb3dc69a1df0118d87e6bf7dfa3e31655799',
}

def main():
 assert __debug__,'Do not use -O'
 # Static and actual runtime pins are checked together after required CLI parsing.
 sys.path.insert(0,str(S));import contract as k
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--gate-result-sha256',required=True,type=k.digest_arg)
 parser.add_argument('--linux-proof-sha256',required=True,type=k.digest_arg)
 parser.add_argument('--transport-proof-sha256',required=True,type=k.digest_arg)
 for key in ('source-review','linux-preflight','source-deployment','authorization','start-receipt'):
  parser.add_argument('--'+key+'-sha256',required=True,type=k.digest_arg)
 a=parser.parse_args()
 runtime_files={'source_review':C/'ROOT_SOURCE_REVIEW.json','linux_preflight':C/'runtime/LINUX_PREFLIGHT.json','source_deployment':C/'runtime/SOURCE_DEPLOYMENT.json','authorization':C/'runtime/AUTHORIZATION.json','start_receipt':C/'runtime/START_RECEIPT.json'}
 for role,path in runtime_files.items():PINNED[path]=getattr(a,role+'_sha256')
 for path,digest in PINNED.items():assert path.is_file() and path.stat().st_size<2_000_000 and sha(path)==digest,str(path)
 review=read(runtime_files['source_review']);pre=read(runtime_files['linux_preflight']);auth=read(runtime_files['authorization'])
 assert review['source_adoptable'] is True and review['package_sha256']==k.PACKAGE
 assert auth['source_review_sha256']==a.source_review_sha256 and auth['linux_preflight_sha256']==a.linux_preflight_sha256 and auth['package_sha256']==k.PACKAGE
 assert pre['package_sha256']==k.PACKAGE and pre['resources_eligible'] is True
 assert not (S/'FILES_SHA256.json').exists() and not (S/'SOURCE_PINS.json').exists()
 assert not any((A/('SAVED_OUTPUTS_AUDIT_NO_REFIT'+suffix)).exists() for suffix in ('.json','.started.json','.failure.json'))
 package=read(C/'FILES_SHA256.json')['files'];static=read(S/'STATIC_SOURCE_SHA256.json')['files']
 for base,pins in ((C,package),(S,static)):
  for rel,pin in pins.items():
   p=(base/rel).resolve();assert p.is_relative_to(base.resolve()) and p.is_file() and p.stat().st_size==pin['bytes']<2_000_000 and sha(p)==pin['sha256'],rel
 manifest=read(C/'MANIFEST.json');assert manifest['exact_ids']==[r['id'] for r in manifest['records']]==k.IDS and len(set(k.IDS))==9
 assert sha(A/'LINUX_SAVED_CHECK.json')==a.linux_proof_sha256 and sha(A/'TRANSPORT_VERIFICATION.json')==a.transport_proof_sha256
 linux=read(A/'LINUX_SAVED_CHECK.json');k.linux_proof(linux)
 transport=read(A/'TRANSPORT_VERIFICATION.json')
 assert transport['status']=='FLGMM9_F_TRANSPORT_ALL_MEMBERS_SHA_PASS_NOT_SCIENTIFIC_ACCEPTANCE' and transport['member_count']==22
 assert transport['linux_proof_sha256']==a.linux_proof_sha256 and transport['gate_result_sha256']==linux['gate_result_sha256']==a.gate_result_sha256
 assert transport['new_CNN']==transport['new_fit']==transport['new_training']==0 and transport['test'] is False and transport['root_adopted'] is False
 k.volume();extract=k.fpath(transport['verified_extract']);archive=k.fpath(transport['archive_path'])
 assert archive.name=='flgmm9_saved_arrays_and_receipts.zip' and archive.parent==extract.parent and archive.is_file() and archive.stat().st_size==transport['archive_bytes']
 k.digest_arg(transport['archive_sha256'])
 expected={'bundle/GATE_RESULT.json','bundle/metadata_receipt.json','LINUX_SAVED_CHECK.json'}|{'bundle/'+i+'/'+n for i in k.IDS for n in ('receipt.json','validation_predictions.npz')}
 members=transport['members'];assert set(members)==expected and len(members)==21
 tm=read(extract/'TRANSPORT_MANIFEST.json')
 assert tm['status']=='FLGMM9_SAVED_ARRAY_TRANSPORT_SOURCE_MEMBERS_UNCHANGED_NOT_ROOT_ADOPTED' and tm['members']==members and tm['exact_ids']==k.IDS
 assert tm['service'].split()[:2]==['guardfed_flgmm_SDFA9_valid','EXITED'] and tm['test'] is False and tm['new_CNN']==tm['new_fit']==0 and tm['model_or_images_downloaded'] is False
 assert sha(extract/'bundle/GATE_RESULT.json')==a.gate_result_sha256 and sha(extract/'LINUX_SAVED_CHECK.json')==a.linux_proof_sha256
 gate=read(extract/'bundle/GATE_RESULT.json');k.gate_proof(gate)
 for rel,pin in members.items():
  k.digest_arg(pin['sha256']);assert type(pin['bytes']) is int and 0<pin['bytes']<10_000_000
  if rel.endswith('.json'):assert sha(extract/rel)==pin['sha256'] and (extract/rel).stat().st_size==pin['bytes'],rel
 for row,receipt,whole in zip(manifest['records'],gate['receipts'],linux['records']):
  ident=row['id'];assert ident==receipt['id']==whole['id']
  assert receipt['status']=='NATIVE_VALID_REPLAY_PASS' and read(extract/'bundle'/ident/'receipt.json')==receipt
  assert receipt['checkpoint_sha256']==whole['checkpoint_sha256']==row['identity']['checkpoint']['sha256']
  assert whole['receipt_sha256']==members['bundle/'+ident+'/receipt.json']['sha256']
  assert receipt['prediction_arrays_sha256']==whole['array_sha256']==members['bundle/'+ident+'/validation_predictions.npz']['sha256']
 command=read(A/'TRANSPORT_COMMAND.json');exit_record=read(A/'TRANSPORT_SSH_EXIT.json')
 assert exit_record.get('exit')==0 and not exit_record.get('timeout',False)
 assert command['gate_result_sha256']==a.gate_result_sha256 and command['linux_proof_sha256']==a.linux_proof_sha256 and command['destination']==archive.parent.as_posix()
 remote=(S/'transport_remote.py').read_text(encoding='utf-8').replace('__GATE_RESULT_SHA256__',a.gate_result_sha256).replace('__LINUX_PROOF_SHA256__',a.linux_proof_sha256)
 assert hashlib.sha256(remote.encode()).hexdigest()==command['remote_source_sha256'] and (A/'TRANSPORT_REMOTE_BOUND.py').read_text(encoding='utf-8')==remote
 summaries=[json.loads(line) for line in (A/'TRANSPORT_STDERR.txt').read_text(encoding='utf-8').splitlines() if line.startswith('{')]
 assert summaries==[{'archive_sha256':transport['archive_sha256'],'archive_bytes':transport['archive_bytes'],'members':22}]
 q=read(T)
 for rel,pin in q['files'].items():assert sha(R/rel)==pin['sha256'] and (R/rel).stat().st_size==pin['bytes'],rel
 for name in ('evaluator','saved_science'):del q['files'][q[name]]
 q['evaluator']=(C/'originals/evaluator.py').relative_to(R).as_posix();q['saved_science']=(S/'saved_science.py').relative_to(R).as_posix()
 q['linux_sha256']=a.linux_proof_sha256;q['transport_sha256']=a.transport_proof_sha256;q['exact_ids']=k.IDS
 diag=R/q['setup_extractor'];node=next(n for n in ast.parse(diag.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='setup_nodes')
 ns={'ast':ast};exec(compile(ast.Module(body=[node],type_ignores=[]),'<original setup extractor only>','exec'),ns)
 prefix=ns['setup_nodes']((S/'verify_arrays.py').read_text());q['setup_AST_sha256']=hashlib.sha256(ast.dump(ast.Module(body=prefix,type_ignores=[]),include_attributes=False).encode()).hexdigest()
 qfiles=list(PINNED)+[C/rel for rel in package]+[S/rel for rel in static]+[A/name for name in ('LINUX_SAVED_CHECK.json','TRANSPORT_VERIFICATION.json','TRANSPORT_COMMAND.json','TRANSPORT_SSH_EXIT.json','TRANSPORT_STDERR.txt','TRANSPORT_REMOTE_BOUND.py')]
 for p in qfiles:
  assert p.is_file() and p.stat().st_size<2_000_000
  q['files'][p.relative_to(R).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
 failure='tmp/celeba_flgmm_closed47_root_execution_20261011/saved_acceptance_actual001/OFFSERVER_ARRAY_REFIT_CHECK.failure.json'
 assert q['files'][failure]['sha256']==q['original_failure_sha256']=='c93d11ee41400659f74597beb1ca0cab19a64b647c805ee0534051fb55f41812' and read(R/failure)['completed']==4
 q['actual_binding']={'gate_result_sha256':a.gate_result_sha256,'archive_path':archive.as_posix(),'archive_sha256':transport['archive_sha256'],'archive_bytes':transport['archive_bytes'],'archive_members':22,'payload_members':21,'members':members,'member_arrays_rehashed_by_binder':False,'new_fit':0,'root_adoption':False}
 q['scope']='Only saved outputs for actual native86 S-DFA exact9; prior81 unchanged. Original Linux whole check supplies root-refit evidence; previous Windows47 refit remains failed (completed4). Binder checks compact proof links only; zero-fit audit verifies saved arrays. No Windows recalibration or scientific adoption claim.'
 with (S/'SOURCE_PINS.json').open('x',encoding='utf8') as f:json.dump(q,f,indent=2,allow_nan=False);f.write('\n')
 files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(S.iterdir()) if p.is_file()}
 with (S/'FILES_SHA256.json').open('x',encoding='utf8') as f:json.dump({'files':files,'scope':q['scope']},f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'source_seal_sha256':sha(S/'FILES_SHA256.json'),'actual_linux_sha256':q['linux_sha256'],'actual_transport_sha256':q['transport_sha256'],'records':9,'new_fit':0,'root_adoption':False}))

if __name__=='__main__':main()
