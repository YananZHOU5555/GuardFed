"""Exact compact Git48 list after root supplies the final snapshot and helper paths."""
from pathlib import Path
import argparse,json,sys
sys.dont_write_bytecode=True
from publish_increment48 import ROOT,H,PARENT,BRANCH,base,configure
TRAIN='docs/server_deployment_20260923/training_20260923/'
NATIVE='docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T072729Z'
TRANSPORT='tmp/celeba_remaining620_A20_transport_20261010'
ADOPTION='tmp/celeba_mechanism_remaining620_A20_root_adoption_20261010'
BINDING='tmp/celeba_mechanism_A20_root_binding_20261010'
GRADIENT='tmp/celeba_gradient64_delta_after5_20261010'
AUTHOR='docs/server_deployment_20260923/revision_20260923/rebuttal_adaptations_20261010'
LOGO='docs/server_deployment_20260923/training_20260923/celeba_logofair100_accepted_20261010'
LOGOREVIEW='tmp/celeba_logofair100_root_review_20261010'
LOGOSUMMARY='tmp/celeba_logofair100_summary_20261010'
A20='docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_IID_two_scenes20_20261010'
A20REVIEW='tmp/celeba_A20_table_root_review_v2_20261010'
A20FAILURE='tmp/celeba_A20_table_root_review_20261010'
DIRECTORIES=[NATIVE,TRANSPORT,ADOPTION,BINDING,GRADIENT,AUTHOR,LOGO,LOGOREVIEW,LOGOSUMMARY,A20,A20REVIEW,A20FAILURE,'tmp/publication_increment48_v2_20261010']

MUTABLE=[TRAIN+'TRAINING_STATE.json',TRAIN+'RUNNING.md',TRAIN+'REBUTTAL_COMPLETION_20261009.md',
 TRAIN+'server_reactivation_20261009/MONITOR_HANDOFF.md','docs/返修实验总览.md',
 'tmp/update_reactivation_state_20261009.py','tmp/update_completion_current_20261009.py','tmp/update_overview_closure100_root_20261009.py']
EXPLICIT=['tmp/adopt_gradient64_after5_root_20261010.py','tmp/adopt_logofair100_root_20261010.py','tmp/adopt_A20_table_root_20261010.py',
 TRAIN+'publication_closed_increment47_verified_20261010.json',
 'tmp/publication47_root_20261010/REMOTE_VERIFICATION.json','tmp/publication47_root_20261010/COMMIT_RECEIPT.json']
PINS={'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T072729Z/ROOT_DELTA_VERIFICATION.json': 'a396108987b67fab13376c00c678be79f9eb2fe57ace6b628a425b48474ce98d', 'tmp/celeba_remaining620_A20_transport_20261010/FILES_SHA256.json': '66b9e054a005786ca3d8d0e32547bbd83d7a8c66520b31b611a2bde759d6f9cc', 'tmp/celeba_mechanism_remaining620_A20_root_adoption_20261010/ROOT_ADOPTION.json': 'e088871fbd98cbc9415cc79a44626667a532389ce0b6dddbaaf2ab25f72a4979', 'tmp/celeba_mechanism_remaining620_A20_root_adoption_20261010/MECHANISM220_INDEX.json': 'af414a0ac6705230c53d324cd1f51e7a76b893dabd7416bb6914a6690b6fdddc', 'tmp/celeba_mechanism_A20_root_binding_20261010/FILES_SHA256.json': 'de223118febfe8e815efbc38d81739d0a54ec889a157f1cdb83b56c3c6c250f9', 'tmp/celeba_gradient64_delta_after5_20261010/ROOT_ADOPTION_REVIEW.json': '99a07fefad41e0f441288f86c5335e175eb5992983fec047ab3a1c9f1eacea57', 'tmp/celeba_gradient64_delta_after5_20261010/DELIVERY_FILES_SHA256.json': '551bf7b32961843ad15fc535aeddf1115402d7d3a3d9d8642ed118d3f7b9d0ec', 'docs/server_deployment_20260923/revision_20260923/rebuttal_adaptations_20261010/ROOT_REVIEW.json': 'bcd0c0e51a99572eb10c9fef0a8305f067cdaae78359349e4afcc9f8baaf83f4', 'docs/server_deployment_20260923/revision_20260923/rebuttal_adaptations_20261010/FILES_SHA256.json': 'f87eb44eb85ac2acc333641e2764d72055191793cc96edd329683a3333fcc44d', 'docs/server_deployment_20260923/training_20260923/celeba_logofair100_accepted_20261010/ROOT_ADOPTION.json': '1529a852b3bd02561d274fdea832186706bb09725c8d02594d09114f44f977c2', 'tmp/celeba_logofair100_root_review_20261010/ROOT_REVIEW.json': 'b4fc447ba86c2fd40b54a1100c1ca2cdba890546b383feb5ddfa6bae19958610', 'tmp/celeba_logofair100_root_review_20261010/FILES_SHA256.json': '7544c45fe44ebdbad6be34bd8b27d811b0b39f3e0cc8cfc6a6b5467c12efd0d9', 'tmp/celeba_logofair100_summary_20261010/FILES_SHA256.json': 'af28607a600ce4ccb03d2ef88bc76e5b83bf3c0415040344950f2f684716a1a1', 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_IID_two_scenes20_20261010/ROOT_VERIFICATION.json': 'dbc2f8fe481c2a049f4e556a500c29d3122fc8392d69e4ab2bc131c0e3f4b3c0', 'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_IID_two_scenes20_20261010/FILES_SHA256.json': '3841a7498dad4a3dc6a0bf3b8262510082cd042671850b5f56320a079c012ef8', 'tmp/celeba_A20_table_root_review_v2_20261010/ROOT_ARITHMETIC_REVIEW.json': 'e6cee89fd0406b006c008200d6fff5a4966967b3dea3271b7a5b5a682bfa8051', 'tmp/celeba_A20_table_root_review_v2_20261010/FILES_SHA256.json': '581a71ded37aaa4fc0d41b8c1f90c45fe5a98eda8060fa7f238aceab11820359', 'tmp/celeba_A20_table_root_review_20261010/FILES_SHA256.json': '02aed724d7ddee7243b7c5f6b22fb30a6901a05a41ec56e6234381c58e2f5ec3', 'tmp/celeba_A20_table_root_review_20261010/REVIEW_FAILURE.json': '51314f65a18108e7d090f4903661d49a411649a9fe674d8a500a0c5342711ca0'}


def build(extras):
    configure({'native':220,'three_view':220})
    receipt=Path('F:/YananResearchStorage/GuardFed/git_publication/increment47/stage001/STAGE_RECEIPT.json')
    assert H(receipt.read_bytes())=='466f5dfc114d3462a0b89fb7ef6786769f0af0bec2010b809d34970aaeb2c79e'
    parent=json.loads((ROOT/EXPLICIT[-2]).read_bytes());assert parent['commit']==PARENT and parent['receipt_sha256']==H(receipt.read_bytes())
    blobs={}; previous_commit=None
    for number,expected in [(45,'7e09337e152a0f16a7b0242a24b26f2e501aa56ed7a949255733ba35de0b5f08'),(46,'775aa77d1249e4cd739f528244f2aaf826a17ff42475940214a44619f3a5a85c'),(47,'466f5dfc114d3462a0b89fb7ef6786769f0af0bec2010b809d34970aaeb2c79e')]:
        prior_receipt=Path(f'F:/YananResearchStorage/GuardFed/git_publication/increment{number}/stage001/STAGE_RECEIPT.json')
        actual=json.loads((ROOT/f'tmp/publication{number}_root_20261010/REMOTE_VERIFICATION.json').read_bytes())
        assert H(prior_receipt.read_bytes())==actual['receipt_sha256']==expected
        if previous_commit is not None:assert actual['parent']==previous_commit
        previous_commit=actual['commit']
        blobs.update({r['path']:r for r in json.loads(prior_receipt.read_bytes())['blobs']})
    assert previous_commit==PARENT
    for name,sha in PINS.items():assert H((ROOT/name).read_bytes())==sha,name
    for directory,name in ((TRANSPORT,'FILES_SHA256.json'),(BINDING,'FILES_SHA256.json'),(AUTHOR,'FILES_SHA256.json'),(GRADIENT,'DELIVERY_FILES_SHA256.json'),(LOGOREVIEW,'FILES_SHA256.json'),(LOGOSUMMARY,'FILES_SHA256.json'),(A20,'FILES_SHA256.json'),(A20REVIEW,'FILES_SHA256.json'),(A20FAILURE,'FILES_SHA256.json')):
        seal=ROOT/directory/name
        for name,pin in json.loads(seal.read_bytes())['files'].items():
            b=(seal.parent/name).read_bytes();assert H(b)==pin['sha256'] and len(b)==pin['bytes']
    names=set(MUTABLE+EXPLICIT+extras);omitted=[]
    for directory in [ROOT/n for n in DIRECTORIES]:
        for p in directory.rglob('*'):
            if not p.is_file():continue
            n=p.relative_to(ROOT).as_posix()
            if p.suffix.lower() not in {'.json','.py','.md','.csv','.patch','.sha256'} or '__pycache__' in p.parts or p.name=='SERVER_GUIDE.md':
                omitted.append(n);continue
            names.add(n)
    files=[];refs=[]
    for n in sorted(names):
        _,b=base.source(n,[n]);e=dict(source=n,sha256=H(b),bytes=len(b));destination=base.destination(n)
        if destination in blobs and all(blobs[destination][k]==e[k] for k in ('sha256','bytes')) and n not in MUTABLE:
            refs.append(dict(e,destination=destination,parent_commit=PARENT))
        else:files.append(e)
    selected={r['source']:r for r in files}
    paths={'current_state':MUTABLE[0],'native220':NATIVE+'/ROOT_DELTA_VERIFICATION.json',
      'replay220':ADOPTION+'/ROOT_ADOPTION.json','gradient10':GRADIENT+'/ROOT_ADOPTION_REVIEW.json','authorpatch':AUTHOR+'/ROOT_REVIEW.json','logofair100':LOGO+'/ROOT_ADOPTION.json','A20table':A20+'/ROOT_VERIFICATION.json'}
    bindings={}
    for role,n in paths.items():
        d=json.loads((ROOT/n).read_bytes());expect=base.REQUIRED_FACTS[role]
        for k,v in expect.items():assert base.pointer(d,k)==v,(role,k)
        bindings[role]=dict(path=n,sha256=selected[n]['sha256'],expect=expect)
    return dict(status='SOURCE_PREPARED_ACTUAL_LIST_NOT_PUBLISHED',parent=PARENT,branch=BRANCH,
      scope='CLOSED220_REPLAY220_GRADIENT10_LOGO100_A20_NO_BULK',accepted={'native':220,'three_view':220},
      bindings=bindings,optional_bindings={},test_started=False,goal_complete=False,
      allowed_paths=sorted(selected),files=files,parent_recovery_references=refs,
      omitted_bulk_or_logs=omitted,large_artifacts_git_restore=False,
      limitation='Only new native2, replay A8, gradient5, root-adopted LoGo100 and A20 table; future full reply/ten-method table only actual root extras; bulk excluded')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--extras',type=Path,required=True,help='Root-reviewed JSON list of exact small source/snapshot/review paths')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert a.out.resolve().is_relative_to(ROOT/'tmp/publication_increment48_v2_root_inputs_20261010')
    extras=json.loads(a.extras.read_bytes());assert isinstance(extras,list) and all(isinstance(n,str) for n in extras)
    d=build(extras);a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(path=str(a.out),sha256=H(a.out.read_bytes()),files=len(d['files']),bytes=sum(e['bytes'] for e in d['files']))))
