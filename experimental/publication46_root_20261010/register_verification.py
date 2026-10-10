"""Register actual Git46 remote bytes without claiming later changes were published."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
TRAIN=ROOT/'docs/server_deployment_20260923/training_20260923'
p=HERE/'REMOTE_VERIFICATION.json';d=json.loads(p.read_bytes())
c=json.loads((HERE/'COMMIT_RECEIPT.json').read_bytes())
assert d['status']=='COMMITTED_BLOB_BYTES_PASS_AND_REMOTE_BRANCH_PASS'
assert d['blobs_verified']==c['staged_blobs'] and d['commit']==c['commit']
assert d['parent']=='e7ea15c5d2e77c40be932e20f0efe941e584fb13'
assert d['accepted']=={'native':212,'three_view':212}
proof=dict(status='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS',commit=d['commit'],branch=d['branch'],parent=d['parent'],
    verified_utc=d['root_review_utc'],committed_blobs_sha256_verified=d['blobs_verified'],changed_paths=d['changed_paths'],
    original_proof_path=p.relative_to(ROOT).as_posix(),original_proof_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    external_checkout=d['external_checkout'],acceptance_cutoff=d['accepted'],test_started=False,goal_complete=False,
    later_local_acceptance_is_not_in_this_commit=True,LoGoFair100_startup_retained_via_parent=True,
    additional_compact_accepted=dict(Hybrid_screen32=27,FLGMM_new96=28,LoGoFair_screen32=32,gradient_screen64=5),
    A_IID_Benign_paired_table=10,gradient_fullcoverage_source_only=True,
    LoGoFair100_new_independent_accepted=0,compact_evidence_only_bulk_at_original_F_or_server_paths=True)
out=TRAIN/'publication_closed_increment46_verified_20261010.json'
with out.open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),commit=d['commit'])))
