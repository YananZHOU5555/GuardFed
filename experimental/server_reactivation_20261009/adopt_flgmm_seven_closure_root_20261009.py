"""Root actual archive/metadata-chain adoption after saved-output verification."""
from pathlib import Path
import argparse,datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=Path,required=True);parser.add_argument('--receipt-sha256',required=True);parser.add_argument('--offserver-sha256',required=True);a=parser.parse_args()
folder=a.attempt.resolve();assert folder.parent==(ROOT/'tmp/celeba_flgmm_seven_canary_closure_20261009').resolve()
receipt=folder/'BACKUP_RECEIPT.json';proof=folder/'verified/OFFSERVER_VERIFICATION.json'
assert sha(receipt)==a.receipt_sha256 and sha(proof)==a.offserver_sha256
r=read(receipt);p=read(proof)
assert p['status']=='PASS_FULL_MEMBER_SHA_AND_ORIGINAL_SAVED_COMPARISON' and p['receipt_sha256']==sha(receipt)
assert sha(folder/'seven_canaries.tar.gz')==r['archive_sha256']==p['archive_sha256']
assert p['member_count']==r['archive_members'] and p['CNN_calls']==p['formal_table_samples']==0
inventory=folder/'verified/MEMBERS.json';assert sha(inventory)==r['inventory_sha256']==p['inventory_sha256']
pins=read(inventory)['files'];assert len(pins)+1==p['member_count']
for n,pin in pins.items():assert sha(folder/'verified'/n)==pin['sha256'] and (folder/'verified'/n).stat().st_size==pin['bytes']
gate=folder/'verified/stage/GATE_ACCEPTANCE.json';local=p['local']
assert (local['accepted_new'],local['same_horizon_pairs'],local['total_runs'],local['rounds'],local['formal_table_samples'])==(5,2,7,3,0)
assert sha(gate)==local['gate_sha256']==r['gate_sha256'] and local['package_sha256']==r['package_sha256']=='6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230'
remote=read(folder/'verified/evidence/REMOTE_STRICT.json');assert remote['source_data_before']==remote['source_data_after'] and remote['pair_ids']==local['pair_ids']
startup=ROOT/'tmp/celeba_flgmm_fullcoverage_binding_20261009/ROOT_CANARY_STARTUP.json'
assert sha(startup)=='22486ecfd6e4012fcb30a27c120574dee918c62069dec55c6ad225e24d68d68b'
authorization=folder/'verified/stage/EXECUTION_AUTHORIZATION.json'
assert sha(authorization)==read(ROOT/read(startup)['start_receipt_path'])['authorization_sha256']
assert read(authorization)['scope']=='seven_same_horizon_3round_canaries'
result=dict(status='ROOT_SEVEN_CANARY_CLOSURE_ADOPTED',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    package_sha256=r['package_sha256'],gate_sha256=sha(gate),offserver_sha256=sha(proof),receipt_sha256=sha(receipt),archive_sha256=r['archive_sha256'],
    offserver_path=proof.relative_to(ROOT).as_posix(),attempt_path=folder.relative_to(ROOT).as_posix(),archive_members_verified=p['member_count'],
    canary_authorization_sha256=sha(authorization),accepted_new_canaries=5,same_horizon_pairs=2,total_canary_runs=7,rounds=3,
    original_saved_comparison_reexecuted=True,source_data_before_after_exact=True,models_repacked_from_70round=0,
    formal_table_samples=0,formal100_started=False,final_test=False,negative_results_retained=True,
    limitations='Three-round selected Tg20 gate covers early implementation/attack and RNG interfaces; it cannot establish later monitoring or universal70-round equivalence. Local verifier runtime is explicitly distinct from training.')
out=ROOT/'tmp/celeba_flgmm_fullcoverage_binding_20261009/ROOT_SEVEN_CANARY_CLOSURE.json'
with out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=sha(out))))
