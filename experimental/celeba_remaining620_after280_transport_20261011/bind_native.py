"""Bind one actual adopted native288 proof before any future execution; no SSH/arrays."""
from pathlib import Path
import argparse,hashlib,json
from verify_native_inputs import validate
H=Path(__file__).resolve().parent;R=H.parents[1]
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    assert not (H/'EXECUTION_INPUTS.json').exists() and not (H/'EXECUTION_FAILURE.json').exists()
    assert sha(H/'SOURCE_FILES_SHA256.json')==a.prepared_seal_sha256
    for name,pin in read(H/'SOURCE_FILES_SHA256.json')['files'].items():
        assert sha(H/name)==pin['sha256'] and (H/name).stat().st_size==pin['bytes']
    proof_path=a.native_root.resolve();assert proof_path.is_file() and sha(proof_path)==a.native_root_sha256
    assert proof_path.is_relative_to(R/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009')
    proof=read(proof_path);assert proof_path.parent.name==proof['tag'] and proof_path.name=='ROOT_DELTA_VERIFICATION.json'
    p=read(H/'PREPARED.json')
    e=dict(native_root_verified=True,native_root_path=str(proof_path),native_root_sha256=a.native_root_sha256,
        native_inspection_path=str(proof_path.parent/'inspection/inspection.json'),native_inspection_sha256=proof['inspection_sha256'],
        native_ledger_path=str(proof_path.parent/'verified_ledger.json'),native_ledger_sha256=proof['ledger_sha256'],
        exact_candidate_ids=p['candidate_ids'],source_seal_sha256=a.prepared_seal_sha256,
        native_archive_roots=p['known_native_archive_roots']+[dict(path=str(proof_path),sha256=a.native_root_sha256)],
        root_native_total=288,transport_target_only=8,proposed_replay_total=288,Full_inference=0,new_training=0,test=False)
    validate(e,p)
    with (H/'EXECUTION_INPUTS.json').open('x',encoding='utf8',newline='\n') as f:f.write(json.dumps(e,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='ACTUAL_NATIVE288_BOUND_NOT_TRANSPORTED_NOT_ADOPTED',execution_inputs_sha256=sha(H/'EXECUTION_INPUTS.json'))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-root',type=Path,required=True);p.add_argument('--native-root-sha256',required=True)
    p.add_argument('--prepared-seal-sha256',required=True);main(p.parse_args())
