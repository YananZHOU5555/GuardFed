from pathlib import Path
import hashlib,json,tarfile,datetime
p=Path('/workspace/guardfed_checks/celeba_native_mismatch_diagnostic_execution_20261009')
out=p/'attempt2_backup';out.mkdir(exist_ok=False)
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
files=sorted(f for f in (p/'attempt2').rglob('*') if f.is_file() and f.name!='deployment.tar.gz' and '__pycache__' not in f.parts)+sorted(f for f in (p/'runs').rglob('*') if f.is_file())
assert not (p/'attempt2/COMPLETED.json').exists()
assert (p/'attempt2/failure.json').exists()
assert len(list((p/'runs').glob('*/validation_predictions.npz')))==1
members={str(f.relative_to(p)):{'sha256':sha(f),'bytes':f.stat().st_size} for f in files}
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'GPU_INFERENCE_NATIVE_MATCH_POSTPROCESS_FAILED_PRESERVED','members':members,'prior_source_archive_sha256':'f45ff6a479386ce2e53c533d11d9b23f40036b895f4d363d1df22063a0e66988','prepared_seal_sha256':'25d6c79f1d70b40f762bd9a372f1e9d2c2f64d8e9e82919c85f1667ce9a0bc61','no_cohort_acceptance':True}
(out/'MEMBERS.json').write_text(json.dumps(record,indent=2)+'\n')
with tarfile.open(out/'evidence.tar.gz','w:gz') as t:
 for f in files:t.add(f,arcname=str(f.relative_to(p)),recursive=False)
 t.add(out/'MEMBERS.json',arcname='MEMBERS.json',recursive=False)
receipt={'archive_sha256':sha(out/'evidence.tar.gz'),'inventory_sha256':sha(out/'MEMBERS.json'),'members':len(members),'archive_bytes':(out/'evidence.tar.gz').stat().st_size}
(out/'BACKUP.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
