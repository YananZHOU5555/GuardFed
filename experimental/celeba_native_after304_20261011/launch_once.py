from pathlib import Path
import datetime,hashlib,json,subprocess,sys
B=Path(__file__).resolve().parent; R=B.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent=R/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/verified_ledger.json'
assert sha(parent)=='44afd080678b145eb1d014926d28ff60d46ce3c0c006ad98566e9e48fc64b306'
assert len(json.loads(parent.read_bytes())['entries'])==44
assert not (B/'PARENT_LEDGER.json').exists() and not (B/'ACTUAL_COMMAND.json').exists()
(B/'PARENT_LEDGER.json').write_bytes(parent.read_bytes())
source=B/'run_once.py'; compile(source.read_bytes(),str(source),'exec')
argv=[sys.executable,'-B',str(source)]
review={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(source),'original_sha256':'9745525453d797b72a36aee66bade4fb3d3db04345fecd93535b68f44d142e88','scope':'namespace-only; original strict inspection/archive/offserver loops retained; no training, refit, CNN or test','parent_ledger_sha256':sha(parent),'guide_read_complete':True,'argv':argv}
(B/'ROOT_COLLECTOR_SOURCE_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf8')
(B/'ACTUAL_COMMAND.json').write_text(json.dumps(argv,indent=2)+'\n',encoding='utf8')
with (B/'ACTUAL.stdout.json').open('xb') as out,(B/'ACTUAL.stderr.log').open('xb') as err:
    p=subprocess.run(argv,stdout=out,stderr=err)
(B/'ACTUAL.EXIT.json').write_text(json.dumps({'returncode':p.returncode,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})+'\n',encoding='utf8')
print(json.dumps({'returncode':p.returncode,'source_sha256':sha(source),'stdout':str(B/'ACTUAL.stdout.json')}))
raise SystemExit(p.returncode)
