from pathlib import Path
import argparse,base64,hashlib,json,subprocess
H=Path(__file__).resolve().parent; C=H.parent; R=C.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal=json.loads((C/'FILES_SHA256.json').read_bytes()); package=sha(C/'FILES_SHA256.json')
review=C/'ROOT_SOURCE_REVIEW.json'
parser=argparse.ArgumentParser();parser.add_argument('--source-review-sha256',required=True);args=parser.parse_args()
assert sha(review)==args.source_review_sha256
assert json.loads(review.read_bytes())['source_adoptable'] is True and json.loads(review.read_bytes())['package_sha256']==package
members={}
for rel in list(seal['files'])+['FILES_SHA256.json']:
 p=C/rel;b=p.read_bytes();h=sha(p)
 if rel in seal['files']:assert h==seal['files'][rel]['sha256']
 members['source/'+rel]={'base64':base64.b64encode(b).decode(),'sha256':h,'bytes':len(b)}
for rel,p in [('ROOT_SOURCE_REVIEW.json',review),('source/originals/saved_science.py',R/'tmp/celeba_flgmm_three_view_closed_batch_preparation_20261011/originals/saved_science.py')]:
 b=p.read_bytes();members[rel]={'base64':base64.b64encode(b).decode(),'sha256':sha(p),'bytes':len(b)}
assert members['source/originals/saved_science.py']['sha256']=='d512e5b2b6614b762d921dd94b2b5162687c0bbddde4caaf8b584b3b22dba745'
old=R/'tmp/celeba_flgmm_closed47_root_execution_20261011/SOURCE_DEPLOYMENT_remote.py'
assert sha(old)=='7a1eaff809262b4c2264529374d18433154a2b8ee856a6e6a18ca09549b25156'
source="assert __import__('hashlib').sha256(__import__('pathlib').Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'\n"+old.read_text().replace('celeba_flgmm_three_view_closed_batch_20261011','hybrid_FFlip10_three_view_prepare_20261011')
compile(source,'deploy_remote','exec')
(H/'DEPLOY_REMOTE.py').write_text(source)
code="import base64;exec(compile(base64.b64decode('"+base64.b64encode(source.encode()).decode()+"'),'<source-deploy>','exec'))"
cmd=['ssh','-p','60350','-o','BatchMode=yes','root@89.22.197.55','python3 -c "'+code+'"']
assert not (H/'DEPLOY_COMMAND.json').exists()
(H/'DEPLOY_COMMAND.json').write_text(json.dumps({'argv':cmd,'package_sha256':package,'members':len(members)},indent=2))
p=subprocess.run(cmd,input=json.dumps({'base':'/workspace/guardfed_checks/hybrid_FFlip10_three_view_prepare_20261011','members':members}).encode(),capture_output=True,timeout=60)
(H/'DEPLOY.stdout').write_bytes(p.stdout);(H/'DEPLOY.stderr').write_bytes(p.stderr)
(H/'DEPLOY_EXIT.json').write_text(json.dumps({'exit':p.returncode})+'\n')
assert p.returncode==0,p.stderr.decode(errors='replace')
j=json.loads(p.stdout);(H/'SOURCE_DEPLOYMENT.json').write_text(json.dumps(j,indent=2)+'\n')
print(json.dumps({'status':j['status'],'members':len(members),'source_review_sha256':sha(review)}))
