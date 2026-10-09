"""Promote the sealed single C scene after independent source/statistic checks."""
from pathlib import Path
import datetime,hashlib,json,statistics
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_mechanism_native_C_Benign10_20261009'
OUT=ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_C_Benign10_20261009'
REVIEW=ROOT/'tmp/celeba_native_C_Benign10_independent_review_20261009/ROOT_INDEPENDENT_REVIEW.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(BASE/'FILES_SHA256.json')=='488bd8bec04ea180e5252966ea7d24f1b057bb9a203ac7b8974557faf63fe7ad'
assert sha(REVIEW)=='fb1b8601be263b18b9617a0a141e0e2b963e8eb21d60b5d287ef8530ef706c1c'
seal=read(BASE/'FILES_SHA256.json');assert len(seal['members'])==13
for row in seal['members']:
 assert sha(BASE/row['path'])==row['sha256'] and (BASE/row['path']).stat().st_size==row['bytes']
for path,pin in read(BASE/'INPUTS.json')['files'].items():
 assert sha(ROOT/path)==pin['sha256'] and (ROOT/path).stat().st_size==pin['bytes']
records=read(BASE/'records.json')['records'];index={(r['variant'],r['seed']):r for r in records}
assert len(index)==len(records)==20
source=read(ROOT/read(BASE/'INPUTS.json')['inspection'])['records']
assert all(r in source for r in records)
table=read(BASE/'tables.json');cells=0;scalars=0
for panel in table['panels']:
 for row in panel['rows']:
  assert len(panel['seeds'])==row['n']==row['expected_n'] and row['complete']
  for metric in ('accuracy_pct','aeod','aspd'):
   values=[index['minus_C',s][metric]-index['Full',s][metric] if row['variant']=='minus_C minus Full' else index[row['variant'],s][metric] for s in panel['seeds']]
   assert row[metric]['mean']==statistics.mean(values)
   assert row[metric]['sample_sd_ddof1']==statistics.stdev(values)
   scalars+=2;cells+=1
assert (scalars,cells)==(54,27)
assert not OUT.exists();OUT.mkdir(parents=True)
for row in seal['members']+[dict(path='FILES_SHA256.json',sha256=sha(BASE/'FILES_SHA256.json'))]:
 (OUT/row['path']).write_bytes((BASE/row['path']).read_bytes())
 assert sha(OUT/row['path'])==row['sha256']
(OUT/'INDEPENDENT_REVIEW.json').write_bytes(REVIEW.read_bytes())
proof=dict(status='ROOT_NATIVE_C_SINGLE_SCENE_SEAL_IDENTITIES_AND_STATISTICS_ADOPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 source=str(BASE.relative_to(ROOT)),source_seal_sha256=sha(BASE/'FILES_SHA256.json'),source_members_verified=13,
 independent_review_sha256=sha(REVIEW),original_input_pins_verified=19,original_native_inspection_sha256=sha(ROOT/read(BASE/'INPUTS.json')['inspection']),
 displayed_checkpoint_pairs=10,display_cells=cells,mean_SD_scalars=scalars,single_scene='IID Benign',variant='minus_C',
 seed_panels=[10,9,6],other_C_scenes_complete=False,original_U100_unchanged=True,new_CNN=0,new_training=0,test=False,
 mixed_environment_and_selection_history_retained=True,necessity_or_causal_claim=False,goal_complete=False)
with (OUT/'ROOT_VERIFICATION.json').open('x',encoding='utf8',newline='\n') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
print(json.dumps(dict(status=proof['status'],root_proof_sha256=sha(OUT/'ROOT_VERIFICATION.json'),directory=str(OUT))))
