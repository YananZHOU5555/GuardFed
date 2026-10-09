"""Pure-data final-six gates; importing this module performs no acceptance or I/O."""
EXPECTED_CHAIN='3844e24261b7e3a89a550b0fcb0016bbf2fb1b8a579f6e51b6a3e5afff23032b'
EXPECTED_PACKAGE='aec95ceb5e8c9b7aa9cec89e9d70d2700c2f242c29e918792088648d6269bad4'
SOURCE_PINS={
 'PACKAGE_SHA256.json':EXPECTED_PACKAGE,
 'source/protocol.json':'63bfa927f841fee637b4d80e08e0f902a493361eae28b999f13302dc69c4304c',
 'jobs/manifest.json':'d888509df45f2392c38cc8a73c9a01fae1a4b3af104857f391a32cdc3a950632',
 'frozen_score.py':'1b31c0322b06bc1901f2a2f49a2b7b3524fed0164b6191d8bb9a5ed7e5f24b27',
 'screen_common.py':'dea4756c20a364156dd3bd0bebd719e0f82b11f34f5d7d9e42bbce18a84983aa',
 'source/accept_result.py':'000b2711673431bd83487eeee0d79913e71805d221bb25829646538f630a06c5',
 'source/worker.py':'a3fe21a5a334b63de9c2167ed28293cd7f38d4411edf11de7d226d895b550fdc',
 'source/flgmm_adapter.py':'3d03b2ac53883ffecf40a5025bf6a5440824ccbf7c05c032535de18b74ff9453'}

def validate_previous(previous,chain_sha,package_sha,manifest):
 assert chain_sha==EXPECTED_CHAIN and package_sha==EXPECTED_PACKAGE
 assert previous['status']=='PARTIAL_STRICT_OFFSERVER_ROOT_RECORD_REVIEW_ADOPTED'
 assert previous['accepted_total']==len(previous['accepted_job_ids'])==len(set(previous['accepted_job_ids']))==26
 assert previous['planned']==32 and previous['selected_recipe'] is None
 assert previous['test_evaluated'] is False and previous['formal100_started'] is False
 planned=[r['id'] for r in manifest['jobs']]
 assert len(planned)==len(set(planned))==32 and set(previous['accepted_job_ids'])<=set(planned)
 wanted=[identity for identity in planned if identity not in previous['accepted_job_ids']]
 assert len(wanted)==6
 return wanted

def validate_snapshot(snapshot,manifest):
 assert all(snapshot['source_sha256'][name]==digest for name,digest in SOURCE_PINS.items())
 queue=snapshot['queue'];assert queue['completed']==32 and queue['pending']==0 and queue['active']==[]
 rows=snapshot['rows'];planned={r['id'] for r in manifest['jobs']}
 assert len(rows)==len({r['id'] for r in rows})==32 and {r['id'] for r in rows}==planned
 assert not snapshot['failure_paths']
 for row in rows:
  assert row['active'] is False and row['progress']['round']==70
  assert row['result_exists'] and row['acceptance_exists'] and row['screen_identity_exists']

def validate_live(snapshot,manifest,resource):
 queue=snapshot['queue'];assert queue['completed']==32 and queue['pending']==0 and queue['active']==[]
 rows=snapshot['flgmm']['rows'];planned={r['id'] for r in manifest['jobs']}
 assert len(rows)==len({r['id'] for r in rows})==32 and {r['id'] for r in rows}==planned
 assert all(r['terminal_acceptance'] and not r['active'] and r['progress']['round']==70 for r in rows)
 assert not any('--job-id' in r['argv'] and planned.intersection(r['argv']) for r in resource)
