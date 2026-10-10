"""Retained stdlib-only actual-binding guard fixtures; no scientific execution."""
from pathlib import Path
import copy, importlib.util, json
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('F_FFlip10_identity',H/'verify_native_inputs.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=m.read(H/'PREPARED.json');old=m.read(p['native312_inspection_path']);ledger=m.read(p['native312_ledger_path'])
realread=m.read;realsha=m.sha

def fixture(extra=0):
    ids=p['candidate_ids'][2:]+[f'minus_F_IID_FedSA_seed{91001+i}' for i in range(extra)]
    rows=[dict(id=i,variant='minus_F',distribution='IID',attack='F Flip',seed=int(i[-5:]),prediction_support=dict(prediction_count=19867)) for i in p['candidate_ids'][2:]]
    rows.extend(dict(id=i,variant='minus_F',distribution='IID',attack='FedSA',seed=int(i[-5:]),prediction_support=dict(prediction_count=19867)) for i in ids[8:])
    proof=dict(root_adopted=True,total_new_strict_and_offserver=312+len(ids),test=False,inspection_sha256='fixture-inspection',ledger_sha256='fixture-ledger',previous_ledger_sha256=p['native312_ledger_sha256'],new_ids=ids)
    d={'fixture-root':proof,'fixture-inspection':dict(records=copy.deepcopy(old['records'])+rows,accepted_new_ids=old['accepted_new_ids']+ids),'fixture-ledger':dict(entries=copy.deepcopy(ledger['entries'])+[{'fixture':True}])}
    e=dict(native_root_verified=True,exact_candidate_ids=p['candidate_ids'],native_root_path='fixture-root',native_root_sha256='fixture-root',native_inspection_path='fixture-inspection',native_inspection_sha256='fixture-inspection',native_ledger_path='fixture-ledger',native_ledger_sha256='fixture-ledger',native_archive_roots=copy.deepcopy(p['known_native_archive_roots'])+[dict(path='fixture-root',sha256='fixture-root')],root_native_total=312+len(ids),transport_target_only=10,proposed_replay_total=320)
    q=copy.deepcopy(p)
    return d,e,q

def run(case,edit=None,reject=False,extra=0):
    d,e,q=fixture(extra)
    if edit:edit(d,e,q)
    m.read=lambda path: d[str(path)] if str(path) in d else realread(path)
    m.sha=lambda path: str(path) if str(path) in d else realsha(path)
    try:m.validate(e,q)
    except AssertionError:
        assert reject,case
    else:assert not reject,case
    return dict(case=case,expected_rejection=reject,passed=True)
cases=[run('native320_two_old_eight_new_FFlip'),run('native321_extra_FedSA_excluded',extra=1),run('native322_extra_two_FedSA_excluded',extra=2),
 run('missing_selected_new_FFlip',lambda d,e,q:d['fixture-root']['new_ids'].pop(),True),
 run('missing_old_native_archive',lambda d,e,q:e['native_archive_roots'].pop(0),True),
 run('wrong_selected_seed',lambda d,e,q:d['fixture-inspection']['records'][-1].update(seed=91011),True),
 run('wrong_selected_attack',lambda d,e,q:d['fixture-inspection']['records'][-1].update(attack='FedSA'),True),
 run('old412_object_tamper',lambda d,e,q:d['fixture-inspection']['records'][0].update(id='tampered'),True),
 run('old45_ledger_prefix_tamper',lambda d,e,q:d['fixture-ledger']['entries'][0].update(fixture_tamper=True),True),
 run('duplicate_native_new',lambda d,e,q:d['fixture-root']['new_ids'].__setitem__(0,d['fixture-root']['new_ids'][1]),True),
 run('duplicate_selected',lambda d,e,q:q['candidate_ids'].__setitem__(0,q['candidate_ids'][1]),True),
 run('replay_count_expansion',lambda d,e,q:e.update(proposed_replay_total=321),True)]
result=dict(status='METADATA_SCOPE_FIXTURES_PASS_NO_SCIENTIFIC_EXECUTION',cases=cases,fit=0,CNN=0,SSH=0)
with (H/'EXACT10_SELECTION_FIXTURES.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
