"""Small metadata-only fixtures; no transport, arrays, fit, or SSH."""
from pathlib import Path
import copy, importlib.util, json
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('after295_identity',H/'verify_native_inputs.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=m.read(H/'PREPARED.json');old=m.read(p['native295_inspection_path']);ledger=m.read(p['native295_ledger_path'])
realread=m.read;realsha=m.sha
def fixture(extra=False):
    ids=p['candidate_ids']+(['minus_F_IID_Benign_seed91001'] if extra else [])
    rows=[dict(id=i,variant='minus_A',distribution='non-IID',attack='Sp-DFA',seed=int(i[-5:]),prediction_support=dict(prediction_count=19867)) for i in p['candidate_ids']]
    if extra: rows.append(dict(id=ids[-1],variant='minus_F'))
    proof=dict(root_adopted=True,total_new_strict_and_offserver=295+len(ids),test=False,inspection_sha256='fixture-inspection',ledger_sha256='fixture-ledger',previous_ledger_sha256=p['native295_ledger_sha256'],new_ids=ids)
    d={'fixture-root':proof,'fixture-inspection':dict(records=copy.deepcopy(old['records'])+rows,accepted_new_ids=ids),'fixture-ledger':dict(entries=copy.deepcopy(ledger['entries'])+[{'fixture':True}])}
    e=dict(native_root_verified=True,exact_candidate_ids=p['candidate_ids'],native_root_path='fixture-root',native_root_sha256='fixture-root',native_inspection_path='fixture-inspection',native_inspection_sha256='fixture-inspection',native_ledger_path='fixture-ledger',native_ledger_sha256='fixture-ledger',native_archive_roots=[dict(path='fixture-root',sha256='fixture-root')],root_native_total=295+len(ids),transport_target_only=5,proposed_replay_total=300)
    q=copy.deepcopy(p);q['actual_native_root_path']='fixture-root';q['actual_native_root_sha256']='fixture-root'
    return d,e,q
def run(case,edit=None,reject=False,extra=False):
    d,e,q=fixture(extra)
    if edit:edit(d,e,q)
    m.read=lambda path: d[str(path)] if str(path) in d else realread(path)
    m.sha=lambda path: str(path) if str(path) in d else realsha(path)
    try:m.validate(e,q)
    except AssertionError:
        assert reject,case
    else:assert not reject,case
    return dict(case=case,expected_rejection=reject,passed=True)
cases=[run('exact5_native300'),run('native301_extraF_rejected_by_exact300',extra=True,reject=True),
 run('missing_A_in_native_new',lambda d,e,q:d['fixture-root']['new_ids'].pop(),True),
 run('F_in_replay_scope',lambda d,e,q:q['candidate_ids'].__setitem__(0,'minus_F_IID_Benign_seed91001'),True),
 run('old395_object_tamper',lambda d,e,q:d['fixture-inspection']['records'][0].update(id='tampered'),True),
 run('old42_ledger_prefix_tamper',lambda d,e,q:d['fixture-ledger']['entries'][0].update(fixture_tamper=True),True),
 run('duplicate_native_new',lambda d,e,q:d['fixture-root']['new_ids'].__setitem__(0,d['fixture-root']['new_ids'][1]),True),
 run('replay_count_expansion',lambda d,e,q:e.update(proposed_replay_total=301),True)]
result=dict(status='METADATA_SCOPE_FIXTURES_PASS_NO_SCIENTIFIC_EXECUTION',cases=cases,fit=0,CNN=0,SSH=0)
with (H/'EXACT300_SELECTION_FIXTURES.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
