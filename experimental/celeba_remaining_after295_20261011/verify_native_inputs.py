"""Metadata-only actual native300 / old295; replay exact5 binding; imports never perform transport or science."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate(e,p):
    assert e['native_root_verified'] is True and e['exact_candidate_ids']==p['candidate_ids']
    for field in ['native_root','native_inspection','native_ledger']:
        assert sha(e[field+'_path'])==e[field+'_sha256']
    for field in ['prior_root','prior_index','native295_root','native295_inspection','native295_ledger']:
        assert sha(p[field+'_path'])==p[field+'_sha256']
    proof=read(e['native_root_path']);inspection=read(e['native_inspection_path']);ledger=read(e['native_ledger_path'])
    old=read(p['native295_inspection_path']);oldledger=read(p['native295_ledger_path'])
    assert proof['root_adopted'] is True and proof['total_new_strict_and_offserver']==e['root_native_total']==300 and proof['test'] is False
    assert proof['inspection_sha256']==e['native_inspection_sha256'] and proof['ledger_sha256']==e['native_ledger_sha256']
    assert proof['previous_ledger_sha256']==p['native295_ledger_sha256']
    assert len(proof['new_ids'])==len(set(proof['new_ids']))==e['root_native_total']-295
    assert p['candidate_ids']==proof['new_ids']
    assert e['native_root_path']==p['actual_native_root_path'] and e['native_root_sha256']==p['actual_native_root_sha256']
    assert p['candidate_ids']==[f'minus_A_non-IID_Sp-DFA_seed{s}' for s in range(91006,91011)]
    assert e['transport_target_only']==5 and e['proposed_replay_total']==300
    assert len(old['records'])==395 and len(inspection['records'])==100+e['root_native_total']
    assert [row for row in inspection['records'] if row['id'] not in proof['new_ids']]==old['records']
    assert len(oldledger['entries'])==42 and ledger['entries'][:-1]==oldledger['entries'] and len(ledger['entries'])==43
    prior=read(p['prior_index_path']);priorproof=read(p['prior_root_path'])
    assert p['prior_accepted']==priorproof['cumulative_accepted']==len(prior['all_ids'])==295
    assert set(p['candidate_ids']).isdisjoint(prior['all_ids'])
    rows={row['id']:row for row in inspection['records']}
    assert len(rows)==len(inspection['records']) and set(p['candidate_ids'])<=set(inspection['accepted_new_ids'])
    for identity in p['candidate_ids']:
        row=rows[identity]
        assert (row['variant'],row['distribution'])==('minus_A','non-IID') and row['attack']=='Sp-DFA'
        assert identity==row['variant']+'_'+row['distribution']+'_'+row['attack']+'_seed'+str(row['seed'])
        assert row['prediction_support']['prediction_count']==19867
    roots=e['native_archive_roots']
    assert len(roots)==1 and roots[-1]=={'path':e['native_root_path'],'sha256':e['native_root_sha256']}
    covered=set()
    for pin in roots:
        assert sha(pin['path'])==pin['sha256']
        root=read(pin['path']);assert root['root_adopted'] is True and root['test'] is False
        selected=set(root['new_ids'])&set(p['candidate_ids'])
        assert selected and not selected&covered
        covered.update(selected)
    assert covered==set(p['candidate_ids'])
    return e


def verify_native_inputs():
    e=read(H/'EXECUTION_INPUTS.json');p=read(H/'PREPARED.json')
    assert sha(H/'SOURCE_FILES_SHA256.json')==e['source_seal_sha256']
    for name,pin in read(H/'SOURCE_FILES_SHA256.json')['files'].items():
        assert sha(H/name)==pin['sha256'] and (H/name).stat().st_size==pin['bytes']
    return validate(e,p)
