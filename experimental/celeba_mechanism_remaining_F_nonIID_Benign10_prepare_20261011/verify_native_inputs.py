"""Metadata-only actual native >=360 / old350; replay exact10 binding; imports never perform transport or science."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate(e,p):
    assert e['native_root_verified'] is True and e['exact_candidate_ids']==p['candidate_ids']
    for field in ['native_root','native_inspection','native_ledger']:
        assert sha(e[field+'_path'])==e[field+'_sha256']
    for field in ['prior_root','prior_index','native350_root','native350_inspection','native350_ledger']:
        assert sha(p[field+'_path'])==p[field+'_sha256']
    proof=read(e['native_root_path']);inspection=read(e['native_inspection_path']);ledger=read(e['native_ledger_path'])
    old=read(p['native350_inspection_path']);oldledger=read(p['native350_ledger_path'])
    assert e['root_native_total']>=p['native_target_minimum']==360 and p['native_binding'] is None
    assert p['required_native_root_path'] is None and p['required_native_root_sha256'] is None
    assert e['native_root_path']!=p['native350_root_path']
    assert proof['root_adopted'] is True and proof['total_new_strict_and_offserver']==e['root_native_total'] and proof['test'] is False
    assert proof['inspection_sha256']==e['native_inspection_sha256'] and proof['ledger_sha256']==e['native_ledger_sha256']
    assert proof['previous_ledger_sha256']==p['native350_ledger_sha256']
    assert len(proof['new_ids'])==len(set(proof['new_ids']))==e['root_native_total']-350
    assert set(p['required_new_native_candidate_ids'])<=set(proof['new_ids'])
    assert p['already_native_candidate_ids']==[] and p['known_native_archive_roots']==[]
    assert not set(p['candidate_ids'])&{r['id'] for r in old['records']}
    assert p['required_new_native_candidate_ids']==p['candidate_ids']
    assert not set(proof['new_ids'])&{r['id'] for r in old['records']}
    assert p['candidate_ids']==[f'minus_F_non-IID_Benign_seed{s}' for s in range(91001,91011)]
    assert e['transport_target_only']==10 and e['proposed_replay_total']==360
    assert len(old['records'])==450 and len(inspection['records'])==100+e['root_native_total']
    assert [row for row in inspection['records'] if row['id'] not in proof['new_ids']]==old['records']
    assert len(oldledger['entries'])==49 and ledger['entries'][:-1]==oldledger['entries'] and len(ledger['entries'])==50
    prior=read(p['prior_index_path']);priorproof=read(p['prior_root_path'])
    assert p['prior_accepted']==priorproof['cumulative_accepted']==len(prior['all_ids'])==350
    assert set(p['candidate_ids']).isdisjoint(prior['all_ids'])
    rows={row['id']:row for row in inspection['records']}
    assert len(rows)==len(inspection['records']) and set(p['candidate_ids'])<=set(inspection['accepted_new_ids'])
    for identity in p['candidate_ids']:
        row=rows[identity]
        assert (row['variant'],row['distribution'])==('minus_F','non-IID') and row['attack']=='Benign'
        assert identity==row['variant']+'_'+row['distribution']+'_'+row['attack']+'_seed'+str(row['seed'])
        assert row['prediction_support']['prediction_count']==19867
    roots=e['native_archive_roots']
    assert len(roots)==1 and roots[:-1]==p['known_native_archive_roots'] and roots[-1]=={'path':e['native_root_path'],'sha256':e['native_root_sha256']}
    covered=set()
    for pin in roots:
        assert sha(pin['path'])==pin['sha256']
        root=read(pin['path']);assert root['root_adopted'] is True and root['test'] is False
        selected=set(root['new_ids'])&set(p['candidate_ids'])
        assert selected and not selected&covered
        covered.update(selected)
    assert covered==set(p['candidate_ids'])
    assert p['authorized_snapshot_path'] is None and p['authorized_snapshot_sha256'] is None
    assert sha(e['authorized_snapshot_path'])==e['authorized_snapshot_sha256']
    snapshot=read(e['authorized_snapshot_path'])
    assert set(p['candidate_ids'])<=set(snapshot['remaining620']['remote_closed_ids'])
    assert not snapshot['remaining620']['failures'] and not snapshot['source_data_binding']['changed_paths']
    return e


def verify_native_inputs():
    e=read(H/'EXECUTION_INPUTS.json');p=read(H/'PREPARED.json')
    assert sha(H/'SOURCE_FILES_SHA256.json')==e['source_seal_sha256']
    for name,pin in read(H/'SOURCE_FILES_SHA256.json')['files'].items():
        assert sha(H/name)==pin['sha256'] and (H/name).stat().st_size==pin['bytes']
    return validate(e,p)
