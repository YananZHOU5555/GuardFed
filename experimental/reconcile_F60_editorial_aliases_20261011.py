"""Resolve only stale generic editorial aliases; scientific evidence is unchanged."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json
R=Path(__file__).resolve().parents[1];T=R/'docs/server_deployment_20260923/training_20260923'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=T/'TRAINING_STATE.json'
assert sha(p)=='7ec6894aedf96dd9837db605c78e088f1f8c18020fdc8daad0c1e26c22b4644d'
old=json.loads(p.read_bytes());s=copy.deepcopy(old)
assert s['latest_rebuttal_draft']['F_paired_models']==20
assert s['latest_working_rebuttal_draft']['content_cutoff']['F_pairs']==60
s['historical_integrated_F20_editorial_entry']=s['latest_rebuttal_draft']
s['latest_rebuttal_draft']=copy.deepcopy(s['latest_working_rebuttal_draft'])
s['latest_rebuttal_draft']['entry_role']='Current adopted full24 working draft; matching submitted manuscript application remains pending.'
for k in ('latest_rebuttal_addendum','latest_rebuttal_evidence_addendum'):
    s[k]['historical_artifact_not_current_full_draft']=True
    s[k]['current_full_draft_entry']=s['latest_working_rebuttal_draft']['entry']
for k in ('celeba_mechanism_v1','mechanism_remaining620_valid_20261010','flgmm_fullcoverage_v2_20261009','hybrid100_fullcoverage_20261010','gradient200_fullcoverage_source_preparation_20261010'):
    assert s[k]==old[k]
now=datetime.now(timezone.utc).isoformat()
s['current_editorial_alias_reconciliation']=dict(path=Path(__file__).relative_to(R).as_posix(),sha256=sha(__file__),utc=now)
s['updated_unix']=datetime.now(timezone.utc).timestamp()
with p.open('w',encoding='utf8',newline='\n') as f:json.dump(s,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
proof=R/'tmp/root_F60_current_entries_20261011/EDITORIAL_ALIAS_RECONCILIATION.json'
with proof.open('x',encoding='utf8',newline='\n') as f:json.dump(dict(status='ROOT_F60_GENERIC_EDITORIAL_ALIASES_RECONCILED',utc=now,writer=s['current_editorial_alias_reconciliation'],prior_STATE_sha256='7ec6894aedf96dd9837db605c78e088f1f8c18020fdc8daad0c1e26c22b4644d',STATE_sha256=sha(p),old_F20_entry_preserved=True,current_full_draft=s['latest_rebuttal_draft']['entry'],current_full_draft_sha256=s['latest_rebuttal_draft']['sha256'],addenda_explicitly_historical=True,all_scientific_stage_objects_unchanged=True,five_entry_bytes_unchanged=True,new_science=0,final_test=False,whole_rebuttal_complete=False),f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(STATE_sha256=sha(p),proof_sha256=sha(proof))))
