"""Metadata gates only. No result acceptance, model import, resource or network actions."""
from pathlib import Path
import hashlib, json

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,why):
 if not ok:raise ValueError(why)

def validate(auth,contract,here):
 here=Path(here)
 need(auth['status']=='ROOT_BOUND_EXACT_NGA96_DELTA_NOT_ACCEPTED','Actual root-bound delta required')
 for field in ['manifest_sha256','source_approval_sha256','source_seal_sha256','checker_sha256']:
  need(auth[field]==contract['expected_'+field],'Actual frozen identity differs: '+field)
 need(auth['test'] is False and auth['automatic_retry'] is False,'No test or retry')
 expected=contract['new_ids'];ids=auth['authorized_ids'];old=auth['accepted_prior_ids']
 need(isinstance(ids,list) and ids and len(ids)==len(set(ids)),'Actual nonempty unique terminal IDs required')
 need(isinstance(old,list) and len(old)==len(set(old)) and set(old)<=set(expected),'Prior IDs differ')
 need(not set(ids)&set(old) and [i for i in expected if i in ids]==ids,'Repeated/foreign/unordered delta')
 need(set(ids)<=set(auth['observed_terminal_ids']) and not auth['observed_failure_paths'],'Actual failure-free terminal observation required')
 need(auth['snapshot_path'] is not None and isinstance(auth['snapshot_sha256'],str) and len(auth['snapshot_sha256'])==64,'Snapshot not bound')
 need(auth['source_approval_path'] is not None and auth['gate14_adoption_path'] is not None and isinstance(auth['gate14_adoption_sha256'],str) and len(auth['gate14_adoption_sha256'])==64,'Actual frozen-source/gate14 proof paths required')
 need(auth['accepted_prior_count']==len(old) and len(old)+len(ids)<=96,'Incorrect denominator')
 if old:
  need(auth['prior_root_path'] is not None and auth['prior_offserver_path'] is not None,'Prior actual roots required')
  need(sha(here/'PREVIOUS_ROOT_ADOPTION.json')==auth['prior_root_sha256'] and sha(here/'PREVIOUS_OFFSERVER_ACCEPTANCE.json')==auth['prior_offserver_sha256'],'Prior proof hash differs')
  prior=read(here/'PREVIOUS_ROOT_ADOPTION.json');off=read(here/'PREVIOUS_OFFSERVER_ACCEPTANCE.json')
  need(prior['root_adopted'] is True and prior['test'] is False and prior['scientific_package_sha256']==contract['expected_source_seal_sha256'],'Actual coverage root required')
  need(prior['accepted_ids']==off['accepted_job_ids']==old and prior['accepted_total']==off['accepted_total']==len(old),'Prior coverage chain differs')
 else:
  need(all(auth[k] is None for k in ['prior_root_path','prior_root_sha256','prior_offserver_path','prior_offserver_sha256']),'First batch must not invent a previous adoption')
  prior={'accepted_ids':[],'accepted_total':0};off={'accepted_job_ids':[],'accepted_total':0}
 return prior,off

def manifest_jobs(manifest,contract):
 need(manifest['total']==100 and len(manifest['new_jobs'])==96 and manifest['reused_jobs']==contract['reused_screen4'],'Exact96+explicit4 changed')
 need([j['id'] for j in manifest['new_jobs']]==contract['new_ids'],'Frozen new-ID order changed')
 return dict(manifest,jobs=[dict(r,job_sha256=r['sha256']) for r in manifest['new_jobs']])
