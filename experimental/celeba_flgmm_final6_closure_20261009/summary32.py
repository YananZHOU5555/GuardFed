"""Read root-adopted strict records only; never import an acceptor or evaluate a model."""
from pathlib import Path
import argparse,hashlib,json,math,statistics
from closure_guard import EXPECTED_CHAIN,EXPECTED_PACKAGE,SOURCE_PINS
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())

def summarize_records(records,manifest,protocol):
 assert len(records)==len({r['id'] for r in records})==32
 expected={r['id']:r for r in manifest['jobs']};assert {r['id'] for r in records}==set(expected)
 assert sha(HERE/'frozen_score.py')=='1b31c0322b06bc1901f2a2f49a2b7b3524fed0164b6191d8bb9a5ed7e5f24b27'
 from frozen_score import score
 rows=[]
 for record in records:
  item=expected[record['id']]
  assert (record['candidate'],record['distribution'],record['attack'],record['seed'],record['rounds'])==(item['tuning_candidate'],item['distribution'],item['attack'],91001,70)
  assert all(math.isfinite(record['metrics'][key]) for key in ('accuracy','aeod','aspd'))
  row=dict(record,**record['metrics']);rows.append(dict(row,score=score(row)))
 candidates=[]
 for candidate in sorted({row['candidate'] for row in rows}):
  group=[row for row in rows if row['candidate']==candidate]
  assert len(group)==4 and {(r['distribution'],r['attack']) for r in group}=={(d,a) for d in protocol['distributions'] for a in protocol['attacks']}
  candidates.append(dict(candidate=candidate,n_seeds=1,**{key:statistics.mean(row[key] for row in group) for key in ['accuracy','aeod','aspd','score']}))
 assert len(candidates)==8 and {r['candidate'] for r in candidates}=={r['id'] for r in protocol['candidates']}
 selected=min(candidates,key=lambda row:(-row['score'],row['candidate']))
 champion=min(candidates,key=lambda row:(-row['accuracy'],row['candidate']))
 def dominates(a,b):
  return a['accuracy']>=b['accuracy'] and a['aeod']<=b['aeod'] and a['aspd']<=b['aspd'] and (a['accuracy']>b['accuracy'] or a['aeod']<b['aeod'] or a['aspd']<b['aspd'])
 pareto=[r for r in candidates if not any(dominates(other,r) for other in candidates)]
 return dict(status='ROOT_ACCEPTED32_RECORD_ONLY_RECIPE_SUMMARY',accepted=32,records=rows,candidates=candidates,
  selected_per_method={manifest['method']:selected},selected_candidate_recipe=next(c for c in protocol['candidates'] if c['id']==selected['candidate']),accuracy_champion=champion,three_metric_pareto=pareto,
  score_semantics='Original frozen score per condition, then mean over four conditions; exact ties use candidate lexical order',
  seed_n=1,sample_SD_reported=False,significance_claimed=False,all_negative_results_retained=True,
  final_test=False,formal100_started=False,recipe_adopted=False,
  limitation='Exposed valid-only seed91001 search. Four conditions are not independent seeds. Recipe summary is not paper performance or final protocol adoption.')

def same_records(server,offserver):
 assert offserver['status']=='PARTIAL_ACCEPTED_OFFSERVER_VERIFIED' and offserver['different_host_observed']
 original={r['id']:r for r in server['records']};verified={r['id']:r for r in offserver['records']}
 assert len(original)==len(server['records'])==len(verified)==len(offserver['records']) and set(original)==set(verified)
 for identity,row in original.items():
  assert all(row[key]==verified[identity][key] for key in ('rounds','seed','distribution','attack','metrics','checkpoint_sha256'))
 return list(original.values())

def validate_final_proof(proof):
 assert proof['status']=='ROOT_LOCKED_AUXILIARY_DELTA_ARCHIVE_MEMBER_AND_ORIGINAL_RECORD_ACCEPTANCE_PASS'
 assert (proof['accepted_before'],proof['accepted_new'],proof['accepted_total'])==(26,6,32)
 assert proof['previous_chain_sha256']==EXPECTED_CHAIN and proof['new_inference']==0 and not proof['final_test']

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--final-directory',type=Path,required=True);parser.add_argument('--root-proof-sha256',required=True);parser.add_argument('--prepared-seal-sha256',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 assert not args.output.exists()
 assert sha(HERE/'FILES_SHA256.json')==args.prepared_seal_sha256
 for name,row in read(HERE/'FILES_SHA256.json')['files'].items():assert sha(HERE/name)==row['sha256']
 root=HERE.parents[1];index=read(HERE/'PRIOR26_RECORD_SOURCES.json');manifest=read(HERE/'manifest.json');protocol=read(HERE/'protocol.json')
 assert sha(HERE/'manifest.json')==SOURCE_PINS['jobs/manifest.json'] and sha(HERE/'protocol.json')==SOURCE_PINS['source/protocol.json']
 assert sha(HERE/'PREVIOUS_CHAIN.json')==EXPECTED_CHAIN
 assert sha(root/index['parent_chain_path'])==EXPECTED_CHAIN
 records=[];bindings=[]
 for entry in index['sources']:
  for name,digest in entry['pins'].items():assert sha(root/name)==digest
  records+=same_records(read(root/entry['server_path']),read(root/entry['offserver_path']));bindings.append(entry)
 assert len(records)==26 and {r['id'] for r in records}==set(read(HERE/'PREVIOUS_CHAIN.json')['accepted_job_ids'])
 final=args.final_directory;proof=read(final/'ROOT_ADOPTION_REVIEW.json')
 assert sha(final/'ROOT_ADOPTION_REVIEW.json')==args.root_proof_sha256
 validate_final_proof(proof)
 assert sha(final/'PARTIAL_ACCEPTANCE.json')==proof['strict_receipt_sha256'] and sha(final/'OFFSERVER_ACCEPTANCE.json')==proof['offserver_proof_sha256']
 assert sha(final/'accepted_final6_delta.tar.gz')==proof['archive_sha256']
 receipt=read(final/'BACKUP_SHA256.json');server=read(final/'PARTIAL_ACCEPTANCE.json');offserver=read(final/'OFFSERVER_ACCEPTANCE.json')
 assert receipt['archive_sha256']==proof['archive_sha256'] and receipt['package_sha256']==EXPECTED_PACKAGE
 assert receipt['accepted_total']==server['accepted_total_including_previous']==32 and receipt['accepted_new']==6
 assert server['before_source_data']==server['after_source_data']==protocol['source_hashes']
 added=same_records(server,offserver);wanted=read(HERE/'EXACT_DELTA.json')['selected_ids']
 assert {r['id'] for r in added}==set(wanted) and len(added)==6
 records+=added
 expected={r['id']:r for r in manifest['jobs']}
 for row in records:
  assert row['job_sha256']==expected[row['id']]['job_sha256'] and row['source_hashes']==protocol['source_hashes']
 result=summarize_records(records,manifest,protocol)
 result['source_bindings']=dict(prior26_sources=bindings,final_root_proof_sha256=args.root_proof_sha256,final_archive_sha256=proof['archive_sha256'],final_strict_sha256=proof['strict_receipt_sha256'],final_offserver_sha256=proof['offserver_proof_sha256'])
 args.output.mkdir(parents=True)
 with (args.output/'SUMMARY32.json').open('x',encoding='utf8',newline='\n') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(status=result['status'],summary_sha256=sha(args.output/'SUMMARY32.json'),recipe_adopted=False)))
if __name__=='__main__':main()
