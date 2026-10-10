"""Source-only Hybrid100 metadata helpers. No binding, output jobs, inference or dispatch."""
from pathlib import Path
import ast,hashlib,json,math,sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SCREEN=ROOT/'tmp/celeba_hybrid_screen_execution_20261009'
ATTACKS=('Benign','F Flip','FedSA','S-DFA','Sp-DFA')
SEEDS=tuple(range(91001,91011));DISTRIBUTIONS={'IID':5000.0,'non-IID':5.0}
REUSED_KEYS={(d,a,91001) for d in DISTRIBUTIONS for a in ('Benign','S-DFA')}
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def frozen_python310_sum(items,start=0):
    # Preserve the adopted local Python3.10 summary exactly across Python3.12.
    for value in items:start=start+value
    return start

def original_rank():
    require(not sys.flags.optimize,'Optimized Python is refused')
    source=SCREEN/'summarize.py'
    require(sha(source)=='47a83a21b35983b6fab1d499784df15e01f81792a795b2ed3eb06361eec4403f','Original score/rank source changed')
    tree=ast.parse(source.read_text('utf8'))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('score','rank')]
    require(len(functions)==2,'Exact original score/rank required')
    env={'require':require,'sum':frozen_python310_sum};exec(compile(ast.Module(body=functions,type_ignores=[]),str(source)+':score_rank_only','exec'),env)
    return env['rank']

def check_complete_summary(summary,protocol,scope,*,summary_path=None,root_path=None,root_sha256=None):
    """Original frozen ranking; actual summary status requires SHA-bound root adoption."""
    adopted=None
    if summary['status']=='ALL32_ORIGINAL_STRICT_OFFSERVER_VERIFIED_ORIGINAL_FROZEN_RANK_ROOT_PENDING':
        require(summary_path is not None and root_path is not None and root_sha256 is not None,'Actual summary requires external root proof and SHA')
        require(sha(root_path)==root_sha256,'External root adoption SHA changed')
        adopted=json.loads(Path(root_path).read_text('utf8'))
        require(adopted['status']=='ROOT_HYBRID32_SUMMARY_ADOPTED' and adopted['accepted_total']==32 and adopted['all32_offserver_verified'] is True and adopted['final_test'] is False,'Actual complete32 valid-only root adoption required')
        require(json.loads(Path(summary_path).read_text('utf8'))==summary and sha(summary_path)==adopted['summary_sha256'],'Original summary bytes/SHA changed')
        require(adopted['source_seal_sha256']==summary['source_seal_sha256']==sha(SCREEN/'FILES_SHA256.json')=='2c496ae11369465d27ed223f8552321ec8cd424e5fd87e9f2d34e5ea8531e06f','Original screen source seal changed')
    else:
        require(summary['status']=='32_VALID_ONLY_TERMINALS_STRICT_ACCEPTED_SUMMARIZED_BACKUP_PENDING','Complete original summary schema required')
    require(summary['seed_n']==1 and summary['final_test_evaluated'] is False,'Single-seed valid screen only')
    records=[r for candidate in summary['all_candidates'] for r in candidate['records']]
    require({r['id'] for r in records}=={e['id'] for e in scope['jobs']} and len(records)==32,'Exact original32 IDs required')
    for r in records:
        require(r['seed']==91001 and all(math.isfinite(r['metrics'][k]) for k in ('accuracy','aeod','aspd')),'Finite same-record metrics and seed91001 required')
        require(all(len(r[k])==64 and set(r[k])<=set('0123456789abcdef') for k in ('checkpoint_sha256','acceptance_sha256')),'Original model/strict identity required')
    ranked=original_rank()(records,protocol['candidates'])
    require(all(summary[k]==value for k,value in ranked.items()),'Original ranking/accuracy/Pareto/all-negative records changed')
    if adopted is not None:require(adopted['selected_recipe']==ranked['selected_recipe'],'Root winner differs from original frozen ranking')
    return ranked['selected_recipe']

def coverage_layout(candidate_id,protocol,scope,screen_jobs):
    """Pure metadata proposal, not a frozen job manifest or recipe approval."""
    require(not sys.flags.optimize,'Optimized Python is refused')
    require(protocol['distributions']==DISTRIBUTIONS and protocol['attacks']==['Benign','S-DFA'],'Original screen axes changed')
    candidates={r['id']:r for r in protocol['candidates']};require(len(candidates)==8 and candidate_id in candidates,'One original recipe required')
    candidate=candidates[candidate_id]
    require(len(scope['jobs'])==len({e['id'] for e in scope['jobs']})==32,'Original32 scope required')
    selected=[e for e in scope['jobs'] if e['tuning_candidate']==candidate_id]
    require(len(selected)==4 and {(e['distribution'],e['attack'],91001) for e in selected}==REUSED_KEYS,'Exactly selected recipe original four references')
    refs=[]
    for entry in selected:
        job=screen_jobs[entry['id']];d,a=entry['distribution'],entry['attack']
        require(job['id']==entry['id'],'Original job ID changed')
        expected=dict(protocol['base_config'],rounds=70,seed=91001,client_alpha=DISTRIBUTIONS[d],learning_rate=candidate['learning_rate'],guardfed_fairness_lambda=candidate['adapter']['fairness_lambda'],trust_threshold=candidate['adapter']['threshold'],experiment_suite=protocol['version'],experiment_tag=entry['id'])
        require(job['config']==expected and job['method']=='CosineFairnessHybrid' and job['dataset']=='celeba','Original four configs/method changed')
        require(job['distribution']==d and job['attack']==a and job['tuning_candidate']==candidate_id and job['adapter']==candidate['adapter'],'No cross-method/recipe/condition reuse')
        require(job['source_hashes']==protocol['source_hashes'] and job['phase']=='screen' and job['evidence_stage']=='validation_screen','Original source/stage changed')
        refs.append(dict(id=entry['id'],job=entry['job'],job_sha256=entry['job_sha256'],output=entry['output'],distribution=d,attack=a,seed=91001,source='ORIGINAL_HYBRID_SCREEN_ONLY',strict_recheck_required=True))
    new=[]
    for d,alpha in DISTRIBUTIONS.items():
        for a in ATTACKS:
            for seed in SEEDS:
                if (d,a,seed) in REUSED_KEYS:continue
                new.append(dict(distribution=d,attack=a,seed=seed,alpha=alpha,rounds=70,evaluation_split='valid',candidate=candidate_id))
    keys={(r['distribution'],r['attack'],r['seed']) for r in new}
    require(len(new)==len(keys)==96 and not keys&REUSED_KEYS and len(keys|REUSED_KEYS)==100,'Exact96+4 coverage required')
    return dict(status='METADATA_PROPOSAL_NOT_FROZEN_NOT_EXECUTABLE',selected_recipe=candidate_id,new_grid=new,reused_references=refs,planned_new=96,planned_reused=4,planned_total=100,execution_authorized=False,actual_recipe_root_adopted=False,job_files_generated=False)
