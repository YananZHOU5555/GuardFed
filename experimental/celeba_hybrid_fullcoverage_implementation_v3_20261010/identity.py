"""Hybrid stage/job identities only; no algorithm, tensor or metric computation."""
import copy
from pathlib import Path
METHOD='CosineFairnessHybrid';VERSION='celeba_hybrid_selected_fullcoverage_20261010_v1'
ATTACKS=('Benign','F Flip','FedSA','S-DFA','Sp-DFA');DISTRIBUTIONS={'IID':5000.0,'non-IID':5.0}
def require(ok,message):
    if not ok:raise ValueError(message)
def key(job):return job['distribution'],job['attack'],job['config']['seed']
def reused(d,a,s):return s==91001 and d in DISTRIBUTIONS and a in ('Benign','S-DFA')
def make_job(protocol,candidate,d,a,seed,phase,reference=False):
    require(d in DISTRIBUTIONS and a in ATTACKS and type(seed) is int and 91001<=seed<=91010,'Unfrozen grid')
    require(phase in {'fullcoverage','canary'},'Unknown stage')
    if phase=='fullcoverage':require(not reference and not reused(d,a,seed),'Old four cannot be retrained')
    else:require(d=='non-IID' and seed==91002 and (not reference or a in ('Benign','S-DFA')),'Exact seven new-interface canaries only')
    ident=f"{candidate['id']}_{d}_{a}_seed{seed}_{phase}"+('_legacy' if reference else '')
    config=dict(protocol['base_config'],seed=seed,rounds=70 if phase=='fullcoverage' else 3,client_alpha=DISTRIBUTIONS[d],learning_rate=candidate['learning_rate'],guardfed_fairness_lambda=candidate['adapter']['fairness_lambda'],trust_threshold=candidate['adapter']['threshold'],experiment_suite=VERSION,experiment_tag=ident)
    return dict(id=ident,dataset='celeba',method='GuardFed' if reference else METHOD,distribution=d,attack=a,phase=phase,evidence_stage='validation_fullcoverage' if phase=='fullcoverage' else 'real_image_cuda_pipeline_gate_only',config=config,adapter=copy.deepcopy(candidate['adapter']),tuning_candidate=candidate['id'],source_hashes=copy.deepcopy(protocol['source_hashes']))
def validate_job(job,protocol,candidate):
    expected=make_job(protocol,candidate,job['distribution'],job['attack'],job['config']['seed'],job['phase'],job['method']=='GuardFed')
    require(job==expected,'Whole job/config/seed/source identity differs')
    c=job['config'];require(c['device']=='cuda' and c['celeba_evaluation_split']=='valid' and c['celeba_train_limit']==c['celeba_eval_limit']==0,'Full train/valid CUDA only')
def validate_grid(jobs,gates,refs):
    desired={(d,a,s) for d in DISTRIBUTIONS for a in ATTACKS for s in range(91001,91011)}
    refkeys={(r['distribution'],r['attack'],r['seed']) for r in refs}
    require(len(refs)==len(refkeys)==4 and all(reused(*x) for x in refkeys),'Four exact Hybrid screen references')
    newkeys={key(j) for j in jobs};require(len(jobs)==len(newkeys)==96 and not newkeys&refkeys and newkeys|refkeys==desired,'96+4 exact100')
    expected=[('Benign',False),('Benign',True),('S-DFA',False),('S-DFA',True),('F Flip',False),('FedSA',False),('Sp-DFA',False)]
    require([(g['attack'],g['method']=='GuardFed') for g in gates]==expected and all(key(g)[0]=='non-IID' and key(g)[2]==91002 for g in gates),'Seven fixed same-horizon gate jobs')
