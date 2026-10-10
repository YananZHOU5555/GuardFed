"""Metadata-only 64 -> two (96 new + 4 referenced) preparations. Never freezes or trains."""
from pathlib import Path
import argparse,ast,copy,difflib,hashlib,json,math,os,sys
if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
    raise RuntimeError('Identity assertions require unoptimized Python')
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'tmp/celeba_gradient_screen64_v2_20261010'
SEAL='11e2ae63c87e0440669a047c930f5465b13babf559cca17bd8808bf693ce7ced'
BRIDGE='snapshot/gradient_bridge_20261010/'
METHODS=('Fed-NGA-gradient','Huber-BRFL-gradient')
ATTACKS=('Benign','F Flip','FedSA','S-DFA','Sp-DFA')
SEEDS=tuple(range(91001,91011));DIST={'IID':5000.0,'non-IID':5.0}
H=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())

def inputs():
    assert H((OLD/'FILES_SHA256.json').read_bytes())==SEAL
    pins=read(OLD/'FILES_SHA256.json')['files']
    for n in [BRIDGE+x for x in ('worker.py','accept_result.py','protocol.json')]+['jobs/manifest.json','frozen_score.py']:
        assert H((OLD/n).read_bytes())==pins[n],n
    ns={};exec(compile((OLD/'frozen_score.py').read_bytes(),'<pinned-score>','exec'),ns)
    return read(OLD/BRIDGE/'protocol.json'),read(OLD/'jobs/manifest.json'),pins,ns['score']

def selected(records,protocol,manifest,score):
    """Requires all original strict rows, not a best-so-far or selected subset."""
    assert len(records)==len({r['id'] for r in records})==64
    original={r['id']:r for r in manifest['jobs']}
    assert {r['id'] for r in records}==set(original)
    groups={c['id']:[] for c in protocol['candidates']}
    for r in records:
        j=read(OLD/'jobs'/original[r['id']]['job'])
        assert H((OLD/'jobs'/original[r['id']]['job']).read_bytes())==original[r['id']]['job_sha256']
        assert r['job_sha256']==original[r['id']]['job_sha256']
        assert (r['method'],r['candidate'],r['distribution'],r['attack'],r['seed'],r['rounds'],r['alpha'],r['evaluation_split'])==(
            j['method'],j['tuning_candidate'],j['distribution'],j['attack'],91001,70,DIST[j['distribution']],'valid')
        assert r['strict_pass'] is True and r['offserver_verified'] is True
        assert r['source_hashes']==j['source_hashes'] and r['component_hashes']==j['component_hashes'] and r['local_hashes']==j['local_hashes']
        for k in ('result_sha256','model_sha256','acceptance_sha256'):
            assert isinstance(r[k],str) and len(r[k])==64 and set(r[k])<=set('0123456789abcdef')
        assert r['output'] and all(type(r['metrics'][k]) in (float,int) and math.isfinite(r['metrics'][k]) and 0<=r['metrics'][k]<=1 for k in ('accuracy','aeod','aspd'))
        groups[r['candidate']].append(r)
    rank={}
    for method in METHODS:
        candidates=[c for c in protocol['candidates'] if c['method']==method]
        values=[]
        for c in candidates:
            rows=groups[c['id']]
            assert len(rows)==4 and {(r['distribution'],r['attack']) for r in rows}=={(d,a) for d in DIST for a in ('Benign','S-DFA')}
            rows=sorted(rows,key=lambda r:(r['distribution'],r['attack']))
            values.append((sum(score(r['metrics']) for r in rows)/4,c['id']))
        rank[method]=sorted(values,key=lambda x:(-x[0],x[1]))
    return {m:rank[m][0][1] for m in METHODS},rank

def render():
    """Only identity/seed and checker attack coverage; scientific loops remain whole-source exact."""
    worker=(OLD/BRIDGE/'worker.py').read_text(encoding='utf8')
    accept=(OLD/BRIDGE/'accept_result.py').read_text(encoding='utf8')
    changes={
      'worker.py':{
        'LOCAL_SOURCES = {"worker.py", "accept_result.py", "prepare.py", "protocol.json"}':'LOCAL_SOURCES = {"worker.py", "accept_result.py", "protocol.json"}',
        'identity = candidate["id"] + "_" + distribution + "_" + attack + "_seed91001_screen"':'seed = job.get("config", {}).get("seed")\n    if type(seed) is not int or seed not in range(91001, 91011):\n        raise ValueError("Fullcoverage requires shared model seed91001..91010")\n    if seed == 91001 and attack in {"Benign", "S-DFA"}:\n        raise ValueError("Selected original four screen records must be referenced, not rerun")\n    identity = candidate["id"] + "_" + distribution + "_" + attack + "_seed" + str(seed) + "_coverage"',
        'dict(protocol["base_config"], seed=91001, rounds=70,':'dict(protocol["base_config"], seed=seed, rounds=70,',
        'identity, "celeba", "validation_screen", expected':'identity, "celeba", "validation_fullcoverage", expected',
        'result.update(status="screen_complete", evidence_stage="validation_screen", revision_job=job,':'result.update(status="coverage_complete", evidence_stage="validation_fullcoverage", revision_job=job,'},
      'accept_result.py':{
        'result["status"] != "screen_complete" or result["evidence_stage"] != "validation_screen"':'result["status"] != "coverage_complete" or result["evidence_stage"] != "validation_fullcoverage"',
        '91001, 70, job["config"]["client_alpha"], 20, 4':'job["config"]["seed"], 70, job["config"]["client_alpha"], 20, 4',
        'root_expected = job["attack"] == "S-DFA"':'root_expected = job["attack"] in {"FedSA", "S-DFA", "Sp-DFA"}',
        'if root_expected and any(row["attack_types"] != ["fflip", "foe"] or "gradient sign-conjugacy" not in row["foe_impl"] for row in audits[:4]):\n        raise ValueError("Incomplete S-DFA gradient upload audit")':
        'for cid, row in enumerate(audits):\n        expected = [] if cid >= 4 or job["attack"] == "Benign" else {\n            "F Flip": ["fflip"], "FedSA": ["foe"], "S-DFA": ["fflip", "foe"],\n            "Sp-DFA": ["fflip"] if cid < 2 else ["foe"]}[job["attack"]]\n        if row["attack_types"] != expected:\n            raise ValueError("Changed five-scene client assignment")\n        if "foe" in expected and (row["foe_mode"] != "fedsa" or "gradient sign-conjugacy" not in row["foe_impl"]):\n            raise ValueError("Changed gradient upload threat")'}}
    result={};diff=[]
    for name,text in [('worker.py',worker),('accept_result.py',accept)]:
        old=text
        for a,b in changes[name].items():
            assert text.count(a)==1,(name,a)
            text=text.replace(a,b)
        ast.parse(text);result[name]=text
        diff.extend(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='original/'+name,tofile='prepared/'+name))
    return result,''.join(diff)

def grid(candidate,protocol,local_hashes):
    rows=[]
    for d,alpha in DIST.items():
        for attack in ATTACKS:
            for seed in SEEDS:
                if seed==91001 and attack in ('Benign','S-DFA'):continue
                identity=f'{candidate["id"]}_{d}_{attack}_seed{seed}_coverage'
                cfg=dict(protocol['base_config'],seed=seed,rounds=70,client_alpha=alpha,
                    use_reweighting=False,experiment_suite=protocol['version'],experiment_tag=identity)
                rows.append(dict(id=identity,dataset='celeba',method=candidate['method'],distribution=d,attack=attack,
                    evidence_stage='validation_fullcoverage',config=cfg,adapter=candidate['adapter'],tuning_candidate=candidate['id'],
                    source_hashes=protocol['source_hashes'],component_hashes=protocol['component_hashes'],local_hashes=local_hashes))
    assert len(rows)==len({r['id'] for r in rows})==96
    return rows

def bind(summary_path,summary_sha,root_path,root_sha,out):
    protocol,manifest,pins,score=inputs()
    summary_path,root_path,out=map(Path,(summary_path,root_path,out))
    assert H(summary_path.read_bytes())==summary_sha and H(root_path.read_bytes())==root_sha
    root=read(root_path);summary=read(summary_path)
    assert root['status']=='ROOT_GRADIENT64_COMPLETE_STRICT_OFFSERVER_ADOPTED'
    assert root['accepted_count']==64 and root['all64_offserver_verified'] is True and root['test'] is False
    assert root['summary_sha256']==summary_sha and root['screen_source_seal_sha256']==SEAL
    winners,rank=selected(summary['records'],protocol,manifest,score)
    assert root['selected_candidates']==winners and set(winners)==set(METHODS)
    assert not out.exists(),'Never overwrite a partial binding'
    source,diff=render();out.mkdir(parents=True)
    def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf8')
    save(out/'BOUND_INPUTS.json',dict(status='SELECTED_SOURCE_PREPARED_NOT_APPROVED_FOR_EXECUTION',
        summary_path=str(summary_path.resolve()),summary_sha256=summary_sha,root_path=str(root_path.resolve()),root_sha256=root_sha,
        winners=winners,rank=rank,execution_authorized=False,final_test=False))
    for method,winner in winners.items():
        candidate=next(c for c in protocol['candidates'] if c['id']==winner)
        stage=out/method/'snapshot/gradient_bridge_fullcoverage';stage.mkdir(parents=True)
        p=copy.deepcopy(protocol);p.update(status='PREPARED_NOT_FROZEN',version='celeba_gradient_fullcoverage_20261010',
            attacks=list(ATTACKS),candidates=[candidate],component_hashes=read(OLD/'jobs'/manifest['jobs'][0]['job'])['component_hashes'])
        for name,text in source.items():(stage/name).write_text(text,encoding='utf8',newline='\n')
        save(stage/'protocol.json',p)
        local={n:H((stage/n).read_bytes()) for n in ('worker.py','accept_result.py','protocol.json')}
        jobs=out/method/'jobs';jobs.mkdir()
        entries=[]
        for j in grid(candidate,p,local):
            save(jobs/(j['id']+'.json'),j);entries.append(dict(id=j['id'],job=j['id']+'.json',sha256=H((jobs/(j['id']+'.json')).read_bytes())))
        reused=[r for r in summary['records'] if r['candidate']==winner]
        assert len(reused)==4
        save(jobs/'manifest.json',dict(status='PREPARED_NOT_FROZEN',new_jobs=entries,reused_jobs=reused,total=100,
            original_component_sources={n:str((OLD/'snapshot'/n).resolve()) for n in p['component_hashes']},
            summary_sha256=summary_sha,root_adoption_sha256=root_sha,executed=False))
    (out/'SOURCE_DIFF.patch').write_text(diff,encoding='utf8')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--summary',required=True);a.add_argument('--summary-sha256',required=True)
    a.add_argument('--root-adoption',required=True);a.add_argument('--root-sha256',required=True);a.add_argument('--out',required=True)
    x=a.parse_args();bind(x.summary,x.summary_sha256,x.root_adoption,x.root_sha256,x.out)
