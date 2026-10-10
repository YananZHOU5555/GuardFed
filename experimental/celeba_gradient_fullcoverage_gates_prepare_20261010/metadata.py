"""Exact14 short-horizon gate identities; requires actual64 and frozen192 before binding."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,os,sys
sys.dont_write_bytecode=True
if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):raise RuntimeError('Unoptimized Python required')
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
COVER=ROOT/'tmp/celeba_gradient_fullcoverage_prepare_20261010'
OLD=ROOT/'tmp/celeba_gradient_realimage_gate_20261009'
STAGE='exploratory_gradient_coverage_14_cpu_gate3_only'
PINS={'gate.py':'4c57c10fe00d5c21424f1265d067ec7c8bd5a6183c134cba31509351f69b7833',
 'shared_cache_wrapper.py':'3c61385450d99824a3983627e61b98c672c3016788b8016d7830435a0b0b8946',
 'shared_cache_bindings.json':'49d04722814f49f48f9ca995e226bf3156a8d7267a1852e086d4a5790c2d0b03'}
H=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())

def coverage_source():
    assert H((COVER/'FILES_SHA256.json').read_bytes())=='ea5a565d35c4f8df7d108679d1010baec92993da8f4b181a14ccde359334ed39'
    for n,p in read(COVER/'FILES_SHA256.json')['files'].items():
        raw=(COVER/n).read_bytes();assert H(raw)==p['sha256'] and len(raw)==p['bytes']
    for n,h in PINS.items():assert H((OLD/n).read_bytes())==h
    spec=importlib.util.spec_from_file_location('coverage_metadata',COVER/'prepare.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def jobs_for(candidate,protocol):
    """Metadata fixtures may call this; binding uses only root-selected candidates."""
    jobs=[]
    cases=[(a,implementation) for a in ('F Flip','FedSA','Sp-DFA') for implementation in ('screen','coverage')]+[('Benign','screen')]
    for attack,implementation in cases:
        identity=f'{candidate["id"]}_non-IID_{attack}_seed91002_{implementation}_cpu_gate3'
        config=dict(protocol['base_config'],seed=91002,rounds=3,device='cpu',client_alpha=5.0,use_reweighting=False,
            experiment_suite='celeba_gradient_fullcoverage_gates_20261010',experiment_tag=identity)
        jobs.append(dict(id=identity,dataset='celeba',method=candidate['method'],distribution='non-IID',attack=attack,
            pilot_candidate=candidate['id'],implementation=implementation,adapter=candidate['adapter'],config=config,
            evidence_stage=STAGE,scientific_table_records=0,source_hashes=protocol['source_hashes']))
    assert len(jobs)==len({j['id'] for j in jobs})==7
    return jobs

def validate_jobs(jobs,candidates,protocol):
    expected=[j for c in candidates for j in jobs_for(c,protocol)]
    assert len(jobs)==14 and len({j['id'] for j in jobs})==14
    assert jobs==expected,'Only exact14 ordered same-input references are allowed'

def bind(bound,approval,approval_sha,out):
    m=coverage_source();original,manifest,_,score=m.inputs();bound,approval,out=map(Path,(bound,approval,out))
    assert H(approval.read_bytes())==approval_sha
    approved=read(approval);binding=read(bound/'BOUND_INPUTS.json')
    assert approved['status']=='ROOT_GRADIENT200_FROZEN_SOURCE_ADOPTED' and approved['test'] is False
    assert approved['bound_inputs_sha256']==H((bound/'BOUND_INPUTS.json').read_bytes())
    assert H(Path(binding['root_path']).read_bytes())==binding['root_sha256']
    assert H(Path(binding['summary_path']).read_bytes())==binding['summary_sha256']
    root64=read(binding['root_path']);assert root64['status']=='ROOT_GRADIENT64_COMPLETE_STRICT_OFFSERVER_ADOPTED'
    assert root64['accepted_count']==64 and root64['all64_offserver_verified'] is True and root64['test'] is False
    assert root64['screen_source_seal_sha256']==m.SEAL and root64['summary_sha256']==binding['summary_sha256']
    summary=read(binding['summary_path']);selected,_=m.selected(summary['records'],original,manifest,score)
    assert selected==root64['selected_candidates']==binding['winners']
    rendered,_=m.render();candidates=[];worker_paths={};identities={}
    for method in m.METHODS:
        stage=bound/method/'snapshot/gradient_bridge_fullcoverage';p=read(stage/'protocol.json');mf=bound/method/'jobs/manifest.json'
        assert H(mf.read_bytes())==approved['manifest_sha256'][method]
        candidate=next(c for c in original['candidates'] if c['id']==selected[method]);candidates.append(candidate)
        expected=copy.deepcopy(original);expected.update(status='FROZEN',version='celeba_gradient_fullcoverage_20261010',
            attacks=list(m.ATTACKS),candidates=[candidate],component_hashes=read(m.OLD/'jobs'/manifest['jobs'][0]['job'])['component_hashes'])
        assert p==expected
        for n,text in rendered.items():assert (stage/n).read_text(encoding='utf8')==text
        local={n:H((stage/n).read_bytes()) for n in ('worker.py','accept_result.py','protocol.json')}
        wanted={j['id']:j for j in m.grid(candidate,p,local)};entries=read(mf)
        assert len(entries['new_jobs'])==96 and {j['id'] for j in entries['new_jobs']}==set(wanted)
        for row in entries['new_jobs']:
            path=mf.parent/row['job'];assert H(path.read_bytes())==row['sha256'] and read(path)==wanted[row['id']]
        assert entries['reused_jobs']==[r for r in summary['records'] if r['candidate']==selected[method]] and len(entries['reused_jobs'])==4
        worker_paths[method]={'screen':str((m.OLD/m.BRIDGE/'worker.py').resolve()),'coverage':str((stage/'worker.py').resolve())}
        identities[str(mf.resolve())]=H(mf.read_bytes())
        for n in local:identities[str((stage/n).resolve())]=local[n]
        for name,sha in expected['component_hashes'].items():
            path=stage.parent/name;assert H(path.read_bytes())==sha;identities[str(path.resolve())]=sha
    for n,h in PINS.items():identities[str((OLD/n).resolve())]=h
    for n in ('worker.py','protocol.json','accept_result.py'):
        path=m.OLD/m.BRIDGE/n;identities[str(path.resolve())]=H(path.read_bytes())
    for name,h in expected['component_hashes'].items():
        path=m.OLD/'snapshot'/name;assert H(path.read_bytes())==h;identities[str(path.resolve())]=h
    for path in (approval,bound/'BOUND_INPUTS.json',Path(binding['root_path']),Path(binding['summary_path'])):
        identities[str(path.resolve())]=H(path.read_bytes())
    for p in HERE.glob('*.py'):identities[str(p.resolve())]=H(p.read_bytes())
    jobs=[j for c in candidates for j in jobs_for(c,original)];validate_jobs(jobs,candidates,original)
    assert not out.exists(),'Preserve partial/existing binding; no automatic retry'
    out.mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    entries=[]
    for j in jobs:
        path=out/'jobs'/(j['id']+'.json');path.parent.mkdir(exist_ok=True);write(path,j)
        entries.append(dict(id=j['id'],job=path.relative_to(out).as_posix(),job_sha256=H(path.read_bytes()),output='runs/'+j['id']))
    (out/'shared_cache_bindings.json').write_bytes((OLD/'shared_cache_bindings.json').read_bytes())
    local={e['job']:e['job_sha256'] for e in entries};local['shared_cache_bindings.json']=PINS['shared_cache_bindings.json']
    write(out/'scope.json',dict(status='PREPARED_EXACT14_NOT_DISPATCHED',evidence_stage=STAGE,scientific_table_records=0,
        test=False,jobs=entries,worker_paths=worker_paths,external_identities=identities,local_hashes=local,
        protected_source_hashes=original['source_hashes'],candidates=candidates,original_protocol=original,
        root64_sha256=binding['root_sha256'],frozen192_approval_sha256=approval_sha,rounds=3,seed=91002,
        methods=list(m.METHODS),max_workers=1,cpu_threads=8,device='cpu',execution_authorized=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--bound',required=True);a.add_argument('--stage-approval',required=True)
    a.add_argument('--approval-sha256',required=True);a.add_argument('--out',required=True);x=a.parse_args()
    bind(x.bound,x.stage_approval,x.approval_sha256,x.out)
