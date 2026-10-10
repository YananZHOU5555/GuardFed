"""Prepare a new actual-input spec only; never invoke Git, freeze, stage, or fit."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
S=ROOT/'tmp/publication_increment44_20261010'
H=lambda b:hashlib.sha256(b).hexdigest()
assert H((S/'FILES_SHA256.json').read_bytes())=='1abced4226020698b083d56dc5ddff4f772a24a8c7136c3d1c58988f088eb939'
for name,entry in json.loads((S/'FILES_SHA256.json').read_bytes())['files'].items():
    b=(S/name).read_bytes();assert H(b)==entry['sha256'] and len(b)==entry['bytes']
sp=importlib.util.spec_from_file_location('publisher44',S/'publish_increment44.py');p=importlib.util.module_from_spec(sp);sp.loader.exec_module(p)
T='docs/server_deployment_20260923/training_20260923'
B='tmp/celeba_hybrid_screen_execution_20261009'
D=B+'/accepted_delta_after23_20261010'

def prepare(main_live,startup=None,startup_sha=None,extra=()):
    spec=json.loads((S/'DRAFT_SPEC.json').read_bytes());assert H((S/'DRAFT_SPEC.json').read_bytes())=='28ee0e93e42c13657d5949e67a44bc17c627c0936c910a617aceae746d6f6088'
    files={e['source']:e for e in spec['files']};fixed=0
    for name,e in files.items():
        _,b=p.source(name,[name])
        if e['sha256'] is not None:
            assert H(b)==e['sha256'] and len(b)==e['bytes'];fixed+=1
        else:e.update(sha256=H(b),bytes=len(b))
    def add(name,expected=None):
        _,b=p.source(name,[name])
        if expected:assert H(b)==expected,name
        files[name]=dict(source=name,sha256=H(b),bytes=len(b));return json.loads(b) if name.endswith('.json') else None
    def bind(name,expected,expect):
        d=add(name,expected)
        for k,v in expect.items():assert p.pointer(d,k)==v,(name,k)
        return dict(path=name,sha256=expected,expect=expect)
    # Check every original delivery member before excluding bulk/console/duplicate bodies.
    sealname=D+'/DELIVERY_FILES_SHA256.json';seal=add(sealname,'59705f5d592e5bb03009b42288ac86aee3cf94fd18838e368e21c655cfe448ae')
    omitted=[]
    for e in seal['members']:
        name=D+'/'+e['path'];b=(ROOT/name).read_bytes();assert H(b)==e['sha256'] and len(b)==e['size']
        why=None
        if Path(name).suffix in {'.tar','.txt'} or e['path']=='SERVER_GUIDE.md':why='Bulk/console/guide remains at original source; not published'
        elif e['path'].startswith('local_record_bridge_v2_verified/'):why='Byte-identical verified copy retained externally; primary bridge provenance included'
        elif e['path'].endswith('/checked_record_body.py'):why='Unchanged scientific body recovered from explicit parent-commit path'
        if why:omitted.append(dict(source=name,sha256=e['sha256'],bytes=e['size'],reason=why))
        else:add(name,e['sha256'])
    for name in ['RAW_STORAGE_LOCATION.json','METADATA_SEAL_CORRECTION.json','LOCAL_RECORD_FAILURE.json','ROOT_READY_CHAIN_LINK.json']:
        add(D+'/'+name)
    spec['optional_bindings']['hybrid27']=bind(D+'/ROOT_ADOPTION_REVIEW.json','26bab727c431679b2913ac76cefe7dc28729c49d660bb3c8141489426bd28e6d',{'/accepted_before':23,'/accepted_new':4,'/accepted_total':27,'/formal100_started':False})
    chain=B+'/BACKUP_CHAIN_accepted_delta_after23_20261010.json'
    add(chain,'b430c1a4ea3fef5786d7719c69af2771d59aebdd93031dda1014bd9b6604b234')
    latest=add(B+'/LATEST_BACKUP.json');assert latest['accepted']==27 and latest['chain_sha256']=='b430c1a4ea3fef5786d7719c69af2771d59aebdd93031dda1014bd9b6604b234'
    add('tmp/adopt_hybrid27_root_20261010.py')
    original=B+'/accepted_delta_after18_20261010/local_record_bridge_v2/checked_record_body.py'
    b=(ROOT/original).read_bytes();assert H(b)=='99d227afd02c3e48b3f24cb1ba3db0cc089c6abb1552643abba6d86ae21b4dc0'
    spec['parent_recovery_references'].append(dict(source=original,destination=p.destination(original),sha256=H(b),bytes=len(b),parent_commit=p.PARENT))
    state_name=T+'/TRAINING_STATE.json';state=json.loads((ROOT/state_name).read_bytes())
    facts={**p.REQUIRED_FACTS['current_state'],'/hybrid_screen32_20261009/offserver_accepted70round_jobs':27,
           '/flgmm_fullcoverage_v2_20261009/new_accepted':22,'/logofair32_validation_search_20261010/root_adopted':32}
    # ROOT current schema is inspected; avoid inventing observed==accepted semantics.
    for k,v in facts.items():assert p.pointer(state,k)==v,(k,v)
    spec['bindings']['current_state']=bind(state_name,files[state_name]['sha256'],facts)
    live=add(main_live);assert live==state['last_health_check']
    add(T+'/server_reactivation_20261009/latest_formal_live.json',files.get(T+'/server_reactivation_20261009/latest_formal_live.json',{}).get('sha256'))
    observed=[]
    for key in ['gradient64_validation_search_20261010','mechanism_remaining620_valid_20261010']:
        entry=state[key]['latest_measured_observation'];add(entry['path'],entry['sha256']);observed.append(dict(role=key,**entry))
    spec['optional_bindings']['logofair100_startup']=None
    if startup is not None:
        assert startup_sha
        d=add(startup,startup_sha);status=d['status']
        assert status.startswith('ROOT_') and all(w not in status for w in ['PREPARED','PENDING','FAIL'])
        spec['optional_bindings']['logofair100_startup']=dict(path=startup,sha256=startup_sha,expect={'/status':status})
    else:assert startup_sha is None
    for name in extra:add(name)
    add(HERE.relative_to(ROOT).as_posix()+'/'+Path(__file__).name)
    spec.update(status='SOURCE_PREPARED_ACTUAL_INPUT_SNAPSHOT_NOT_PUBLISHED',files=sorted(files.values(),key=lambda e:e['source']),allowed_paths=sorted(files))
    spec['hybrid27_omitted_delivery_members']=omitted
    spec['input_preparation']=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),fixed_source_files_rechecked=fixed,
        hybrid_sealed_members_verified=len(seal['members']),main_live=main_live,measured_utc=live['checked_utc'],observed_main_terminal=live['queue_completed'],
        scientific_native_accepted=200,three_view_accepted=200,Hybrid_accepted=27,FL_new_accepted=22,LoGo_screen_accepted=32,
        logofair100_startup_bound=startup is not None,Git_calls=0,SSH=0,fit=0,freeze=False,stage=False,
        note='Mutable inputs are actual at this snapshot. Root must refresh before final publication if these change.')
    assert sum(e['bytes'] for e in spec['files'])<p.ns['LIMIT']
    return spec

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--main-live',required=True);a.add_argument('--startup');a.add_argument('--startup-sha256')
    a.add_argument('--extra',action='append',default=[]);a.add_argument('--output',type=Path,required=True);x=a.parse_args()
    assert x.output.resolve().is_relative_to(HERE.resolve()) and not x.output.exists()
    try:
        d=prepare(x.main_live,x.startup,x.startup_sha256,x.extra)
        p.ns['save'](x.output,d)
        print(json.dumps(dict(path=str(x.output),sha256=H(x.output.read_bytes()),files=len(d['files']),bytes=sum(e['bytes'] for e in d['files']),startup=d['optional_bindings']['logofair100_startup'])))
    except Exception as e:
        failure=x.output.with_name(x.output.stem+'_FAILURE.json')
        p.ns['save'](failure,dict(error=repr(e),source_unchanged=True,noGit=True,noSSH=True,no_fit=True))
        raise
