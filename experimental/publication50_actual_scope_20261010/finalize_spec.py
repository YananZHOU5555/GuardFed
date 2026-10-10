"""Root binds only the exact eight mutable files; extras are already closed."""
from pathlib import Path
import argparse,json
import publish_increment50 as p
HERE=Path(__file__).resolve().parent


def validate_actual(actual,prepared):
    assert set(actual)=={'status','parent','accepted','mutable_files'}, 'No arbitrary extra names or fields'
    assert actual['status']=='ROOT_CLOSED50_MUTABLE_BINDINGS_READY'
    assert actual['parent']==p.PARENT and actual['accepted']==p.ACCEPTED
    assert set(actual['mutable_files'])==set(prepared['pending_mutable']), 'Exact eight mutable files'
    for pin in actual['mutable_files'].values():
        assert set(pin)=={'sha256','bytes'}
        assert isinstance(pin['sha256'],str) and len(pin['sha256'])==64 and all(c in '0123456789abcdef' for c in pin['sha256'])
        assert type(pin['bytes']) is int and 0<pin['bytes']<p.ns['LIMIT']


def finalize(actual_path,actual_sha,output):
    assert output.parent.resolve()==HERE and not output.exists(), 'Fresh owned final spec only'
    raw=actual_path.read_bytes();assert p.sha(raw)==actual_sha
    actual=json.loads(raw);d=p.read(HERE/'PREPARED_MANIFEST.json')
    validate_actual(actual,d)
    entries={e['source']:e for e in d['files']}
    refs={e['source']:e for e in d['parent_recovery_references']}
    allowed=set(d['allowed_paths'])

    def add(name,pin):
        assert name in allowed, 'Exact closed source paths only'
        _,b=p.source(name,sorted(allowed))
        assert p.sha(b)==pin['sha256'] and len(b)==pin['bytes'], 'Actual source changed: '+name
        entries[name]=dict(source=name,destination=p.destination(name),sha256=pin['sha256'],bytes=pin['bytes'])
        refs.pop(name,None)

    for name,pin in actual['mutable_files'].items():add(name,pin)
    state=d['pending_mutable'][0]
    d['bindings']['current_state']=dict(path=state,sha256=actual['mutable_files'][state]['sha256'],expect=p.FACTS['current_state'])
    # This packet's sealed members are the sole additional permitted names.
    seal=HERE/'FILES_SHA256.json';packet=p.read(seal)
    own=HERE.relative_to(p.ROOT).as_posix()+'/'
    allowed.update(own+rel for rel in packet['files']);allowed.add(own+seal.name)
    for rel,pin in packet['files'].items():add(own+rel,pin)
    add(own+seal.name,{'sha256':p.sha(seal.read_bytes()),'bytes':seal.stat().st_size})
    role_paths={pin['path'] for pin in d['bindings'].values()}
    assert role_paths<=set(entries), 'All16 role proofs must be carried as files'
    p.storage()
    parent_paths=set(p.git('ls-tree','-r','--name-only','-z',p.PARENT).decode().split('\0'))-{''}
    for name,e in list(entries.items()):
        if name not in role_paths and e['destination'] in parent_paths:
            old=p.git('show',p.PARENT+':'+e['destination'])
            if p.sha(old)==e['sha256'] and len(old)==e['bytes']:
                refs[name]=dict(e,parent_commit=p.PARENT);del entries[name]
    d.update(status='ROOT_FINAL_CLOSED50_INPUTS_READY',pending_mutable=[],files=sorted(entries.values(),key=lambda e:e['source']),
             parent_recovery_references=sorted(refs.values(),key=lambda e:e['source']),allowed_paths=sorted(allowed),
             root_actual_input_sha256=actual_sha)
    result=p.plan(d)
    p.ns['save'](output,d)
    return {'status':'ROOT_ACTUAL_SPEC_READY_NOT_FROZEN_OR_STAGED','path':str(output),'sha256':p.sha(output.read_bytes()),
            'files':len(result['files']),'bytes':result['total_bytes'],'parent_references':len(refs)}


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root-inputs',type=Path,required=True)
    a.add_argument('--root-inputs-sha256',required=True);a.add_argument('--output',type=Path,required=True);v=a.parse_args()
    print(json.dumps(finalize(v.root_inputs,v.root_inputs_sha256,v.output),ensure_ascii=False))
