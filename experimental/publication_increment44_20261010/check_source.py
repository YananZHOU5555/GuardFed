"""Small source/metadata checks only: no Git calls or scientific execution."""
import ast,copy,hashlib,json
import publish_increment44 as p

def dump(x):return json.dumps(x,sort_keys=True).encode()

if __name__=='__main__':
    prior=p.raw.decode('utf8');tree=ast.parse(prior)
    bodies={n.name:ast.get_source_segment(prior,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
    stage=p.stage_text.replace('publication44_frozen_bytes_v1','publication43_frozen_bytes_v1').replace('# Git44 sealed evidence/source bytes','# Git43 sealed source/startup bytes')
    stage=stage.replace("assert d['accepted'] == ACCEPTED and set(d['bindings']) == ROLES",p.old)
    assert stage==bodies['stage']
    assert p.freeze is p.ns['freeze'] and p.freeze.__code__.co_filename==str(p.PRIOR)
    verifier=p.ROOT/'tmp/publication_increment43_20261010/verify_increment43.py'
    import verify_increment44 as v
    assert v.verify.__code__.co_filename==str(verifier)
    # Explicit synthetic metadata fixture, not a scientific result or current-state snapshot.
    data={};bindings={};entries=[]
    for role,facts in p.REQUIRED_FACTS.items():
        obj={}
        for key,value in facts.items():
            cur=obj;parts=key[1:].split('/')
            for segment in parts[:-1]:cur=cur.setdefault(segment,{})
            cur[parts[-1]]=value
        path='tmp/fixture/'+role+'.json';content=dump(obj);data[path]=content
        bindings[role]=dict(path=path,sha256=p.sha(content),expect=facts.copy())
        entries.append(dict(source=path,sha256=p.sha(content),bytes=len(content)))
    s=dict(parent=p.PARENT,branch=p.BRANCH,scope='CLOSED200_COMPACT_EVIDENCE_AND_AUTHOR_REVIEW_NO_BULK',
        accepted=p.ACCEPTED,bindings=bindings,optional_bindings={r:None for r in p.OPTIONAL},
        test_started=False,goal_complete=False,allowed_paths=list(data),files=entries,parent_recovery_references=[])
    real_source=p.source
    p.source=lambda name,allowed:(p.ROOT/name,data[name])
    assert len(p.plan(s)['files'])==6
    mutations=[lambda x:x.update(parent='0'*40),lambda x:x.update(goal_complete=True),
        lambda x:x.update(test_started=True),lambda x:x['bindings'].update(current_state=None),
        lambda x:x['files'][0].update(sha256='0'*64),lambda x:x['files'].append(x['files'][0]),
        lambda x:x.update(accepted={'native':200,'three_view':199})]
    for mutate in mutations:
        t=copy.deepcopy(s);mutate(t)
        try:p.plan(t)
        except (AssertionError,KeyError,TypeError):pass
        else:raise AssertionError('Malformed metadata accepted')
    p.source=real_source
    for path in ['tmp/x/model.pt','tmp/x/preds.npz','tmp/x/backup.tar.gz','tmp/x/model.safetensors','tmp/../bad.json']:
        try:p.relative(path)
        except AssertionError:pass
        else:raise AssertionError('Bulk/unsafe path accepted')
    draft=p.read(p.HERE/'DRAFT_SPEC.json')
    try:p.plan(draft)
    except AssertionError as e:assert 'Actual root binding missing: current_state' in str(e)
    else:raise AssertionError('Unfrozen draft unexpectedly ready')
    proof=dict(status='SOURCE_AND_METADATA_ONLY_PASS_NOT_PUBLISHED',
        original_publisher_sha256=p.PRIOR_SHA,original_verifier_sha256=hashlib.sha256(verifier.read_bytes()).hexdigest(),
        stage_inverse_three_edits_byte_exact=True,freeze_original_function=True,verify_original_function=True,
        synthetic_metadata_positive=1,metadata_refusals=len(mutations),path_refusals=5,
        draft_missing_current_state_rejected=True,Git_calls=0,SSH=0,scientific_work=0,
        final_actual_plan_executed=False)
    p.ns['save'](p.HERE/'SELF_CHECK.json',proof)
    print(json.dumps(proof))
