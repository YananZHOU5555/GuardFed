"""Source-only byte reuse and exact cutoff guards; no freeze, stage, Git or science."""
import ast,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import publish_increment45 as p
from prepare_spec import build

def check():
    inverse=p.stage_text.replace('publication45_frozen_bytes_v1','publication44_frozen_bytes_v1').replace('# Git45 sealed evidence/source bytes','# Git44 sealed evidence/source bytes')
    assert inverse==p.base.stage_text
    assert p.freeze is p.base.ns['freeze']
    for name in ('publish_increment45.py','verify_increment45.py','prepare_spec.py','check_source.py'):
        ast.parse((p.HERE/name).read_text(encoding='utf8'))
    p.configure({'native':208,'three_view':200})
    refused=0
    for counts in ({'native':200,'three_view':200},{'native':208,'three_view':208},{'native':210,'three_view':200}):
        try:p.configure(counts)
        except AssertionError:refused+=1
        else:raise AssertionError('wrong cutoff accepted')
    d=build()
    assert all(e['bytes']<100_000_000 for e in d['files'])
    assert sum(e['bytes'] for e in d['files'])<100_000_000
    assert len({p.base.destination(e['source']) for e in d['files']})==len(d['files'])
    return dict(status='SOURCE_AND_ACTUAL_BINDINGS_PASS_NOT_FROZEN_OR_PUBLISHED',
      original_stage_transport_byte_exact_after_schema_comment_reversal=True,
      original_freeze_reused=True,original_committed_blob_verifier_AST_reused=True,
      cutoff_refusals=refused,actual_bindings=4,seal_member_checks=d['original_seal_member_checks'],
      git_or_network_operations=0,scientific_execution=0,
      parent_commit_blob_references_pending_original_plan=True)

if __name__=='__main__':
    out=Path(__file__).with_name('SELF_CHECK.json')
    try:
        d=check()
        with out.open('x',encoding='utf8') as f:json.dump(d,f,indent=2);f.write('\n')
        print(json.dumps(d))
    except Exception as e:
        fail=out.with_name('SOURCE_CHECK_FAILURE.json')
        with fail.open('x',encoding='utf8') as f:json.dump({'error_type':type(e).__name__,'error':str(e)},f,indent=2)
        raise
