"""Small metadata and unchanged byte-transport check; no scientific evaluation."""
from pathlib import Path
import ast,hashlib,json,runpy,sys,copy
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent;R=B.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
n=runpy.run_path(str(B/'publish_increment42.py'));g=n['namespace']['closed_guard'];c=json.loads((B/'ACTUAL_CLOSED_INPUTS.json').read_bytes());g(c)
refusals=0
for key,value in [('status','PREPARED'),('parent_commit','0'*40),('counts',dict(c['counts'],three_view=170)),('counts',dict(c['counts'],Hybrid=24)),('roots',dict(c['roots'],future={}))]:
    bad=copy.deepcopy(c);bad[key]=value
    try:g(bad)
    except AssertionError:refusals+=1
    else:raise AssertionError(key)
# The original41 main still extracts the exact original40 AST Try; these lines are unchanged.
original=(R/'tmp/publication41_preparation_20261010/publish_increment41.py').read_text('utf8')
marker='    # Execute the exact existing40 copy/-f/-text/index/failure block, not a reimplementation.'
assert original[original.index(marker):]==n['text'][n['text'].index(marker):]
v=runpy.run_path(str(B/'verify_increment42.py'))
old=(R/'tmp/publication40_preparation_20261010/verify_increment40.py').read_text('utf8');new=v['text']
start="    payload = git('cat-file'";end="    changed = set(filter"
assert old[old.index(start):old.index(end)]==new[new.index(start):new.index(end)]
out=dict(status='PASS_MINIMAL42_SCOPE_AND_ORIGINAL_BYTE_TRANSPORT',positive=1,refusals=refusals,original41_transport_dispatch_suffix_byte_exact=True,original40_committed_blob_loop_byte_exact=True,no_Git_mutations=True,no_scientific_computation=True)
with (B/'SOURCE_REUSE_CHECK.json').open('x',encoding='utf8',newline='\n') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
