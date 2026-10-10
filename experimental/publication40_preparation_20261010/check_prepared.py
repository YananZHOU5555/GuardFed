from pathlib import Path
import ast,copy,hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent;R=B.parents[1];O=R/'tmp/publication39_preparation_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
sp=importlib.util.spec_from_file_location('publication40_check',B/'publish_increment40.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
c=read(B/'ACTUAL_CLOSED_INPUTS.json');r=read(R/m.BACKUP/'ROOT_ADOPTION_REVIEW.json');t=read(R/m.C70/'ROOT_VERIFICATION.json')
m.closed_guard(c);m.closure_guard(r);m.table_guard(t)
refused=[]
for label,fn,obj,key,val in [('parent',m.closed_guard,c,'parent_commit','3da6cbb1f72e0d750597e8390ab6ac1c2ff7ae44'),('oldviews',m.closed_guard,c,'counts',dict(m.COUNTS,three_view=160)),('future',m.closed_guard,c,'counts',dict(m.COUNTS,three_view=180)),('C10partial',m.closure_guard,r,'accepted_new',9),('C10oldchanged',m.closure_guard,r,'original160_unchanged',False),('C10wrongid',m.closure_guard,r,'accepted_new_ids',[]),('C70prepared',m.table_guard,t,'status','PREPARED'),('C70wrongcount',m.table_guard,t,'complete_scenes',10),('C70allnonIID',m.table_guard,t,'full_nonIID_coverage',True),('C70fullreply',m.table_guard,t,'incorporated_into_full_rebuttal',True),('C70test',m.table_guard,t,'test',True)]:
 d=copy.deepcopy(obj);d[key]=val
 try:fn(d)
 except (AssertionError,KeyError,TypeError):refused.append(label)
 else:raise AssertionError(label+' incorrectly accepted')
s=(B/'publish_increment40.py').read_text('utf8');old=(O/'publish_increment39.py').read_text('utf8')
assert s[s.index('    try:\n        for name,source'):]==old[old.index('    try:\n        for name,source'):]
v=(B/'verify_increment40.py').read_text('utf8');ov=(O/'verify_increment39.py').read_text('utf8')
assert v[v.index("    payload = git('cat-file'"):v.index('    changed = set')]==ov[ov.index("    payload = git('cat-file'"):ov.index('    changed = set')]
for p in B.glob('*.py'):ast.parse(p.read_text('utf8'))
proof=dict(status='PASS_SOURCE_METADATA_ONLY_NO_STAGE',positive_guard_calls=3,refusal_checks=len(refused),refused=refused,actual_C10_C70_roots=True,copy_index_failure_body_byteexact39=True,committed_blob_parser_byteexact39=True,Git_mutations=0,SSH=0,CNN=0,new_statistics=0)
(B/'SELF_CHECK.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8',newline='\n');print(json.dumps(proof))
