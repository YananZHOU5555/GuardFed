from pathlib import Path
import ast,copy,datetime,hashlib,json,subprocess,sys
sys.dont_write_bytecode=True
R=Path.cwd();H=R/'tmp/publication_increment36_root_review_20261010';B=R/'tmp/publication_increment36_prepared_20261010';O=R/'tmp/publication_increment35_prepared_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes())
def save(n,v):
 with (H/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
assert sha(B/'FILES_SHA256.json')=='2e22fb9c4ed8de3d27f439a310263f5a02932f548c508d49becf136a668dcaca'
seal=read(B/'FILES_SHA256.json')
for n,v in seal['files'].items():assert sha(B/n)==v['sha256'] and (B/n).stat().st_size==v['bytes']
assert len(seal['files'])==15
p=read(B/'PREPARED_INPUTS.json')
for n,s in p['ready_sha256'].items():assert sha(R/n)==s
assert len(p['ready_sha256'])==31
# Original bounded fixture entry has no report argument and makes no source writes/Git calls.
check=subprocess.run([sys.executable,'-B',str(B/'check_prepared.py')],capture_output=True,timeout=60)
(H/'CHECK_STDOUT.json').write_bytes(check.stdout);(H/'CHECK_STDERR.log').write_bytes(check.stderr)
assert check.returncode==0
actual=json.loads(check.stdout);assert actual==read(B/'SELF_CHECK_FINAL.json') and actual['refusal_count']==26
s=(B/'publish_increment36.py').read_text();old=(O/'publish_increment35.py').read_text();tree=ast.parse(s)
fn=lambda src,n:ast.get_source_segment(src,next(x for x in ast.parse(src).body if isinstance(x,ast.FunctionDef) and x.name==n))
assert fn(s,'verify_blobs')==fn(old,'verify_blobs')
marker='    try:\n        for name, source in sources.items():';assert s.split(marker,1)[1]==old.split(marker,1)[1]
plan=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='plan')
# Independently run actual two scope assertions without entering plan or Git.
assertions=[x for x in plan.body if isinstance(x,ast.Assert)]
count=next(x for x in assertions if "closed['counts']" in ast.unparse(x)); table=next(x for x in assertions if "closed.get('C50_table')" in ast.unparse(x))
ns={'closed':{'counts':{'native':156,'three_view':156,'FL_new':11,'Hybrid':19,'baseline_valid':900},'C50_table':None}}
def run(node,scope):exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_source_scope_fixture','exec'),scope)
run(count,ns);run(table,ns);denied=[]
for k in ns['closed']['counts']:
 bad=copy.deepcopy(ns);bad['closed']['counts'][k]+=1
 try:run(count,bad)
 except AssertionError:denied.append(k)
 else:raise AssertionError(k)
bad=copy.deepcopy(ns);bad['closed']['C50_table']={}
try:run(table,bad)
except AssertionError:denied.append('C50_already_published_nonnull')
else:raise AssertionError('C50 accepted')
# Actual C50 author-review source, not future C6 or table claim.
rel=Path('docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C50_20261010');a=R/rel
assert sha(a/'ROOT_REVIEW.json')=='64b083fb61cb8bff17304031af01a0d692d03fdc16e956de0dcd154f5a524691'
asl=read(a/'FILES_SHA256.json'); rows=asl.get('members')
if rows is None:
 rows=[dict(path=n,**v) for n,v in asl['files'].items()]
assert len(rows)==13
for row in rows:
 assert sha(a/row['path'])==row['sha256']
 assert (rel/row['path']).as_posix() in p['required_extra_paths']
for n in ['FILES_SHA256.json','ROOT_REVIEW.json']:assert (rel/n).as_posix() in p['required_extra_paths']
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
report=dict(status='PASS_SOURCE_ONLY_INCREMENT36_READY_FOR_ROOT_CLOSED_INPUT_BINDING',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_seal_sha256=sha(B/'FILES_SHA256.json'),source_members=15,ready_pins_verified=31,publisher_sha256=sha(B/'publish_increment36.py'),verifier_sha256=sha(B/'verify_increment36.py'),metadata_check_sha256=sha(B/'check_prepared.py'),original_fixture_reexecuted=True,positive_exact6_fixture=True,original_refusals=26,independent_scope_positive_checks=2,independent_scope_refusals=denied,parent_commit=p['parent_commit'],parent_HEAD_or_live_state_not_rechecked=True,index_mutation_block_byte_exact_to_adopted35=True,checks=dict(closed156_C6_before_mutation=True,closed_FL11_Hybrid19_baseline900=True,C6_same_checkpoint_strict_receipt_root_chain=True,Hybrid19_bound_actual_published35_SHA=True,Hybrid_old_namespace_excludes_archive_seal_LATEST=True,C50_table_forced_null=True,complete_C50_author_review_root_actual_SHA_bound=True,C50_author_review_required_extra_files=15,force_add_text_renormalization_blob_SHA_retained=True,index_commit_remote_branch_checks_retained=True,unique_new_archive_SHA_and_no_tracked_archive=True,individual_total_under100MB=True,raw_weights_extracted_trees_excluded=True,secret_pattern_guard_retained=True,no_commit_push_in_publisher=True),blocking_findings=[],limits=['No plan/main, Git, SSH, staging, commit, push, live observation or science acceptance performed.','Root must bind actual C6 closure and consistent156/156 state and review external extra_pins before staging.','C50 author-review text is an adopted source artifact, not manuscript application or submission; C50 table remains null.','Origin URL and exhaustive credentials are not independently audited; original branch and pattern-based guards retained.'])
save('ROOT_REVIEW.json',report)
(H/'README.md').write_text('Increment36 source-only PASS.15 sealed sources/31 ready pins checked; original26 refusal fixtures plus6 independent scope refusals. C6 exact6 strict closure required; native/views156, FL11, Hybrid19 and baseline900. Hybrid old namespace allows unchanged already-published ROOT proof only; C50 table forced null, adopted author-review text15 extra pins bound. Original force-add/-text/index/commit checks retained. No Git/SSH/publish/science execution. Root still supplies actual closed input and stage authorization.\n',encoding='utf8')
files=[dict(path=f.name,size=f.stat().st_size,sha256=sha(f)) for f in sorted(H.iterdir()) if f.is_file()]
save('REVIEW_FILES_SHA256.json',dict(status='SOURCE_REVIEW_ONLY',members=files))
print(json.dumps(dict(review_sha256=sha(H/'ROOT_REVIEW.json'),seal_sha256=sha(H/'REVIEW_FILES_SHA256.json'),members=len(files))))
