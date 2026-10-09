"""Adopt the sealed complete 24-comment writing copy; no manuscript application."""
from pathlib import Path, PurePosixPath
import datetime, hashlib, json, re, shutil, subprocess, sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tmp/guardfed_rebuttal_integrated100_20261009'
DEST=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated100_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(SOURCE/'FINAL_FILES_SHA256.json')=='250cae33a02ba400225939e258586ba57009a812749696785190df29eabb983b'
seal=read(SOURCE/'FINAL_FILES_SHA256.json')
assert len(seal['files'])==17 and not DEST.exists()
for name,pin in seal['files'].items():
 rel=PurePosixPath(name)
 assert not rel.is_absolute() and '..' not in rel.parts
 assert sha(SOURCE/name)==pin['sha256'] and (SOURCE/name).stat().st_size==pin['bytes']
handoff=read(SOURCE/'FINAL_HANDOFF.json')
assert handoff['accepted_U_scope']==dict(complete_scenes=10,Full_models=100,minus_U_models=100,views=['native','raw','shared_calibration'],seed_panels=[10,9,6])
assert handoff['C4_excluded_from_three_view_table'] and handoff['primary_endpoint_pending'] and handoff['final_test_pending']
assert handoff['other_seven_mechanism_controls_pending'] and handoff['remaining_eight_method_matrix_pending']
for name,pin in read(SOURCE/'SOURCE_MAP.json')['input_files'].items():
 assert sha(ROOT/name)==pin['sha256'] and (ROOT/name).stat().st_size==pin['bytes']
checks=subprocess.run([sys.executable,'-B',str(SOURCE/'verify_delivery.py')],capture_output=True,text=True,check=True)
assert json.loads(checks.stdout)==read(SOURCE/'DELIVERY_CHECKS.json')
old=ROOT/'docs/server_deployment_20260923/revision_20260923/rebuttal_integrated71_v2_20261009'
quote=r'\*\*Original comment \(verbatim\)\.\*\*\n\n(.*?)\n\n\*\*Response\.\*\*'
text=(SOURCE/'rebuttal_integrated_20261009.md').read_text(encoding='utf8')
assert len(re.findall(quote,text,re.S))==24
assert re.findall(quote,text,re.S)==re.findall(quote,(old/'rebuttal_integrated_20261009.md').read_text(encoding='utf8'),re.S)
DEST.mkdir()
for name in list(seal['files'])+['FINAL_FILES_SHA256.json']:
 shutil.copyfile(SOURCE/name,DEST/name)
 assert sha(DEST/name)==sha(SOURCE/name)
proof=dict(status='ROOT_FULL24_U100_WRITING_COPY_SOURCE_NUMBERS_COMMENTS_AND_DIFF_PASS',
 checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_seal_sha256=sha(SOURCE/'FINAL_FILES_SHA256.json'),
 sealed_members_verified=17,input_members_verified=37,comments_verbatim=24,numeric_pointer_format_checks=37,
 local_links_existing=36,external_links_syntax_only=1,changed_paragraphs=21,unchanged_paragraphs=206,
 prior_two_documents_reversibly_exact=True,source_inputs_unchanged=True,old_canonical_draft_unchanged=True,
 negative_results_and_pending_limits_retained=True,submission_gate='DO_NOT_SUBMIT_BEFORE_FULL_COHORT',
 upstream_U100_root_sha256=handoff['upstream_root_evidence']['three_view100_ROOT'],
 new_statistics=False,new_inference=False,new_threshold_fit=False,submitted_manuscript_edited=False,
 verifier_stdout_sha256=hashlib.sha256(checks.stdout.encode()).hexdigest(),
 entry=(DEST/'rebuttal_integrated_20261009.md').relative_to(ROOT).as_posix(),
 manuscript_candidate=(DEST/'manuscript_insertions_integrated_20261009.md').relative_to(ROOT).as_posix())
for target in (SOURCE/'ROOT_REVIEW.json',DEST/'ROOT_REVIEW.json'):
 with target.open('x',encoding='utf8',newline='\n') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
assert sha(SOURCE/'ROOT_REVIEW.json')==sha(DEST/'ROOT_REVIEW.json')
print(json.dumps(dict(status=proof['status'],root_sha256=sha(DEST/'ROOT_REVIEW.json'),entry=proof['entry'])))
