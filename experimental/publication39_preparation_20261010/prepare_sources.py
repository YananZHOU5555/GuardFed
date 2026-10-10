from pathlib import Path
import hashlib,json,difflib
ROOT=Path(__file__).resolve().parents[2];B=Path(__file__).resolve().parent;O=ROOT/'tmp/publication38_preparation_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def write(n,s):(B/n).write_text(s,encoding='utf8',newline='\n')
old=(O/'publish_increment38.py').read_text('utf8');s=old
changes=[('increment38','increment39'),('INCREMENT38','INCREMENT39'),('8c71a8f76ac69fbbf9aa84d77a6264d294509daf','3da6cbb1f72e0d750597e8390ab6ac1c2ff7ae44'),('delta_after12_20261010','delta_after14_20261010'),('accepted_delta_after19_20261010','accepted_delta_after21_20261010'),('hybrid_after19_delta','hybrid_after21_delta'),('native=160,three_view=160,FL_new=14,Hybrid=21','native=170,three_view=160,FL_new=16,Hybrid=22'),("'C60_reply_root','FL_root','Hybrid_root'","'native_root','semantic_root','FL_root','Hybrid_root'"),('(12,2,14)','(14,2,16)'),("len(set(fl['accepted_job_ids']))==14","len(set(fl['accepted_job_ids']))==16"),('(19,2,21)','(21,1,22)'),("[:12]","[:14]"),("[12:]","[14:]"),("[:19]","[:21]"),("[19:]","[21:]"),("len(set(link['accepted_job_ids']))==21 and len(set(link['accepted_new_ids']))==2","len(set(link['accepted_job_ids']))==22 and len(set(link['accepted_new_ids']))==1"),("state['flgmm_fullcoverage_v2_20261009']['new_accepted']==14","state['flgmm_fullcoverage_v2_20261009']['new_accepted']==16"),("['offserver_accepted70round_jobs']==21","['offserver_accepted70round_jobs']==22"),("latest_fl['accepted_total']==14","latest_fl['accepted_total']==16"),("latest_hy['accepted']==21","latest_hy['accepted']==22"),('publication_closed_increment37','publication_closed_increment38')]
for a,b in changes:assert a in s,a;s=s.replace(a,b)
s=s.replace("REPLY=Path('docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C60_20261010')","SEMANTIC=Path('docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C60_20261010/semantic_review')\nNATIVE=CHECKS/'mechanism_science_backups_20261009';TAG='root_delta_20261010T013518Z'\nC10=Path('tmp/celeba_mechanism_valid_C_after60_20261010')")
start=s.index('def reply_guard(');end=s.index('def delta_guard(',start)
s=s[:start]+'''def native_guard(r):
    assert r['status']=='ROOT_NATIVE_INCREMENT_MEMBERS_OLD260_RECORDS_AND_EXACT_C_DELTA_PASS'
    assert (r['native_accepted'],r['added_n'],r['full_reused'],r['minus_C_accepted'])==(170,10,100,70)
    assert r['added_ids']==[f'minus_C_non-IID_F Flip_seed{s}' for s in range(91001,91011)]
    assert r['ledger_previous25_entries_exact'] and r['original260_raw_json_record_bytes_exact'] and r['receipt_chain_verified']
    assert not r['three_view_acceptance_changed'] and r['new_inference']==r['oldFull_models_repacked']==0
    assert not r['test'] and not r['whole_rebuttal_complete']

def semantic_guard(r):
    assert r['status']=='INDEPENDENT_C60_AFFECTED_PROSE_SEMANTIC_PASS_NO_CANONICAL_EDIT'
    assert not r['findings'] and not r['required_corrections'] and r['original_comments_verbatim_and_ordered']==24
    assert r['canonical_edits']==r['new_statistics']==r['SSH_calls']==r['Git_actions']==0

'''+s[end:]
a=s.index("    reply,r=pin(");b=s.index("    flp,fl=pin(",a)
s=s[:a]+'''    native,nr=pin('native_root',ROOT/NATIVE/TAG/'ROOT_INDEPENDENT_REVIEW.json');native_guard(nr)
    semantic,sr=pin('semantic_root',ROOT/SEMANTIC/'ROOT_SEMANTIC_REVIEW.json');semantic_guard(sr)
    for rel,row in sr['source_pins'].items():assert sha(ROOT/rel)==row['sha256']
    native_proof=read(ROOT/NATIVE/TAG/'ROOT_DELTA_VERIFICATION.json')
    assert sha(ROOT/NATIVE/TAG/'ROOT_DELTA_VERIFICATION.json')==nr['source_root_proof_sha256']
    for name,key in [(TAG+'.tar.gz','archive_sha256'),(TAG+'.tar.gz.receipt.json','receipt_sha256'),(TAG+'_offserver_verification.json','offserver_proof_sha256')]:assert sha(ROOT/NATIVE/name)==nr[key]==native_proof[key]
    assert sha(ROOT/NATIVE/('mechanism_inspection_v4_'+TAG)/'inspection.json')==nr['inspection_sha256']
    ledger=read(ROOT/NATIVE/TAG/'verified_ledger.json');previous_ledger=read(ROOT/prep['native_previous_ledger'])
    assert sha(ROOT/NATIVE/TAG/'verified_ledger.json')==nr['ledger_sha256'] and sha(ROOT/prep['native_previous_ledger'])==nr['previous_ledger_sha256']
    assert len(ledger['entries'])==26 and ledger['entries'][:-1]==previous_ledger['entries'] and ledger['entries'][-1]['receipt_sha256']==nr['receipt_sha256']
    receipt=read(ROOT/NATIVE/(TAG+'.tar.gz.receipt.json'));assert receipt['accepted_new_ids']==nr['added_ids'] and receipt['reused_full_weights_repacked']==0
    assert receipt['manifest_sha256']==nr['manifest_sha256'] and receipt['previous_receipt_sha256']==previous_ledger['entries'][-1]['receipt_sha256']
    startup=read(ROOT/C10/'execution_candidate/ROOT_STARTUP_OBSERVATION.json')
    assert startup['status']=='ROOT_C_AFTER60_REAL_LINUX_STARTUP_AND_ALLOCATION_PASS' and startup['scientific_offserver_new_accepted']==0
    assert startup['new_training']==startup['new_Full_inference']==0 and startup['test_inference'] is False and startup['original160_not_rerun']
    assert startup['deployment_receipt_sha256']==sha(ROOT/C10/'execution_candidate/deployment_receipt.json')
'''+s[b:]
s=s.replace("assert m['scientific_results_offserver_verified']==m['three_view_new_models_offserver_verified']==160", "assert m['scientific_results_offserver_verified']==170 and m['three_view_new_models_offserver_verified']==160")
s=s.replace("==sha(reply) and state['latest_rebuttal_draft']", "==prep['unchanged_C60_reply_root_sha256'] and state['latest_rebuttal_draft']")
s=s.replace("    tracked=set(git('ls-tree'", "    verify_blobs(PARENT+':',prep['parent_recovery_blobs'])\n    tracked=set(git('ls-tree'",1)
a=s.index("    rows=sealed(ROOT/REPLY");b=s.index("    sealed(ROOT/FL",a)
s=s[:a]+"    sealed(ROOT/SEMANTIC,'FILES_SHA256.json',prep['semantic_seal_sha256'])\n    for row in prep['ready_manifest']:add(ROOT/row['path'],row['sha256'])\n"+s[b:]
s=s.replace("sealed(ROOT/HY,'DELIVERY_FILES_SHA256.json',hy['delivery_seal_sha256'],[omit]);add", "hyrows=sealed(ROOT/HY,'DELIVERY_FILES_SHA256.json',hy['delivery_seal_sha256'],[omit]);assert len(hyrows)==57;add")
s=s.replace('for p in (reply,flp,hyp,state_path,live,prevp,','for p in (native,semantic,flp,hyp,state_path,live,prevp,')
s=s.replace('args.C60_reply_root=reply','args.native_root=native;args.semantic_root=semantic')
s=s.replace("C60_reply_root_path=args.C60_reply_root.relative_to(ROOT).as_posix(),C60_reply_root_sha256=sha(args.C60_reply_root)","native_root_path=args.native_root.relative_to(ROOT).as_posix(),native_root_sha256=sha(args.native_root),semantic_review_path=args.semantic_root.relative_to(ROOT).as_posix(),semantic_review_sha256=sha(args.semantic_root)")
s=s.replace('mechanism_offserver_verified=160','mechanism_offserver_verified=170').replace('FLGMM_fullcoverage_new_offserver_verified=14,Hybrid_offserver_verified=21','FLGMM_fullcoverage_new_offserver_verified=16,Hybrid_offserver_verified=22')
s=s.replace('C60_full_reply_included=True','C60_full_reply_included=False,C60_semantic_review_included=True,C_after60_new_views_accepted=0')
s=s.replace("scope='Adopted C60 author-review text and exact auxiliary deltas only; no new endpoint, manuscript application or final test.'", "scope='Native170, accepted views160 and exact auxiliary deltas. C_after60 startup is not acceptance; semantic review adds no prose/statistics. No C70 table or final test.'")
write('publish_increment39.py',s);write('PUBLISHER_DIFF.patch',''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='increment38',tofile='increment39')))
oldv=(O/'verify_increment38.py').read_text('utf8');v=oldv.replace('increment38','increment39').replace('8c71a8f76ac69fbbf9aa84d77a6264d294509daf','3da6cbb1f72e0d750597e8390ab6ac1c2ff7ae44').replace('[900, 160, 160, 14, 21]','[900, 170, 160, 16, 22]').replace("receipt['C60_full_reply_included'] is True", "receipt['C60_full_reply_included'] is False and receipt['C60_semantic_review_included'] is True and receipt['C_after60_new_views_accepted']==0").replace('mechanism_offserver_verified=160','mechanism_offserver_verified=170').replace('FLGMM_fullcoverage_new_offserver_verified=14, Hybrid_offserver_verified=21','FLGMM_fullcoverage_new_offserver_verified=16, Hybrid_offserver_verified=22')
write('verify_increment39.py',v);write('VERIFIER_DIFF.patch',''.join(difflib.unified_diff(oldv.splitlines(True),v.splitlines(True),fromfile='increment38',tofile='increment39')))
print('Prepared source only; input manifest not yet frozen.')
