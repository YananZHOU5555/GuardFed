"""Derive increment36 from the successful35 source without Git or future receipts."""
from pathlib import Path
import ast,hashlib,re
H=Path(__file__).resolve().parent;O=H.with_name('publication_increment35_prepared_20261010')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(O/'publish_increment35.py')=='0772f210851449811934614913d2cb40a18cf852eb882d973df2a744b0da012d'
assert sha(O/'verify_increment35.py')=='8267976b8f2b89f14c9c2440e25bc11c8b626049a56f9183dd13c7207f5230b6'
s=(O/'publish_increment35.py').read_text('utf8')
changes={
 'increment35':'increment36','INCREMENT35':'INCREMENT36',
 'ae94e17a7b1c3b3f6298eaa9d7f09bbf31cca17d':'fbe6027794ce045bdf79254b995c8d2b1de2fb56',
 'root_delta_20261009T233607Z':'root_delta_20261010T000754Z',
 'C3':'C6','C_after47':'C_after50','C_AFTER47':'C_AFTER50',
 "IDS = ['minus_C_IID_Sp-DFA_seed' + str(n) for n in range(91008, 91011)]":"IDS = ['minus_C_non-IID_Benign_seed' + str(n) for n in range(91001, 91007)]",
 '(147, 3, 150)':'(150, 6, 156)',
 "len(scope['excluded_prior_ids']) == 147":"len(scope['excluded_prior_ids']) == 150",
 'original147_unchanged':'original150_unchanged',
 'prior147_root_adoption_sha256':'prior150_root_adoption_sha256',
 '64732337f35f81bed49e40c60fa1d5229fb6a7557c45272505fee488c5ad20ea':'3859d49fb57c3ecc4b23244012431224d255b02590dab221b2d6464e28aa7dd8',
 "'native':150, 'three_view':150, 'FL_new':9":"'native':156, 'three_view':156, 'FL_new':11",
 'publication_closed_increment34_verified_20261010.json':'publication_closed_increment35_verified_20261010.json',
 'EXACT3_APPROVAL':'EXACT6_APPROVAL','(27,72,9)':'(54,144,18)',
 '[147,147,7,18]':'[150,150,9,19]',
 'celeba_flgmm_fullcoverage_delta_after7_20261010':'celeba_flgmm_fullcoverage_delta_after9_20261010',
 '(7,2,9)':'(9,2,11)',"len(set(fl_proof['accepted_job_ids'])) == 9":"len(set(fl_proof['accepted_job_ids'])) == 11",
 '4403439d39196206e169f14428d68b59b779e1cdd4a5a9fb7d0dd0a3b13dabcf':'ecaaa936289589c2b8b28ff42fa81e7e4eb09206fce49a187781be9ed4143577',
 "main['three_view_new_models_offserver_verified'] == 150":"main['three_view_new_models_offserver_verified'] == 156",
 "['new_accepted'] == 9":"['new_accepted'] == 11",
 "['native_accepted'] == 150":"['native_accepted'] == 156",
 'mechanism_offserver_verified=150, mechanism_three_view_offserver_verified=150':'mechanism_offserver_verified=156, mechanism_three_view_offserver_verified=156',
 'FLGMM_fullcoverage_new_offserver_verified=9':'FLGMM_fullcoverage_new_offserver_verified=11'
}
s=re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda m:changes[m.group(0)],s)
start=s.index("    table = closed.get('C50_table');")
end=s.index("    seal(HERE,'FILES_SHA256.json')",start)
s=s[:start]+"    assert closed.get('C50_table') is None, 'C50 table already published35; do not rebind old C3 to C6'\n    args.table_included = False\n"+s[end:]
needle="    assert previous['status'] == 'COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS'"
pos=s.index(needle)
s=s[:pos]+"    prior_receipt_path = ROOT / TRAIN / 'publication_closed_increment35_20261010.json'\n    assert sha(prior_receipt_path) == previous['publication_receipt_sha256']\n    hybrid_unchanged_guard(hy_proof,sha(hy_adoption),read(prior_receipt_path),'experimental/' + hy_adoption.relative_to(ROOT / 'tmp').as_posix(),prepared['unchanged_Hybrid_root_sha256'])\n"+s[pos:]
needle="        path = path.resolve(); rel = path.relative_to(ROOT); assert not EXCLUDED & set(rel.parts)"
assert s.count(needle)==1
s=s.replace(needle,needle+"\n        if path.is_relative_to(ROOT / 'tmp/celeba_hybrid_screen_execution_20261009'): assert path == hy_adoption, 'Already published Hybrid scope allows unchanged ROOT proof only'",1)
needle="    seal(hy_adoption.parent,'DELIVERY_FILES_SHA256.json',hy_proof['delivery_seal_sha256']); add(hy_adoption)"
assert s.count(needle)==1;s=s.replace(needle,"    add(hy_adoption)  # Already published35; no old Hybrid seal tree/archive.",1)
needle="    assert len(fl_proof['accepted_new_ids']) == len(set(fl_proof['accepted_new_ids'])) == 2"
pos=s.index(needle)
s=s[:pos]+"    assert fl_proof['accepted_new_ids'] == ['FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed' + str(n) + '_fullcoverage' for n in (91001,91002)]\n"+s[pos:]
pos=s.index('def verify_blobs(')
s=s[:pos]+"def hybrid_unchanged_guard(proof,digest,previous_receipt,published_path,expected):\n    assert digest == expected and previous_receipt['copied_sha256'][published_path] == digest\n    assert proof['accepted_total'] == 19 and proof['new_inference'] == 0\n    assert proof['selection_performed'] is proof['scientific_changes'] is proof['final_test'] is False\n\n"+s[pos:]
ast.parse(s)
with (H/'publish_increment36.py').open('x',encoding='utf8',newline='\n') as f:f.write(s)
v=(O/'verify_increment35.py').read_text('utf8')
changes={'increment35':'increment36','ae94e17a7b1c3b3f6298eaa9d7f09bbf31cca17d':'fbe6027794ce045bdf79254b995c8d2b1de2fb56','[900, 150, 150, 9, 19]':'[900, 156, 156, 11, 19]','mechanism_offserver_verified=150, mechanism_three_view_offserver_verified=150':'mechanism_offserver_verified=156, mechanism_three_view_offserver_verified=156','FLGMM_fullcoverage_new_offserver_verified=9':'FLGMM_fullcoverage_new_offserver_verified=11'}
v=re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda m:changes[m.group(0)],v);ast.parse(v)
with (H/'verify_increment36.py').open('x',encoding='utf8',newline='\n') as f:f.write(v)
marker='    try:\n        for name, source in sources.items():'
assert (O/'publish_increment35.py').read_text('utf8').split(marker,1)[1]==s.split(marker,1)[1]
print('INCREMENT36_SOURCE_ONLY_PUBLISHER_AND_VERIFIER_DERIVED')
