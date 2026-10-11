"""Editorial release-pointer correction only; no Git or scientific execution."""
from pathlib import Path
import json,hashlib,re,difflib
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
H=lambda b:hashlib.sha256(b).hexdigest()
target=R/'docs/server_deployment_20260923/revision_20260923/rebuttal_working_F50_FL81_gradient64_20261011.md'
before=target.read_bytes();assert H(before)=='bde1bf9d49f6ab5c2d46899b6c34ba794c07ba40e5222487a3127c94e56a0daa'
oldproof=(O/'EDITORIAL_VERIFICATION.json').read_bytes();assert H(oldproof)=='fdb60319e8aa8f37d23b197355faaff89d522d5f711a10965c367e0c8676fe5b'
receipt_path=R/'docs/server_deployment_20260923/training_20260923/publication_closed_increment64_verified_20261011.json'
receipt_bytes=receipt_path.read_bytes();assert H(receipt_bytes)=='ceb120bf2ebc7fe8d635c2fd554f1c69787f70dc0e112398600ca265cdc9add2'
receipt=json.loads(receipt_bytes);commit=receipt['commit'];assert commit=='3edbd954d91bbc5e4993546950202645d82ccbb0'
assert receipt['status']=='COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS'
c=receipt['acceptance_cutoff'];assert (c['native'],c['three_view'],c['F_table_pairs'],c['FLGMM_three_views'],c['FLGMM_table_complete_scene_records'],c['gradient_search'])==(350,350,50,81,80,64)
old='''**Response.** A concrete revision snapshot is available at [GuardFed, commit 817f6fd](https://github.com/YananZHOU5555/GuardFed/tree/817f6fd395564b044ed05e360771817f2f4585de). The release includes the accepted comparison tables, per-seed results, frozen configurations, adapter/source records, aggregation code and documented limitations. Large models and raw array archives are retained separately with archive/member hashes; publishing code is not itself a model backup.

That historical snapshot includes the earlier A80 response update. The current draft adds complete A100, five-scene F-deletion, the FLGMM eight-scene table with its separate partial record, IID Benign hybrid-control evidence and the completed 64-run validation search while retaining the A90 findings; these additions do not establish completion of the remaining benchmark or final evaluation. The final revision release must additionally contain any subsequently accepted methods/results, the frozen evaluation protocol, matching manuscript source and runnable setup instructions. Missing historical checkpoint binaries and adaptation boundaries remain disclosed.'''
new='''**Response.** A verified revision snapshot is available at [GuardFed, commit 3edbd954](https://github.com/YananZHOU5555/GuardFed/tree/3edbd954d91bbc5e4993546950202645d82ccbb0). This fixed release includes the accepted F50 evidence, FLGMM's 81 three-view records and eight-complete-scene table of 80 records, and the completed 64-run Fed-NGA/Huber validation search. It also contains compact per-seed results, frozen configurations, adapter/source records and documented limitations. Large models and raw array archives are retained separately with archive/member hashes; publishing code is not itself a model backup.

The earlier [817f6fd snapshot](https://github.com/YananZHOU5555/GuardFed/tree/817f6fd395564b044ed05e360771817f2f4585de) remains a historical A80-era release. The current author-review draft integrates the already released FLGMM evidence into the complete responses, alongside A100, F50, the IID Benign hybrid-control evidence and the gradient search. This FLGMM prose expansion has not yet been published in that fixed commit. Neither the release nor this draft completes the remaining benchmark or final evaluation. The final revision release must additionally contain any subsequently accepted methods/results, the frozen evaluation protocol, matching manuscript source and runnable setup instructions. Missing historical checkpoint binaries and adaptation boundaries remain disclosed.'''
s=before.decode();assert s.count(old)==1;s=s.replace(old,new);after=s.encode()
assert s.replace(new,old).encode()==before
q=lambda b:[x for x in b.splitlines(keepends=True) if x.startswith(b'>')]
assert q(after)==q(before) and len(q(after))==38
blocks=lambda b:re.findall(rb'(?:^>[^\r\n]*(?:\r?\n|$))+',b,re.M)
assert blocks(after)==blocks(before) and len(blocks(after))==24
assert [x for x in after.splitlines() if x.startswith(b'#')]==[x for x in before.splitlines() if x.startswith(b'#')]
parent=Path(json.loads(oldproof)['parent_path']);original=(R/parent).read_bytes()
back=s.replace(new,old)
for ch in reversed(json.loads((O/'CHANGES.json').read_bytes())):back=back.replace(ch['new'],ch['old'])
assert back.encode()==original
preserved=O/'before_release_pointer';assert not preserved.exists();preserved.mkdir()
(preserved/target.name).write_bytes(before);(preserved/'EDITORIAL_VERIFICATION.json').write_bytes(oldproof)
assert H((preserved/target.name).read_bytes())==H(before) and H((preserved/'EDITORIAL_VERIFICATION.json').read_bytes())==H(oldproof)
assert target.read_bytes()==before;target.write_bytes(after)
(O/'RELEASE_POINTER_DIFF.patch').write_text(''.join(difflib.unified_diff(before.decode().splitlines(True),s.splitlines(True),fromfile='before_release_pointer/'+target.name,tofile=target.name)),encoding='utf8')
(O/'FULL_DIFF_V2.patch').write_text(''.join(difflib.unified_diff(original.decode().splitlines(True),s.splitlines(True),fromfile=parent.name,tofile=target.name)),encoding='utf8')
proof=json.loads(oldproof)
proof.update(status='AUTHOR_EDITORIAL_FL81_AND_VERIFIED_RELEASE_POINTER_CHECK_PASS_NOT_ROOT_ADOPTION',draft_sha256=H(after),editor_source_sha256=H(Path(__file__).read_bytes()),edits=9,
 previous_draft_sha256=H(before),previous_proof_sha256=H(oldproof),previous_snapshots_path=preserved.relative_to(R).as_posix(),
 release_pointer_changes=dict(old=old,new=new),publication_receipt=dict(path=receipt_path.relative_to(R).as_posix(),sha256=H(receipt_bytes),commit=commit),
 release_pointer_only_scientific_cutoff_unchanged=True,new_FL81_prose_not_in_fixed_commit=True,Git_invoked=False,network_verification_repeated=False,
 previous_editor_source_path=(O/'edit_and_verify.py').relative_to(R).as_posix(),original_full_parent_reverse_reconstruction_exact=True)
proof['source_pins']['publication_receipt']={'path':receipt_path.relative_to(R).as_posix(),'sha256':H(receipt_bytes)}
out=O/'EDITORIAL_VERIFICATION_V2.json';assert not out.exists();out.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(draft_sha256=H(after),proof_sha256=H(out.read_bytes()),full_diff_sha256=H((O/'FULL_DIFF_V2.patch').read_bytes()))))
