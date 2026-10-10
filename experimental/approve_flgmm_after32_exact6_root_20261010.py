"""Source review only for six frozen original FL96 terminals."""
from pathlib import Path
import ast,datetime,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'tmp/celeba_flgmm_fullcoverage_delta_after32_20261010'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def main():
    seal=HERE/'PREPARED_FILES_SHA256.json'
    assert sha(seal)=='43d177ed79a8ec119629b02d9d220422d141ee4e1333d82d7f062ff1c110dd3b'
    for name,pin in read(seal)['files'].items():assert sha(HERE/name)==pin['sha256'] and (HERE/name).stat().st_size==pin['bytes']
    handoff=read(HERE/'PREPARED_HANDOFF.json');checks=read(HERE/'PREPARE_CHECKS.json')
    ids=[f'FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed{s}_fullcoverage' for s in range(91005,91011)]
    assert handoff['authorized_ids']==ids and handoff['prior_root_accepted']==32
    assert checks['per_ID_scientific_loop_bytes_exact'] and checks['archive_and_strict_body_bytes_exact'] and checks['original_verifier_bytes_exact']
    source=(HERE/'run_once.py').read_text('utf8');ast.parse(source)
    assert "S = Path('F:/YananResearchStorage/GuardFed')" in source
    assert "review['source_adoptable'] is True" in source and "review['exact_selected_ids']==IDS" in source
    assert "review['prepared_seal_sha256']==sha(H/'PREPARED_FILES_SHA256.json')" in source
    diff=(HERE/'SOURCE_DIFF.patch').read_text('utf8')
    assert 'wanted==authorized_ids' in diff and "assert not queue['failed']" in diff and '    before=repo_identity' not in diff
    review=dict(status='ROOT_EXACT6_SOURCE_AND_TRANSPORT_REVIEW_PASS_NOT_ACCEPTANCE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_adoptable=True,exact_selected_ids=ids,prepared_seal_sha256=sha(seal),
        handoff_sha256=sha(HERE/'PREPARED_HANDOFF.json'),source_diff_sha256=sha(HERE/'SOURCE_DIFF.patch'),
        prior_accepted32_prefix_required=True,original_scientific_loop_strict_archive_and_offserver_checker_unchanged=True,
        new_refusals_only='Exact6 complete/order and queue.failed before science',
        source_review='Read entire run_once.py, exact effective diff, prior finalize and actual pinned findings; inherited original resource/transport/verifier lifecycle preserved.',
        collector_CPU=110,collector_threads=1,CUDA_VISIBLE_DEVICES='',bulk_F_only=True,
        actual_resource_preflight_still_required=True,collect_authorized_once=True,new_accepted=0,
        root_actual_acceptance_required=True,final_test=False,training_or_CNN=False)
    with (HERE/'ROOT_SOURCE_REVIEW.json').open('x',encoding='utf8') as f:json.dump(review,f,indent=2);f.write('\n')
    print(json.dumps(dict(path=(HERE/'ROOT_SOURCE_REVIEW.json').relative_to(ROOT).as_posix(),sha256=sha(HERE/'ROOT_SOURCE_REVIEW.json'))))
if __name__=='__main__':main()
