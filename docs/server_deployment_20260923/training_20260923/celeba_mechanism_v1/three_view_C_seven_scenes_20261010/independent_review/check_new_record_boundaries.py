"""Correct two auxiliary fixture targets; no scientific/source/output changes."""
import copy, json
from check_source import B, H, R, function, need, read, refuse, sha
stage = R/'tmp/celeba_mechanism_valid_C_after60_20261010'
prior = read(R/'tmp/celeba_mechanism_valid_C_after56_20261010/inventory_actual160_Full100refs.json')
current = read(stage/'inventory_actual170_Full100refs.json')
expected = [f'minus_C_non-IID_F Flip_seed{s}' for s in range(91001,91011)]
scenes = [('IID',a) for a in ['Benign','F Flip','FedSA','S-DFA','Sp-DFA']]+[('non-IID','Benign'),('non-IID','F Flip')]
namespace = dict(need=need, EXPECTED=expected, SCENES=scenes)
exec(compile(function(B/'build.py','scope'),str(B/'build.py')+':scope','exec'),namespace)
assert namespace['scope'](prior,current)[expected[0]]['terminal_round'] == 70
refusals = []
for key, value in [('terminal_round',69),('actual_alpha',5000)]:
    bad = copy.deepcopy(current)
    target = next(r for r in bad['records'] if r['id'] == expected[0])
    target[key] = value
    refusals.append(refuse('new exact10 record '+key,lambda: namespace['scope'](prior,bad)))
proof = dict(status='AUXILIARY_FIXTURE_TARGET_CORRECTION_PASS_SOURCE_UNCHANGED',original_source_review_sha256=sha(H/'SOURCE_REVIEW.json'),original_fixture_target=current['records'][-1]['id'],correct_target=expected[0],original_two_labels_exercised_old_record_drift=True,corrected_two_new_record_boundaries_refused=refusals,scientific_source_changed=False,actual_C70_statistics=0,CNN=0)
with (H/'SOURCE_BOUNDARY_CORRECTION.json').open('x',encoding='utf8',newline='\n') as f:
    json.dump(proof,f,ensure_ascii=False,indent=2); f.write('\n')
print(json.dumps(dict(path=str(H/'SOURCE_BOUNDARY_CORRECTION.json'),sha256=sha(H/'SOURCE_BOUNDARY_CORRECTION.json'))))
