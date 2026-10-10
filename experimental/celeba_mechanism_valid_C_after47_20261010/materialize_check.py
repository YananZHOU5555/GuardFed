"""Reuse accepted metadata guards for actual3; no scientific execution."""
from pathlib import Path
import ast,re
H=Path(__file__).resolve().parent
s=(H.with_name('celeba_mechanism_valid_C_after40_20261010')/'check_prepared.py').read_text(encoding='utf-8-sig')
changes={
 'inventory_actual147_Full100refs.json':'inventory_actual150_Full100refs.json',
 '==147':'==150','==653':'==650',"len(scope['excluded_prior_ids'])==140":"len(scope['excluded_prior_ids'])==147",
 'for n in (0,1,3,5,8,11)':'for n in (0,1,2,4,8,11)',
 'prior140_in_scope':'prior147_in_scope','closed140_must_not_replay':'closed147_must_not_replay',
 "[f'minus_C_IID_Sp-DFA_seed{s}' for s in range(91001,91008)]":"[f'minus_C_IID_Sp-DFA_seed{s}' for s in range(91008,91011)]",
 'C_AFTER40':'C_AFTER47','bounded C7 check':'bounded C3 check',
 'celeba_mechanism_valid_C_after36_20261010':'celeba_mechanism_valid_C_after40_20261010',
 "'len(chosen)==7'":"'len(chosen)==3'",
 "service = 'guardfed_celeba_mechanism_valid_C_after36'":"service = 'guardfed_celeba_mechanism_valid_C_after40'",
 'eb1f6c3120febb138e32af25484ac790cf962cd15d5ea9921140119bc5a3317a':'64732337f35f81bed49e40c60fa1d5229fb6a7557c45272505fee488c5ad20ea',
 "'positive_inventory_records':147":"'positive_inventory_records':150",
 "'batch_approval_positive_exact7'":"'batch_approval_positive_exact3'",
 "'per_child_exact7_science_approval_positive':7":"'per_child_exact3_science_approval_positive':3",
}
s=re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda m:changes[m.group(0)],s)
ast.parse(s)
with (H/'check_prepared.py').open('x',encoding='utf-8',newline='\n') as f:f.write(s)
with (H/'run_checks.py').open('x',encoding='utf-8',newline='\n') as f:f.write((H.with_name('celeba_mechanism_valid_C_after40_20261010')/'run_checks.py').read_text(encoding='utf-8-sig'))
