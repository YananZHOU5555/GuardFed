"""Reuse the prior metadata guard exercise; no scientific execution."""
from pathlib import Path
import ast,re
H=Path(__file__).resolve().parent
s=(H.with_name('celeba_mechanism_valid_C_after36_20261010')/'check_prepared.py').read_text(encoding='utf-8-sig')
changes={
 'inventory_actual140_Full100refs.json':'inventory_actual147_Full100refs.json',
 '==140':'==147','==660':'==653',"len(scope['excluded_prior_ids'])==136":"len(scope['excluded_prior_ids'])==140",
 'prior136_in_scope':'prior140_in_scope','closed136_must_not_replay':'closed140_must_not_replay',
 "[f'minus_C_IID_S-DFA_seed{s}' for s in range(91007,91011)]":"[f'minus_C_IID_Sp-DFA_seed{s}' for s in range(91001,91008)]",
 'C_AFTER36':'C_AFTER40','bounded C4 check':'bounded C7 check',
 'celeba_mechanism_valid_C_after28_20261009':'celeba_mechanism_valid_C_after36_20261010',
 "'len(chosen)==4'":"'len(chosen)==7'",
 "service = 'guardfed_celeba_mechanism_valid_C_after28'":"service = 'guardfed_celeba_mechanism_valid_C_after36'",
 'cd197e57b42dda87bc239736e401cb5accd029313546533f6a48caf27950e1e0':'eb1f6c3120febb138e32af25484ac790cf962cd15d5ea9921140119bc5a3317a',
 "'positive_inventory_records':140":"'positive_inventory_records':147",
 "'batch_approval_positive_exact4'":"'batch_approval_positive_exact7'",
 "'per_child_exact4_science_approval_positive':4":"'per_child_exact7_science_approval_positive':7",
}
s=re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda m:changes[m.group(0)],s)
ast.parse(s)
with (H/'check_prepared.py').open('x',encoding='utf-8',newline='\n') as f:f.write(s)
