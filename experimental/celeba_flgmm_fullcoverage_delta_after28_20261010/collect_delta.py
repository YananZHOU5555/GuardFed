from pathlib import Path
import hashlib
p=Path('/workspace/guardfed_checks/celeba_flgmm_fullcoverage_delta_after14_20261010/collect_delta.py')
s=p.read_text('utf8')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='575ecbd7f948ad66d58d28cb336100da1efc965559ee05ba500f3ef259d98c20'
a="    authorized_ids=['FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed91006_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed91007_fullcoverage']\n"
b="    authorized_ids=['FLGMM_Tg20_L2.0_lr0.001_IID_FedSA_seed91010_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91002_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91003_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91004_fullcoverage']\n"
assert s.count(a)==1
exec(compile(s.replace(a,b),__file__,'exec'),globals())
