from pathlib import Path
import hashlib
p=Path('/workspace/guardfed_checks/celeba_flgmm_fullcoverage_delta_after14_20261010/collect_delta.py')
s=p.read_text('utf8')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='575ecbd7f948ad66d58d28cb336100da1efc965559ee05ba500f3ef259d98c20'
for a,b in [("    authorized_ids=['FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed91006_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed91007_fullcoverage']\n", "    authorized_ids=['FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91005_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91006_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91007_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91008_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91009_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91010_fullcoverage']\n"), ("    queue_snapshot=read(RELEASE/'queue_progress.json')\n", "    queue_snapshot=read(RELEASE/'queue_progress.json')\n    assert not queue_snapshot['failed'], 'Frozen queue failure forbids collection'\n"), ('    wanted=[identity for identity in wanted if identity in authorized_ids]\n', "    wanted=[identity for identity in wanted if identity in authorized_ids]\n    assert wanted==authorized_ids, 'All six authorized outputs must be terminal, unique and ordered'\n"), ("    queue=read(RELEASE/'queue_progress.json')\n", "    queue=read(RELEASE/'queue_progress.json')\n    assert not queue['failed'], 'Queue failure forbids collection'\n")]:
    assert s.count(a)==1
    s=s.replace(a,b)
exec(compile(s,__file__,'exec'),globals())
