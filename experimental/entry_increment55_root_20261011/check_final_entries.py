"""Verify current navigation after actual adoptions; no scientific recomputation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
R=Path(__file__).resolve().parents[2]
T=R/'docs/server_deployment_20260923/training_20260923'
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
S=read(T/'TRAINING_STATE.json');m=S['celeba_mechanism_v1'];a=m['A_three_view_nine_scene_table'];reply=S['latest_rebuttal_draft']
assert (m['scientific_results_offserver_verified'],m['three_view_new_models_offserver_verified'])==(295,295)
assert m['three_view_counts_by_variant']=={'minus_U':100,'minus_C':100,'minus_A':95}
assert (a['paired_models'],a['complete_scenes'],a['mean_SD_scalars'],a['display_cells'])==(90,9,1620,810)
assert a['incorporated_into_full_rebuttal'] and not a['final_test']
assert S['flgmm_fullcoverage_v2_20261009']['new_accepted']==59
assert S['FLGMM_after48_valid_three_view_20261011']['FLGMM_total_three_view_records']==61
assert S['FLGMM_after48_valid_three_view_20261011']['six_scene_table']['complete_scene_records']==60
assert S['gradient64_validation_search_20261010']['offserver_accepted']==42
assert S['hybrid100_fullcoverage_20261010']['new_accepted']==9
assert reply['A90_incorporated'] and reply['clear_A90_incorporated'] and reply['comments_verbatim']==24
assert reply['root_proof_sha256']==H(R/reply['root_proof_path'])=='17ad95f3c558c78b5c9cadd495c3d7802389ecf780840680fc09e958ffe05d11'
assert reply['clear_reader_root_sha256']==H(R/reply['clear_reader_root_path'])=='f7c00fe0d4eb35268d8e21500591f022f330c3a6021ba2794954b759d840f1e7'
assert reply['clear_reader_sha256']==H(R/reply['clear_reader_entry'])=='c2041b79aa3ceedc3fe62f0c76b18ffa7656d7e9fb9b994ef7571a4344d89e46'
assert not reply['whole_rebuttal_complete'] and not reply['manuscript_applied'] and not reply['final_test']
assert S['latest_five_queue_readonly_observation']['growth_proof_sha256']=='8705fb39e987f1800af363d60e988579f74d4e6c0897dbb9c0935b57d263d4f1'
entries={}
for rel,old in read(Path(__file__).with_name('ENTRIES_BEFORE.json')).items():
    p=R/rel;text=p.read_bytes().decode('utf8');_,sep,suffix=text.partition(old['marker']);assert sep
    b=(sep+suffix).encode('utf8');assert hashlib.sha256(b).hexdigest()==old['suffix_sha256'] and len(b)==old['bytes']
    current=text.partition(old['marker'])[0]
    if p.name in ['RUNNING.md','返修实验总览.md','REBUTTAL_COMPLETION_20261009.md']:
        assert 'A90' in current and reply['clear_reader_entry'].removeprefix('docs/') in current
    entries[rel]=dict(sha256=H(p),historical_suffix_unchanged=True)
overview=(R/'docs/返修实验总览.md').read_text(encoding='utf8').partition('以下为 2026-10-04')[0]
for line in overview.splitlines():
    if '最新累计机制288' in line:assert line.startswith('历史增量（发布54截止，当前见下）：')
    if line.startswith('| 英文回复 |'):assert 'A90九场景' in line and 'rebuttal_clear_A90_20261011' in line
completion=(T/'REBUTTAL_COMPLETION_20261009.md').read_text(encoding='utf8').partition('## Historical accepted increment')[0]
assert 'No complete10-seed scene summary yet' not in completion
sources={str(p.relative_to(R)).replace('\\','/'):H(p) for p in [R/'tmp'/n for n in ['update_reactivation_state_20261009.py','update_completion_current_20261009.py','update_overview_closure100_root_20261009.py']]}
proof=dict(status='ROOT55_CURRENT_ENTRIES_AND_ACTUAL_A90_EDITORIAL_ADOPTIONS_PASS',utc=datetime.now(timezone.utc).isoformat(),native=295,three_views=295,variant_counts=m['three_view_counts_by_variant'],A_pairs=90,A_scenes=9,A_stats=1620,A_cells=810,FL_native_new=59,FL_three_view_records=61,FL_complete_scene_records=60,gradient=42,Hybrid_native_new=9,detailed_A90_root=reply['root_proof_sha256'],clear_A90_root=reply['clear_reader_root_sha256'],entries=entries,generators_sha256=sources,historical_suffixes_unchanged=4,science_recomputed=False,whole_rebuttal_complete=False,final_test=False)
out=Path(__file__).with_name('FINAL_ENTRIES_CHECK.json');assert not out.exists();out.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=proof['status'],sha256=H(out))))
