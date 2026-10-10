"""Verify current navigation after actual adoptions; no scientific recomputation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
R=Path(__file__).resolve().parents[2]
T=R/'docs/server_deployment_20260923/training_20260923'
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
S=read(T/'TRAINING_STATE.json');m=S['celeba_mechanism_v1'];a=m['A_three_view_ten_scene_table'];reply=S['latest_rebuttal_draft']
assert (m['scientific_results_offserver_verified'],m['three_view_new_models_offserver_verified'])==(304,304)
assert m['three_view_counts_by_variant']=={'minus_U':100,'minus_C':100,'minus_A':100,'minus_F':4}
assert (a['paired_models'],a['complete_scenes'],a['mean_SD_scalars'],a['display_cells'])==(100,10,2106,1053)
assert a['incorporated_into_full_rebuttal'] and not a['final_test']
assert S['flgmm_fullcoverage_v2_20261009']['new_accepted']==63
assert S['FLGMM_after48_valid_three_view_20261011']['FLGMM_total_three_view_records']==61
assert S['FLGMM_after48_valid_three_view_20261011']['six_scene_table']['complete_scene_records']==60
assert S['gradient64_validation_search_20261010']['offserver_accepted']==46
assert S['hybrid100_fullcoverage_20261010']['new_accepted']==12
assert reply['A100_incorporated'] and reply['clear_A100_incorporated'] and reply['comments_verbatim']==24
assert (reply['remaining_control_variants'],reply['remaining_method_coverages'])==(5,7)
assert 'other_six_controls_pending' not in reply and 'remaining_eight_methods_pending' not in reply
assert reply['superseded_pending_flags_before_A100']=={'other_six_controls_pending':True,'remaining_eight_methods_pending':True}
assert reply['root_proof_sha256']==H(R/reply['root_proof_path'])=='195eb98c26cc2a4f0c3576cbb7a6b57a73fbc0d4ccc17a683f9817569e164c54'
assert reply['clear_reader_root_sha256']==H(R/reply['clear_reader_root_path'])=='6c071c87b07e8fc6b36f93553875cbce33bf034b8b3d71584d37e6b70535096c'
assert reply['clear_reader_sha256']==H(R/reply['clear_reader_entry'])=='e3ed0f76c452e6f6a0ac03922802f46ccb7ab6aae336631f48f0e0400f9ec452'
assert not reply['whole_rebuttal_complete'] and not reply['manuscript_applied'] and not reply['final_test']
assert S['latest_five_queue_readonly_observation']['growth_proof_sha256']=='333fd6ee69973b716781f5d0d52a96561503b15992b8cebf81ba9438cda44067'
observed=S['latest_five_queue_readonly_observation']
assert m['queue_completed_observed']==observed['main_terminal']==305 and m['queue_observation_utc']==observed['utc']=='2026-10-10T18:17:26.732274+00:00'
assert (m['queue_active_observed'],m['queue_pending_observed'],m['queue_failures_observed'])==(8,487,0)
assert 'Historical main-only resource sample' in S['last_health_check_scope']
assert S['gradient64_validation_search_20261010']['Huber_screen_accepted']==14 and S['gradient64_validation_search_20261010']['recipe_selected'] is False
entries={}
for rel,old in read(Path(__file__).with_name('ENTRIES_BEFORE.json')).items():
    p=R/rel;text=p.read_bytes().decode('utf8');_,sep,suffix=text.partition(old['marker']);assert sep
    b=(sep+suffix).encode('utf8');assert hashlib.sha256(b).hexdigest()==old['suffix_sha256'] and len(b)==old['bytes']
    current=text.partition(old['marker'])[0]
    if p.name in ['RUNNING.md','返修实验总览.md','REBUTTAL_COMPLETION_20261009.md']:
        assert 'A100' in current and reply['clear_reader_entry'].removeprefix('docs/') in current
    entries[rel]=dict(sha256=H(p),historical_suffix_unchanged=True)
overview=(R/'docs/返修实验总览.md').read_text(encoding='utf8').partition('以下为 2026-10-04')[0]
for line in overview.splitlines():
    if '最新累计机制288' in line:assert line.startswith('历史增量（发布54截止，当前见下）：')
    if line.startswith('| 英文回复 |'):assert 'A100十场景' in line and 'rebuttal_clear_A100_20261011' in line
completion=(T/'REBUTTAL_COMPLETION_20261009.md').read_text(encoding='utf8').partition('## Historical accepted increment')[0]
assert 'No complete10-seed scene summary yet' not in completion
sources={str(p.relative_to(R)).replace('\\','/'):H(p) for p in [R/'tmp'/n for n in ['update_reactivation_state_20261009.py','update_completion_current_20261009.py','update_overview_closure100_root_20261009.py']]}
proof=dict(status='ROOT57_CURRENT_ENTRIES_AND_ACTUAL_INCREMENT_ADOPTIONS_PASS',utc=datetime.now(timezone.utc).isoformat(),native=304,three_views=304,variant_counts=m['three_view_counts_by_variant'],A_pairs=100,A_scenes=10,A_stats=2106,A_cells=1053,FL_native_new=63,FL_three_view_records=61,FL_complete_scene_records=60,gradient=46,Hybrid_native_new=12,detailed_A100_root=reply['root_proof_sha256'],clear_A100_root=reply['clear_reader_root_sha256'],entries=entries,generators_sha256=sources,historical_suffixes_unchanged=4,science_recomputed=False,whole_rebuttal_complete=False,final_test=False)
out=Path(__file__).with_name('FINAL_ENTRIES_CHECK.json');assert not out.exists();out.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=proof['status'],sha256=H(out))))
