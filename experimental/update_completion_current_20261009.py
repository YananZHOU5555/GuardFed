"""Refresh only the current completion section from accepted receipts."""
from pathlib import Path
import json, hashlib
ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/'docs/server_deployment_20260923/training_20260923'
state=json.loads((TRAIN/'TRAINING_STATE.json').read_bytes())
fl47_current=state.get('FLGMM_closed47_valid_three_view_20261011',{})
if fl47_current.get('complementary_adopted'):
    fl47_proof=ROOT/'tmp/celeba_flgmm_closed47_root_execution_20261011/ROOT_SCIENTIFIC_ADOPTION.json'
    assert hashlib.sha256(fl47_proof.read_bytes()).hexdigest()==fl47_current['root_proof_sha256']=='22fc113add73814ae40accf563ce5f63bd63d50bcf53b7262bdb2b996b016dfe'
    a47=json.loads(fl47_proof.read_bytes())
    assert a47['status']==fl47_current['status']=='ROOT_FLGMM_CLOSED47_COMPLEMENTARY_EVIDENCE_ADOPTED_WINDOWS_REFIT_FAILED_PRESERVED'
    assert fl47_current['root_proof_path']==fl47_proof.relative_to(ROOT).as_posix() and a47['root_adoption']
    assert a47['new_three_view_records_accepted']==fl47_current['root_new_three_view_acceptances']==47 and a47['FLGMM_total_three_view_records']==fl47_current['FLGMM_total_three_view_records']==48
    assert a47['exact_ids']==fl47_current['exact_ids'] and len(set(a47['exact_ids']))==47 and len(a47['prior_interface_explicitly_reused'])==1
    assert a47['native_training_acceptance_unchanged_by_this_action']==fl47_current['source_native_training_records']==44 and a47['original_screen_reuse_separate']==fl47_current['separate_screen_reuse']==4
    assert a47['mechanism_three_view_cutoff_unchanged']==260 and state['celeba_mechanism_v1']['three_view_new_models_offserver_verified']>=260
    assert a47['Linux_whole_original_saved_check_pass'] and a47['Linux_root_fit_verified'] and a47['Linux_original_root_refit_records']==47
    assert a47['Windows_saved_outputs_audit_pass'] and a47['Windows_saved_outputs_audit_fit_calls']==0
    assert not any(a47[k] for k in ('Windows_exact_refit_pass','Windows_whole_saved_check_pass','Windows_original_array_refit_block_pass','previous_Windows_exact_refit_condition_satisfied','cross_platform_bitwise_recalibration_claimed','test'))
    assert not fl47_current['validation_hold'] and fl47_current['original_validation_hold_preserved']
    assert hashlib.sha256((ROOT/fl47_current['validation_hold_path']).read_bytes()).hexdigest()==fl47_current['validation_hold_sha256']=='a87f1dac4b42d6ec1754d88b772c44a5c0d7c95ca3bb3d684cc7a97a34dfa6f1'
main=state['celeba_mechanism_v1'];baseline=state['final_evaluator_runtime_20261009']
A80=main.get('A_three_view_eight_scene_table')
if A80:
    A80_root=ROOT/A80['root_proof_path']
    assert hashlib.sha256(A80_root.read_bytes()).hexdigest()==A80['root_proof_sha256']=='d2245e4d9de68b415931dccc7d1f70c39a871c226c59d9973675dfc1d3cc1bc7'
    A80_proof=json.loads(A80_root.read_bytes())
    assert A80_proof['root_adoption'] and A80_proof['complete_scenes']==8 and A80_proof['paired_models']==80
    assert A80_proof['mean_SD_scalars_recomputed']==1458 and A80_proof['display_cells']==729
accepted=main['scientific_results_offserver_verified'];replayed=baseline['actual_native_valid_image_replays_accepted']
mechanism_views=main['three_view_new_models_offserver_verified']
C_views=main.get('three_view_counts_by_variant',{}).get('minus_C',0)
C_view_note=(f'{C_views} accepted C terminal checkpoints. C IID Benign has ten paired seeds only when the exact C11 increment is offserver-adopted; its three-view table requires a separate source/statistics review. Other C scenes remain incomplete. ' if C_views>1 else
    f'{C_views} accepted C-interface gate. The single C gate does not supply a complete C scene or a component performance conclusion. ')
if main.get('C_three_view_single_scene_table'):
    C_view_note=('12 accepted C terminal checkpoints. The IID Benign ten-pair raw/native/shared table now passes separate root adoption:162 mean/SD scalars,81 display cells,216 count-reconstructed metrics and54 exact native scalar matches. '
        'The two F Flip pairs remain excluded from means; other C scenes are incomplete. Raw deletion lowers ACC by0.014pp and both disparity means; calibrated views retain ACC/AEOD/ASPD tradeoffs, with native/shared identical. '
        'All10/9/6 panels, Full2CPU/8GPU versus C10CPU, training environments and selection/test history remain explicit. No C necessity, causality or significance is established. ')
complete_native_scenes=main['latest_interim_paper_table']['complete_paired_scenes']
published=state['latest_publication_verification']
paired_table=main.get('latest_paired_three_view_table')
paired_note=('Full paired three-view receipts remain missing and will join actual baseline replay acceptance, without new Full inference or substituting old calibration. '
    if not paired_table else
    'A subsequent root-reviewed identity join now supplies actual Full100 replay receipts and the exact60 Full counterparts for six complete scenes. '
    'The raw/native/shared tables include all three consistent10/9/6-seed panels and paired differences, with972 independently recomputed mean/sampleSD checks. '
    'For these120 records, native and shared-calibration metrics are exactly equal. DeletingU lowers native accuracy in all six scenes by0.309–1.384 percentage points and lowersASPD in all six, '
    'whileAEOD increases in five; rawAEOD decreases in five. These tradeoffs do not establish that every term is necessary. '
    'Full replay mixes5CPU/95GPU overall (5CPU/55GPU in the displayed60); controls useCPU. The disclosed training-build/driver and validation-selection history remain limitations, '
    'and neither a uniform-device final comparison nor a primary endpoint has been established. ')
if paired_table and paired_table['complete_scenes'] == 7:
    paired_note = ('A root-reviewed join supplies70 complete Full–minus_U pairs in seven scenes;71 pairs/142 individual records are retained, with the incomplete non-IID FedSA91001 pair excluded from means. '
        'Raw/native/shared10/9/6-seed tables and paired differences passed1134 independent mean/sampleSD checks. '
        'Native/shared metrics and counts coincide for all140 displayed models. DeletingU lowers native accuracy in all seven scenes by0.309–1.384pp andASPD in all seven;AEOD increases in six, while rawAEOD decreases in five and increases in two(IIDSp-DFA andnon-IID F Flip). '
        'The original six-scene records/statistics are unchanged. Full5CPU/65GPU and69cu128/1cu130 versus70CPU/cu128 controls, driver/runtime and validation-selection limits remain explicit. Neither a uniform-device final comparison nor a primary endpoint has been established. ')
if paired_table and paired_table['complete_scenes'] == 9:
    paired_note = ('The later independently accepted nine-scene join contains90 complete Full–minus_U pairs and retains92 pairs/184 individual records. '
        'The two non-IID Sp-DFA pairs are incomplete and excluded from scene means. Raw/native/shared fixed10/9/6-seed panels include729 displayed cells; '
        '1458 mean/sampleSD scalars and1656 metrics reconstructed from group counts pass independent checks. All189 prior seven-scene statistic rows and81 native92 display rows remain exact. '
        'Native and shared outcomes coincide for all184 models. In the newly complete non-IID FedSA and S-DFA scenes, deletingU lowers raw accuracy by0.492 and0.453pp, with disparity tradeoffs retained. '
        'Complete-scene Full inference is5CPU/85GPU and its training88cu128/2cu130; controls use90CPU/cu128. Mixed devices, historical environments, validation selection and prior test exposure remain limitations. '
        'This is a descriptive validation comparison, not an isolated aggregation effect, a selected primary endpoint or evidence that every scoring term is indispensable. ')
if paired_table and paired_table['complete_scenes'] == 10:
    paired_note = ('The independently accepted ten-scene join contains100 complete Full–minus_U pairs and200 records, with all IID/non-IID, five scenarios and ten paired seeds complete. '
        'Raw/native/shared fixed10/9/6-seed panels contain810 display cells;1620 mean/sampleSD scalars,1800 count-reconstructed metrics and every display cell pass root checks. '
        'All243 prior nine-scene statistic rows and184 individual records remain exact. Native and shared views coincide for all200 models. '
        'In the newly complete non-IID Sp-DFA scene, removingU lowers raw ACC by0.151pp and raises both disparity means; native ACC is lower by0.121pp butASPD also falls. '
        'These view-dependent tradeoffs are retained. Full inference5CPU/95GPU and training98cu128/2cu130 versus controls100CPU/cu128, historical environments, validation selection and prior test exposure remain explicit. '
        'Only the minus_U comparison is complete; the other seven variants, isolated-aggregation claims, primary-endpoint choice and final-test evaluation remain open. ')
increments=main['incremental_science_backups']
latest_increment=increments[-1]
flgmm=state['flgmm_screen32_20261009']
fl_selection_note=('Numerical recipe-summary adoption remains a separate step.' if not flgmm.get('selected_recipe') else
    'Independent and root record-only reviews adopted the unchanged frozen-score choice Tg20/L2/lr0.001, with32 condition scores and32 candidate mean scalars exact. The score margin over second place is0.0000497792; all eight candidates, the different ACC champion(Tg20/L3/lr0.001) and six Pareto candidates remain retained. This is exposed seed91001 validation selection, not stable superiority or a final endpoint.')
diagnostic=state.get('native_mismatch_GPU_diagnostic_20261009')
fl_bound_note=('FLGMM coverage metadata has separately passed actual server binding and root144-member SHA/scope checks:96 new70-round jobs plus four explicit original references, five new3-round gates and two same-horizon original references. No canary or full-coverage training has started. The local Python3.10 extraction-API failure is retained; a new safe per-file local extraction passed without repeating any remote bind or checkpoint inference.' if state.get('flgmm_fullcoverage_v2_20261009') else '')
if state.get('flgmm_fullcoverage_v2_20261009',{}).get('canary_runner_started'):
    fl_bound_note=fl_bound_note.replace('No canary or full-coverage training has started.',
        'The exact seven-canary queue has actually started after source/data, protected-main growth and real Linux resource checks. No canary has yet been adopted; the96 new70-round queue has not started. Three rounds cannot exercise the selected Tg20 transition or prove70-round equivalence.')
if state.get('flgmm_fullcoverage_v2_20261009',{}).get('formal100_started'):
    fl_bound_note=fl_bound_note.replace('No canary has yet been adopted; the96 new70-round queue has not started.',
        'All seven canary runs passed original strict/tensor/RNG checks and315 archive-member offserver/root checks. The frozen96-new+4-reused validation queue has actually started with one worker on eachGPU, both observed at round1; zero new70-round results are adopted at this startup boundary.')
if main.get('C_after12_valid_replay'):
    C_view_note+=(' An independent eight-checkpoint C/IID/F Flip terminal replay actually started onCPU112..119 with eight threads, nice10/idleI/O and hiddenCUDA. Source, actual original artifacts and resource guards passed; the prior112 and Full are not replayed. New offserver adoption and a complete F Flip table remain pending.')
    if main['C_after12_valid_replay']['offserver_new_accepted']==8:
        C_view_note=C_view_note.replace('12 accepted C terminal checkpoints.','20 accepted C terminal checkpoints.').replace('The two F Flip pairs remain excluded from means; other C scenes are incomplete.',
            'The previous two F Flip pairs were excluded from that Benign table; all ten F Flip checkpoint replays are now accepted, with its separate complete paired table still pending. Other C scenes remain incomplete.').replace(
            'New offserver adoption and a complete F Flip table remain pending.',
            'The exact eight increment completed normally and passed89 archive-member checks,72 reconstructed metrics,192 confusion counts,24 rules and root adoption. Native discrepancies are all zero; only the F Flip paired-table review remains pending.')
if main.get('C_three_view_two_scene_table'):
    C_view_note+=(' The actual20-pair,40-record Benign/F Flip table now passes root recomputation of324 mean/sampleSD scalars and360 count metrics, with162 display cells and the old24 records/162 Benign scalars exact. All three views and fixed10/9/6-seed panels are retained; Full replay2CPU/18GPU versus C20CPU remains explicit. Other eight C scenes are incomplete. ')
    C_view_note=C_view_note.replace('with its separate complete paired table still pending.','with its separate complete paired table now independently adopted.').replace('only the F Flip paired-table review remains pending.','the F Flip paired-table review is now independently adopted.')
if main.get('C_after20_valid_replay',{}).get('offserver_new_accepted')==5:
    C_view_note+=(' Another exact five IID FedSA C terminal checkpoints (91001/03/05/06/08) are now strictly accepted and off-server adopted, closing U100+C25=125. All68 archive members,45 saved-array metrics,120 confusion counts and15 rules passed, with zero native discrepancy. The prior120 and Full were not reinferred; FedSA has only5/10 pairs and is excluded from complete-scene means. ')
    C_view_note=C_view_note.replace('20 accepted C terminal checkpoints.',f'{C_views} accepted C terminal checkpoints. The complete Benign/F Flip comparison uses20 of these checkpoints; remaining individual records are coverage only.')
if main.get('C_after25_valid_replay'):
    c3=main['C_after25_valid_replay']
    C_view_note+=(f' The exact next three IID FedSA checkpoints (91002/04/07) have status {c3["status"]}, with{c3["offserver_new_accepted"]} new off-server acceptances; the prior125 and Full are excluded from inference. ')
    if c3['offserver_new_accepted']==3:
        C_view_note+=('All54 archive members,27 saved-array metrics,72 confusion counts and9 rules passed, with zero native discrepancy. This closes U100+C28=128; FedSA is still8/10 and supplies no complete-scene mean or sample SD. ')
if main.get('C_after28_valid_replay'):
    c28=main['C_after28_valid_replay']
    C_view_note+=(f' The next exact eight terminal checkpoints (two IID FedSA and six IID S-DFA) have actual source-bound Linux startup, status {c28["status"]}, with {c28["offserver_new_accepted"]} new off-server acceptances. The prior128 and Full are excluded from inference. ')
    if c28['offserver_new_accepted']==8:
        C_view_note=('36 accepted C terminal checkpoints. The new exact8 passed89 archive-member checks,72 saved-array metric checks,192 confusion counts and24 prediction rules; native discrepancies are zero. The prior128 and Full were not reinferred. Benign, F Flip and FedSA each have ten matched C checkpoints; S-DFA has only six and is excluded from complete-scene means. The previously accepted two-scene table remains valid; the three-scene table requires separate independent statistical adoption. All views remain validation-only, with calibration, device/environment and selection/test-exposure limitations retained. ')
if main.get('C_three_view_three_scene_table'):
    C_view_note=C_view_note.replace('The previously accepted two-scene table remains valid; the three-scene table requires separate independent statistical adoption.',
        'The three-scene table is now independently adopted:30 matched pairs/60 records,486 mean/sampleSD scalars,243 cells and540 metrics reconstructed from group counts. The old40 records,324 scalars and162 display cells remain exact. Full replay2CPU/28GPU versus C30CPU is explicit; seven other C scenes and the full mechanism grid remain incomplete.')
if main.get('C_after36_valid_replay',{}).get('offserver_new_accepted')==4:
    C_view_note=('40 C terminal checkpoints now pass strict off-server/root adoption. The exact four new IID S-DFA checkpoints passed61 archive-member checks,36 saved-array metrics,96 confusion counts and12 prediction rules; native discrepancies are zero. Prior136 and Full were not reinferred. Benign/F Flip/FedSA/S-DFA each have ten paired checkpoints; the four-scene table requires separate statistical adoption. The remaining six C scenes and six other controls are incomplete. Validation, mixed-device/environment and selection/test-exposure limitations remain. ')
if main.get('C_three_view_four_scene_table'):
    C_view_note=C_view_note.replace('the four-scene table requires separate statistical adoption.',
        'the four-scene table is independently adopted:40 pairs/80 records,648 mean/sampleSD scalars,324 cells and720 count-derived metrics. Old60 records/486 statistics/243 cells and the original S-DFA six records remain exact. Full replay3CPU/37GPU versus C40CPU is explicit; all fixed10/9/6 panels and negative results are retained.')
if main.get('C_after47_valid_replay',{}).get('offserver_new_accepted')==3:
    C_view_note=('50 C terminal checkpoints now pass strict off-server/root adoption; the latest exact three pass54 archive members,27 saved-array metrics,72 confusion counts and9 prediction rules, with native discrepancies zero. Prior147 and Full were not reinferred. All five IID scenes each have ten paired checkpoints; the five non-IID C scenes and six other controls remain incomplete. The complete five-scene table needs its own independent adoption. Validation, mixed-device/environment and selection/test-exposure limitations remain. ')
if main.get('C_three_view_five_scene_table'):
    C_view_note=C_view_note.replace('The complete five-scene table needs its own independent adoption.',
        'The five-scene table is independently adopted:50 pairs/100 records,810 mean/sampleSD scalars,405 cells,900 count-derived metrics and162 seed-first cross-scene scalars. The old80 records/648 statistics/324 cells remain exact. All10/9/6 panels and negative results are retained; this is descriptive validation evidence, not a component-necessity or final-test conclusion.')
if main.get('C_after50_valid_replay'):
    C56=main['C_after50_valid_replay']
    C_view_note+=(' The six non-IID Benign terminal checkpoint replays are strictly accepted off server:75 archive members,54 metrics,144 confusion counts and18 prediction rules passed, with native discrepancies zero. C is cumulative56, but this scene is only6/10 and excluded from complete-scene means. ' if C56['offserver_new_accepted']==6 else
        ' Six non-IID Benign terminal checkpoint replays have actually started; strict off-server acceptance remains0. The partial scene is excluded from complete-scene means. ')
if main.get('C_after56_valid_replay',{}).get('offserver_new_accepted')==4:
    C_view_note=('60 C terminal checkpoints now pass strict off-server/root adoption, closing U100+C60=160. The latest exact four non-IID Benign checkpoints passed61 archive members/60 content members,36 metrics,96 confusion counts and12 prediction rules; native discrepancies are zero. Original156 and Full were not reinferred. The five IID scenes and non-IID Benign now each have ten pairs; the six-scene table requires separate independent adoption. Four other non-IID C scenes and six other controls remain incomplete. Validation, mixed-device/environment and historical selection/test-exposure limits remain. ')
if main.get('C_three_view_six_scene_table'):
    C_view_note=C_view_note.replace('the six-scene table requires separate independent adoption.',
        'the six-scene table is independently adopted:60 pairs/120 records,972 mean/sampleSD scalars,486 display cells,1080 count-derived metrics and2880 confusion counts. Old100 records/810 scalars/405 cells and the162 IID seed-first aggregate scalars remain byte-exact. All raw/native/shared10/9/6 panels and negative results are retained. The aggregate covers only the original five IID scenes; no imbalanced six-scene overall mean was calculated. The C60 table is a separate entry; the complete author-review response still incorporates C50, not C60. No primary endpoint, necessity, significance or final-test claim is made.')
if main.get('C_three_view_six_scene_table',{}).get('incorporated_into_full_rebuttal'):
    C_view_note=C_view_note.replace('The C60 table is a separate entry; the complete author-review response still incorporates C50, not C60.', 'The complete author-review response now incorporates C60; the prior C50 documents remain preserved and the submitted manuscript has not been changed.')
if main.get('C_three_view_seven_scene_table'):
    C_view_note=('C70 is independently adopted as a separate seven-scene table:70 pairs/140 records, five IID scenes and non-IID Benign/F Flip, each with ten shared seeds. '
        '1134 mean/sampleSD scalars,567 display cells,1260 count-derived metrics,3360 confusion counts and630 paired metric checks pass. Old120 records/972 statistics/486 cells and162 IID seed-first aggregate scalars remain exact. '
        'All raw/native/shared10/9/6 panels and negative results are retained; no imbalanced seven-scene aggregate is computed. In the added non-IID F Flip scene, deletion of C raises native/shared ACC by0.943pp, lowers AEOD by0.00308 and raises ASPD by0.01177. '
        'Full5CPU/65GPU versus C70CPU, training environments and selection/test history are disclosed. The complete author-review response still incorporates C60; C70 is not yet integrated into that preserved text. '
        'Three non-IID C scenes, six other controls and final-test evaluation remain incomplete; no selected endpoint, necessity, causality or significance is claimed. ')
if main.get('C_after70_valid_replay',{}).get('offserver_new_accepted')==10:
    C_view_note+=(' The subsequent exact ten non-IID FedSA terminal checkpoints have also passed strict acceptance,103 archive-member checks,90 reconstructed metrics,240 confusion counts and30 prediction rules, with zero native discrepancy. '
        'The old170 and Full were not reinferred; cumulative accepted validation replays are U100+C80=180. The eighth-scene table still requires separate statistical adoption; only non-IID S-DFA/Sp-DFA and the other six variants remain unclosed for C checkpoint coverage. ')
if main.get('C_three_view_eight_scene_table'):
    C_view_note=('C80 is independently adopted as a separate eight-scene table:80 pairs/160 records, five IID scenes and non-IID Benign/F Flip/FedSA, ten shared seeds each. '
        '1296 mean/sampleSD scalars,648 display cells,1440 count-derived metrics,3840 confusion counts,720 paired metric checks and162 original IID aggregate scalars pass. Old140 records/1134 statistics/567 cells and the IID aggregate remain exact; no imbalanced eight-scene aggregate is computed. '
        'In the added FedSA scene, deletion raises native/shared ACC by0.410pp and AEOD by0.00190, with ASPD difference about-0.0000093; ASPD changes direction in the9/6-seed panels. All raw/native/shared10/9/6 panels and negative results remain. '
        'Full5CPU/75GPU versus C80CPU, training environments and selection/test history are disclosed. The exact new10 increment has103 verified archive members and zero native discrepancy; prior170/Full were not reinferred. '
        'The complete author-review response still incorporates C60; C70/C80 tables are separate. Non-IID S-DFA/Sp-DFA, six other controls and final-test evaluation remain incomplete. No necessity, causality, selected endpoint or significance is claimed. ')
if state.get('latest_rebuttal_addendum') and state['latest_rebuttal_addendum']['complete_C_scenes']>state['latest_rebuttal_draft'].get('complete_C_scenes',0):
    addendum=state['latest_rebuttal_addendum']
    C_view_note+=(f" A separate C{10*addendum['complete_C_scenes']} English author-review addendum for R3.2/R3.7 is source-checked at{addendum['scalar_pointer_checks']} scalar pointers,{addendum['display_cells_checked']} display values,{addendum['scope_fact_checks']} scope facts and{addendum['links_checked']} links; the frozen24-comment response and submitted manuscript remain unchanged. Entry: "+addendum['entry']+'. ')
if main.get('C_three_view_full100_table'):
    C_view_note=('C100 complete paired ten-scene tables independently adopted:100 pairs/200 records across IID/non-IID and all five scenarios, with fixed10/9/6 panels. '
        '1620 mean/sampleSD scalars,810 display cells,1800 count-derived metrics,4800 confusion counts and900 paired metric checks pass; old160/1296/648 and162 IID aggregate scalars remain exact. '
        'Separate nonIID5 and balanced10 aggregates each have162 checked scalars, averaging scenes within each seed first. Native/shared deletion contrasts show ACC+0.532038pp/AEOD−0.00111223/ASPD−0.00083292 under nonIID S-DFA and+0.680525pp/−0.000584879/+0.00478524 under Sp-DFA. '
        'All subsets and unfavorable effects remain. Full replay5CPU95GPU/training98cu1282cu130 versus C100CPU/cu128, selection history and previous test exposure are disclosed. Six other controls remain unfinished, with no necessity, causality, significance or universal-win claim. ')
A_view_note=''
if main.get('three_view_counts_by_variant',{}).get('minus_A')==12:
    A_view_note=(' An additional12 minus_A terminal replays have original strict/offserver checks and root adoption against the native212 restore chain:110 archive members,108 metrics,288 confusion counts,36 prediction rules and24 original model/result members. '
        'IID Benign has ten matched seeds; IID F Flip has only two and is excluded from means. Its separate A table still requires source/statistical adoption. ')
if main.get('A_three_view_single_scene_table'):
    A_view_note=A_view_note.replace('Its separate A table still requires source/statistical adoption.',
        'The IID Benign three-view table is independently adopted:162 statistics/81 cells/216 count-derived metrics, all10/9/6 panels retained. Native/shared paired deletion is ACC−0.429859pp, AEOD+0.002313 and ASPD−0.003142, with subset direction changes. Full2CPU8GPU versus A10CPU and commoncu128 training are disclosed; no necessity, causal or significance claim. Entry: '+main['A_three_view_single_scene_table']['table_path']+'.')
recovery=state.get('baseline_valid_recovery_prepared_20261009')
if main.get('three_view_counts_by_variant',{}).get('minus_A')==20:
    A_view_note=(' Eight additional IID F Flip minus_A replays pass original strict/offserver and root native218/220 restore-chain adoption:74 archive members,72 metrics,192 counts,24 rules and16 native model/result member rehashes. '
        'U100+C100+A20=220 saved three-view models are adopted; A IID Benign and F Flip each have ten matched seeds. The original Benign table remains adopted; the new two-scene statistics await separate root arithmetic review. Eight other A scenes and other controls remain unfinished. ')
if main.get('A_three_view_two_scene_table'):
    A_view_note=A_view_note.replace('The original Benign table remains adopted; the new two-scene statistics await separate root arithmetic review.',
        'The new two-scene three-view table is independently adopted:324 mean/sampleSD scalars,162 display cells,360 count-derived metrics and960 integer-count checks. Old24 records and Benign162 scalars/81 display values remain exact. IID F Flip native/shared deletion is approximately ACC+0.001pp, AEOD+0.00001 and ASPD−0.00088; all10/9/6 panels and direction changes remain, without necessity, causal or significance claims. Entry: '+main['A_three_view_two_scene_table']['table_path']+'.')
if main.get('three_view_counts_by_variant',{}).get('minus_A')==28:
    A_view_note=' An additional28 minus_A replays are root-adopted, including the retained20 paired IID Benign/F Flip checkpoints and their independent two-scene table. Eight further IID FedSA records pass exact native228 recovery-chain joining,74 archive members,72 metrics,192 counts,24 rules and16 original model/result hashes. FedSA is only8/10 seeds, so no extra scene mean or table is produced; prior220 remain exact. '
if main.get('three_view_counts_by_variant',{}).get('minus_A')==36:
    A_view_note=' Cumulative36 minus_A terminal replays are root-adopted, retaining the20 paired IID Benign/F Flip checkpoints and their independently adopted two-scene table. The latest exact8 pass original strict/offserver74-member,72-metric,192-count,24-rule checks and native236 restore-chain joining with16 original model/result hashes; the prior228 index remains exact. IID FedSA now has ten paired checkpoints at record level, whereas S-DFA has only6/10 and is excluded from complete-scene means. No additional A scenario table has been adopted; A100 and remaining controls are incomplete. '
if main.get('A_three_view_four_scene_table'):
    A_view_note=' Cumulative40 minus_A saved terminal replays are root-adopted against the native243 recovery chain, bringing three-view controls to240. Four complete IID scenes (Benign/F Flip/FedSA/S-DFA), each ten paired model seeds, are independently adopted with648 mean/sampleSD scalars,324 display cells,720 count-derived metrics and1920 integer-count checks; old A20 forty JSON object bytes/order,324 scalars and162 cells remain exact. All native/raw/shared and10/9/6 panels remain. Native/shared deletion differences under S-DFA are approximately ACC−0.361pp/AEOD+0.00542/ASPD−0.00804 and under FedSA+0.134pp/−0.00065/+0.00141, showing tradeoffs. Full3CPU37GPU versus A40CPU and commoncu128 training are disclosed. Six remaining A scenes and other controls are incomplete; no necessity, significance, causal-isolation or final-test claim. Entry: '+main['A_three_view_four_scene_table']['table_path']+'. '
if A80:
    A_view_note=' '+'Current A80 table is root-adopted: five IID scenes plus non-IID Benign/F Flip/FedSA, eight scenes with ten paired seeds each,80 pairs/160 records,1458 mean/sampleSD scalars and729 display cells; all10/9/6 panels remain. Old A60 preserves120 object bytes/order,972 scene statistics/486 cells and IID seed-first aggregate bytes. Raw FedSA deletion improves all three means; native/shared deletion slightly raises ACC while worsening both gaps. All negative outcomes and view-dependent tradeoffs remain, without necessity or significance claims. Full5CPU75GPU/79cu128+1cu130 versus A80CPU/cu128, selection history and historical test exposure are disclosed. Non-IID S-DFA/Sp-DFA, remaining controls and final evaluation remain unfinished. '+'Entry: '+A80['table_path']+'. '
recovery_paragraph=('The exact 900−424 complement is sealed as a prepared-only proposal: 465 unexecuted models, '
    '10 preserved CPU partial results and one separately verified GPU diagnostic. '
    'The new GPU worker and strict acceptance entry are being implemented separately; '
    'the proposal has dispatched no inference and registered no new acceptance. ' if recovery else '')
gpu_recovery = state.get('baseline_valid_GPU_recovery_20261009')
if gpu_recovery:
    recovery_paragraph = ('The independently reviewed first new GPU replay has passed the original strict checks, '
        '73 archive-member SHA checks and off-server saved-array reconstruction of nine metrics, '
        '24 confusion counts and three prediction rules. Native discrepancy is exactly zero; the old424 collector is unchanged. '
        'The cumulative425 collector explicitly retains CPU/GPU source provenance. The remaining464 are not dispatched; '
        'the10 preserved CPU partial results and one original GPU diagnostic are not registered. '
        'Mixed-device raw/shared results are not a uniform-device final comparison. ')
    if gpu_recovery.get('CPU_partial10_registered') and gpu_recovery.get('diagnostic1_registered'):
        recovery_paragraph = ('The first new GPU replay passed original strict acceptance and73 off-server archive-member checks, '
            'with exact native metrics and saved-array reconstruction. A separate explicit review then revalidated and '
            'registered the original10 CPU partial records and one saved successful GPU diagnostic without CNN inference. '
            'The new436 collector retains CPU434/GPU2 provenance, unchanged historical424/425 collectors and the invalid '
            'original CPU failure. The remaining464 are not dispatched. Mixed-device raw/shared results are not a '
            'uniform-device final comparison. ')
    if gpu_recovery.get('remaining464_dispatched'):
        recovery_paragraph = recovery_paragraph.replace('The remaining464 are not dispatched. ',
            'The exact remaining464 GPU queue is now running in43 bounded chunks with one GPU worker. '
            'CPU106 coordination, CPU105 workers and nice10 were independently observed. '
            'Worker zero exits and remote closures do not increment436 before off-server registration. ')
    if gpu_recovery.get('GPU_remaining464_offserver_accepted'):
        recovery_paragraph = (f'The original436 source chain and {gpu_recovery["GPU_remaining464_offserver_accepted"]} '
            f'new GPU records were strictly accepted and verified off server, closing the legacy chain at460/900 '
            '(CPU434/GPU26). The exact original464 scope is divided into43 bounded chunks; '
            'only independently verified chunks enter separate cumulative collectors. Original424/425/436 collectors '
            'and the invalid CPU output are unchanged. Saved arrays reconstruct all three rules, nine metrics and24 '
            'confusion counts; root refitting is checked by the bound original remote strict tool, not repeated locally. '
            'Mixed CPU/GPU results are not a uniform-device final comparison. ')
        if not gpu_recovery.get('queue_running'):
            recovery_paragraph += ('The queue stopped in chunk002 during a worker resource preflight before CNN inference, '
                'with Protected main800 health failed. The main mechanism queue is currently healthy; the instantaneous '
                'guard inputs were not preserved, so a transient handover is a hypothesis, not a proved unique cause. '
                'The failure archive is retained and the original service has not been restarted. Two completed partial '
                'records remain unregistered; any repair needs a separate reviewed version and exact missing-ID scope. ')
        if gpu_recovery.get('completed_partial2_explicitly_registered'):
            recovery_paragraph = recovery_paragraph.replace(
                'Two completed partial records remain unregistered; ',
                'The two saved partial records subsequently passed original partial strict acceptance and independent saved-array checks and were explicitly imported without new CNN inference; ')
v2 = state.get('baseline_valid_GPU_remaining440_v2_20261009')
if v2:
    recovery_paragraph += (f'The exact remaining440 complement was subsequently deployed in a new V2 namespace with unchanged scientific inference/acceptance and native1e-12. '
        f'Actual Linux processes, CPU106 coordination, CPU105 single-GPU workers, nice10/idle I/O and resource receipts were verified at {v2["measured_utc"]}; '
        f'{v2["worker_exit_complete_observed"]} worker zero exits and {v2["remote_closed_n"]} remote closures were observed in that snapshot. '
        f'Separately, {v2["new_offserver_accepted"]} V2 records have passed original strict acceptance and independent off-server checks, for the current cumulative{replayed}/900. '
        'The first supervisor attempt exited before contract/CNN because its review filename differed from the installed file; original configuration/log bytes were preserved and only the external config path was corrected before starting the fresh queue. '
        'The failed original464 queue and historical collectors remain unchanged. ')
diagnostic_paragraph=('A separately approved one-model GPU diagnostic reproduces the original three native metrics exactly. '
    'Its saved CPU/GPU arrays differ on one native/raw prediction (image172599); shared-calibration predictions match. '
    'The unchanged source/data/checkpoint and all saved metrics, counts and rules were independently verified. '
    'This is one diagnostic execution, not a cohort acceptance or restoration of the failed CPU queue. '
    'Two engineering failures are preserved; the comparison was completed offline without another CNN execution. '
    'Historical GPU per-image outputs are unavailable, so unique historical causality remains unproved. '
    if diagnostic else '')
if baseline.get('GPU_diagnostic_later_explicit_versioned_import'):
    diagnostic_paragraph = diagnostic_paragraph.replace(
        'This is one diagnostic execution, not a cohort acceptance or restoration of the failed CPU queue. ',
        'Its original diagnostic receipt remains unchanged; a later explicit saved-array import is included in the current derived collector. The failed CPU queue remains stopped. ')
screen=state.get('hybrid_screen32_20261009')
failure=state['final_evaluator_runtime_20261009'].get('failed_model_id')
failure_paragraph=("The CPU replay service has fail-stopped on FairGuard/IID/F Flip/seed91009: native metrics differ from the original record despite matching model/config/data identities and unchanged tensors. The original1e-12 tolerance is preserved and the65-member failure archive is verified off server. Chunk036's10 strict partial results were subsequently imported after explicit review into the derived436 source chain; the failed CPU result remains invalid. No original metric, model, threshold or selection rule has changed and the failed CPU service remains stopped. Previously accepted records remain valid. " if failure else "")
paragraph=(f"The original32-item Hybrid validation search has actually started under `{screen['service']}`; its independent source/startup archive and root verification bind the unchanged eight recipes, four conditions, seed91001 and70 rounds. It uses one GPU0 worker, CPU104, one compute thread and nice10. No100-job multi-seed confirmation, test or automatic retry is authorized. "
    if screen else "The unchanged original32-item Hybrid validation search is root-approved for a separate frozen execution copy; exact source/scope approval is complete, while actual dispatch and source-bound startup acceptance remain separate requirements. ")
if screen and screen.get('offserver_accepted70round_jobs'):
    paragraph += (f"The first{screen['offserver_accepted70round_jobs']}/32 terminal jobs have original server strict acceptance and linked off-server backups, including53 members in the original four-job increment and35 in the next two-job increment, with an independent CPU record-layer replay. "
        "Only three current-host runtime metadata queries are bound to the original server receipt; all scientific checks and null diagnostic policies are unchanged. Local torch2.8 CPU is disclosed separately from server torch2.11 cu128 and no local CNN inference or CUDA context was used. "
        "The still-running shared service log is saved as a prefix, not a closed per-job log; terminal artifacts were checked before and after backup. No final recipe is selected from this partial snapshot. ")
if screen.get('status')=='ROOT_HYBRID32_SUMMARY_ADOPTED':
    paragraph='The original32-item Hybrid search has exited normally with zero workers. All32 terminals have original strict acceptance and off-server verification; an independent review checks the exact27+5 chain and frozen score/rank, with maximum independent difference1.11e-16. The frozen selection is CosineFairness_lam20.0_tau0.1_lr0.001, also the accuracy champion and sole three-metric Pareto candidate. All eight candidates remain. This is n=1 validation search, without sampleSD or significance. The100-cell coverage has not started; the actual-summary status interface requires a record-layer repair and seven new real-image short-run gates remain prerequisites. The immutable SUMMARY32 and its score/rank are unchanged.'
paragraph=paragraph.rstrip()
next37_status=main.get('next37_valid_replay',{})
next37_note='Dispatch and new off-server acceptance require their own actual receipts.'
if next37_status.get('startup_root_proof_sha256'):
    next37_note=(f"The exact37 scope has actually started under guardfed_celeba_mechanism_valid_next37, observed at {next37_status['measured_utc']}, "
        "after fresh Linux full source/data/terminal hash, CPU owner and quota checks. One eight-thread CPU worker is restricted to112–119, nice10 and idle I/O with CUDA hidden. "
        "Source-bound startup and actual approval receipts are saved; no new off-server scientific acceptance is inferred from this startup. The old23 replays and Full weights are excluded.")
if next37_status.get('offserver_new_accepted'):
    next37_note += (f" Separately, {next37_status['offserver_new_accepted']} new terminals have passed original strict checks, archive-member verification, independent saved-array reconstruction and root adoption; "
        f"{next37_status['offserver_remaining']} of this37-item scope remain unaccepted off server. All native discrepancies are zero. These acceptances bind later backup receipts rather than the startup snapshot.")
if next37_status.get('status') in ('COMPLETE_STRICT_OFFSERVER_NO_FULL_JOIN','COMPLETE_STRICT_OFFSERVER_PAIRED_SIX_SCENES'):
    next37_note += (f" The later terminal observation at{next37_status['latest_progress_utc']} verified service exit,37 completed IDs,zero residual workers and no failure. "
        "Both disjoint backups are now strict/off-server accepted; the exact37 scope is complete. Later native terminals remain outside this scope, and the subsequent paired Full join has its own actual root proof.")
next11_note = ('Execution has not been inferred from preparation.')
if main.get('next11_valid_replay',{}).get('execution_started'):
    next11_note = ('The separate execution passed39 refusals and actually started under guardfed_celeba_mechanism_valid_next11 after109 Linux source/data/terminal-member checks and actual quota/CPU-owner checks; one eight-thread CPU worker on112–119, nice10/idleI/O and CUDA hidden was independently observed. Old37 and next8 were not restarted.')
if main.get('next11_valid_replay',{}).get('offserver_new_accepted') == 11:
    next11_note += (' The later terminal snapshot verifies normalEXITED,11 complete,zero residual workers and no failure. All11 have original strict acceptance,120 verified archive members and independent saved-array reconstruction of99 metrics,264 confusion counts and33 rules, followed by root source/checkpoint/terminal adoption. Native discrepancy is exactly zero. This closes71 minus_U three-view models; the prior60 are unchanged. The fixed six-scene table remains separate from a future explicitly accepted seven-scene join.')
    if paired_table and paired_table['complete_scenes'] == 7:
        next11_note = next11_note.replace('a future explicitly accepted seven-scene join','the subsequent explicitly root-accepted seven-scene join')
after71_note=''
if main.get('after71_valid_replay'):
    after71=main['after71_valid_replay']
    after71_note=(f'The separate after71 exact11 replay status is {after71["status"]}, with{after71["offserver_new_accepted"]} new offserver acceptances. '
        'It excludes all previously closed71 and Full inference, covering only non-IID FedSA91002–91009 and S-DFA91001–91003. '
        'Remote completion is not acceptance. Existing seven-scene mean/SD tables remain a separate sealed evidence snapshot; these IDs cannot create a complete eighth scene because FedSA91010 is outside the frozen replay scope. '
        'At that historical replay scope, the other seven variants had source-only preparation. Their later native terminals require separate acceptance; they do not enter this eleven-model replay.')
after82_failure_note=''
if main.get('after82_failed_attempt'):
    failure82=main['after82_failed_attempt']
    after82_failure_note=('The exact10 after82 replay attempt stopped before runtime dependency binding and CNN because its approval guard retained the old11-ID cardinality. '
        'Actual service exit, zero workers/completions and an empty output tree are preserved with source/log/approval evidence; zero new scientific acceptances are inferred. '
        'The original82 accepted records and main800 training are unaffected. A separate engineering version requires a positive valid10 approval gate and fresh namespace; the failed service is not restarted. '
        f'Root failure review: {failure82["root_failure_review_path"]}.')
if main.get('after82_v2_valid_replay'):
    replay82=main['after82_v2_valid_replay']
    after82_failure_note+=(f' The separately reviewed V2 service has actual startup evidence: {replay82["status"]}, '
        f'{replay82["offserver_new_accepted"]} new off-server acceptances; one8-thread CPU worker on112–119, nice10/idleI/O and CUDA hidden. '
        'Positive valid10 approvals and invalid-approval refusals pass; only cardinality and namespace/pins changed, and the failed original service is retained.')
after92_note=''
if main.get('after92_valid_replay'):
    replay92=main['after92_valid_replay']
    after92_note=(f'The final exact8 minus_U replay has actual source-bound Linux startup: {replay92["status"]}, '
        f'{replay92["offserver_new_accepted"]} new off-server acceptances. It covers only non-IID Sp-DFA91003–91010, excluding the prior92, Full inference and the separately accepted C4. '
        'One8-thread CPU worker on112–119, nice10/idleIO and hidden CUDA is actually observed. This startup does not establish100 accepted three-view models or a complete ten-scene three-view table. ')
    if replay92['offserver_new_accepted'] == 8:
        after92_note = (f'The final exact8 replay completed normally with no residual worker or failure; all8 are strictly accepted and backed up off server, closing100 minus_U three-view models. '
            f'{replay92["archive_members_verified"]} archive members,72 metrics,192 counts and24 prediction rules passed checks, with zero native discrepancy. '
            'The prior92 and Full were not reinferred; C4 is excluded. The complete ten-scene three-view table has its own independently accepted identity/statistics join, cited above.')
fl_multiseed_status=('The100-job multi-seed coverage has not started.' if not state.get('flgmm_fullcoverage_v2_20261009',{}).get('formal100_started') else
    'The frozen100-cell multi-seed validation coverage is running as96 new jobs plus four explicitly reused results; new70-round acceptances are tracked separately from the completed search.')
if state.get('flgmm_fullcoverage_v2_20261009',{}).get('new_accepted'):
    fl_multiseed_status+=(f' New70-round checkpoints have passed original strict/offserver checks and independent root archive/record adoption; cumulative new coverage acceptance is{state["flgmm_fullcoverage_v2_20261009"]["new_accepted"]}/96, with four prior results reused separately. No complete10-seed scene summary has yet been adopted; these acceptances are validation evidence, not final-test results.')
reply_progress_note=('Its24 original comments,37 numeric pointers and37 links passed review;21 necessary paragraph updates leave206 prior paragraphs exact.' if not state['latest_rebuttal_draft'].get('complete_C_scenes') else
    'The latest copy also includes the separately adopted C20 two-scene comparison. All24 original comments and both previous full documents recover exactly by reversing ten edits;22 new C20 scalar pointers,12 scope/environment facts and41 links passed root checks. The prior37 numeric references are preserved.')
if state['latest_rebuttal_draft'].get('complete_C_scenes')==5:
    reply_progress_note='The complete copy includes all five IID C scenes. Its24 original comments remain verbatim and both previous full documents recover byte-exactly by reversing11 edits;38 C scalar pointers,19 scope/environment facts,24 direction checks and44 links passed root review. All unaffected text, prior numeric references, counterexamples and pending items remain intact.'
if state['latest_rebuttal_draft'].get('complete_C_scenes')==6:
    reply_progress_note='The complete author-review copy now includes the five IID scenes and non-IID Benign C evidence. All24 original comments remain verbatim;10 reversible edits recover both C50 documents exactly. Root checks cover36 scalar pointers,18 new mean/SD cells,27 directions and50 links. Prior numbers and negative outcomes remain intact. Four non-IID C attack scenes, six other controls and P1–P6 remain incomplete; the submitted manuscript and final test remain pending.'
if state['latest_rebuttal_draft'].get('complete_C_scenes')==10:
    reply_progress_note='The complete C100 author-review copy incorporates allIID/non-IID/five-scenario U/C evidence. All24 original comments remain verbatim;14 reversible edits restore the previous C60 documents exactly. Root reran the original writing checker for90 scalar pointers,45 cells,54 directions and60 links, with all prior numeric displays preserved. Six other image controls and P1–P6 remain unfinished, including final evaluation and submitted-manuscript integration.'
if state['latest_rebuttal_draft'].get('A_complete_scenes')==2:
    reply_progress_note='The latest complete author-review copy integrates U/C100, paired A20 in two IID scenes, LoGoFair100 and the ten-method native1000 table. Root and independent semantic reviews retain all24 original comments and old numbers;25 reversible edits restore both C100 documents exactly. The original checker passed23 added mean/SD cells,59 source facts,54 directions,six tradeoff means and82 links. Seven method cohorts,six image controls,P1-P6,final evaluation and submitted-manuscript integration remain incomplete.'
if state['latest_rebuttal_draft'].get('A_complete_scenes')==4:
    reply_progress_note='The latest complete author-review copy integrates U/C100, paired A40 in four IID scenes, LoGoFair100 and the ten-method native1000 table. Root and independent semantic reviews preserve all24 original comments and old numbers; eight reversible spans restore both prior full documents exactly. Checks cover59 mean/SD cells,118 scalar pointers,69 source facts,108 directions and88 links. A40 tradeoffs and all10/9/6 panels remain. Seven method cohorts,six image controls,P1-P6,final evaluation and submitted-manuscript integration remain incomplete.'
current=f'''## Current accepted increment — measured {state['last_health_check']['checked_utc']}

The main mechanism queue has{main['queue_completed_observed']} terminal jobs observed, with{accepted} independently accepted and backed up off server in{len(increments)} linked increments;100 Full controls remain explicit reuse. The latest{len(latest_increment['new_ids'])}-ID increment has{latest_increment['members_verified']} verified members, archive `{latest_increment['archive_sha256']}`. The queue continues with eight workers and no observed failures. These counts do not establish all800 controls or the whole rebuttal.

The existing nine-method terminal-model validation replay has {replayed} distinct accepted and off-server-verified models, with {900-replayed} still missing. Each closed increment has original strict acceptance and independent raw/native/shared prediction-array checks; all preserve native metrics exactly. The cumulative collector is `{baseline['accepted_collection_path']}`. {failure_paragraph}{diagnostic_paragraph}{recovery_paragraph}This is validation replay, not final-test evaluation or new model training.

The approved exact15 mechanism replay increment is complete and its service is EXITED: three disjoint backups contain149 content members plus three inventories, with135 metrics,360 confusion counts and45 prediction rules independently reconstructed. Together with the original eight, the historical increment closed23 actual minus_U terminal checkpoints. Subsequent explicitly adopted increments bring the current total to{mechanism_views} strict, off-server raw/native/shared validation replays: complete U100 plus{C_view_note}{A_view_note}Native discrepancies in these accepted increments are zero. {paired_note}This does not establish the complete mechanism comparison.

The accepted native mechanism data currently support{complete_native_scenes} complete Full–minus_U scenes, with ten paired seeds per scene and consistent additional nine-seed and six-seed panels. In non-IID Benign, Full hasACC88.594±1.168%, AEOD0.00696±0.00423 andASPD0.06511±0.00902; minus_U has87.290±1.870%,0.01300±0.00732 and0.05140±0.01598. Full therefore has higher mean accuracy and lower meanAEOD but higher meanASPD in this scene. These native end-to-end results retain calibration effects and do not establish an isolated aggregation cause or that every score term is necessary. The next37 replay scope is exactly the historical accepted60 minus the already closed23; source review and36 scientific no-CNN rejection checks plus32 execution rejection checks have passed. {next37_note}

FLGMM has{flgmm['offserver_accepted70round_jobs']}/32 full70-round validation-search jobs strictly accepted and backed up off server using the frozen original acceptor. {'The original search continues; incomplete candidates do not supply a final selection.' if flgmm['offserver_accepted70round_jobs']<32 else 'The original service has exited normally with zero workers; all eight candidates are now complete. The final-six archive and every member have been checked off server.'} {fl_selection_note} {fl_multiseed_status} The n=1 search cannot establish across-seed SD or significance. The exact earliest-two legacy receipt lacks the later host flag; a separately pinned reader preserves its actual21-source/record checks, does not fabricate a host field and rejects missing fields in other receipts. The original schema failure is preserved.

Hybrid's CPU4 and separate CUDA4 pipeline gates are complete, strict and backed up off server. The CUDA increment has49 verified members; both same-GPU Hybrid/legacy pairs match terminal tensors, every-round metrics, attacks, diagnostic fields and RNG exactly, excluding only cumulative wall time as in the original comparator. All four CUDA terminals predict a constant negative class, withACC0.516686 and zero gaps. These negative short-run outcomes are preserved and cannot establish performance advantage, CPU/CUDA equivalence or70-round equivalence. {paragraph}

The exact11 subsequent evaluation scope is native71 minus the closed60: non-IID F Flip seeds91001–91010 and FedSA91001. Its source review preserves scientific functions and passes42 refusals. {next11_note}

{fl_bound_note}

{after71_note}

{after82_failure_note}

{after92_note}

The latest previously verified Git publication is `{published['commit']}`, with{published['committed_blobs_sha256_verified']} committed blob SHA checks against the remote branch. New completion evidence is published in a separate increment. Source preparation, approval, dispatch and completed scientific results are distinct. The native three-hour chat monitor remains PAUSED; supervisor-managed training does not restore it. Final test, the remaining baselines, complete mechanism comparisons and final manuscript claims remain unfinished.

The complete900-record three-view descriptive paper tables are now independently root-reviewed:900 original receipts rejoined,8100 metrics reconstructed from saved group counts and4860 mean/sampleSD scalars checked. AllIID/non-IID/five-scene/fixed10/9/6-seed views are retained under outputs/guardfed_tables/celeba_nine_method_three_view_20261009. Old native metrics/displayed values remain exact;94 oldJSON sampleSD last-bit differences(max2.78e-17) are disclosed without changing tolerance. The24-comment complete author-review reply now integrates U100 ten-scene results and900 calibration attribution at {state['latest_rebuttal_draft']['entry']}. {reply_progress_note} COMPAS counterexamples, actual TableII repeat counts, device/selection history and all pending items remain explicit. The previous sealed writing copy is retained. Submission remains gated on the full cohort, and the submitted manuscript source has not been edited.

An independently checked accepted900 descriptive attribution now accompanies that draft:2052 scalar checks pass after averaging the ten scenarios within each seed. Native ACC/AEOD/ASPD mean advantages involve6/8,8/8,7/8 baselines; common calibration gives7/8,4/8,1/8. These are direction counts, not seed win rates or significance. GuardFed's native and shared outcomes are identical, and the baseline calibration changes the comparison. The English validation900 addendum preserves all negative differences, selection/device limits and the undecided primary endpoint; no new inference/refitting/test occurred.

The accepted nine-method three-view tables are also compiled into a nine-page A3 landscape PDF, covering all views and fixed10/9/6 panels. Independent root verification checks2430 displayed mean/SD pairs(4860 scalar strings), exact original fragment bytes and label-only compile copies, all nine page bounds and visual renders; the original67-member seal is unchanged. This is display compilation only, not new statistical or experimental evidence.

The bounded manuscript locator found historical IEEEtran source in paper.md, but its title, method and table structure differ from the submitted PDF and its three named bibliography/figure dependencies were not found in the searched paths. Nineteen input SHA pins and three source variants were independently checked. No matching submitted-version source or complete build is claimed, and the original manuscript was not edited. The submitted source-path question is pending while independent experimental work continues.

A concrete historical Fig3 correction candidate now plots original round70 ACC against AEOD and ASPD, retaining all26 settings and260 same-round records;78 mean checks and PDF/PNG review pass. It is explicitly a single-seed author-review candidate, not an adopted replacement or recovery of original plotting/execution/checkpoint identities. No FairScore, metric-wise extrema, test-based round selection or scene-as-seed SD is used. Historical test exposure, unresolved ForestDiffusion execution and PCA-label limitations remain in the caption and figure. P4 remains open.

'''
if state.get('author_decisions_20261010'):
    current+='\nThe2026-10-10 author accepts Huber identity projection on R^p as a CNN empirical adaptation, without inherited theoretical guarantees. The author delegates LoGoFair population definition; root selects fixed image-ID20 virtual cohorts, root-only DP fitting, explicitly not true training-client fairness. These H/L choices are resolved; original prepared records remain preserved.\n'
if state.get('gradient64_validation_search_20261010'):
    g=state['gradient64_validation_search_20261010']
    observed=g.get('latest_measured_observation',{})
    current+=f"\nThe new Fed-NGA32+Huber32 validation search actually started with one physicalGPU1 worker, CPU105, nice10/idleIO. Root verifies startup round{g['root_startup']['observed_round']}, physicalGPU UUID and original scientific/job bytes. The latest actual snapshot separately observes{observed.get('terminal70_observed',0)}/64 terminal70 jobs; {g['offserver_accepted']} offserver acceptances are root-adopted. The first Fed-NGA candidate is constant-negative, withACC0.516686 and zero gaps; it is retained without a champion claim. Two earlier resource-preflight engineering failures occurred before training and are preserved. Entry: {g['root_startup_path']}.\n"
if state.get('logofair32_validation_search_20261010'):
    l=state['logofair32_validation_search_20261010']
    current+=f"\nThe original8×4 LoGoFair validation-only search uses30 post-rounds per job from four accepted FedAvg models/margin caches, with zero CNN calls or new training. {l['original_strict_closed']}/32 are locally closed by the original strict bridge, root-adopted{l['root_adopted']}; no recipe is selected before the complete search. All outputs stay on checked F storage. The earlier real-score3-round gate passed40 Beta fits and exact serialized replay, but its constant-negative prediction is preserved as interface evidence only.\n"
    if l['root_adopted']==32:
        current+='The actual complete32 passed the separate original strict/saved-prediction review and root artifact/score checks. Frozen four-condition score selectsLoGoFair-DP_07; accuracy champion differs, all eight candidates/four Pareto entries/eight constant predictions remain. Seed91001/fit_seed1719 and virtual population limits remain; the separate100-cell coverage is tracked below.\n'
if state.get('logofair100_fullcoverage_20261010'):
    l=state['logofair100_fullcoverage_20261010']
    if l.get('root_adopted')==100:
        current+='\nLoGoFair fixed-recipe100 coverage is fully root-adopted:96 new30-postround fits and four explicit reuses, IID/non-IID×five scenarios×ten model seeds, fixed fitseed1719. Original strict and1,986,700 saved predictions/300 metric checks pass with zero error; independent1027 artifact hashes,198 statistics and99 display cells pass. The single constant outcome and all10/9/6 panels are retained; virtual20cohorts are not training clients. No new CNN training/inference or final test. Bulk remains on F and only compact reports/restore references are copied locally. Entry: '+l['canonical_table']+'.\n'
    else:
        current+=f"\nLoGoFair fixed-recipe100 coverage has started as96 new plus four reused records; latest local original strict closures are{l['local_strict_closed_observed']}/96. Complete independent acceptance is still pending. Entry: {l['root_startup_path']}.\n"
if state.get('celeba_native_ten_method_table_20261010'):
    current+='\nThe native-only ten-method1000-cell descriptive paper table is independently adopted:IID/non-IID, five scenarios each and all10/9/6seed panels. All810 original nine-method metric objects remain exact;1800 scene-level mean/SD scalars,900 rendered cells,540 seed-first aggregate scalars and270 rendered aggregate cells pass. LoGoFair uses its fitted DP native predictions. This does not expand the original900 three-view or final-evaluation scope. Seven method coverages remain incomplete. Entry: '+state['celeba_native_ten_method_table_20261010']['table_path']+'.\n'
if state.get('mechanism_remaining620_valid_20261010'):
    r=state['mechanism_remaining620_valid_20261010']
    current+=f"\nThe finite remaining620 CPU terminal-validation queue actually started under {r['actual_service']}, excluding prior180 and Full reinference. A single eight-thread CPU evaluator on112–119, nice10/idleIO with hidden CUDA, binds only completed original70-round producers. The latest actual snapshot observes{r.get('latest_measured_observation',{}).get('remote_strict_closed',r['remote_strict_closed_at_startup'])} remote closures, of which{r['new_offserver_accepted']} newly adopted offserver records have original archive and saved-array verification plus exact accepted native model/result restore-chain identity. The root wrapper taskset syntax failure occurred before Python and is preserved. The first transport failed at a missing pinned verifier path, then its unchanged49-member archive passed one explicitly reviewed recovery,9 metrics/24 counts/3 rules and native discrepancy0; no scientific work was rerun. Entry: {r['root_startup_path']}.\n"
    if r.get('C100_table_adopted'):
        current+='C100 complete ten-scene statistics and three-view tables are separately root-adopted; the exact oldC80 snapshot remains preserved. Entry: '+main['C_three_view_full100_table']['table_path']+'. Other six controls, frozen final evaluation and submitted manuscript remain unfinished.\n'
    elif r.get('C100_replay_complete'):
        current+='The final19 C records are now root-adopted:173 archive members/171 metrics/456 confusion counts/57 prediction rules, zero native discrepancy, exact20 native200 record identities and40 original model/result member rehashes. Prior180 and first181 are unchanged; cumulative U100+C100=200. C100 statistical tables await separate review; six further mechanism controls and final evaluation remain unfinished.\n'
    else:current+='The cumulative181 includes one partial C-scene record beyond the C80 table, so no additional scene mean is reported.\n'
if state.get('gradient200_fullcoverage_source_preparation_20261010'):
    current+='\nFed-NGA/Huber200 coverage source preparation passes independent review only:192 new plus eight reuses are planned, with no selected recipe, actual jobs or dispatch. Actual complete64 offserver acceptance, source freeze and new-attack real-image gates remain prerequisites; the prepared source is not a scientific result.\n'
if state.get('gradient200_new_attack_gates_source_20261010'):
    current+='\nThe14 common-three-round new-attack gate sources pass independent source review only; actual image gates remain0. The formal70-round checker is unchanged.\n'
if state.get('added_baseline_three_view_scope_20261010'):
    current+='\nAdded-baseline three-view compatibility is source-reviewed, without new evaluation or fit. Four CNN methods need their original strict-identity bridges. LoGoFair native must retain its fitted DP state and virtual mapping; the cache valid_native_prediction denotes FedAvg raw, not LoGoFair native. Any backbone raw/shared replacement diagnostic requires explicit labelling, and the final primary endpoint remains undecided.\n'
if state.get('hybrid100_fullcoverage_20261010'):
    h=state['hybrid100_fullcoverage_20261010']
    hybrid_note='\nHybrid100 actual v3 metadata is bound and root-adopted:96 new jobs, four explicit original screen reuses and seven pipeline gates. All147 returned metadata members pass identity/grid checks. The Python3.10 summary replays exactly on server3.12 with explicit sequential summation; original values/ranking/tolerances are retained and the earlier pre-job binding failure is preserved. '
    if h['canaries_offserver_adopted']==7:
        hybrid_note+='All seven real three-round gates have passed the original saved comparison and254-member offserver/root checks, including two same-horizon implementation pairs. They supply zero70-round scientific samples; saved terminal weights/final RNG summaries do not imply per-round weights or universal70-round equivalence. Entry: '+h['canary_closure_path']+'. '
    else:
        hybrid_note+='The seven three-round gates have actually started onCPU104/GPU0, with observed first-job round2. Gate closure remains pending. '
    if h['formal100_started']:
        hybrid_note+='The frozen96-new70-round validation queue has actually started, with oneCPU104/GPU0 worker observed at round1 and physically mapped to the boundGPU UUID. The four original records are reused and the old canary authorization retained byte-for-byte. New70-round acceptance remains0 at this startup boundary. No test or automatic retry. Entry: '+h['coverage_start_path']+'.\n'
    else:
        hybrid_note+='The formal96 queue has not started.\n'
    if h.get('new_accepted',0):
        hybrid_note=hybrid_note.replace('New70-round acceptance remains0 at this startup boundary.',f"At startup new70-round acceptance was0; the original strict/offserver/root chain now accepts{h['new_accepted']}/96, with four reuse separate.")
        if h['new_accepted']==9:
            hybrid_note+='The latest exact8 IID Benign91003–91010 increment passes272 archive-member hashes and64 terminal tensor identities. The prior accepted seed91002 remains exact. A full ten-seed IID Benign table requires independent statistical adoption; other scenarios, whole100 coverage and final test remain incomplete. Entry: '+h['latest_delta_root_path']+'.\n'
        else:
            hybrid_note+='First IID Benign91002 record passes188 archive-member hashes and eight saved terminal tensor identity checks,70 complete rounds and valid19867 with all metrics from one checkpoint. No single-seed scenario SD or runtime-equivalence claim. Entry: '+h['first_delta_root_path']+'.\n'
    if h.get('native_IID_Benign_table'):
        hybrid_note+='The independent and root-adopted native IID Benign table now contains10 checkpoints (nine formal plus one screen reuse), fixed10/9/6 panels,18 mean/sampleSD scalars and9 display cells. It is one complete exposed-validation scene; other nine scenes and three-view/full100/final-test coverage remain incomplete. Entry: '+h['native_IID_Benign_table']['table_path']+'.\n'
    current+=hybrid_note
p=TRAIN/'REBUTTAL_COMPLETION_20261009.md';text=p.read_text(encoding='utf8')
if state.get('author_adaptation_reply_patch_20261010'):
    current+='\nThe author-approved Huber identity-projection and delegated LoGoFair virtual-cohort definitions now have root-reviewed, reversible English response/manuscript patches. All24 original comments and original numbers are retained; no scientific acceptance or manuscript application is implied. Entry: '+state['author_adaptation_reply_patch_20261010']['reply_patch']+'.\n'
if main.get('three_view_new_models_offserver_verified')==251:
    current+='\nLatest saved-array increment adds exactly11 records, retaining the original240; U100+C100+A51 now total251 strict/offserver/root-adopted three-view terminal checkpoints. All five A IID scenes have ten paired seeds, while A non-IID Benign has only one. The original A40 table is retained; the separately adopted A50 table is described below. All five non-IID A scenes remain unfinished. No new inference/fit was used for this transport. Entry: '+state['mechanism_remaining620_valid_20261010']['root_adoption_path']+'.\n'
if state.get('celeba_native_ten_method_PDF_20261010'):
    pinfo=state['celeba_native_ten_method_PDF_20261010']
    current+='\nThe ten-method native validation table PDF passes all three page visual checks and exact matching of900 mean/sampleSD pairs(1800 numbers) to the accepted source. Pages contain10/9/6 seed panels and both distributions/five scenarios. This is a display artifact, not final test evidence or complete17-method coverage. Entry: '+pinfo['pdf_path']+'.\n'
if state.get('added_CNN_three_view_bridge_20261010'):
    current+='\nThe four-CNN metadata bridge source passes64 focused root-replayed gates and preserves17 original scientific functions. Only existing exact FL6/Hybrid1/NGA8 root-adopted chunks are registered; Huber without an original70-round proof is refused. This prepares identity handling only: new scientific evaluation, fitting, training and test calls are zero.\n'
if main.get('A_three_view_five_scene_table') and not main.get('A_three_view_six_scene_table') and not A80:
    a=main['A_three_view_five_scene_table']
    current+='\nCurrent A table supersedes the four-scene display boundary: five complete IID scenes,50 Full–minus_A pairs,972 recomputed scalars/486 display cells,900 count-derived metrics and2400 integer-count checks; old80 record bytes/order and648 scalars/324 cells exact. The separate five-IID aggregate first averages within seed; the non-IID singleton is excluded. Native/shared Sp-DFA deletion means are −0.230pp/−0.00950/+0.00117 (ACC/AEOD/ASPD), retaining opposite fairness directions and all10/9/6 panels; the ACC direction is a mean, not every seed. Full3CPU47GPU versus A50CPU/commoncu128 and prior selection/test exposure remain disclosed. Five non-IID A scenes and other controls remain incomplete. Entry: '+a['table_path']+'.\n'
if main.get('A_three_view_six_scene_table') and not A80:
    a=main['A_three_view_six_scene_table']
    current+='\nAt A60 table adoption the mechanism acceptance was native264 and three-view260 (U100+C100+A60), with distinct cutoffs. The A60 table covers five IID scenes plus non-IID Benign, each ten paired seeds:120 records,1134 mean/sampleSD scalars,567 cells,1080 count-derived metrics and2880 base-count checks. Original A50 records/statistics and IID-only seed-first aggregate bytes remain exact. Non-IID Benign native/shared deletion means are ACC−0.540pp,AEOD+0.00488,ASPD−0.00582: all tradeoffs and10/9/6 panels remain. Full5CPU55GPU/59cu128+1cu130 versus A60CPU/cu128 is disclosed. Four other non-IID A scenes remain incomplete. '+('A60 is incorporated into the current complete24-comment author-review reply and insertion draft. ' if state['latest_rebuttal_draft'].get('A60_incorporated') else 'The reader reply still has the separate A50 evidence cutoff; A60 is not yet incorporated. ')+'No final test or manuscript application. Entry: '+a['table_path']+'.\n'
if A80:
    current+='\n'+'Current A80 table is root-adopted: five IID scenes plus non-IID Benign/F Flip/FedSA, eight scenes with ten paired seeds each,80 pairs/160 records,1458 mean/sampleSD scalars and729 display cells; all10/9/6 panels remain. Old A60 preserves120 object bytes/order,972 scene statistics/486 cells and IID seed-first aggregate bytes. Raw FedSA deletion improves all three means; native/shared deletion slightly raises ACC while worsening both gaps. All negative outcomes and view-dependent tradeoffs remain, without necessity or significance claims. Full5CPU75GPU/79cu128+1cu130 versus A80CPU/cu128, selection history and historical test exposure are disclosed. Non-IID S-DFA/Sp-DFA, remaining controls and final evaluation remain unfinished. '+('A80 is incorporated into the root-adopted reader reply; use its linked adoption evidence. ' if state['latest_rebuttal_draft'].get('A80_incorporated') else 'The current reader reply remains at A60; A80 table adoption does not imply reply integration. ')+'Entry: '+A80['table_path']+'.\n'
if state['latest_rebuttal_draft'].get('editorial_reversible_edits'):
    current+='\nThe complete24-comment English reply and manuscript insertion candidate retain all original quotes, prior scientific numbers and tables. Historical details follow the responses. '+('The root-adopted A80 reader integration is recorded at the linked reply entry. ' if state['latest_rebuttal_draft'].get('A80_incorporated') else 'The latest A60 increment adds21 exactly reversible spans and6 adopted JSON mean/SD pointer pairs; all2457 prior numeric strings and98 links remain, with five IID scenes plus non-IID Benign and the remaining four non-IID boundary explicit. ' if state['latest_rebuttal_draft'].get('A60_incorporated') else 'The A50 increment retains the five IID scenes and their paired/seed-first tradeoffs. ')+'This is author-review material, not submission-ready or applied manuscript text. Entry: '+state['latest_rebuttal_draft']['entry']+'.\n'
if fl47_current and not fl47_current.get('validation_hold') and not fl47_current.get('complementary_adopted'):
    e=state['FLGMM_closed47_valid_three_view_20261011']
    current+='\nFinite47 previously accepted FLGMM checkpoints are now being evaluated on CPU120–127/eight threads/single process, with no training or test. At '+e['observed_utc']+' observed saved replay receipts are '+str(e['receipts_observed'])+'/47; scientific acceptance remains0 pending original whole-saved checks and offserver verification. Source scope44new+4screenreuse−1already-adopted interface; original science and native tolerance unchanged. Entry: '+e['start_receipt_path']+'.\n'
if state.get('added_CNN_exact3_valid_interface_20261010'):
    e=state['added_CNN_exact3_valid_interface_20261010']
    if e['root_scientific_acceptances']==3:
        current+='Exactly three representative added-CNN validation interfaces are root-adopted: FLGMM, Hybrid and one FedNGA search checkpoint. The unchanged whole original check passes on Linux; the independent original saved-array block on F reproduces root-only calibration, predictions,27 metrics,72 base counts and9 rules exactly, with native discrepancy0. The original Windows whole-check failure at FLGMM audit group_kl (~2.2e-19) remains preserved; its full root receipt is not claimed exact on Windows, and tolerance is unchanged. Three interfaces do not supply complete100-cell comparisons or a final-test endpoint. No training or test. '+e['root_proof_path']+'.\n'
    else:
        current+='Exactly three added-CNN real-image valid interface checks have started; offserver acceptance remains0 until the separate root proof. '+e['start_receipt_path']+'.\n'
if fl47_current.get('validation_hold') and not fl47_current.get('complementary_adopted'):
    e=state['FLGMM_closed47_valid_three_view_20261011']
    current+='\nAll47 FLGMM replay receipts and the Linux whole original checker pass; the98-member F archive is verified. The unchanged Windows original array block stopped at record5 (IID Benign91006): Root-only threshold fit changed. Its failure and prior four loop completions are retained, but new three-view adoption remains0. The failed command is not retried and no tolerance changes. Prior native training acceptances and queues remain unchanged. Entry: '+e['validation_hold_path']+'.\n'
    if e.get('single_failed_record_diagnostic'):
        current+='The separately named single-failed-record diagnostic measures one ULP (−1.1102230246251565e−16) in shared calibration fit_diagnostics/server_adaptive_lambda and its dependent fit SHA; effective thresholds, all three predictions, metrics, counts and root receipt are exact. The first diagnostic comparator key-type failure remains preserved. This is one record only, not an adoption or a complete Windows47 pass; no platform cause is established. Entry: '+e['single_failed_record_diagnostic']['root_review_path']+'.\n'
        if e['single_failed_record_diagnostic'].get('operation_trace_path'):
            current+='A zero-fit stdlib trace using identical operands in the existing two runtimes first differs at math.log1p and reproduces each captured coefficient. OS/libm/Python factors are not individually isolated; this arithmetic diagnostic does not accept47 records. Entry: '+e['single_failed_record_diagnostic']['operation_trace_path']+'.\n'
if fl47_current.get('complementary_adopted'):
    current+='\n'+'Exactly47 valid terminal-checkpoint three-view records are now root-adopted using complementary evidence:47 new plus one separately retained prior FLGMM interface total48, from44 accepted native training records plus4 screen reuses; mechanism three-view acceptance was260 at this FL adoption and remains separately reported. The whole unchanged Linux checker supplies47 root-only refits; the98-member F transport passes. The independent Windows SAVED_OUTPUTS_AUDIT_NO_REFIT passes47 records with zero fits,423 metrics,1128 base counts,141 rules and native discrepancy0. Original Windows exact refit and whole checks remain FAIL: the original hold, failures, single-record and operation diagnostics are retained; dual-platform bitwise recalibration is not established. Four root-audit group_kl differences of -2.168404344971009e-19 remain disclosed. No tolerance changes, new training/CNN/test, complete100-cell or17-method claim. Entry: '+fl47_current['root_proof_path']+'. Historical hold: '+fl47_current['validation_hold_path']+'.\n'
if state.get('final_split_metadata_20261011'):
    current+='\nOfficial candidate final partition2 identity is independently verified:19,962 ordered image IDs, with train/valid identities unchanged and all three official partitions complete/disjoint. Only image_id/split arrays were decoded; the metadata ZIP was hashed as a whole for identity. No label-array decoding, pixels, models, calibration, inference or test metrics. The original prepared protocol remains byte-exact and unfrozen; primary endpoint and final evaluation remain undecided, with historical test exposure retained. Entry: '+state['final_split_metadata_20261011']['root_proof_path']+'.\n'
if state.get('FLGMM_after48_valid_three_view_20261011'):
    e=state['FLGMM_after48_valid_three_view_20261011']
    current+='\nA subsequent exact13 complementary adoption now totals61 FLGMM validation checkpoints (57 native training acceptances plus4 original screen reuses), preserving prior48. Linux whole13 and F30-member verification pass; Windows saved-output audit verifies117 metrics/312 base counts/39 rules with zero refit and native discrepancy0. One additional root-audit group_kl difference of -2.168404344971009e-19 is disclosed. Windows new13 refit was not performed; historical47 refit/whole remain failed. This does not change mechanism288 or establish full100/final-test completion. Entry: '+e['root_proof_path']+'.\n'
if state.get('FLGMM_after48_valid_three_view_20261011',{}).get('six_scene_table'):
    t=state['FLGMM_after48_valid_three_view_20261011']['six_scene_table']
    current+='\nFLGMM six complete scenes (five IID plus non-IID Benign, ten seeds each) now have root-adopted raw/native/shared-calibration tables. Fixed10/9/6 panels use sample SD(ddof1);324 statistical scalars,162 rendered cells and549 count-derived metrics were independently recomputed. All61 accepted records remain; the single non-IID S-DFA screen record is excluded from complete-scene statistics. IID alpha5000/non-IID alpha5, validation recipe exposure, calibration tradeoffs and preserved Windows refit limits are disclosed. This is60 complete-scene records, not full100 or final test. Entry: '+t['table_path']+'.\n'
start=text.index('## Current accepted increment');end=text.index('## Historical accepted increment',start)
p.write_text(text[:start]+current+text[end:],encoding='utf8')
print(json.dumps(dict(updated_current_section=True,mechanism_strict=accepted,baseline_replays=replayed,mechanism_views=mechanism_views,goal_complete=False)))
