"""Refresh only the current completion section from accepted receipts."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/'docs/server_deployment_20260923/training_20260923'
state=json.loads((TRAIN/'TRAINING_STATE.json').read_bytes())
main=state['celeba_mechanism_v1'];baseline=state['final_evaluator_runtime_20261009']
accepted=main['scientific_results_offserver_verified'];replayed=baseline['actual_native_valid_image_replays_accepted']
published=state['latest_publication_verification']
screen=state.get('hybrid_screen32_20261009')
failure=state['final_evaluator_runtime_20261009'].get('failed_model_id')
failure_paragraph=("The CPU replay service has fail-stopped on FairGuard/IID/F Flip/seed91009: native metrics differ from the original record despite matching model/config/data identities and unchanged tensors. The original1e-12 tolerance is preserved, the65-member failure archive is verified off server, and chunk036's10 strict partial results are not counted. The cause is not established; no original metric, model, threshold or selection rule has changed and no retry has started. Previously accepted records remain valid. " if failure else "")
paragraph=(f"The original32-item Hybrid validation search has actually started under `{screen['service']}`; its independent source/startup archive and root verification bind the unchanged eight recipes, four conditions, seed91001 and70 rounds. It uses one GPU0 worker, CPU104, one compute thread and nice10. No100-job multi-seed confirmation, test or automatic retry is authorized. "
    if screen else "The unchanged original32-item Hybrid validation search is root-approved for a separate frozen execution copy; exact source/scope approval is complete, while actual dispatch and source-bound startup acceptance remain separate requirements. ")
current=f'''## Current accepted increment — measured {state['last_health_check']['checked_utc']}

The main mechanism queue has{main['queue_completed_observed']} terminal jobs observed, with{accepted} independently accepted and backed up off server in seven linked increments;100 Full controls remain explicit reuse. The latest seven-ID increment has86 verified members, archive `abf0c8a30726a786ecebcff4395e07ffbda77295b3389106f6d7628011f0b9cf`. The queue continues with eight workers and no observed failures. These counts do not establish all800 controls or the whole rebuttal.

The existing nine-method terminal-model validation replay has{replayed} distinct accepted and off-server-verified models, with{900-replayed} still missing. Each closed increment has original strict acceptance and independent raw/native/shared prediction-array checks; all preserve native metrics exactly. The cumulative collector is `{baseline['accepted_collection_path']}`. {failure_paragraph}This is validation replay, not final-test evaluation or new model training.

The approved exact15 mechanism replay increment is complete and its service is EXITED: three disjoint backups contain149 content members plus three inventories, with135 metrics,360 confusion counts and45 prediction rules independently reconstructed. Together with the original eight,23 actual minus_U terminal checkpoints have strict raw/native/shared validation replays; native discrepancy is zero. Full paired three-view receipts remain missing and will join actual baseline replay acceptance, without new Full inference or substituting old calibration. This does not establish the complete mechanism comparison.

FLGMM's first2/32 full70-round validation-search jobs are strictly accepted, backed up off server and checked with the frozen original acceptor. They cover IID/Benign and IID/S-DFA for one candidate; no complete four-condition recipe or winner has been selected. The22-member archive is `638d7bba07fcc1f3a9953a8f71fb412516d8b5ffcc134eb3b3794cf82dedcbbf`; the original search continues.

Hybrid's CPU4 and separate CUDA4 pipeline gates are complete, strict and backed up off server. The CUDA increment has49 verified members; both same-GPU Hybrid/legacy pairs match terminal tensors, every-round metrics, attacks, diagnostic fields and RNG exactly, excluding only cumulative wall time as in the original comparator. All four CUDA terminals predict a constant negative class, withACC0.516686 and zero gaps. These negative short-run outcomes are preserved and cannot establish performance advantage, CPU/CUDA equivalence or70-round equivalence. {paragraph}

The latest previously verified Git publication is `{published['commit']}`, with{published['committed_blobs_sha256_verified']} committed blob SHA checks against the remote branch. New completion evidence is published in a separate increment. Source preparation, approval, dispatch and completed scientific results are distinct. The native three-hour chat monitor remains PAUSED; supervisor-managed training does not restore it. Final test, the remaining baselines, complete mechanism comparisons and final manuscript claims remain unfinished.

'''
p=TRAIN/'REBUTTAL_COMPLETION_20261009.md';text=p.read_text(encoding='utf8')
start=text.index('## Current accepted increment');end=text.index('## Historical accepted increment',start)
p.write_text(text[:start]+current+text[end:],encoding='utf8')
print(json.dumps(dict(updated_current_section=True,mechanism_strict=accepted,baseline_replays=replayed,mechanism_views=23,goal_complete=False)))
