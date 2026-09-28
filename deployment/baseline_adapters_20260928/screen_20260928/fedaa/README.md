# FedAA bounded validation screen — 2026-09-28

Prepared, **not launched**. Frozen pilot files and main training core remain unchanged. This package extends only the already validated official-DDPG adaptation's candidate settings and run length.

## Frozen search coverage

Eight candidates = synchronized actor/critic LR `{.001,.01}` × retained clients `{10,16}` of 20 × local Adam LR `{.0005,.001}`. Each candidate covers IID alpha5000 / non-IID alpha5 × Benign / S-DFA, shared seed91001. Total **32 new 70-useful-round validation-only jobs** in `screen_jobs/manifest.json`; all use train162770 and valid19867. No test evaluation, historical-result replacement, metric mixing or automatic expansion.

`config` stores the unchanged core dataclass fields; `policy_config`, `aggre_num`, `policy_seed` are explicit job/result/checkpoint identity fields. Both policy rates are checked against their optimizer learning rates. Root reward remains raw accuracy on the clean train-derived root, and final reporting is raw ACC/AEOD/ASPD from the final validation checkpoint. Preserve the existing score exactly, all candidates, negative outcomes and accuracy/Pareto alternatives. n=1 does not support sample SD or significance claims. Selection aggregation across conditions must follow the parent stage protocol rather than a new rule invented here.

## Deployment and gates

Preserve this relative directory structure under a deployment root:

```
celeba_baselines/
  fedaa/fedaa_official_adapter.py
  fedaa/official/DDPG/DDPG.py
  integration_20260928/compare_pilot_resume.py
  screen_20260928/fedaa/...
```

The worker uses the separately specified `--repo` for frozen CNN/data/attack code and verifies every inherited source/data hash. It verifies its new wrapper/runner and pinned official-source hashes against each job before training. `--official-dir` may point to the existing official checkout.

`gate_jobs/manifest.json` contains two new-code default-parameter regression gates: non-IID, seed91001, Benign/S-DFA, 3 useful rounds, local LR .001, policy LR .01, keep10. Run these first against the existing real-data pilot checkpoints. Compare with:

```
python compare_pilot_equivalence.py --pilot OLD/training_state.pt --screen NEW/training_state.pt --out equality.json
```

This checks CNN weights, full controller including policy/replay/optimizer/RNG, external RNG, validation metrics, diagnostics and attack audits. Only identity metadata is excluded because the new stage and source files necessarily differ. It requires both checkpoints to be at round3. The default CPU controller was already checked bitwise against the frozen pilot for three synthetic parameter cohorts; actual GPU/image equivalence remains the deployment gate.

Worker command:

```
python run_fedaa_screen.py --repo /workspace/GuardFed-celeba-expanded --job JOB.json --out NEW_RUN_DIR
```

Each round saves progress plus the complete `training_state.pt`; the first boundary remains in `checkpoint_round1.pt`. At a controlled boundary, `--stop-after 1` stops after saving round1 without calling the result complete. After external-interruption diagnosis, `--resume PRIOR/training_state.pt` can continue only an identical job/source/data/environment. The first checkpoint is retained in the resume chain. Do not resume an old pilot directly into a new screen: identities intentionally differ. There is no automatic retry loop, and nonempty output without `--resume` is rejected.

Strict result acceptance:

```
python accept_result.py --job JOB.json --out RUN_DIR
```

It verifies complete configured rounds, matching final raw metrics, validation denominator, policy transitions and selected IDs, core/policy config, source/data/adapter identities, model/full/first-checkpoint hashes, and equality between standalone CNN weights and the complete final state. Use it before summary or result reuse. Failed evidence is retained separately.

## Local checks completed

`python check_screen.py` passed on torch2.8.0+cpu: default controller's three rounds exactly match the frozen pilot; all four policyLR×keep combinations retain the specified actor/critic rates and restore state; 32 tasks form the exact eight-candidate/four-condition product; test-split mutations are rejected. CLI parsing and Python compilation pass. `local_checks.json` explicitly records that real CelebA/GPU execution was not performed by this preparation task.

The official policy remains pinned to commit `1fba884934cbec1b3506812d9d612e6a181b1d79`. Keep16 is explicitly a search variant, not the upstream 50%-retention default. The established useful-round timing, CPU policy, ordered client IDs and zero-distance guard remain disclosed adaptations. Perfect-root terminal reward still raises rather than silently changing upstream stopping behavior.
