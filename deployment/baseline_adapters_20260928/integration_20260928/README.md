# FedAA CNN integration boundary — 2026-09-28

Status: **local component and fabricated-image plumbing checks passed; real CelebA/GPU gate pending**. No main worker, server, frozen result or training queue was changed. No formal experiment was launched.

## What is reviewable

- `fedaa_round_adapter.py`: persistent `FedAARounds` invokes the existing pinned `OfficialFedAA` wrapper; no substitute policy or aggregation heuristic. `train_and_aggregate_round` reuses the image core's actual local trainer, attack helper and root evaluation order.
- `check_round_adapter.py`, `verification.json`: CPU measurements on torch 2.8.0+cpu. Three fabricated RGB64 rounds pass through the actual CelebACNN and actual local Adam training. Three policy transitions are learned, parameters remain finite, full-model aggregation matches direct actor-weight averaging. A serialized model/policy/external-RNG boundary resumes two subsequent fabricated CNN rounds bitwise. Separately, controller replay/optimizers/targets/pending transition resume three rounds bitwise; actor parameters change and external RNG remains isolated.
- `source_snapshot/`: partial checkout used for interface verification only, recovered from existing local `celeba_expanded_code_20260924.bundle`, commit `6ddb4da67de3ad4f4cacd4becf2405bbff2ba3da`. Its Git object database references the existing local clone; it is **not** a portable deployment archive.

The current image core's LF-normalized SHA256 is `cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed`, matching the existing expansion protocol. CelebA loader LF SHA is `0f48fbc6d241e7d0cde692839659e817feab407991795a6f92a33052a9cc07ce`. Windows checkout byte hashes are separately recorded in `verification.json`.

Runnable check:

```powershell
python tmp/celeba_baselines/integration_20260928/check_round_adapter.py
```

These synthetic checks do not measure CelebA accuracy/fairness or establish CUDA equivalence. The source worker still labels `FedAA` as a stateless core heuristic; it must not silently be reused as this method identifier.

## Minimal real-data integration

Use an isolated, newly versioned worker and method name such as `FedAA-DDPG-adapted-v1`:

1. Reuse existing `load_bundle`, seeded CNN initialization and `client_runtime_data`. Assert official valid-only evaluation, unchanged data/cache hashes, train-root/client/validation disjointness, clean root (no noise/synthetic data), actual client IDs 0–19, and a recorded policy seed.
2. Evaluate the initial CNN on `bundle['server_X']`, `server_y`, `server_sensitive`; **only this train-derived root** can supply policy reward. Instantiate `FedAARounds(model, official_dir, policy_seed, initial_root_accuracy, participants=20, keep=10)` once per run.
3. Each useful round calls `train_and_aggregate_round(core, model, bundle, clients, audits, cfg, controller, device)`. Validate attack audits through existing `validate_attack_audit`, log diagnostics, and evaluate unchanged validation metrics from this model. Do not calibrate FedAA using GuardFed's thresholds unless it is explicitly the shared-calibration comparison.
4. Write final `model.pt` as the ordinary CNN state dict for existing result acceptance, and save a separate full `training_state.pt` for resumability. The new worker/result schema must record its adapter hashes, DDPG commit/SHA, adaptation settings and checkpoint identities. Existing `checked_result` should be retained, with an extra check that all 70 rounds have 70 policy updates and the final pending transition is preserved.
5. Run a bounded real-CelebA, 20-client, >=3-round CPU/GPU canary before any 70-round sweep; include both Benign and S-DFA, verify root split access and attacked full-parameter inputs. Then independently choose bounded policy/local-LR search settings on validation, freeze and run paired seeds/scenarios. The present local check authorizes no scientific baseline claim.

## Explicit adaptation contract

Official source remains https://github.com/Gp1g/FedAA at `1fba884934cbec1b3506812d9d612e6a181b1d79`, DDPG SHA `e9bdbccdaf3aba7e1eece10370cbf7cedfe66104a691bc24a9d4e1300cd12d68`. It is loaded from the existing separate checkout; no license is invented.

Retained: actor/critic/targets/replay/optimizer implementation, noisy actor, replay batch 16, learning rates .01, discount .99, tau .001, default buffer 100000, full-parameter pairwise distance states, scalar accuracy reward, 50% participant retention (10/20), alpha=0 full-model weighted aggregation. Full local models are reconstructed after this project's attack helper; deltas are not mistaken for the state vectors.

Disclosed adaptations: common project CNN initialization and local Adam/data-attack protocol; clean train-root reward instead of upstream `server_test`; CPU policy with isolated RNG; all 20 clients in ascending ID order rather than upstream random permutation; all-zero distance guard inherited from existing wrapper. This is an official-policy adaptation, not bitwise reproduction of upstream's whole training program.

Timing: upstream starts with an initial no-op aggregation and ends with an unused trained cohort. Here virtual initialization stores `(ones, action(ones), initial root accuracy)`. Each of 70 useful cohorts finishes the previous transition before choosing current weights, resulting in 70 useful aggregations and 70 DDPG updates. Last reward/action/state remains pending; no fake next-state or extra training is added. If a pending reward is exactly 1.0, the wrapper raises before proceeding because upstream would terminate; any different 70-round stopping policy needs an explicit amendment rather than silent retry.

## Required recovery state

Save at the completed `train_and_aggregate_round` boundary:

- Model full parameter state; controller `state_dict` (actor/critic/targets, both optimizers, replay contents/pointer, policy torch/NumPy RNG, transition count, pending state/action/reward, selected IDs, model layout).
- External torch CPU RNG, `torch.cuda.get_rng_state_all()` on CUDA, NumPy RNG, Python random RNG; current round, attack audits and complete trajectory/round diagnostics.
- Config/recipe/source/data/partition identities and environment. Load and verify identities before any computation; reconstruct fixed bundle/clients/model/controller, load state, then restore external RNG **last**. Current local Adam is recreated for each client/round by the shared core, so no optimizer persists between completed rounds. Checkpoints in mid-client training are unsupported.

The original `model.pt` alone is evaluation-only; restarting FedAA from it would discard policy learning and is not a continuation. Retain failed evidence; no automatic retry following nonfinite policy outputs or source/schema mismatch.

## Why other adapters are not immediate substitutions

- Fed-NGA: existing new component only implements the normalized-gradient equation. Current image worker uploads multi-step Adam deltas; faithful integration needs same-global-point true gradients, proper sample averaging and attack semantics for gradient messages. Delta/LR is not equivalent.
- LoGoFair: current centralized evaluation has no per-evaluation-sample client IDs. The official DP adapter needs a frozen client mapping and independent local calibration data; beta-calibration dependency is not yet tested. Current global two-threshold branch cannot populate its official row. EO upstream defects are documented in its existing audit.

Neither is blocked by insufficient compute; each requires a concrete protocol/interface decision and its own real-data gate.
