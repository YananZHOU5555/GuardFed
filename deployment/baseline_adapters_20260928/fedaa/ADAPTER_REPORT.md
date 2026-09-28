# FedAA official DDPG integration audit — 2026-09-24

Status: component prototype verified on CPU; not yet a CelebA end-to-end reproduction.

## Source and deliverables

- Official repository: https://github.com/Gp1g/FedAA
- Pinned commit: `1fba884934cbec1b3506812d9d612e6a181b1d79`.
- DDPG.py SHA256: `e9bdbccdaf3aba7e1eece10370cbf7cedfe66104a691bc24a9d4e1300cd12d68`.
- No LICENSE/COPYING file was found in the tracked repository. The new adapter loads a separately obtained, hash-verified upstream checkout; do not silently copy/relicense that source into GuardFed.
- `fedaa_official_adapter.py`: new wrapper, no replacement actor/critic implementation.
- `check_adapter.py`, `verification.json`: runnable CPU checks and measured outcome.
- Existing `fedaa_aggregate` in GuardFed is a proximity + softmax reward heuristic, with no policy learning; do not label its output as this official FedAA implementation.

## Fixed algorithm features

Official actor and critic each use 256/256 hidden units. Actor adds Gaussian noise (std 0.1) inside every forward pass, including action selection and target actor. Its nonnegative output is abs + L1 normalization, not softmax. DDPG discount .99, target tau .001; Adam actor/critic learning rates .01 and weight decay 1e-5. Upstream CLI help says .001 but actual defaults are .01; record actual values.

Replay batch16, sampled with replacement even for the first transition; one update per round. Upstream buffer default100000; keep it or document a capacity reduction (70 transitions fit either way). There is one episode by default (`episodes=0`, loop `episodes+1`). No pretrained policy.

State: pairwise L2 distances among full local model parameter vectors; sum each row; select k smallest sums; divide selected sums by their maximum. Upstream defaults select100 clients and aggregate50, i.e. 50% retention. For 20 participants, k10 preserves that ratio; k16 is a separate setting, not the official default. State and action slots follow the selected-order ranking, not permanent client identities.

Reward: scalar accuracy of the aggregated global model on server data; no AEOD/ASPD reward, no root-gradient alignment reward. For this project, supply a train-derived clean root set, never official valid/test. Upstream uses a file named `server_test`; transferring that name literally to CelebA's official test split would leak data. Global mixing alpha defaults0, so weighted local-model averaging is exact.

## Minimal timing integration

Create one `OfficialFedAA(...)` per independent FL run. Do not instantiate it inside a stateless `aggregate_round` call. Maintain `(previous_state, previous_action, previous_reward)` across rounds and include it in any resumable checkpoint.

Official source ordering:

1. Initialize global model; all clients initially receive this same model. Initial state is all ones and initial k clients are randomly selected. Compute initial actor action; aggregation here combines identical initial models (a no-op mathematically).
2. Measure root accuracy of that aggregated model, then train the first local cohort, applying the project's attacks.
3. Compute next state / selected IDs from that cohort. Add `(initial state, initial action, next state, initial root accuracy, done)` and update DDPG once.
4. At the next decision, choose action using current state; aggregate the selected local models; measure root accuracy. Train the next local cohort; compute next state; learn the transition. Repeat.

Two possible integrations must be explicitly distinguished:

- **Literal upstream timing:** preserve initial no-op aggregation, count round0 accordingly. Its last local cohort is trained but never aggregated. This does not match GuardFed's meaning of70 completed local-train/aggregate rounds.
- **Common 70-useful-update adaptation (recommended for a new, explicit protocol):** perform a virtual initialization transition before the first useful aggregation; each of70 rounds performs local training then selected aggregation. At round r>1, use current local cohort's state to finish learning previous round's transition **before** choosing current action. At the last round keep the final pending transition in the checkpoint; do not fake next_state or train an extra cohort. This yields70 useful aggregations,70 policy updates including bootstrap, and an explicitly disclosed one-round alignment adaptation. The official initialization action is sampled from ones; initial reward is initial-model root accuracy.

Compute state from the same full local models received by the aggregator (after attacks). Pairwise differences of deltas are algebraically equivalent but floating-point translation can change ties; passing full parameters avoids claiming bitwise equivalence where there is none. Upstream randomly permutes all clients before distance ranking; fixed ID order is a disclosed tie-order convention if adopted. For ties, explicitly fix and record ordering.

## Determinism and persistence

Policy computation stays on CPU; this is a deliberate device adaptation. It leaves actor stochasticity intact and isolates policy torch/NumPy RNG from training RNG. The upstream replay buffer auto-selects CUDA independent of policy device; wrapper sets it to CPU to match policy.

Do not call `eval()` expecting deterministic actor output: upstream noise remains active in eval. Keep policy seed in manifest, one stream per run. Checkpoint actor, critic, both targets, both optimizers, replay contents/pointer, RNGs and transition counter using wrapper state_dict. Also save pending FL transition, current selected client IDs/state, global model and external training RNG states. Upstream save/load alone omits replay and RNGs and overwrites target networks; it is not exact resume.

The wrapper's only numerical guard is all-zero distance sums -> zero state (upstream produces NaN). This guard must be disclosed. Do not change actor outputs, distance metric, reward shaping or update count to improve benchmark performance under the FedAA name.

## Measured checks and missing gates

Python3.10 / torch2.8.0+cpu: action and one complete learning step match the directly imported upstream module bitwise for actor, critic and both targets. Actor parameters change, four further steps after state restore match bitwise, surrounding RNG is unchanged, outlier selection and zero-distance handling pass. Command: `python tmp/celeba_baselines/fedaa/check_adapter.py`.

Still required before queueing long experiments: integrate caller timing, verify no official valid/test in reward, run >=3 useful CPU/GPU image rounds and ensure finite weights/losses, policy transitions grow and all client IDs/config/checkpoint identity are recorded. A 2-round task that only confirms a process exits does not establish correct DDPG learning. No server code or training jobs were changed by this subtask.
