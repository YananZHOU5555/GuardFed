# CelebA baseline screen v1 — frozen 2026-09-28

User-authorized continuation of baseline completion. This bounded search adds two faithful, explicitly disclosed adapters; it does not complete all 17 manuscript methods. Previously accepted results and configurations remain immutable.

## Scope and selection

64 new useful-70-round jobs: FedAA-DDPG-adapted-v1 and LASA-official aggregation adapted to local Adam deltas, each 8 candidates × IID(alpha5000)/non-IID(alpha5) × Benign/S-DFA, shared seed91001. Full train162770, official valid19867, fixed final checkpoint. Eight concurrent processes distributed across the two current RTX5090 GPUs. No test labels, checkpoint/seed cherry-picking, early-stopping selection, automatic retry or automatic expansion to a full multi-seed queue.

FedAA candidates: actor/critic LR jointly {.001,.01}, retained clients {10,16} of20, local Adam LR {.0005,.001}. Official DDPG source pinned to 1fba884934cbec1b3506812d9d612e6a181b1d79. Clean train-root reward, CPU policy, ascending client-ID ordering, virtual bootstrap/useful-round timing, zero-distance guard, pending final transition and keep16 variation are disclosed adaptations. Perfect-root terminal events are preserved as failures, never silently altered.

LASA candidates: sparsity {.3,.5}, lambda_n=lambda_s {1,2}, local Adam LR {.0005,.001}. Official aggregation pinned to 8477367a4e8708cde264f7572805040c650af59f; model-delta inputs, CPU median for deterministic selection and zero-overall-norm guard are disclosed. Preserve official strict mask, norm/sign filters and fallback behavior. No fabricated full-paper fidelity claim for shared local optimization.

Every method reports raw/native final-round ACC, AEOD and ASPD. No additional calibration is fitted in this search. The existing 700-checkpoint common-calibration experiment is a separate attribution experiment, not merged as duplicate training seeds.

For each candidate, compute the unchanged score per scenario:
`ACC - .35*(.45*AEOD + .45*ASPD + .10*max(AEOD,ASPD)) - .10*max(0,max(AEOD,ASPD)-.06)`.
Select ONE configuration per method by equal mean score over all four scenarios; break exact ties by ascending candidate string. Also publish each method's mean-accuracy champion (same tie rule), three-metric Pareto set (maximize ACC, minimize both gaps), every candidate and every individual result. Missing/failed candidates are explicitly incomplete, never dropped to create a favorable comparison. Selection requires all64 strict acceptances; interim rankings are descriptive only. n=1 seed; four scenarios are not independent seed samples. No sample SD, significance, universal superiority or expected-performance claim.

## Gates, identity and recovery

Four new 3-round real-image gates (two methods × Benign/S-DFA at default non-IID settings) must exactly reproduce earlier pilots' model tensors, trajectories, diagnostics and attacks. FedAA additionally compares full controller/replay/optimizer/RNG state. Gate IDs/directories are separate, never counted among64. Earlier FedAA round1-to3 exact resume gates also remain preserved.

Freeze all64 job files, protocol, worker/adapter/official source, original core and data hashes in manifest before launching. Result acceptance verifies complete1..70 rounds, seed91001, exact config/real alpha/attack, valid split/denominators, source identities, finite final metrics and final checkpoint hash. FedAA acceptance additionally checks full controller transitions/state and first/final checkpoint identity. LASA acceptance checks original job, diagnostics, provenance and its immutable acceptance receipt. All metrics belong to one final checkpoint.

The queue stops new dispatch after a failed worker or failed acceptance, lets already active workers finish, and preserves evidence. No automatic numerical retry. Existing nonempty partial directories are rejected. Only after diagnosing an external interruption and verifying source/data/protocol identity, absence of duplicates and strict accepted-result skip behavior may a bounded recovery be prepared. FedAA may resume identical full-state checkpoints; LASA has no mid-round recovery guarantee. Do not restart old formal/fullcoverage queues or alter driver/instance settings.

Incrementally back up source/protocol/gates immediately and accepted results, checkpoints and logs at key batches; verify archive SHA and member hashes off-server. At64 accepted, generate the complete candidate summary, verify off-server backup and pause this stage's monitor. Remaining methods, full IID/non-IID five-attack multi-seed confirmation, mechanism ablations and frozen final evaluation remain separate work; no automatic unreviewed new protocol.
