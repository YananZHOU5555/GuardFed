# Gradient fullcoverage new-attack gate — SOURCE PREPARED ONLY

No candidate has been selected, no real job bound, and no model or image operation run by this package. The existing 64-job screen and its 70-round strict checker are unchanged. This package only prepares the missing attack-interface check before a separately authorized 192-new/8-reference fullcoverage stage.

## Exact future scope

For each of Fed-NGA-gradient and Huber-BRFL-gradient, use its **actual complete-64, root-adopted selected recipe**. Fix non-IID alpha=5, model seed=91002, CPU strict FP32, eight CPU threads, one fresh process at a time, three rounds, all 162770 train / 16277 clean root / 19867 validation images. For each of F Flip, FedSA and Sp-DFA run the pinned screen worker and derived coverage worker on identical inputs. Add one screen-worker Benign reference per method. This is 7 jobs per method, 14 jobs total and 42 rounds; no job contributes a formal table cell. The prior Benign/S-DFA four-image gate is not repeated.

The six screen/coverage pairs require exact saved tensors, three-round metrics/diagnostics, gradient-upload SHA oracles, native checkpoint replay and saved Python/NumPy-global/Torch-CPU RNG fingerprints. The two F Flip/Benign comparisons additionally check the frozen metadata-only F Flip null: identical model, metrics and numerical gradient audit, while attack assignment labels remain different. F Flip changes neither Smiling labels nor CNN inputs in this unweighted CE bridge; this is a limitation, not demonstrated robustness to an input/label attack.

FedSA uses FOE on malicious clients 0..3. Sp-DFA assigns metadata F Flip to 0..1 and FOE to 2..3. Each FOE-bearing round uses the existing clean-root local-Adam descent reference once; that reference is adversary input, never defense fairness/calibration. The original sign-conjugacy and norm-cap checks remain unchanged. Huber uses the accepted identity projection in R^p and the original solver stopping rule; a convergence or oracle failure is retained and stops the gate, without increasing tolerances or iterations.

## Reuse and boundary

`metadata.py` imports the pinned, already reviewed coverage `prepare.py` for complete-64 selection and 96+4 identity checks. It requires the **actual** complete-64 summary/root receipt and an external root approval binding both frozen coverage manifests. No future SHA or selected recipe is supplied here. Coverage component files must already be installed with their exact original SHA and each selected protocol must be FROZEN; prepared coverage protocols are rejected. Binding writes only a separate exploratory scope and 14 job configurations.

`adapter.py` reuses the original real-image `gate.py` at run time, with exact source replacements shown in `SOURCE_DIFF.patch`: seed expectation, root-call expectation for the three FOE scenes, explicit attack assignments, and evidence-stage wording. All 14 whole scientific worker functions remain byte-identical. The six nested gradient/data/sign routines remain byte-identical; the aggregate oracle changes only its expected root-call predicate. The original formal `accept_result.py` still requires 70 rounds and is never used to accept these three-round canaries. No scientific source file is copied or edited.

The original exact six shared-cache target registrations are reused, without broadening path escape permissions. Stage-local and external source/job/summary/approval identities are checked before scientific imports and after execution. Existing/partial outputs and per-job attempt files are not overwritten. Original failures and an outer failure receipt remain in place. The comparison refuses any retained failure.

## Future commands — not authorization

Keep the dependency directory layout relative to the project root; run the metadata binder on the eventual server so absolute dependency paths are real there. A root-approved frozen coverage release must be produced first, including refreshed job-local hashes after protocol freezing. This package neither performs that freeze nor starts a queue.

```text
python metadata.py --bound <actual_frozen_192_root> --stage-approval <actual_root_source_approval.json> --approval-sha256 <actual_sha> --out /workspace/guardfed_checks/gradient14_gate/<new_stage>
```

The source approval contract is `status=ROOT_GRADIENT200_FROZEN_SOURCE_ADOPTED`, `test=false`, `bound_inputs_sha256`, and `manifest_sha256` for the two method names. `BOUND_INPUTS.json` comes from the original coverage binder and points to the actual accepted-64 summary/root receipts. This new gate checks the original four references of each method exactly against that summary.

Root must then inspect the bound scope and actual resources. Each fresh child needs an external receipt with `status=ROOT_APPROVED_GRADIENT14_CPU3_CANARIES_ONLY`, exact `scope_sha256` and ordered job-ID/SHA map, `measured_unix` no more than 120 seconds old, eight allocated CPU IDs, actual CPU quota/RAM, no restricted-affinity owner overlap, no duplicate gate worker, old screen64 exited, and protected main-queue health/growth (or verified completion). `test_authorized`, `coverage192_authorized` and `automatic_retry` must all be false. Broad scheduler affinity is not an exclusive CPU reservation. Root must additionally verify live guide, all thread owners, actual I/O class, GPU health and aggregate resource budget; prepared metadata checks are not a live preflight.

```text
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c <approved_8_cpu_ids> nice -n 10 ionice -c 3 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python adapter.py --stage <stage> --repo /workspace/GuardFed-celeba-expanded --job <one_exact_id> --approval <fresh_actual_receipt> --approval-sha256 <sha>
python compare_saved.py --stage <stage> --repo /workspace/GuardFed-celeba-expanded
```

Run entries only in their bound order, fresh child per entry, stopping at the first failure. No queue/service/monitor implementation is added here. Saved comparison loads tensors and existing JSON only; it does not perform another CNN pass. Native replay inside the original gate remains part of each authorized image gate. A successful CPU comparison does not establish CPU/GPU equivalence, 70-round behavior, all RNG-generator state equivalence, scientific superiority, or permission to launch 192 jobs. Root adoption, offserver backup and actual GPU/runtime review remain necessary.

Code and compact records may live on E. Any future local model/array/archive/log transport must use a separately verified `F:/YananResearchStorage/GuardFed/` destination (volume label `Yanan 2TB`), with no E fallback. This source-only package writes no bulk artifact.

## Actual local verification

`python check_prepared.py` checks whole scientific-source equality, exact14 metadata and four attack assignments; it does not import Torch or NumPy, select from actual results, bind jobs or run images. Its candidate examples are explicitly metadata fixtures. `SELF_CHECK.json` and the source diff are evidence of preparation only; actual accepted short gates remain zero.
