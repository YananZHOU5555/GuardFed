# CelebA numerical execution protocol

The earlier image pilots used cudnn.benchmark=True. Matched seed123 repeats had identical data/configuration but different first-round adaptive candidate/selected-client decisions and final weights. They remain exploratory learning evidence; no formal test comparison was launched under those settings.

This independent source version enforces strict deterministic algorithms for CelebA only, disables cuDNN benchmark and both cuDNN/matmul TF32, sets cuDNN deterministic and CUBLAS_WORKSPACE_CONFIG=:4096:8 before worker CUDA use. Both dataset setup and the later run-level reseeding preserve these flags. Unsupported deterministic operations fail rather than silently fall back. Each image data contract records the execution settings.

Architecture, data partitions, batch64, Adam.001, candidate table, AD2+ aggregation, attacks, calibration and final70-round reporting remain unchanged. Historical/tabular worktrees and results are untouched. New images must freeze this source hash, pass matched cross-GPU first-round tensor/candidate checks and3-round concurrency checks before formal launch. This controls observed execution variability, not cross-version/hardware reproducibility in general. All failed/weak pilots remain separate.
