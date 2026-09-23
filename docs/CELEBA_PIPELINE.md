# CelebA image pipeline

This image worktree adds a CNN and image data adapter to the existing AD2+ implementation. It does not change the candidate bank, candidate selection, score terms, hard gate, root update, threshold calibration, client reweighting, attack definitions, or tabular numerical path.

## Data and cache

- Official target: Smiling; sensitive attribute: Male. Male remains metadata, not an image input.
- Official train/valid/test contain 162770/19867/19962 images. Root and clients come only from train. Validation and test never enter local/root training.
- HF shard rows are identity-sorted, not official image order. Cache uses image.path numeric stem minus one, and checks every row's official split and all 40 attributes.
- Cache: data/celeba/derived/rgb64_v1/images.npy, uint8 NCHW [202599,3,64,64], deterministic Pillow RGB bilinear resize, no crop/augmentation.
- metadata.npz stores official numeric IDs, split, Smiling, Male. Cache manifest records source-shard hashes, official metadata hashes, complete coverage and derived-file hashes.
- Preprocessing uses the independent /tmp/guardfed-dataset-fetch-venv/bin/python with Arrow/Pillow/NumPy. The shared training environment was not changed.
- Rebuild/resume: run scripts/build_celeba_cache.py --root data/celeba --workers 8 in that preprocessing environment. Incomplete caches are rejected by the runtime.

## Model and memory

Conv(3,32,3,padding1), ReLU, MaxPool2; Conv(32,64,3,padding1), ReLU, MaxPool2; Conv(64,128,3,padding1), ReLU, MaxPool2; global average pool; Linear(128,2). No BatchNorm, dropout or augmentation. There are 93,506 parameters.

Pixels are CPU uint8. Client/root/evaluation partitions use compact CPU uint8 tensors, and one batch is copied/cast to GPU float32 and divided by 255 in CNN.forward. Labels and sample weights move per batch. No all-dataset float32 GPU allocation is made.

The full train+test pixel partitions total approximately 2.09 GiB of CPU uint8 data per job, plus array/metadata/temporary indexing overhead. The mmap file itself is approximately 2.32 GiB on disk. A synthetic single 256-image training batch measured peak torch allocated GPU memory 1,250,659,840 bytes on RTX 5090. This is a single-batch measurement, not a complete-experiment memory ceiling.

## Pilot versus formal identity

ExperimentConfig adds celeba_train_limit and celeba_eval_limit (0 means full official split), celeba_evaluation_split (valid or test), and optional celeba_cache_dir. A subset is sampled deterministically from the complete official split by seed; it is not the first identity-sorted HF rows.

Use valid during learning-curve/model-selection pilots, keeping test for the frozen final comparison. Run IDs and data_contract.image_data_contract record subset limits, selected evaluation split, selected image-ID hashes, cache hash and pilot_subset/full_official_split status. Dataset is 'celeba'; distribution and client_alpha retain the existing client-partition meaning.

The common scripts/run_revision_ablation.py worker accepts a job with dataset='celeba'. Materialize config with asdict(ExperimentConfig(...)), use a separate output directory, and generate source_hashes only after the complete cache is fixed. The image worker hash set includes the CNN/data adapter, cache builder, official metadata and actual cache files. The core reproduction CLI retains its existing tabular defaults.

No image model was trained on real data during implementation. The parent task will separately launch the benign validation learning-curve pilot.

## Checks

scripts/check_celeba_pipeline.py runs a small synthetic CPU check:

- A complete AD2+ tabular round matches frozen b1a808b exactly at checkpoint tensors, reported metrics and per-round diagnostics.
- The image CNN produces two logits and all ten adaptive candidates plus threshold reporting run successfully.
- F Flip leaves image pixels and target labels unchanged, changes sensitive metadata/reweighting, and changes a local optimization step.

deployment/celeba_checks also records a real 512-train/128-valid loading/evaluation-only check and the synthetic GPU memory measurement. These checks are not scientific experimental results.
