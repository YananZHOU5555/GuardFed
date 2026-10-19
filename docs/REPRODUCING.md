# Reproducing experiments

## Data

Provide the following paths under the repository root:

```text
data/adult/adult.data
data/adult/adult.test
data/adult/adult.names
data/compas/compas-scores-two-years.csv
data/celeba/derived/rgb64_v1/images.npy
data/celeba/derived/rgb64_v1/metadata.npz
data/celeba/derived/rgb64_v1/available.npy
data/celeba/derived/rgb64_v1/manifest.json
```

Fixed tabular input URLs and SHA256 values are in [DATA_SOURCES.json](DATA_SOURCES.json). Data are not bundled.

CelebA uses Smiling as the target and Male as sensitive metadata. The official train / valid / test splits contain 162770 / 19867 / 19962 images. Root and clients use train; the released image templates evaluate valid. The CNN receives RGB images without the sensitive attribute as an input feature.

To build the complete RGB64 cache, provide `data/celeba/list_attr_celeba.txt`, `list_eval_partition.txt` and Parquet shards under `hf_flwrlabs/img_align+identity+attr/`, then run:

```bash
python -m pip install -r requirements-image.txt
python scripts/build_celeba_cache.py --root data/celeba --workers 8
```

The builder checks official image IDs, splits and attributes, uses Pillow RGB bilinear resize to 64×64 without augmentation, and records cache hashes. It does not download source data. Keep large data, caches, checkpoints and logs on suitable research storage or the server. Directory links can preserve the expected project paths.

## Run one fixed configuration

```bash
python scripts/run_experiment.py \
  --job configs/celeba_smallalpha/GuardFed-AD2+.json \
  --output /research-storage/GuardFed/runs/full_alpha1_seed91013
```

This command performs training. The output path must be new; existing scientific outputs are never overwritten. The three provided templates contain all 63 server configuration fields. They describe a frozen alpha=1, Benign, seed91013 member of the small-alpha cohort, not a completed 180-run result. Dataset, distribution, attack, seed and the complete configuration are recorded with the terminal checkpoint.

For other jobs, specify the intended `ExperimentConfig` explicitly. The old dataclass default image evaluation split is `test`; the release runner requires explicit `valid` for its image protocol. The reference tabular preprocessing includes a legacy COMPAS option whose scaler fit scope differs from `train_only`; reproduce the applicable configuration rather than silently changing it.

`run_experiment` preserves server training, attacks, model calibration and random-number consumption, including trajectory diagnostics. All three terminal metrics come from the saved terminal model and the method's defined postprocessing. A one-round installation check is not a paper result.

## Aggregate a complete cohort

Concatenate the per-run `raw_results.jsonl` records from one experiment cohort into a JSONL input without editing the records, then run:

```bash
python scripts/summarize_results.py \
  --input /research-storage/GuardFed/cohort.jsonl \
  --manifest configs/celeba_smallalpha/manifest.json \
  --output /research-storage/GuardFed/summary.csv
```

The manifest declares the fixed ten seeds and 70 rounds. Every configuration group must contain exactly those seeds. Different hyperparameters form separate groups; missing, extra or duplicate seeds and incomplete trajectories are rejected. The terminal trajectory and record metrics must agree. The output reports each metric's `n_total`, `n_defined`, mean and sample SD (`ddof=1`) in the input's fractional units. Undefined metrics retain their denominator counts.

The supplied seed manifest is for the small-alpha cohort. Other historical cohorts have different prescribed seeds and need their own recorded manifest. Do not substitute a new manifest to reinterpret an old experiment.

## Tests and limits

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
python baselines/fednga/fednga.py
python baselines/flgmm/check_flgmm.py
```

The package includes metric and strict-summary tests independent of data or Git history. Historical regression tests need reference commits `ec419b7` / `9e62b78`; the COMPAS regression also needs its CSV. They skip if those inputs are absent. CUDA determinism needs CUDA. A full historical clone may be large; obtain it on suitable research storage when needed.

Source/metric correspondence is recorded in [VERIFICATION.json](VERIFICATION.json). Component checks do not establish end-to-end GPU parity for every baseline, and no new training campaign was run for this cleanup. Published validation evidence retains its model-selection and historical test-exposure limitations.
