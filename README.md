# GuardFed

Fairness-aware Byzantine-robust federated learning research code for Adult, COMPAS and CelebA.

This release separates the shared training engine, experiment execution, terminal-result aggregation and baseline operators. The training and metric functions preserve the server implementation. Data, checkpoints, development searches, machine launchers and generated reports are distributed separately.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_experiment.py --help
python scripts/summarize_results.py --help
```

Python 3.10 or later is required. On Windows, activate `.venv\Scripts\Activate.ps1`. GPU experiments require a compatible PyTorch build. Image-cache preparation additionally uses `requirements-image.txt`.

## Code

| Component | Location |
|---|---|
| Training, attacks, AD2 / AD2+ and metric definitions | [scripts/reproduce_paper_tables.py](scripts/reproduce_paper_tables.py) |
| Adult / COMPAS loading | [src/data_loader.py](src/data_loader.py) |
| CelebA CNN and data contract | [src/celeba_data.py](src/celeba_data.py) |
| One configured experiment | [scripts/run_experiment.py](scripts/run_experiment.py) |
| Complete-cohort terminal statistics | [scripts/summarize_results.py](scripts/summarize_results.py) |
| Additional baseline operators and third-party sources | [baselines](baselines) |
| Data, execution and reproducibility | [docs/REPRODUCING.md](docs/REPRODUCING.md) |
| Baseline interfaces and scope | [docs/CODE_MAP.md](docs/CODE_MAP.md) |
| Server correspondence | [PROVENANCE.json](PROVENANCE.json) |

## Evaluation

The reported metrics are accuracy, absolute TPR difference (`aeod` in the implementation) and absolute positive-prediction-rate difference (`aspd`). Values are fractions on [0, 1]. Undefined denominators remain undefined and generate diagnostics.

An experiment saves one terminal checkpoint and its three metrics. The separate summarizer requires the complete seed list and fixed round count in a manifest, keeps different configurations separate, and reports the mean and sample standard deviation. Missing or duplicate runs are errors. Model calibration and aggregation-internal candidate selection remain part of the algorithms.

The included CelebA templates are copied from three frozen server jobs for the small-alpha stage. They are configurations, not a claim that this stage has completed. CelebA evidence discussed with this project is validation evidence; historical model selection and test exposure prevent a claim of an untouched final test.

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

Some historical regression tests require Git reference commits and the COMPAS CSV; they skip when those inputs are absent. CUDA-specific checks require CUDA. Verification scope and measured agreement are recorded in [docs/VERIFICATION.json](docs/VERIFICATION.json).

Third-party notices remain with their source files. A project-wide release license has not been assigned; do not assume that one dependency's license applies to the whole repository.
