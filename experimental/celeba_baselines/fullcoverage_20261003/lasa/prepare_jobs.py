"""Prepare 100 fixed-recipe coverage jobs and 5 separate canaries; no training.

The parent excludes explicitly reused accepted records from its new-run queue.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_FILES = ["adapter.py", "worker.py", "protocol.json", "prepare_jobs.py"]


def make_jobs(phase):
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in CODE_FILES}
    if hashes["adapter.py"] != protocol["upstream_adapter_sha256"]:
        raise ValueError("adapter differs from the passed real-image pilot")
    candidates = protocol["candidates"]
    distributions = protocol["distributions"]
    seeds = protocol["seeds"]
    if phase == "preflight":
        candidates = [v for v in candidates if v["id"] == protocol["preflight_candidate"]]
        distributions = {protocol["preflight_distribution"]: protocol["distributions"][protocol["preflight_distribution"]]}
        seeds = [protocol["preflight_seed"]]
    elif phase != "fullcoverage":
        raise ValueError("unknown phase")
    jobs = []
    rounds = protocol["coverage_rounds"] if phase == "fullcoverage" else protocol["preflight_rounds"]
    for candidate in candidates:
        for distribution, alpha in distributions.items():
            for attack in protocol["attacks"]:
                for seed in seeds:
                    identity = f"{candidate['id']}_{distribution}_{attack}_seed{seed}_{phase}"
                    config = copy.deepcopy(protocol["base_config"])
                    config.update(seed=seed, rounds=rounds, learning_rate=candidate["learning_rate"], client_alpha=alpha,
                                  experiment_suite=protocol["version"] + "_" + phase, experiment_tag=identity)
                    jobs.append(dict(id=identity, dataset="celeba", distribution=distribution,
                        method=protocol["method"], attack=attack, phase=phase, config=config,
                        adapter=copy.deepcopy(candidate["adapter"]), tuning_candidate=candidate["id"],
                        source_hashes=copy.deepcopy(protocol["source_hashes"]), adapter_source_hashes=hashes,
                        evidence_stage=protocol["evidence_stage"] if phase == "fullcoverage" else "pipeline_canary_only",
                        message_semantics="frozen-core local Adam model deltas after attack injection",
                        reporting=protocol["reporting"]))
    return jobs


def write_frozen(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError("refusing to overwrite a different frozen job: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "jobs")
    args = parser.parse_args()
    counts = {}
    for phase in ("fullcoverage", "preflight"):
        jobs = make_jobs(phase)
        counts[phase] = len(jobs)
        for job in jobs:
            write_frozen(args.output / phase / (job["id"] + ".json"), job)
        write_frozen(args.output / (phase + "_manifest.json"), dict(
            method="LASA-official", phase=phase, expected_jobs=len(jobs), jobs=jobs,
            status="prepared_not_started", dataset="celeba", evaluation_split="valid"))
    print(json.dumps({"coverage_jobs": counts["fullcoverage"], "preflight_jobs": counts["preflight"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
