"""Materialize the parent's reviewed FLGMM protocol; never run or freeze jobs."""
import argparse
import itertools
import json
from pathlib import Path

from worker import HERE, METHOD, digest, validate_job, write_json


def prepare(output):
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf8"))
    if protocol["status"] not in {"PREPARED_NOT_FROZEN", "FROZEN"}:
        raise ValueError("Unexpected protocol review status")
    adapter_sources = ["flgmm_adapter.py", "worker.py", "prepare_jobs.py", "protocol.json",
                       "sources/flgmm_pinned.py", "sources/flgmm_fedavg.py", "sources/flgmm_license.txt"]
    hashes = {name: digest(HERE / name) for name in adapter_sources}
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for candidate, distribution, attack in itertools.product(
            protocol["candidates"], protocol["distributions"], protocol["attacks"]):
        identity = f"{candidate['id']}_{distribution}_{attack}_seed91001_screen"
        config = dict(protocol["base_config"], rounds=70, seed=91001,
                      learning_rate=candidate["learning_rate"],
                      client_alpha=protocol["distributions"][distribution],
                      experiment_suite=protocol["version"], experiment_tag=identity)
        job = dict(id=identity, dataset="celeba", method=METHOD, distribution=distribution,
                   attack=attack, phase="screen", evidence_stage="validation_screen",
                   config=config, adapter=candidate["adapter"], tuning_candidate=candidate["id"],
                   source_hashes=protocol["source_hashes"], adapter_source_hashes=hashes)
        validate_job(job, protocol, require_frozen=False)
        path = output / (identity + ".json")
        write_json(path, job)
        jobs.append(dict(id=identity, job=path.name, job_sha256=digest(path),
                         method=METHOD, tuning_candidate=candidate["id"], distribution=distribution, attack=attack))
    assert len(jobs) == len({item["id"] for item in jobs}) == 32
    manifest = dict(status=protocol["status"], execution_started=False, method=METHOD,
                    protocol_sha256=digest(HERE / "protocol.json"), jobs=jobs,
                    next_action="Parent real-image gate and protocol review/freeze before any launch")
    write_json(output / "manifest.json", manifest)
    print(json.dumps(dict(status=manifest["status"], jobs=len(jobs), output=str(output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "screen_jobs_draft")
    prepare(parser.parse_args().out)
