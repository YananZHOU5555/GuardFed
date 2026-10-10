"""Generate gradient jobs from a draft or already approved frozen protocol.

Never change protocol status, approve a decision, train or launch a queue.
"""
import argparse
import itertools
import json
from pathlib import Path

from worker import COMPONENTS, HERE, LOCAL_SOURCES, digest, validate_job, write_json


def prepare(output):
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf8"))
    if protocol["status"] not in {"PREPARED_NOT_FROZEN", "FROZEN"}:
        raise ValueError("Expected a draft or independently reviewed frozen protocol")
    if protocol["status"] == "FROZEN" and any(row.get("status") != "APPROVED" for row in protocol["protocol_decisions"].values()):
        raise ValueError("Frozen protocol still has unresolved decisions")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    local_hashes = {name: digest(HERE / name) for name in sorted(LOCAL_SOURCES)}
    records = []
    for candidate, distribution, attack in itertools.product(protocol["candidates"], protocol["distributions"], protocol["attacks"]):
        identity = candidate["id"] + "_" + distribution + "_" + attack + "_seed91001_screen"
        config = dict(protocol["base_config"], seed=91001, rounds=70,
            client_alpha=protocol["distributions"][distribution],
            use_reweighting=candidate["adapter"]["objective"] == "shared_reweighted_ce",
            experiment_suite=protocol["version"], experiment_tag=identity)
        job = dict(id=identity, dataset="celeba", method=candidate["method"], distribution=distribution, attack=attack,
            evidence_stage="validation_screen", config=config, adapter=candidate["adapter"], tuning_candidate=candidate["id"],
            source_hashes=protocol["source_hashes"], component_hashes=COMPONENTS, local_hashes=local_hashes)
        validate_job(job, protocol, require_frozen=False)
        path = output / (identity + ".json")
        write_json(path, job)
        records.append(dict(id=identity, method=job["method"], job=path.name, job_sha256=digest(path)))
    assert len(records) == len({row["id"] for row in records}) == 64
    write_json(output / "manifest.json", dict(status=protocol["status"], execution_started=False,
        protocol_sha256=digest(HERE / "protocol.json"), jobs=records,
        limitation="Frozen validation-only 64-job search; resource preflight and root dispatch remain required; n=1, no test or automatic retry"))
    print(json.dumps(dict(status=protocol["status"], prepared_jobs=len(records))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "screen_jobs_draft")
    prepare(parser.parse_args().out)
