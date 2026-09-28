"""Derive two explicit pipeline gates from an immutable Stage A job config."""
import argparse
import hashlib
import json
from pathlib import Path

from run_fedaa_pilot import ADAPTATIONS, validate_job


def prepare(base_path, out):
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if base["dataset"] != "celeba" or base["distribution"] != "non-IID" or base["config"]["seed"] != 91001:
        raise ValueError("Expected seed91001 non-IID Stage A image job")
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for attack in ["Benign", "S-DFA"]:
        ident = f"FedAA-DDPG_non-IID_{attack}_seed91001_gate3"
        cfg = dict(base["config"], learning_rate=.001, rounds=3,
                   ad2_calibration_enabled=False, full_round_diagnostics=True,
                   experiment_suite="fedaa_image_gate_v1", experiment_tag=ident)
        job = dict(id=ident, dataset="celeba", distribution="non-IID",
                   method="FedAA-DDPG-adapted-v1", attack=attack, config=cfg,
                   policy_seed=91001, aggre_num=10, evidence_stage="pipeline_canary_only",
                   source_hashes=base["source_hashes"], adaptations=ADAPTATIONS,
                   derivation=dict(base_job_id=base["id"], base_job_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
                     base_source_output=base["output"],
                     changes={k: dict(before=base["config"].get(k), after=v)
                              for k, v in cfg.items() if base["config"].get(k) != v},
                     attack_override=dict(before=base["attack"], after=attack)))
        validate_job(job)
        path = out / f"{ident}.json"
        text = json.dumps(job, indent=2, ensure_ascii=False)
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Frozen job differs: {path}")
        path.write_text(text, encoding="utf-8")
        jobs.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    print(json.dumps(dict(jobs=jobs, training_started=False), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-job", type=Path, default=Path(__file__).with_name("base_job.json"))
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("jobs"))
    args = parser.parse_args()
    prepare(args.base_job, args.out)
