"""Read-only metadata bridge; no dispatch, inference, fitting or Torch import."""
from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import types

import numpy as np

HERE = Path(__file__).resolve().parent
REUSE_SHA256 = "36d686bf467b0ed4b5df489acf229f2ce56f967d5549225ba582992b5262cdef"
PROOFS_SHA256 = "b42f5766254abdcd3064cb5621285fac3b66012c4d330b62547b8ab5baee1912"
METHODS = frozenset({"FLGMM", "CosineFairnessHybrid", "Fed-NGA-gradient", "Huber-BRFL-gradient"})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_pin(pin):
    path = Path(pin["path"])
    require(digest(path) == pin["sha256"], "Pinned bytes changed: " + str(path))
    if "bytes" in pin:
        require(path.stat().st_size == pin["bytes"], "Pinned size changed")
    return json.loads(path.read_text(encoding="utf-8"))


def science_bindings(*, torch_module=None, pandas_module=None):
    """Original function AST, separate globals. Caller supplies actual modules later.

    This only constructs functions. It does not call root reconstruction, the
    shared fitter, model_margins, or an original GPU strict checker.
    """
    manifest = read_pin({"path": str(HERE / "SOURCE_REUSE.json"), "sha256": REUSE_SHA256})
    require(digest(manifest["original_core"]["path"]) == manifest["original_core"]["sha256"],
            "Original core source changed")
    ns = {"__builtins__": __builtins__, "np": np, "pd": pandas_module,
          "torch": torch_module, "dataclasses": dataclasses, "hashlib": hashlib,
          "json": json, "types": types, "Path": Path}
    for entry in manifest["function_sources"]:
        pin = entry["file"]
        require(digest(pin["path"]) == pin["sha256"], "Original science source changed")
        source = Path(pin["path"]).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=pin["path"])
        selected = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                selected.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name in entry["functions"]:
                text = ast.get_source_segment(source, node)
                require(hashlib.sha256(text.encode()).hexdigest() == entry["functions"][node.name],
                        "Original function source changed")
                selected.append(node)
            elif isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id in entry["constants"] for t in node.targets):
                selected.append(node)
        require(sum(isinstance(n, ast.FunctionDef) for n in selected) == len(entry["functions"]),
                "Missing original function")
        exec(compile(ast.Module(body=selected, type_ignores=[]), pin["path"], "exec", dont_inherit=True), ns)
    # The sole evaluator adaptation; the original module is never imported or mutated.
    ns["METHODS"] = set(METHODS)
    return types.SimpleNamespace(**{name: value for name, value in ns.items()
                                   if not name.startswith("__")})


def validate_metadata(method, job, result, provenance, acceptance, strict, off, scope=None):
    """Identity checks after byte pins and original external strict/offserver proofs."""
    require(method in METHODS, "Unsupported method (LoGoFair is not a CNN native branch)")
    source_method = "FLGMM-author-code" if method == "FLGMM" else method
    require(job["method"] == result["method"] == source_method, "Wrong method")
    require(job["id"] == strict["id"] == off["id"], "Wrong proof record ID")
    require(result["dataset"] == job["dataset"] == "celeba", "Wrong dataset")
    require(all(result["revision_job"].get(k) == v for k, v in job.items()), "Wrong revision job")
    require(result["config"] == job["config"], "Wrong config")
    for key in ("distribution", "attack"):
        require(result[key] == job[key], "Wrong scenario/candidate")
    candidate = result["revision_job"]["tuning_candidate"] if method == "CosineFairnessHybrid" else result["tuning_candidate"]
    require(candidate == job["tuning_candidate"], "Wrong candidate")
    cfg = job["config"]
    require(type(cfg["seed"]) is int and 91001 <= cfg["seed"] <= 91010
            and result["seed"] == cfg["seed"], "Wrong seed")
    require(cfg["rounds"] == result["rounds"] == 70 and cfg["num_clients"] == 20
            and cfg["num_malicious"] == 4, "Incomplete original 70-round record")
    require(cfg["celeba_evaluation_split"] == "valid" and cfg["celeba_train_limit"] == 0
            and cfg["celeba_eval_limit"] == 0, "Only full official valid is admissible")
    require(result["alpha"] == cfg["client_alpha"] ==
            (5000.0 if job["distribution"] == "IID" else 5.0), "Wrong alpha")
    if method == "CosineFairnessHybrid":
        require(scope is not None and provenance["source_hashes"] == scope["protected_source_hashes"]
                and provenance["local_hashes"] == scope["local_hashes"]
                and all(provenance["source_hashes"].get(k) == v for k, v in job["source_hashes"].items()),
                "Wrong scope source")
    else:
        require(provenance["source_hashes"] == job["source_hashes"], "Wrong source")
        require(result["provenance"] == provenance, "Wrong embedded provenance")
    require(job["source_hashes"]["scripts/reproduce_paper_tables.py"] ==
            "cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed", "Wrong original core")
    for key in ("adapter_source_hashes", "local_hashes", "component_hashes"):
        if key in job:
            require(provenance[key] == job[key], "Wrong adapter source")
    if method in {"Fed-NGA-gradient", "Huber-BRFL-gradient"}:
        require(result["gradient_recipe"] == job["adapter"], "Wrong gradient recipe")
    require(acceptance["status"] == "PASS", "Failed original acceptance")
    require(acceptance["artifact_hashes"]["model.pt"] == strict["checkpoint_sha256"]
            == off["checkpoint_sha256"], "Wrong checkpoint")
    require(result["metrics"] == strict["metrics"] == off["metrics"], "Wrong checkpoint metrics")
    require(all(type(result["metrics"][k]) in (int, float) and np.isfinite(result["metrics"][k])
                for k in ("accuracy", "aeod", "aspd")), "Undefined native metrics")
    require(result["evaluation_stats"]["prediction_count"] == 19867, "Partial valid predictions")
    image = result["data_contract"]["image_data_contract"]
    require(image["evaluation_split"] == "valid" and image["actual_train_rows"] == 162770
            and image["actual_evaluation_rows"] == 19867 and image["train_eval_disjoint"] is True
            and image["root_client_disjoint"] is True, "Wrong image/root/valid contract")
    require(len(image["client_sample_counts"]) == 20 and image["root_image_ids_sha256"],
            "Missing root identity")


def identity_record(method, job_id):
    """Only registered, root-adopted exact chunks; returns metadata, never evaluates.

    Existing GPU strict proofs are read as external evidence. The CPU process
    never impersonates their GPU runtime and never invokes checked_result.
    """
    require(method in METHODS, "Unsupported method")
    manifest = read_pin({"path": str(HERE / "PROOF_PINS.json"), "sha256": PROOFS_SHA256})
    require(method in manifest["chains"], "No adopted original strict/offserver/root proof for method")
    chain = manifest["chains"][method]
    require(digest(chain["checker"]["path"]) == chain["checker"]["sha256"], "Original strict source changed")
    root, off, strict = (read_pin(chain[k]) for k in ("root", "offserver", "strict"))
    require(all(doc["status"] == chain[k]["status"] for k, doc in
                (("root", root), ("offserver", off), ("strict", strict))), "Wrong proof status")
    require(job_id in root["accepted_new_ids"] and job_id in chain["records"], "Job not in adopted exact chunk")
    require(root[chain["root_offserver_key"]] == chain["offserver"]["sha256"], "Broken root/offserver link")
    if "offserver_strict_key" in chain:
        require(off[chain["offserver_strict_key"]] == chain["strict"]["sha256"], "Broken strict/offserver link")
    members = read_pin(chain["raw_index"])
    binding = read_pin(chain["index_binding"])
    require(binding[chain["index_binding_key"]] == chain["raw_index"]["sha256"], "Broken root/raw index link")
    indexed = members[chain["index_files_key"]]
    evidence = chain["records"][job_id]
    for pin in evidence.values():
        require(indexed[pin["index_key"]]["sha256"] == pin["sha256"], "Raw index/member mismatch")
    # FL's strict receipt is bound by the same root-pinned raw index.
    if "strict_index_key" in chain:
        require(indexed[chain["strict_index_key"]]["sha256"] == chain["strict"]["sha256"], "Broken strict/raw link")
    def one(records):
        found = [r for r in records if r["id"] == job_id]
        require(len(found) == 1, "Missing/duplicate external proof record")
        return found[0]
    sr = one(strict["records"])
    off_records = off["local"]["records"] if method == "CosineFairnessHybrid" else off["records"]
    ore = one(off_records)
    if method == "FLGMM":
        require(strict["original_checker_sha256"] == chain["checker"]["sha256"], "Wrong original checker")
    elif method == "CosineFairnessHybrid":
        require(strict["original_checked_source_sha256"] == chain["checker_function_sha256"], "Wrong original checked body")
        require(root["remote_strict_sha256"] == chain["strict"]["sha256"]
                and root["handoff_sha256"] == chain["index_binding"]["sha256"], "Broken Hybrid root proof link")
    else:
        require(sr["status"] == "ORIGINAL_CHECKED_RESULT_PASS"
                and sr["validator_sha256"] == chain["checker"]["sha256"], "Wrong original gradient checker")
    job, result, prov, acceptance = (read_pin(evidence[k]) for k in ("job", "result", "provenance", "acceptance"))
    if "stored_job" in evidence:
        require(read_pin(evidence["stored_job"]) == job, "Stored/original job identity changed")
    job_sha = evidence["job"]["sha256"]
    require(prov["job_sha256"] == job_sha, "Wrong provenance/job SHA")
    for r in (sr, ore):
        if "job_sha256" in r:
            require(r["job_sha256"] == job_sha, "Wrong strict/job SHA")
        for key in ("acceptance_sha256", "original_acceptance_sha256"):
            if key in r:
                require(r[key] == evidence["acceptance"]["sha256"], "Wrong strict/acceptance SHA")
    if method == "CosineFairnessHybrid":
        require(root["job_sha256"] == job_sha and acceptance["job_sha256"] == job_sha, "Wrong Hybrid root/job SHA")
        require(acceptance["scope_sha256"] == prov["scope_sha256"]
                == result["revision_job"]["scope_sha256"] == evidence["scope"]["sha256"], "Wrong Hybrid scope SHA")
    for name, sha in acceptance["artifact_hashes"].items():
        p = Path(evidence["result"]["path"]).parent / name
        require(p.name == name and p.parent == Path(evidence["result"]["path"]).parent, "Unsafe artifact path")
        require(digest(p) == sha, "Changed accepted artifact: " + name)
    output = Path(evidence["result"]["path"]).parent
    require(not list(output.glob("failure*.json")) and not (output / "FAILED.json").exists(),
            "Failure artifact present")
    validate_metadata(method, job, result, prov, acceptance, sr, ore,
                      read_pin(evidence["scope"]) if "scope" in evidence else None)
    # Only this new private record is normalized. Every original artifact stays byte-exact.
    return {"id": job_id, "method": method, "source_method": job["method"],
            "config": copy.deepcopy(job["config"]), "actual_alpha": result["alpha"],
            "data_contract": copy.deepcopy(result["data_contract"]["image_data_contract"]),
            "prior_validation_metrics": copy.deepcopy(result["metrics"]),
            "checkpoint": copy.deepcopy(evidence["model"]),
            "original_training_provenance": copy.deepcopy(prov),
            "external_proof_sha256": {k: chain[k]["sha256"] for k in ("root", "offserver", "strict")},
            "original_artifact_pins": copy.deepcopy(evidence),
            "status": "IDENTITY_ONLY_NO_NEW_PREDICTION_OR_FIT"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=sorted(METHODS))
    parser.add_argument("job_id")
    args = parser.parse_args()
    print(json.dumps(identity_record(args.method, args.job_id), indent=2, allow_nan=False))
