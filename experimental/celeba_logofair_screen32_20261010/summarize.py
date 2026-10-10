"""Descriptive frozen-score summary only after all32 original strict results exist."""
import argparse
import math
from pathlib import Path
import statistics
from frozen_score import score
from stage_inputs import HERE, bulk_path, digest, read, require, verify_delivery, write


def summarize(rows):
    require(len(rows) == len({r["id"] for r in rows}) == 32, "Complete32 only; no partial selection")
    candidates = []
    for candidate in sorted({r["candidate"] for r in rows}):
        group = [r for r in rows if r["candidate"] == candidate]
        require(len(group) == 4 and {(r["distribution"], r["attack"]) for r in group}
                == {("IID", "Benign"), ("IID", "S-DFA"), ("non-IID", "Benign"), ("non-IID", "S-DFA")}, "Missing four-condition identity")
        require(all(all(math.isfinite(r["metrics"][k]) and 0 <= r["metrics"][k] <= 1
                        for k in ("accuracy", "aeod", "aspd")) for r in group), "Nonfinite/out-of-range result")
        candidates.append(dict(candidate=candidate, seed_n=1,
            **{k: statistics.mean(r["metrics"][k] for r in group) for k in ("accuracy", "aeod", "aspd")},
            score=statistics.mean(score(r["metrics"]) for r in group)))
    require(len(candidates) == 8, "All eight candidates required")
    selected = min(candidates, key=lambda r: (-r["score"], r["candidate"]))
    champion = min(candidates, key=lambda r: (-r["accuracy"], r["candidate"]))
    def dominates(a, b):
        return a["accuracy"] >= b["accuracy"] and a["aeod"] <= b["aeod"] and a["aspd"] <= b["aspd"] and (
            a["accuracy"] > b["accuracy"] or a["aeod"] < b["aeod"] or a["aspd"] < b["aspd"])
    return dict(status="DESCRIPTIVE_LOCAL_STRICT32_SUMMARY_ROOT_REVIEW_PENDING", records=rows, candidates=candidates,
        selected_per_method={"LoGoFair-DP-official-adapted": selected}, accuracy_champion=champion,
        three_metric_pareto=[r for r in candidates if not any(dominates(a, r) for a in candidates)],
        score_semantics="Original frozen score per condition, then mean over four conditions; exact ties lexical candidate ID",
        seed_n=1, sample_SD_reported=False, significance_claimed=False, all_negative_results_retained=True,
        recipe_adopted=False, formal100_started=False, final_test=False, true_training_client_fairness=False)


def run(index_path, output):
    verify_delivery()
    index_path, _ = bulk_path(index_path, 0)
    output, _ = bulk_path(output, 4 * 1024 * 1024)
    require(not output.exists() and not (index_path.parent / "QUEUE_FAILURE.json").exists(), "Preserve old/failed output")
    index = read(index_path)
    require(index["status"] == "LOCAL_ORIGINAL_STRICT32_COMPLETE_ROOT_REVIEW_PENDING"
            and index["source_seal_sha256"] == digest(HERE / "FILES_SHA256.json"), "Actual strict32 closure required")
    manifest = read(HERE / "jobs/manifest.json")
    expected = {r["id"]: r for r in manifest["jobs"]}
    refs = {e["id"]: e for e in read(HERE / "snapshot/logofair_bridge_20261010/reuse_manifest.json")["entries"]}
    rows = []
    for entry in index["records"]:
        require(digest(entry["result"]) == entry["result_sha256"] and digest(entry["acceptance"]) == entry["acceptance_sha256"], "Strict result/proof bytes changed")
        job_entry = expected[entry["id"]]
        job = read(HERE / "jobs" / job_entry["job"])
        result, acceptance = read(entry["result"]), read(entry["acceptance"])
        require(result["job"] == job and result["settings"] == job["settings"] and acceptance["status"] == "PASS"
                and acceptance["job_sha256"] == job_entry["job_sha256"], "Strict job/result binding differs")
        for name, wanted in acceptance["artifact_hashes"].items():
            require(digest(Path(entry["result"]).parent / name) == wanted, "Original strict artifact changed")
        source = refs[job["baseline_id"]]["source_job"]
        rows.append(dict(id=entry["id"], candidate=job["candidate"], distribution=source["distribution"], attack=source["attack"],
                         seed=job["seed"], fit_seed=job["fit_seed"], metrics=result["metrics"], checkpoint_sha256=result["checkpoint_sha256"]))
    require({r["id"] for r in rows} == set(expected), "Strict32 IDs differ")
    write(output, dict(summarize(rows), strict_index_sha256=digest(index_path)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.index, a.out)
