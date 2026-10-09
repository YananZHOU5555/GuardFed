"""Audit historical synthetic tables against preserved raw records, without training."""
from pathlib import Path
import csv
import hashlib
import json
import math
import tarfile
import statistics
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "tmp/git_sync_assets/guardfed-5090-20260923.tar.gz"
RAW = ROOT / "tmp/table2_rawaudit_20261002/paper_tables_raw.jsonl"
TABLES = ROOT / "outputs/guardfed_tables/expanded_experiments_final"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def key(record, config=None):
    config = config or record
    return (record["dataset"], record["distribution"], record["attack"],
            int(record["seed"]), config["synthetic_method"],
            round(float(config["server_ratio"]), 10),
            round(float(config["synthetic_ratio"]), 10))


def close(actual, expected):
    assert math.isfinite(float(actual)) and math.isfinite(float(expected))
    assert abs(float(actual) - float(expected)) <= 1e-12, (actual, expected)


def main():
    archive_sha = sha(ARCHIVE.read_bytes())
    assert archive_sha == "a2a0384c9e02d1e3c0cca176d2ea53ef99e37016eff3475ccc779765961772e0"
    raw_sha = sha(RAW.read_bytes())
    assert raw_sha == "c94441fc33b4a460008bcb6eac409a856ae4ad79ed8baa132866342120f4e541"
    member_receipts = {}
    source_hits = []
    inspected = 0
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        assert sha(archive.extractfile("./results/paper_tables/raw_results.jsonl").read()) == raw_sha
        for member in archive.getmembers():
            if not member.isfile() or member.size > 2_000_000:
                continue
            selected_source = member.name.endswith((".py", ".sh", ".txt", ".toml"))
            selected_evidence = member.name in [
                "./results/paper_tables/advisor_experiments/forest_diffusion_run.log",
                "./results/paper_tables/expanded_experiments/expanded_synthetic_ratios_raw.csv"]
            if not (selected_source or selected_evidence):
                continue
            data = archive.extractfile(member).read()
            if selected_evidence or "reproduce_paper_tables.py" in member.name:
                member_receipts[member.name] = {"bytes": len(data), "sha256": sha(data)}
            if selected_source:
                inspected += 1
                lines = data.decode("utf8", errors="replace").splitlines()
                hits = [{"line": i + 1, "text": line} for i, line in enumerate(lines)
                        if any(term in line.lower() for term in
                               ["forestdiffusion", "forest_diffusion", "forest-diffusion"])]
                if hits:
                    source_hits.append({"member": member.name, "sha256": sha(data), "hits": hits})

    records = {}
    suite_counts = Counter()
    for lineno, line in enumerate(RAW.read_bytes().splitlines(), 1):
        record = json.loads(line)
        config = record.get("config", {})
        suite = config.get("experiment_suite", "")
        if "synthe" in suite or "generation" in suite:
            suite_counts[suite] += 1
        if suite != "expanded_synthetic_ratios":
            continue
        identity = key(record, config)
        assert identity not in records, identity
        assert record["rounds"] == config["rounds"] == 70
        assert record["method"] == "GuardFed-AD2+"
        assert [v["round"] for v in record["last10_metrics"]] == list(range(61, 71))
        close(record["alpha"], 5000 if record["distribution"] == "IID" else 5)
        records[identity] = (record, lineno, sha(line))
    assert len(records) == 840

    checks = []
    mixed_legacy = []
    for filename, selected in [("expanded_synthetic_ratios_raw.csv", False),
                               ("synthetic_joint_raw.csv", True)]:
        data = (TABLES / filename).read_bytes()
        rows = list(csv.DictReader(data.decode("utf-8-sig").splitlines()))
        assert len(rows) == len(records)
        seen = set()
        for row in rows:
            identity = key(row)
            assert identity not in seen
            seen.add(identity)
            record, lineno, line_sha = records[identity]
            if selected:
                best = max(record["last10_metrics"], key=lambda item:
                           item["metrics"]["accuracy"] - .5 *
                           (item["metrics"]["aeod"] + item["metrics"]["aspd"]))
                metrics = best["metrics"]
                assert int(row["joint_round"]) == best["round"]
                for column, metric in [("joint_acc", "accuracy"), ("joint_aeod", "aeod"), ("joint_aspd", "aspd")]:
                    close(row[column], metrics[metric])
                close(row["joint_score"], metrics["accuracy"] - .5 * (metrics["aeod"] + metrics["aspd"]))
                for column, metric in [("final_acc", "accuracy"), ("final_aeod", "aeod"), ("final_aspd", "aspd")]:
                    close(row[column], record["metrics"][metric])
                checks.append({"dataset": identity[0], "distribution": identity[1], "attack": identity[2],
                               "seed": identity[3], "synthetic_method": identity[4],
                               "server_ratio": identity[5], "synthetic_ratio": identity[6],
                               "raw_line": lineno, "raw_line_sha256": line_sha,
                               "selected_round": best["round"], "raw_config_sha256": sha(json.dumps(record["config"], sort_keys=True, separators=(",", ":")).encode()),
                               "acc": metrics["accuracy"], "aeod": metrics["aeod"], "aspd": metrics["aspd"],
                               "root_clean_rows": record["data_contract"]["root_clean_rows"],
                               "root_synthetic_rows": record["data_contract"]["root_synthetic_rows"]})
            else:
                history = record["last10_metrics"]
                # This older export takes the best of each column separately.
                # Recover that rule explicitly; never call it one checkpoint.
                for column, metric, select in [("ACC", "accuracy", max), ("AEOD", "aeod", min), ("ASPD", "aspd", min)]:
                    close(row[column], select(v["metrics"][metric] for v in history))
                close(row["ACC_pct"], 100 * float(row["ACC"]))
                close(row["fair_avg"], .5 * (float(row["AEOD"]) + float(row["ASPD"])))
                close(row["score"], float(row["ACC"]) - float(row["fair_avg"]))
                common = [v["round"] for v in history if all(abs(float(row[column]) - v["metrics"][metric]) <= 1e-12
                          for column, metric in [("ACC", "accuracy"), ("AEOD", "aeod"), ("ASPD", "aspd")])]
                if not common:
                    mixed_legacy.append({"identity": identity, "raw_line": lineno,
                                         "acc_rounds": [v["round"] for v in history if v["metrics"]["accuracy"] == float(row["ACC"])],
                                         "aeod_rounds": [v["round"] for v in history if v["metrics"]["aeod"] == float(row["AEOD"])],
                                         "aspd_rounds": [v["round"] for v in history if v["metrics"]["aspd"] == float(row["ASPD"])]})
        assert seen == set(records)
        member_receipts["local:" + filename] = {"bytes": len(data), "sha256": sha(data)}

    with (OUT / "per_record_840.csv").open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    grouped = defaultdict(list)
    for row in checks:
        grouped[(row["dataset"], row["synthetic_method"], row["server_ratio"], row["synthetic_ratio"], row["seed"])].append(row)
    per_seed = []
    for identity, rows in sorted(grouped.items()):
        assert {(r["distribution"], r["attack"]) for r in rows} == {
            (distribution, attack) for distribution in ["IID", "non-IID"] for attack in ["Benign", "FedSA"]}
        assert len(rows) == 4
        record = dict(zip(["dataset", "synthetic_method", "server_ratio", "synthetic_ratio", "seed"], identity))
        record.update({metric: statistics.fmean(r[metric] for r in rows) for metric in ["acc", "aeod", "aspd"]})
        per_seed.append(record)
    setting_groups = defaultdict(list)
    for row in per_seed:
        setting_groups[(row["dataset"], row["synthetic_method"], row["server_ratio"], row["synthetic_ratio"])].append(row)
    summaries = []
    for identity, rows in sorted(setting_groups.items()):
        assert sorted(r["seed"] for r in rows) == [123, 456, 789]
        record = dict(zip(["dataset", "synthetic_method", "server_ratio", "synthetic_ratio"], identity))
        record["n_seeds"] = 3
        record["generator_implementation_status"] = ("NOT_APPLICABLE_REAL_ONLY" if identity[1] == "none"
                                                       else "PER_RUN_CODE_AND_FIT_IDENTITY_UNRESOLVED")
        for metric in ["acc", "aeod", "aspd"]:
            record[metric + "_mean"] = statistics.fmean(r[metric] for r in rows)
            record[metric + "_sample_sd"] = statistics.stdev(r[metric] for r in rows)
        summaries.append(record)
    for name, rows in [("joint_per_seed_210.csv", per_seed), ("joint_setting_summary_70.csv", summaries)]:
        with (OUT / name).open("w", encoding="utf8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    receipt = {"status": "NUMERICAL_TABLE_LINEAGE_ACCEPTED_GENERATOR_AND_FIGURE_LINEAGE_PARTIAL",
               "archive": str(ARCHIVE.relative_to(ROOT)), "archive_sha256": archive_sha,
               "raw_member_sha256": raw_sha, "raw_records": len(records),
               "verified_legacy_independent_extrema_rows": 840, "verified_joint_checkpoint_rows": 840,
               "legacy_rows_without_any_common_checkpoint": len(mixed_legacy),
               "legacy_mixed_checkpoint_details": mixed_legacy,
               "seeds": sorted({k[3] for k in records}),
               "within_seed_scenario_means": len(per_seed), "three_seed_setting_summaries": len(summaries),
               "summary_rule": "First average IID/non-IID x Benign/FedSA within each seed, then mean and sample SD over the three shared seeds. This does not fix historical test-based checkpoint selection.",
               "synthetic_method_record_counts": dict(Counter(k[4] for k in records)),
               "all_synthetic_suite_counts": dict(suite_counts),
               "member_receipts": member_receipts, "inspected_text_sources": inspected,
               "forest_source_hits": source_hits,
               "limits": ["Numerical tables are linked to archived raw records, not proven identical to every plotted Fig.3 point.",
                          "The older expanded_synthetic_ratios_raw.csv independently maximizes ACC and minimizes AEOD/ASPD over the last ten rounds. Its derived score/fairness values generally do not describe any one model checkpoint; use the audited joint table for coherent historical comparisons.",
                          "Historical joint checkpoints use the evaluated last-ten-round metrics; they are not frozen terminal/untouched-test results.",
                          "The three archived reproduce_paper_tables source snapshots do not implement or accept forest_diffusion. Driver lists and successful logs do not recover its generator implementation/package/fit-data identity.",
                          "The raw record has no per-run immutable source, generator-cache or fitted-generator hash. Current source cannot be certified as the code executed for every historical generator.",
                          "Three seeds are independent replications; two distributions and two attacks are scenarios, not twelve independent seeds."]}
    receipt["per_record_sha256"] = sha((OUT / "per_record_840.csv").read_bytes())
    (OUT / "acceptance.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: receipt[k] for k in ["status", "raw_records", "verified_legacy_independent_extrema_rows", "verified_joint_checkpoint_rows", "legacy_rows_without_any_common_checkpoint", "synthetic_method_record_counts"]}))


if __name__ == "__main__":
    main()
