"""Independently check retained 10-seed record statistics and export an appendix."""
from pathlib import Path
import hashlib
import json
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
AGENT = ROOT / "tmp/table2_cohort_audit_20261002"


def main():
    records = {}
    for line in (ROOT / "tmp/table2_rawaudit_20261002/attack_strength_raw.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("dataset") == "adult" and r.get("study") == "main" and r.get("rounds") == 70:
            key = (r["method"], r["distribution"], r["attack"], r["seed"])
            assert key not in records, key
            records[key] = r
    cohorts = [x for x in json.loads((AGENT / "attack_strength_ten_seed.json").read_text(encoding="utf-8")) if x["identity"]["method"] not in {"GuardFed-AD2", "GuardFed-AD2+"}]
    assert len(cohorts) == 120
    errors = []
    expected_seeds = [123, 456, 789, 1001, 2024, 3141, 4242, 5050, 6060, 7070]
    for c in cohorts:
        i = c["identity"]
        assert c["seeds"] == expected_seeds and c["n_seeds"] == 10
        raw = [records[(i["method"], i["distribution"], i["attack"], s)] for s in expected_seeds]
        for r in raw:
            assert [x["round"] for x in r["trajectory_metrics"]] == list(range(1, 71))
            assert r["metrics"] == r["trajectory_metrics"][-1]["metrics"]
        for k, stats in c["final70_statistics"].items():
            values = [r["metrics"][k] for r in raw]
            errors.extend([abs(statistics.mean(values) - stats["mean"]), abs(statistics.stdev(values) - stats["sample_std"])])
    assert len(errors) == 720 and max(errors) < 1e-14
    archive = ROOT / "tmp/git_sync_assets/guardfed-5090-20260923.tar.gz"
    h = hashlib.sha256()
    with archive.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    assert h.hexdigest() == "a2a0384c9e02d1e3c0cca176d2ea53ef99e37016eff3475ccc779765961772e0"
    (OUT / "后期三场景十种子记录.json").write_text(json.dumps(cohorts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for src, dst in [("summary.json", "cohort_audit_summary.json"), ("paper_FedSA_three_seed.json", "旧FedSA三种子记录.json"), ("README.md", "独立种子审查.md")]:
        shutil.copyfile(AGENT / src, OUT / dst)
    shutil.copyfile(ROOT / "tmp/table2_rawaudit_20261002/source_receipt.json", OUT / "source_receipt.json")
    check = {"status": "VERIFIED_RETAINED_RECORD_STATISTICS_NOT_SUBMISSION_PROVENANCE", "raw_methods": 20, "conditions": 120, "records": 1200, "metrics": 360, "mean_and_sd_checks": 720, "max_numeric_error": max(errors), "sample_std_ddof": 1, "checkpoint_hash_verified": False, "new_training": 0, "archive_sha256": h.hexdigest()}
    (OUT / "statistics_verification.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    methods = list(dict.fromkeys(c["identity"]["method"] for c in cohorts))
    lookup = {(c["identity"]["method"], c["identity"]["distribution"], c["identity"]["attack"]): c for c in cohorts}
    lines = ["# Adult：保留日志三场景的十种子重统计", "", "**历史归档统计附表，不能直接替换投稿Table II。** 保留原始代码方法名；不认证原法忠实度或名称映射。采用70轮终轮记录的mean±sampleSD，n=10；ACC单位为%，AEOD/ASPD为0–1。F Flip=all_unprivileged，FedSA gain=4.5、norm_ratio=3；root=训练分区10%。S-DFA/Sp-DFA的同配方全方法十种子队列未在本次档案核实中找到。", "", "完整轨迹与终轮记录已检查；历史checkpoint/训练源码/数据内容hash缺失，不能据此称checkpoint严格验收。保留低准确率和真实零公平数值，不按准确率隐藏指标。", ""]
    for dist in ["IID", "non-IID"]:
        lines += [f"## {dist}", "", "| 原始代码方法名 | 指标 | Benign | F Flip | FedSA |", "|---|---|---:|---:|---:|"]
        for method in methods:
            for label, key, scale, digits in [("ACC (%)", "accuracy", 100, 3), ("AEOD", "aeod", 1, 5), ("ASPD", "aspd", 1, 5)]:
                values = []
                for attack in ["Benign", "F Flip", "FedSA"]:
                    s = lookup[(method, dist, attack)]["final70_statistics"][key]
                    values.append(f"{s['mean']*scale:.{digits}f} ± {s['sample_std']*scale:.{digits}f}")
                lines.append(f"| {method} | {label} | " + " | ".join(values) + " |")
        lines.append("")
    (OUT / "三场景十种子重统计.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(check, ensure_ascii=False))


if __name__ == "__main__":
    main()
