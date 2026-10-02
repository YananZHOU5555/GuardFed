"""Reconstruct every historical Table II cell and a same-round correction candidate."""
from pathlib import Path
from collections import defaultdict, Counter
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/guardfed_tables/table2_recovered_20261002"
KEYS = {"ACC": "accuracy", "AEOD": "aeod", "ASPD": "aspd"}
ATTACKS = ["Benign", "F-Flip", "FedSA", "S-DFA", "Sp-DFA"]


def escape(s):
    return s.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%").replace("→", r"$\to$")


def label(method):
    if method == "FedWA":
        return r"FedWA$^{\dagger}$"
    if method == "Cosine Similarity + Fairness Deviation":
        return r"Cosine + fairness$^{\ddagger}$"
    return escape(method)


def formatted(c, tex=False):
    scale = 100 if c["metric"] == "ACC" else 1
    digits = 2 if c["metric"] == "ACC" else 4
    v = c["recovered_value"] * scale
    if c["n"] == 1:
        return f"${v:.{digits}f}^{{1}}$" if tex else f"{v:.{digits}f} [n=1]"
    sd = c["sample_std"] * scale
    return f"${v:.{digits}f} \\pm {sd:.{digits}f}$" if tex else f"{v:.{digits}f} ± {sd:.{digits}f} [n={c['n']}]"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    packet = json.loads((EVIDENCE / "source_map_complete.json").read_text(encoding="utf-8"))
    historical = packet["source_map"]
    assert len(historical) == 480
    assert Counter(c["n"] for c in historical) == {1: 294, 10: 180, 3: 6}
    assert Counter(c["status"] for c in historical) == {"EXACT_PUBLISHED_ROUNDING": 435, "HIDDEN_NE_VALUE_RECOVERED": 44, "PUBLISHED_VALUE_DIFFERENCE": 1}
    needed = defaultdict(set)
    for c in historical:
        assert math.isfinite(c["recovered_value"])
        assert (c["sample_std"] is None) == (c["n"] == 1)
        assert len(c["seed_identity"]) == c["n"]
        if c["n"] == 1:
            for row in c["round_identity"]:
                needed[c["atomic_source_file"]].add(row["atomic_jsonl_line"])
    atomic = {}
    for filename, lines in needed.items():
        for i, line in enumerate((ROOT / filename).open(encoding="utf-8"), 1):
            if i in lines:
                atomic[(filename, i)] = json.loads(line)
        assert all((filename, i) in atomic for i in lines)
    multi = json.loads((EVIDENCE / "后期三场景十种子记录.json").read_text(encoding="utf-8"))
    ten = {(c["identity"]["method"], c["identity"]["distribution"], c["identity"]["attack"]): c for c in multi}
    three = json.loads((EVIDENCE / "旧FedSA三种子记录.json").read_text(encoding="utf-8"))
    errors = []
    revised = []
    scalar_check_count = 0
    for c in historical:
        r = dict(c)
        metric = KEYS[c["metric"]]
        if c["n"] == 1:
            ids = c["round_identity"]
            assert len(ids) == 1
            src = atomic[(c["atomic_source_file"], ids[0]["atomic_jsonl_line"])]
            assert (src["dataset"], src["distribution"], src["method"], src["attack"], src["seed"], src["rounds"]) == ("adult", c["distribution"], c["source_method"], c["actual_attack"], 123, 70)
            vals = [x["metrics"][metric] for x in src.get("last10_metrics", [])]
            selected = (max if metric == "accuracy" else min)(vals) if vals else src["metrics"][metric]
            errors.append(abs(selected - c["recovered_value"]))
            r["recovered_value"] = src["metrics"][metric]
            r["sample_std"] = None
            r["round70_source"] = {"file": c["atomic_source_file"], "line": ids[0]["atomic_jsonl_line"], "run_id": src["run_id"]}
            scalar_check_count += 1
        elif c["n"] == 10:
            cohort = ten[(c["source_method"], c["distribution"], c["actual_attack"])]
            s = cohort["final70_statistics"][metric]
            r["recovered_value"], r["sample_std"] = s["mean"], s["sample_std"]
            r["round70_source"] = {"cohort_id": cohort["cohort_id"], "source": "后期三场景十种子记录.json", "seeds": cohort["seeds"]}
        else:
            matches = [x for x in three if x["identity"]["method"] == c["source_method"] and x["identity"]["distribution"] == c["distribution"] and x["identity"]["attack"] == c["actual_attack"]]
            assert len(matches) == 1
            cohort = matches[0]
            vals = [x["metrics"][metric] for x in cohort["seed_identities"]]
            assert len(vals) == 3
            r["recovered_value"], r["sample_std"] = statistics.mean(vals), statistics.stdev(vals)
            s = cohort["final70_statistics"][metric]
            errors.extend([abs(r["recovered_value"] - s["mean"]), abs(r["sample_std"] - s["sample_std"])])
            scalar_check_count += 2
            r["round70_source"] = {"cohort_id": cohort["cohort_id"], "source": "旧FedSA三种子记录.json", "seeds": cohort["seeds"]}
        r["selection_rule"] = "same run final round70; arithmetic mean and sample SD across actual seeds when n>=2"
        r["scientific_status"] = "SAME_CHECKPOINT_RECORD_CANDIDATE; historical method identity limits retained"
        revised.append(r)
    assert scalar_check_count == 306 and max(errors) < 1e-12
    methods = list(dict.fromkeys(c["method"] for c in historical))
    all_fragments = []
    for version, cells in [("round70_revision", revised), ("historical_reconstruction", historical)]:
        lookup = {(c["method"], c["distribution"], c["attack"], c["metric"]): c for c in cells}
        assert len(lookup) == 480
        title = "Same-checkpoint round70 correction candidate" if version == "round70_revision" else "Historical selections reconstructed"
        md = [f"# Adult Table II — {title}", "", "所有480格有数值。n=1表示单次结果、SD不可计算；n≥2为mean±sampleSD。原始方法名/混合分支限制仍需写入正文，不认证原法忠实度。", ""]
        for dist in ["IID", "non-IID"]:
            md += [f"## {dist}", "", "| 投稿展示名 | 指标 | " + " | ".join(ATTACKS) + " |", "|---|---|" + "---:|" * 5]
            tex = [r"\begin{center}", r"\textbf{Adult Table II: " + escape(dist) + " --- " + title + "}", r"\end{center}", r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.05}", r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l|l|ccccc}", r"\toprule", r"Historical label & Metric & Benign & F-Flip & FedSA & S-DFA & Sp-DFA \\", r"\midrule"]
            for method in methods:
                for j, metric in enumerate(KEYS):
                    row = [lookup[(method, dist, attack, metric)] for attack in ATTACKS]
                    md.append("| " + method + " | " + metric + " | " + " | ".join(formatted(c) for c in row) + " |")
                    meth = r"\multirow{3}{*}{" + label(method) + "}" if j == 0 else ""
                    metriclabel = r"ACC (\%)" if metric == "ACC" else metric
                    tex.append(meth + " & " + metriclabel + " & " + " & ".join(formatted(c, True) for c in row) + r" \\")
                tex.append(r"\midrule")
            tex[-1] = r"\bottomrule"
            tex += [r"\end{tabular*}", r"\vspace{4mm}", r"\begin{minipage}{\linewidth}\footnotesize", r"Superscript $1$: single seed, SD unavailable. Other cells: mean $\pm$ sample SD (ddof=1), n=10 except FedWA/FedSA n=3. No accuracy-based suppression of fairness values. AEOD denotes the implemented absolute TPR gap.", r"\par $\dagger$ FedWA is a historical display label whose numeric source is AdaAggRL. $\ddagger$ The Cosine row combines GuardFed-AD2 in Benign/S-DFA/Sp-DFA with GuardFed in F-Flip/FedSA. These are source identifications, not certifications of the named baseline algorithms."]
            if version == "historical_reconstruction":
                tex += [r"\par Historical n=1 cells independently maximize ACC and minimize AEOD/ASPD over rounds61--70; 94/98 conditions have no common selected round. F-Flip/FedSA n=10 use opposite selection directions for different method groups. FedWA/FedSA n=3 jointly select ACC-.5(AEOD+ASPD). These test-selected results do not provide an unbiased matched-method ranking.", r"\par FairGuard IID FedSA ACC is restored as54.13\% from the source, versus59.13\% in the submitted PDF. Original submission retained separately."]
            else:
                tex += [r"\par Every metric uses the same final-round record within a seed. Mixed historical training recipes and sample sizes remain disclosed; n=1 has no repeated-seed uncertainty. This candidate removes checkpoint-selection mixing but does not resolve baseline implementation fidelity or create missing seeds."]
            tex += [r"\end{minipage}"]
            fragment = OUT / f"table2_{version}_{'iid' if dist == 'IID' else 'noniid'}.tex"
            fragment.write_text("\n".join(tex) + "\n", encoding="utf-8")
            all_fragments.append(fragment.name)
            md.append("")
        (OUT / f"table2_{version}.md").write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
        (OUT / f"table2_{version}_cells.json").write_text(json.dumps(cells, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    doc = [r"\documentclass[10pt]{article}", r"\usepackage[a3paper,landscape,margin=12mm]{geometry}", r"\usepackage{booktabs,multirow,array}", r"\pagestyle{plain}", r"\begin{document}", r"\small"]
    for i, name in enumerate(all_fragments):
        if i:
            doc.append(r"\clearpage")
        doc.append(r"\input{" + name + "}")
    doc.append(r"\end{document}")
    (OUT / "table2_complete_review.tex").write_text("\n".join(doc) + "\n", encoding="utf-8")
    result = {"status": "ALL480_NUMERIC_SOURCES_AND_SAME_ROUND_CANDIDATE_BUILT", "all_cells": 480, "hidden_values_restored": 44, "source_rounding_matches": 435, "published_numeric_discrepancies": 1, "n_cell_counts": dict(Counter(c["n"] for c in historical)), "single_seed_projection_checks": 294, "three_seed_final70_checks": 12, "max_numeric_error": max(errors), "new_training": 0, "all_cells_have_ten_seeds": False, "manuscript_assembly_script_found": False, "method_fidelity_certified": False}
    (EVIDENCE / "complete_reconstruction_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
