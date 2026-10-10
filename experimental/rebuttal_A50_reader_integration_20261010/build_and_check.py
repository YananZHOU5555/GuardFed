"""Bounded A50 prose integration; --check is read-only and reuses the frozen writing checker."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "SOURCE_PINS.json").read_bytes())
DOCS = ("rebuttal_integrated_20261010.md", "manuscript_insertions_integrated_20261010.md")
BASE = "E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_IID_five_scenes50_20261010/"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def text_sha(value):
    return sha(value.encode("utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(key):
    pin = PINS[key]
    data = Path(pin["path"]).read_bytes()
    require(sha(data) == pin["sha256"] and len(data) == pin["bytes"], "Source changed: " + key)
    return json.loads(data)


def pointer(value, path):
    for key in path.split("/")[1:]:
        key = key.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def original_checker():
    pin = PINS["original_writing_checker"]
    require(sha(Path(pin["path"]).read_bytes()) == pin["sha256"], "Original checker changed")
    spec = importlib.util.spec_from_file_location("frozen_reader_writing_checker", pin["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(name, value):
    (HERE / name).write_bytes((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def build():
    for key in PINS:
        pin = PINS[key]
        data = Path(pin["path"]).read_bytes()
        require(sha(data) == pin["sha256"] and len(data) == pin["bytes"], "Input pin: " + key)
    root, table, seed_first = load("A50_root"), load("tables"), load("iid_seed_first")
    require(root["root_adoption"] is True and root["paired_models"] == 50 and root["complete_scenes"] == 5,
            "A50 is not actually adopted")
    facts = []

    def pair(key, panel, row, metric, shared_panel=None):
        path = f"/panels/{panel}/rows/{row}/{metric}"
        value = pointer(table if key == "tables" else seed_first, path)
        digits = 3 if metric == "accuracy_pct" else 5
        display = f"{value['mean']:+.{digits}f} ± {value['sample_sd_ddof1']:.{digits}f}"
        facts.append({"source": key, "mean_pointer": path + "/mean", "sd_pointer": path + "/sample_sd_ddof1",
                      "mean": value["mean"], "sd": value["sample_sd_ddof1"], "digits": digits, "display": display,
                      "same_shared_pointer": None if shared_panel is None else path.replace(f"/panels/{panel}/", f"/panels/{shared_panel}/")})
        return display

    metrics = ("accuracy_pct", "aeod", "aspd")
    sp_native = " / ".join(pair("tables", 0, 14, m, 6) for m in metrics)
    sp_raw = " / ".join(pair("tables", 3, 14, m) for m in metrics)
    avg_native = " / ".join(pair("iid_seed_first", 0, 2, m, 6) for m in metrics)
    avg_raw = " / ".join(pair("iid_seed_first", 3, 2, m) for m in metrics)
    block = f"""**Current A50 extension: all five IID scenes.** The [accepted A50 table]({BASE}TABLES.md) and [root verification]({BASE}ROOT_VERIFICATION.json) add IID Sp-DFA to the historical A40 comparison above, completing Benign, F Flip, FedSA, S-DFA and Sp-DFA with 50 minus_A models paired to 50 Full checkpoints. Each scene has ten matched seeds; all three views retain each model’s same round-70 checkpoint and the same 19,867 validation images. Only the A score contribution is removed; geometric hard filtering and prediction calibration remain enabled. The single available non-IID Benign minus_A record (seed 91001) is excluded from every displayed average. All five non-IID A scenes and the other five incomplete image-control variants remain pending.

**Added IID Sp-DFA evidence.** The ten-seed paired differences, minus_A−Full, are {sp_native} for native and shared calibration, and {sp_raw} for raw, in ΔACC / ΔAEOD / ΔASPD order. Values are means ± sample SD (ddof=1); ΔACC is in percentage points and disparity differences are absolute gaps. Accuracy decreases refer to the paired mean: in each view, four seeds have positive accuracy differences and six have negative differences. Deletion lowers native/shared mean AEOD but raises mean ASPD; under raw prediction it raises both disparity means. Raw ASPD reverses sign in the fixed six-seed panel, whereas native/shared ASPD remains positive. The complete fixed 10/9/6-seed panels are retained in the linked table, without selecting a favorable subset.

**Five-IID seed-first summary.** We first average the five scenes within each seed, then summarize across the ten seeds; the sampling unit is the seed, not 50 independent scene–seed observations. The paired ΔACC / ΔAEOD / ΔASPD summaries are {avg_native} for native/shared calibration and {avg_raw} for raw, with the same units and sample-SD convention. The [per-scene values]({BASE}tables.json) and [seed-first values]({BASE}IID_SEED_FIRST.json) also retain the fixed nine- and six-seed summaries. Raw aggregate ASPD is positive in the ten- and nine-seed panels and negative in the six-seed panel. These descriptive averages do not establish significance, component necessity or an isolated causal mechanism.

**Current A50 comparability and scope.** Full replay comprises three CPU and 47 GPU checkpoints; all 50 minus_A replays use CPU. Both training cohorts contain 50 cu128 checkpoints. The A40 device counts above describe that historical subset only; mixed replay devices and driver provenance do not establish device equivalence. Native and shared-calibration metrics and group counts coincide within each of the 100 records, so these views are not independent confirmations. Root-only calibration, configuration-selection seed 91001, validation exposure and prior official-test exposure remain disclosed; the fixed sensitivity panels do not restore an untouched evaluation. U100/C100 and the baseline comparisons remain separate. Five non-IID A scenes remain incomplete, and the final primary endpoint and final test remain pending.

"""
    manifest = {"status": "REVERSIBLE_A50_INTEGRATION_FOR_AUTHOR_REVIEW", "input_reader_root": PINS["reader_root"],
                "input_A50_root": PINS["A50_root"], "documents": [], "historical_scope_normalization": {},
                "new_science_block": block, "new_scientific_measurements": 0}
    labels = {
        "**A40 extension: four complete IID scenes.**": "**Historical A40 extension: four complete IID scenes.**",
        "**A40 fixed-panel direction check.**": "**Historical A40 fixed-panel direction check.**",
        "**Current A40 comparability and coverage boundary.**": "**Historical A40 comparability and coverage boundary.**",
        "The A40 comparison in R3.2 adds": "The historical A40 comparison in R3.2 adds",
    }
    manifest["historical_scope_normalization"] = {after: before for before, after in labels.items()}
    history = "The later [accepted A40 comparison](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_IID_four_scenes40_20261010/TABLES.md) adds IID FedSA/S-DFA, giving four IID scenes with ten paired seeds and fixed 10/9/6 sensitivity panels. Six A scenes remain outside this accepted table; U100 and C100 are unchanged.\n\n"
    current = f"The [accepted A50 comparison]({BASE}TABLES.md) now covers all five IID scenes with 50 matched Full/minus_A pairs and fixed 10/9/6-seed panels. Five non-IID A scenes remain incomplete; the single available non-IID Benign minus_A record is excluded from the displayed averages. U100 and C100 are unchanged.\n\n"
    for name in DOCS:
        source = Path(PINS[name]["path"]).read_bytes().decode("utf-8")
        value, operations = source, []

        def edit(before, after, reason):
            nonlocal value
            require(value.count(before) == 1, "Nonunique edit: " + reason)
            at = value.index(before)
            op = {"reason": reason, "offset_codepoints": at, "before": before, "after": after,
                  "document_sha256_before": text_sha(value)}
            value = value[:at] + after + value[at + len(before):]
            op["document_sha256_after"] = text_sha(value)
            operations.append(op)

        for before, after in labels.items():
            if before in value:
                edit(before, after, "Mark existing A40 evidence as historical; no scientific value changed")
        anchor = "### R3.3 —" if name.startswith("rebuttal") else "**Root/heterogeneity paragraph.**"
        if name.startswith("rebuttal"):
            anchor = next(line for line in value.splitlines() if line.startswith(anchor))
        edit(anchor, block + anchor, "Add adopted A50 Sp-DFA and seed-first evidence with explicit scope")
        edit(history, current, "Current evidence history points to A50; retain A40 unchanged below")
        lineage = next(line for line in value.splitlines() if line.startswith("The separately [accepted A20 comparison]")) + "\n\n"
        edit(lineage, lineage + history, "Move unchanged A40 snapshot into historical lineage")
        if name.startswith("rebuttal"):
            ae_anchor = "**Revision material supplied.** The companion insertion text"
            ae = "The A-deletion extension now completes all five IID scenes with 50 matched Full/minus_A pairs, including IID Sp-DFA. Raw and native/shared prediction show different accuracy–disparity trade-offs; all five non-IID A scenes remain pending. R3.2 gives the paired and seed-first summaries, and R3.7 explains their limits.\n\n"
            edit(ae_anchor, ae + ae_anchor, "Add concise current A50 answer to the Associate Editor")
            r37_anchor = "The tabular COMPAS counterexamples above remain unchanged."
            r37 = "The new A50 IID Sp-DFA comparison reinforces this boundary: deletion lowers native/shared mean AEOD but raises mean ASPD, while both raw disparity means rise in the ten-seed panel. Raw ASPD reverses sign in the fixed six-seed panel. Its accuracy loss is a paired-mean result, with four positive and six negative seed-level differences in each view. The five-IID seed-first summary in R3.2 preserves these view and panel distinctions; it does not include the single non-IID Benign A record or complete the five non-IID scenes. "
            edit(r37_anchor, r37 + r37_anchor, "Interpret added A50 result without claiming universal or causal benefit")
            p2 = next(line for line in value.splitlines() if line.startswith("| P2 —"))
            p2_new = p2.replace("A40 now covers", "The historical A40 comparison covers").replace(
                "with 40 paired checkpoints; complete IID Sp-DFA, all five non-IID A scenes and the other five incomplete image controls.",
                "with 40 paired checkpoints. A50 adds IID Sp-DFA, completing all five IID scenes with 50 paired checkpoints; complete all five non-IID A scenes and the other five incomplete image controls.")
            require(p2 != p2_new and "A50 adds IID Sp-DFA" in p2_new, "P2 scope edit failed")
            edit(p2, p2_new, "Update only P2 completed/pending scope to adopted A50")
        candidate_name = name.replace("_20261010.md", "_A50_reader_20261010.md")
        (HERE / candidate_name).write_bytes(value.encode("utf-8"))
        manifest["documents"].append({"source": PINS[name]["path"], "source_sha256": text_sha(source),
                                      "candidate": (HERE / candidate_name).as_posix(), "candidate_sha256": text_sha(value),
                                      "operations": operations})
    save("EDIT_MANIFEST.json", manifest)
    save("FACT_BINDINGS.json", {"status": "ADOPTED_A50_VALUE_POINTERS", "mean_sd_pairs": facts,
        "scope": [{"source": "A50_root", "pointer": p, "expected": pointer(root, p)} for p in
                  ("/paired_models", "/complete_scenes", "/preserved_records", "/replay_devices", "/training_torch", "/seed_panels", "/aggregate_scope", "/test", "/primary_endpoint_selected")],
        "exclusion": {"source": "summary", "pointer": "/partial_nonIID_excluded", "expected": ["minus_A_non-IID_Benign_seed91001"]},
        "native_shared_equality": {"source": "focused_checks", "pointer": "/per_scene/native_shared_metrics_and_counts_exact", "expected": True},
        "source_scopes": "Only existing adopted tables and paired differences are read; no new scientific aggregation or model execution."})


def check():
    for key, pin in PINS.items():
        data = Path(pin["path"]).read_bytes()
        require(sha(data) == pin["sha256"] and len(data) == pin["bytes"], "Frozen input: " + key)
    old = original_checker()
    # This is the original checker, actually executed on its frozen reader candidates.
    old_result = old.run()
    reader_root = load("reader_root")
    require(reader_root["author_review_only"] is True and reader_root["A50_incorporated"] is False,
            "Unexpected reader input boundary")
    require(reader_root["documents_sha256"] == {name: PINS[name]["sha256"] for name in DOCS},
            "Reader root/document identity mismatch")
    require({d["candidate_sha256"] for d in old_result["documents"]} == {PINS[name]["sha256"] for name in DOCS},
            "Current reader documents differ from the actually checked editorial candidates")
    require(old.digest("reader") == text_sha("reader"), "Original digest reuse")
    root = load("A50_root")
    seal = load("A50_candidate_seal")
    for key in ("tables", "iid_seed_first", "official_table", "summary", "saved_verification"):
        file_name = Path(PINS[key]["path"]).name
        require(root["files_sha256"][file_name] == PINS[key]["sha256"] == seal["files"][file_name]["sha256"],
                "Canonical A50/source seal mismatch: " + key)
    for key in ("focused_checks", "paired_per_seed"):
        require(seal["files"][Path(PINS[key]["path"]).name]["sha256"] == PINS[key]["sha256"], "Source evidence seal")
    require(root["source_seal_sha256"] == PINS["A50_candidate_seal"]["sha256"] and root["root_adoption"], "A50 root source binding")
    bindings = json.loads((HERE / "FACT_BINDINGS.json").read_bytes())
    data = {key: load(key) for key in ("tables", "iid_seed_first", "summary", "focused_checks", "paired_per_seed", "A50_root")}
    for fact in bindings["mean_sd_pairs"]:
        mean = pointer(data[fact["source"]], fact["mean_pointer"])
        sd = pointer(data[fact["source"]], fact["sd_pointer"])
        require((mean, sd) == (fact["mean"], fact["sd"]), "Mean/SD pointer mismatch")
        require(f"{mean:+.{fact['digits']}f} ± {sd:.{fact['digits']}f}" == fact["display"], "Numeric formatting changed")
        if fact["same_shared_pointer"]:
            shared = pointer(data[fact["source"]], fact["same_shared_pointer"])
            require(shared == {"mean": mean, "sample_sd_ddof1": sd}, "Native/shared values differ")
    for fact in bindings["scope"] + [bindings["exclusion"], bindings["native_shared_equality"]]:
        require(pointer(data[fact["source"]], fact["pointer"]) == fact["expected"], "Scope/source fact mismatch")
    directions = []
    native_sp_aspd = [pointer(data["tables"], f"/panels/{p}/rows/14/aspd/mean") for p in (0, 1, 2, 6, 7, 8)]
    require(all(value > 0 for value in native_sp_aspd), "Native/shared Sp-DFA ASPD direction changed")
    for key, row in (("tables", 14), ("iid_seed_first", 2)):
        signs = [pointer(data[key], f"/panels/{p}/rows/{row}/aspd/mean") for p in (3, 4, 5)]
        require(signs[0] > 0 and signs[1] > 0 and signs[2] < 0, "Raw ASPD fixed-panel reversal missing")
        directions.append({"source": key, "pointers": [f"/panels/{p}/rows/{row}/aspd/mean" for p in (3, 4, 5)], "values": signs})
    seed_signs = {}
    for view in ("native", "raw", "shared_calibration"):
        records = data["paired_per_seed"][view]
        chosen = [(i, r) for i, r in enumerate(records) if r["distribution"] == "IID" and r["attack"] == "Sp-DFA"]
        require(len(chosen) == 10 and len({r["seed"] for _, r in chosen}) == 10, "Sp-DFA seed identity")
        require(sum(r["accuracy_pct"] > 0 for _, r in chosen) == 4 and sum(r["accuracy_pct"] < 0 for _, r in chosen) == 6,
                "Accuracy direction count differs")
        seed_signs[view] = {"positive": 4, "negative": 6,
                            "source_pointers": [f"/{view}/{i}/accuracy_pct" for i, _ in chosen],
                            "seeds": [r["seed"] for _, r in chosen]}
    manifest = json.loads((HERE / "EDIT_MANIFEST.json").read_bytes())
    num = lambda s: Counter(re.findall(r"[+−-]?\d+(?:[.,]\d+)*(?:[eE][+−-]?\d+)?", s))
    links = lambda s: Counter(re.findall(r"\]\(([^)]+)\)", s))
    tables = lambda s: Counter(re.findall(r"(?m)(?:^\|[^\n]*\n)+", s))
    reports = []
    new_targets = set()
    for doc in manifest["documents"]:
        source = Path(doc["source"]).read_bytes().decode("utf-8")
        candidate = Path(doc["candidate"]).read_bytes().decode("utf-8")
        require(text_sha(source) == doc["source_sha256"] and text_sha(candidate) == doc["candidate_sha256"], "Document identity")
        forward = source
        for op in doc["operations"]:
            at = op["offset_codepoints"]
            require(text_sha(forward) == op["document_sha256_before"] and forward[at:at + len(op["before"])] == op["before"], "Forward exact span")
            forward = forward[:at] + op["after"] + forward[at + len(op["before"]):]
            require(text_sha(forward) == op["document_sha256_after"], "Forward post-hash")
        require(forward == candidate, "Candidate forward replay")
        reverse = candidate
        for op in reversed(doc["operations"]):
            at = op["offset_codepoints"]
            require(text_sha(reverse) == op["document_sha256_after"] and reverse[at:at + len(op["after"])] == op["after"], "Reverse exact span")
            reverse = reverse[:at] + op["before"] + reverse[at + len(op["after"]):]
            require(text_sha(reverse) == op["document_sha256_before"], "Reverse post-hash")
        require(reverse == source, "Original bytes not recovered")
        require(not (num(source) - num(candidate)), "An original number string is missing")
        require(not (links(source) - links(candidate)), "An original technical link is missing")
        urls = lambda s: Counter(re.findall(r"https?://[^\s<>)]+", s))
        require(not (urls(source) - urls(candidate)), "An original external URL is missing")
        normalized = candidate
        for after, before in manifest["historical_scope_normalization"].items():
            normalized = normalized.replace(after, before)
        # Aside from explicit historical labels, every old prose sentence remains verbatim.
        require(not (Counter(old.sentences(source)) - Counter(old.sentences(normalized))), "Original prose or negative finding removed")
        require(old.quotations(source) == old.quotations(candidate), "Reviewer comment text/order changed")
        require(candidate.count(manifest["new_science_block"]) == 1, "A50 body differs between complete drafts")
        for fact in bindings["mean_sd_pairs"]:
            require(fact["display"] in manifest["new_science_block"], "A50 display absent")
        # P2 is the only changed old table row; every scientific table and other pending row is exact.
        p2 = lambda s: re.sub(r"(?m)^\| P2 —[^\n]*\n", "", s)
        require(tables(p2(source)) == tables(p2(candidate)), "Old table/P1/P3–P6 bytes changed")
        for pattern in (r"\\\[.*?\\\]", r"(?m)^```[^\n]*\n.*?^```"):
            require(re.findall(pattern, source, re.S) == re.findall(pattern, candidate, re.S), "Math/code changed")
        extra_links = links(candidate) - links(source)
        for target in extra_links:
            require(target.startswith(BASE), "Unscoped new link: " + target)
            pin = next((p for p in PINS.values() if p["path"] == target), None)
            require(pin is not None and Path(target).is_file() and sha(Path(target).read_bytes()) == pin["sha256"], "New link/source identity")
            new_targets.add(target)
        preface = {}
        if Path(doc["source"]).name.startswith("rebuttal"):
            require(len(old.quotations(candidate)) == 24, "Expected 24 original comments")
            require(source[:source.index("## Associate Editor")] == candidate[:candidate.index("## Associate Editor")], "Front matter changed")
            preface["front_matter_exact"] = True
        reports.append({"candidate": Path(doc["candidate"]).name, "sha256": text_sha(candidate), "bytes": len(candidate.encode("utf-8")),
                        "edit_operations": len(doc["operations"]), "forward_and_inverse_bytes_exact": True,
                        "original_comment_count": len(old.quotations(candidate)), "original_comment_text_and_order_exact": True,
                        "all_old_number_string_occurrences_preserved": sum(num(source).values()),
                        "added_number_string_occurrences": sum((num(candidate) - num(source)).values()),
                        "all_old_links_preserved": sum(links(source).values()), "added_link_occurrences": sum(extra_links.values()),
                        "all_old_external_URLs_preserved": True,
                        "all_old_scientific_tables_exact": True, "only_P2_pending_row_updated": Path(doc["source"]).name.startswith("rebuttal"),
                        "all_old_prose_preserved_except_declared_historical_labels": True, **preface})
    return {"status": "PASS_A50_READER_INTEGRATION_FOR_AUTHOR_REVIEW", "documents": reports,
            "original_writing_checker_actually_executed": True, "original_checker_sha256": PINS["original_writing_checker"]["sha256"],
            "original_checker_result": old_result, "original_quotations_sentences_digest_reused": True,
            "A50_root_sha256": PINS["A50_root"]["sha256"], "reader_root_sha256": PINS["reader_root"]["sha256"],
            "new_mean_sd_pairs_bound_to_adopted_JSON_pointers": len(bindings["mean_sd_pairs"]),
            "raw_ASPD_fixed_panel_sign_reversals": directions, "SpDFA_accuracy_seed_signs": seed_signs,
            "native_shared_SpDFA_ASPD_positive_in_all_fixed_panels": True,
            "new_link_targets_checked_locally": sorted(new_targets), "external_links_reopened": False,
            "manifest_sha256": sha((HERE / "EDIT_MANIFEST.json").read_bytes()),
            "fact_bindings_sha256": sha((HERE / "FACT_BINDINGS.json").read_bytes()),
            "checker_sha256": sha(Path(__file__).read_bytes()),
            "science_recomputed": False, "canonical_documents_written": False, "author_adopted": False,
            "manuscript_applied": False, "primary_endpoint_selected": False, "final_test": False,
            "complete17_methods_or_all_mechanisms_claimed": False, "new_CNN_fit_training_SSH_bulk_writes": 0}


if __name__ == "__main__":
    require(sys.argv[1:] in ([], ["--check"]), "Only --check is accepted")
    if not sys.argv[1:]:
        build()
    result = check()
    if not sys.argv[1:]:
        save("SELF_CHECK.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
