"""A60 prose-only extension; reuse the original writing-check AST without a parallel checker."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "SOURCE_PINS.json").read_bytes())
DOCS = ("rebuttal_integrated_20261010.md", "manuscript_insertions_integrated_20261010.md")
BASE = str(Path(PINS["tables"]["path"]).parent.as_posix()) + "/"


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(key):
    pin = PINS[key]
    data = Path(pin["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == pin["sha256"] and len(data) == pin["bytes"], key
    return json.loads(data)


def save(name, value):
    (HERE / name).write_bytes((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def original_checks():
    """Load exact helper functions and the original document-invariant loop, not its old cohort execution."""
    source_path = PINS["A50_build_checker"]["path"]
    source = Path(source_path).read_bytes().decode()
    assert file_sha(source_path) == PINS["A50_build_checker"]["sha256"]
    tree = ast.parse(source)
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("sha", "text_sha", "require", "pointer")]
    exec(compile(ast.Module(body=helpers, type_ignores=[]), source_path, "exec"), globals())
    writing_path = PINS["original_writing_checker"]["path"]
    writing = Path(writing_path).read_bytes().decode()
    assert file_sha(writing_path) == PINS["original_writing_checker"]["sha256"]
    functions = [node for node in ast.parse(writing).body if isinstance(node, ast.FunctionDef) and node.name in ("sentences", "quotations")]
    scope = dict(re=re)
    exec(compile(ast.Module(body=functions, type_ignores=[]), writing_path, "exec"), scope)
    globals()["old"] = SimpleNamespace(**{node.name: scope[node.name] for node in functions})
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "check")
    first = next(i for i, node in enumerate(original.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "manifest" for target in node.targets))
    body = original.body[first:-1]
    assert isinstance(original.body[-1], ast.Return)
    # These statements are executed unchanged; the old A50-specific facts/old checker.run() are not rerun.
    loop_sha = hashlib.sha256(ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False).encode()).hexdigest()
    returned = ast.parse("return {'documents': reports, 'new_link_targets_checked_locally': sorted(new_targets)}").body[0]
    fn = ast.FunctionDef(name="check_documents_original", args=original.args, body=body + [returned],
                         decorator_list=[], returns=None, type_comment=None)
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(module, source_path + " [unchanged document checks]", "exec"), globals())
    return {"source_sha256": PINS["A50_build_checker"]["sha256"], "document_check_AST_sha256": loop_sha,
            "document_check_statements_reused_unchanged": len(body),
            "original_quotation_sentence_helpers_sha256": PINS["original_writing_checker"]["sha256"],
            "old_cohort_checks_rerun": False}


REUSE = original_checks()


def build():
    root, table = read("A60_root"), read("tables")
    facts = []

    def pair(panel, metric, shared=None):
        rows = table["panels"][panel]["rows"]
        index = next(i for i, row in enumerate(rows) if (row["distribution"], row["attack"], row["variant"]) == ("non-IID", "Benign", "minus_A minus Full"))
        path = f"/panels/{panel}/rows/{index}/{metric}"
        value = pointer(table, path)
        digits = 3 if metric == "accuracy_pct" else 5
        display = f"{value['mean']:+.{digits}f} ± {value['sample_sd_ddof1']:.{digits}f}"
        facts.append({"source": "tables", "mean_pointer": path + "/mean", "sd_pointer": path + "/sample_sd_ddof1",
                      "display": display, "digits": digits, "mean": value["mean"], "sd": value["sample_sd_ddof1"],
                      "shared_pointer": None if shared is None else path.replace(f"/panels/{panel}/", f"/panels/{shared}/")})
        return display

    metrics = ("accuracy_pct", "aeod", "aspd")
    native = " / ".join(pair(0, metric, 6) for metric in metrics)
    raw = " / ".join(pair(3, metric) for metric in metrics)
    block = f"""**Current A60 extension: the first complete non-IID A scene.** The [accepted A60 table]({BASE}TABLES.md) and [root verification]({BASE}ROOT_VERIFICATION.json) retain the five IID scenes above and add non-IID Benign with ten matched seeds, giving six complete scenes and 60 Full/minus_A pairs (120 records). The formerly isolated non-IID Benign seed 91001 now belongs to this complete scene; its exclusion described in the historical A50 snapshot does not apply to the current per-scene table. All three views use each model’s same terminal round-70 checkpoint and 19,867 validation images. Deleting A removes its score contribution while geometric hard filtering and prediction calibration remain enabled.

**Added non-IID Benign evidence.** The ten-seed paired differences, minus_A−Full, are {native} for native/shared calibration and {raw} for raw, in ΔACC / ΔAEOD / ΔASPD order. Values are means ± sample SD (ddof=1); ΔACC is in percentage points and disparity differences are absolute gaps. In every view and each fixed 10/9/6-seed panel, deletion lowers mean ACC and mean ASPD but raises mean AEOD. Thus the result preserves a utility–disparity trade-off rather than an advantage on all three metrics. These are paired-mean directions, not a claim that every seed changes in the same direction, a significance test or evidence of component necessity. All panels and unfavorable outcomes remain in the linked table and [saved numerical values]({BASE}tables.json).

**Aggregation and evaluation boundary.** The [five-IID seed-first aggregate]({BASE}IID_SEED_FIRST.json) is preserved byte for byte from A50: scenes are averaged within each seed before the across-seed mean and sample SD. Non-IID Benign is reported separately and does not enter this aggregate; no mean over the imbalanced six-scene set is supplied. The fixed nine-seed panel excludes configuration-selection seed 91001 and the fixed six-seed panel retains 91005–91010, identically for Full and minus_A. Root-only calibration, exposed validation and prior official-test exposure remain disclosed; sensitivity subsetting does not restore an untouched confirmation set. AEOD remains the absolute TPR gap, not full equalized odds.

**Current A60 comparability and remaining scope.** Full replay comprises five CPU and 55 GPU checkpoints; all 60 minus_A replays use CPU. Full training provenance comprises 59 cu128 and one cu130 checkpoints, while all 60 minus_A checkpoints report cu128. The A50 environment counts above apply only to that historical IID subset. Mixed devices, training builds and driver provenance do not establish numerical or complete-trajectory equivalence. Native/shared metric values and saved group counts coincide within each of the 120 records, so these views are not independent confirmations or evidence of an additional calibration gain. The remaining four non-IID A scenes—F Flip, FedSA, S-DFA and Sp-DFA—and the other five image-control variants remain incomplete. The final primary endpoint, frozen final evaluation and submitted-manuscript integration remain pending; no final test or complete all-component study is claimed.

"""
    manifest = {"status": "REVERSIBLE_A60_INTEGRATION_FOR_AUTHOR_REVIEW", "input_reader_root": PINS["reader_root"],
                "input_A60_root": PINS["A60_root"], "documents": [], "historical_scope_normalization": {},
                "new_science_block": block, "new_scientific_measurements": 0}
    labels = {
        "**Current A50 extension: all five IID scenes.**": "**Historical A50 extension: all five IID scenes.**",
        "**Added IID Sp-DFA evidence.**": "**Historical A50 IID Sp-DFA evidence.**",
        "**Five-IID seed-first summary.**": "**Preserved five-IID seed-first summary.**",
        "**Current A50 comparability and scope.**": "**Historical A50 comparability and scope.**",
        "The new A50 IID Sp-DFA comparison reinforces this boundary:": "The historical A50 IID Sp-DFA comparison reinforces this boundary:",
        "The five-IID seed-first summary in R3.2 preserves these view and panel distinctions; it does not include the single non-IID Benign A record or complete the five non-IID scenes.": "The seed-first summary in R3.2 provides a broader five-IID summary; it averages only the IID scenes and remains separate from the added non-IID Benign comparison.",
    }
    manifest["historical_scope_normalization"] = {after: before for before, after in labels.items()}
    for name in DOCS:
        source = Path(PINS[name]["path"]).read_bytes().decode("utf-8")
        value, operations = source, []

        def edit(before, after, reason):
            nonlocal value
            require(value.count(before) == 1, "Nonunique exact span: " + reason)
            at = value.index(before)
            op = {"reason": reason, "offset_codepoints": at, "before": before, "after": after,
                  "document_sha256_before": text_sha(value)}
            value = value[:at] + after + value[at + len(before):]
            op["document_sha256_after"] = text_sha(value)
            operations.append(op)

        for before, after in labels.items():
            if before in value:
                edit(before, after, "Historical A50 scope or explicit IID-aggregate interpretation; scientific values unchanged")
        anchor = "**Root/heterogeneity paragraph.**" if name.startswith("manuscript") else next(line for line in value.splitlines() if line.startswith("### R3.3 —"))
        edit(anchor, block + anchor, "Add adopted A60 non-IID Benign evidence, environment and scope")
        history = next(line for line in value.splitlines() if line.startswith("The [accepted A50 comparison]")) + "\n\n"
        current = f"The [accepted A60 comparison]({BASE}TABLES.md) covers six complete scenes: all five IID scenes and non-IID Benign, with 60 matched Full/minus_A pairs and fixed 10/9/6-seed panels. The other four non-IID A scenes and five other image-control variants remain incomplete. The five-IID aggregate is preserved separately, with no mixed-distribution aggregate; U100 and C100 are unchanged.\n\n"
        edit(history, current, "Point current accepted snapshot to A60")
        lineage = next(line for line in value.splitlines() if line.startswith("The later [accepted A40 comparison]")) + "\n\n"
        edit(lineage, lineage + "**Historical A50 snapshot.**\n\n" + history, "Retain original A50 snapshot under historical lineage")
        if name.startswith("rebuttal"):
            ae_old = next(line for line in value.splitlines() if line.startswith("The A-deletion extension now completes all five IID scenes"))
            ae = "The A-deletion extension now covers six complete CelebA scenes with 60 matched Full/minus_A pairs: all five IID scenes and non-IID Benign. In the added non-IID scene, deletion lowers mean accuracy and ASPD but raises mean AEOD under raw, native and shared prediction. R3.2/R3.7 retain the paired results, fixed sensitivity panels and limits; the other four non-IID A scenes and five other image controls remain pending."
            edit(ae_old, ae, "Update the Associate Editor answer to current A60 evidence")
            hist_anchor = "**Historical A50 snapshot.**\n\n" + history
            edit(hist_anchor, hist_anchor + "**Historical A50 Associate Editor summary.**\n\n" + ae_old + "\n\n", "Retain original AE scientific text and numbers in historical scope")
            guide = next(line for line in value.splitlines() if line.startswith("**Image-control reading guide.**"))
            edit(guide, guide + "\n\nThe current A comparison covers five IID scenes plus non-IID Benign, with 60 paired checkpoints; the remaining four non-IID A scenes and five other control variants remain pending.", "Synchronize the mechanism overview with A60")
            r37_anchor = "The tabular COMPAS counterexamples above remain unchanged."
            r37 = "The added A60 non-IID Benign comparison supplies a further counterexample to uniform component benefit: deleting A lowers mean accuracy and ASPD but raises mean AEOD in every reported prediction view and fixed seed panel. The current evidence therefore supports bounded utility–disparity trade-offs, without isolating a causal mechanism or completing the other four non-IID A scenes. "
            edit(r37_anchor, r37 + r37_anchor, "Add the current non-IID interpretation without three-metric dominance")
            p2 = next(line for line in value.splitlines() if line.startswith("| P2 —"))
            p2_new = p2.replace("A50 adds IID Sp-DFA, completing all five IID scenes with 50 paired checkpoints; complete all five non-IID A scenes and the other five incomplete image controls.", "The historical A50 extension adds IID Sp-DFA, completing all five IID scenes with 50 paired checkpoints. A60 adds non-IID Benign, giving six complete scenes and 60 pairs; complete the other four non-IID A scenes and the other five incomplete image controls.")
            require(p2_new != p2, "P2 update did not match")
            edit(p2, p2_new, "Update only P2 current completed/pending coverage")
        target = name.replace("_20261010.md", "_A60_reader_20261011.md")
        (HERE / target).write_bytes(value.encode())
        manifest["documents"].append({"source": PINS[name]["path"], "source_sha256": text_sha(source),
                                      "candidate": (HERE / target).as_posix(), "candidate_sha256": text_sha(value), "operations": operations})
    save("EDIT_MANIFEST.json", manifest)
    directions = []
    for i, panel in enumerate(table["panels"]):
        j = next(j for j, row in enumerate(panel["rows"]) if (row["distribution"], row["attack"], row["variant"]) == ("non-IID", "Benign", "minus_A minus Full"))
        directions.append({"view": panel["view"], "seed_count": len(panel["seeds"]), "seeds_pointer": f"/panels/{i}/seeds",
                           "mean_pointers": {m: f"/panels/{i}/rows/{j}/{m}/mean" for m in metrics}})
    save("FACT_BINDINGS.json", {"mean_sd_pairs": facts, "nonIID_Benign_direction_panels": directions,
         "scope": [{"source": "A60_root", "pointer": path, "expected": pointer(root, path)} for path in
                   ("/paired_models", "/preserved_records", "/complete_scenes", "/complete_IID_scenes", "/complete_nonIID_scenes", "/replay_devices", "/training_torch", "/seed_panels", "/IID_seed_first_JSON_bytes_exact", "/aggregate_scope", "/test", "/primary_endpoint_selected")],
         "native_shared_equality": {"source": "summary", "pointer": "/native_shared_metrics_and_counts_exact", "expected": True},
         "source_statistics_recomputed": False})


def check():
    for key, pin in PINS.items():
        data = Path(pin["path"]).read_bytes()
        require(hashlib.sha256(data).hexdigest() == pin["sha256"] and len(data) == pin["bytes"], "Input drift: " + key)
    root, reader, table, summary = read("A60_root"), read("reader_root"), read("tables"), read("summary")
    require(reader["author_review_only"] and reader["A50_incorporated"] and reader["documents_sha256"] == {n: PINS[n]["sha256"] for n in DOCS}, "A50 reader/root identity")
    require(root["root_adoption"] and root["paired_models"] == 60 and root["complete_IID_scenes"] == 5 and root["complete_nonIID_scenes"] == ["Benign"], "A60 adoption/scope")
    for key in ("tables", "iid_seed_first", "summary", "official_table", "saved_verification"):
        require(root["files_sha256"][Path(PINS[key]["path"]).name] == PINS[key]["sha256"], "A60 canonical/root SHA")
    facts = json.loads((HERE / "FACT_BINDINGS.json").read_bytes())
    for fact in facts["mean_sd_pairs"]:
        mean, sd = pointer(table, fact["mean_pointer"]), pointer(table, fact["sd_pointer"])
        require((mean, sd) == (fact["mean"], fact["sd"]), "Mean/SD pointer")
        require(f"{mean:+.{fact['digits']}f} ± {sd:.{fact['digits']}f}" == fact["display"], "Numeric display")
        if fact["shared_pointer"]:
            require(pointer(table, fact["shared_pointer"]) == {"mean": mean, "sample_sd_ddof1": sd}, "Shared/native equality")
    for panel in facts["nonIID_Benign_direction_panels"]:
        signs = {m: pointer(table, path) for m, path in panel["mean_pointers"].items()}
        require(signs["accuracy_pct"] < 0 and signs["aeod"] > 0 and signs["aspd"] < 0, "Non-IID fixed-panel mean direction")
        require(len(pointer(table, panel["seeds_pointer"])) == panel["seed_count"], "Panel seed identity")
    for fact in facts["scope"] + [facts["native_shared_equality"]]:
        require(pointer(root if fact["source"] == "A60_root" else summary, fact["pointer"]) == fact["expected"], "Root/scope fact")
    globals()["bindings"] = facts
    result = check_documents_original()
    result.update(status="PASS_A60_READER_INTEGRATION_AUTHOR_REVIEW_ONLY", original_document_checks_actually_executed_once=True,
                  original_check_reuse=REUSE, initial_bridge_failure_preserved="CHECK_BRIDGE_FAILURE.json", new_mean_sd_pairs_bound_to_JSON_pointers=len(facts["mean_sd_pairs"]),
                  fixed_10_9_6_mean_directions_bound=True, root_reader_sha256=PINS["reader_root"]["sha256"],
                  A60_root_sha256=PINS["A60_root"]["sha256"], source_statistics_recomputed=False,
                  self_check=True, independent_review=False, canonical_written=False, manuscript_applied=False,
                  author_review=True, primary_endpoint_selected=False, final_test=False, new_CNN_fit_training_SSH_Git_bulk_writes=0,
                  manifest_sha256=file_sha(HERE / "EDIT_MANIFEST.json"), fact_bindings_sha256=file_sha(HERE / "FACT_BINDINGS.json"),
                  checker_sha256=file_sha(__file__))
    return result


if __name__ == "__main__":
    require(sys.argv[1:] in ([], ["--check"]), "Only --check is accepted")
    started = datetime.now(timezone.utc).isoformat()
    if not sys.argv[1:]:
        build()
    result = check()
    if not sys.argv[1:]:
        save("SELF_CHECK.json", result)
        save("ACTUAL_COMMAND.json", {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
             "command": "python -B tmp/rebuttal_A60_reader_integration_20261011/build_and_check.py", "exit_code": 0,
             "candidate_check_invocations": 2, "initial_partial_check_exit_code": 1,
             "complete_successful_candidate_checks": 1, "science_statistics_recomputed": False})
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
