"""Classify saved public responses without importing or executing downloaded code."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"


def main():
    receipts = []
    for file in sorted(ROOT.glob("*_receipts.json")):
        if file.name == "classified_receipts.json":
            continue
        receipts.extend(json.loads(file.read_text(encoding="utf-8")))
    for row in receipts:
        file = ROOT / row["saved_as"]
        assert hashlib.sha256(file.read_bytes()).hexdigest() == row["sha256"], file
        row["sha_verified"] = True
        row["kind"] = "source_code" if row["name"].startswith(("btp_", "hyperledger_")) and row["name"].endswith("_py") else "metadata_or_landing_page"
        row["target_paper_fulltext"] = False

    code_checks = []
    repositories = []
    for label, tree_name, owner_repo, raw_prefix in [
        ("btp", "feddna_author_btp_tree", "Adityagarg8384/btp", "btp_"),
        ("hyperledger", "feddna_author_hyperledger_tree", "Adityagarg8384/Federated-Learning-Using-Hyperledger-fabric", "hyperledger_"),
    ]:
        commit = json.loads((RAW / (label + "_commit.json")).read_text())
        tree = json.loads((RAW / (tree_name + ".json")).read_text())
        # GitHub's tree endpoint for a branch reports its resolved commit SHA.
        assert commit["sha"] == tree["sha"], (commit["sha"], tree["sha"])
        assert not tree["truncated"]
        blobs = {entry["path"]: entry for entry in tree["tree"] if entry["type"] == "blob"}
        saved = []
        for path, entry in blobs.items():
            filename = raw_prefix + path.replace("/", "_").replace(".", "_") + ".html"
            local = RAW / filename
            if not local.exists():
                continue
            content = local.read_bytes()
            git_blob_sha1 = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            assert git_blob_sha1 == entry["sha"], path
            saved.append(path)
            if path.endswith(".py"):
                text = content.decode("utf-8-sig")
                try:
                    ast.parse(text, filename=path)
                    parse_status = "PASS"
                except SyntaxError as exc:
                    parse_status = {"error": type(exc).__name__, "line": exc.lineno, "message": exc.msg}
                target_terms = [word for word in ["feddna", "fingerprint", "median_absolute_deviation", "median_abs_deviation", "adaptive_threshold", "probe"] if word in text.lower()]
                code_checks.append({"repo": owner_repo, "commit": commit["sha"], "path": path,
                                    "sha256": hashlib.sha256(content).hexdigest(), "git_blob_sha1_verified": True,
                                    "ast_parse": parse_status, "target_terms": target_terms})
        repositories.append({"repo": owner_repo, "commit": commit["sha"], "commit_tree_sha": commit["commit"]["tree"]["sha"],
                             "resolved_tree_sha": tree["sha"], "truncated": False, "saved_paths": saved,
                             "nonempty_python_total": len([entry for path, entry in blobs.items() if path.endswith(".py") and entry.get("size", 0) > 0]),
                             "saved_python": len([path for path in saved if path.endswith(".py")])})

    institution = (RAW / "smartfl_boyu_institution.html").read_text(encoding="utf-8")
    links = re.findall(r'href\s*=\s*["\']([^"\']+)', institution, re.IGNORECASE)
    relevant_links = [url for url in links if any(term in url.lower() for term in ["smartfl", ".pdf", "github"])]
    result = {"status": "BOUNDED_SOURCE_SEARCH_COMPLETE_SPECIFICATION_STILL_INCOMPLETE", "response_count": len(receipts),
              "response_sha_checks": "PASS", "repositories": repositories, "source_code_checks": code_checks,
              "smartfl_boyu_institution": {"exact_title_present": "SmartFL" in institution,
                                          "pdf_github_smartfl_links": relevant_links},
              "feddna_fulltext_found": False, "smartfl_fulltext_found": False, "faithful_adapter_unblocked": False,
              "negative_evidence_scope": "Only these saved responses and repository commits; not proof that public full text or other code cannot exist."}
    (ROOT / "classified_receipts.json").write_text(json.dumps(receipts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / "source_inspection.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"responses": len(receipts), "code_files": len(code_checks), "repositories": repositories,
                      "smartfl_links": relevant_links, "status": result["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
