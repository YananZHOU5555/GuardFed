"""Check the bounded C80 writing references; no metric/statistical computation."""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(value, path):
    for part in path.strip("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def comment(text, number):
    block = text.split("### " + number + " — ", 1)[1]
    block = block.split("**Original comment (verbatim).**", 1)[1]
    return block.split("**Response.**", 1)[0].strip()


def main():
    sources = read(HERE / "SOURCES.json")
    for name, pin in sources["source_pins"].items():
        assert sha(REPO / name) == pin["sha256"], name
    text = (HERE / "ADDENDUM.md").read_text(encoding="utf-8")
    old = (REPO / sources["original_reply"]).read_text(encoding="utf-8")
    for number in ("R3.2", "R3.7"):
        assert comment(old, number) in text, number
    table = read(REPO / sources["table"])
    references = read(HERE / "NUMERIC_REFERENCES.json")
    for ref in references["numeric_references"]:
        actual = pointer(table, ref["pointer"])
        assert actual == ref["value"], ref["pointer"]
        if ref["kind"] == "mean_sd":
            dp = ref["mean_decimals"]
            rendered = f'{actual["mean"]:+.{dp}f} ± {actual["sample_sd_ddof1"]:.{ref["sd_decimals"]}f}'
        else:
            rendered = f'{actual:+.{ref["decimals"]}f}'
        assert rendered == ref["display"] and rendered in text, ref["pointer"]
    root = read(REPO / sources["root"])
    for ref in references["scope_references"]:
        assert pointer(root, ref["pointer"]) == ref["value"], ref["pointer"]
    for i, seeds in enumerate((list(range(91001, 91011)), list(range(91002, 91011)), list(range(91005, 91011)))):
        for offset in (0, 3, 6):
            assert table["panels"][i + offset]["seeds"] == seeds
    records = read(REPO / sources["records"])["records"]
    assert len(records) == 160
    assert all(r["views"]["native"] == r["views"]["shared_calibration"] for r in records)
    # Direction checks read accepted means; they do not refit or recompute them.
    assert table["panels"][0]["rows"][23]["aspd"]["mean"] < 0
    assert all(table["panels"][i]["rows"][23]["aspd"]["mean"] > 0 for i in (1, 2))
    assert table["panels"][3]["rows"][20]["aeod"]["mean"] < 0 < table["panels"][5]["rows"][20]["aeod"]["mean"]
    links = re.findall(r"\]\(([^)]+)\)", text)
    for target in links:
        assert Path(target).is_absolute() and Path(target).is_file(), target
    result = {"status": "C80_ADDENDUM_SOURCE_AND_QUOTATION_CHECKS_PASS", "source_pins": len(sources["source_pins"]), "original_comments_exact": ["R3.2", "R3.7"], "numeric_pointer_displays": len(references["numeric_references"]), "scope_pointer_checks": len(references["scope_references"]), "seed_panels": [10, 9, 6], "native_shared_dictionary_identity_records": 160, "links_checked": len(links), "original_C60_documents_unchanged": True, "new_statistics": 0, "new_inference": 0, "whole_rebuttal_complete": False, "primary_endpoint": "PENDING_AUTHOR"}
    output = HERE / "CHECK_RESULTS.json"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
