"""Read-only checks of the two fixed editorial candidates and reversible edits."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sentences(text):
    # Work line by line; decimal dots and URL dots lack following whitespace.
    return [re.sub(r"\s+", " ", part).strip()
            for line in text.splitlines() if line.strip() and not line.startswith(("#", "|", ">"))
            for part in re.split(r"(?<=[.!?])\s+(?=[A-Z*])", line)]


def quotations(text):
    return re.findall(r"(?m)^\*\*Original comment \(verbatim\)\.\*\*\n\n((?:>[^\n]*\n)+)", text)


def run():
    manifest = json.loads((HERE / "EDIT_MANIFEST.json").read_bytes())
    root_pin = manifest["input_root_review"]
    root_bytes = Path(root_pin["path"]).read_bytes()
    require(hashlib.sha256(root_bytes).hexdigest() == root_pin["sha256"], "Input root review changed")
    root = json.loads(root_bytes)
    navigation = {v["after"]:v["before"] for v in manifest["navigation_only_sentence_changes"]}
    # These exceptions change locations only: the digits occur in cohort labels.
    for after, before in navigation.items():
        require(not re.search(r"\d", re.sub(r"\b(?:A20|C100)\b", "", before + after)),
                "Navigation exception contains a quantitative value")
    reports = []
    for doc in manifest["documents"]:
        source = Path(doc["source"]).read_bytes().decode("utf-8")
        candidate = Path(doc["candidate"]).read_bytes().decode("utf-8")
        require(digest(source) == doc["source_sha256"] == root["documents_sha256"][Path(doc["source"]).name],
                "Canonical source bytes changed or not the accepted input")
        require(digest(candidate) == doc["candidate_sha256"], "Candidate bytes changed")
        # Replay every exact span, checking whole-document SHA before and after.
        forward = source
        for op in doc["operations"]:
            at = op["offset_codepoints"]
            require(digest(forward) == op["document_sha256_before"], "Forward pre-SHA")
            require(forward[at:at + len(op["before"])] == op["before"], "Forward span")
            forward = forward[:at] + op["after"] + forward[at + len(op["before"]):]
            require(digest(forward) == op["document_sha256_after"], "Forward post-SHA")
        require(forward == candidate, "Forward candidate not exact")
        reverse = candidate
        for op in reversed(doc["operations"]):
            at = op["offset_codepoints"]
            require(digest(reverse) == op["document_sha256_after"], "Reverse pre-SHA")
            require(reverse[at:at + len(op["after"])] == op["after"], "Reverse span")
            reverse = reverse[:at] + op["before"] + reverse[at + len(op["after"]):]
            require(digest(reverse) == op["document_sha256_before"], "Reverse post-SHA")
        require(reverse == source, "Inverse did not recover original bytes")

        num = lambda s: Counter(re.findall(r"[+−-]?\d+(?:[.,]\d+)*(?:[eE][+−-]?\d+)?", s))
        links = lambda s: Counter(re.findall(r"\]\(([^)]+)\)", s))
        urls = lambda s: Counter(re.findall(r"https?://[^\s<>)]+", s))
        tables = lambda s: Counter(re.findall(r"(?m)(?:^\|[^\n]*\n)+", s))
        math = lambda s: Counter(re.findall(r"\\\[.*?\\\]", s, flags=re.S))
        code = lambda s: Counter(re.findall(r"(?m)^```[^\n]*\n.*?^```", s, flags=re.S))
        for name, extract in (("number strings", num), ("Markdown links", links), ("raw URLs", urls),
                              ("whole tables", tables), ("math blocks", math), ("code blocks", code)):
            require(extract(source) == extract(candidate), name + " differ")
        original_sentences, edited_sentences = sentences(source), sentences(candidate)
        normalized = [navigation.get(s, s) for s in edited_sentences]
        numeric_old = Counter(s for s in original_sentences if re.search(r"\d", s))
        numeric_new = Counter(s for s in normalized if re.search(r"\d", s))
        require(numeric_old == numeric_new, "A number-containing sentence changed beyond declared navigation")
        require(not (Counter(original_sentences) - Counter(normalized)),
                "An original prose sentence or negative finding is missing")
        changed_navigation = [{"before": before, "after": after,
                               "occurrences": edited_sentences.count(after)}
                              for after, before in navigation.items() if after in edited_sentences]
        old_quotes, new_quotes = quotations(source), quotations(candidate)
        require(old_quotes == new_quotes, "Original-comment bytes or order changed")
        if Path(doc["source"]).name.startswith("rebuttal"):
            require(len(old_quotes) == 24, "Expected exactly 24 Original comments")
            pending = lambda s: re.search(r"(?m)(?:^\|[^\n]*\n)+", s[s.index("## Pending register"):]).group(0)
            require(pending(source) == pending(candidate), "Pending table changed")
            first_response = lambda s: len(re.findall(r"\b[\w’'-]+\b", s[:s.index("## Associate Editor")]))
            preface = {"source_words_before_first_response":first_response(source),
                       "candidate_words_before_first_response":first_response(candidate)}
        else:
            preface = {}
        reports.append({"candidate":Path(doc["candidate"]).name, "source_sha256":digest(source),
                        "candidate_sha256":digest(candidate), "edit_operations":len(doc["operations"]),
                        "forward_exact":True, "inverse_original_bytes_exact":True,
                        "original_comments_exact_order":len(old_quotes), "number_strings_multiset_exact":True,
                        "number_string_occurrences":sum(num(source).values()),
                        "Markdown_links_multiset_exact":True, "Markdown_link_occurrences":sum(links(source).values()),
                        "raw_URL_multiset_exact":True, "whole_tables_exact":True,
                        "table_blocks":sum(tables(source).values()), "math_code_blocks_exact":True,
                        "numeric_prose_sentences":sum(numeric_old.values()),
                        "quantitative_sentences_changed":0,
                        "declared_cohort_label_navigation_changes":changed_navigation,
                        "all_original_prose_retained_except_declared_navigation":True, **preface})
    return {"status":"PASS_EDITORIAL_CANDIDATES_FOR_AUTHOR_REVIEW", "documents":reports,
            "input_root_review_sha256":root_pin["sha256"],
            "edit_manifest_sha256":hashlib.sha256((HERE / "EDIT_MANIFEST.json").read_bytes()).hexdigest(),
            "checker_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "science_recomputed":False, "external_links_reopened":False,
            "new_scientific_results":0, "canonical_documents_written":False,
            "manuscript_applied":False, "author_adopted":False,
            "final_test":False, "full17_completed":False, "new_CNN_fit_training_SSH_bulk_writes":0}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, allow_nan=False))
