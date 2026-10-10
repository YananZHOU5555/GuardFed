"""Exact, reversible metadata/scope bridge to the sealed A50 sources."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

H = Path(__file__).resolve().parent
R = H.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def mapped(name):
    contract = json.loads((H / "SOURCE_ADAPTATIONS.json").read_bytes())[name]
    source = (R / contract["source"]).read_bytes().decode("utf-8")
    need(hashlib.sha256(source.encode()).hexdigest() == contract["sha256"], "A50 source changed: " + name)
    text = source
    for before, after in contract["replacements"]:
        need(text.count(before) == 1, "Scope replacement must occur once: " + before)
        text = text.replace(before, after, 1)
    inverse = text
    for before, after in reversed(contract["replacements"]):
        need(inverse.count(after) == 1, "Ambiguous inverse replacement: " + after)
        inverse = inverse.replace(after, before, 1)
    need(inverse == source, "A50 inverse is not byte exact")
    ast.parse(text)
    return text


def namespace(name, filename):
    scope = {"__file__": filename, "__name__": "A60_scoped_source_no_main"}
    exec(compile(mapped(name), filename + " [exact scoped A50 bridge]", "exec"), scope)
    return {key: value for key, value in scope.items() if key not in ("__file__", "__name__", "__builtins__")}


def preflight(name):
    need(not sys.flags.optimize, "Optimized Python forbidden")
    binding_path = H / "ROOT_BINDING.json"
    need(binding_path.is_file(), "Actual root-adopted MECHANISM260 inputs not bound; no generation authorized yet")
    binding = json.loads(binding_path.read_bytes())
    need(binding["status"] == "ACTUAL_ROOT260_BOUND_FOR_A60_TABLE" and binding["root_adopted"] is True,
         "Actual root adoption required")
    for key in ("adoption", "index"):
        need(sha(R / binding[key]) == binding[key + "_sha256"], "Actual root/index pin drift")
    if name == "build.py":
        parser = argparse.ArgumentParser()
        for key in ("adoption", "adoption-sha256", "index", "index-sha256"):
            parser.add_argument("--" + key, required=True)
        args = vars(parser.parse_args())
        need(all(args[key] == binding[key] for key in args), "CLI differs from frozen ROOT_BINDING")
    else:
        proof = json.loads((H / "SOURCE_BINDINGS.json").read_bytes())
        need(proof["actual_A60_root_adoption_sha256"] == binding["adoption_sha256"]
             and proof["accepted_index_sha256"] == binding["index_sha256"], "Generated source/root binding mismatch")
