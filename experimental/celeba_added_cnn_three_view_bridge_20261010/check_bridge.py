"""Focused metadata rejection and original zero-margin-rule checks; no model/fit."""
import copy
import json
import sys

import numpy as np

import bridge


def main():
    passed = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        passed.append(name)
    def reject(name, call, expected):
        try:
            call()
        except ValueError as e:
            check(name, expected in str(e))
        else:
            raise AssertionError("Accepted: " + name)

    pins = bridge.read_pin({"path": str(bridge.HERE / "PROOF_PINS.json"), "sha256": bridge.PROOFS_SHA256})
    actual = {}
    for method, chain in pins["chains"].items():
        jid = next(iter(chain["records"]))
        record = bridge.identity_record(method, jid)
        check(method + ": actual adopted metadata chain", record["id"] == jid)
        evidence = chain["records"][jid]
        strict = bridge.read_pin(chain["strict"])
        off = bridge.read_pin(chain["offserver"])
        off_records = off["local"]["records"] if method == "CosineFairnessHybrid" else off["records"]
        args = [method] + [bridge.read_pin(evidence[k]) for k in ("job", "result", "provenance", "acceptance")]
        args += [next(r for r in strict["records"] if r["id"] == jid),
                 next(r for r in off_records if r["id"] == jid),
                 bridge.read_pin(evidence["scope"]) if "scope" in evidence else None]
        actual[method] = args
        # Independent changed metadata, keeping other actual checkpoint/proof fields fixed.
        mutations = [
            ("wrong method", 2, lambda d: d.update(method="FedAvg"), "Wrong method"),
            ("wrong config", 2, lambda d: d["config"].update(learning_rate=0.123), "Wrong config"),
            ("wrong seed", 2, lambda d: d.update(seed=91010 if d["seed"] != 91010 else 91009), "Wrong seed"),
            ("wrong source", 3, lambda d: d["source_hashes"].update({"scripts/reproduce_paper_tables.py": "0" * 64}), "source"),
            ("wrong checkpoint", 4, lambda d: d["artifact_hashes"].update({"model.pt": "0" * 64}), "Wrong checkpoint"),
            ("wrong proof record", 5, lambda d: d.update(id="wrong_job"), "Wrong proof record ID"),
            ("partial horizon", 2, lambda d: d.update(rounds=69), "70-round"),
        ]
        for name, index, mutate, expected in mutations:
            altered = copy.deepcopy(args)
            mutate(altered[index])
            reject(method + ": " + name, lambda a=altered: bridge.validate_metadata(*a), expected)
        bad_pin = dict(chain["root"], sha256="0" * 64)
        reject(method + ": wrong root bytes", lambda p=bad_pin: bridge.read_pin(p), "Pinned bytes changed")
        reject(method + ": unadopted job", lambda m=method: bridge.identity_record(m, "not_adopted"), "adopted exact chunk")

    reject("Huber cannot invent a root proof", lambda: bridge.identity_record("Huber-BRFL-gradient", "Huber_delta0.1"), "No adopted")
    reject("LoGoFair is not registered", lambda: bridge.identity_record("LoGoFair", "ordinary_cnn"), "Unsupported")
    science = bridge.science_bindings()
    check("four private labels only", science.METHODS == set(bridge.METHODS))
    # Construct sealed prediction-stage fixtures; no threshold fitting takes place.
    margins = np.array([-1e-9, -0.0, 0.0, 1e-9])
    sensitive = np.array([0, 0, 1, 1])
    for method in sorted(bridge.METHODS):
        fits = {}
        for view in ("native", "raw", "shared_calibration"):
            shared = view == "shared_calibration"
            fits[view] = science._seal_fit({"view": view, "method": method,
                "rule": "group_margin_greater_equal" if shared else "argmax_margin_strictly_positive",
                "thresholds": {0: 0.0, 1: 0.0} if shared else None,
                "fit_data": "clean_train_root_only" if shared else "none"})
        pred = science.predict_views(margins, sensitive, fits)
        check(method + ": native/raw strictly positive incl negative zero", pred["native"].tolist() == pred["raw"].tolist() == [0, 0, 0, 1])
        check(method + ": shared threshold tie inclusive", pred["shared_calibration"].tolist() == [0, 1, 1, 1])
        forged = copy.deepcopy(fits)
        forged["native"]["method"] = "LoGoFair"
        forged["native"] = science._seal_fit({k:v for k,v in forged["native"].items() if k != "fit_sha256"})
        reject(method + ": LoGo native fit refused", lambda f=forged: science.predict_views(margins, sensitive, f), "Prediction fit identity changed")

    reuse = bridge.read_pin({"path": str(bridge.HERE / "SOURCE_REUSE.json"), "sha256": bridge.REUSE_SHA256})
    functions_checked = 0
    for entry in reuse["function_sources"]:
        source = open(entry["file"]["path"], encoding="utf-8").read()
        # Full original compilation is inert: no imports or module-level code run.
        original_code = compile(source, entry["file"]["path"], "exec", dont_inherit=True)
        definitions = {c.co_name:c for c in original_code.co_consts if hasattr(c, "co_code")}
        for name in entry["functions"]:
            selected = getattr(science, name).__code__
            check("original function bytecode: " + name,
                  selected.co_code == definitions[name].co_code
                  and selected.co_flags == definitions[name].co_flags)
            functions_checked += 1
    check("native tolerance unchanged", science.TOLERANCE == 1e-12)
    check("no Torch loaded", "torch" not in sys.modules)
    return {"status":"PASS_SOURCE_IDENTITY_GATES_ONLY", "checks_passed":len(passed),
            "checks":passed, "actual_metadata_positive_methods":sorted(actual),
            "original_functions_bytecode_checked":functions_checked,
            "synthetic_margin_fixture_only":True, "new_CNN_calls":0, "new_fit_calls":0,
            "new_training_calls":0, "Torch_imported":False, "SSH_calls":0,
            "scientific_evaluation_completed":False, "dispatch_authorized":False,
            "Huber_existing_70round_root_proof_available":False,
            "proof_pins_sha256":bridge.PROOFS_SHA256, "source_reuse_sha256":bridge.REUSE_SHA256,
            "checker_source_sha256":bridge.digest(__file__), "bridge_sha256":bridge.digest(bridge.__file__)}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, allow_nan=False))
