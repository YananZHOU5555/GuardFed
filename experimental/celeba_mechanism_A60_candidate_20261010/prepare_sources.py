"""Prepare only source/metadata wiring; never read models or execute table generation."""
import ast
import hashlib
import json
from pathlib import Path

H = Path(__file__).resolve().parent
R = H.parents[1]
A50 = R / "tmp/celeba_mechanism_A50_IID_candidate_20261010"
P = "docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_IID_five_scenes50_20261010"
S5 = "[('IID','Benign'),('IID','F Flip'),('IID','FedSA'),('IID','S-DFA'),('IID','Sp-DFA')]"
S6 = S5[:-1] + ",('non-IID','Benign')]"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def save(name, value):
    with (H / name).open("x", encoding="utf8", newline="\n") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")


def main():
    assert not (H / "SOURCE_ADAPTATIONS.json").exists(), "Fresh preparation only"
    source_seal = A50 / "FILES_SHA256.json"
    assert sha(source_seal) == "0600c7ac5d04f8be45c9acf20cc00bb6162f618b46340fd11b6a3619dbf52f70"
    sealed = json.loads(source_seal.read_bytes())["files"]
    adaptations = {}

    def source(name, pairs):
        data = (A50 / name).read_bytes()
        assert sha(A50 / name) == sealed[name]["sha256"] and len(data) == sealed[name]["bytes"]
        adaptations[name] = {"source": (A50 / name).relative_to(R).as_posix(), "sha256": sha(A50 / name),
                             "bytes": len(data), "replacements": pairs}

    source("build.py", [
        ("Read actual adopted A51 receipts; display only complete IID A50", "Read actual adopted A60 receipts; display five IID scenes plus non-IID Benign"),
        ("PREV = R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_IID_four_scenes40_20261010'", "PREV = R/'" + P + "'"),
        ("SCENES = [('IID', 'Benign'), ('IID', 'F Flip'), ('IID', 'FedSA'), ('IID', 'S-DFA'), ('IID', 'Sp-DFA')]", "SCENES = [('IID', 'Benign'), ('IID', 'F Flip'), ('IID', 'FedSA'), ('IID', 'S-DFA'), ('IID', 'Sp-DFA'), ('non-IID', 'Benign')]"),
        ("new_ids = [f'minus_A_IID_Sp-DFA_seed{seed}' for seed in range(91001,91011)]+['minus_A_non-IID_Benign_seed91001']", "new_ids = [f'minus_A_non-IID_Benign_seed{seed}' for seed in range(91002,91011)]"),
        ("adopted['status']=='ROOT_AFTER240_EXACT11_SAVED_ARRAYS_NATIVE251_ADOPTED'", "adopted['status']==read(H/'ROOT_BINDING.json')['root_status']"),
        ("adopted['new_accepted']==11 and adopted['prior_accepted']==240", "adopted['new_accepted']==9 and adopted['prior_accepted']==251"),
        ("adopted['cumulative_accepted']==251 and adopted['original240_unchanged']", "adopted['cumulative_accepted']==260 and adopted['original251_unchanged']"),
        ("Actual exact11 root adoption required", "Actual exact9 root adoption required"),
        ("Follow six SHA-bound adopted increments", "Follow seven SHA-bound adopted increments"),
        ("len(visited)<6", "len(visited)<7"),
        ("[12,8,8,8,4,11]", "[12,8,8,8,4,11,9]"),
        ("[212,220,228,236,240,251]", "[212,220,228,236,240,251,260]"),
        ("Wrong exact six A increments", "Wrong exact seven A increments"),
        ("Five complete IID scenes required", "Five complete IID scenes plus non-IID Benign required"),
        ("'edf4a520c3b797e2376005ef132bc21021ab5dbd0962b4105299d67309f8ebf9'", "'2ab91a854691624c79124a5eccbb43efe30541474cb7ef542b76db300554f7c4'"),
        ("Original A40 root adoption changed", "Original A50 root adoption changed"),
        ("['build.py','ROOT_VERIFICATION.json','FILES_SHA256.json','records.json','tables.json','TABLES.md']", "['ROOT_VERIFICATION.json','records.json','tables.json','TABLES.md','IID_SEED_FIRST.json']"),
        ("for name,pin in read(PREV/'FILES_SHA256.json')['files'].items():\n        need(sha(PREV/name)==pin['sha256'] and (PREV/name).stat().st_size==pin['bytes'], 'Original A40 sealed file drift: '+name)", "for name,pin in read(PREV/'ROOT_VERIFICATION.json')['files_sha256'].items():\n        need(sha(PREV/name)==pin, 'Original root-adopted A50 file drift: '+name)"),
        ("len(records)==len({r['id'] for r in records})==100 and len(links)==50", "len(records)==len({r['id'] for r in records})==120 and len(links)==60"),
        ("Exactly50 A +50 paired Full required", "Exactly60 A +60 paired Full required"),
        ("expected=" + S5, "expected=" + S6),
        ("Only the exact five A IID ten-shared-seed scenes are publishable", "Only five A IID scenes plus non-IID Benign with ten shared seeds are publishable"),
        ("assert len(records)==100 and len({r['id'] for r in records})==100", "assert len(records)==120 and len({r['id'] for r in records})==120"),
        ("assert len(bycell)==100", "assert len(bycell)==120"),
        ("scenes=" + S5.replace("[", "{").replace("]", "}"), "scenes=" + S6.replace("[", "{").replace("]", "}")),
        ("assert len(rows)==15", "assert len(rows)==18"),
        ("assert len(errors)==810", "assert len(errors)==972"),
        ("assert metric_checks==900 and count_checks==2400", "assert metric_checks==1080 and count_checks==2880"),
        ("Old80 records/order changed", "Old100 records/order changed"),
        ("Old80 serialized object bytes/order changed", "Old100 serialized object bytes/order changed"),
        ("if r['attack'] in ('Benign','F Flip','FedSA','S-DFA')", "if r['distribution']=='IID'"),
        ("Old A40 648 statistics changed", "Old A50 810 per-scene statistics changed"),
        ("old_cells==324 and checks['mean_sd_scalars']==810", "old_cells==405 and checks['mean_sd_scalars']==972"),
        ("# CelebA Full–minus_A: five complete IID scenes, three views", "# CelebA Full–minus_A: five IID scenes plus non-IID Benign, three views"),
        ("IID Benign, F Flip, FedSA, S-DFA and Sp-DFA only.", "IID Benign, F Flip, FedSA, S-DFA and Sp-DFA plus non-IID Benign only; the other four non-IID scenes are incomplete."),
        ("need(cells==405, 'Display cell count incomplete')", "need(cells==486, 'Display cell count incomplete')"),
        ("Actual replay devices across50 pairs", "Actual replay devices across60 pairs"),
        ("these50 actual source records", "these60 actual source records"),
        ("These are all five IID minus_A scenes; all five non-IID scenes remain outside this delivery (one non-IID Benign seed is retained in the accepted source index only).", "These are all five IID minus_A scenes plus the complete ten-seed non-IID Benign scene; the other four non-IID scenes remain outside this delivery. No mixed-distribution six-scene aggregate is reported."),
        ("table_record_count=100,preserved_records=100,paired_models=50,complete_scenes=5", "table_record_count=120,preserved_records=120,paired_models=60,complete_scenes=6"),
        ("old80_record_JSON_bytes_and_order_exact=True,old648_scalars_exact=True,old324_display_cells_exact=True", "old100_record_JSON_bytes_and_order_exact=True,old810_scalars_exact=True,old405_display_cells_exact=True"),
        ("ACTUAL_FIVE_A_IID_SCENE_TABLE_CANDIDATE_ROOT_REVIEW_PENDING", "ACTUAL_A60_FIVE_IID_PLUS_NONIID_BENIGN_CANDIDATE_ROOT_REVIEW_PENDING"),
        ("complete_scenes=5,paired_models=50,\n        displayed_records=100,preserved_records=100", "complete_scenes=6,paired_models=60,\n        displayed_records=120,preserved_records=120"),
        ("actual_A51_root_adoption=args.adoption,actual_A51_root_adoption_sha256=args.adoption_sha256", "actual_A60_root_adoption=args.adoption,actual_A60_root_adoption_sha256=args.adoption_sha256"),
        ("original80_provenance_kept=True", "original100_provenance_kept=True"),
        ("remaining_A_scenes=5,excluded_partial_id=\"minus_A_non-IID_Benign_seed91001\"", "remaining_A_scenes=4,excluded_partial_id=None"),
        ("ACTUAL_A50_FIVE_IID_SCENE_THREE_VIEW_TABLE_BUILT_ROOT_REVIEW_PENDING',records=100,pairs=50,stats=810,cells=405,metrics=900,counts=2400", "ACTUAL_A60_SIX_SCENE_THREE_VIEW_TABLE_BUILT_ROOT_REVIEW_PENDING',records=120,pairs=60,stats=972,cells=486,metrics=1080,counts=2880"),
    ])
    old_aggregate_put = "put('IID_SEED_FIRST.json',dict(status='ACTUAL_FIVE_IID_SCENE_SEED_FIRST_CANDIDATE',\n        scene_count_per_seed=5,distribution='IID',panels=aggregate,\n        definition='For each model seed, equally average its five IID scenes; then mean/sample SD across paired model seeds.',\n        not_all_ten_scenes=True,final_test=False))"
    source("finish.py", [
        ("need(len(records)==100 and all(r['distribution']=='IID' for r in records), 'Only exact five IID scenes')", "need(len(records)==120, 'Exactly six paired scenes required')\n    iid_records=[r for r in records if r['distribution']=='IID']\n    need(len(iid_records)==100 and len([r for r in records if r['distribution']=='non-IID' and r['attack']=='Benign'])==20, 'Only original five IID scenes plus non-IID Benign')"),
        ("aggregate = panels.aggregate_panels(records, evidence)", "aggregate = panels.aggregate_panels(iid_records, evidence)"),
        ("aggregate_check = numeric.verify_aggregate(records, aggregate)", "aggregate_check = numeric.verify_aggregate(iid_records, aggregate)\n    need(aggregate==read(base['PREV']/'IID_SEED_FIRST.json')['panels'], 'Original A50 IID aggregate statistics changed')"),
        (old_aggregate_put, "with (H/'IID_SEED_FIRST.json').open('xb') as f:\n        f.write((base['PREV']/'IID_SEED_FIRST.json').read_bytes())"),
        ("scene_display_cells=405", "scene_display_cells=486"),
        ("excluded_partial_id='minus_A_non-IID_Benign_seed91001'", "excluded_partial_id=None"),
        ("'CelebA, five IID scenarios. '", "'CelebA, five IID scenarios plus non-IID Benign. '"),
        ("attack='Five-IID mean' if is_aggregate else row['attack']", "attack='Five-IID mean' if is_aggregate else ('non-IID / '+row['attack'] if row['distribution']=='non-IID' else row['attack'])"),
        ("seeds=p['seeds'],attack=row['attack']", "seeds=p['seeds'],distribution=row['distribution'],attack=row['attack']"),
        ("per_scene_scalars=810,aggregate_scalars=162,\n        total_scalars=972,total_display_cells=486", "per_scene_scalars=checks['mean_sd_scalars'],aggregate_scalars=aggregate_check['mean_sd_scalars'],\n        total_scalars=checks['mean_sd_scalars']+aggregate_check['mean_sd_scalars'],total_display_cells=486+81"),
    ])
    source("verify_saved.py", [
        ("aggregate_numeric.verify_aggregate(records,aggregate)", "aggregate_numeric.verify_aggregate([r for r in records if r['distribution']=='IID'],aggregate)"),
        ("and len(previous)==80", "and len(previous)==100"),
        ("if row['attack']!='Sp-DFA'", "if row['distribution']=='IID'"),
        ("assert cells==486", "assert cells==567"),
        ("assert all(r['distribution']=='IID' for r in records)\n    assert bindings['limits']['excluded_partial_id']=='minus_A_non-IID_Benign_seed91001'", "assert len(records)==120 and len([r for r in records if r['distribution']=='IID'])==100\n    assert len([r for r in records if r['distribution']=='non-IID' and r['attack']=='Benign'])==20\n    assert bindings['limits']['excluded_partial_id'] is None\n    assert sha(H/'IID_SEED_FIRST.json')==sha(base['PREV']/'IID_SEED_FIRST.json')\n    assert aggregate==read(base['PREV']/'IID_SEED_FIRST.json')['panels']"),
        ("status='PASS',all_scalars=972,all_display_cells=486", "status='PASS',all_scalars=result['per_scene']['mean_sd_scalars']+result['seed_first']['mean_sd_scalars'],all_display_cells=cells"),
        ("old80_object_bytes_order_exact=True,old648_scalars_and324_cells_exact=True", "old100_object_bytes_order_exact=True,old810_scalars_and405_cells_exact=True,old_IID_seed_first_JSON_bytes_exact=True"),
    ])
    save("SOURCE_ADAPTATIONS.json", adaptations)
    for name in adaptations:
        code = '"""Scoped original A50 source; no scientific function copied or rewritten."""\nfrom source_adapter import namespace, preflight\n_RUN = (__name__ == "__main__")\nglobals().update(namespace(' + repr(name) + ', __file__))\nif _RUN:\n    preflight(' + repr(name) + ')\n    main()\n'
        with (H / name).open("x", encoding="utf8", newline="\n") as file:
            file.write(code)
    with (H / "AGGREGATE_SOURCE_PINS.json").open("xb") as file:
        file.write((A50 / "AGGREGATE_SOURCE_PINS.json").read_bytes())
    import source_adapter
    report = []
    for name, contract in adaptations.items():
        mapped = source_adapter.mapped(name)
        report.append({"source": contract["source"], "source_sha256": contract["sha256"],
                       "replacements_each_unique": len(contract["replacements"]), "inverse_original_source_bytes_exact": True,
                       "mapped_source_sha256": hashlib.sha256(mapped.encode()).hexdigest(), "mapped_AST_valid": True})
    save("SOURCE_PREPARATION.json", {"status": "SOURCE_ONLY_READY_AWAITING_ACTUAL_ROOT260_BINDING", "sources": report,
         "A50_source_seal_sha256": sha(source_seal), "expected_exact_new_ids": [f"minus_A_non-IID_Benign_seed{s}" for s in range(91002,91011)],
         "scope": "Original five IID scenes plus complete non-IID Benign; original IID seed-first JSON preserved byte-for-byte; no mixed-distribution aggregate.",
         "ROOT_BINDING_present": False, "table_generation_executed": False, "new_CNN_fit_training_SSH_bulk_writes": 0})
    print(json.dumps({"status": "SOURCE_ONLY_READY", "sources": report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
