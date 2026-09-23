"""Freeze the planned fixed-reservoir protected-share experiment."""
import json
import sys
import contextlib
import io
from dataclasses import asdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "scripts"))
import reproduce_paper_tables as core
import run_revision_ablation as runner

out=ROOT/"results/revision_20260923/rootshare_v1"
if (out/"manifest.json").exists(): raise RuntimeError("Already frozen")
frozen=runner.source_hashes()
frozen["deployment/prepare_rootshare.py"]=runner.digest(__file__)
jobs=[];canaries=[];audits=[]
for dataset in ["adult","compas"]:
    source=Path("/workspace/GuardFed-next/results/revision_20260923/compas_ablation_v1/jobs/non-IID_Full_seed123.json")
    base=json.loads(source.read_text())["config"]
    base.update(compas_preprocessing_version="legacy" if dataset=="adult" else "train_only",device="cpu",server_sampling="fixed_reservoir_protected_share_v1",client_alpha=5.0,
                full_round_diagnostics=True,root_label_noise=0.,root_sensitive_noise=0.,
                root_reserve_ratio=.2,ablation_component="none",experiment_suite="revision_rootshare_v1")
    for seed in runner.SEEDS:
        base["seed"]=seed
        with contextlib.redirect_stdout(io.StringIO()):
            loader=core.DatasetLoader(dataset_name=dataset,seed=seed,device="cpu",**({"preprocessing_version":"train_only"} if dataset=="compas" else {}))
        label="income" if dataset=="adult" else "two_year_recid"
        for share in [.5,.1,.02,0.]:
            cfg=core.ExperimentConfig(**dict(base,root_protected_share=share))
            root,clients,audit=core.sample_reservoir_root(loader.train_df,label,loader.sensitive_column,cfg,dataset)
            audits.append(dict(dataset=dataset,seed=seed,**audit))
            for attack in ["Benign","S-DFA"]:
                ident=f"{dataset}_{attack}_share{share}_seed{seed}"
                config=asdict(cfg);config["experiment_tag"]=ident
                job=dict(id=ident,dataset=dataset,distribution="non-IID",method="GuardFed-AD2+",attack=attack,config=config,
                         output=str(out/"runs"/ident),source_hashes=frozen,evidence_stage="formal_supplement")
                path=out/"jobs"/(ident+".json");runner.write_json(path,job);jobs.append(str(path))
                if seed==123 and share in [.5,0.]:
                    pilot=dict(job,config=dict(config,rounds=2,experiment_suite="rootshare_canary_v1"),
                               output=str(out/"canaries/runs"/ident),evidence_stage="pipeline_canary")
                    path=out/"canaries/jobs"/(ident+".json");runner.write_json(path,pilot);canaries.append(str(path))
for ds in ["adult","compas"]:
    for seed in runner.SEEDS:
        group=[a for a in audits if a["dataset"]==ds and a["seed"]==seed]
        assert len(group)==4
        for field in ["server_rows","reserve_index_sha256","client_index_sha256","client_rows"]:
            assert len({a[field] for a in group})==1,(ds,seed,field)
manifest=dict(protocol="rootshare_v1",output=str(out),jobs=jobs,new_run_count=160,reused_full_count=0,
    seeds=runner.SEEDS,source_hashes=frozen,statistics="Fixed round70, all10 unique seeds, mean and sample std; no selection.",
    research_question="End-to-end AD2+ sensitivity to missing protected groups in root, with invariant clients and root size.",
    reserve_design="20% train joint-stratified reservoir. Remaining80% clients fixed. Root min(10% train, nonprotected reserve,2*protected reserve). Unused reservoir never returns to clients.",
    protected_groups={"adult":"Female=0","compas":"African-American=1"},
    limitations=["Zero-share fairness is unidentifiable; preserve and label original fallback behavior.",
                 "S-DFA attacker reference update also changes with root; not defense-only causal effect.",
                 "COMPAS train_only preprocessing; no historical controls reused for this reservoir protocol."])
runner.write_json(out/"manifest.json",manifest)
runner.write_json(out/"partition_audit.json",audits)
runner.write_json(out/"canaries/manifest.json",dict(manifest,output=str(out/"canaries"),jobs=canaries,new_run_count=len(canaries),evidence_stage="pipeline_canary"))
print(json.dumps({"manifest":str(out/"manifest.json"),"jobs":len(jobs),"canaries":len(canaries),"audited_partitions":len(audits)}))
