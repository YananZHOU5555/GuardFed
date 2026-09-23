#!/usr/bin/env python3
"""Prepare isolated COMPAS train-only preprocessing ablations. No formal training."""
import hashlib,json,sys,subprocess
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import reproduce_paper_tables as core
import run_revision_ablation as runner
OUT=ROOT/"results/revision_20260923/compas_ablation_v1"
CHECK=ROOT/"results/revision_20260923/compas_pipeline_checks"
SEEDS=runner.SEEDS
def write(path,value):runner.write_json(path,value)
def main():
 commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 assert commit.startswith("b1a808b"),("unexpected frozen commit",commit)
 source=Path("/workspace/GuardFed/results/attack_strength/raw_results.jsonl")
 controls={}
 for line_no,line in enumerate(source.open(),1):
  r=json.loads(line)
  if (r.get("study")=="ratio" and r.get("dataset")=="compas" and r.get("method")=="GuardFed-AD2+" and r.get("attack")=="S-DFA" and r.get("rounds")==70 and r.get("num_malicious")==4 and r.get("seed") in SEEDS):
   key=(r["distribution"],r["seed"]);assert key not in controls
   controls[key]=(r,line_no,hashlib.sha256(line.encode()).hexdigest())
 assert len(controls)==20
 frozen=runner.source_hashes()
 compas=ROOT/"data/compas/compas-scores-two-years.csv"
 frozen[str(compas.relative_to(ROOT))]=runner.digest(compas)
 frozen[str(Path(__file__).resolve().relative_to(ROOT))]=runner.digest(__file__)
 jobs=[];preparations=[]
 for seed in SEEDS:
  for dist in ["IID","non-IID"]:
   old,line_no,line_hash=controls[(dist,seed)]
   for component in ["none","U","C","A","F","V","N"]:
    profile="Full" if component=="none" else "minus"+component
    identity=f"{dist}_{profile}_seed{seed}"
    cfg=asdict(core.ExperimentConfig(**dict(old["config"],device="cuda",rounds=70,
      ablation_component=component,client_alpha=None,full_round_diagnostics=True,
      root_label_noise=0.0,root_sensitive_noise=0.0,compas_preprocessing_version="train_only",
      experiment_suite="revision_compas_ablation_v1",experiment_tag=identity)))
    job={"id":identity,"dataset":"compas","distribution":dist,"method":"GuardFed-AD2+","attack":"S-DFA",
      "config":cfg,"output":str(OUT/"runs"/identity),"source_hashes":frozen,"evidence_stage":"formal_supplement",
      "historical_config_source":{"source":str(source),"line":line_no,"line_sha256":line_hash},
      "control_note":"Full is a necessary new train_only preprocessing control. No historical metrics are reused."}
    path=OUT/"jobs"/(identity+".json")
    write(path,job);jobs.append(str(path))
    preparations.append(job)
 assert len(jobs)==140 and len({j["id"] for j in preparations})==140
 assert len({(j["distribution"],j["config"]["seed"],j["config"]["ablation_component"]) for j in preparations})==140
 manifest={"protocol":"revision_compas_ablation_v1","jobs":jobs,"output":str(OUT),"source_hashes":frozen,
  "source_commit":commit,"reused_full_count":0,"new_run_count":140,"seeds":SEEDS,"formal_started_by_preparer":False,
  "statistics":"final round common checkpoint; mean/sample std; no test or seed selection",
  "preprocessing":"train_only; new Full controls, legacy COMPAS results unchanged and not pooled",
  "old_experiments_rerun":False}
 write(OUT/"manifest.json",manifest)
 pilots=[]
 for dist,component,gpu in [("IID","none",0),("non-IID","N",1)]:
  original=next(j for j in preparations if j["distribution"]==dist and j["config"]["seed"]==123 and j["config"]["ablation_component"]==component)
  j=dict(original);j["id"]="pipeline_"+original["id"];j["config"]=dict(original["config"],rounds=2,experiment_tag=j["id"])
  j["output"]=str(CHECK/"runs"/j["id"]);j["evidence_stage"]="pipeline_check_not_formal";j["visible_gpu"]=gpu
  path=CHECK/"jobs"/(j["id"]+".json");write(path,j);pilots.append(str(path))
 write(CHECK/"manifest.json",dict(manifest,jobs=pilots,output=str(CHECK),new_run_count=2,seeds=[123],evidence_stage="pipeline_check_not_formal"))
 provenance={"frozen_commit":commit,"source_hashes":frozen,"historical_raw_sha256":runner.digest(source),"new_jobs":140,
   "pipeline_jobs":2,"pipeline_scope":"IID Full and non-IID minus N, each seed123/2 rounds; pipeline checks only, not a comparative effect estimate",
   "formal_launched":False}
 write(OUT/"preparation.json",provenance)
 print(json.dumps(provenance),flush=True)
if __name__=="__main__":main()
