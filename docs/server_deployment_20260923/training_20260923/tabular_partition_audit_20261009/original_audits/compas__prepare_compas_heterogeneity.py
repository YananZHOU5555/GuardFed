#!/usr/bin/env python3
"""Prepare 180 COMPAS E2 jobs; execute only two 2-round checks."""
import contextlib, hashlib, io, json, os, subprocess, sys, time
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np
import torch
import reproduce_paper_tables as core
import run_revision_ablation as runner
OUT=ROOT/"results/revision_20260923/compas_heterogeneity_v1"
CHECK=ROOT/"results/revision_20260923/compas_heterogeneity_pipeline_checks"
SOURCE=Path("/workspace/GuardFed/results/attack_strength/raw_results.jsonl")
ALPHAS=(1.0,0.5,0.1)
METHODS=("GuardFed-AD2+","FedAvg","FLTrust")
ATTACKS=("Benign","S-DFA")
def comparable(cfg):
 return {k:v for k,v in cfg.items() if k not in ("experiment_suite","experiment_tag")}
def verify(manifest):
 assert len(manifest["jobs"])==len(set(manifest["jobs"]))==180
 for name,expected in manifest["source_hashes"].items():
  assert runner.digest(ROOT/name)==expected,("changed source",name)
 identities=set();outputs=set();groups={}
 for path in manifest["jobs"]:
  job=json.loads(Path(path).read_text());cfg=job["config"]
  assert cfg==asdict(core.ExperimentConfig(**cfg))
  assert cfg["rounds"]==70 and cfg["device"]=="cpu"
  assert cfg["compas_preprocessing_version"]=="train_only" and cfg["ablation_component"]=="none"
  assert cfg["root_label_noise"]==cfg["root_sensitive_noise"]==0
  assert job["dataset"]=="compas" and job["distribution"]=="non-IID"
  assert cfg["client_alpha"] in ALPHAS and cfg["seed"] in runner.SEEDS
  assert job["method"] in METHODS and job["attack"] in ATTACKS
  assert job["source_hashes"]==manifest["source_hashes"]
  identity=(cfg["client_alpha"],job["method"],job["attack"],cfg["seed"])
  assert identity not in identities and job["output"] not in outputs
  identities.add(identity);outputs.add(job["output"])
  groups.setdefault(identity[:-1],[]).append(cfg["seed"])
 assert len(groups)==18 and all(sorted(s)==sorted(runner.SEEDS) for s in groups.values())
 audit=json.loads((OUT/"client_distribution_audit.json").read_text())
 assert len(audit)==len({(r["alpha"],r["seed"]) for r in audit})==30
 assert all(len(r["clients"])==20 for r in audit)
 return {"unique_jobs":180,"condition_groups":18,"seeds_per_condition":10,"partition_audits":30,"source_hashes":"passed"}
def main():
 torch.set_num_threads(1)
 assert not (OUT/"manifest.json").exists(),"Refuse to overwrite frozen manifest"
 historical={}
 for line_no,line in enumerate(SOURCE.open(),1):
  r=json.loads(line)
  if (r.get("dataset")=="compas" and r.get("distribution")=="non-IID" and r.get("method")=="GuardFed-AD2+" and r.get("attack")=="S-DFA" and r.get("rounds")==70 and r.get("num_malicious")==4 and r.get("seed") in runner.SEEDS):
   seed=r["seed"];assert seed not in historical
   expected=asdict(core.ExperimentConfig(seed=seed,rounds=70,device="cuda",server_ratio=.1,synthetic_method="none",fflip_mode="all_unprivileged",sdfa_foe_mode="fedsa",spdfa_foe_mode="fedsa",fedsa_gain=4.5,fedsa_norm_ratio=3))
   assert comparable(asdict(core.ExperimentConfig(**r["config"])))==comparable(expected),("config mismatch",seed)
   assert r["alpha"]==5 and [x["round"] for x in r["trajectory_metrics"]]==list(range(1,71))
   historical[seed]=(r,line_no,hashlib.sha256(line.encode()).hexdigest())
 assert len(historical)==10
 frozen=runner.source_hashes()
 for path in (ROOT/"data/compas/compas-scores-two-years.csv",Path(__file__).resolve()):
  frozen[str(path.relative_to(ROOT))]=runner.digest(path)
 jobs=[];partitions=[]
 for seed in runner.SEEDS:
  old,line_no,line_hash=historical[seed]
  for alpha in ALPHAS:
   cfg=asdict(core.ExperimentConfig(**dict(old["config"],device="cpu",rounds=70,ablation_component="none",client_alpha=alpha,full_round_diagnostics=True,root_label_noise=0.,root_sensitive_noise=0.,compas_preprocessing_version="train_only")))
   with contextlib.redirect_stdout(io.StringIO()):
    bundle=core.load_bundle("compas",alpha,core.ExperimentConfig(**cfg),torch.device("cpu"))
   clients=[]
   for cid,client in bundle["clients"].items():
    y=client["y"].cpu().numpy();s=client["sensitive"]
    clients.append({"client_id":cid,"samples":len(y),"is_malicious_under_attack":cid<4,
     "group_label_counts":{f"{g}|{label}":int(np.sum((s==g)&(y==label))) for g in (0,1) for label in (0,1)},
     "sensitive_one_share":float(np.mean(s==1)) if len(y) else None,"positive_label_share":float(np.mean(y==1)) if len(y) else None})
   assert sum(c["samples"] for c in clients)==bundle["train_rows"]-bundle["root_clean_rows"]
   partitions.append({"alpha":alpha,"seed":seed,"partition_basis":"sensitive_group_Dirichlet","preprocessing":"train_only",
    "train_rows":bundle["train_rows"],"test_rows":bundle["test_rows"],"root_rows":bundle["root_clean_rows"],
    "root_noise_audit":bundle["root_noise_audit"],"empty_client_ids":[c["client_id"] for c in clients if not c["samples"]],
    "clients":clients,"resampled":False})
   for method in METHODS:
    for attack in ATTACKS:
     identity=f"alpha{alpha:g}_{method}_{attack}_seed{seed}"
     config=asdict(core.ExperimentConfig(**dict(cfg,experiment_suite="revision_compas_heterogeneity_v1",experiment_tag=identity)))
     job={"id":identity,"dataset":"compas","distribution":"non-IID","method":method,"attack":attack,"config":config,
      "output":str(OUT/"runs"/identity),"source_hashes":frozen,"evidence_stage":"formal_supplement",
      "historical_config_source":{"path":str(SOURCE),"line":line_no,"line_sha256":line_hash},
      "note":"New train_only preprocessing and client alpha. Historical metric tables are neither overwritten nor pooled."}
     path=OUT/"jobs"/f"{identity}.json";runner.write_json(path,job);jobs.append(str(path))
 runner.write_json(OUT/"client_distribution_audit.json",partitions)
 manifest={"protocol":"revision_compas_heterogeneity_v1","jobs":jobs,"output":str(OUT),"source_hashes":frozen,
  "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
  "new_run_count":180,"reused_full_count":0,"seeds":runner.SEEDS,"old_experiments_rerun":False,"formal_started_by_preparer":False,
  "statistics":"Round70 common checkpoint; all seeds retained; mean/sample std; paired seeds and partitions.",
  "preprocessing":"COMPAS train_only; old main tables unchanged, historical numerical metrics not reused.",
  "config_changes":{"client_alpha":list(ALPHAS),"device":"cpu","compas_preprocessing_version":"train_only",
   "provenance_only":["experiment_suite","experiment_tag"],"logging_only":["full_round_diagnostics"]},
  "partition_audit":str(OUT/"client_distribution_audit.json"),"created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
 runner.write_json(OUT/"manifest.json",manifest)
 verification=verify(manifest);pilots=[]
 for path in jobs:
  original=json.loads(Path(path).read_text())
  if original["config"]["seed"]!=123 or original["config"]["client_alpha"]!=.1 or original["method"]!="GuardFed-AD2+":continue
  pilot=dict(original,id="pipeline_"+original["id"],evidence_stage="pipeline_check_not_formal")
  pilot["config"]=dict(original["config"],rounds=2,experiment_tag=pilot["id"])
  pilot["output"]=str(CHECK/"runs"/pilot["id"])
  target=CHECK/"jobs"/(pilot["id"]+".json");runner.write_json(target,pilot);pilots.append(str(target))
 assert len(pilots)==2
 runner.write_json(CHECK/"manifest.json",dict(manifest,jobs=pilots,output=str(CHECK),new_run_count=2,seeds=[123],evidence_stage="pipeline_check_not_formal"))
 statuses=[]
 for path in pilots:
  job=json.loads(Path(path).read_text());out=Path(job["output"]);out.mkdir(parents=True,exist_ok=True)
  env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",GUARDFED_CPU_THREADS="1",PYTHONUNBUFFERED="1")
  with (out/"worker.log").open("a") as log:
   proc=subprocess.run([sys.executable,str(ROOT/"scripts/run_revision_ablation.py"),"worker","--job",path],env=env,stdout=log,stderr=subprocess.STDOUT)
  item={"job":job["id"],"exitcode":proc.returncode}
  if proc.returncode:runner.write_json(out/"failure.json",dict(item,time=time.time()))
  else:
   result=runner.checked_result(job)
   assert result["alpha"]==.1 and len(result["round_summaries"])==2
   assert result["data_contract"]["preprocessing_version"]=="train_only"
   assert all(x["root_noise_audit"]["label"]["flipped_count"]==0 and x["root_noise_audit"]["sensitive"]["flipped_count"]==0 for x in result["round_summaries"])
   item.update(metrics=result["metrics"],duration_sec=result["duration_sec"],status="complete")
  statuses.append(item)
 status=runner.summarize(manifest)
 assert status["new_complete"]==status["failed"]==0
 verification.update(formal_started=False,pilots=statuses,status="passed" if all(x["exitcode"]==0 for x in statuses) else "pilot_failure_retained")
 runner.write_json(OUT/"verification.json",verification)
 print(json.dumps({"manifest":str(OUT/"manifest.json"),**verification}),flush=True)
 if not all(x["exitcode"]==0 for x in statuses):raise SystemExit(1)
if __name__=="__main__":
 if "--verify-only" in sys.argv:print(json.dumps(verify(json.loads((OUT/"manifest.json").read_text()))))
 else:main()
