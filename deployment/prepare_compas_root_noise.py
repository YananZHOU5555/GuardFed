#!/usr/bin/env python3
"""Prepare nonduplicate COMPAS root sensitivity jobs; canaries only here."""
import contextlib, io, json, os, subprocess, sys, time
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np
import torch
import reproduce_paper_tables as core
import run_revision_ablation as runner
OUT=ROOT/"results/revision_20260923/compas_root_noise_v1"
CHECK=OUT/"pipeline_checks"
def semantic(cfg):
 return {k:v for k,v in cfg.items() if k not in ("experiment_suite","experiment_tag","device")}
def main():
 torch.set_num_threads(1)
 assert not (OUT/"manifest.json").exists(),"Refuse to overwrite frozen manifest"
 templates={}
 for seed in runner.SEEDS:
  path=ROOT/"results/revision_20260923/compas_ablation_v1/jobs"/f"non-IID_Full_seed{seed}.json"
  job=json.loads(path.read_text());result=runner.checked_result(job);assert result is not None
  assert result["alpha"]==5 and result["data_contract"]["preprocessing_version"]=="train_only"
  cfg=asdict(core.ExperimentConfig(**job["config"]))
  assert cfg["rounds"]==70 and cfg["compas_preprocessing_version"]=="train_only"
  assert cfg["ablation_component"]=="none" and cfg["root_label_noise"]==cfg["root_sensitive_noise"]==0
  templates[seed]=cfg
 candidates={}
 scan=[]
 for manifest_path in sorted((ROOT/"results/revision_20260923").glob("*/manifest.json")):
  if manifest_path.parent==OUT:continue
  manifest=json.loads(manifest_path.read_text())
  for job_path in manifest.get("jobs",[]):
   job=json.loads(Path(job_path).read_text());cfg=job["config"]
   alpha=cfg.get("client_alpha") if cfg.get("client_alpha") is not None else core.DISTRIBUTIONS.get(job["distribution"])
   if not (job["dataset"]=="compas" and job["method"]=="GuardFed-AD2+" and job["attack"] in ("Benign","S-DFA")
           and cfg["rounds"]==70 and cfg["seed"] in runner.SEEDS and alpha==5
           and cfg.get("compas_preprocessing_version")=="train_only"
           and cfg.get("ablation_component","none")=="none" and not cfg.get("root_label_noise",0) and not cfg.get("root_sensitive_noise",0)):continue
   if semantic(asdict(core.ExperimentConfig(**cfg)))!=semantic(templates[cfg["seed"]]):continue
   result=runner.checked_result(job)
   if result is None:continue
   key=(job["attack"],cfg["seed"])
   candidates.setdefault(key,[]).append((job_path,job,result))
   scan.append({"job":job_path,"attack":key[0],"seed":key[1],"device":cfg["device"],"checkpoint_verified":True})
 frozen=runner.source_hashes()
 for path in (ROOT/"data/compas/compas-scores-two-years.csv",Path(__file__).resolve()):
  frozen[str(path.relative_to(ROOT))]=runner.digest(path)
 reused={}
 for key,matches in candidates.items():
  matches.sort(key=lambda item:item[1]["config"]["device"]!="cpu")
  source_path,job,result=matches[0];reused[key]=result
  runner.write_json(OUT/"reused_full"/f"{key[0]}_{key[1]}.json",{
   "status":"reused_completed_train_only_control","source_job":source_path,
   "source_result":str(Path(job["output"])/"result.json"),
   "source_result_sha256":runner.digest(Path(job["output"])/"result.json"),
   "result":result,"equivalence":"All model, partition, attack, preprocessing, root and evaluation configuration fields equal; suite/tag are provenance-only.",
   "device_limitation":"Prior CUDA control vs new CPU noise where applicable; short canaries do not prove 70-round device equivalence."})
 jobs=[];missing=[]
 for seed in runner.SEEDS:
  for attack in ("Benign","S-DFA"):
   conditions=[(field,rate) for field in ("root_label_noise","root_sensitive_noise") for rate in (.1,.2,.4)]
   if (attack,seed) not in reused:conditions.insert(0,("clean",0.));missing.append({"attack":attack,"seed":seed})
   for field,rate in conditions:
    identity=f"{attack}_{field}_{int(rate*100)}pct_seed{seed}"
    cfg=dict(templates[seed],device="cpu",experiment_suite="revision_compas_root_noise_v1",experiment_tag=identity)
    if field!="clean":cfg[field]=rate
    config=asdict(core.ExperimentConfig(**cfg))
    job={"id":identity,"dataset":"compas","distribution":"non-IID","method":"GuardFed-AD2+","attack":attack,
     "config":config,"output":str(OUT/"runs"/identity),"source_hashes":frozen,"evidence_stage":"formal_supplement",
     "condition":"new_clean_control" if field=="clean" else "root_noise",
     "interpretation":"Root end-to-end sensitivity: FedSA-inspired attack also uses the root update; no defense-only attribution."}
    path=OUT/"jobs"/f"{identity}.json";runner.write_json(path,job);jobs.append(str(path))
 assert len(jobs)==120+len(missing) and len(reused)+len(missing)==20
 manifest={"protocol":"revision_compas_root_noise_v1","jobs":jobs,"output":str(OUT),"source_hashes":frozen,
  "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
  "seeds":runner.SEEDS,"new_run_count":len(jobs),"noise_run_count":120,"new_clean_count":len(missing),
  "reused_full_count":len(reused),"old_experiments_rerun":False,
  "statistics":"Fixed round70 all metrics; paired seeds; mean/sample std; no test/seed selection.",
  "preprocessing":"train_only","partition_alpha":5,
  "config_differences":{"intervention":["root_label_noise OR root_sensitive_noise"],"provenance_only":["experiment_suite","experiment_tag"],"environment":"CPU noise; prior CUDA clean reused with disclosed numerical limitation"},
  "interpretation":"End-to-end root sensitivity including FedSA attack direction changes.",
  "created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
 runner.write_json(OUT/"manifest.json",manifest)
 runner.write_json(OUT/"control_reuse_audit.json",{"completed_matching_jobs":scan,"reused":len(reused),"new_missing_controls":missing})
 identities=set()
 for path in jobs:
  job=json.loads(Path(path).read_text());c=job["config"]
  identity=(job["attack"],c["root_label_noise"],c["root_sensitive_noise"],c["seed"])
  assert identity not in identities;identities.add(identity)
  assert c==asdict(core.ExperimentConfig(**c)) and c["device"]=="cpu" and c["rounds"]==70
  if identity[1:3]==(0.,0.):assert (job["attack"],c["seed"]) not in reused
 for name,expected in frozen.items():assert runner.digest(ROOT/name)==expected,name
 # Verify root-only interventions on actual seed123 data without training.
 clean_cfg=core.ExperimentConfig(**dict(templates[123],device="cpu"))
 with contextlib.redirect_stdout(io.StringIO()):
  clean=core.load_bundle("compas",5.,clean_cfg,torch.device("cpu"))
 pilots=[]
 for attack,field in (("Benign","root_label_noise"),("S-DFA","root_sensitive_noise")):
  original=next(json.loads(Path(path).read_text()) for path in jobs if
   json.loads(Path(path).read_text())["attack"]==attack and
   json.loads(Path(path).read_text())["config"]["seed"]==123 and
   json.loads(Path(path).read_text())["config"][field]==.2)
  cfg=core.ExperimentConfig(**original["config"])
  with contextlib.redirect_stdout(io.StringIO()):
   noisy=core.load_bundle("compas",5.,cfg,torch.device("cpu"))
  assert clean["rw_weights"]==noisy["rw_weights"]
  assert torch.equal(clean["X_test"],noisy["X_test"]) and torch.equal(clean["y_test"],noisy["y_test"])
  np.testing.assert_array_equal(clean["test_sensitive"],noisy["test_sensitive"])
  for cid in clean["clients"]:
   for name in ("X","y"):assert torch.equal(clean["clients"][cid][name],noisy["clients"][cid][name])
   np.testing.assert_array_equal(clean["clients"][cid]["sensitive"],noisy["clients"][cid]["sensitive"])
  pilot=dict(original,id="pipeline_"+original["id"],evidence_stage="pipeline_check_not_formal")
  pilot["config"]=dict(original["config"],rounds=2,experiment_tag=pilot["id"])
  pilot["output"]=str(CHECK/"runs"/pilot["id"])
  path=CHECK/"jobs"/(pilot["id"]+".json");runner.write_json(path,pilot);pilots.append(str(path))
 runner.write_json(CHECK/"manifest.json",dict(manifest,jobs=pilots,output=str(CHECK),new_run_count=2,reused_full_count=0,evidence_stage="pipeline_check_not_formal"))
 statuses=[]
 for path in pilots:
  job=json.loads(Path(path).read_text());out=Path(job["output"]);out.mkdir(parents=True,exist_ok=True)
  env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",GUARDFED_CPU_THREADS="1",PYTHONUNBUFFERED="1")
  with (out/"worker.log").open("a") as log:
   proc=subprocess.run([sys.executable,str(ROOT/"scripts/run_revision_ablation.py"),"worker","--job",path],env=env,stdout=log,stderr=subprocess.STDOUT)
  status={"job":job["id"],"exitcode":proc.returncode}
  if proc.returncode:runner.write_json(out/"failure.json",dict(status,time=time.time()))
  else:
   result=runner.checked_result(job)
   assert result["alpha"]==5 and result["data_contract"]["preprocessing_version"]=="train_only"
   audit=result["data_contract"]["root_noise_audit"]
   assert audit["label"]["flipped_count"]+audit["sensitive"]["flipped_count"]==int(.2*audit["root_rows"])
   status.update(duration_sec=result["duration_sec"],metrics=result["metrics"],noise_audit=audit)
  statuses.append(status)
 runner.summarize(manifest)
 passed=all(x["exitcode"]==0 for x in statuses)
 runner.write_json(OUT/"verification.json",{"passed":passed,"new_jobs":len(jobs),"noise_jobs":120,
  "unique_conditions":len(identities),"reused_controls":len(reused),"new_clean_controls":len(missing),
  "root_only_check":"client tensors/test tensors/global reweighting exact equal","pilots":statuses,"formal_started":False})
 print(json.dumps({"manifest":str(OUT/"manifest.json"),"new_jobs":len(jobs),"reused_controls":len(reused),"new_clean_controls":len(missing),"canaries_passed":passed}),flush=True)
 if not passed:raise SystemExit(1)
if __name__=="__main__":main()
