#!/usr/bin/env python3
import argparse, contextlib, csv, hashlib, importlib.util, io, json, math, os, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import statistics
os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
OUT=Path(__file__).resolve().parent
SOURCES={"adult":Path("/workspace/GuardFed-revision"),"compas":Path("/workspace/GuardFed-next")}
MODULE=None
ROOT_LOADED=None
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
 return h.hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"))
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+".tmp")
 temp.write_text(json.dumps(value,indent=2));temp.replace(path)
def init_worker(root):
 global MODULE,ROOT_LOADED
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 ROOT_LOADED=Path(root);sys.path.insert(0,str(ROOT_LOADED))
 path=ROOT_LOADED/"scripts/reproduce_paper_tables.py"
 spec=importlib.util.spec_from_file_location("frozen_analysis_core",path)
 MODULE=importlib.util.module_from_spec(spec);sys.modules[spec.name]=MODULE;spec.loader.exec_module(MODULE)
 assert not torch.cuda.is_available(),"Offline analysis must not see CUDA"
def group_stats(y,pred,sensitive):
 import numpy as np
 result={}
 for group in (0,1):
  mask=sensitive==group;pos=mask&(y==1);neg=mask&(y==0)
  result[str(group)]={"n":int(mask.sum()),"positive_support":int(pos.sum()),"negative_support":int(neg.sum()),
   "predicted_positive":int(np.sum(pred[mask]==1)),
   "true_positive":int(np.sum(pred[pos]==1)),"false_positive":int(np.sum(pred[neg]==1)),
   "TPR":float(np.mean(pred[pos]==1)) if pos.any() else None,
   "FPR":float(np.mean(pred[neg]==1)) if neg.any() else None}
 return result
def evaluate(job_path):
 import torch,numpy as np
 job_path=Path(job_path);job=json.loads(job_path.read_text());dataset=job["dataset"]
 assert ROOT_LOADED==SOURCES[dataset]
 destination=OUT/"cases"/f"{dataset}__{job['id']}.json"
 original_path=Path(job["output"])/"result.json";checkpoint=Path(job["output"])/"model.pt"
 original=json.loads(original_path.read_text());reported=original["metrics"]
 checkpoint_sha=sha(checkpoint)
 assert checkpoint_sha==original["revision_job"]["checkpoint_sha256"]
 assert original["config"]==job["config"]==original["revision_job"]["config"]
 assert original["rounds"]==70 and original["trajectory_metrics"][-1]["metrics"]==reported
 for name,digest in job["source_hashes"].items():assert sha(ROOT_LOADED/name)==digest,(job["id"],name)
 if destination.exists():
  existing=json.loads(destination.read_text())
  assert existing["checkpoint_sha256"]==checkpoint_sha and existing["source_result_sha256"]==sha(original_path)
  return existing
 start=time.time();cfg=MODULE.ExperimentConfig(**dict(job["config"],device="cpu"))
 with contextlib.redirect_stdout(io.StringIO()):
  bundle=MODULE.load_bundle(dataset,original["alpha"],cfg,torch.device("cpu"))
 dc=original["data_contract"]
 for field in ("num_features","train_rows","test_rows","root_clean_rows","root_synthetic_rows","server_sampling_audit"):
  assert bundle[field]==dc[field],(job["id"],"reloaded data contract",field)
 model=MODULE.SimpleMLP(bundle["num_features"],cfg.seed)
 model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True));model.eval()
 thresholds,calibration_info=MODULE.fit_group_thresholds(model,bundle,cfg)
 margins=MODULE.model_margins(model,bundle["X_test"],cfg.batch_size)
 y=bundle["y_test"].numpy();sensitive=np.asarray(bundle["test_sensitive"])
 calibrated=MODULE.metrics_from_group_thresholds(y,margins,sensitive,thresholds)
 core_metrics={k:calibrated[k] for k in MODULE.METRICS}
 payload={"dataset":dataset,"distribution":job["distribution"],"alpha":original["alpha"],"ablation":cfg.ablation_component,
  "seed":cfg.seed,"job_id":job["id"],"source_config":original["config"],"source_job":str(job_path),
  "source_hashes":job["source_hashes"],"source_result":str(original_path),"source_result_sha256":sha(original_path),
  "source_config_sha256":hashlib.sha256(canonical(original["config"]).encode()).hexdigest(),
  "checkpoint":str(checkpoint),"checkpoint_sha256":checkpoint_sha,"checkpoint_round":70,
  "evaluation_device":"cpu","gpu_visible":False,"new_training":False,"new_seed":False,
  "torch_version":torch.__version__,"reported_calibrated":reported,
  "cpu_recalibrated":core_metrics,"cpu_thresholds":thresholds,"calibration_info":calibration_info}
 if core_metrics!=reported:
  payload.update(status="cpu_recalibration_mismatch",formal_stat_eligible=False,
   limitation="CPU recalibration differs from original reported metrics. Original metrics remain unchanged; this case is not relabeled or used in paired formal summaries.",
   mismatch={k:{"original":reported[k],"cpu_recalibrated":core_metrics[k]} for k in MODULE.METRICS if reported[k]!=core_metrics[k]})
  write(destination,payload);return payload
 preds=[]
 with torch.no_grad():
  for start_idx in range(0,len(bundle["X_test"]),cfg.batch_size):
   preds.append(model(bundle["X_test"][start_idx:start_idx+cfg.batch_size]).argmax(dim=1).numpy())
 raw_pred=np.concatenate(preds)
 cal_pred=np.zeros_like(y)
 for group in (0,1):
  mask=sensitive==group;cal_pred[mask]=(margins[mask]>=thresholds[group]).astype(int)
 raw=MODULE.compute_metrics(y,raw_pred,sensitive)
 assert {k:MODULE.compute_metrics(y,cal_pred,sensitive)[k] for k in MODULE.METRICS}==core_metrics
 payload.update(status="accepted_exact_recalibration",formal_stat_eligible=True,
  raw={k:raw[k] for k in MODULE.METRICS},calibrated=core_metrics,
  raw_groups=group_stats(y,raw_pred,sensitive),calibrated_groups=group_stats(y,cal_pred,sensitive),
  paired_delta_calibrated_minus_raw={k:core_metrics[k]-raw[k] for k in MODULE.METRICS},
  duration_sec=time.time()-start)
 write(destination,payload)
 return payload
def csv_write(name,rows):
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with (OUT/name).open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def summarize(manifest):
 grouped={};perseed=[];statuses=[];metric_keys=["accuracy","aeod","aspd","group0_TPR","group0_FPR","group1_TPR","group1_FPR"]
 for dataset,paths in manifest["jobs"].items():
  for path in paths:
   job=json.loads(Path(path).read_text());cfg=job["config"];condition=(dataset,job["distribution"],cfg["ablation_component"])
   grouped.setdefault(condition,[]);dest=OUT/"cases"/f"{dataset}__{job['id']}.json"
   if not dest.exists():
    statuses.append({"dataset":dataset,"job_id":job["id"],"status":"not_evaluated"});continue
   result=json.loads(dest.read_text());statuses.append({"dataset":dataset,"job_id":job["id"],"status":result["status"]})
   if result["status"]!="accepted_exact_recalibration":continue
   grouped[condition].append(result)
   row={"dataset":dataset,"distribution":job["distribution"],"ablation":cfg["ablation_component"],"seed":cfg["seed"],"checkpoint_sha256":result["checkpoint_sha256"]}
   for view in ("raw","calibrated"):
    row.update({view+"_"+k:v for k,v in result[view].items()})
    for group,stats in result[view+"_groups"].items():
     row.update({f"{view}_group{group}_{k}":v for k,v in stats.items()})
   perseed.append(row)
 summary=[];coverage=[]
 for condition,rows in sorted(grouped.items()):
  planned=[json.loads(Path(path).read_text())["config"]["seed"] for path in manifest["jobs"][condition[0]]
           if (json.loads(Path(path).read_text())["dataset"],json.loads(Path(path).read_text())["distribution"],json.loads(Path(path).read_text())["config"]["ablation_component"])==condition]
  complete=len(rows)==len({r["seed"] for r in rows})==10 and {r["seed"] for r in rows}==set(planned)
  info=dict(zip(("dataset","distribution","ablation"),condition))
  coverage.append(dict(info,n_exact_pairs=len(rows),n_planned=10,status="complete_10seed" if complete else "incomplete_no_formal_mean"))
  if not complete:continue
  for metric in metric_keys:
   def values(view):
    if metric.startswith("group"):
     group,rate=metric.split("_");return [r[view+"_groups"][group[-1]][rate] for r in rows]
    return [r[view][metric] for r in rows]
   raw=values("raw");cal=values("calibrated")
   if not all(v is not None and math.isfinite(v) for v in raw+cal):continue
   delta=[b-a for a,b in zip(raw,cal)]
   summary.append(dict(info,metric=metric,n_pairs=10,raw_mean=statistics.mean(raw),raw_sample_std=statistics.stdev(raw),
    calibrated_mean=statistics.mean(cal),calibrated_sample_std=statistics.stdev(cal),
    paired_delta_mean=statistics.mean(delta),paired_delta_sample_std=statistics.stdev(delta),std_ddof=1))
 csv_write("per_seed_raw_calibrated.csv",perseed);csv_write("condition_coverage.csv",coverage);csv_write("formal_paired_summary.csv",summary)
 report={"planned_checkpoint_cases":sum(map(len,manifest["jobs"].values())),"accepted_exact_pairs":len(perseed),
  "mismatches":[r for r in statuses if r["status"]=="cpu_recalibration_mismatch"],
  "not_evaluated":[r for r in statuses if r["status"]=="not_evaluated"],
  "complete_10seed_conditions":sum(r["status"]=="complete_10seed" for r in coverage),
  "new_training_runs":0,"new_seeds":0,"adult_historical_full_missing_checkpoints":20,
  "interpretation":"Same-checkpoint argmax vs root-fitted group thresholds; isolates prediction postprocessing, not a retrained calibration-off algorithm. No performance ranking.",
  "limitations":["Strict equality in three aggregate reported metrics does not prove original GPU and CPU per-example logits or predictions were identical.",
   "Only complete sets of ten exact-matched checkpoint pairs enter formal summaries; any mismatched case leaves its condition incomplete, not a silently filtered mean.",
   "Historical Adult Full has no available checkpoint in these cohorts and is not reconstructed or retrained."]}
 write(OUT/"verification.json",report);print(json.dumps(report),flush=True)
def prepare_manifest():
 jobs={}
 for dataset,root in SOURCES.items():
  m=json.loads((root/f"results/revision_20260923/{dataset}_ablation_v1/manifest.json").read_text())
  jobs[dataset]=m["jobs"]
 assert len(jobs["adult"])==120 and len(jobs["compas"])==140
 manifest={"jobs":jobs,"evaluation":"CPU only, 16 workers after per-dataset equality canary","new_training_runs":0}
 write(OUT/"manifest.json",manifest)
 history=SOURCES["adult"]/"results/revision_20260923/adult_ablation_v1/reused_full"
 missing=[{"wrapper":str(path),"status":"checkpoint_unavailable_not_retrained"} for path in sorted(history.glob("*.json"))]
 assert len(missing)==20;write(OUT/"adult_full_missing_checkpoints.json",missing)
 return manifest
def main():
 parser=argparse.ArgumentParser();parser.add_argument("action",choices=["prepare","case","batch","summarize"])
 parser.add_argument("--job");parser.add_argument("--dataset",choices=list(SOURCES));args=parser.parse_args()
 if args.action=="prepare":prepare_manifest();return
 manifest=json.loads((OUT/"manifest.json").read_text())
 if args.action=="case":
  job=json.loads(Path(args.job).read_text());init_worker(str(SOURCES[job["dataset"]]))
  result=evaluate(args.job);print(json.dumps({"job":result["job_id"],"dataset":result["dataset"],"status":result["status"],"mismatch":result.get("mismatch")}),flush=True);return
 if args.action=="summarize":summarize(manifest);return
 for dataset in ([args.dataset] if args.dataset else list(SOURCES)):
  paths=manifest["jobs"][dataset]
  canary_job=json.loads(Path(paths[0]).read_text())
  canary=json.loads((OUT/"cases"/f"{dataset}__{canary_job['id']}.json").read_text())
  assert canary["status"]=="accepted_exact_recalibration",(dataset,"canary did not pass")
  with ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context("spawn"),initializer=init_worker,initargs=(str(SOURCES[dataset]),)) as pool:
   futures={pool.submit(evaluate,path):path for path in paths}
   for i,future in enumerate(as_completed(futures),1):
    try:result=future.result()
    except Exception as exc:
     write(OUT/"failures"/(dataset+"__"+Path(futures[future]).stem+".json"),{"source_job":futures[future],"error":repr(exc)})
     raise
    if i%10==0 or result["status"]!="accepted_exact_recalibration":
     print(json.dumps({"dataset":dataset,"completed_evaluations":i,"job":result["job_id"],"status":result["status"]}),flush=True)
 summarize(manifest)
if __name__=="__main__":main()
