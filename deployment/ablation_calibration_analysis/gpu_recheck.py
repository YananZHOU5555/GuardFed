#!/usr/bin/env python3
import argparse,contextlib,hashlib,importlib.util,io,json,os,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
OUT=Path(__file__).resolve().parent
DEST=OUT/"gpu_rechecks"
ROOT=Path("/workspace/GuardFed-next")
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+".tmp");temp.write_text(json.dumps(obj,indent=2));temp.replace(path)
def groups(y,pred,sensitive):
 import numpy as np
 result={}
 for group in (0,1):
  mask=sensitive==group;positive=mask&(y==1);negative=mask&(y==0)
  result[str(group)]={"n":int(mask.sum()),"positive_support":int(positive.sum()),"negative_support":int(negative.sum()),
   "predicted_positive":int(np.sum(pred[mask]==1)),"true_positive":int(np.sum(pred[positive]==1)),"false_positive":int(np.sum(pred[negative]==1)),
   "TPR":float(np.mean(pred[positive]==1)) if positive.any() else None,"FPR":float(np.mean(pred[negative]==1)) if negative.any() else None}
 return result
def worker(case_path):
 case=json.loads(Path(case_path).read_text());assert case["status"]=="cpu_recalibration_mismatch"
 job=json.loads(Path(case["source_job"]).read_text());original=json.loads(Path(case["source_result"]).read_text())
 visible=original["revision_job"]["visible_gpu"];assert visible in ("0","1")
 os.environ.update(CUDA_VISIBLE_DEVICES=visible,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
 import torch,numpy as np
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 sys.path.insert(0,str(ROOT))
 spec=importlib.util.spec_from_file_location("original_gpu_analysis_core",ROOT/"scripts/reproduce_paper_tables.py")
 core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
 assert torch.cuda.device_count()==1
 for name,digest in job["source_hashes"].items():assert sha(ROOT/name)==digest
 assert sha(case["checkpoint"])==case["checkpoint_sha256"]==original["revision_job"]["checkpoint_sha256"]
 assert sha(case["source_result"])==case["source_result_sha256"]
 assert original["config"]==job["config"]==case["source_config"]
 cfg=core.ExperimentConfig(**original["config"]);assert cfg.device=="cuda"
 with contextlib.redirect_stdout(io.StringIO()):
  bundle=core.load_bundle(job["dataset"],original["alpha"],cfg,torch.device("cuda"))
 for key in ("num_features","train_rows","test_rows","root_clean_rows","root_synthetic_rows","server_sampling_audit"):
  assert bundle[key]==original["data_contract"][key]
 model=core.SimpleMLP(bundle["num_features"],cfg.seed).to("cuda")
 model.load_state_dict(torch.load(case["checkpoint"],map_location="cuda",weights_only=True));model.eval()
 thresholds,info=core.fit_group_thresholds(model,bundle,cfg)
 margins=core.model_margins(model,bundle["X_test"],cfg.batch_size)
 y=bundle["y_test"].cpu().numpy();s=np.asarray(bundle["test_sensitive"])
 calibrated=core.metrics_from_group_thresholds(y,margins,s,thresholds)
 metrics={k:calibrated[k] for k in core.METRICS};reported=original["metrics"]
 output=dict(case,evaluation_device="cuda",gpu_visible=True,physical_visible_gpu=visible,
  gpu_name=torch.cuda.get_device_name(0),torch_version=torch.__version__,gpu_recalibrated=metrics,
  gpu_thresholds=thresholds,calibration_info=info,original_cpu_result=str(case_path),original_cpu_result_sha256=sha(case_path),
  recheck_script_sha256=sha(__file__),status="gpu_recalibration_mismatch",formal_stat_eligible=False)
 if metrics==reported:
  preds=[]
  with torch.no_grad():
   for i in range(0,len(bundle["X_test"]),cfg.batch_size):
    preds.append(model(bundle["X_test"][i:i+cfg.batch_size]).argmax(dim=1).cpu().numpy())
  raw_pred=np.concatenate(preds);cal_pred=np.zeros_like(y)
  for group in (0,1):
   mask=s==group;cal_pred[mask]=(margins[mask]>=thresholds[group]).astype(int)
  raw=core.compute_metrics(y,raw_pred,s)
  output.update(status="accepted_exact_recalibration",formal_stat_eligible=True,
   raw={k:raw[k] for k in core.METRICS},calibrated=metrics,raw_groups=groups(y,raw_pred,s),
   calibrated_groups=groups(y,cal_pred,s),paired_delta_calibrated_minus_raw={k:metrics[k]-raw[k] for k in core.METRICS},
   recovered_on_original_gpu=True,limitation="CPU mismatch retained unchanged; this pair was reproduced on the original physical GPU. Recheck is evaluation only, not a new training seed.")
 else:
  output["gpu_mismatch"]={k:{"original":reported[k],"gpu":metrics[k]} for k in core.METRICS if metrics[k]!=reported[k]}
 write(DEST/"cases"/Path(case_path).name,output)
 print(json.dumps({"job":job["id"],"gpu":visible,"status":output["status"],"gpu_mismatch":output.get("gpu_mismatch")}),flush=True)
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--case");args=parser.parse_args()
 if args.case:worker(args.case);return
 pending={"0":[],"1":[]};original_hashes={}
 for path in sorted((OUT/"cases").glob("*.json")):
  case=json.loads(path.read_text());original_hashes[str(path)]=sha(path)
  if case["status"]=="cpu_recalibration_mismatch":
   original=json.loads(Path(case["source_result"]).read_text());pending[original["revision_job"]["visible_gpu"]].append(str(path))
 assert sum(map(len,pending.values()))==6
 write(DEST/"manifest.json",{"cases_by_original_gpu":pending,"original_cpu_case_hashes":original_hashes,
  "concurrency_per_gpu":1,"torch_cpu_threads":1,"new_training":False})
 def queue(gpu,paths):
  for path in paths:
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
   subprocess.run([sys.executable,__file__,"--case",path],env=env,check=True)
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(queue,gpu,paths) for gpu,paths in pending.items()]
  for future in futures:future.result()
 assert all(sha(path)==digest for path,digest in original_hashes.items()),"Original CPU records changed"
 results=[json.loads(p.read_text()) for p in (DEST/"cases").glob("*.json")]
 write(DEST/"verification.json",{"evaluated":len(results),"exact_matches":sum(r["status"]=="accepted_exact_recalibration" for r in results),
  "mismatches":[r["job_id"] for r in results if r["status"]!="accepted_exact_recalibration"],
  "all260_original_cpu_case_hashes_unchanged":True,"new_training_runs":0,"new_seeds":0})
 print((DEST/"verification.json").read_text(),flush=True)
if __name__=="__main__":main()
