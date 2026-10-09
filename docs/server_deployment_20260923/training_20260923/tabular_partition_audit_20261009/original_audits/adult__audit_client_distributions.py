import os,sys,json,hashlib,collections,time
from pathlib import Path
os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",GUARDFED_CPU_THREADS="1",CUDA_VISIBLE_DEVICES="")
ROOT=Path("/workspace/GuardFed-revision")
OUT=ROOT/"results/revision_20260923/adult_heterogeneity_v1"
sys.path.insert(0,str(ROOT/"scripts"))
import torch,numpy as np
import reproduce_paper_tables as core
torch.set_num_threads(1);torch.set_num_interop_threads(1)
manifest=json.loads((OUT/"manifest.json").read_text())
for path,h in manifest["source_hashes"].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
def counts(y,s):
 y=np.asarray(y,dtype=int);s=np.asarray(s,dtype=int)
 return {f"{a}|{b}":int(((s==a)&(y==b)).sum()) for a in [0,1] for b in [0,1]}
seen=set();reports=[];started=time.time()
for path in manifest["jobs"]:
 job=json.loads(Path(path).read_text());c=job["config"];key=(c["client_alpha"],c["seed"])
 if key in seen:continue
 seen.add(key);cfg=core.ExperimentConfig(**c)
 bundle=core.load_bundle("adult",cfg.client_alpha,cfg,torch.device("cpu"))
 clients=[]
 for cid,client in sorted(bundle["clients"].items()):
  y=client["y"].cpu().numpy();s=client["sensitive"];cross=counts(y,s)
  sens={str(a):int((s==a).sum()) for a in [0,1]};labels={str(a):int((y==a).sum()) for a in [0,1]}
  clients.append({"client_id":cid,"is_malicious_under_SDFA":cid<cfg.num_malicious,"n":len(y),"sensitive_counts":sens,"label_counts":labels,
   "sensitive_label_counts":cross,"empty_client":len(y)==0,"empty_sensitive_groups":[a for a,n in sens.items() if n==0],
   "empty_sensitive_label_groups":[a for a,n in cross.items() if n==0],
   "partition_tensor_sha256":hashlib.sha256(client["X"].cpu().numpy().tobytes()+y.tobytes()+np.asarray(s,dtype=np.int64).tobytes()).hexdigest()})
 total=sum(x["n"] for x in clients)
 report={"dataset":"adult","client_alpha":cfg.client_alpha,"seed":cfg.seed,"partition_stage":"before adversarial sensitive-attribute corruption",
  "source_hashes":manifest["source_hashes"],"client_count":len(clients),"client_total_samples":total,"clients":clients,
  "empty_client_ids":[x["client_id"] for x in clients if x["empty_client"]],
  "clients_with_missing_sensitive_group":sum(bool(x["empty_sensitive_groups"]) for x in clients),
  "clients_with_missing_cross_group":sum(bool(x["empty_sensitive_label_groups"]) for x in clients),
  "malicious_client_samples":sum(x["n"] for x in clients if x["is_malicious_under_SDFA"]),
  "malicious_sample_fraction":sum(x["n"] for x in clients if x["is_malicious_under_SDFA"])/total,
  "root_cross_counts":counts(bundle["server_y"].cpu().numpy(),bundle["server_sensitive"]),
  "test_cross_counts":counts(bundle["y_test"].cpu().numpy(),bundle["test_sensitive"]),
  "server_sampling_audit":bundle["server_sampling_audit"],"train_rows":bundle["train_rows"],"root_clean_rows":bundle["root_clean_rows"],
  "no_training":True,"no_resampling":True}
 assert total+bundle["root_clean_rows"]==bundle["train_rows"]
 if cfg.seed==123:
  matches=list((OUT/"pilot").glob(f"cpu_c*/runs/a{cfg.client_alpha:g}_*/result.json"))
  assert len(matches)==6
  assert all([x["samples"] for x in json.loads(p.read_text())["attack_audit"]]==[x["n"] for x in clients] for p in matches)
 reports.append(report)
 print(json.dumps({"alpha":cfg.client_alpha,"seed":cfg.seed,"empty_clients":len(report["empty_client_ids"]),"malicious_sample_fraction":report["malicious_sample_fraction"]}),flush=True)
assert len(reports)==30
payload={"kind":"actual_client_partition_audit","source":"same frozen core.load_bundle as training","no_training":True,"no_resampling":True,
 "elapsed_sec":time.time()-started,"alpha_seed_count":30,"client_records":600,"cross_key":"sensitive|label","reports":reports}
(OUT/"client_distribution_audit.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({"audit_complete":True,"pairs":30,"clients":600,"elapsed_sec":payload["elapsed_sec"]}))
