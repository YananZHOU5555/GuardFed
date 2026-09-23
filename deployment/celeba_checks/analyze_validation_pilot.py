#!/usr/bin/env python3
"""Read-only checkpoint analysis of CelebA validation pilots. Never touches test."""
import argparse,json,os,sys,time
from pathlib import Path
os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="0")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import reproduce_paper_tables as core
import run_revision_ablation as runner
torch.set_num_threads(1);torch.set_num_interop_threads(1)

def rates(y,pred,s):
    result=core.compute_metrics(y,pred,s)
    result.update({"predicted_positive_count":int(pred.sum()),"constant_prediction":bool(np.unique(pred).size==1),
        "confusion":{"tn":int(((y==0)&(pred==0)).sum()),"fp":int(((y==0)&(pred==1)).sum()),
                     "fn":int(((y==1)&(pred==0)).sum()),"tp":int(((y==1)&(pred==1)).sum())}})
    return result

def evaluate(model,bundle,cfg):
    thresholds,calibration=core.fit_group_thresholds(model,bundle,cfg)
    margins=core.model_margins(model,bundle["X_test"],cfg.batch_size)
    y=bundle["y_test"].numpy().astype(int);s=bundle["test_sensitive"]
    raw=(margins>0).astype(int)
    calibrated=np.zeros(len(y),dtype=int)
    for group in [0,1]:
        mask=s==group;calibrated[mask]=(margins[mask]>=thresholds[group]).astype(int)
    return {"raw":rates(y,raw,s),"calibrated":rates(y,calibrated,s),
       "validation_roc_auc":float(roc_auc_score(y,margins)),
       "validation_cross_entropy":float(np.mean(np.logaddexp(0,margins)-y*margins)),
       "margin_mean":float(np.mean(margins)),"margin_std":float(np.std(margins)),
       "calibration_thresholds":thresholds,"calibration_fit_source":"train-root only",
       "root_calibration_info":calibration}

def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--output",required=True);p.add_argument("--wait",action="store_true")
    args=p.parse_args();manifest=json.loads(Path(args.manifest).read_text())
    jobs=[json.loads(Path(path).read_text()) for path in manifest["jobs"]]
    for job in jobs:
        assert job["dataset"]=="celeba" and job["config"]["celeba_evaluation_split"]=="valid"
    while any(not (Path(j["output"])/"result.json").exists() for j in jobs):
        failed=[j["id"] for j in jobs if (Path(j["output"])/"failure.json").exists()]
        if failed:raise RuntimeError("Training failure retained: "+str(failed))
        if not args.wait:raise RuntimeError("Results not complete; no training was started by evaluator")
        time.sleep(10)
    for name,h in manifest["source_hashes"].items():assert runner.digest(ROOT/name)==h, name
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs=[]
    initial_cache={}
    for job in jobs:
        result=runner.checked_result(job)
        cfg=core.ExperimentConfig(**job["config"])
        bundle=core.load_bundle("celeba",result["alpha"],cfg,device)
        assert bundle["image_data_contract"]["evaluation_split"]=="valid"
        expected_rows = cfg.celeba_eval_limit or bundle["image_data_contract"]["official_split_sizes"]["1"]
        assert bundle["test_rows"]==expected_rows
        model=core.make_model(bundle,cfg,device)
        initial_state=core.clone_state(model)
        initial=evaluate(model,bundle,cfg)
        state=torch.load(Path(job["output"])/"model.pt",map_location=device,weights_only=True)
        model.load_state_dict(state)
        final=evaluate(model,bundle,cfg)
        curve=result["trajectory_metrics"]
        delta=np.sqrt(sum(float(((state[k]-initial_state[k])**2).sum().cpu()) for k in state))
        outputs.append({"job_id":job["id"],"method":job["method"],"checkpoint_sha256":result["revision_job"]["checkpoint_sha256"],
            "training_rounds":cfg.rounds,"data_contract":bundle["image_data_contract"],"initial_reference":initial,"final_same_checkpoint":final,
            "parameter_l2_change":float(delta),"recorded_curve_reporting":"calibrated" if job["method"]=="GuardFed-AD2+" and cfg.ad2_calibration_enabled else "raw",
            "recorded_curve":curve,"first_round":curve[0],"last_round":curve[-1],
            "final_raw_accuracy_gain_over_initial":final["raw"]["accuracy"]-initial["raw"]["accuracy"],
            "final_raw_accuracy_gain_over_majority":final["raw"]["accuracy"]-final["raw"]["majority_accuracy"],
            "validation_auc_gain":final["validation_roc_auc"]-initial["validation_roc_auc"],
            "validation_cross_entropy_reduction":initial["validation_cross_entropy"]-final["validation_cross_entropy"]})
    report={"kind":"validation_only_readonly_pilot_analysis","manifest":args.manifest,"evaluation_device":str(device),"jobs":outputs,
        "training_started_by_evaluator":False,"test_data_used":False,
        "limitations":["Single-seed learning pilots on the train/official-validation sizes in each data contract; not formal generalization evidence.",
        "Recorded GuardFed curve is calibrated whereas FedAvg curve is raw. Compare explicitly raw or calibrated columns at the same checkpoint.",
        "Raw/calibrated are two evaluations of each unchanged final checkpoint; thresholds are fitted only on that run's training-root data.",
        "Weight changes or low fairness gaps alone do not establish learning. Constant predictions and majority baselines are reported.",
        "Initial reference uses the identical seeded model. Validation is allowed for pilot decisions; official test remains untouched.",
        "A 70-round extension would be a predefined learning-curve pilot, not a search for the best test checkpoint. Preserve all curves and endpoints."]}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    runner.write_json(out/"evaluation.json",report)
    lines=["# CelebA验证集pilot：同一checkpoint原始/校准评估","",
        "只读现有checkpoint；没有启动训练，也没有使用官方test。阈值仅由train-root拟合。","",
        "| 方法 | 初始raw ACC | 最终raw ACC | 最终cal ACC | 多数类ACC | 最终raw正例率 | 初始/最终AUC | 初始/最终CE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in outputs:
        a=r["initial_reference"];b=r["final_same_checkpoint"]
        lines.append(f"| {r['method']} | {a['raw']['accuracy']:.4f} | {b['raw']['accuracy']:.4f} | {b['calibrated']['accuracy']:.4f} | {b['raw']['majority_accuracy']:.4f} | {b['raw']['positive_rate']:.4f} | {a['validation_roc_auc']:.4f}/{b['validation_roc_auc']:.4f} | {a['validation_cross_entropy']:.4f}/{b['validation_cross_entropy']:.4f} |")
    lines+=["","## 已记录曲线（GuardFed校准、FedAvg原始，不能混为同一口径）",""]
    for r in outputs:
        values=", ".join(f"{p['round']}:{p['metrics']['accuracy']:.4f}" for p in r["recorded_curve"])
        lines.append(f"- {r['method']}: {values}")
        lines.append(f"  最终raw常数预测={r['final_same_checkpoint']['raw']['constant_prediction']}；cal常数预测={r['final_same_checkpoint']['calibrated']['constant_prediction']}。")
    lines+=["","## 科学限制",""]+["- "+s for s in report["limitations"]]
    (out/"evaluation.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(out),"summary":[{"method":r["method"],"initial":r["initial_reference"],"final":r["final_same_checkpoint"],"raw_gain_majority":r["final_raw_accuracy_gain_over_majority"]} for r in outputs]}),flush=True)
if __name__=="__main__":main()
