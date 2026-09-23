#!/usr/bin/env python3
"""Small CPU checks for CNN plumbing, F Flip effect, and exact tabular compatibility."""
import copy,json,os,subprocess,sys,tempfile,types
from dataclasses import asdict
from pathlib import Path
os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np
import torch
import reproduce_paper_tables as core
from src.celeba_data import CelebACNN
torch.set_num_threads(1);torch.set_num_interop_threads(1)

def bundle(image=False):
    gen=torch.Generator().manual_seed(7)
    def x(n):
        return torch.randint(0,256,(n,3,64,64),generator=gen,dtype=torch.uint8) if image else torch.randn(n,3,generator=gen)
    labels=torch.tensor([0,1,0,1,1,0,1,0],dtype=torch.long)
    sens=np.array([0,0,1,1,0,0,1,1])
    clients={i:{"X":x(8),"y":labels.clone(),"sensitive":sens.copy(),"sensitive_feature_index":None} for i in range(4)}
    return {"dataset":"celeba" if image else "adult","label_col":"label","sensitive_col":"sensitive","feature_includes_label":False,
        "feature_includes_sensitive":False,"num_features":3,"train_rows":40,"test_rows":8,"clients":clients,
        "server_X":x(8),"server_y":labels.clone(),"server_sensitive":sens.copy(),"root_clean_rows":8,"root_synthetic_rows":0,
        "X_test":x(8),"y_test":labels.clone(),"test_sensitive":sens.copy(),
        "rw_weights":{(0,0):0.4,(1,0):1.4,(0,1):0.6,(1,1):1.6},"synthetic_method":"none",
        "server_sampling_audit":{},"root_noise_audit":{}}

def main():
    base=types.ModuleType("guardfed_tabular_base")
    base.__file__=str(ROOT/"scripts/reproduce_paper_tables.py")
    sys.modules[base.__name__]=base
    source=subprocess.check_output(["git","show","b1a808b:scripts/reproduce_paper_tables.py"],cwd=ROOT,text=True)
    exec(compile(source,base.__file__,"exec"),base.__dict__)
    cfg=core.ExperimentConfig(seed=123,num_clients=4,num_malicious=1,batch_size=8,rounds=1,
        device="cpu",synthetic_method="none",full_round_diagnostics=True,ad2_calibration_quantiles=7)
    b=bundle(False);base.load_bundle=lambda *args:b;original_loader=core.load_bundle;core.load_bundle=lambda *args:b
    base_cfg=base.ExperimentConfig(**{k:v for k,v in asdict(cfg).items() if k in base.ExperimentConfig.__dataclass_fields__})
    with tempfile.TemporaryDirectory() as td:
        old=base.run_experiment("adult","IID","GuardFed-AD2+","Benign",base_cfg,"check",torch.device("cpu"),checkpoint_path=Path(td)/"base.pt")
        new=core.run_experiment("adult","IID","GuardFed-AD2+","Benign",cfg,"check",torch.device("cpu"),checkpoint_path=Path(td)/"new.pt")
        a=torch.load(Path(td)/"base.pt",weights_only=True);z=torch.load(Path(td)/"new.pt",weights_only=True)
        assert all(torch.equal(a[k],z[k]) for k in a)
        assert old["metrics"]==new["metrics"]
        assert old["round_summaries"]==new["round_summaries"]
    b=bundle(True)
    model=core.make_model(b,cfg,torch.device("cpu"))
    assert isinstance(model,CelebACNN) and model(b["server_X"]).shape==(8,2)
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    raw=b["clients"][0]
    benign,audit_b=core.client_runtime_data(raw,0,"Benign",[0],b["rw_weights"],torch.device("cpu"),"all_unprivileged","state","fedsa","fedsa")
    attack,audit_a=core.client_runtime_data(raw,0,"F Flip",[0],b["rw_weights"],torch.device("cpu"),"all_unprivileged","state","fedsa","fedsa")
    assert torch.equal(benign["X"],attack["X"]) and torch.equal(benign["y"],attack["y"])
    assert not torch.equal(benign["weights"],attack["weights"])
    assert audit_a["label_changed_count"]==0 and audit_a["fflip_changed"]>0
    core.set_seed(123);u=core.train_local_model(model,benign,cfg)
    core.set_seed(123);v=core.train_local_model(model,attack,cfg)
    changed=max(float((u[k]-v[k]).abs().max()) for k in u)
    assert changed>0
    core.load_bundle=lambda *args:b
    with tempfile.TemporaryDirectory() as td:
        r=core.run_experiment("celeba","IID","GuardFed-AD2+","Benign",cfg,"synthetic_check",torch.device("cpu"),checkpoint_path=Path(td)/"cnn.pt")
        info=r["round_summaries"][0]["aggregate"]
        assert len(info["ad2_plus_candidates"])==10
        assert len(info["client_weights"])==4 and abs(sum(info["client_weights"])-1)<1e-6
        assert all(np.isfinite(x) for x in r["metrics"].values())
        assert Path(td,"cnn.pt").exists()
    core.load_bundle=original_loader
    report={"tabular_bitwise_checkpoint_equal":True,"tabular_round_diagnostics_equal":True,
        "cnn_model_parameters":sum(p.numel() for p in model.parameters()),"cnn_shape":[8,2],"cnn_bn":False,
        "fflip_changes_pixels":False,"fflip_changes_labels":False,"fflip_changes_weights":True,
        "fflip_model_max_abs_change":changed,"adaptive_candidate_count":10,
        "cnn_synthetic_one_round_complete":True,"formal_training_started":False}
    print(json.dumps(report))
if __name__=="__main__":main()
