"""Frozen checkpoint-only, root-fitted shared calibration; no model training."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path('/workspace/GuardFed-celeba-expanded')
STAGE = ROOT / 'results/revision_20260928/celeba_shared_calibration_v1'
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def checked_output(job, manifest_hash, core, np):
    path = STAGE/'runs'/job['id']/'evaluation.json'
    if not path.exists(): return None
    result = json.loads(path.read_text())
    for key in ['id','method','distribution','attack']:
        assert result[key] == job[key], ('evaluation identity', key)
    assert result['seed'] == job['config']['seed'] and result['round'] == 70 and result['evaluation_split'] == 'valid'
    assert result['checkpoint_sha256'] == job['checkpoint_sha256'] == digest(Path(job['output'])/'model.pt')
    assert result['manifest_sha256'] == manifest_hash
    assert digest(Path(job['output'])/'result.json') == result['original_result_sha256']
    assert digest(path.parent/'margins.npz') == result['cache_sha256']
    original = json.loads((Path(job['output'])/'result.json').read_text())
    with np.load(path.parent/'margins.npz') as cache:
        assert len(cache['valid_y']) == 19867
        raw = core.compute_metrics(cache['valid_y'], (cache['valid_margins'] > 0).astype(int),cache['valid_sensitive'])
        shared = core.metrics_from_group_thresholds(cache['valid_y'],cache['valid_margins'],cache['valid_sensitive'],{int(k):v for k,v in result['thresholds'].items()})
    for k in core.METRICS:
        assert abs(raw[k]-result['raw'][k]) <= 1e-12 and abs(shared[k]-result['shared_calibration'][k]) <= 1e-12
        assert abs(result['native'][k]-original['metrics'][k]) <= 1e-12
    return result

def worker(shard, canary=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(shard % 2)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    import reproduce_paper_tables as core
    import run_revision_ablation as original
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    manifest = json.loads((STAGE/'manifest.json').read_text())
    manifest_hash = digest(STAGE/'manifest.json')
    assert digest(STAGE/'PROTOCOL.md') == manifest['protocol_sha256']
    for name, expected in manifest['evaluation_source_hashes'].items():
        assert digest(ROOT/name) == expected, ('evaluation source drift', name)
    for name, expected in manifest['data_hashes'].items():
        assert digest(ROOT/name) == expected, ('data drift', name)
    device = torch.device('cuda')
    selected = [j for j in manifest['jobs'] if j['id'] in manifest['canary_ids']] if canary else manifest['jobs']
    groups = sorted({(j['distribution'], j['config']['seed']) for j in selected})
    for gi, (dist, seed) in enumerate(groups):
        if not canary and gi % manifest['workers'] != shard: continue
        group = [j for j in selected if (j['distribution'], j['config']['seed']) == (dist, seed)]
        todo = []
        for j in group:
            if checked_output(j, manifest_hash, core, np) is None: todo.append(j)
        if not todo: continue
        if list((STAGE/'runs').glob('*/failure.json')) or list(STAGE.glob('shard_failure_*.json')):
            raise RuntimeError('Unresolved queue failure; dispatch stopped')
        cfg = core.ExperimentConfig(**todo[0]['config'])
        core.set_seed(seed, deterministic_image=True)
        bundle = core.load_bundle('celeba', cfg.client_alpha, cfg, device)
        # Reuse the original split routine, then release client pixels: evaluation uses root/valid only.
        del bundle['clients']
        root_y = bundle['server_y'].numpy(); root_s = np.asarray(bundle['server_sensitive'])
        valid_y = bundle['y_test'].numpy(); valid_s = np.asarray(bundle['test_sensitive'])
        assert len(valid_y) == 19867 and bundle['train_rows'] == 162770
        for j in todo:
            if list((STAGE/'runs').glob('*/failure.json')) or list(STAGE.glob('shard_failure_*.json')):
                raise RuntimeError('Unresolved queue failure; dispatch stopped')
            out = STAGE/'runs'/j['id']; out.mkdir(parents=True, exist_ok=True)
            if (out/'failure.json').exists(): raise RuntimeError('Unresolved failed evaluation: '+j['id'])
            start = time.time()
            try:
                save(out/'progress.json', {'phase':'inference','pid':os.getpid(),'started_unix':start})
                old = original.checked_result(j)
                assert old is not None and digest(Path(j['output'])/'model.pt') == j['checkpoint_sha256']
                contract = bundle['image_data_contract']; prior = old['data_contract']['image_data_contract']
                for key in ['root_image_ids_sha256','train_image_ids_sha256','evaluation_image_ids_sha256','cache_manifest_sha256','evaluation_split']:
                    assert contract[key] == prior[key], (j['id'], key)
                config = core.ExperimentConfig(**j['config'])
                core.set_seed(seed, deterministic_image=True)
                model = core.make_model(bundle, config, device)
                model.load_state_dict(torch.load(Path(j['output'])/'model.pt', map_location=device, weights_only=True))
                root_m = core.model_margins(model, bundle['server_X'], config.batch_size)
                valid_m = core.model_margins(model, bundle['X_test'], config.batch_size)
                assert np.isfinite(root_m).all() and np.isfinite(valid_m).all()
                raw = core.compute_metrics(valid_y, (valid_m > 0).astype(int), valid_s)
                shared_cfg = dataclasses.replace(config, **manifest['calibration'])
                saved_fn = core.model_margins
                try:
                    # Feed cached root margins into the unchanged original threshold search.
                    def cached_margins(_model, X, _batch):
                        assert X is bundle['server_X']
                        return root_m
                    core.model_margins = cached_margins
                    thresholds, calibration_info = core.fit_group_thresholds(model, bundle, shared_cfg)
                    shared = core.metrics_from_group_thresholds(valid_y, valid_m, valid_s, thresholds)
                    if j['method'] == 'GuardFed-AD2+':
                        native_thresholds, _ = core.fit_group_thresholds(model, bundle, config)
                        native = core.metrics_from_group_thresholds(valid_y, valid_m, valid_s, native_thresholds)
                    else: native = raw
                finally: core.model_margins = saved_fn
                differences = {k: native[k]-old['metrics'][k] for k in core.METRICS}
                save(out/'native_replay_check.json', {'differences':differences,'native':native,'original':old['metrics']})
                assert all(abs(v) <= 1e-12 for v in differences.values()), ('native replay mismatch', j['id'], differences)
                if canary:
                    direct = core.evaluate_for_reporting(j['method'],model,bundle,config)
                    direct_shared = core.evaluate_model_calibrated(model,bundle,shared_cfg)
                    assert all(abs(direct[k]-native[k]) <= 1e-12 and abs(direct_shared[k]-shared[k]) <= 1e-12 for k in core.METRICS)
                np.savez_compressed(out/'margins.npz', root_margins=root_m, valid_margins=valid_m,
                                    root_y=root_y,root_sensitive=root_s,valid_y=valid_y,valid_sensitive=valid_s)
                result = {'id':j['id'],'method':j['method'],'distribution':dist,'attack':j['attack'],'seed':seed,
                          'round':70,'evaluation_split':'valid','checkpoint_sha256':j['checkpoint_sha256'],
                          'original_result_sha256':digest(Path(j['output'])/'result.json'),
                          'manifest_sha256':manifest_hash,'cache_sha256':digest(out/'margins.npz'),
                          'root_contract':contract,'raw':raw,'shared_calibration':shared,'native':native,
                          'thresholds':thresholds,'calibration_info':calibration_info,'native_replay_differences':differences,
                          'canary_direct_replay':canary,'root_counts':{str(g):{'n':int((root_s==g).sum()),'positives':int(((root_s==g)&(root_y==1)).sum())} for g in [0,1]},
                          'valid_counts':{str(g):{'n':int((valid_s==g).sum()),'positives':int(((valid_s==g)&(valid_y==1)).sum())} for g in [0,1]},
                          'torch_version':torch.__version__,'gpu':os.environ['CUDA_VISIBLE_DEVICES'],
                          'duration_sec':time.time()-start,'finished_unix':time.time()}
                save(out/'evaluation.json',result)
                checked_output(j, manifest_hash, core, np)
                save(out/'progress.json',{'phase':'complete','pid':os.getpid(),'finished_unix':time.time()})
                print(json.dumps({'complete':j['id'],'seconds':result['duration_sec']}),flush=True)
                del model
            except Exception:
                save(out/'failure.json',{'id':j['id'],'traceback':traceback.format_exc(),'time':time.time()})
                raise
        del bundle

def run():
    manifest = json.loads((STAGE/'manifest.json').read_text()); processes=[]
    assert not list((STAGE/'runs').glob('*/failure.json')) and not list(STAGE.glob('shard_failure_*.json')), 'Resolve preserved failures before restart'
    for shard in range(manifest['workers']):
        log=(STAGE/f'worker_{shard}.log').open('a')
        p=subprocess.Popen([sys.executable,__file__,'worker','--shard',str(shard)],stdout=log,stderr=subprocess.STDOUT)
        processes.append((p,log))
    failed=False
    while any(p.poll() is None for p,_ in processes):
        if any(p.poll() not in (None,0) for p,_ in processes):
            failed=True
            save(STAGE/'shard_failure_coordinator.json',{'reason':'nonzero worker exit; stop new dispatch','time':time.time()})
            # Finish in-flight checkpoint evaluations; do not relaunch failed workers.
        time.sleep(3)
    codes=[p.returncode for p,_ in processes]
    for _,log in processes: log.close()
    save(STAGE/'queue_exit.json',{'codes':codes,'failed':failed or any(codes),'time':time.time()})
    if any(codes): raise SystemExit(1)
    assert len(list((STAGE/'runs').glob('*/evaluation.json'))) == manifest['total'], 'Incomplete queue'

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['worker','canary','run']);parser.add_argument('--shard',type=int,default=0);args=parser.parse_args()
    try:
        if args.action=='run': run()
        else: worker(args.shard,args.action=='canary')
    except Exception:
        save(STAGE/f'shard_failure_{args.shard}.json',{'traceback':traceback.format_exc(),'time':time.time()})
        raise
