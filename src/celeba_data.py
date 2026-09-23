"""CelebA CPU uint8 cache and the existing GuardFed split/reweighting protocol."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn


class CelebACNN(nn.Module):
    """64x64 RGB, three Conv/ReLU/Pool blocks, global pooling, two logits."""
    def __init__(self, seed=123):
        super().__init__()
        torch.manual_seed(seed)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        # Pixels stay uint8 on CPU until this batch; no augmentation or BN.
        if x.dtype != torch.uint8 or tuple(x.shape[1:]) != (3, 64, 64):
            raise ValueError("CelebACNN expects NCHW uint8 RGB64 batches")
        x = x.to(device=self.classifier.weight.device, dtype=torch.float32)
        return self.classifier(self.features(x / 255.0).flatten(1))


def load_celeba_bundle(alpha, config, core):
    if config.include_sensitive_feature:
        raise ValueError("Male is metadata only; it is never an image input feature")
    if config.synthetic_ratio:
        raise ValueError("Tabular root synthesis is not supported for images")
    if config.celeba_evaluation_split not in {"valid", "test"}:
        raise ValueError("CelebA evaluation split must be valid or test")
    cache = Path(config.celeba_cache_dir) if config.celeba_cache_dir else core.ROOT / "data/celeba/derived/rgb64_v1"
    manifest = json.loads((cache / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("CelebA RGB cache is incomplete; do not silently omit missing images")
    metadata = np.load(cache / "metadata.npz")
    images = np.load(cache / "images.npy", mmap_mode="r")
    ids, split, labels, sensitive = (metadata[k] for k in ["image_id", "split", "Smiling", "Male"])
    assert images.shape == (len(ids), 3, 64, 64) and images.dtype == np.uint8
    assert np.array_equal(ids, np.arange(1, len(ids) + 1))
    assert np.load(cache / "available.npy", mmap_mode="r").all()
    train_pos = np.flatnonzero(split == 0)
    eval_split = 1 if config.celeba_evaluation_split == "valid" else 2
    eval_pos = np.flatnonzero(split == eval_split)
    official_sizes = {str(i): int((split == i).sum()) for i in [0, 1, 2]}
    # A seeded subset of the full official split, not identity-sorted parquet rows.
    rng = np.random.default_rng(config.seed)
    for name, limit in [("train", config.celeba_train_limit), ("eval", config.celeba_eval_limit)]:
        if limit < 0:
            raise ValueError("Pilot subset limits must be nonnegative")
    if config.celeba_train_limit:
        train_pos = np.sort(rng.choice(train_pos, min(config.celeba_train_limit, len(train_pos)), replace=False))
    if config.celeba_eval_limit:
        eval_pos = np.sort(rng.choice(eval_pos, min(config.celeba_eval_limit, len(eval_pos)), replace=False))
    df = pd.DataFrame({"image_id": ids[train_pos], "Smiling": labels[train_pos], "Male": sensitive[train_pos]})
    root_df, sampling_audit = core.sample_server_dataframe(df, "Smiling", "Male", config)
    client_df = df.drop(root_df.index).reset_index(drop=True)
    raw_clients = core.create_client_data_dict(client_df, ["image_id"], "Smiling", "Male",
                                               config.num_clients, alpha, torch.device("cpu"), config.seed)
    root_df, noise_audit = core.apply_root_noise(root_df.reset_index(drop=True), "Smiling", "Male", config)
    def pixels(image_ids):
        # One compact CPU uint8 copy per partition; runtime shares it without another clone.
        return torch.from_numpy(np.array(images[np.asarray(image_ids, dtype=np.int64) - 1], copy=True))
    for client in raw_clients.values():
        client_ids = client["X"][:, 0].numpy().astype(np.int64)
        client["X"] = pixels(client_ids)
        client["image_ids"] = client_ids
        client["sensitive_feature_index"] = None
    root_ids = root_df["image_id"].to_numpy(dtype=np.int64)
    union = np.concatenate([v["image_ids"] for v in raw_clients.values()])
    assert len(set(root_ids) & set(union)) == 0
    assert np.array_equal(np.sort(np.concatenate([root_ids, union])), ids[train_pos])
    assert not np.intersect1d(ids[train_pos], ids[eval_pos]).size
    rw = core.compute_reweighing_weights(df, "Male", "Smiling") if config.use_reweighting else {(s, y): 1.0 for s in [0, 1] for y in [0, 1]}
    cache_hash = hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest()
    image_contract = {
        "numerical_execution": {"version": "celeba_deterministic_fp32_v1",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cublas_workspace_config": ":4096:8"},
        "model": "Conv32/64/128_3x3_ReLU_MaxPool_GAP_Linear2", "image_shape": [3, 64, 64],
        "normalization": "uint8 divided by 255 per batch", "augmentation": "none",
        "target": "Smiling", "sensitive": "Male", "evaluation_split": config.celeba_evaluation_split,
        "official_split_sizes": official_sizes, "actual_train_rows": len(train_pos), "actual_evaluation_rows": len(eval_pos),
        "evidence_stage": "pilot_subset" if config.celeba_train_limit or config.celeba_eval_limit else "full_official_split",
        "train_image_ids_sha256": hashlib.sha256(ids[train_pos].tobytes()).hexdigest(),
        "evaluation_image_ids_sha256": hashlib.sha256(ids[eval_pos].tobytes()).hexdigest(),
        "root_image_ids_sha256": hashlib.sha256(root_ids.tobytes()).hexdigest(),
        "cache_manifest_sha256": cache_hash, "official_metadata": manifest["official_metadata"],
        "train_eval_disjoint": True, "root_client_disjoint": True,
        "client_sample_counts": [int(len(v["y"])) for v in raw_clients.values()],
    }
    return {"dataset": "celeba", "label_col": "Smiling", "sensitive_col": "Male",
            "feature_cols": ["RGB"], "feature_includes_sensitive": False, "feature_includes_label": False,
            "server_X": pixels(root_ids), "server_y": torch.tensor(root_df["Smiling"].to_numpy(), dtype=torch.long),
            "server_sensitive": root_df["Male"].to_numpy(dtype=int), "root_clean_rows": len(root_ids),
            "root_synthetic_rows": 0, "server_sampling_audit": sampling_audit, "root_noise_audit": noise_audit,
            "synthetic_method": "none", "clients": raw_clients, "X_test": pixels(ids[eval_pos]),
            "y_test": torch.tensor(labels[eval_pos], dtype=torch.long), "test_sensitive": sensitive[eval_pos].astype(int),
            "num_features": 3, "rw_weights": rw, "train_rows": len(train_pos), "test_rows": len(eval_pos),
            "image_data_contract": image_contract}
