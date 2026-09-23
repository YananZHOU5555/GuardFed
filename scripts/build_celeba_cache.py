#!/usr/bin/env python3
"""Build only the official CelebA RGB64 cache; run in the separate Arrow/Pillow tools venv."""
import argparse
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, __version__ as pillow_version

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def build(root, workers=8):
    root = Path(root)
    out = root / "derived/rgb64_v1"
    out.mkdir(parents=True, exist_ok=True)
    attr_path, split_path = root / "list_attr_celeba.txt", root / "list_eval_partition.txt"
    with attr_path.open() as f:
        count = int(next(f)); names = next(f).split()
        attrs = np.empty((count, len(names)), dtype=np.bool_)
        seen_attrs = set()
        for line in f:
            items = line.split(); idx = int(Path(items[0]).stem) - 1
            assert idx not in seen_attrs and 0 <= idx < count
            seen_attrs.add(idx)
            attrs[idx] = np.asarray(items[1:], dtype=np.int8) == 1
    assert len(seen_attrs) == count
    split = np.full(count, -1, dtype=np.int8)
    for line in split_path.read_text().splitlines():
        filename, value = line.split(); idx = int(Path(filename).stem) - 1
        assert split[idx] == -1
        split[idx] = int(value)
    assert all((split == i).sum() == n for i, n in [(0, 162770), (1, 19867), (2, 19962)])
    official = {attr_path.name: sha(attr_path), split_path.name: sha(split_path)}
    manifest_path = out / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous:
        assert previous["official_metadata"] == official
        assert previous["resize"] == "Pillow RGB BILINEAR 64x64, no crop"
    image_path, available_path = out / "images.npy", out / "available.npy"
    images = np.lib.format.open_memmap(image_path, mode="r+" if image_path.exists() else "w+", dtype=np.uint8, shape=(count, 3, 64, 64))
    available = np.lib.format.open_memmap(available_path, mode="r+" if available_path.exists() else "w+", dtype=np.bool_, shape=(count,))
    if not (out / "metadata.npz").exists():
        np.savez(out / "metadata.npz", image_id=np.arange(1, count + 1, dtype=np.int64),
                 split=split, Smiling=attrs[:, names.index("Smiling")].astype(np.int64),
                 Male=attrs[:, names.index("Male")].astype(np.int64))
    recorded = dict(previous["source_shards"]) if previous else {}
    manifest = {"version": 1, "official_metadata": official, "resize": "Pillow RGB BILINEAR 64x64, no crop",
                "image_dtype": "uint8", "shape": [count, 3, 64, 64], "row_index": "official image numeric id minus one",
                "source_shards": recorded, "complete": False, "pillow_version": pillow_version, "numpy_version": np.__version__}
    def save():
        images.flush(); available.flush()
        manifest["available_images"] = int(available.sum())
        manifest["complete"] = bool(available.all())
        if manifest["complete"] and "cache_files_sha256" not in manifest:
            manifest["cache_files_sha256"] = {p.name: sha(p) for p in [image_path, available_path, out / "metadata.npz"]}
        tmp = manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        tmp.replace(manifest_path)
    save()
    for path in sorted((root / "hf_flwrlabs/img_align+identity+attr").glob("*.parquet")):
        rel = str(path.relative_to(root))
        checksum = sha(path)
        if rel in recorded:
            assert recorded[rel]["sha256"] == checksum
            continue
        expected_split = {"train": 0, "valid": 1, "test": 2}[path.name.split("-")[0]]
        rows = 0
        def decode(item):
            idx = int(Path(item["path"]).stem) - 1
            with Image.open(io.BytesIO(item["bytes"])) as image:
                arr = np.asarray(image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.uint8)
            return idx, arr.transpose(2, 0, 1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=["image"] + names):
                values = batch.to_pydict()
                for i, item in enumerate(values["image"]):
                    idx = int(Path(item["path"]).stem) - 1
                    assert 0 <= idx < count and int(split[idx]) == expected_split, (path.name, item["path"], "official split mismatch")
                    assert all(bool(values[name][i]) == bool(attrs[idx, j]) for j, name in enumerate(names)), (item["path"], "official attributes mismatch")
                # Reprocessing an interrupted shard is safe; no row-position assumptions.
                for idx, arr in pool.map(decode, values["image"]):
                    images[idx] = arr
                    available[idx] = True
                    rows += 1
        recorded[rel] = {"sha256": checksum, "rows": rows}
        save()
        print(json.dumps({"shard": rel, "rows": rows, "available": int(available.sum()), "complete": manifest["complete"]}), flush=True)
    save()
    print(json.dumps({"cache": str(out), "complete": manifest["complete"], "available": int(available.sum())}), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    build(args.root, args.workers)
