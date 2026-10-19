#!/usr/bin/env python3
"""Summarize fixed-protocol terminal metrics; never select runs or rounds."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


METRICS = ("accuracy", "aeod", "aspd")
GROUP_FIELDS = ("dataset", "distribution", "method", "attack")
FIELDS = (*GROUP_FIELDS, "config_sha256", "config_json", "metric", "n_total",
          "n_defined", "mean", "sample_std")


def number(value, context):
    if type(value) not in (int, float):
        raise ValueError(f"{context}: expected a numeric metric")
    try:
        return float(value)
    except OverflowError as exc:
        raise ValueError(f"{context}: metric cannot be represented as a float") from exc


def summarize_records(records, manifest):
    """Validate every record before returning per-configuration, per-metric rows.

    Metrics retain their input units. Nonfinite values count toward n_total only;
    sample_std uses ddof=1 and is undefined when fewer than two values are finite.
    """
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    seeds, rounds = manifest.get("seeds"), manifest.get("rounds")
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("manifest.seeds must be a nonempty list of unique integers")
    if type(rounds) is not int or rounds < 1:
        raise ValueError("manifest.rounds must be a positive integer")
    expected_seeds = set(seeds)
    groups = {}
    for index, record in enumerate(records, 1):
        context = f"record {index}"
        if not isinstance(record, dict):
            raise ValueError(f"{context}: expected an object")
        config, seed = record.get("config"), record.get("seed")
        if (not isinstance(config, dict) or type(seed) is not int
                or type(config.get("seed")) is not int or config["seed"] != seed):
            raise ValueError(f"{context}: record.seed and config.seed must be equal integers")
        if seed not in expected_seeds:
            raise ValueError(f"{context}: seed {seed} is outside manifest.seeds")
        if (type(record.get("rounds")) is not int or record["rounds"] != rounds
                or type(config.get("rounds")) is not int or config["rounds"] != rounds):
            raise ValueError(f"{context}: record/config rounds must match manifest.rounds")
        identity = tuple(record.get(field) for field in GROUP_FIELDS)
        if any(not isinstance(value, str) or not value for value in identity):
            raise ValueError(f"{context}: group identity fields must be nonempty strings")
        points = record.get("trajectory_metrics")
        if not isinstance(points, list) or len(points) != rounds:
            raise ValueError(f"{context}: trajectory must contain every round 1..{rounds}")
        for expected_round, point in enumerate(points, 1):
            if (not isinstance(point, dict) or type(point.get("round")) is not int
                    or point["round"] != expected_round):
                raise ValueError(f"{context}: trajectory rounds must be exactly 1..{rounds} in order")
            if not isinstance(point.get("metrics"), dict):
                raise ValueError(f"{context}: round {expected_round} metrics are missing")
            for metric in METRICS:
                number(point["metrics"].get(metric), f"{context}, round {expected_round}, {metric}")
        if not isinstance(record.get("metrics"), dict):
            raise ValueError(f"{context}: terminal metrics are missing")
        terminal = {}
        for metric in METRICS:
            value = number(record["metrics"].get(metric), f"{context}, {metric}")
            final = number(points[-1]["metrics"][metric], f"{context}, final {metric}")
            if not (value == final or (math.isnan(value) and math.isnan(final))
                    or math.isclose(value, final, rel_tol=0, abs_tol=1e-12)):
                raise ValueError(f"{context}: {metric} differs from the final trajectory point")
            terminal[metric] = value
        shared_config = {key: value for key, value in config.items()
                         if key not in {"seed", "experiment_tag", "experiment_suite"}}
        config_json = json.dumps(shared_config, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=True, allow_nan=False)
        key = (*identity, config_json)
        by_seed = groups.setdefault(key, {})
        if seed in by_seed:
            raise ValueError(f"{context}: duplicate seed {seed} in the same configuration group")
        by_seed[seed] = terminal
    if not groups:
        raise ValueError("input contains no records")
    output = []
    for key, by_seed in groups.items():
        if set(by_seed) != expected_seeds:
            raise ValueError(f"group {key[:4]}: seed coverage differs from manifest.seeds; "
                             f"expected {seeds}, received {list(by_seed)}")
        config_json = key[-1]
        group = dict(zip(GROUP_FIELDS, key[:4]), config_json=config_json,
                     config_sha256=hashlib.sha256(config_json.encode("utf-8")).hexdigest())
        for metric in METRICS:
            values = [row[metric] for row in by_seed.values() if math.isfinite(row[metric])]
            output.append(dict(group, metric=metric, n_total=len(by_seed), n_defined=len(values),
                               mean=statistics.mean(values) if values else None,
                               sample_std=statistics.stdev(values) if len(values) > 1 else None))
    return output


def read_records(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="JSON object with fixed seeds and rounds")
    parser.add_argument("--output", type=Path, required=True,
                        help="New CSV path; existing files are never overwritten")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        rows = summarize_records(read_records(args.input), manifest)
        with args.output.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
