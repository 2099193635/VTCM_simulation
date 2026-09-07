from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


COMPONENT_NAMES = ("fastener", "ballast", "subgrade")
CATEGORY_NAMES = ("normal", "counterfactual_control", "defect")


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": float("nan"), "p05": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "min": float(np.min(finite)),
        "p05": float(np.percentile(finite, 5)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def audit_dataset(path: str | Path) -> dict[str, Any]:
    report: dict[str, Any] = {"path": str(path), "splits": {}}
    split_seeds: dict[str, set[int]] = {}
    with h5py.File(path, "r") as h5:
        report["schema_version"] = str(h5.attrs.get("schema_version", ""))
        report["attrs"] = {
            "ds_m": float(h5.attrs["ds_m"]),
            "window_m": float(h5.attrs["window_m"]),
            "stride_m": float(h5.attrs["stride_m"]),
        }
        for split in ("train", "val", "test"):
            group = h5[split]
            seeds = set(np.asarray(group["random_seed"], dtype=np.int64).tolist())
            split_seeds[split] = seeds
            speed = np.asarray(group["context"][:, 0], dtype=np.float64)
            category = np.asarray(group["window_category"], dtype=np.int64)
            eta = np.asarray(group["component_eta"], dtype=np.float32)
            gap = np.asarray(group["void_gap_m"], dtype=np.float32)
            geometry_mm = np.asarray(group["geometry_abs"], dtype=np.float32) * 1000.0
            split_report: dict[str, Any] = {
                "sample_count": int(group["obs"].shape[0]),
                "unique_seed_count": len(seeds),
                "speed_counts": {
                    str(int(value)): int(np.count_nonzero(np.isclose(speed, value)))
                    for value in sorted(set(speed.tolist()))
                },
                "window_category_counts": {
                    CATEGORY_NAMES[index]: int(np.count_nonzero(category == index))
                    for index in range(len(CATEGORY_NAMES))
                },
                "geometry_mm": _stats(geometry_mm),
                "void_gap_mm": _stats(gap * 1000.0),
                "component_eta": {},
            }
            for index, name in enumerate(COMPONENT_NAMES):
                split_report["component_eta"][name] = _stats(eta[:, :, index, :])
            report["splits"][split] = split_report

    overlaps = {}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = sorted(split_seeds[first] & split_seeds[second])
        overlaps[f"{first}_{second}"] = shared
    report["seed_overlap"] = overlaps
    report["seed_leakage"] = any(bool(shared) for shared in overlaps.values())
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a component inverse HDF5 dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_dataset(args.dataset)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    if report["seed_leakage"]:
        raise SystemExit("Seed leakage detected between dataset splits.")


if __name__ == "__main__":
    main()