from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import h5py
import numpy as np

from pipeline.build_joint_inverse_dataset import (
    GEOM_CHANNELS,
    MASK_CHANNELS,
    _boundary_mask,
    _build_obs,
    _find_npz_files,
    _geom_labels,
    _interp_to_grid,
    _regular_grid,
    _run_context,
    _source_distance,
)


SCHEMA_VERSION = "1.0"
COMPONENT_NAMES = ("fastener", "ballast", "subgrade")
SIDE_NAMES = ("L", "R")
COMPONENT_KEYS = (
    ("Fastener_eta_k_L_ref", "Fastener_eta_k_R_ref"),
    ("Ballast_eta_k_L_ref", "Ballast_eta_k_R_ref"),
    ("Subgrade_eta_k_L_ref", "Subgrade_eta_k_R_ref"),
)
VOID_GAP_KEYS = ("Sleeper_void_gap_L_m_ref", "Sleeper_void_gap_R_m_ref")
VOID_ACTIVE_KEYS = ("Sleeper_void_active_L_ref", "Sleeper_void_active_R_ref")
K_FASTENER_0 = 60.0e6
K_BALLAST_0 = 240.0e6
K_SUBGRADE_0 = 65.0e6


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _node_spacing(data: np.lib.npyio.NpzFile) -> float:
    if "Structure_window_meta" in data.files:
        meta = np.asarray(data["Structure_window_meta"], dtype=float).reshape(-1)
        if meta.size > 6 and meta[6] > 0:
            return float(meta[6])
    return 0.6


def _record_sides(side: str) -> tuple[bool, bool]:
    words = {item.strip().lower() for item in str(side).replace("|", ",").split(",")}
    return (
        bool(words & {"left", "l", "both", "all"}),
        bool(words & {"right", "r", "both", "all"}),
    )


def _component_labels_from_summary(
    npz_path: Path,
    abs_s: np.ndarray,
    data: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eta = np.ones((abs_s.size, len(COMPONENT_NAMES), len(SIDE_NAMES)), dtype=np.float32)
    gap = np.zeros((abs_s.size, len(SIDE_NAMES)), dtype=np.float32)
    active = np.zeros((abs_s.size, len(SIDE_NAMES)), dtype=np.float32)
    spacing_m = _node_spacing(data)
    records = _load_json(npz_path.with_name("structure_defects_summary.json")).get("records", [])

    part_index = {
        "fastener_failure": 0,
        "ballast_condition": 1,
        "subgrade_condition": 2,
    }
    for record in records:
        kind = str(record.get("kind", "")).lower()
        start_m = float(record.get("abs_start_m", np.nan))
        count = max(1, int(record.get("count", 1)))
        if not np.isfinite(start_m):
            continue
        end_m = start_m + count * spacing_m
        tol = max(1e-9, 1e-9 * spacing_m)
        region = (abs_s >= start_m - tol) & (abs_s < end_m - tol)
        left, right = _record_sides(record.get("side", "both"))
        side_indices = ([0] if left else []) + ([1] if right else [])

        if kind == "sleeper_void":
            value = max(0.0, float(record.get("delta_gap_m", 0.0)))
            for side_idx in side_indices:
                gap[region, side_idx] = np.maximum(gap[region, side_idx], value)
                active[region, side_idx] = 1.0
        elif kind in part_index:
            factor = float(record.get("stiffness_factor", 1.0))
            for side_idx in side_indices:
                eta[region, part_index[kind], side_idx] *= factor
    return eta, gap, active


def _component_labels(
    npz_path: Path,
    data: np.lib.npyio.NpzFile,
    abs_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if all(key in data.files for pair in COMPONENT_KEYS for key in pair):
        eta = np.stack(
            [
                np.column_stack(
                    [
                        np.asarray(data[left], dtype=np.float32).reshape(-1),
                        np.asarray(data[right], dtype=np.float32).reshape(-1),
                    ]
                )
                for left, right in COMPONENT_KEYS
            ],
            axis=1,
        )
    else:
        eta, _, _ = _component_labels_from_summary(npz_path, abs_s, data)

    if all(key in data.files for key in VOID_GAP_KEYS):
        gap = np.column_stack(
            [np.asarray(data[key], dtype=np.float32).reshape(-1) for key in VOID_GAP_KEYS]
        )
    else:
        _, gap, _ = _component_labels_from_summary(npz_path, abs_s, data)

    if all(key in data.files for key in VOID_ACTIVE_KEYS):
        active = np.column_stack(
            [np.asarray(data[key], dtype=np.float32).reshape(-1) for key in VOID_ACTIVE_KEYS]
        )
    else:
        _, _, active = _component_labels_from_summary(npz_path, abs_s, data)
    return eta, gap, active


def _interp_nearest(source_s: np.ndarray, values: np.ndarray, target_s: np.ndarray) -> np.ndarray:
    source_s = np.asarray(source_s, dtype=np.float64).reshape(-1)
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    n = min(source_s.size, values.shape[0])
    source_s = source_s[:n]
    values = values[:n]
    order = np.argsort(source_s)
    source_s = source_s[order]
    values = values[order]
    source_s, unique_idx = np.unique(source_s, return_index=True)
    values = values[unique_idx]
    idx = np.searchsorted(source_s, target_s, side="left")
    idx = np.clip(idx, 0, source_s.size - 1)
    prev = np.maximum(idx - 1, 0)
    choose_prev = np.abs(target_s - source_s[prev]) <= np.abs(source_s[idx] - target_s)
    idx = np.where(choose_prev, prev, idx)
    return values[idx].astype(np.float32)


def _series_stiffness(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    out = np.zeros(np.broadcast_shapes(first.shape, second.shape), dtype=np.float64)
    first_b, second_b = np.broadcast_arrays(first, second)
    valid = (first_b > 1e-12) & (second_b > 1e-12)
    out[valid] = 1.0 / (1.0 / first_b[valid] + 1.0 / second_b[valid])
    return out


def support_eta_from_components(
    component_eta: np.ndarray,
    void_active: np.ndarray,
) -> np.ndarray:
    component_eta = np.asarray(component_eta, dtype=np.float64)
    void_active = np.asarray(void_active, dtype=bool)
    kf = component_eta[:, 0, :] * K_FASTENER_0
    kb = component_eta[:, 1, :] * K_BALLAST_0
    ks = component_eta[:, 2, :] * K_SUBGRADE_0
    kb = np.where(void_active, 0.0, kb)
    k_under = _series_stiffness(kb, ks)
    k_support = _series_stiffness(kf, k_under)
    k_under_0 = 1.0 / (1.0 / K_BALLAST_0 + 1.0 / K_SUBGRADE_0)
    k_support_0 = 1.0 / (1.0 / K_FASTENER_0 + 1.0 / k_under_0)
    return (k_support / k_support_0).astype(np.float32)


def _seed_split_map(
    seeds: list[int],
    catalog_path: str | None,
    split_seed: int,
) -> dict[int, str]:
    if catalog_path:
        catalog = _load_json(Path(catalog_path))
        mapping = catalog.get("seed_splits", {})
        parsed = {int(key): str(value) for key, value in mapping.items()}
        missing = sorted(set(seeds) - set(parsed))
        if missing:
            raise ValueError(f"Catalog is missing split assignments for seeds: {missing[:5]}")
        return parsed

    unique = np.asarray(sorted(set(seeds)), dtype=np.int64)
    rng = np.random.default_rng(split_seed)
    rng.shuffle(unique)
    n = unique.size
    n_train = int(round(n * (68.0 / 96.0)))
    n_val = int(round(n * (14.0 / 96.0)))
    n_train = min(max(n_train, 1), max(n - 2, 1)) if n >= 3 else max(n - 1, 1)
    n_val = min(max(n_val, 1), max(n - n_train - 1, 0)) if n >= 3 else 0
    mapping: dict[int, str] = {}
    for idx, seed in enumerate(unique):
        mapping[int(seed)] = "train" if idx < n_train else ("val" if idx < n_train + n_val else "test")
    return mapping


def _contiguous_intervals(distance: np.ndarray, active: np.ndarray, ds_m: float) -> list[tuple[float, float]]:
    mask = np.asarray(active, dtype=bool)
    if not np.any(mask):
        return []
    edges = np.diff(np.concatenate([[False], mask, [False]]).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(float(distance[start]), float(distance[end - 1] + ds_m)) for start, end in zip(starts, ends)]


def _window_overlaps(start: float, end: float, intervals: list[tuple[float, float]]) -> bool:
    return any(start < interval_end and end > interval_start for interval_start, interval_end in intervals)


def build_windows(args: argparse.Namespace) -> dict[str, np.ndarray]:
    npz_files = _find_npz_files([Path(item) for item in args.result_roots])
    if not npz_files:
        raise FileNotFoundError("No simulation_result.npz files found.")

    runs: list[dict[str, Any]] = []
    seed_intervals: dict[int, list[tuple[float, float]]] = {}
    obs_names: list[str] | None = None

    for run_index, npz_path in enumerate(npz_files):
        with np.load(npz_path, allow_pickle=True) as data:
            if "A" not in data.files:
                continue
            rel_s, abs_s = _source_distance(data)
            grid_rel = _regular_grid(rel_s, args.ds_m)
            if grid_rel.size < 2:
                continue
            grid_abs = _interp_to_grid(rel_s, abs_s, grid_rel)[:, 0]
            obs_raw, names = _build_obs(data, include_force=False)
            obs_names = obs_names or names
            geom_raw = _geom_labels(data)
            eta_raw, gap_raw, void_raw = _component_labels(npz_path, data, abs_s)
            boundary_raw = _boundary_mask(data, min(rel_s.size, data["A"].shape[0]))

            obs = _interp_to_grid(rel_s, obs_raw, grid_rel)
            geom = _interp_to_grid(rel_s, geom_raw, grid_rel)
            eta = np.clip(
                _interp_to_grid(rel_s, eta_raw.reshape(eta_raw.shape[0], -1), grid_rel),
                0.0,
                args.max_eta,
            ).reshape(grid_rel.size, len(COMPONENT_NAMES), len(SIDE_NAMES))
            gap = _interp_nearest(rel_s, gap_raw, grid_rel)
            void_active = _interp_nearest(rel_s, void_raw, grid_rel) > 0.5
            boundary = _interp_nearest(rel_s, boundary_raw, grid_rel)[:, 0]
            support_eta = support_eta_from_components(eta, void_active)

            context, run_meta = _run_context(npz_path, run_index, args.ds_m)
            seed = int(run_meta["random_seed"])
            lead_distance_m = max(0.0, float(run_meta.get("lead_time_s", 0.0)) * (context[0] / 3.6))
            transition = grid_rel >= lead_distance_m
            finite = (
                np.isfinite(obs).all(axis=1)
                & np.isfinite(geom).all(axis=1)
                & np.isfinite(eta).all(axis=(1, 2))
                & np.isfinite(gap).all(axis=1)
                & np.isfinite(support_eta).all(axis=1)
            )
            valid = finite & transition & (boundary > 0.5)
            component_active = np.any(np.abs(eta - 1.0) > 1e-6, axis=(1, 2))
            defect_active = component_active | np.any(void_active, axis=1)
            intervals = _contiguous_intervals(grid_rel, defect_active, args.ds_m)
            seed_intervals.setdefault(seed, []).extend(intervals)

            runs.append(
                {
                    "npz_path": str(npz_path),
                    "run_name": str(run_meta["run_name"]),
                    "seed": seed,
                    "context": context,
                    "rel": grid_rel,
                    "abs": grid_abs,
                    "obs": obs,
                    "geom": geom,
                    "eta": eta,
                    "gap": gap,
                    "void_active": void_active.astype(np.float32),
                    "support_eta": support_eta,
                    "valid": valid.astype(np.float32),
                    "transition": transition.astype(np.float32),
                    "boundary": (boundary > 0.5).astype(np.float32),
                    "defect_active": defect_active,
                }
            )

    split_map = _seed_split_map([run["seed"] for run in runs], args.case_catalog, args.split_seed)
    window_points = max(2, int(round(args.window_m / args.ds_m)))
    stride_points = max(1, int(round(args.stride_m / args.ds_m)))
    rows: dict[str, list[Any]] = {
        "obs": [],
        "geometry_abs": [],
        "component_eta": [],
        "component_log_eta": [],
        "void_gap_m": [],
        "void_active": [],
        "support_eta": [],
        "valid_mask": [],
        "context": [],
        "run_id": [],
        "random_seed": [],
        "window_start_rel_m": [],
        "window_start_abs_m": [],
        "defect_coverage": [],
        "window_category": [],
        "split": [],
    }

    for run in runs:
        size = run["rel"].size
        for start in range(0, size - window_points + 1, stride_points):
            end = start + window_points
            if float(np.mean(run["valid"][start:end])) < args.min_valid_fraction:
                continue
            coverage = float(np.mean(run["defect_active"][start:end]))
            if coverage > 0.0:
                category = 2
            elif _window_overlaps(
                float(run["rel"][start]),
                float(run["rel"][end - 1] + args.ds_m),
                seed_intervals.get(run["seed"], []),
            ):
                category = 1
            else:
                category = 0
            mask = np.column_stack(
                [run["valid"][start:end], run["transition"][start:end], run["boundary"][start:end]]
            ).astype(np.float32)
            rows["obs"].append(run["obs"][start:end])
            rows["geometry_abs"].append(run["geom"][start:end])
            rows["component_eta"].append(run["eta"][start:end])
            rows["component_log_eta"].append(
                np.log(np.clip(run["eta"][start:end], args.min_eta, args.max_eta))
            )
            rows["void_gap_m"].append(run["gap"][start:end])
            rows["void_active"].append(run["void_active"][start:end])
            rows["support_eta"].append(run["support_eta"][start:end])
            rows["valid_mask"].append(mask)
            rows["context"].append(run["context"])
            rows["run_id"].append(run["run_name"])
            rows["random_seed"].append(run["seed"])
            rows["window_start_rel_m"].append(float(run["rel"][start]))
            rows["window_start_abs_m"].append(float(run["abs"][start]))
            rows["defect_coverage"].append(coverage)
            rows["window_category"].append(category)
            rows["split"].append(split_map[run["seed"]])

    if not rows["obs"]:
        raise RuntimeError("No valid windows were generated.")

    arrays: dict[str, np.ndarray] = {}
    for key, values in rows.items():
        if key in {"run_id", "split"}:
            arrays[key] = np.asarray(values, dtype=object)
        elif key in {"random_seed", "window_category"}:
            arrays[key] = np.asarray(values, dtype=np.int64)
        else:
            arrays[key] = np.asarray(values, dtype=np.float32)

    arrays["sample_weight"] = np.ones(arrays["window_category"].shape, dtype=np.float32)
    train = arrays["split"] == "train"
    targets = {2: 0.50, 1: 0.30, 0: 0.20}
    for category, target in targets.items():
        selected = train & (arrays["window_category"] == category)
        count = int(np.count_nonzero(selected))
        if count:
            arrays["sample_weight"][selected] = np.float32(target / count)
    if np.any(train):
        mean_weight = float(np.mean(arrays["sample_weight"][train]))
        if mean_weight > 0:
            arrays["sample_weight"][train] /= mean_weight

    arrays["obs_channel_names"] = np.asarray(obs_names or [], dtype=object)
    return arrays


def _write_dataset(group: h5py.Group, key: str, values: np.ndarray) -> None:
    if values.dtype == object:
        dtype = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(key, data=values.astype(dtype), dtype=dtype)
        return
    if values.shape[0] == 0:
        group.create_dataset(key, data=values)
        return
    chunks = True if values.ndim == 1 else (min(64, values.shape[0]), *values.shape[1:])
    group.create_dataset(
        key,
        data=values,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        chunks=chunks,
    )


def write_hdf5(path: str | Path, arrays: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_keys = [
        "obs",
        "geometry_abs",
        "component_eta",
        "component_log_eta",
        "void_gap_m",
        "void_active",
        "support_eta",
        "valid_mask",
        "context",
        "run_id",
        "random_seed",
        "window_start_rel_m",
        "window_start_abs_m",
        "defect_coverage",
        "window_category",
        "sample_weight",
    ]
    with h5py.File(output, "w") as h5:
        h5.attrs["schema_version"] = SCHEMA_VERSION
        h5.attrs["ds_m"] = float(args.ds_m)
        h5.attrs["window_m"] = float(args.window_m)
        h5.attrs["stride_m"] = float(args.stride_m)
        h5.attrs["geometry_units"] = "m"
        h5.attrs["component_eta_units"] = "ratio"
        h5.attrs["void_gap_units"] = "m"
        h5.attrs["component_names"] = json.dumps(COMPONENT_NAMES)
        h5.attrs["side_names"] = json.dumps(SIDE_NAMES)
        h5.attrs["obs_channel_names"] = json.dumps(arrays["obs_channel_names"].tolist())
        h5.attrs["geometry_channel_names"] = json.dumps(GEOM_CHANNELS)
        h5.attrs["mask_channel_names"] = json.dumps(MASK_CHANNELS)
        h5.attrs["context_names"] = json.dumps(["speed_kmh", "random_seed", "ds_m", "run_index"])
        h5.attrs["window_category_names"] = json.dumps(["normal", "counterfactual_control", "defect"])
        h5.attrs["base_stiffness_npm"] = json.dumps(
            {"fastener": K_FASTENER_0, "ballast": K_BALLAST_0, "subgrade": K_SUBGRADE_0}
        )
        h5.attrs["eta_defect_thresholds"] = json.dumps({"lower": 0.8, "upper": 1.25})
        h5.attrs["void_gap_threshold_m"] = 0.0002
        h5.attrs["void_severity_reference_m"] = 0.002
        for split in ("train", "val", "test"):
            group = h5.create_group(split)
            selected = arrays["split"] == split
            group.attrs["sample_count"] = int(np.count_nonzero(selected))
            for key in sample_keys:
                _write_dataset(group, key, arrays[key][selected])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an HDF5 dataset for component-wise stiffness inversion.")
    parser.add_argument("--result-roots", nargs="+", required=True)
    parser.add_argument("--output", default="results/joint_inverse_dataset/component_inverse_dataset.h5")
    parser.add_argument("--case-catalog", default=None)
    parser.add_argument("--ds-m", type=float, default=0.25)
    parser.add_argument("--window-m", type=float, default=128.0)
    parser.add_argument("--stride-m", type=float, default=16.0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.95)
    parser.add_argument("--min-eta", type=float, default=0.02)
    parser.add_argument("--max-eta", type=float, default=50.0)
    parser.add_argument("--split-seed", type=int, default=20260715)
    parser.add_argument("--format", choices=("hdf5", "npz"), default="hdf5")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    arrays = build_windows(args)
    if args.format == "hdf5":
        write_hdf5(args.output, arrays, args)
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **arrays)
    counts = {split: int(np.count_nonzero(arrays["split"] == split)) for split in ("train", "val", "test")}
    print(f"Saved component inverse dataset: {args.output}")
    print(f"  windows: {arrays['obs'].shape[0]}")
    print(f"  split counts: {counts}")
    print(f"  obs shape: {arrays['obs'].shape}")
    print(f"  geometry shape: {arrays['geometry_abs'].shape}")
    print(f"  component eta shape: {arrays['component_eta'].shape}")


if __name__ == "__main__":
    main()