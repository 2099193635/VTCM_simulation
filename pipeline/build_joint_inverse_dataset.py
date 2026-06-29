from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except Exception:
    yaml = None


OBS_CHANNELS = [
    ("carbody_ay_mps2", "A", 0),
    ("carbody_az_mps2", "A", 1),
    ("bogie1_ay_mps2", "A", 5),
    ("bogie1_az_mps2", "A", 6),
    ("bogie2_ay_mps2", "A", 10),
    ("bogie2_az_mps2", "A", 11),
    ("wheelset1_az_mps2", "A", 16),
    ("wheelset2_az_mps2", "A", 21),
    ("wheelset3_az_mps2", "A", 26),
    ("wheelset4_az_mps2", "A", 31),
]

GEOM_CHANNELS = [
    "Irre_bz_L_ref",
    "Irre_bz_R_ref",
    "Irre_by_L_ref",
    "Irre_by_R_ref",
]

STIFFNESS_CHANNELS = [
    "Stiffness_eta_k_L_ref",
    "Stiffness_eta_k_R_ref",
]

MASK_CHANNELS = [
    "valid_mask",
    "transition_mask",
    "boundary_mask",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists() or yaml is None:
        return {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _interp_to_grid(source_s: np.ndarray, values: np.ndarray, target_s: np.ndarray) -> np.ndarray:
    source_s = np.asarray(source_s, dtype=np.float64).reshape(-1)
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    n = min(source_s.size, values.shape[0])
    source_s = source_s[:n]
    values = values[:n]
    finite = np.isfinite(source_s)
    if np.count_nonzero(finite) < 2:
        raise ValueError("source mileage has fewer than two finite points")
    source_s = source_s[finite]
    values = values[finite]
    order = np.argsort(source_s)
    source_s = source_s[order]
    values = values[order]
    source_s, unique_idx = np.unique(source_s, return_index=True)
    values = values[unique_idx]

    out = np.empty((target_s.size, values.shape[1]), dtype=np.float32)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(target_s, source_s, values[:, col]).astype(np.float32)
    return out


def _source_distance(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    if "Track_rel_mileage_m" in data.files:
        rel = np.asarray(data["Track_rel_mileage_m"], dtype=np.float64).reshape(-1)
    elif "Irre_distance_m" in data.files:
        rel = np.asarray(data["Irre_distance_m"], dtype=np.float64).reshape(-1)
    else:
        dt = float(data["dt"])
        rel = np.arange(data["A"].shape[0], dtype=np.float64) * dt

    if "Track_abs_mileage_m" in data.files:
        abs_s = np.asarray(data["Track_abs_mileage_m"], dtype=np.float64).reshape(-1)
    else:
        abs_s = rel.copy()
    n = min(rel.size, abs_s.size, data["A"].shape[0])
    return rel[:n], abs_s[:n]


def _regular_grid(rel_s: np.ndarray, ds_m: float) -> np.ndarray:
    start = np.ceil(float(np.nanmin(rel_s)) / ds_m) * ds_m
    stop = np.floor(float(np.nanmax(rel_s)) / ds_m) * ds_m
    return np.arange(start, stop + 0.5 * ds_m, ds_m, dtype=np.float64)


def _build_obs(data: np.lib.npyio.NpzFile, include_force: bool) -> tuple[np.ndarray, list[str]]:
    a = np.asarray(data["A"], dtype=np.float32)
    parts = []
    names = []
    for name, key, idx in OBS_CHANNELS:
        if key in data.files and data[key].ndim == 2 and data[key].shape[1] > idx:
            parts.append(np.asarray(data[key][:, idx], dtype=np.float32).reshape(-1, 1))
        else:
            parts.append(np.zeros((a.shape[0], 1), dtype=np.float32))
        names.append(name)

    if include_force and "TotalVerticalForce" in data.files:
        total_v = np.asarray(data["TotalVerticalForce"], dtype=np.float32)
        if "TotalVerticalForce_Point2" in data.files:
            p2 = np.asarray(data["TotalVerticalForce_Point2"], dtype=np.float32)
            if p2.shape == total_v.shape:
                total_v = total_v + p2
        if total_v.ndim == 2:
            parts.append(total_v)
            names.extend([f"wheelrail_fz_{i + 1}_n" for i in range(total_v.shape[1])])

    return np.concatenate(parts, axis=1), names


def _geom_labels(data: np.lib.npyio.NpzFile) -> np.ndarray:
    n = data["A"].shape[0]
    parts = []
    for key in GEOM_CHANNELS:
        if key in data.files:
            parts.append(np.asarray(data[key], dtype=np.float32).reshape(-1, 1))
        else:
            parts.append(np.zeros((n, 1), dtype=np.float32))
    return np.concatenate(parts, axis=1)


def _stiffness_from_summary(npz_path: Path, abs_s: np.ndarray, data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    eta = np.ones((abs_s.size, 2), dtype=np.float32)
    mask = np.zeros((abs_s.size, 2), dtype=np.float32)
    summary_path = npz_path.with_name("structure_defects_summary.json")
    summary = _load_json(summary_path)
    records = summary.get("records", []) if isinstance(summary, dict) else []
    if "Structure_window_meta" in data.files:
        meta = np.asarray(data["Structure_window_meta"], dtype=float).reshape(-1)
        spacing_m = float(meta[6]) if meta.size > 6 and meta[6] > 0 else 0.6
    else:
        spacing_m = 0.6

    for record in records:
        if str(record.get("kind", "")).lower() != "ballast_condition":
            continue
        start_m = float(record.get("abs_start_m", np.nan))
        count = int(record.get("count", 1))
        factor = float(record.get("stiffness_factor", 1.0))
        side = str(record.get("side", "both")).lower()
        if not np.isfinite(start_m):
            continue
        end_m = start_m + count * spacing_m
        tol = max(1e-9, 1e-9 * spacing_m)
        active = (abs_s >= start_m - tol) & (abs_s < end_m - tol)
        if side in ("left", "l", "both", "all"):
            eta[active, 0] *= factor
            mask[active, 0] = 1.0
        if side in ("right", "r", "both", "all"):
            eta[active, 1] *= factor
            mask[active, 1] = 1.0
    return eta, mask


def _stiffness_labels(npz_path: Path, data: np.lib.npyio.NpzFile, abs_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if all(key in data.files for key in STIFFNESS_CHANNELS):
        eta = np.column_stack([
            np.asarray(data["Stiffness_eta_k_L_ref"], dtype=np.float32).reshape(-1),
            np.asarray(data["Stiffness_eta_k_R_ref"], dtype=np.float32).reshape(-1),
        ])
        if "Stiffness_irregularity_mask_L" in data.files and "Stiffness_irregularity_mask_R" in data.files:
            mask = np.column_stack([
                np.asarray(data["Stiffness_irregularity_mask_L"], dtype=np.float32).reshape(-1),
                np.asarray(data["Stiffness_irregularity_mask_R"], dtype=np.float32).reshape(-1),
            ])
        else:
            mask = (np.abs(eta - 1.0) > 1e-6).astype(np.float32)
        return eta, mask
    return _stiffness_from_summary(npz_path, abs_s, data)


def _boundary_mask(data: np.lib.npyio.NpzFile, n: int) -> np.ndarray:
    if "Structure_boundary_weight" not in data.files:
        return np.ones(n, dtype=np.float32)
    bw = np.asarray(data["Structure_boundary_weight"], dtype=np.float32)
    if bw.ndim == 2 and bw.shape[1] > 3:
        return (bw[:n, 3] >= 0.999).astype(np.float32)
    if bw.ndim == 1:
        return (bw[:n] >= 0.999).astype(np.float32)
    return np.ones(n, dtype=np.float32)


def _run_context(npz_path: Path, run_index: int, ds_m: float) -> tuple[np.ndarray, dict[str, Any]]:
    meta = _load_yaml(npz_path.with_name("run_meta.yaml"))
    args = _load_json(npz_path.with_name("argparse_params.json"))
    speed_kmh = float(args.get("vx_set", meta.get("vx_set", 0.0)) or 0.0)
    random_seed = float(args.get("random_seed", -1) or -1)
    lead_time_s = float(meta.get("irr_lead_time_s", args.get("irr_lead_time", 0.0)) or 0.0)
    context = np.array([speed_kmh, random_seed, ds_m, float(run_index)], dtype=np.float32)
    meta_out = {
        "npz_path": str(npz_path),
        "run_name": meta.get("run_name", npz_path.parent.parent.name),
        "run_note": meta.get("run_note", ""),
        "vehicle_type": meta.get("vehicle_type", ""),
        "irr_type": meta.get("irr_type", ""),
        "speed_kmh": speed_kmh,
        "random_seed": int(random_seed),
        "lead_time_s": lead_time_s,
    }
    return context, meta_out


def _find_npz_files(result_roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in result_roots:
        if root.is_file() and root.suffix.lower() == ".npz":
            files.append(root)
        elif root.exists():
            files.extend(root.glob("*/files/simulation_result.npz"))
    return sorted(set(p.resolve() for p in files))


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    npz_files = _find_npz_files([Path(p) for p in args.result_roots])
    if not npz_files:
        raise FileNotFoundError("No simulation_result.npz files found.")

    samples_obs = []
    samples_geom = []
    samples_eta = []
    samples_log_eta = []
    samples_mask = []
    samples_context = []
    samples_meta = []
    obs_names: list[str] | None = None

    window_points = max(2, int(round(args.window_m / args.ds_m)))
    stride_points = max(1, int(round(args.stride_m / args.ds_m)))

    for run_index, npz_path in enumerate(npz_files):
        with np.load(npz_path, allow_pickle=True) as data:
            if "A" not in data.files:
                continue
            rel_s, abs_s = _source_distance(data)
            grid_rel = _regular_grid(rel_s, args.ds_m)
            if grid_rel.size < window_points:
                continue
            grid_abs = _interp_to_grid(rel_s, abs_s, grid_rel)[:, 0]

            obs_raw, names = _build_obs(data, include_force=args.include_force)
            obs_names = obs_names or names
            geom_raw = _geom_labels(data)
            eta_raw, stiffness_mask_raw = _stiffness_labels(npz_path, data, abs_s)
            boundary_raw = _boundary_mask(data, min(rel_s.size, data["A"].shape[0]))

            obs = _interp_to_grid(rel_s, obs_raw, grid_rel)
            geom = _interp_to_grid(rel_s, geom_raw, grid_rel)
            eta = np.clip(_interp_to_grid(rel_s, eta_raw, grid_rel), args.min_eta, args.max_eta)
            stiffness_mask = (_interp_to_grid(rel_s, stiffness_mask_raw, grid_rel) > 0.5).astype(np.float32)
            boundary = _interp_to_grid(rel_s, boundary_raw, grid_rel)[:, 0]

            context, run_meta = _run_context(npz_path, run_index, args.ds_m)
            lead_distance_m = max(0.0, float(run_meta.get("lead_time_s", 0.0)) * (context[0] / 3.6))
            transition = (grid_rel >= lead_distance_m).astype(np.float32)
            finite = np.isfinite(obs).all(axis=1) & np.isfinite(geom).all(axis=1) & np.isfinite(eta).all(axis=1)
            valid = (finite.astype(np.float32) * transition * boundary).astype(np.float32)
            mask = np.column_stack([valid, transition, boundary]).astype(np.float32)

            for start in range(0, grid_rel.size - window_points + 1, stride_points):
                end = start + window_points
                if float(np.mean(valid[start:end])) < args.min_valid_fraction:
                    continue
                samples_obs.append(obs[start:end])
                samples_geom.append(geom[start:end])
                samples_eta.append(eta[start:end])
                samples_log_eta.append(np.log(eta[start:end]))
                samples_mask.append(mask[start:end])
                samples_context.append(context)
                sample_meta = dict(run_meta)
                sample_meta.update({
                    "window_start_rel_m": float(grid_rel[start]),
                    "window_end_rel_m": float(grid_rel[end - 1]),
                    "window_start_abs_m": float(grid_abs[start]),
                    "window_end_abs_m": float(grid_abs[end - 1]),
                    "has_stiffness_irregularity": bool(np.any(stiffness_mask[start:end] > 0.5)),
                })
                samples_meta.append(json.dumps(sample_meta, ensure_ascii=False))

    if not samples_obs:
        raise RuntimeError("No valid windows were generated. Try smaller window_m or lower min_valid_fraction.")

    return {
        "obs": np.stack(samples_obs).astype(np.float32),
        "geom_irregularity": np.stack(samples_geom).astype(np.float32),
        "stiffness_irregularity": np.stack(samples_eta).astype(np.float32),
        "stiffness_log_eta_k": np.stack(samples_log_eta).astype(np.float32),
        "mask": np.stack(samples_mask).astype(np.float32),
        "context": np.stack(samples_context).astype(np.float32),
        "meta_json": np.asarray(samples_meta),
        "obs_channel_names": np.asarray(obs_names or []),
        "geom_channel_names": np.asarray(GEOM_CHANNELS),
        "stiffness_channel_names": np.asarray(["eta_k_L", "eta_k_R"]),
        "mask_channel_names": np.asarray(MASK_CHANNELS),
        "context_names": np.asarray(["speed_kmh", "random_seed", "ds_m", "run_index"]),
        "source_npz": np.asarray([str(p) for p in npz_files]),
        "ds_m": np.asarray(args.ds_m, dtype=np.float32),
        "window_m": np.asarray(args.window_m, dtype=np.float32),
        "stride_m": np.asarray(args.stride_m, dtype=np.float32),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build joint inverse dataset for geometry and stiffness irregularities.")
    parser.add_argument("--result-roots", nargs="+", default=["results/ballast_condition_random_scan"], help="Result roots or npz files.")
    parser.add_argument("--output", default="results/joint_inverse_dataset/joint_inverse_dataset.npz", help="Output dataset npz path.")
    parser.add_argument("--ds-m", type=float, default=0.25, help="Uniform spatial grid spacing in meters.")
    parser.add_argument("--window-m", type=float, default=128.0, help="Window length in meters.")
    parser.add_argument("--stride-m", type=float, default=16.0, help="Window stride in meters.")
    parser.add_argument("--include-force", action="store_true", help="Append simulated wheel-rail vertical force channels to obs.")
    parser.add_argument("--min-valid-fraction", type=float, default=0.95, help="Minimum valid-mask fraction per window.")
    parser.add_argument("--min-eta", type=float, default=0.02, help="Lower clamp before log eta.")
    parser.add_argument("--max-eta", type=float, default=50.0, help="Upper clamp before log eta.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **dataset)
    print(f"Saved joint inverse dataset: {output}")
    print(f"  windows: {dataset['obs'].shape[0]}")
    print(f"  obs shape: {dataset['obs'].shape}")
    print(f"  geom shape: {dataset['geom_irregularity'].shape}")
    print(f"  stiffness shape: {dataset['stiffness_irregularity'].shape}")


if __name__ == "__main__":
    main()
