from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml


DEFAULT_MANIFEST = Path("configs/sweeps/sleeper_void_2mm_position_scan.yaml")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML top level must be a mapping: {path}")
    return data


def selected_cases(manifest: dict[str, Any], include: list[str] | None) -> list[dict[str, Any]]:
    cases = manifest.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest.cases must be a non-empty list")
    include_set = set(include or [])
    return [case for case in cases if not include_set or str(case.get("case_id")) in include_set]


def expected_run_note(manifest: dict[str, Any], case_id: str) -> str:
    common = manifest.get("common", {}) or {}
    return f'{common.get("note_prefix", "sweep")}_{case_id}'


def find_case_result(workspace: Path, manifest: dict[str, Any], case: dict[str, Any]) -> Path | None:
    common = manifest.get("common", {}) or {}
    project_name = str(common.get("project_name", manifest.get("manifest_name", "default_project")))
    result_root = workspace / "results" / project_name
    if not result_root.exists():
        return None

    case_id = str(case["case_id"])
    run_note = expected_run_note(manifest, case_id)
    candidates: list[tuple[float, Path]] = []
    for meta_path in result_root.glob("*/files/run_meta.yaml"):
        try:
            meta = load_yaml(meta_path)
        except Exception:
            continue
        meta_run_note = str(meta.get("run_note", ""))
        meta_profile = str(meta.get("param_profile_dir", "")).replace("\\", "/")
        if meta_run_note != run_note and not meta_profile.endswith(case_id):
            continue
        npz_path = meta_path.with_name("simulation_result.npz")
        if npz_path.exists():
            candidates.append((npz_path.stat().st_mtime, npz_path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def run_args_for_result(npz_path: Path) -> dict[str, Any]:
    return load_json(npz_path.with_name("argparse_params.json"))


def load_series(npz_path: Path, case: dict[str, Any]) -> dict[str, Any]:
    data = np.load(npz_path)
    a = data["A"]
    dt = float(data["dt"])
    nt = a.shape[0]
    t = np.arange(nt) * dt

    def col(index: int) -> np.ndarray:
        return a[:, index] if a.shape[1] > index else np.zeros(nt)

    total_v = data["TotalVerticalForce"] if "TotalVerticalForce" in data.files else None
    total_v_p2 = data["TotalVerticalForce_Point2"] if "TotalVerticalForce_Point2" in data.files else None
    if (
        total_v is not None
        and total_v_p2 is not None
        and total_v.ndim == 2
        and total_v_p2.ndim == 2
        and total_v.shape == total_v_p2.shape
        and np.any(np.abs(total_v_p2) > 1e-9)
    ):
        total_v = total_v + total_v_p2

    if total_v is not None and total_v.ndim == 2 and total_v.shape[0] == nt and total_v.shape[1] >= 8:
        fz_sum_kn = np.sum(total_v[:, :8], axis=1) / 1000.0
        fz_axle1_kn = (total_v[:, 0] + total_v[:, 1]) / 1000.0
    elif total_v is not None and total_v.ndim == 2 and total_v.shape[0] == nt and total_v.shape[1] >= 2:
        fz_sum_kn = np.sum(total_v, axis=1) / 1000.0
        fz_axle1_kn = (total_v[:, 0] + total_v[:, 1]) / 1000.0
    else:
        fz_sum_kn = np.zeros(nt)
        fz_axle1_kn = np.zeros(nt)

    mileage_m = None
    if "Track_rel_mileage_m" in data.files:
        mileage_m = np.asarray(data["Track_rel_mileage_m"], dtype=float).reshape(-1)[:nt]

    return {
        "case_id": str(case["case_id"]),
        "note": str(case.get("note", "")),
        "npz_path": npz_path,
        "dt": dt,
        "t": t,
        "mileage_m": mileage_m,
        "carbody_az_g": col(1) / 9.81,
        "bogie1_az_g": col(6) / 9.81,
        "wheelset1_az_g": col(16) / 9.81,
        "fz_axle1_kn": fz_axle1_kn,
        "fz_sum_kn": fz_sum_kn,
        "args": run_args_for_result(npz_path),
        "defects": case.get("structure_defects", []) or [],
    }


def downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(x), len(y))
    if n <= max_points:
        return x[:n], y[:n]
    step = int(np.ceil(n / max_points))
    return x[:n:step], y[:n:step]


def rms(x: np.ndarray) -> float:
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((values - np.mean(values)) ** 2)))


def void_ranges_m(item: dict[str, Any], spacing_m: float = 0.6) -> list[tuple[float, float, int]]:
    ranges: list[tuple[float, float, int]] = []
    for defect in item.get("defects", []):
        if str(defect.get("type", defect.get("kind", ""))).lower() not in {"sleeper_void", "void"}:
            continue
        start_m = float(defect.get("start_m", defect.get("relative_m", 0.0)))
        count = max(1, int(float(defect.get("count", 1))))
        end_m = start_m + max(0, count - 1) * spacing_m
        ranges.append((start_m, end_m, count))
    return ranges


def vehicle_constants(item: dict[str, Any]) -> tuple[float, float, float, float]:
    args = item.get("args", {}) or {}
    vx_kmh = float(args.get("vx_set", 215.0))
    x0 = float(args.get("X0", 20.0))
    lc = 9.0
    lt = 1.2
    return vx_kmh / 3.6, x0, lc, lt


def void_ranges_time(item: dict[str, Any]) -> list[tuple[float, float, float, int]]:
    v, x0, lc, lt = vehicle_constants(item)
    axle_lead = 2.0 * (lc + lt)
    ranges: list[tuple[float, float, float, int]] = []
    for start_m, end_m, count in void_ranges_m(item):
        t_first_axle = (start_m - x0 - axle_lead) / v
        t_last_axle = (end_m - x0) / v
        t_ref = (start_m - x0) / v
        ranges.append((t_first_axle, t_last_axle, t_ref, count))
    return ranges


def mark_voids_time(ax, item: dict[str, Any]) -> None:
    for t0, t1, tref, count in void_ranges_time(item):
        ax.axvspan(t0, t1, color="#F4A261", alpha=0.16, lw=0)
        ax.axvline(tref, color="#E76F51", lw=0.9, ls="--", alpha=0.75)
        ylim = ax.get_ylim()
        ax.text(tref, ylim[1], f"void x{count}", color="#B44B35", fontsize=7, rotation=90,
                ha="right", va="top")


def mark_voids_mileage(ax, item: dict[str, Any]) -> None:
    for start_m, end_m, count in void_ranges_m(item):
        ax.axvspan(start_m, end_m, color="#F4A261", alpha=0.16, lw=0)
        ax.axvline(start_m, color="#E76F51", lw=0.9, ls="--", alpha=0.75)
        ylim = ax.get_ylim()
        ax.text(start_m, ylim[1], f"void x{count}", color="#B44B35", fontsize=7, rotation=90,
                ha="right", va="top")


def plot_overlay(series: list[dict[str, Any]], out_dir: Path, x_axis: str) -> list[Path]:
    x_key = "t" if x_axis == "time" else "mileage_m"
    xlabel = "Time (s)" if x_axis == "time" else "Relative mileage (m)"
    suffix = "time" if x_axis == "time" else "mileage"
    mark_voids = mark_voids_time if x_axis == "time" else mark_voids_mileage

    panels = [
        ("carbody_az_g", "Carbody vertical acceleration", "g"),
        ("bogie1_az_g", "Bogie 1 vertical acceleration", "g"),
        ("wheelset1_az_g", "Wheelset 1 vertical acceleration", "g"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, 8.5), sharex=True, constrained_layout=True)
    for ax, (key, title, ylabel) in zip(axes, panels):
        for item in series:
            x = item.get(x_key)
            if x is None:
                continue
            x_ds, y_ds = downsample_xy(np.asarray(x), np.asarray(item[key]))
            ax.plot(x_ds, y_ds, lw=1.0, label=item["case_id"])
        for item in series:
            mark_voids(ax, item)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, ls="--")
    axes[-1].set_xlabel(xlabel)
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    path_acc = out_dir / f"void2mm_acceleration_overlay_{suffix}_marked.png"
    fig.savefig(path_acc, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True, constrained_layout=True)
    for ax, key, title in [
        (axes[0], "fz_axle1_kn", "Wheel-rail vertical force, axle 1 sum"),
        (axes[1], "fz_sum_kn", "Wheel-rail vertical force, all contacts sum"),
    ]:
        for item in series:
            x = item.get(x_key)
            if x is None:
                continue
            x_ds, y_ds = downsample_xy(np.asarray(x), np.asarray(item[key]))
            ax.plot(x_ds, y_ds, lw=1.0, label=item["case_id"])
        for item in series:
            mark_voids(ax, item)
        ax.set_title(title)
        ax.set_ylabel("kN")
        ax.grid(True, alpha=0.25, ls="--")
    axes[-1].set_xlabel(xlabel)
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    path_force = out_dir / f"void2mm_wheelrail_force_overlay_{suffix}_marked.png"
    fig.savefig(path_force, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [path_acc, path_force]


def baseline_for_case(item: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    case_id = item["case_id"]
    if "_noirr_" in case_id:
        return by_id.get("case_01_noirr_no_void")
    if "_irr_" in case_id:
        return by_id.get("case_05_irr_no_void")
    return None


def plot_delta(series: list[dict[str, Any]], out_dir: Path, x_axis: str) -> Path | None:
    by_id = {item["case_id"]: item for item in series}
    pairs = [(item, baseline_for_case(item, by_id)) for item in series]
    pairs = [(item, base) for item, base in pairs if base is not None and item is not base]
    if not pairs:
        return None

    x_key = "t" if x_axis == "time" else "mileage_m"
    xlabel = "Time (s)" if x_axis == "time" else "Relative mileage (m)"
    suffix = "time" if x_axis == "time" else "mileage"
    mark_voids = mark_voids_time if x_axis == "time" else mark_voids_mileage
    panels = [
        ("carbody_az_g", "Delta carbody Az", "g"),
        ("bogie1_az_g", "Delta bogie1 Az", "g"),
        ("wheelset1_az_g", "Delta wheelset1 Az", "g"),
        ("fz_axle1_kn", "Delta axle1 wheel-rail Fz", "kN"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
    for ax, (key, title, ylabel) in zip(axes.ravel(), panels):
        for item, base in pairs:
            x = item.get(x_key)
            bx = base.get(x_key)
            if x is None or bx is None:
                continue
            n = min(len(x), len(item[key]), len(base[key]))
            y = np.asarray(item[key])[:n] - np.asarray(base[key])[:n]
            x_ds, y_ds = downsample_xy(np.asarray(x)[:n], y)
            ax.plot(x_ds, y_ds, lw=1.0, label=item["case_id"])
        for item, _ in pairs:
            mark_voids(ax, item)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, ls="--")
    axes[0, 0].legend(fontsize=7, ncol=2, frameon=False)
    path = out_dir / f"void2mm_delta_from_group_baseline_{suffix}_marked.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_bars(series: list[dict[str, Any]], out_dir: Path) -> Path:
    labels = [item["case_id"] for item in series]
    metrics = [
        ("carbody_az_g", "Carbody Az RMS (g)"),
        ("bogie1_az_g", "Bogie1 Az RMS (g)"),
        ("wheelset1_az_g", "Wheelset1 Az RMS (g)"),
        ("fz_axle1_kn", "Axle1 Fz RMS (kN)"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 2.4 * len(metrics)), constrained_layout=True)
    x = np.arange(len(series))
    for ax, (key, title) in zip(axes, metrics):
        values = [rms(item[key]) for item in series]
        ax.bar(x, values, color="#2A9D8F", alpha=0.9)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=7)
        ax.grid(True, axis="y", alpha=0.25, ls="--")
    path = out_dir / "void2mm_metric_summary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot sleeper void response overlays with void location markers.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Sweep manifest YAML.")
    parser.add_argument("--cases", nargs="*", help="Optional case_id list.")
    parser.add_argument("--project-name", default="", help="Override manifest common.project_name.")
    parser.add_argument("--x-axis", choices=["time", "mileage", "both"], default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    manifest_path = Path(args.manifest).resolve()
    workspace = manifest_path.parent.parent.parent
    manifest = load_yaml(manifest_path)
    if args.project_name:
        manifest.setdefault("common", {})["project_name"] = args.project_name

    cases = selected_cases(manifest, args.cases)
    loaded: list[dict[str, Any]] = []
    missing: list[str] = []
    for case in cases:
        result = find_case_result(workspace, manifest, case)
        if result is None:
            missing.append(str(case["case_id"]))
            continue
        loaded.append(load_series(result, case))

    if not loaded:
        raise FileNotFoundError("No matching simulation_result.npz files found for selected cases.")

    project_name = str((manifest.get("common", {}) or {}).get("project_name", manifest.get("manifest_name", "default_project")))
    out_dir = workspace / "results" / project_name / "_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    axes = ["time", "mileage"] if args.x_axis == "both" else [args.x_axis]
    for axis in axes:
        if axis == "mileage" and not any(item.get("mileage_m") is not None for item in loaded):
            continue
        paths.extend(plot_overlay(loaded, out_dir, axis))
        delta_path = plot_delta(loaded, out_dir, axis)
        if delta_path is not None:
            paths.append(delta_path)
    paths.append(plot_metric_bars(loaded, out_dir))

    print(f"Loaded {len(loaded)} case(s).")
    if missing:
        print("Missing case result(s): " + ", ".join(missing))
    print("Wrote:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
