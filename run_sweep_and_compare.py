from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from analyze_results import (
    default_dynamic_irr_path,
    fft_amp,
    read_numeric_column,
    resample_pair_to_common_grid,
    run_all_analyses,
    welch_spatial_psd,
    welch_time_psd_no_preprocess,
)


DEFAULT_MANIFEST = Path("configs/sweeps/high_speed_passenger_primary_secondary_suspension_scan.yaml")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML top level must be a mapping: {path}")
    return data


def command_to_text(command: list[str]) -> str:
    return subprocess.list2cmdline([str(x) for x in command])


def run_command(command: list[str], cwd: Path, dry_run: bool = False) -> None:
    print(command_to_text(command))
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def selected_cases(manifest: dict[str, Any], include: list[str] | None) -> list[dict[str, Any]]:
    cases = manifest.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest.cases must be a non-empty list")
    include_set = set(include or [])
    return [case for case in cases if not include_set or case["case_id"] in include_set]


def expected_run_note(manifest: dict[str, Any], case_id: str) -> str:
    common = manifest.get("common", {}) or {}
    return f'{common.get("note_prefix", "sweep")}_{case_id}'


def case_param_updates(case: dict[str, Any]) -> dict[str, float]:
    vehicle_patch = case.get("updates", {}).get("vehicle_params.yaml", {})
    params: dict[str, float] = {}

    def collect_numeric_leaves(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if isinstance(value, dict):
                collect_numeric_leaves(value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                params[str(key)] = float(value)

    collect_numeric_leaves(vehicle_patch)
    return params


def find_case_result(workspace: Path, manifest: dict[str, Any], case: dict[str, Any]) -> Path | None:
    common = manifest.get("common", {}) or {}
    project_name = str(common.get("project_name", manifest.get("manifest_name", "default_project")))
    run_note = expected_run_note(manifest, str(case["case_id"]))
    result_root = workspace / "results" / project_name
    if not result_root.exists():
        return None

    candidates: list[tuple[float, Path]] = []
    for meta_path in result_root.glob("*/files/run_meta.yaml"):
        try:
            meta = load_yaml(meta_path)
        except Exception:
            continue
        meta_run_note = str(meta.get("run_note", ""))
        meta_profile = str(meta.get("param_profile_dir", ""))
        if meta_run_note != run_note and not meta_profile.replace("\\", "/").endswith(str(case["case_id"])):
            continue
        npz_path = meta_path.with_name("simulation_result.npz")
        if npz_path.exists():
            candidates.append((npz_path.stat().st_mtime, npz_path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def load_case_series(npz_path: Path, case_id: str, note: str) -> dict[str, Any]:
    data = np.load(npz_path)
    A = data["A"]
    dt = float(data["dt"])
    nt = A.shape[0]
    t = np.arange(nt) * dt

    def col(index: int) -> np.ndarray:
        return A[:, index] if A.shape[1] > index else np.zeros(nt)

    def arr(name: str):
        return data[name] if name in data.files else None

    total_v = arr("TotalVerticalForce")
    total_v_p2 = arr("TotalVerticalForce_Point2")
    if (
        total_v is not None
        and total_v_p2 is not None
        and total_v.ndim == 2
        and total_v_p2.ndim == 2
        and total_v.shape == total_v_p2.shape
        and np.any(np.abs(total_v_p2) > 1e-9)
    ):
        total_v = total_v + total_v_p2

    total_l = arr("TotalLateralForce")
    erxi_z = arr("Erxi_Force_z")
    yixi_z = arr("Yixi_Force_z")
    defect_fastener_nodes = arr("Structure_defect_fastener_nodes")
    defect_void_nodes = arr("Structure_defect_void_nodes")
    defect_void_contact_nodes = arr("Structure_defect_void_contact_nodes")
    defect_max_gap_m = arr("Structure_defect_max_gap_m")
    defect_ballast_nodes = arr("Structure_defect_ballast_nodes")
    defect_ballast_eta_k = arr("Structure_defect_ballast_eta_k_max")
    defect_ballast_eta_c = arr("Structure_defect_ballast_eta_c_max")
    defect_ballast_condition_force = arr("Structure_defect_ballast_condition_FV_sum")

    fz_axle1_sum_kn = np.zeros(nt)
    if total_v is not None and total_v.ndim == 2 and total_v.shape[0] == nt and total_v.shape[1] >= 2:
        fz_axle1_sum_kn = (total_v[:, 0] + total_v[:, 1]) / 1000.0

    fy_axle1_mean_kn = np.zeros(nt)
    if total_l is not None and total_l.ndim == 2 and total_l.shape[0] == nt and total_l.shape[1] >= 2:
        fy_axle1_mean_kn = 0.5 * (total_l[:, 0] + total_l[:, 1]) / 1000.0

    primary_z_mean_kn = np.zeros(nt)
    if yixi_z is not None and yixi_z.ndim == 2 and yixi_z.shape[0] == nt:
        primary_z_mean_kn = np.mean(yixi_z, axis=1) / 1000.0

    secondary_z_mean_kn = np.zeros(nt)
    if erxi_z is not None and erxi_z.ndim == 2 and erxi_z.shape[0] == nt:
        secondary_z_mean_kn = np.mean(erxi_z, axis=1) / 1000.0

    mileage_km = None
    if "Track_abs_mileage_m" in data.files:
        mileage_m = np.asarray(data["Track_abs_mileage_m"]).reshape(-1)[:nt]
        mileage_km = mileage_m / 1000.0
    elif "Irre_distance_m" in data.files and "Track_abs_mileage_m" in data.files:
        mileage_km = (float(data["Track_abs_mileage_m"][0]) + data["Irre_distance_m"][:nt]) / 1000.0

    return {
        "case_id": case_id,
        "note": note,
        "npz_path": npz_path,
        "dt": dt,
        "t": t,
        "mileage_km": mileage_km,
        "carbody_ay_g": col(0) / 9.81,
        "carbody_az_g": col(1) / 9.81,
        "bogie1_ay_g": col(5) / 9.81,
        "bogie1_az_g": col(6) / 9.81,
        "wheelset1_ay_g": col(15) / 9.81,
        "wheelset1_az_g": col(16) / 9.81,
        "fz_axle1_sum_kn": fz_axle1_sum_kn,
        "fy_axle1_mean_kn": fy_axle1_mean_kn,
        "primary_z_mean_kn": primary_z_mean_kn,
        "secondary_z_mean_kn": secondary_z_mean_kn,
        "structure_defect_fastener_nodes": defect_fastener_nodes,
        "structure_defect_void_nodes": defect_void_nodes,
        "structure_defect_void_contact_nodes": defect_void_contact_nodes,
        "structure_defect_max_gap_m": defect_max_gap_m,
        "structure_defect_ballast_nodes": defect_ballast_nodes,
        "structure_defect_ballast_eta_k": defect_ballast_eta_k,
        "structure_defect_ballast_eta_c": defect_ballast_eta_c,
        "structure_defect_ballast_condition_force": defect_ballast_condition_force,
    }


def defect_metric_max(arr: np.ndarray | None) -> float:
    if arr is None:
        return 0.0
    values = np.asarray(arr, dtype=float)
    return float(np.nanmax(values)) if values.size else 0.0


def defect_abs_metric_max(arr: np.ndarray | None) -> float:
    if arr is None:
        return 0.0
    values = np.asarray(arr, dtype=float)
    return float(np.nanmax(np.abs(values))) if values.size else 0.0


def defect_contact_ratio(void_nodes: np.ndarray | None, contact_nodes: np.ndarray | None) -> float:
    if void_nodes is None or contact_nodes is None:
        return 0.0
    void_values = np.asarray(void_nodes, dtype=float)
    contact_values = np.asarray(contact_nodes, dtype=float)
    if void_values.size == 0 or contact_values.size == 0:
        return 0.0
    total_void = float(np.nansum(void_values))
    if total_void <= 0.0:
        return 0.0
    return float(np.nansum(contact_values) / total_void)


def downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(x), len(y))
    if n <= max_points:
        return x[:n], y[:n]
    step = int(np.ceil(n / max_points))
    return x[:n:step], y[:n:step]


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean((x - np.mean(x)) ** 2))) if len(x) else np.nan


def write_summary(cases: list[dict[str, Any]], series: list[dict[str, Any]], out_dir: Path) -> Path:
    rows = []
    updates_by_case = {case["case_id"]: case_param_updates(case) for case in cases}
    for item in series:
        row = {
            "case_id": item["case_id"],
            "note": item["note"],
            "result_npz": str(item["npz_path"]),
            "carbody_az_rms_g": rms(item["carbody_az_g"]),
            "carbody_ay_rms_g": rms(item["carbody_ay_g"]),
            "bogie1_az_rms_g": rms(item["bogie1_az_g"]),
            "wheelset1_az_rms_g": rms(item["wheelset1_az_g"]),
            "wheelrail_fz_axle1_sum_rms_kn": rms(item["fz_axle1_sum_kn"]),
            "wheelrail_fz_axle1_sum_peak_kn": float(np.nanmax(np.abs(item["fz_axle1_sum_kn"]))),
            "primary_z_mean_rms_kn": rms(item["primary_z_mean_kn"]),
            "secondary_z_mean_rms_kn": rms(item["secondary_z_mean_kn"]),
            "structure_defect_fastener_nodes_max": defect_metric_max(item.get("structure_defect_fastener_nodes")),
            "structure_defect_void_nodes_max": defect_metric_max(item.get("structure_defect_void_nodes")),
            "structure_defect_void_contact_ratio": defect_contact_ratio(
                item.get("structure_defect_void_nodes"),
                item.get("structure_defect_void_contact_nodes"),
            ),
            "structure_defect_max_gap_mm": defect_metric_max(item.get("structure_defect_max_gap_m")) * 1000.0,
            "structure_defect_ballast_nodes_max": defect_metric_max(item.get("structure_defect_ballast_nodes")),
            "structure_defect_ballast_eta_k_max": defect_metric_max(item.get("structure_defect_ballast_eta_k")),
            "structure_defect_ballast_eta_c_max": defect_metric_max(item.get("structure_defect_ballast_eta_c")),
            "structure_defect_ballast_condition_force_peak_kn": defect_abs_metric_max(item.get("structure_defect_ballast_condition_force")) / 1000.0,
        }
        row.update(updates_by_case.get(item["case_id"], {}))
        rows.append(row)
    path = out_dir / "sweep_response_summary.csv"
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(path, index=False, encoding="utf-8-sig")

    if not summary_df.empty:
        baseline = summary_df.iloc[0]
        rel_df = summary_df[["case_id", "note"]].copy()
        metric_cols = [
            col for col in summary_df.columns
            if col.endswith("_rms_g") or col.endswith("_rms_kn") or col.endswith("_peak_kn")
        ]
        for col in metric_cols:
            base = float(baseline[col])
            rel_df[col + "_pct_vs_baseline"] = (summary_df[col].astype(float) / (base + 1e-12) - 1.0) * 100.0
        rel_df.to_csv(out_dir / "sweep_response_relative_to_baseline.csv", index=False, encoding="utf-8-sig")
    return path


def plot_metric_changes(summary_csv: Path, out_dir: Path) -> Path | None:
    df = pd.read_csv(summary_csv)
    if df.empty:
        return None

    baseline = df.iloc[0]
    metrics = [
        ("carbody_az_rms_g", "Carbody Az RMS"),
        ("carbody_ay_rms_g", "Carbody Ay RMS"),
        ("bogie1_az_rms_g", "Bogie1 Az RMS"),
        ("wheelset1_az_rms_g", "WS1 Az RMS"),
        ("wheelrail_fz_axle1_sum_rms_kn", "Wheel-rail Fz RMS"),
        ("secondary_z_mean_rms_kn", "Secondary Fz RMS"),
    ]
    available = [(col, label) for col, label in metrics if col in df.columns]
    if not available:
        return None

    fig, axes = plt.subplots(len(available), 1, figsize=(13, 2.2 * len(available)), constrained_layout=True)
    if len(available) == 1:
        axes = [axes]

    x = np.arange(len(df))
    for ax, (col, label) in zip(axes, available):
        base = float(baseline[col])
        pct = (df[col].astype(float).to_numpy() / (base + 1e-12) - 1.0) * 100.0
        colors = ["#2A9D8F" if value <= 0 else "#E76F51" for value in pct]
        ax.bar(x, pct, color=colors, alpha=0.9)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_ylabel("%")
        ax.set_title(f"{label}: change relative to {df.iloc[0]['case_id']}")
        ax.grid(True, axis="y", alpha=0.25, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(df["case_id"], rotation=28, ha="right", fontsize=8)

    path = out_dir / "sweep_metric_change_vs_baseline.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_delta_from_baseline(series: list[dict[str, Any]], out_dir: Path, trim_time_s: float = 2.0) -> Path | None:
    if len(series) < 2:
        return None

    baseline = series[0]
    panels = [
        ("carbody_az_g", "Delta carbody vertical acceleration", "g"),
        ("carbody_ay_g", "Delta carbody lateral acceleration", "g"),
        ("bogie1_az_g", "Delta bogie1 vertical acceleration", "g"),
        ("secondary_z_mean_kn", "Delta secondary vertical force mean", "kN"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 8.5), constrained_layout=True)
    for ax, (key, title, ylabel) in zip(axes.ravel(), panels):
        for item in series[1:]:
            n = min(len(item["t"]), len(item[key]), len(baseline["t"]), len(baseline[key]))
            t = item["t"][:n]
            mask = t >= trim_time_s
            if not np.any(mask):
                mask = np.ones(n, dtype=bool)
            y = item[key][:n] - baseline[key][:n]
            x_ds, y_ds = downsample_xy(t[mask], y[mask])
            ax.plot(x_ds, y_ds, lw=1.1, label=item["case_id"])
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.28, ls="--")
    axes[0, 0].legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle(f"Response difference from baseline, t >= {trim_time_s:g}s", fontsize=14)
    path = out_dir / "sweep_delta_from_baseline.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _log_psd_rmse(
    freq: np.ndarray,
    psd_sim: np.ndarray,
    psd_meas: np.ndarray,
    f_low: float,
    f_high: float,
) -> float:
    freq = np.asarray(freq, dtype=float)
    psd_sim = np.asarray(psd_sim, dtype=float)
    psd_meas = np.asarray(psd_meas, dtype=float)
    mask = (
        np.isfinite(freq)
        & np.isfinite(psd_sim)
        & np.isfinite(psd_meas)
        & (freq >= f_low)
        & (freq <= f_high)
        & (psd_sim > 0)
        & (psd_meas > 0)
    )
    if np.count_nonzero(mask) < 3:
        return np.nan
    diff = np.log10(psd_sim[mask] + 1e-30) - np.log10(psd_meas[mask] + 1e-30)
    return float(np.sqrt(np.mean(diff**2)))


def _band_rms_from_psd(freq: np.ndarray, psd: np.ndarray, f_low: float, f_high: float) -> float:
    freq = np.asarray(freq, dtype=float)
    psd = np.asarray(psd, dtype=float)
    mask = np.isfinite(freq) & np.isfinite(psd) & (freq >= f_low) & (freq <= f_high)
    if np.count_nonzero(mask) < 2:
        return np.nan
    return float(np.sqrt(np.trapz(psd[mask], freq[mask])))


def _psd_ratio_db(sim_rms: float, meas_rms: float) -> float:
    if not np.isfinite(sim_rms) or not np.isfinite(meas_rms) or sim_rms <= 0 or meas_rms <= 0:
        return np.nan
    return float(20.0 * np.log10(sim_rms / meas_rms))


def compute_bogie_psd_fit(
    cases: list[dict[str, Any]],
    series: list[dict[str, Any]],
    manifest: dict[str, Any],
    dynamic_irr_path: Path,
    out_dir: Path,
    buffer_time_s: float = 2.0,
) -> tuple[Path | None, list[Path]]:
    """Score each case by Bogie1 acceleration PSD distance from measured acceleration."""
    common = manifest.get("common", {}) or {}
    vx_kmh = float(common.get("vx_set", 215.0))
    v_mps = vx_kmh / 3.6
    if v_mps <= 0:
        return None, []

    dynamic_df = pd.read_csv(dynamic_irr_path, encoding="utf-8")
    meas_mileage_km = read_numeric_column(dynamic_df, "里程")
    meas_vert = read_numeric_column(dynamic_df, "垂向加速度(g)")
    meas_lat = read_numeric_column(dynamic_df, "横向加速度(g)")
    updates_by_case = {case["case_id"]: case_param_updates(case) for case in cases}

    rows: list[dict[str, Any]] = []
    spectra: list[dict[str, Any]] = []
    bands = [(0.5, 2.0), (2.0, 8.0), (8.0, 20.0), (20.0, 80.0)]

    for item in series:
        if item["mileage_km"] is None:
            continue
        mask_time = item["t"] >= buffer_time_s
        if not np.any(mask_time):
            mask_time = np.ones_like(item["t"], dtype=bool)

        row: dict[str, Any] = {"case_id": item["case_id"], "note": item["note"]}
        row.update(updates_by_case.get(item["case_id"], {}))
        spec_item: dict[str, Any] = {"case_id": item["case_id"]}

        for direction, sim_key, meas_sig, suffix in [
            ("vertical", "bogie1_az_g", meas_vert, "z"),
            ("lateral", "bogie1_ay_g", meas_lat, "y"),
        ]:
            try:
                _, sim_grid, meas_grid, dx_m = resample_pair_to_common_grid(
                    np.asarray(item["mileage_km"])[mask_time],
                    np.asarray(item[sim_key])[mask_time],
                    meas_mileage_km,
                    meas_sig,
                )
                f_sim, psd_sim, _ = welch_time_psd_no_preprocess(sim_grid, dx_m, v_mps)
                f_meas, psd_meas, _ = welch_time_psd_no_preprocess(meas_grid, dx_m, v_mps)
            except Exception as exc:
                print(f"[PSD-Fit] {item['case_id']} {direction} skipped: {exc}")
                continue

            n = min(len(f_sim), len(f_meas), len(psd_sim), len(psd_meas))
            f = f_sim[:n]
            psd_sim = psd_sim[:n]
            psd_meas = psd_meas[:n]
            row[f"bogie1_{suffix}_psd_log_rmse_0p5_30hz"] = _log_psd_rmse(f, psd_sim, psd_meas, 0.5, 30.0)
            row[f"bogie1_{suffix}_psd_log_rmse_0p5_80hz"] = _log_psd_rmse(f, psd_sim, psd_meas, 0.5, 80.0)
            for f_low, f_high in bands:
                sim_rms = _band_rms_from_psd(f, psd_sim, f_low, f_high)
                meas_rms = _band_rms_from_psd(f, psd_meas, f_low, f_high)
                label = f"{str(f_low).replace('.', 'p')}_{str(f_high).replace('.', 'p')}hz"
                row[f"bogie1_{suffix}_rms_ratio_db_{label}"] = _psd_ratio_db(sim_rms, meas_rms)

            spec_item[f"freq_{suffix}"] = f
            spec_item[f"sim_{suffix}"] = psd_sim
            spec_item[f"meas_{suffix}"] = psd_meas

        z_score = row.get("bogie1_z_psd_log_rmse_0p5_30hz", np.nan)
        y_score = row.get("bogie1_y_psd_log_rmse_0p5_30hz", np.nan)
        row["bogie1_psd_fit_score_0p5_30hz"] = float(np.nanmean([z_score, y_score]))
        rows.append(row)
        spectra.append(spec_item)

    if not rows:
        return None, []

    df = pd.DataFrame(rows).sort_values("bogie1_psd_fit_score_0p5_30hz", na_position="last")
    csv_path = out_dir / "bogie_psd_fit_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    saved: list[Path] = []
    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    x = np.arange(len(df))
    ax.bar(x, df["bogie1_psd_fit_score_0p5_30hz"], color="#457B9D", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(df["case_id"], rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("log10 PSD RMSE")
    ax.set_title("Bogie1 acceleration PSD fit to measured data (lower is better, 0.5-30 Hz)")
    ax.grid(True, axis="y", alpha=0.25, ls="--")
    rank_path = out_dir / "bogie_psd_fit_ranking.png"
    fig.savefig(rank_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(rank_path)

    spectra_by_case = {item["case_id"]: item for item in spectra}
    top_case_ids = df["case_id"].head(3).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), constrained_layout=True)
    for suffix, ax, title in [
        ("z", axes[0], "Bogie1 vertical acceleration PSD"),
        ("y", axes[1], "Bogie1 lateral acceleration PSD"),
    ]:
        plotted_meas = False
        for case_id in top_case_ids:
            spec = spectra_by_case.get(case_id, {})
            f = spec.get(f"freq_{suffix}")
            psd_sim = spec.get(f"sim_{suffix}")
            psd_meas = spec.get(f"meas_{suffix}")
            if f is None or psd_sim is None:
                continue
            keep = (f > 0) & (f <= 100.0)
            ax.loglog(f[keep], psd_sim[keep] + 1e-30, lw=1.4, label=f"Sim {case_id}")
            if not plotted_meas and psd_meas is not None:
                ax.loglog(f[keep], psd_meas[keep] + 1e-30, lw=1.7, ls="--", color="black", label="Measured")
                plotted_meas = True
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (g^2/Hz)")
        ax.grid(True, which="both", alpha=0.28, ls="--")
        ax.legend(fontsize=8, frameon=False)
    overlay_path = out_dir / "bogie_psd_fit_top3_overlay.png"
    fig.savefig(overlay_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(overlay_path)

    return csv_path, saved


def plot_overlay(series: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    saved: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 11), constrained_layout=True)
    panels = [
        ("carbody_az_g", "Carbody vertical acceleration", "g"),
        ("carbody_ay_g", "Carbody lateral acceleration", "g"),
        ("bogie1_az_g", "Bogie1 vertical acceleration", "g"),
        ("wheelset1_az_g", "Wheelset1 vertical acceleration", "g"),
        ("fz_axle1_sum_kn", "Wheel-rail vertical force, axle1 sum", "kN"),
        ("secondary_z_mean_kn", "Secondary vertical force mean", "kN"),
    ]
    for ax, (key, title, ylabel) in zip(axes.ravel(), panels):
        for item in series:
            x, y = downsample_xy(item["t"], item[key])
            ax.plot(x, y, lw=1.1, label=item["case_id"])
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.28, ls="--")
    axes[0, 0].legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle("Suspension sweep dynamic response comparison", fontsize=14)
    path = out_dir / "sweep_dynamic_response_overlay.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
    panels = [
        ("primary_z_mean_kn", "Primary vertical force mean", "kN"),
        ("secondary_z_mean_kn", "Secondary vertical force mean", "kN"),
        ("fz_axle1_sum_kn", "Wheel-rail vertical force, axle1 sum", "kN"),
        ("fy_axle1_mean_kn", "Wheel-rail lateral force, axle1 mean", "kN"),
    ]
    for ax, (key, title, ylabel) in zip(axes.ravel(), panels):
        for item in series:
            x, y = downsample_xy(item["t"], item[key])
            ax.plot(x, y, lw=1.1, label=item["case_id"])
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.28, ls="--")
    axes[0, 0].legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle("Suspension and wheel-rail force comparison", fontsize=14)
    path = out_dir / "sweep_force_overlay.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for item in series:
        f, amp = fft_amp(item["carbody_az_g"], item["dt"])
        keep = f <= 100.0
        axes[0].plot(f[keep], amp[keep], lw=1.1, label=item["case_id"])
        f2, amp2 = fft_amp(item["fz_axle1_sum_kn"], item["dt"])
        keep2 = f2 <= 100.0
        axes[1].plot(f2[keep2], amp2[keep2], lw=1.1, label=item["case_id"])
    axes[0].set_title("Carbody vertical acceleration FFT")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Amplitude (g)")
    axes[1].set_title("Wheel-rail vertical force FFT")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude (kN)")
    for ax in axes:
        ax.grid(True, alpha=0.28, ls="--")
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    path = out_dir / "sweep_frequency_overlay.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for item in series:
        if item["mileage_km"] is None:
            continue
        mileage_m = item["mileage_km"] * 1000.0
        dx = np.diff(mileage_m)
        dx = dx[np.isfinite(dx) & (dx > 0)]
        if len(dx) == 0:
            continue
        dx_m = float(np.median(dx))
        f_az, psd_az = welch_spatial_psd(item["carbody_az_g"], dx_m)
        f_fz, psd_fz = welch_spatial_psd(item["fz_axle1_sum_kn"], dx_m)
        axes[0].loglog(f_az[f_az > 0], psd_az[f_az > 0] + 1e-30, lw=1.1, label=item["case_id"])
        axes[1].loglog(f_fz[f_fz > 0], psd_fz[f_fz > 0] + 1e-30, lw=1.1, label=item["case_id"])
    axes[0].set_title("Carbody vertical acceleration spatial PSD")
    axes[0].set_xlabel("Spatial frequency (1/m)")
    axes[0].set_ylabel("PSD (g^2/(1/m))")
    axes[1].set_title("Wheel-rail vertical force spatial PSD")
    axes[1].set_xlabel("Spatial frequency (1/m)")
    axes[1].set_ylabel("PSD (kN^2/(1/m))")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.28, ls="--")
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    path = out_dir / "sweep_spatial_psd_overlay.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build/run a suspension sweep, run per-case analyze_results plots, and compare cases."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Sweep manifest path.")
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable.")
    parser.add_argument("--cases", nargs="*", help="Optional subset of case_id values.")
    parser.add_argument("--dry-run", action="store_true", help="Print build/run commands without running simulation.")
    parser.add_argument("--compare-only", action="store_true", help="Skip simulation and compare existing result folders.")
    parser.add_argument("--skip-single-analysis", action="store_true", help="Skip per-case analyze_results figures.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop the simulation sweep on the first failed case.")
    parser.add_argument("--dynamic-irr-path", default="", help="Measured dynamic inspection CSV for analyze_results.")
    parser.add_argument("--out-dir", default="", help="Comparison figure output directory.")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[], help="Extra args forwarded to generate_main.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(__file__).resolve().parent
    manifest_path = (workspace / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest)
    manifest = load_yaml(manifest_path)
    cases = selected_cases(manifest, args.cases)

    if not args.compare_only:
        build_cmd = [args.python_exe, str(workspace / "utils" / "build_param_sweep.py"), "--manifest", str(manifest_path)]
        run_command(build_cmd, workspace, dry_run=args.dry_run)

        run_cmd = [
            args.python_exe,
            str(workspace / "utils" / "run_param_sweep.py"),
            "--manifest",
            str(manifest_path),
        ]
        if args.cases:
            run_cmd.extend(["--cases", *args.cases])
        if args.stop_on_error:
            run_cmd.append("--stop-on-error")
        if args.extra_args:
            run_cmd.extend(["--extra-args", *args.extra_args])
        run_command(run_cmd, workspace, dry_run=args.dry_run)
        if args.dry_run:
            return

    common = manifest.get("common", {}) or {}
    project_name = str(common.get("project_name", manifest.get("manifest_name", "sweep")))
    out_dir = Path(args.out_dir) if args.out_dir else workspace / "results" / project_name / "_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    dynamic_irr_path = Path(args.dynamic_irr_path) if args.dynamic_irr_path else default_dynamic_irr_path()

    loaded = []
    missing = []
    for case in cases:
        result_npz = find_case_result(workspace, manifest, case)
        if result_npz is None:
            missing.append(str(case["case_id"]))
            continue
        if not args.skip_single_analysis:
            run_all_analyses(
                target_file=str(result_npz),
                dynamic_irr_path=dynamic_irr_path,
                run_core=True,
                run_irre=True,
                run_accel=True,
                run_psd=True,
                show=False,
            )
        loaded.append(load_case_series(result_npz, str(case["case_id"]), str(case.get("note", ""))))

    if missing:
        print("No result found for cases: " + ", ".join(missing))
    if not loaded:
        raise RuntimeError("No case results were found to compare.")

    summary_path = write_summary(cases, loaded, out_dir)
    fig_paths = plot_overlay(loaded, out_dir)
    metric_path = plot_metric_changes(summary_path, out_dir)
    delta_path = plot_delta_from_baseline(loaded, out_dir)
    psd_fit_path, psd_fit_figs = compute_bogie_psd_fit(cases, loaded, manifest, dynamic_irr_path, out_dir)
    for extra_path in (metric_path, delta_path):
        if extra_path is not None:
            fig_paths.append(extra_path)
    fig_paths.extend(psd_fit_figs)
    print(f"Summary saved: {summary_path}")
    if psd_fit_path is not None:
        print(f"Bogie PSD fit summary saved: {psd_fit_path}")
    for path in fig_paths:
        print(f"Figure saved: {path}")


if __name__ == "__main__":
    main()
