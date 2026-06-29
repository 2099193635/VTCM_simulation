from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "joint_inverse_stiffness_geometry_scan"
OUT_DIR = RESULT_ROOT / "_comparison"

CASE_ORDER = [
    ("case_00_seed20260627_baseline", "Baseline", "#595959"),
    ("case_01_seed20260627_loose_eta0p2_100m", "Loose eta_k=0.2", "#3B6EA8"),
    ("case_02_seed20260627_harden_eta5_100m", "Hardened eta_k=5", "#B24A3F"),
]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def find_case_npz(case_key: str) -> Path:
    matches = [p for p in RESULT_ROOT.rglob("simulation_result.npz") if case_key in str(p)]
    if not matches:
        raise FileNotFoundError(f"No simulation_result.npz found for {case_key}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_case(case_key: str, label: str, color: str) -> dict:
    path = find_case_npz(case_key)
    data = np.load(path, allow_pickle=True)
    x = np.asarray(data["Irre_distance_m"], dtype=float)
    X = np.asarray(data["X"], dtype=float)
    A = np.asarray(data["A"], dtype=float)
    Fv = np.asarray(data["TotalVerticalForce"], dtype=float)
    mask = np.maximum(
        np.asarray(data["Stiffness_irregularity_mask_L"], dtype=int),
        np.asarray(data["Stiffness_irregularity_mask_R"], dtype=int),
    )
    return {
        "case_key": case_key,
        "label": label,
        "color": color,
        "path": str(path),
        "distance_m": x,
        "mask": mask,
        "carbody_az_mps2": A[:, 1],
        "carbody_z_mm": X[:, 1] * 1000.0,
        "bogie_az_mean_mps2": 0.5 * (A[:, 6] + A[:, 11]),
        "bogie_z_mean_mm": 0.5 * (X[:, 6] + X[:, 11]) * 1000.0,
        "wr_force_mean_kN": np.mean(Fv, axis=1) / 1000.0,
        "wr_force_max_kN": np.max(Fv, axis=1) / 1000.0,
    }


def mask_span(cases: list[dict]) -> tuple[float, float] | None:
    spans = []
    for case in cases:
        idx = np.flatnonzero(case["mask"] > 0)
        if idx.size:
            spans.append((float(case["distance_m"][idx[0]]), float(case["distance_m"][idx[-1]])))
    if not spans:
        return None
    return min(s[0] for s in spans), max(s[1] for s in spans)


def format_axes(ax: plt.Axes, span: tuple[float, float] | None) -> None:
    if span is not None:
        ax.axvspan(span[0], span[1], color="#D7D7D7", alpha=0.24, lw=0, zorder=0)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6)
    ax.tick_params(length=3, width=0.7)


def add_legend(ax: plt.Axes, cases: list[dict], ncol: int = 3) -> None:
    handles = [
        mpl.lines.Line2D([0], [0], color=case["color"], lw=1.6, label=case["label"])
        for case in cases
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.12),
        ncol=ncol,
        borderaxespad=0,
        columnspacing=1.0,
        handlelength=1.8,
    )


def set_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.02,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=8.5,
        va="bottom",
        ha="left",
    )


def robust_ylim(ax: plt.Axes, values: list[np.ndarray], pad: float = 0.15) -> None:
    arr = np.concatenate([np.asarray(v, dtype=float) for v in values])
    lo, hi = np.nanpercentile(arr, [0.8, 99.2])
    if np.isclose(lo, hi):
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    margin = max((hi - lo) * pad, 1e-6)
    ax.set_ylim(lo - margin, hi + margin)


def plot_vehicle_response(cases: list[dict], out_base: Path, xlim: tuple[float, float]) -> None:
    configure_matplotlib()
    span = mask_span(cases)
    panels = [
        ("carbody_az_mps2", "Carbody vertical\nacceleration (m s$^{-2}$)", "a"),
        ("carbody_z_mm", "Carbody vertical\ndisplacement (mm)", "b"),
        ("bogie_az_mean_mps2", "Bogie vertical\nacceleration (m s$^{-2}$)", "c"),
        ("bogie_z_mean_mm", "Bogie vertical\ndisplacement (mm)", "d"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.9), sharex=True, constrained_layout=True)
    axes = axes.ravel()
    for ax, (key, ylabel, panel_label) in zip(axes, panels):
        format_axes(ax, span)
        vals = []
        for case in cases:
            ax.plot(case["distance_m"], case[key], color=case["color"], lw=1.0, alpha=0.92)
            vals.append(case[key][(case["distance_m"] >= xlim[0]) & (case["distance_m"] <= xlim[1])])
        robust_ylim(ax, vals)
        ax.set_xlim(*xlim)
        ax.set_ylabel(ylabel)
        set_panel_label(ax, panel_label)
    axes[2].set_xlabel("Distance from simulation start (m)")
    axes[3].set_xlabel("Distance from simulation start (m)")
    add_legend(axes[0], cases)
    if span is not None:
        axes[0].text(
            span[0],
            axes[0].get_ylim()[1],
            f"  stiffness anomaly {span[0]:.0f}-{span[1]:.0f} m",
            ha="left",
            va="top",
            fontsize=6.4,
            color="#5A5A5A",
        )
    fig.suptitle(
        "Vehicle vertical response under identical geometry but different support stiffness",
        x=0.01,
        y=1.03,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    save_all(fig, out_base)


def plot_wheel_rail_force(cases: list[dict], out_base: Path, xlim: tuple[float, float]) -> None:
    configure_matplotlib()
    span = mask_span(cases)
    panels = [
        ("wr_force_mean_kN", "Mean wheel-rail\nvertical force (kN)", "a"),
        ("wr_force_max_kN", "Maximum contact\nvertical force (kN)", "b"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True, constrained_layout=True)
    for ax, (key, ylabel, panel_label) in zip(axes, panels):
        format_axes(ax, span)
        vals = []
        for case in cases:
            ax.plot(case["distance_m"], case[key], color=case["color"], lw=1.0, alpha=0.92)
            vals.append(case[key][(case["distance_m"] >= xlim[0]) & (case["distance_m"] <= xlim[1])])
        robust_ylim(ax, vals, pad=0.10)
        ax.set_xlim(*xlim)
        ax.set_ylabel(ylabel)
        set_panel_label(ax, panel_label)
    axes[-1].set_xlabel("Distance from simulation start (m)")
    add_legend(axes[0], cases)
    if span is not None:
        axes[0].text(
            span[0],
            axes[0].get_ylim()[1],
            f"  stiffness anomaly {span[0]:.0f}-{span[1]:.0f} m",
            ha="left",
            va="top",
            fontsize=6.4,
            color="#5A5A5A",
        )
    fig.suptitle(
        "Wheel-rail vertical force under identical geometry but different support stiffness",
        x=0.01,
        y=1.03,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    save_all(fig, out_base)


def save_all(fig: plt.Figure, out_base: Path) -> None:
    for suffix in [".png", ".pdf", ".svg", ".tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix in {".png", ".tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(out_base.with_suffix(suffix), **kwargs)
    plt.close(fig)


def metric_summary(cases: list[dict], out_json: Path, out_csv: Path, xlim: tuple[float, float]) -> None:
    metric_keys = [
        "carbody_az_mps2",
        "carbody_z_mm",
        "bogie_az_mean_mps2",
        "bogie_z_mean_mm",
        "wr_force_mean_kN",
        "wr_force_max_kN",
    ]
    rows = []
    for case in cases:
        distance = case["distance_m"]
        in_view = (distance >= xlim[0]) & (distance <= xlim[1])
        in_defect = case["mask"] > 0
        row = {"case": case["label"], "source_npz": case["path"]}
        for region_name, selector in [("view", in_view), ("defect", in_defect)]:
            for key in metric_keys:
                vals = np.asarray(case[key][selector], dtype=float)
                if vals.size == 0:
                    row[f"{region_name}_{key}_rms"] = float("nan")
                    row[f"{region_name}_{key}_peak_abs"] = float("nan")
                    row[f"{region_name}_{key}_mean"] = float("nan")
                    continue
                row[f"{region_name}_{key}_rms"] = float(np.sqrt(np.nanmean(vals * vals)))
                row[f"{region_name}_{key}_peak_abs"] = float(np.nanmax(np.abs(vals)))
                row[f"{region_name}_{key}_mean"] = float(np.nanmean(vals))
        rows.append(row)

    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_source_data(cases: list[dict], out_csv: Path) -> None:
    keys = [
        "carbody_az_mps2",
        "carbody_z_mm",
        "bogie_az_mean_mps2",
        "bogie_z_mean_mm",
        "wr_force_mean_kN",
        "wr_force_max_kN",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "distance_m", "stiffness_mask", *keys])
        writer.writeheader()
        for case in cases:
            for i, s in enumerate(case["distance_m"]):
                row = {
                    "case": case["label"],
                    "distance_m": f"{s:.6g}",
                    "stiffness_mask": int(case["mask"][i]),
                }
                for key in keys:
                    row[key] = f"{case[key][i]:.8g}"
                writer.writerow(row)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [load_case(*case_info) for case_info in CASE_ORDER]
    xlim = (150.0, 298.5)
    plot_vehicle_response(
        cases,
        OUT_DIR / "joint_inverse_vehicle_vertical_response_comparison",
        xlim,
    )
    plot_wheel_rail_force(
        cases,
        OUT_DIR / "joint_inverse_wheel_rail_force_comparison",
        xlim,
    )
    write_source_data(cases, OUT_DIR / "joint_inverse_vehicle_force_source_data.csv")
    metric_summary(
        cases,
        OUT_DIR / "joint_inverse_vehicle_force_summary.json",
        OUT_DIR / "joint_inverse_vehicle_force_summary.csv",
        xlim,
    )
    print("Saved vehicle response figure:", OUT_DIR / "joint_inverse_vehicle_vertical_response_comparison.png")
    print("Saved wheel-rail force figure:", OUT_DIR / "joint_inverse_wheel_rail_force_comparison.png")
    print("Saved summary:", OUT_DIR / "joint_inverse_vehicle_force_summary.csv")


if __name__ == "__main__":
    main()
