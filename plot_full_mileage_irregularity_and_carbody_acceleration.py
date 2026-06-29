from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCAN_ROOTS = {
    "Deterministic settlement": Path("results/transition_settlement_scan"),
    "Random irregularity + settlement": Path("results/transition_settlement_random_scan"),
}
OUT_DIR = Path("results/transition_settlement_full_mileage_input_response")
SETTLEMENT_ZONE_M = (80.0, 100.0)


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.4,
    "legend.fontsize": 6.0,
    "legend.frameon": False,
    "figure.dpi": 160,
})


COLORS = {
    "baseline": "#74787C",
    "cosine": "#0077A3",
    "kink": "#C24D2C",
    "grid": "#D9D9D9",
    "zone": "#E9EEF3",
}


def _case_id_from_path(path: Path) -> str:
    name = path.parents[1].name
    if "settlement_random_" in name:
        return name.split("settlement_random_")[-1].rsplit("-", 1)[0]
    if "settlement_" in name:
        return name.split("settlement_")[-1].rsplit("-", 1)[0]
    return name


def _metadata(data: np.lib.npyio.NpzFile) -> list[dict]:
    if "Settlement_metadata_json" not in data.files:
        return []
    raw = data["Settlement_metadata_json"]
    text = str(raw.item() if raw.shape == () else raw.tolist())
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def load_scan(scan_name: str, root: Path) -> list[dict]:
    cases = []
    for path in sorted(root.rglob("simulation_result.npz")):
        if "_publication_figures" in path.parts:
            continue
        data = np.load(path, allow_pickle=True)
        meta = _metadata(data)
        if meta:
            spec = meta[0]
            family = str(spec.get("type", "")).lower()
            amplitude_mm = float(spec.get("amplitude_mm", abs(np.nanmin(data["Settlement_profile_mm"]))))
        else:
            family = "baseline"
            amplitude_mm = 0.0

        distance_m = np.asarray(data["Settlement_distance_rel_m"], dtype=float)
        vertical_irre_mm = 0.5 * (np.asarray(data["Irre_bz_L_ref"], dtype=float) + np.asarray(data["Irre_bz_R_ref"], dtype=float)) * 1000.0
        lateral_irre_mm = 0.5 * (np.asarray(data["Irre_by_L_ref"], dtype=float) + np.asarray(data["Irre_by_R_ref"], dtype=float)) * 1000.0
        carbody_ay = np.asarray(data["A"][:, 0], dtype=float)
        carbody_az = np.asarray(data["A"][:, 1], dtype=float)

        cases.append({
            "scan": scan_name,
            "case_id": _case_id_from_path(path),
            "family": family,
            "amplitude_mm": amplitude_mm,
            "distance_m": distance_m,
            "vertical_irre_mm": vertical_irre_mm,
            "lateral_irre_mm": lateral_irre_mm,
            "carbody_ay_mps2": carbody_ay,
            "carbody_az_mps2": carbody_az,
            "peak_full_carbody_ay_mps2": float(np.nanmax(np.abs(carbody_ay))),
            "peak_full_carbody_az_mps2": float(np.nanmax(np.abs(carbody_az))),
            "rms_full_carbody_ay_mps2": float(np.sqrt(np.nanmean(carbody_ay ** 2))),
            "rms_full_carbody_az_mps2": float(np.sqrt(np.nanmean(carbody_az ** 2))),
            "path": path,
        })
    order = {"baseline": 0, "cosine": 1, "kink": 2}
    return sorted(cases, key=lambda row: (order.get(row["family"], 9), row["amplitude_mm"]))


def load_all() -> dict[str, list[dict]]:
    scans = {}
    for scan_name, root in SCAN_ROOTS.items():
        cases = load_scan(scan_name, root)
        if len(cases) != 9:
            raise RuntimeError(f"Expected 9 cases in {root}, found {len(cases)}")
        scans[scan_name] = cases
    return scans


def _style_axis(ax):
    ax.grid(True, color=COLORS["grid"], lw=0.42, alpha=0.72)
    ax.tick_params(length=2.4, width=0.7)
    ax.axvspan(SETTLEMENT_ZONE_M[0], SETTLEMENT_ZONE_M[1], color=COLORS["zone"], zorder=0)
    ax.axhline(0.0, color="#5A5A5A", lw=0.55)


def _trace_label(row: dict) -> str:
    if row["family"] == "baseline":
        return "baseline"
    if row["family"] == "cosine":
        return f"cos {row['amplitude_mm']:.0f} mm"
    return f"kink {row['amplitude_mm']:.0f} mm"


def _plot_all_cases(ax, cases: list[dict], y_key: str, show_legend: bool = False):
    family_max = {}
    for family in ("cosine", "kink"):
        vals = [row["amplitude_mm"] for row in cases if row["family"] == family]
        family_max[family] = max(vals) if vals else 1.0

    for row in cases:
        family = row["family"]
        if family == "baseline":
            color = COLORS["baseline"]
            alpha = 0.86
            lw = 0.85
            zorder = 2
        else:
            color = COLORS[family]
            alpha = 0.25 + 0.72 * row["amplitude_mm"] / family_max[family]
            lw = 0.62 + 0.55 * row["amplitude_mm"] / family_max[family]
            zorder = 3 if family == "cosine" else 4
        ax.plot(row["distance_m"], row[y_key], color=color, alpha=alpha, lw=lw, label=_trace_label(row), zorder=zorder)
    if show_legend:
        ax.legend(loc="upper right", ncol=2, handlelength=1.3, columnspacing=0.8)
    _style_axis(ax)


def make_figure(scans: dict[str, list[dict]]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    row_defs = [
        ("vertical_irre_mm", "Vertical irregularity (mm)", "Vertical track irregularity"),
        ("lateral_irre_mm", "Lateral irregularity (mm)", "Lateral track irregularity"),
        ("carbody_ay_mps2", "Carbody $a_y$ (m s$^{-2}$)", "Carbody lateral acceleration"),
        ("carbody_az_mps2", "Carbody $a_z$ (m s$^{-2}$)", "Carbody vertical acceleration"),
    ]
    scan_names = list(scans.keys())

    fig, axes = plt.subplots(4, 2, figsize=(7.5, 8.6), sharex=False)
    for col, scan_name in enumerate(scan_names):
        cases = scans[scan_name]
        xmin = min(float(np.nanmin(item["distance_m"])) for item in cases)
        xmax = max(float(np.nanmax(item["distance_m"])) for item in cases)
        for row_idx, (y_key, ylabel, row_title) in enumerate(row_defs):
            ax = axes[row_idx, col]
            _plot_all_cases(ax, cases, y_key, show_legend=False)
            ax.set_xlim(xmin, xmax)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row_idx == len(row_defs) - 1:
                ax.set_xlabel("Distance from valid start (m)")
            else:
                ax.set_xlabel("")
            if row_idx == 0:
                ax.set_title(scan_name, loc="left", pad=4, fontweight="bold")
            ax.text(0.01, 0.88, row_title, transform=ax.transAxes, fontsize=6.5, color="#333333", va="top")

    # Stable row limits make the two experimental backgrounds directly comparable.
    for ax in axes[0, :]:
        ax.set_ylim(-38, 4)
    for ax in axes[1, :]:
        ax.set_ylim(-2.8, 2.8)
    for ax in axes[2, :]:
        ax.set_ylim(-0.035, 0.035)
    axes[3, 0].set_ylim(-0.95, 1.38)
    axes[3, 1].set_ylim(-0.95, 1.38)

    for ax, letter in zip(axes.flat, "abcdefgh"):
        ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=8.5, va="top")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.58, 0.955), ncol=5, handlelength=1.4, columnspacing=0.9)
    fig.suptitle("Full-mileage irregularity inputs and carbody acceleration responses", x=0.02, y=0.99, ha="left", fontsize=9.5, fontweight="bold")
    fig.text(
        0.02,
        0.006,
        "Vertical/lateral irregularity is the left-right rail average; shaded band marks the imposed 20 m transition settlement zone.",
        fontsize=6.4,
        color="#444444",
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.90, bottom=0.075, wspace=0.22, hspace=0.24)

    base = OUT_DIR / "full_mileage_irregularity_and_carbody_acceleration"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return base.with_suffix(".png")


def export_source_data(scans: dict[str, list[dict]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    trace_rows = []
    for scan_cases in scans.values():
        for row in scan_cases:
            metric_rows.append({
                "scan": row["scan"],
                "case_id": row["case_id"],
                "family": row["family"],
                "amplitude_mm": row["amplitude_mm"],
                "peak_full_carbody_ay_mps2": row["peak_full_carbody_ay_mps2"],
                "peak_full_carbody_az_mps2": row["peak_full_carbody_az_mps2"],
                "rms_full_carbody_ay_mps2": row["rms_full_carbody_ay_mps2"],
                "rms_full_carbody_az_mps2": row["rms_full_carbody_az_mps2"],
                "result_npz": str(row["path"]),
            })
            for i in range(len(row["distance_m"])):
                trace_rows.append({
                    "scan": row["scan"],
                    "case_id": row["case_id"],
                    "family": row["family"],
                    "amplitude_mm": row["amplitude_mm"],
                    "distance_m": row["distance_m"][i],
                    "vertical_irre_mm": row["vertical_irre_mm"][i],
                    "lateral_irre_mm": row["lateral_irre_mm"][i],
                    "carbody_ay_mps2": row["carbody_ay_mps2"][i],
                    "carbody_az_mps2": row["carbody_az_mps2"][i],
                })
    pd.DataFrame(metric_rows).to_csv(OUT_DIR / "full_mileage_input_response_metrics.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(OUT_DIR / "full_mileage_input_response_traces.csv", index=False)


def main() -> None:
    scans = load_all()
    export_source_data(scans)
    figure = make_figure(scans)
    print(f"Saved figure: {figure}")
    print(f"Saved source data: {OUT_DIR}")


if __name__ == "__main__":
    main()
