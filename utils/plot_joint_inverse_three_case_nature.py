from __future__ import annotations

import csv
import json
import re
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
    matches = [
        p
        for p in RESULT_ROOT.rglob("simulation_result.npz")
        if case_key in str(p)
    ]
    if not matches:
        raise FileNotFoundError(f"No simulation_result.npz found for {case_key}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_case(case_key: str, label: str, color: str) -> dict:
    path = find_case_npz(case_key)
    data = np.load(path, allow_pickle=True)
    distance = np.asarray(data["Irre_distance_m"], dtype=float)
    eta_l = np.asarray(data["Stiffness_eta_k_L_ref"], dtype=float)
    eta_r = np.asarray(data["Stiffness_eta_k_R_ref"], dtype=float)
    bz_l = np.asarray(data["Irre_bz_L_ref"], dtype=float)
    bz_r = np.asarray(data["Irre_bz_R_ref"], dtype=float)
    acc_z = np.asarray(data["A"][:, 1], dtype=float)
    mask = np.maximum(
        np.asarray(data["Stiffness_irregularity_mask_L"], dtype=int),
        np.asarray(data["Stiffness_irregularity_mask_R"], dtype=int),
    )
    return {
        "case_key": case_key,
        "label": label,
        "color": color,
        "path": str(path),
        "distance_m": distance,
        "eta_k_mean": 0.5 * (eta_l + eta_r),
        "geometry_bz_mean_mm": 500.0 * (bz_l + bz_r),
        "carbody_az_mps2": acc_z,
        "mask": mask,
    }


def contiguous_mask_span(distance: np.ndarray, mask: np.ndarray) -> tuple[float, float] | None:
    idx = np.flatnonzero(mask > 0)
    if idx.size == 0:
        return None
    return float(distance[idx[0]]), float(distance[idx[-1]])


def write_source_data(cases: list[dict], out_csv: Path) -> None:
    fields = [
        "case",
        "distance_m",
        "eta_k_mean",
        "geometry_bz_mean_mm",
        "carbody_az_mps2",
        "stiffness_mask",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            for i, s in enumerate(case["distance_m"]):
                writer.writerow(
                    {
                        "case": case["label"],
                        "distance_m": f"{s:.6g}",
                        "eta_k_mean": f"{case['eta_k_mean'][i]:.8g}",
                        "geometry_bz_mean_mm": f"{case['geometry_bz_mean_mm'][i]:.8g}",
                        "carbody_az_mps2": f"{case['carbody_az_mps2'][i]:.8g}",
                        "stiffness_mask": int(case["mask"][i]),
                    }
                )


def label_line_end(ax: plt.Axes, x: np.ndarray, y: np.ndarray, text: str, color: str, dy: float = 0.0) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return
    xi = x[finite][-1]
    yi = y[finite][-1] + dy
    ax.text(
        xi,
        yi,
        " " + text,
        color=color,
        va="center",
        ha="left",
        fontsize=6.5,
        clip_on=False,
    )


def put_axis_legend(ax: plt.Axes, cases: list[dict]) -> None:
    handles = [
        mpl.lines.Line2D([0], [0], color=case["color"], lw=1.6, label=case["label"])
        for case in cases
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        ncol=3,
        bbox_to_anchor=(0.0, 1.08),
        handlelength=1.8,
        columnspacing=1.1,
        borderaxespad=0.0,
    )


def plot_figure(cases: list[dict], out_base: Path) -> None:
    configure_matplotlib()

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.05, 1.35], "hspace": 0.10},
        constrained_layout=True,
    )

    nonbaseline_spans = [
        contiguous_mask_span(c["distance_m"], c["mask"])
        for c in cases
        if "Baseline" not in c["label"]
    ]
    nonbaseline_spans = [span for span in nonbaseline_spans if span is not None]
    if nonbaseline_spans:
        defect_start = min(span[0] for span in nonbaseline_spans)
        defect_end = max(span[1] for span in nonbaseline_spans)
    else:
        defect_start = defect_end = None

    for ax in axes:
        if defect_start is not None:
            ax.axvspan(defect_start, defect_end, color="#D7D7D7", alpha=0.24, lw=0, zorder=0)
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)
        ax.tick_params(length=3, width=0.7)

    for case in cases:
        x = case["distance_m"]
        axes[0].plot(x, case["eta_k_mean"], color=case["color"], lw=1.45)
        axes[1].plot(x, case["geometry_bz_mean_mm"], color=case["color"], lw=1.0, alpha=0.82)
        axes[2].plot(x, case["carbody_az_mps2"], color=case["color"], lw=0.75, alpha=0.86)

    axes[0].set_ylabel("Stiffness\neta_k")
    axes[1].set_ylabel("Vertical track\nirregularity (mm)")
    axes[2].set_ylabel("Carbody vertical\nacceleration (m s$^{-2}$)")
    axes[2].set_xlabel("Distance from simulation start (m)")

    axes[0].set_ylim(-0.15, 5.55)
    axes[0].set_yticks([0.2, 1.0, 5.0])
    axes[0].set_yticklabels(["0.2", "1", "5"])

    geom_all = np.concatenate([c["geometry_bz_mean_mm"] for c in cases])
    g_lim = np.nanpercentile(np.abs(geom_all), 99.5)
    axes[1].set_ylim(-max(g_lim, 0.8) * 1.12, max(g_lim, 0.8) * 1.12)

    acc_all = np.concatenate([c["carbody_az_mps2"] for c in cases])
    a_lim = np.nanpercentile(np.abs(acc_all), 99.2)
    axes[2].set_ylim(-max(a_lim, 0.05) * 1.18, max(a_lim, 0.05) * 1.18)

    x_min = min(float(np.nanmin(c["distance_m"])) for c in cases)
    x_max = max(float(np.nanmax(c["distance_m"])) for c in cases)
    axes[2].set_xlim(x_min, x_max + 16.0)

    x_label = x_max + 1.2
    axes[0].text(x_label, 5.0, "Hardened eta_k=5", color="#B24A3F", va="center", ha="left", fontsize=6.5)
    axes[0].text(x_label, 1.0, "Baseline", color="#595959", va="center", ha="left", fontsize=6.5)
    axes[0].text(x_label, 0.2, "Loose eta_k=0.2", color="#3B6EA8", va="center", ha="left", fontsize=6.5)

    label_line_end(
        axes[1],
        cases[0]["distance_m"],
        cases[0]["geometry_bz_mean_mm"],
        "same seed geometry",
        "#333333",
        dy=0.0,
    )
    put_axis_legend(axes[2], cases)

    if defect_start is not None:
        axes[0].text(
            defect_start,
            axes[0].get_ylim()[1],
            f"  stiffness anomaly {defect_start:.0f}-{defect_end:.0f} m",
            ha="left",
            va="top",
            fontsize=6.4,
            color="#5A5A5A",
        )

    for panel, ax in zip(["a", "b", "c"], axes):
        ax.text(
            -0.075,
            1.02,
            panel,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=8.5,
            va="bottom",
            ha="left",
        )

    fig.suptitle(
        "Three-case comparison for joint stiffness-geometry inversion data",
        x=0.01,
        y=1.015,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )

    for suffix in [".png", ".pdf", ".svg", ".tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix in {".png", ".tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(out_base.with_suffix(suffix), **kwargs)
    plt.close(fig)


def summarize(cases: list[dict], out_json: Path) -> None:
    rows = []
    for case in cases:
        span = contiguous_mask_span(case["distance_m"], case["mask"])
        rows.append(
            {
                "case": case["label"],
                "source_npz": case["path"],
                "distance_min_m": float(np.nanmin(case["distance_m"])),
                "distance_max_m": float(np.nanmax(case["distance_m"])),
                "eta_k_min": float(np.nanmin(case["eta_k_mean"])),
                "eta_k_max": float(np.nanmax(case["eta_k_mean"])),
                "stiffness_anomaly_span_m": None if span is None else [span[0], span[1]],
                "geometry_bz_rms_mm": float(np.sqrt(np.nanmean(case["geometry_bz_mean_mm"] ** 2))),
                "carbody_az_rms_mps2": float(np.sqrt(np.nanmean(case["carbody_az_mps2"] ** 2))),
                "carbody_az_peak_abs_mps2": float(np.nanmax(np.abs(case["carbody_az_mps2"]))),
            }
        )
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [load_case(*case_info) for case_info in CASE_ORDER]
    out_base = OUT_DIR / "joint_inverse_three_case_nature_comparison"
    plot_figure(cases, out_base)
    write_source_data(cases, OUT_DIR / "joint_inverse_three_case_source_data.csv")
    summarize(cases, OUT_DIR / "joint_inverse_three_case_summary.json")

    print(f"Saved figure base: {out_base}")
    print(f"Saved source data: {OUT_DIR / 'joint_inverse_three_case_source_data.csv'}")
    print(f"Saved summary: {OUT_DIR / 'joint_inverse_three_case_summary.json'}")


if __name__ == "__main__":
    main()
