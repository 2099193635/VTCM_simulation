from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


G = 9.80665
DEFAULT_ROOT = Path("results/ballast_condition_random_scan")
DEFAULT_OUT = DEFAULT_ROOT / "_comparison" / "publication_ballast_condition_random"


PANEL_FONT = 7.0
LABEL_FONT = 7.2
TICK_FONT = 6.5
TITLE_FONT = 7.4


COLORS = {
    "baseline": "#4d4d4d",
    "harden": "#0072B2",
    "loose": "#D55E00",
    "neutral": "#6a6a6a",
    "vertical": "#009E73",
    "lateral": "#CC79A7",
}


def set_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": PANEL_FONT,
            "axes.labelsize": LABEL_FONT,
            "axes.titlesize": TITLE_FONT,
            "xtick.labelsize": TICK_FONT,
            "ytick.labelsize": TICK_FONT,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.nanmean(x * x)))


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def build_label(row: pd.Series) -> str:
    if row["group"] == "baseline":
        return "baseline"
    eta_k = f'{row["eta_k"]:g}'
    eta_c = f'{row["eta_c"]:g}'
    count = int(row["count"])
    prefix = "H" if row["group"] == "harden" else "L"
    return rf"{prefix}: $\eta_k$={eta_k}, $\eta_c$={eta_c}, n={count}"


def find_case(df: pd.DataFrame, case_id: str) -> pd.Series:
    hit = df.loc[df["case_id"] == case_id]
    if hit.empty:
        raise ValueError(f"Case not found: {case_id}")
    return hit.iloc[0]


def selected_cases(df: pd.DataFrame) -> list[str]:
    candidates = [
        "case_00_random_baseline",
        "case_02_harden_etaK5_etaC1_count5",
        "case_06_loose_etaK0p1_etaC1_count5",
        "case_13_harden_etaK5_etaC1_count10",
        "case_15_loose_etaK0p2_etaC1_count10",
    ]
    return [case_id for case_id in candidates if case_id in set(df["case_id"])]


def case_trace(row: pd.Series) -> pd.DataFrame:
    data = load_npz(row["result_npz"])
    distance = np.asarray(data.get("Irre_distance_m", data["Track_rel_mileage_m"]), dtype=float)
    vertical_irre = 0.5 * (
        np.asarray(data["Irre_bz_L_ref"], dtype=float) + np.asarray(data["Irre_bz_R_ref"], dtype=float)
    ) * 1000.0
    lateral_irre = 0.5 * (
        np.asarray(data["Irre_by_L_ref"], dtype=float) + np.asarray(data["Irre_by_R_ref"], dtype=float)
    ) * 1000.0
    accel = np.asarray(data["A"], dtype=float)
    ballast_force = np.asarray(data.get("Structure_defect_ballast_condition_FV_sum", np.zeros((len(distance), 2))), dtype=float)
    if ballast_force.ndim == 2:
        ballast_force_kn = np.nanmax(np.abs(ballast_force), axis=1) / 1000.0
    else:
        ballast_force_kn = np.abs(ballast_force) / 1000.0
    return pd.DataFrame(
        {
            "case_id": row["case_id"],
            "label": build_label(row),
            "group": row["group"],
            "distance_m": distance,
            "vertical_irregularity_mm": vertical_irre,
            "lateral_irregularity_mm": lateral_irre,
            "carbody_az_mps2": accel[:, 1],
            "carbody_ay_mps2": accel[:, 0],
            "ballast_condition_force_kn": ballast_force_kn,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.105,
        1.055,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def shade_defect(ax: plt.Axes, start_m: float = 200.0, count: int = 10, sleeper_spacing_m: float = 0.6) -> None:
    ax.axvspan(start_m, start_m + count * sleeper_spacing_m, color="#f0c44f", alpha=0.18, lw=0)
    ax.axvline(start_m, color="#a97b00", lw=0.6, ls="--", alpha=0.75)


def plot_irregularity(ax: plt.Axes, trace: pd.DataFrame) -> None:
    active = trace.loc[
        (trace["vertical_irregularity_mm"].abs() + trace["lateral_irregularity_mm"].abs()) > 1e-6,
        "distance_m",
    ]
    x_min = max(float(trace["distance_m"].min()), float(active.min()) - 5.0) if not active.empty else float(trace["distance_m"].min())
    x_max = float(trace["distance_m"].max())
    ax.plot(
        trace["distance_m"],
        trace["vertical_irregularity_mm"],
        color=COLORS["vertical"],
        lw=0.65,
        label="vertical irregularity",
    )
    ax.plot(
        trace["distance_m"],
        trace["lateral_irregularity_mm"],
        color=COLORS["lateral"],
        lw=0.65,
        label="lateral irregularity",
    )
    shade_defect(ax)
    ax.set_xlim(x_min, x_max)
    ax.set_ylabel("Irregularity (mm)")
    ax.set_title("Same fixed-seed random track irregularity is applied to all ballast-condition cases", loc="left")
    ax.legend(loc="upper right", frameon=False, ncol=2, handlelength=1.5, columnspacing=0.8)
    ax.text(201.5, ax.get_ylim()[1] * 0.78, "ballast state\nchanged here", color="#7a5c00", fontsize=6.3)


def plot_local_accel_delta(ax: plt.Axes, traces: dict[str, pd.DataFrame], baseline_id: str) -> pd.DataFrame:
    baseline = traces[baseline_id].set_index("distance_m")["carbody_az_mps2"]
    rows: list[dict[str, float | str]] = []
    window = (190.0, 220.0)
    for case_id, trace in traces.items():
        if case_id == baseline_id:
            continue
        delta = trace["carbody_az_mps2"].to_numpy() - baseline.to_numpy()
        mask = trace["distance_m"].between(*window).to_numpy()
        row_meta = trace.iloc[0]
        color = COLORS["harden"] if row_meta["group"] == "harden" else COLORS["loose"]
        alpha = 0.95 if "count10" in case_id or "etaK0p1" in case_id else 0.72
        lw = 1.0 if "count10" in case_id or "etaK0p1" in case_id else 0.75
        ax.plot(
            trace.loc[mask, "distance_m"],
            delta[mask],
            color=color,
            alpha=alpha,
            lw=lw,
            label=row_meta["label"],
        )
        rows.append(
            {
                "case_id": case_id,
                "label": row_meta["label"],
                "local_delta_az_rms_mps2": rms(delta[mask]),
                "local_delta_az_peak_abs_mps2": float(np.nanmax(np.abs(delta[mask]))),
            }
        )
    shade_defect(ax)
    ax.axhline(0, color="#333333", lw=0.45, alpha=0.65)
    ax.set_xlim(*window)
    ax.set_ylabel(r"$\Delta$ carbody $a_z$ (m s$^{-2}$)")
    ax.set_title("Local carbody-acceleration perturbation relative to the random-irregularity baseline", loc="left")
    ax.legend(loc="upper right", frameon=False, ncol=2, handlelength=1.4, columnspacing=0.7)
    return pd.DataFrame(rows)


def plot_global_metric_changes(ax: plt.Axes, analysis: pd.DataFrame) -> None:
    metrics = [
        ("carbody_az_rms_g_pct_vs_baseline", "carbody"),
        ("bogie1_az_rms_g_pct_vs_baseline", "bogie"),
        ("wheelset1_az_rms_g_pct_vs_baseline", "wheelset"),
    ]
    subset = analysis.loc[analysis["group"] != "baseline"].copy()
    subset["order"] = np.arange(len(subset))
    offsets = [-0.18, 0.0, 0.18]
    markers = ["o", "s", "^"]
    for (col, label), off, marker in zip(metrics, offsets, markers):
        for group, color in [("harden", COLORS["harden"]), ("loose", COLORS["loose"])]:
            part = subset.loc[subset["group"] == group]
            ax.scatter(
                part[col],
                part["order"] + off,
                s=19,
                marker=marker,
                facecolor=color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.9,
                label=f"{label}, {group}" if off == offsets[0] else None,
            )
    ax.axvline(0, color="#333333", lw=0.5)
    ax.set_yticks(subset["order"])
    labels = [f'{g[0].upper()} {ek:g}/{ec:g}, n={int(n)}' for g, ek, ec, n in zip(subset["group"], subset["eta_k"], subset["eta_c"], subset["count"])]
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Vertical RMS change (%)")
    ax.set_title("Full-mileage RMS metrics are weakly separated", loc="left")
    handles = [
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="white", markersize=4.5, label="carbody"),
        mpl.lines.Line2D([0], [0], marker="s", color="none", markerfacecolor="#777777", markeredgecolor="white", markersize=4.5, label="bogie"),
        mpl.lines.Line2D([0], [0], marker="^", color="none", markerfacecolor="#777777", markeredgecolor="white", markersize=4.5, label="wheelset"),
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["harden"], markeredgecolor="white", markersize=4.5, label="harden"),
        mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["loose"], markeredgecolor="white", markersize=4.5, label="loose"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=2, columnspacing=0.7, handletextpad=0.25)
    ax.margins(y=0.02)


def plot_force_peak(ax: plt.Axes, analysis: pd.DataFrame) -> None:
    subset = analysis.loc[analysis["group"] != "baseline"].copy()
    subset["y"] = np.arange(len(subset))
    colors = [COLORS["harden"] if g == "harden" else COLORS["loose"] for g in subset["group"]]
    ax.barh(
        subset["y"],
        subset["structure_defect_ballast_condition_force_peak_kn"],
        color=colors,
        height=0.68,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_yticks(subset["y"])
    labels = [f'{g[0].upper()} {ek:g}/{ec:g}, n={int(n)}' for g, ek, ec, n in zip(subset["group"], subset["eta_k"], subset["eta_c"], subset["count"])]
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Peak condition force (kN)")
    ax.set_title("Local ballast-state force is more discriminative", loc="left")
    ax.margins(y=0.02)


def write_source_data(out_dir: Path, analysis: pd.DataFrame, traces: dict[str, pd.DataFrame], local_summary: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = analysis.copy()
    metrics.to_csv(out_dir / "source_data_metric_summary.csv", index=False)
    trace_data = pd.concat(traces.values(), ignore_index=True)
    trace_data.to_csv(out_dir / "source_data_selected_traces.csv", index=False)
    local_summary.to_csv(out_dir / "source_data_local_delta_summary.csv", index=False)


def make_figure(result_root: Path, out_dir: Path) -> Path:
    set_nature_style()
    comparison = result_root / "_comparison"
    analysis = pd.read_csv(comparison / "ballast_condition_group_analysis.csv")
    summary = pd.read_csv(comparison / "sweep_response_summary.csv")
    analysis = analysis.merge(summary[["case_id", "result_npz", "note"]], on="case_id", how="left")
    analysis["label"] = analysis.apply(build_label, axis=1)

    keep = selected_cases(analysis)
    traces = {case_id: case_trace(find_case(analysis, case_id)) for case_id in keep}
    baseline_id = "case_00_random_baseline"
    if baseline_id not in traces:
        raise ValueError("Baseline trace is required for the comparison figure.")

    fig = plt.figure(figsize=(7.2, 7.25), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[0.85, 0.92, 1.95],
        width_ratios=[1.0, 1.0],
        hspace=0.55,
        wspace=0.35,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :])
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[2, 1])

    plot_irregularity(ax_a, traces[baseline_id])
    local_summary = plot_local_accel_delta(ax_b, traces, baseline_id)
    plot_global_metric_changes(ax_c, analysis)
    plot_force_peak(ax_d, analysis)

    for label, ax in zip(["a", "b", "c", "d"], [ax_a, ax_b, ax_c, ax_d]):
        add_panel_label(ax, label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        grid_axis = "x" if ax in [ax_c, ax_d] else "y"
        ax.grid(axis=grid_axis, color="#d8d8d8", lw=0.35, alpha=0.7)

    ax_b.set_xlabel("Distance from valid start (m)")
    ax_c.set_ylabel("Case")
    ax_d.tick_params(axis="y", labelleft=False)

    caption = (
        "Fixed random irregularity; ballast-state change starts at 200 m. "
        "H, hardened ballast; L, loosened ballast."
    )
    fig.text(0.012, 0.012, caption, ha="left", va="bottom", fontsize=6.4, color="#333333")
    fig.subplots_adjust(left=0.125, right=0.988, top=0.975, bottom=0.08)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "ballast_condition_random_comparison"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    write_source_data(out_dir, analysis, traces, local_summary)
    return base.with_suffix(".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a publication-grade comparison figure for the ballast-condition random-irregularity scan."
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT, help="Root directory of the scan results.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory for figure and source data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    png = make_figure(args.result_root, args.out_dir)
    print(f"Saved figure preview: {png}")


if __name__ == "__main__":
    main()
