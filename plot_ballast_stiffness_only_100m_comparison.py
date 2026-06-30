from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = PROJECT_ROOT / "results" / "ballast_condition_stiffness_only_100m_scan"
DEFAULT_OUT = DEFAULT_ROOT / "_comparison" / "publication_ballast_stiffness_only_100m"
G = 9.80665


COLORS = {
    "baseline": "#111111",
    "loose": "#D55E00",
    "harden": "#0072B2",
    "neutral": "#777777",
    "vertical": "#009E73",
    "lateral": "#CC79A7",
    "force": "#A6611A",
}


def set_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def relocate_result_npz_paths(summary: pd.DataFrame, result_root: Path) -> pd.DataFrame:
    df = summary.copy()
    root = result_root.resolve()

    def relocate(npz_path: str | Path) -> str:
        path = Path(npz_path)
        if path.exists():
            return str(path)
        parts = path.parts
        if root.name in parts:
            idx = parts.index(root.name)
            candidate = root.joinpath(*parts[idx + 1 :])
            if candidate.exists():
                return str(candidate)
        return str(path)

    df["source_npz"] = df["source_npz"].map(relocate)
    missing = [path for path in df["source_npz"].map(Path) if not path.exists()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:3])
        raise FileNotFoundError(f"Missing simulation_result.npz files after path relocation:\n{preview}")
    return df


def load_summary(result_root: Path) -> pd.DataFrame:
    path = result_root / "_comparison" / "ballast_stiffness_only_full_summary.csv"
    df = pd.read_csv(path)
    df = relocate_result_npz_paths(df, result_root)
    df["group"] = np.select(
        [df["kind"].eq("baseline"), df["eta_k"].lt(1), df["eta_k"].gt(1)],
        ["baseline", "loose", "harden"],
        default="neutral",
    )
    return df.sort_values(["speed_kmh", "eta_k", "kind"]).reset_index(drop=True)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def trace_from_row(row: pd.Series) -> pd.DataFrame:
    data = load_npz(row["source_npz"])
    distance = np.asarray(data.get("Irre_distance_m", data.get("Track_rel_mileage_m")), dtype=float)
    vertical_irre = 0.5 * (
        np.asarray(data.get("Irre_bz_L_ref", np.zeros_like(distance)), dtype=float)
        + np.asarray(data.get("Irre_bz_R_ref", np.zeros_like(distance)), dtype=float)
    ) * 1000.0
    lateral_irre = 0.5 * (
        np.asarray(data.get("Irre_by_L_ref", np.zeros_like(distance)), dtype=float)
        + np.asarray(data.get("Irre_by_R_ref", np.zeros_like(distance)), dtype=float)
    ) * 1000.0
    accel = np.asarray(data["A"], dtype=float)
    x = np.asarray(data["X"], dtype=float)
    total_v = np.asarray(data["TotalVerticalForce"], dtype=float)
    eta_k = 0.5 * (
        np.asarray(data.get("Stiffness_eta_k_L_ref", np.ones_like(distance)), dtype=float)
        + np.asarray(data.get("Stiffness_eta_k_R_ref", np.ones_like(distance)), dtype=float)
    )
    mask = np.maximum(
        np.asarray(data.get("Stiffness_irregularity_mask_L", np.zeros_like(distance)), dtype=int),
        np.asarray(data.get("Stiffness_irregularity_mask_R", np.zeros_like(distance)), dtype=int),
    )
    return pd.DataFrame(
        {
            "case_id": row["case_id"],
            "group": row["group"],
            "eta_k": row["eta_k"],
            "speed_kmh": row["speed_kmh"],
            "distance_m": distance,
            "vertical_irregularity_mm": vertical_irre,
            "lateral_irregularity_mm": lateral_irre,
            "stiffness_eta_k": eta_k,
            "stiffness_mask": mask,
            "carbody_az_mps2": accel[:, 1],
            "carbody_z_mm": x[:, 1] * 1000.0,
            "bogie_az_mean_mps2": 0.5 * (accel[:, 6] + accel[:, 11]),
            "wheelrail_force_peak_kn": np.nanmax(total_v, axis=1) / 1000.0,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.105, 1.055, label, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="left")


def shade_window(ax: plt.Axes, start_m: float, end_m: float) -> None:
    ax.axvspan(start_m, end_m, color="#f0c44f", alpha=0.18, lw=0)
    ax.axvline(start_m, color="#a97b00", lw=0.6, ls="--", alpha=0.75)
    ax.axvline(end_m, color="#a97b00", lw=0.45, ls=":", alpha=0.55)


def find_row(df: pd.DataFrame, case_id: str) -> pd.Series:
    hit = df.loc[df["case_id"] == case_id]
    if hit.empty:
        raise ValueError(f"Case not found: {case_id}")
    return hit.iloc[0]


def representative_case_ids(df: pd.DataFrame) -> list[str]:
    candidates = [
        "case_00_seed20260627_baseline_v215",
        "case_01_seed20260627_eta0p1_count167_v215",
        "case_02_seed20260627_eta0p2_count167_v215",
        "case_06_seed20260627_eta5_count167_v215",
        "case_07_seed20260627_eta10_count167_v215",
    ]
    available = set(df["case_id"])
    return [case_id for case_id in candidates if case_id in available]


def build_label(row: pd.Series) -> str:
    if row["group"] == "baseline":
        return "baseline"
    prefix = "loose" if row["eta_k"] < 1 else "harden"
    return rf"{prefix}, $\eta_k$={row['eta_k']:g}"


def plot_input_profile(ax: plt.Axes, traces: dict[str, pd.DataFrame], start_m: float, end_m: float) -> pd.DataFrame:
    baseline = traces["case_00_seed20260627_baseline_v215"]
    loose = traces["case_02_seed20260627_eta0p2_count167_v215"]
    active = baseline.loc[
        (baseline["vertical_irregularity_mm"].abs() + baseline["lateral_irregularity_mm"].abs()) > 1e-6,
        "distance_m",
    ]
    x_min = max(float(baseline["distance_m"].min()), float(active.min()) - 5.0) if not active.empty else start_m - 35.0
    x_max = min(float(baseline["distance_m"].max()), end_m + 75.0)
    ax.plot(baseline["distance_m"], baseline["vertical_irregularity_mm"], color=COLORS["vertical"], lw=0.65, label="vertical irregularity")
    ax.plot(baseline["distance_m"], baseline["lateral_irregularity_mm"], color=COLORS["lateral"], lw=0.65, label="lateral irregularity")
    shade_window(ax, start_m, end_m)
    ax.set_xlim(x_min, x_max)
    ax.set_ylabel("Irregularity (mm)")
    ax.set_title("Same random irregularity with a 100 m stiffness-only ballast window", loc="left")
    ax.legend(loc="upper right", frameon=False, ncol=2, handlelength=1.5, columnspacing=0.8)

    ax2 = ax.twinx()
    ax2.plot(loose["distance_m"], loose["stiffness_eta_k"], color="#4D4D4D", lw=0.75, ls="-.", label=r"$\eta_k$ profile")
    ax2.set_ylim(-0.2, 1.25)
    ax2.set_ylabel(r"$\eta_k$")
    ax2.spines["top"].set_visible(False)
    ax2.legend(loc="lower right", frameon=False, handlelength=1.5)
    return loose.loc[loose["distance_m"].between(x_min, x_max), ["distance_m", "stiffness_eta_k", "stiffness_mask"]].copy()


def plot_local_accel_delta(
    ax: plt.Axes,
    rows: pd.DataFrame,
    traces: dict[str, pd.DataFrame],
    start_m: float,
    end_m: float,
) -> pd.DataFrame:
    baseline = traces["case_00_seed20260627_baseline_v215"].set_index("distance_m")["carbody_az_mps2"]
    source_rows = []
    window = (start_m - 20.0, end_m + 20.0)
    for case_id in representative_case_ids(rows):
        if case_id == "case_00_seed20260627_baseline_v215":
            continue
        row = find_row(rows, case_id)
        trace = traces[case_id].copy()
        base = np.interp(trace["distance_m"], baseline.index.to_numpy(), baseline.to_numpy())
        trace["delta_carbody_az_mps2"] = trace["carbody_az_mps2"].to_numpy() - base
        mask = trace["distance_m"].between(*window)
        color = COLORS["loose"] if row["eta_k"] < 1 else COLORS["harden"]
        alpha = 0.95 if row["eta_k"] in {0.1, 10.0} else 0.75
        lw = 1.05 if row["eta_k"] in {0.1, 10.0} else 0.8
        ax.plot(
            trace.loc[mask, "distance_m"],
            trace.loc[mask, "delta_carbody_az_mps2"],
            color=color,
            alpha=alpha,
            lw=lw,
            label=build_label(row),
        )
        source_rows.append(trace.loc[mask, ["case_id", "eta_k", "speed_kmh", "distance_m", "delta_carbody_az_mps2"]])
    shade_window(ax, start_m, end_m)
    ax.axhline(0, color="#333333", lw=0.45, alpha=0.65)
    ax.set_xlim(*window)
    ax.set_ylabel(r"$\Delta$ carbody $a_z$ (m s$^{-2}$)")
    ax.set_title("Local carbody-acceleration perturbation relative to the no-defect baseline", loc="left")
    ax.legend(loc="upper right", frameon=False, ncol=2, handlelength=1.4, columnspacing=0.7)
    return pd.concat(source_rows, ignore_index=True)


def plot_local_displacement_delta(
    ax: plt.Axes,
    rows: pd.DataFrame,
    traces: dict[str, pd.DataFrame],
    start_m: float,
    end_m: float,
) -> pd.DataFrame:
    baseline = traces["case_00_seed20260627_baseline_v215"].set_index("distance_m")["carbody_z_mm"]
    source_rows = []
    window = (start_m - 20.0, end_m + 20.0)
    for case_id in representative_case_ids(rows):
        if case_id == "case_00_seed20260627_baseline_v215":
            continue
        row = find_row(rows, case_id)
        trace = traces[case_id].copy()
        base = np.interp(trace["distance_m"], baseline.index.to_numpy(), baseline.to_numpy())
        trace["delta_carbody_z_mm"] = trace["carbody_z_mm"].to_numpy() - base
        mask = trace["distance_m"].between(*window)
        color = COLORS["loose"] if row["eta_k"] < 1 else COLORS["harden"]
        alpha = 0.95 if row["eta_k"] in {0.1, 10.0} else 0.75
        lw = 1.05 if row["eta_k"] in {0.1, 10.0} else 0.8
        ax.plot(
            trace.loc[mask, "distance_m"],
            trace.loc[mask, "delta_carbody_z_mm"],
            color=color,
            alpha=alpha,
            lw=lw,
            label=build_label(row),
        )
        source_rows.append(trace.loc[mask, ["case_id", "eta_k", "speed_kmh", "distance_m", "delta_carbody_z_mm"]])
    shade_window(ax, start_m, end_m)
    ax.axhline(0, color="#333333", lw=0.45, alpha=0.65)
    ax.set_xlim(*window)
    ax.set_ylabel(r"$\Delta$ carbody $z$ (mm)")
    ax.set_title("Local carbody vertical-displacement perturbation relative to the no-defect baseline", loc="left")
    ax.legend(loc="upper right", frameon=False, ncol=2, handlelength=1.4, columnspacing=0.7)
    return pd.concat(source_rows, ignore_index=True)


def plot_eta_metric_changes(ax: plt.Axes, rows: pd.DataFrame) -> pd.DataFrame:
    scan = rows.loc[(rows["speed_kmh"] == 215) & rows["kind"].ne("baseline")].copy()
    scan = scan.sort_values("eta_k")
    metrics = [
        ("carbody_z_rms_delta_pct", "carbody displacement", "#34495E", "o"),
        ("bogie_z_rms_delta_pct", "bogie displacement", "#009E73", "s"),
        ("carbody_az_rms_delta_pct", "carbody acceleration", "#7B3294", "^"),
    ]
    for col, label, color, marker in metrics:
        ax.plot(scan["eta_k"], scan[col], marker=marker, ms=3.7, lw=1.0, color=color, label=label)
    ax.axhline(0, color="#333333", lw=0.5, alpha=0.75)
    ax.axvline(1.0, color="#777777", lw=0.55, ls=":", alpha=0.75)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax.get_xaxis().set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax.set_xlabel(r"Stiffness multiplier $\eta_k$ ($\eta_c$=1)")
    ax.set_ylabel("RMS change vs speed-matched baseline (%)")
    ax.set_title("Stiffness softening mainly amplifies displacement response", loc="left")
    ax.legend(loc="upper right", frameon=False, ncol=1)
    return scan[["case_id", "eta_k", "speed_kmh"] + [m[0] for m in metrics]].copy()


def plot_force_peak(ax: plt.Axes, rows: pd.DataFrame) -> pd.DataFrame:
    scan = rows.loc[(rows["speed_kmh"] == 215) & rows["kind"].ne("baseline")].copy()
    scan = scan.sort_values("eta_k")
    colors = [COLORS["loose"] if eta < 1 else COLORS["neutral"] if eta == 1 else COLORS["harden"] for eta in scan["eta_k"]]
    y = np.arange(len(scan))
    ax.barh(y, scan["wr_max_force_peak_delta_pct"], color=colors, height=0.68, edgecolor="white", linewidth=0.35)
    ax.axvline(0, color="#333333", lw=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels([rf"$\eta_k$={eta:g}" for eta in scan["eta_k"]])
    ax.invert_yaxis()
    ax.set_xlabel("Peak wheel-rail force change (%)")
    ax.set_title("Peak wheel-rail force changes remain small relative to vibration metrics", loc="left")
    ax.margins(y=0.04)
    return scan[["case_id", "eta_k", "speed_kmh", "wr_max_force_peak", "wr_max_force_peak_baseline", "wr_max_force_peak_delta_pct"]].copy()


def write_source_data(
    out_dir: Path,
    summary: pd.DataFrame,
    input_profile: pd.DataFrame,
    local_delta: pd.DataFrame,
    local_displacement_delta: pd.DataFrame,
    eta_metrics: pd.DataFrame,
    force_peak: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "source_data_metric_summary.csv", index=False)
    input_profile.to_csv(out_dir / "source_data_input_stiffness_profile.csv", index=False)
    local_delta.to_csv(out_dir / "source_data_local_accel_delta.csv", index=False)
    local_displacement_delta.to_csv(out_dir / "source_data_local_displacement_delta.csv", index=False)
    eta_metrics.to_csv(out_dir / "source_data_eta_metric_changes.csv", index=False)
    force_peak.to_csv(out_dir / "source_data_force_peak_changes.csv", index=False)


def save_outputs(fig: plt.Figure, out_dir: Path, base_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / base_name
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def make_figure(result_root: Path, out_dir: Path) -> Path:
    set_nature_style()
    summary = load_summary(result_root)
    selected = representative_case_ids(summary)
    traces = {case_id: trace_from_row(find_row(summary, case_id)) for case_id in selected}
    start_m = float(summary["span_start_m"].min())
    end_m = float(summary["span_end_m"].max())

    fig = plt.figure(figsize=(7.2, 8.75), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=4,
        ncols=2,
        height_ratios=[0.78, 0.82, 0.82, 1.65],
        width_ratios=[1.0, 1.0],
        hspace=0.58,
        wspace=0.35,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :])
    ax_c = fig.add_subplot(gs[2, :])
    ax_d = fig.add_subplot(gs[3, 0])
    ax_e = fig.add_subplot(gs[3, 1])

    input_profile = plot_input_profile(ax_a, traces, start_m, end_m)
    local_delta = plot_local_accel_delta(ax_b, summary, traces, start_m, end_m)
    local_displacement_delta = plot_local_displacement_delta(ax_c, summary, traces, start_m, end_m)
    eta_metrics = plot_eta_metric_changes(ax_d, summary)
    force_peak = plot_force_peak(ax_e, summary)

    for label, ax in zip(["a", "b", "c", "d", "e"], [ax_a, ax_b, ax_c, ax_d, ax_e]):
        panel_label(ax, label)
        grid_axis = "x" if ax in [ax_e] else "y"
        ax.grid(axis=grid_axis, color="#d8d8d8", lw=0.35, alpha=0.7)
    ax_c.set_xlabel("Distance from simulation start (m)")

    caption = (
        "Fixed random irregularity; the saved stiffness-mask window spans about 100 m and 167 sleepers "
        r"($\eta_c$=1.0 for all defect cases)."
    )
    fig.text(0.012, 0.012, caption, ha="left", va="bottom", fontsize=6.4, color="#333333")
    fig.subplots_adjust(left=0.125, right=0.988, top=0.975, bottom=0.08)

    save_outputs(fig, out_dir, "ballast_stiffness_only_100m_comparison")
    plt.close(fig)
    write_source_data(out_dir, summary, input_profile, local_delta, local_displacement_delta, eta_metrics, force_peak)
    return out_dir / "ballast_stiffness_only_100m_comparison.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publication comparison figure for 100 m stiffness-only ballast scan.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT, help="Root directory of the scan results.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output directory for figure and source data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    png = make_figure(args.result_root, args.out_dir)
    print(f"Saved figure preview: {png}")


if __name__ == "__main__":
    main()
