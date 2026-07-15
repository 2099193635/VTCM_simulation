from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "results" / "stiffness_four_mechanism_comparison"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "_comparison"
G = 9.80665
LOCAL_WINDOW_M = (160.0, 340.0)
DEFECT_WINDOW_M = (200.0, 300.0)


@dataclass(frozen=True)
class CaseStyle:
    key: str
    folder_token: str
    label: str
    short_label: str
    color: str


CASES = (
    CaseStyle("baseline", "case_00_baseline", "baseline", "base", "#4D4D4D"),
    CaseStyle("fastener_failure_local", "case_01_fastener_failure_local", "fastener loss", "fastener", "#0072B2"),
    CaseStyle("sleeper_void_local", "case_02_sleeper_void_local", "sleeper void", "sleeper", "#009E73"),
    CaseStyle("ballast_softening_100m", "case_03_ballast_softening_100m", "ballast softening", "ballast", "#D55E00"),
    CaseStyle("subgrade_weakening_100m", "case_04_subgrade_weakening_100m", "subgrade weakening", "subgrade", "#CC79A7"),
)


CHANNELS = (
    ("carbody_az_millig", r"Carbody vertical $\Delta a_z$ (10$^{-3}$ g)", "Carbody vertical"),
    ("bogie_az_millig", r"Bogie vertical $\Delta a_z$ (10$^{-3}$ g)", "Bogie vertical"),
    ("carbody_ay_millig", r"Carbody lateral $\Delta a_y$ (10$^{-3}$ g)", "Carbody lateral"),
    ("bogie_ay_millig", r"Bogie lateral $\Delta a_y$ (10$^{-3}$ g)", "Bogie lateral"),
)


def set_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def find_result_file(result_root: Path, token: str) -> Path:
    matches = sorted(result_root.glob(f"*{token}*/files/simulation_result.npz"))
    if not matches:
        raise FileNotFoundError(f"No simulation_result.npz found under {result_root} for token {token!r}")
    return matches[-1]


def load_case(result_root: Path, style: CaseStyle) -> dict[str, np.ndarray | str | Path]:
    path = find_result_file(result_root, style.folder_token)
    with np.load(path, allow_pickle=True) as data:
        accel = np.asarray(data["A"], dtype=float)
        distance = np.asarray(data["Track_rel_mileage_m"], dtype=float)

    def col(index: int) -> np.ndarray:
        return accel[:, index] if accel.shape[1] > index else np.zeros(accel.shape[0], dtype=float)

    return {
        "key": style.key,
        "label": style.label,
        "short_label": style.short_label,
        "color": style.color,
        "path": path,
        "distance_m": distance,
        "carbody_ay_millig": col(0) / G * 1000.0,
        "carbody_az_millig": col(1) / G * 1000.0,
        "bogie_ay_millig": 0.5 * (col(5) + col(10)) / G * 1000.0,
        "bogie_az_millig": 0.5 * (col(6) + col(11)) / G * 1000.0,
    }


def window_mask(distance: np.ndarray, window_m: tuple[float, float]) -> np.ndarray:
    return (distance >= window_m[0]) & (distance <= window_m[1])


def interp_baseline(
    baseline: dict[str, np.ndarray | str | Path],
    distance: np.ndarray,
    channel: str,
) -> np.ndarray:
    base_distance = np.asarray(baseline["distance_m"], dtype=float)
    base_signal = np.asarray(baseline[channel], dtype=float)
    return np.interp(distance, base_distance, base_signal)


def rms(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.nanmean(values**2)))


def peak_abs(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanmax(np.abs(values)))


def draw_defect_window(ax: plt.Axes) -> None:
    ax.axvspan(*DEFECT_WINDOW_M, color="#F0C44F", alpha=0.16, lw=0)
    ax.axvline(DEFECT_WINDOW_M[0], color="#9C6B00", lw=0.55, ls="--", alpha=0.7)
    ax.axvline(DEFECT_WINDOW_M[1], color="#9C6B00", lw=0.45, ls=":", alpha=0.65)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.115, 1.055, label, transform=ax.transAxes, fontsize=8.8, fontweight="bold", ha="left", va="bottom")


def build_sources(cases: list[dict[str, np.ndarray | str | Path]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = next(c for c in cases if c["key"] == "baseline")
    rows = []
    metrics = []

    for case in cases:
        distance = np.asarray(case["distance_m"], dtype=float)
        mask = window_mask(distance, LOCAL_WINDOW_M)
        trace = {
            "case": str(case["key"]),
            "label": str(case["label"]),
            "distance_m": distance[mask],
        }
        metric_row = {
            "case": str(case["key"]),
            "label": str(case["label"]),
            "source_npz": str(case["path"]),
        }
        for channel, _, _ in CHANNELS:
            signal = np.asarray(case[channel], dtype=float)
            if case["key"] == "baseline":
                delta = np.zeros_like(signal)
            else:
                delta = signal - interp_baseline(baseline, distance, channel)
            trace[channel] = signal[mask]
            trace[f"delta_{channel}"] = delta[mask]
            metric_row[f"{channel}_rms_local_millig"] = rms(signal[mask])
            metric_row[f"{channel}_peak_local_millig"] = peak_abs(signal[mask])
            metric_row[f"delta_{channel}_rms_local_millig"] = rms(delta[mask])
            metric_row[f"delta_{channel}_peak_local_millig"] = peak_abs(delta[mask])

        rows.append(pd.DataFrame(trace))
        metrics.append(metric_row)

    metrics_df = pd.DataFrame(metrics)
    baseline_metrics = metrics_df.loc[metrics_df["case"] == "baseline"].iloc[0]
    for channel, _, _ in CHANNELS:
        base_rms = float(baseline_metrics[f"{channel}_rms_local_millig"])
        base_peak = float(baseline_metrics[f"{channel}_peak_local_millig"])
        metrics_df[f"{channel}_rms_delta_pct"] = (
            metrics_df[f"{channel}_rms_local_millig"] / base_rms - 1.0
        ) * 100.0
        metrics_df[f"{channel}_peak_delta_pct"] = (
            metrics_df[f"{channel}_peak_local_millig"] / base_peak - 1.0
        ) * 100.0
    return pd.concat(rows, ignore_index=True), metrics_df


def plot_delta_traces(ax: plt.Axes, traces: pd.DataFrame, cases: list[dict[str, np.ndarray | str | Path]], channel: str) -> None:
    draw_defect_window(ax)
    for case in cases:
        if case["key"] == "baseline":
            continue
        part = traces.loc[traces["case"] == case["key"]]
        ax.plot(
            part["distance_m"],
            part[f"delta_{channel}"],
            lw=0.85,
            color=str(case["color"]),
            label=str(case["label"]),
            alpha=0.92,
        )
    ax.axhline(0.0, color="#333333", lw=0.45, alpha=0.72)
    ax.set_xlim(*LOCAL_WINDOW_M)
    ax.grid(axis="y", color="#D8D8D8", lw=0.32, alpha=0.65)


def plot_rms_bars(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    plot_df = metrics.loc[metrics["case"] != "baseline"].copy()
    x = np.arange(len(plot_df))
    width = 0.18
    bar_specs = [
        ("carbody_az_millig_rms_delta_pct", r"carbody $a_z$", "#6A3D9A"),
        ("bogie_az_millig_rms_delta_pct", r"bogie $a_z$", "#009E73"),
        ("carbody_ay_millig_rms_delta_pct", r"carbody $a_y$", "#A6611A"),
        ("bogie_ay_millig_rms_delta_pct", r"bogie $a_y$", "#4C78A8"),
    ]
    for idx, (column, label, color) in enumerate(bar_specs):
        ax.bar(x + (idx - 1.5) * width, plot_df[column], width=width, color=color, label=label)
    ax.axhline(0.0, color="#333333", lw=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["short_label"] if "short_label" in plot_df else plot_df["label"], rotation=0)
    ax.set_ylabel("RMS change vs baseline (%)")
    ax.set_title("Local acceleration RMS response", loc="left")
    ax.grid(axis="y", color="#D8D8D8", lw=0.32, alpha=0.65)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False, columnspacing=0.95, handlelength=1.1)


def save_figure(fig: plt.Figure, out_dir: Path, base_name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / base_name
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    return base.with_suffix(".png")


def make_figure(result_root: Path, out_dir: Path) -> Path:
    set_nature_style()
    cases = [load_case(result_root, style) for style in CASES]
    traces, metrics = build_sources(cases)

    metrics["short_label"] = [style.short_label for style in CASES]
    traces.to_csv(out_dir / "stiffness_four_acceleration_traces.csv", index=False)
    metrics.to_csv(out_dir / "stiffness_four_acceleration_metrics.csv", index=False)
    metrics.to_json(out_dir / "stiffness_four_acceleration_metrics.json", orient="records", indent=2)

    fig = plt.figure(figsize=(7.25, 6.4), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.88], hspace=0.56, wspace=0.36)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    ax_bar = fig.add_subplot(gs[2, :])

    for label, ax, (channel, ylabel, title) in zip(["a", "b", "c", "d"], axes, CHANNELS):
        plot_delta_traces(ax, traces, cases, channel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        panel_label(ax, label)
    axes[2].set_xlabel("Distance from simulation start (m)")
    axes[3].set_xlabel("Distance from simulation start (m)")
    axes[0].legend(ncol=2, loc="upper right", frameon=False, handlelength=1.35, columnspacing=0.75)

    plot_rms_bars(ax_bar, metrics)
    panel_label(ax_bar, "e")

    fig.suptitle(
        "Carbody and bogie acceleration response under four stiffness-defect mechanisms",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=9.2,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.953,
        "Curves show local acceleration perturbation relative to the no-defect baseline; shaded band marks the 200-300 m defect zone.",
        ha="left",
        va="top",
        fontsize=6.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.115, right=0.99, top=0.905, bottom=0.085)
    png = save_figure(fig, out_dir, "stiffness_four_acceleration_comparison")
    plt.close(fig)
    return png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nature-style acceleration comparison for four stiffness-defect mechanisms.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT, help="Root directory of the sweep results.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for figures and source data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    png = make_figure(args.result_root, args.out_dir)
    print(f"Saved acceleration comparison: {png}")


if __name__ == "__main__":
    main()
