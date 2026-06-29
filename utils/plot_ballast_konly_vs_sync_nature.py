from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "ballast_condition_stiffness_only_100m_scan" / "_comparison"

FAMILIES = {
    "stiffness_only": {
        "label": "Stiffness only, eta_c=1",
        "root": ROOT / "results" / "ballast_condition_stiffness_only_100m_scan",
        "color": "#3B6EA8",
        "patterns": {
            "baseline": "case_00_seed20260627_baseline_v215",
            "eta0p2": "case_02_seed20260627_eta0p2_count167_v215",
            "eta5": "case_06_seed20260627_eta5_count167_v215",
        },
    },
    "sync_eta": {
        "label": "Synchronized eta_k=eta_c",
        "root": ROOT / "results" / "ballast_condition_sync_eta_100m_scan",
        "color": "#B36A2E",
        "patterns": {
            "baseline": "case_00_random_baseline",
            "eta0p2": "case_02_eta0p2_count167",
            "eta5": "case_06_eta5_count167",
        },
    },
}

CONDITIONS = {
    "eta0p2": {"label": "Loose eta_k=0.2", "eta_k": 0.2},
    "eta5": {"label": "Hardened eta_k=5", "eta_k": 5.0},
}


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


def latest_npz(root: Path, pattern: str) -> Path:
    matches = [p for p in root.rglob("simulation_result.npz") if pattern in str(p)]
    if not matches:
        raise FileNotFoundError(f"No simulation_result.npz found under {root} for {pattern}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_case(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    distance = np.asarray(data["Irre_distance_m"], dtype=float)
    X = np.asarray(data["X"], dtype=float)
    A = np.asarray(data["A"], dtype=float)
    total_v = np.asarray(data["TotalVerticalForce"], dtype=float)
    if "Stiffness_irregularity_mask_L" in data.files and "Stiffness_irregularity_mask_R" in data.files:
        mask = np.maximum(
            np.asarray(data["Stiffness_irregularity_mask_L"], dtype=int),
            np.asarray(data["Stiffness_irregularity_mask_R"], dtype=int),
        )
    else:
        mask = np.zeros_like(distance, dtype=int)
    if "Stiffness_eta_k_L_ref" in data.files and "Stiffness_eta_k_R_ref" in data.files:
        eta_k = 0.5 * (
            np.asarray(data["Stiffness_eta_k_L_ref"], dtype=float)
            + np.asarray(data["Stiffness_eta_k_R_ref"], dtype=float)
        )
    else:
        eta_k = np.ones_like(distance)
    return {
        "path": str(path),
        "distance_m": distance,
        "eta_k": eta_k,
        "mask": mask,
        "carbody_az_mps2": A[:, 1],
        "carbody_z_mm": X[:, 1] * 1000.0,
        "bogie_az_mean_mps2": 0.5 * (A[:, 6] + A[:, 11]),
        "bogie_z_mean_mm": 0.5 * (X[:, 6] + X[:, 11]) * 1000.0,
        "wr_force_mean_kN": np.mean(total_v, axis=1) / 1000.0,
        "wr_force_max_kN": np.max(total_v, axis=1) / 1000.0,
    }


def load_all() -> dict:
    loaded = {}
    for family_key, family in FAMILIES.items():
        loaded[family_key] = {}
        for condition_key, pattern in family["patterns"].items():
            loaded[family_key][condition_key] = load_case(latest_npz(family["root"], pattern))
    return loaded


def defect_span(loaded: dict) -> tuple[float, float]:
    spans = []
    for family_cases in loaded.values():
        for condition_key in ("eta0p2", "eta5"):
            case = family_cases[condition_key]
            idx = np.flatnonzero(case["mask"] > 0)
            if idx.size:
                spans.append((case["distance_m"][idx[0]], case["distance_m"][idx[-1]]))
    if not spans:
        return (180.0, 280.0)
    return float(min(s[0] for s in spans)), float(max(s[1] for s in spans))


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(np.asarray(values, dtype=float) ** 2)))


def summarize_metrics(loaded: dict, span: tuple[float, float]) -> list[dict]:
    metrics = {
        "carbody_az_rms": ("carbody_az_mps2", "rms"),
        "carbody_z_rms": ("carbody_z_mm", "rms"),
        "bogie_az_rms": ("bogie_az_mean_mps2", "rms"),
        "bogie_z_rms": ("bogie_z_mean_mm", "rms"),
        "wr_mean_force_rms": ("wr_force_mean_kN", "rms"),
        "wr_max_force_peak": ("wr_force_max_kN", "peak"),
    }
    rows = []
    for family_key, family_cases in loaded.items():
        baseline = family_cases["baseline"]
        for condition_key, condition in CONDITIONS.items():
            case = family_cases[condition_key]
            selector = (case["distance_m"] >= span[0]) & (case["distance_m"] <= span[1])
            base_selector = (baseline["distance_m"] >= span[0]) & (baseline["distance_m"] <= span[1])
            row = {
                "family": family_key,
                "family_label": FAMILIES[family_key]["label"],
                "condition": condition_key,
                "condition_label": condition["label"],
                "eta_k": condition["eta_k"],
                "eta_c": 1.0 if family_key == "stiffness_only" else condition["eta_k"],
                "span_start_m": span[0],
                "span_end_m": span[1],
                "source_npz": case["path"],
                "baseline_npz": baseline["path"],
            }
            for metric_name, (signal_key, stat) in metrics.items():
                values = case[signal_key][selector]
                base_values = baseline[signal_key][base_selector]
                if stat == "rms":
                    value = rms(values)
                    base_value = rms(base_values)
                else:
                    value = float(np.nanmax(np.abs(values)))
                    base_value = float(np.nanmax(np.abs(base_values)))
                row[metric_name] = value
                row[f"{metric_name}_baseline"] = base_value
                row[f"{metric_name}_delta_pct"] = 100.0 * (value / base_value - 1.0)
            rows.append(row)
    return rows


def save_all(fig: plt.Figure, base: Path) -> None:
    for suffix in [".png", ".pdf", ".svg", ".tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if suffix in {".png", ".tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(base.with_suffix(suffix), **kwargs)
    plt.close(fig)


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(-0.12, 1.03, text, transform=ax.transAxes, fontweight="bold", fontsize=8.5, va="bottom")


def plot_metric_grid(rows: list[dict], out_base: Path) -> None:
    configure_matplotlib()
    metric_panels = [
        ("carbody_az_rms_delta_pct", "Carbody acceleration\nRMS change (%)"),
        ("carbody_z_rms_delta_pct", "Carbody displacement\nRMS change (%)"),
        ("bogie_az_rms_delta_pct", "Bogie acceleration\nRMS change (%)"),
        ("bogie_z_rms_delta_pct", "Bogie displacement\nRMS change (%)"),
        ("wr_mean_force_rms_delta_pct", "Mean wheel-rail force\nRMS change (%)"),
        ("wr_max_force_peak_delta_pct", "Peak wheel-rail force\nchange (%)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.6), constrained_layout=True)
    axes = axes.ravel()
    x = np.arange(len(CONDITIONS))
    width = 0.34
    offsets = {"stiffness_only": -width / 2, "sync_eta": width / 2}
    labels = [CONDITIONS[k]["label"].replace(" ", "\n", 1) for k in CONDITIONS]
    for idx, (ax, (metric, ylabel)) in enumerate(zip(axes, metric_panels)):
        for family_key, family in FAMILIES.items():
            values = []
            for condition_key in CONDITIONS:
                match = next(r for r in rows if r["family"] == family_key and r["condition"] == condition_key)
                values.append(match[metric])
            ax.bar(
                x + offsets[family_key],
                values,
                width=width,
                color=family["color"],
                alpha=0.86,
                label=family["label"],
            )
        ax.axhline(0, color="#333333", lw=0.7)
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        panel_label(ax, chr(ord("a") + idx))
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.22), ncol=2, borderaxespad=0)
    fig.suptitle(
        "Decoupling ballast stiffness from damping: response changes in the defect segment",
        x=0.01,
        y=1.04,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    save_all(fig, out_base)


def interpolate_to_base(case: dict, baseline: dict, key: str) -> np.ndarray:
    return np.interp(case["distance_m"], baseline["distance_m"], baseline[key])


def plot_delta_traces(loaded: dict, span: tuple[float, float], out_base: Path) -> None:
    configure_matplotlib()
    trace_metrics = [
        ("carbody_az_mps2", "Delta carbody acceleration\n(m s$^{-2}$)"),
        ("carbody_z_mm", "Delta carbody displacement\n(mm)"),
        ("bogie_az_mean_mps2", "Delta bogie acceleration\n(m s$^{-2}$)"),
        ("wr_force_max_kN", "Delta max wheel-rail force\n(kN)"),
    ]
    fig, axes = plt.subplots(
        len(trace_metrics),
        len(CONDITIONS),
        figsize=(7.3, 6.2),
        sharex=True,
        constrained_layout=True,
    )
    xlim = (max(0.0, span[0] - 30.0), span[1] + 55.0)
    for col, (condition_key, condition) in enumerate(CONDITIONS.items()):
        axes[0, col].set_title(condition["label"], fontsize=8)
        for row, (metric_key, ylabel) in enumerate(trace_metrics):
            ax = axes[row, col]
            ax.axvspan(span[0], span[1], color="#D7D7D7", alpha=0.24, lw=0)
            for family_key, family in FAMILIES.items():
                case = loaded[family_key][condition_key]
                baseline = loaded[family_key]["baseline"]
                delta = case[metric_key] - interpolate_to_base(case, baseline, metric_key)
                linestyle = "-" if family_key == "stiffness_only" else "--"
                ax.plot(
                    case["distance_m"],
                    delta,
                    color=family["color"],
                    lw=1.0,
                    alpha=0.92,
                    linestyle=linestyle,
                    label=family["label"],
                )
            ax.axhline(0, color="#333333", lw=0.6)
            ax.grid(axis="y", color="#E6E6E6", lw=0.6)
            ax.set_xlim(*xlim)
            if col == 0:
                ax.set_ylabel(ylabel)
                panel_label(ax, chr(ord("a") + row))
            if row == len(trace_metrics) - 1:
                ax.set_xlabel("Distance from simulation start (m)")
    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            color=family["color"],
            lw=1.4,
            linestyle="-" if family_key == "stiffness_only" else "--",
            label=family["label"],
        )
        for family_key, family in FAMILIES.items()
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.13, 1.02), ncol=2, frameon=False)
    fig.suptitle(
        "Incremental response relative to the matched baseline",
        x=0.01,
        y=1.07,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    save_all(fig, out_base)


def write_rows(rows: list[dict], csv_path: Path, json_path: Path) -> None:
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = load_all()
    span = defect_span(loaded)
    rows = summarize_metrics(loaded, span)
    plot_metric_grid(rows, OUT_DIR / "ballast_konly_vs_sync_metric_grid")
    plot_delta_traces(loaded, span, OUT_DIR / "ballast_konly_vs_sync_delta_traces")
    write_rows(
        rows,
        OUT_DIR / "ballast_konly_vs_sync_summary.csv",
        OUT_DIR / "ballast_konly_vs_sync_summary.json",
    )
    print("Defect span:", span)
    print("Saved:", OUT_DIR / "ballast_konly_vs_sync_metric_grid.png")
    print("Saved:", OUT_DIR / "ballast_konly_vs_sync_delta_traces.png")
    print("Saved:", OUT_DIR / "ballast_konly_vs_sync_summary.csv")


if __name__ == "__main__":
    main()
