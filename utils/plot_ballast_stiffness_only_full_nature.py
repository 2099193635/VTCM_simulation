from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "ballast_condition_stiffness_only_100m_scan"
OUT_DIR = RESULT_ROOT / "_comparison"

SPEED_BASELINE = {
    160: "case_08_seed20260627_baseline_v160",
    215: "case_00_seed20260627_baseline_v215",
    250: "case_11_seed20260627_baseline_v250",
}

CASE_META = {
    "case_00_seed20260627_baseline_v215": (215, 1.0, "baseline"),
    "case_01_seed20260627_eta0p1_count167_v215": (215, 0.1, "defect"),
    "case_02_seed20260627_eta0p2_count167_v215": (215, 0.2, "defect"),
    "case_03_seed20260627_eta0p5_count167_v215": (215, 0.5, "defect"),
    "case_04_seed20260627_eta1_count167_v215": (215, 1.0, "neutral-window"),
    "case_05_seed20260627_eta2_count167_v215": (215, 2.0, "defect"),
    "case_06_seed20260627_eta5_count167_v215": (215, 5.0, "defect"),
    "case_07_seed20260627_eta10_count167_v215": (215, 10.0, "defect"),
    "case_08_seed20260627_baseline_v160": (160, 1.0, "baseline"),
    "case_09_seed20260627_eta0p2_count167_v160": (160, 0.2, "defect"),
    "case_10_seed20260627_eta5_count167_v160": (160, 5.0, "defect"),
    "case_11_seed20260627_baseline_v250": (250, 1.0, "baseline"),
    "case_12_seed20260627_eta0p2_count167_v250": (250, 0.2, "defect"),
    "case_13_seed20260627_eta5_count167_v250": (250, 5.0, "defect"),
}

METRICS = {
    "carbody_az_rms": ("carbody_az_mps2", "rms", "Carbody acceleration\nRMS change (%)"),
    "carbody_z_rms": ("carbody_z_mm", "rms", "Carbody displacement\nRMS change (%)"),
    "bogie_az_rms": ("bogie_az_mean_mps2", "rms", "Bogie acceleration\nRMS change (%)"),
    "bogie_z_rms": ("bogie_z_mean_mm", "rms", "Bogie displacement\nRMS change (%)"),
    "wr_mean_force_rms": ("wr_force_mean_kN", "rms", "Mean wheel-rail force\nRMS change (%)"),
    "wr_max_force_peak": ("wr_force_max_kN", "peak", "Peak wheel-rail force\nchange (%)"),
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


def case_id_from_path(path: Path) -> str:
    text = str(path)
    match = re.search(r"ballast_konly100m_(case_[^\\/]+?)-2026", text)
    if not match:
        raise ValueError(f"Cannot parse case id from {path}")
    return match.group(1)


def load_case(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    distance = np.asarray(data["Irre_distance_m"], dtype=float)
    X = np.asarray(data["X"], dtype=float)
    A = np.asarray(data["A"], dtype=float)
    total_v = np.asarray(data["TotalVerticalForce"], dtype=float)
    mask = np.maximum(
        np.asarray(data.get("Stiffness_irregularity_mask_L", np.zeros_like(distance)), dtype=int),
        np.asarray(data.get("Stiffness_irregularity_mask_R", np.zeros_like(distance)), dtype=int),
    )
    eta_k = 0.5 * (
        np.asarray(data.get("Stiffness_eta_k_L_ref", np.ones_like(distance)), dtype=float)
        + np.asarray(data.get("Stiffness_eta_k_R_ref", np.ones_like(distance)), dtype=float)
    )
    case_id = case_id_from_path(path)
    speed, eta, kind = CASE_META[case_id]
    return {
        "case_id": case_id,
        "path": str(path),
        "speed_kmh": speed,
        "eta_k": eta,
        "kind": kind,
        "distance_m": distance,
        "mask": mask,
        "eta_k_profile": eta_k,
        "carbody_az_mps2": A[:, 1],
        "carbody_z_mm": X[:, 1] * 1000.0,
        "bogie_az_mean_mps2": 0.5 * (A[:, 6] + A[:, 11]),
        "bogie_z_mean_mm": 0.5 * (X[:, 6] + X[:, 11]) * 1000.0,
        "wr_force_mean_kN": np.mean(total_v, axis=1) / 1000.0,
        "wr_force_max_kN": np.max(total_v, axis=1) / 1000.0,
    }


def load_all() -> dict[str, dict]:
    cases = {}
    for path in RESULT_ROOT.rglob("simulation_result.npz"):
        case_id = case_id_from_path(path)
        old = cases.get(case_id)
        if old is None or path.stat().st_mtime > Path(old["path"]).stat().st_mtime:
            cases[case_id] = load_case(path)
    missing = sorted(set(CASE_META) - set(cases))
    if missing:
        raise FileNotFoundError("Missing cases: " + ", ".join(missing))
    return cases


def defect_span(cases: dict[str, dict]) -> tuple[float, float]:
    spans = []
    for case in cases.values():
        idx = np.flatnonzero(case["mask"] > 0)
        if idx.size:
            spans.append((case["distance_m"][idx[0]], case["distance_m"][idx[-1]]))
    return float(min(s[0] for s in spans)), float(max(s[1] for s in spans))


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.nanmean(values * values)))


def metric_value(values: np.ndarray, stat: str) -> float:
    if stat == "rms":
        return rms(values)
    if stat == "peak":
        return float(np.nanmax(np.abs(values)))
    raise ValueError(stat)


def interp_baseline(case: dict, baseline: dict, key: str) -> np.ndarray:
    return np.interp(case["distance_m"], baseline["distance_m"], baseline[key])


def summarize(cases: dict[str, dict], span: tuple[float, float]) -> list[dict]:
    rows = []
    for case_id, case in sorted(cases.items(), key=lambda item: (item[1]["speed_kmh"], item[1]["eta_k"], item[1]["kind"])):
        baseline = cases[SPEED_BASELINE[case["speed_kmh"]]]
        selector = (case["distance_m"] >= span[0]) & (case["distance_m"] <= span[1])
        base_selector = (baseline["distance_m"] >= span[0]) & (baseline["distance_m"] <= span[1])
        row = {
            "case_id": case_id,
            "speed_kmh": case["speed_kmh"],
            "eta_k": case["eta_k"],
            "eta_c": 1.0,
            "kind": case["kind"],
            "span_start_m": span[0],
            "span_end_m": span[1],
            "source_npz": case["path"],
            "baseline_case_id": baseline["case_id"],
        }
        for metric_name, (signal_key, stat, _) in METRICS.items():
            value = metric_value(case[signal_key][selector], stat)
            base_value = metric_value(baseline[signal_key][base_selector], stat)
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


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.03, label, transform=ax.transAxes, fontweight="bold", fontsize=8.5, va="bottom")


def metric_lookup(rows: list[dict], speed: int, eta: float, metric: str) -> float:
    for row in rows:
        if int(row["speed_kmh"]) == speed and np.isclose(row["eta_k"], eta):
            return float(row[f"{metric}_delta_pct"])
    raise KeyError((speed, eta, metric))


def plot_eta_response(rows: list[dict], out_base: Path) -> None:
    configure_matplotlib()
    eta_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.8), constrained_layout=True)
    axes = axes.ravel()
    for idx, (metric_name, (_, _, ylabel)) in enumerate(METRICS.items()):
        ax = axes[idx]
        y = [metric_lookup(rows, 215, eta, metric_name) for eta in eta_values]
        ax.plot(eta_values, y, color="#3B6EA8", marker="o", ms=3.4, lw=1.25)
        ax.axhline(0, color="#333333", lw=0.7)
        ax.axvline(1.0, color="#999999", lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_xticks(eta_values)
        ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)
        ax.set_ylabel(ylabel)
        panel_label(ax, chr(ord("a") + idx))
    fig.suptitle(
        "Stiffness-only ballast condition: response trend at 215 km h$^{-1}$",
        x=0.01,
        y=1.035,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    save_all(fig, out_base)


def plot_speed_sensitivity(rows: list[dict], out_base: Path) -> None:
    configure_matplotlib()
    speeds = [160, 215, 250]
    eta_values = [0.2, 5.0]
    colors = {0.2: "#3B6EA8", 5.0: "#B24A3F"}
    labels = {0.2: "Loose eta_k=0.2", 5.0: "Hardened eta_k=5"}
    selected_metrics = [
        "carbody_z_rms",
        "bogie_z_rms",
        "carbody_az_rms",
        "bogie_az_rms",
        "wr_max_force_peak",
        "wr_mean_force_rms",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.7), constrained_layout=True)
    axes = axes.ravel()
    for idx, metric_name in enumerate(selected_metrics):
        ax = axes[idx]
        for eta in eta_values:
            y = [metric_lookup(rows, speed, eta, metric_name) for speed in speeds]
            ax.plot(speeds, y, color=colors[eta], marker="o", ms=3.5, lw=1.25, label=labels[eta])
        ax.axhline(0, color="#333333", lw=0.7)
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)
        ax.set_xticks(speeds)
        ax.set_ylabel(METRICS[metric_name][2])
        panel_label(ax, chr(ord("a") + idx))
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.22), ncol=1, borderaxespad=0)
    fig.suptitle(
        "Speed sensitivity of stiffness-only ballast-condition response",
        x=0.01,
        y=1.035,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    save_all(fig, out_base)


def plot_representative_traces(cases: dict[str, dict], span: tuple[float, float], out_base: Path) -> None:
    configure_matplotlib()
    baseline = cases[SPEED_BASELINE[215]]
    case_ids = [
        "case_01_seed20260627_eta0p1_count167_v215",
        "case_02_seed20260627_eta0p2_count167_v215",
        "case_03_seed20260627_eta0p5_count167_v215",
        "case_05_seed20260627_eta2_count167_v215",
        "case_06_seed20260627_eta5_count167_v215",
        "case_07_seed20260627_eta10_count167_v215",
    ]
    colors = {
        0.1: "#215A8A",
        0.2: "#3B6EA8",
        0.5: "#78A6C8",
        2.0: "#D98C64",
        5.0: "#B24A3F",
        10.0: "#7A231F",
    }
    trace_metrics = [
        ("carbody_z_mm", "Delta carbody\ndisplacement (mm)"),
        ("bogie_z_mean_mm", "Delta bogie\ndisplacement (mm)"),
        ("carbody_az_mps2", "Delta carbody\nacceleration (m s$^{-2}$)"),
        ("wr_force_max_kN", "Delta peak wheel-rail\nforce (kN)"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(7.3, 6.0), sharex=True, constrained_layout=True)
    xlim = (span[0] - 35.0, span[1] + 45.0)
    for idx, (ax, (metric_key, ylabel)) in enumerate(zip(axes, trace_metrics)):
        ax.axvspan(span[0], span[1], color="#D7D7D7", alpha=0.24, lw=0)
        for case_id in case_ids:
            case = cases[case_id]
            eta = case["eta_k"]
            delta = case[metric_key] - interp_baseline(case, baseline, metric_key)
            ax.plot(case["distance_m"], delta, color=colors[eta], lw=0.9, alpha=0.90, label=f"eta_k={eta:g}")
        ax.axhline(0, color="#333333", lw=0.65)
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)
        ax.set_xlim(*xlim)
        ax.set_ylabel(ylabel)
        panel_label(ax, chr(ord("a") + idx))
    axes[-1].set_xlabel("Distance from simulation start (m)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.26), ncol=6, borderaxespad=0, columnspacing=0.7)
    fig.suptitle(
        "Spatial response increments for stiffness-only ballast defects at 215 km h$^{-1}$",
        x=0.01,
        y=1.04,
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    save_all(fig, out_base)


def write_tables(rows: list[dict], out_csv: Path, out_json: Path) -> None:
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_all()
    span = defect_span(cases)
    rows = summarize(cases, span)
    plot_eta_response(rows, OUT_DIR / "ballast_stiffness_only_eta_response")
    plot_speed_sensitivity(rows, OUT_DIR / "ballast_stiffness_only_speed_sensitivity")
    plot_representative_traces(cases, span, OUT_DIR / "ballast_stiffness_only_delta_traces_215")
    write_tables(
        rows,
        OUT_DIR / "ballast_stiffness_only_full_summary.csv",
        OUT_DIR / "ballast_stiffness_only_full_summary.json",
    )
    print("Defect span:", span)
    print("Saved:", OUT_DIR / "ballast_stiffness_only_eta_response.png")
    print("Saved:", OUT_DIR / "ballast_stiffness_only_speed_sensitivity.png")
    print("Saved:", OUT_DIR / "ballast_stiffness_only_delta_traces_215.png")
    print("Saved:", OUT_DIR / "ballast_stiffness_only_full_summary.csv")


if __name__ == "__main__":
    main()
