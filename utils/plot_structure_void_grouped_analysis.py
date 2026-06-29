from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


DEFAULT_MANIFEST = Path("configs/sweeps/sleeper_void_2mm_position_scan.yaml")
POSITION_ORDER = ["high_peak", "low_trough", "abs_peak", "max_gradient", "near_flat"]
POSITION_LABELS = {
    "high_peak": "High peak",
    "low_trough": "Low trough",
    "abs_peak": "Large amplitude",
    "max_gradient": "Max gradient",
    "near_flat": "Near flat",
}

COUNT_COLORS = {1: "#0072B2", 2: "#009E73", 3: "#D55E00"}
POSITION_COLORS = {
    "high_peak": "#0072B2",
    "low_trough": "#D55E00",
    "abs_peak": "#009E73",
    "max_gradient": "#CC79A7",
    "near_flat": "#E69F00",
}
PANEL_LABELS = "abcdefghijklmnopqrstuvwxyz"


def apply_publication_style() -> None:
    """Apply a compact journal-style Matplotlib theme."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "axes.linewidth": 0.55,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "legend.fontsize": 6,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def style_axes(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=1.5)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color="#D9D9D9", lw=0.35, alpha=0.65)
    ax.set_axisbelow(True)


def add_panel_labels(axes) -> None:
    for label, ax in zip(PANEL_LABELS, np.ravel(axes)):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=8,
                fontweight="bold", ha="left", va="top")


def save_figure(fig, out_dir: Path, stem: str) -> Path:
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    svg_path = out_dir / f"{stem}.svg"
    tiff_path = out_dir / f"{stem}.tiff"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(tiff_path, dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return png_path


def pivot_position_count(metrics: pd.DataFrame, column: str) -> pd.DataFrame:
    df = metrics[(metrics["group"] == "irr") & (metrics["void_count"] > 0)].copy()
    pivot = df.pivot_table(index="position_label", columns="void_count", values=column, aggfunc="first")
    return pivot.reindex([POSITION_LABELS[key] for key in POSITION_ORDER]).reindex(columns=[1, 2, 3])


def draw_heatmap(ax, values: pd.DataFrame, title: str, cbar_label: str,
                 cmap: str = "viridis", vmin: float | None = None,
                 vmax: float | None = None, fmt: str = ".1f") -> None:
    arr = values.to_numpy(dtype=float)
    image = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels([str(col) for col in values.columns])
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels(values.index)
    ax.set_xlabel("Number of consecutive void sleepers")
    ax.set_ylabel("Void position")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=5.5, color="#111111")
    cbar = plt.colorbar(image, ax=ax, fraction=0.034, pad=0.035)
    cbar.set_label(cbar_label)
    cbar.outline.set_linewidth(0.45)
    style_axes(ax, grid_axis="")


def draw_annotated_heatmap(ax, values: pd.DataFrame, title: str, cbar_label: str,
                           cmap: str, vmin: float, vmax: float,
                           fmt: str = ".1f", show_y: bool = True) -> None:
    arr = values.to_numpy(dtype=float)
    image = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels([str(col) for col in values.columns])
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels(values.index if show_y else [])
    ax.set_xlabel("Consecutive void sleepers")
    ax.set_ylabel("Void position" if show_y else "")

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = arr[i, j]
            if not np.isfinite(val):
                continue
            r, g, b, _ = cmap_obj(norm(val))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if lum < 0.45 else "#222222"
            ax.text(j, i, format(val, fmt), ha="center", va="center",
                    fontsize=5.2, color=text_color)

    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=5.4, labelpad=1.8)
    cbar.ax.tick_params(labelsize=5.4, width=0.45, length=2.2, pad=1.2)
    cbar.outline.set_linewidth(0.45)
    style_axes(ax, grid_axis="")


def plot_design_matrix(ax, metrics: pd.DataFrame) -> None:
    ax.set_axis_off()
    ax.set_title("Experiment design")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    band_specs = [
        (0.60, 0.30, "#F3F5FA", "Smooth track", "0-3 voids at one fixed location", "count effect"),
        (0.08, 0.40, "#F7F3EF", "Measured irregularity", "five positions x three counts", ""),
    ]
    for y0, height, face, title, subtitle, question in band_specs:
        ax.add_patch(plt.Rectangle((0.00, y0), 1.00, height, facecolor=face,
                                   edgecolor="#D0D0D0", lw=0.45))
        ax.text(0.03, y0 + height - 0.075, title, fontsize=6.2, weight="bold", va="center")
        ax.text(0.03, y0 + height - 0.17, subtitle, fontsize=5.5, color="#555555", va="center")
        if question:
            ax.text(0.82, y0 + height - 0.075, question, fontsize=5.5, color="#333333",
                    ha="center", va="center")

    # Smooth-track count sweep: one baseline and three count levels at the same defect location.
    x0 = 0.42
    for i, count in enumerate([0, 1, 2, 3]):
        x = x0 + i * 0.105
        ax.scatter(x, 0.79, s=34, color="#484878", edgecolors="white", linewidths=0.5)
        ax.text(x, 0.70, str(count), ha="center", va="center", fontsize=5.7)
    ax.annotate("", xy=(0.80, 0.75), xytext=(0.76, 0.75),
                arrowprops=dict(arrowstyle="->", lw=0.65, color="#555555"))
    ax.text(0.83, 0.71, "RMS dose\nresponse", fontsize=5.2, ha="left", va="center")

    # Measured-irregularity scan: rows are void positions, columns are void counts.
    position_y = np.linspace(0.35, 0.15, len(POSITION_ORDER))
    ax.text(0.39, 0.25, "5 positions", fontsize=5.2, color="#555555", ha="right", va="center")
    ax.plot([0.42, 0.42], [position_y[-1], position_y[0]], color="#767676", lw=0.55)
    ax.plot([0.42, 0.45], [position_y[0], position_y[0]], color="#767676", lw=0.55)
    ax.plot([0.42, 0.45], [position_y[-1], position_y[-1]], color="#767676", lw=0.55)
    for count in [1, 2, 3]:
        ax.text(0.62 + 0.07 * (count - 1), 0.39, str(count), ha="center",
                va="center", fontsize=5.4, color="#555555")
    for y, pos in zip(position_y, POSITION_ORDER):
        sub = metrics[(metrics["group"] == "irr") & (metrics["position_key"] == pos)]
        for count in [1, 2, 3]:
            present = not sub[sub["void_count"] == count].empty
            color = COUNT_COLORS[count] if present else "#E6E6E6"
            ax.scatter(0.62 + 0.07 * (count - 1), y, s=23, color=color,
                       edgecolors="white", linewidths=0.45)
    ax.text(0.78, 0.39, "count", fontsize=5.2, color="#555555", ha="left", va="center")
    ax.annotate("", xy=(0.78, 0.25), xytext=(0.72, 0.25),
                arrowprops=dict(arrowstyle="->", lw=0.65, color="#555555"))
    ax.text(0.82, 0.25, "local response\nand closure", fontsize=5.2, ha="left", va="center")


def add_direct_endpoint_labels(ax, x_values: list[int], df: pd.DataFrame, col: str) -> None:
    for pos in POSITION_ORDER:
        sub = df[(df["position_key"].isin(["", pos])) & ((df["position_key"] == pos) | (df["void_count"] == 0))]
        sub = sub.sort_values("void_count")
        if sub.empty:
            continue
        y = sub[col].to_numpy(dtype=float)
        x = sub["void_count"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", ms=2.4, lw=0.95, color=POSITION_COLORS[pos],
                markeredgewidth=0, label=POSITION_LABELS[pos])


def plot_carbody_mileage_comparison(ax, items: list[dict[str, Any]]) -> None:
    by_id = {item["case_id"]: item for item in items}
    case_ids = [
        "case_01_noirr_no_void",
        "case_02_noirr_void_1_start_199p8m",
        "case_03_noirr_void_2_start_199p8m",
        "case_04_noirr_void_3_start_199p8m",
    ]
    labels = ["0", "1", "2", "3"]
    colors = ["#606060", COUNT_COLORS[1], COUNT_COLORS[2], COUNT_COLORS[3]]
    first_void = by_id.get(case_ids[1])
    center = float(first_void.get("void_start_m", 200.0)) if first_void else 200.0
    x_min, x_max = center - 22.0, center + 28.0

    for case_id, label, color in zip(case_ids, labels, colors):
        item = by_id.get(case_id)
        if item is None or item["mileage_m"] is None:
            continue
        x = np.asarray(item["mileage_m"], dtype=float)
        y = np.asarray(item["carbody_az_g"], dtype=float) * 1000.0
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]
        mask = (x >= x_min) & (x <= x_max)
        if not np.any(mask):
            continue
        ax.plot(x[mask], y[mask], lw=0.9, color=color, label=label)

    if first_void:
        start = float(first_void["void_start_m"])
        end = start + max(0, int(first_void["void_count"]) - 1) * 0.6
        ax.axvspan(start, end, color="#C7C7C7", alpha=0.35, lw=0)
        ax.axvline(start, color="#4D4D4D", lw=0.55, ls="--", alpha=0.9)

    ax.axhline(0.0, color="#222222", lw=0.5)
    ax.set_title(r"Car-body $a_z$ near the void")
    ax.set_xlabel("Relative mileage (m)")
    ax.set_ylabel(r"$a_z$ (mg)")
    ax.set_xlim(x_min, x_max)
    ax.legend(title="Void count", title_fontsize=5.4, loc="upper left",
              ncol=4, fontsize=5.2, handlelength=1.1, columnspacing=0.7)
    style_axes(ax)


def plot_experiment_comparison_master(metrics: pd.DataFrame, out_dir: Path,
                                      items: list[dict[str, Any]]) -> Path:
    """Create the redesigned main comparison figure for the grouped experiment."""
    noirr = metrics[metrics["group"] == "noirr"].sort_values("void_count")
    x = noirr["void_count"].to_numpy(dtype=float)
    irr = metrics[metrics["group"] == "irr"].copy()

    local_wheel = pivot_position_count(metrics, "void_local_wheelset1_az_rms_pct_vs_group_base")
    local_bogie = pivot_position_count(metrics, "void_local_bogie1_az_rms_pct_vs_group_base")
    contact = pivot_position_count(metrics, "void_contact_ratio") * 100.0

    fig = plt.figure(figsize=(7.6, 5.35))
    gs = fig.add_gridspec(
        3, 18,
        height_ratios=[0.98, 1.12, 0.94],
        left=0.07,
        right=0.97,
        top=0.95,
        bottom=0.08,
        wspace=1.35,
        hspace=0.44,
    )
    ax_a = fig.add_subplot(gs[0, :9])
    ax_b = fig.add_subplot(gs[0, 10:])
    ax_c = fig.add_subplot(gs[1, :5])
    ax_d = fig.add_subplot(gs[1, 6:11])
    ax_e = fig.add_subplot(gs[1, 12:17])
    ax_f = fig.add_subplot(gs[2, :8])
    ax_g = fig.add_subplot(gs[2, 10:])

    plot_carbody_mileage_comparison(ax_a, items)

    ax_b.plot(x, noirr["post2_wheelset1_az_rms_g"], marker="o", ms=2.9, lw=1.05,
              color="#0F4D92", label=r"Wheelset 1 $a_z$")
    ax_b.set_xlabel("Consecutive void sleepers")
    ax_b.set_ylabel("Wheelset RMS (g)", color="#0F4D92")
    ax_b.tick_params(axis="y", colors="#0F4D92")
    ax_b.set_xticks([0, 1, 2, 3])
    style_axes(ax_b)
    ax_b2 = ax_b.twinx()
    ax_b2.plot(x, noirr["post2_axle1_fz_rms_kn"], marker="s", ms=2.7, lw=1.05,
               color="#B64342", label=r"Axle 1 $F_z$")
    ax_b2.set_ylabel("Axle force RMS (kN)", color="#B64342")
    ax_b2.tick_params(axis="y", colors="#B64342", direction="out", pad=1.5, width=0.55, length=2.8)
    ax_b2.spines["top"].set_visible(False)
    ax_b2.spines["right"].set_linewidth(0.55)
    ax_b.set_title("Smooth track: count-dose response")

    wheel_vmax = max(20.0, float(np.nanmax(local_wheel.to_numpy(dtype=float))))
    bogie_vmax = max(8.0, float(np.nanmax(local_bogie.to_numpy(dtype=float))))
    draw_annotated_heatmap(
        ax_c, local_wheel, r"Local wheelset $a_z$", "",
        cmap="YlOrRd", vmin=0.0, vmax=wheel_vmax, fmt=".1f", show_y=True,
    )
    draw_annotated_heatmap(
        ax_d, local_bogie, r"Local bogie $a_z$", "",
        cmap="PuBuGn", vmin=min(0.0, float(np.nanmin(local_bogie.to_numpy(dtype=float)))),
        vmax=bogie_vmax, fmt=".1f", show_y=False,
    )
    draw_annotated_heatmap(
        ax_e, contact, "Void closure", "",
        cmap="Greys", vmin=0.0, vmax=max(0.35, float(np.nanmax(contact.to_numpy(dtype=float)))),
        fmt=".2f", show_y=False,
    )

    add_direct_endpoint_labels(
        ax_f, [0, 1, 2, 3], irr, "void_local_wheelset1_az_rms_pct_vs_group_base"
    )
    ax_f.axhline(0.0, color="#222222", lw=0.55)
    ax_f.set_title(r"Wheelset response trend")
    ax_f.set_xlabel("Consecutive void sleepers")
    ax_f.set_ylabel("Local RMS change (%)")
    ax_f.set_xticks([0, 1, 2, 3])
    ax_f.set_xlim(-0.08, 3.08)
    ax_f.legend(loc="upper left", ncol=2, fontsize=5.0, handlelength=1.3, columnspacing=0.8)
    style_axes(ax_f)

    rank = metrics[(metrics["group"] == "irr") & (metrics["void_count"] == 3)].copy()
    rank["position_key"] = pd.Categorical(rank["position_key"], categories=POSITION_ORDER, ordered=True)
    rank = rank.sort_values("position_key")
    labels = [POSITION_LABELS[p] for p in POSITION_ORDER]
    y = np.arange(len(labels))
    vals = rank.set_index("position_key").reindex(POSITION_ORDER)["void_local_wheelset1_az_rms_pct_vs_group_base"].to_numpy(dtype=float)
    colors = [POSITION_COLORS[p] for p in POSITION_ORDER]
    ax_g.barh(y, vals, color=colors, alpha=0.92, height=0.58)
    ax_g.set_yticks(y)
    ax_g.set_yticklabels(labels)
    ax_g.tick_params(axis="y", labelsize=5.8)
    ax_g.invert_yaxis()
    ax_g.set_xlabel("Local wheelset RMS change (%)")
    ax_g.set_title("Three-void position ranking")
    for yi, val in zip(y, vals):
        ax_g.text(val + 0.25, yi, f"{val:.1f}", va="center", ha="left", fontsize=5.3)
    style_axes(ax_g, grid_axis="x")

    label_specs = [
        ("a", ax_a, -0.08, 1.06),
        ("b", ax_b, -0.10, 1.06),
        ("c", ax_c, -0.10, 1.07),
        ("d", ax_d, -0.10, 1.07),
        ("e", ax_e, -0.10, 1.07),
        ("f", ax_f, -0.09, 1.07),
        ("g", ax_g, -0.09, 1.07),
    ]
    for label, ax, lx, ly in label_specs:
        ax.text(lx, ly, label, transform=ax.transAxes, fontsize=8,
                fontweight="bold", ha="left", va="top")

    path = save_figure(fig, out_dir, "fig00_experiment_comparison_master")
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML top level must be a mapping: {path}")
    return data


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def expected_run_note(manifest: dict[str, Any], case_id: str) -> str:
    common = manifest.get("common", {}) or {}
    return f'{common.get("note_prefix", "sweep")}_{case_id}'


def find_case_result(workspace: Path, manifest: dict[str, Any], case_id: str) -> Path:
    common = manifest.get("common", {}) or {}
    project_name = str(common.get("project_name", manifest.get("manifest_name", "default_project")))
    result_root = workspace / "results" / project_name
    run_note = expected_run_note(manifest, case_id)
    candidates: list[tuple[float, Path]] = []
    for meta_path in result_root.glob("*/files/run_meta.yaml"):
        try:
            meta = load_yaml(meta_path)
        except Exception:
            continue
        if str(meta.get("run_note", "")) != run_note:
            continue
        npz_path = meta_path.with_name("simulation_result.npz")
        if npz_path.exists():
            candidates.append((npz_path.stat().st_mtime, npz_path))
    if not candidates:
        raise FileNotFoundError(f"No simulation result found for {case_id}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((values - np.mean(values)) ** 2)))


def dyn_peak(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan
    return float(np.nanmax(np.abs(values - np.nanmean(values))))


def downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(x), len(y))
    if n <= max_points:
        return x[:n], y[:n]
    step = int(np.ceil(n / max_points))
    return x[:n:step], y[:n:step]


def case_defect(case: dict[str, Any]) -> dict[str, Any]:
    defects = case.get("structure_defects") or []
    return defects[0] if defects else {}


def classify_case(case: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case["case_id"])
    defect = case_defect(case)
    count = int(defect.get("count", 0) or 0)
    start_m = defect.get("start_m", np.nan)
    position = ""
    for name in POSITION_ORDER:
        if name in case_id:
            position = name
            break
    group = "noirr" if "_noirr_" in case_id else "irr"
    return {
        "case_id": case_id,
        "note": str(case.get("note", "")),
        "group": group,
        "void_count": count,
        "void_start_m": float(start_m) if start_m != "" and not pd.isna(start_m) else np.nan,
        "position_key": position,
        "position_label": POSITION_LABELS.get(position, ""),
        "gap_mm": float(defect.get("delta_gap_m", 0.0) or 0.0) * 1000.0,
    }


def compact_case_label(item: dict[str, Any]) -> str:
    count = int(item.get("void_count", 0) or 0)
    if count <= 0:
        return "baseline"
    pos = item.get("position_label", "")
    if pos:
        return f"{pos}, n={count}"
    return f"n={count}"


def pass_window(start_m: float, count: int, v: float = 215.0 / 3.6, x0: float = 20.0,
                lc: float = 9.0, lt: float = 1.2, spacing_m: float = 0.6) -> tuple[float, float]:
    end_m = float(start_m) + max(0, int(count) - 1) * spacing_m
    t0 = (float(start_m) - x0 - 2.0 * (lc + lt)) / v - 0.2
    t1 = (end_m - x0) / v + 0.5
    return max(0.0, t0), t1


def load_case_series(npz_path: Path, case: dict[str, Any]) -> dict[str, Any]:
    data = np.load(npz_path)
    a = data["A"]
    nt = a.shape[0]
    dt = float(data["dt"])
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

    if total_v is not None and total_v.ndim == 2 and total_v.shape[0] == nt and total_v.shape[1] >= 2:
        axle1_fz_kn = (total_v[:, 0] + total_v[:, 1]) / 1000.0
    else:
        axle1_fz_kn = np.zeros(nt)

    mileage_m = None
    if "Track_rel_mileage_m" in data.files:
        mileage_m = np.asarray(data["Track_rel_mileage_m"], dtype=float).reshape(-1)[:nt]

    void_nodes = data["Structure_defect_void_nodes"] if "Structure_defect_void_nodes" in data.files else np.zeros(nt)
    contact_nodes = data["Structure_defect_void_contact_nodes"] if "Structure_defect_void_contact_nodes" in data.files else np.zeros(nt)

    return {
        **classify_case(case),
        "npz_path": npz_path,
        "args": load_json(npz_path.with_name("argparse_params.json")),
        "t": t,
        "mileage_m": mileage_m,
        "carbody_az_g": col(1) / 9.81,
        "bogie1_az_g": col(6) / 9.81,
        "wheelset1_az_g": col(16) / 9.81,
        "axle1_fz_kn": axle1_fz_kn,
        "void_nodes": void_nodes,
        "contact_nodes": contact_nodes,
    }


def mark_void_mileage(ax, item: dict[str, Any]) -> None:
    if not np.isfinite(item.get("void_start_m", np.nan)) or item.get("void_count", 0) <= 0:
        return
    start = float(item["void_start_m"])
    end = start + max(0, int(item["void_count"]) - 1) * 0.6
    ax.axvspan(start, end, color="#C7C7C7", alpha=0.28, lw=0)
    ax.axvline(start, color="#4D4D4D", lw=0.65, ls="--", alpha=0.9)


def compute_metrics(items: list[dict[str, Any]]) -> pd.DataFrame:
    by_id = {item["case_id"]: item for item in items}
    baselines = {
        "noirr": by_id["case_01_noirr_no_void"],
        "irr": by_id["case_05_irr_no_void"],
    }
    rows = []
    signal_map = [
        ("carbody_az", "carbody_az_g", "g"),
        ("bogie1_az", "bogie1_az_g", "g"),
        ("wheelset1_az", "wheelset1_az_g", "g"),
        ("axle1_fz", "axle1_fz_kn", "kn"),
    ]
    for item in items:
        base = baselines[item["group"]]
        total_void = float(np.nansum(item["void_nodes"]))
        contact_ratio = float(np.nansum(item["contact_nodes"]) / total_void) if total_void > 0.0 else 0.0
        row = {
            "case_id": item["case_id"],
            "group": item["group"],
            "baseline_case_id": base["case_id"],
            "position_key": item["position_key"],
            "position_label": item["position_label"],
            "void_start_m": item["void_start_m"],
            "void_count": int(item["void_count"]),
            "gap_mm": float(item["gap_mm"]),
            "void_contact_ratio": contact_ratio,
        }
        masks = {"post2": item["t"] >= 2.0}
        if item["void_count"] > 0 and np.isfinite(item["void_start_m"]):
            t0, t1 = pass_window(float(item["void_start_m"]), int(item["void_count"]))
            masks["void_local"] = (item["t"] >= t0) & (item["t"] <= t1)
        else:
            masks["void_local"] = item["t"] >= 2.0

        for scope, mask in masks.items():
            base_mask = base["t"] >= 2.0 if scope == "post2" else mask[:len(base["t"])]
            if base_mask.size != base["t"].size:
                base_mask = base["t"] >= 2.0
            for stem, key, unit in signal_map:
                value = rms(item[key][mask])
                base_value = rms(base[key][base_mask])
                peak = dyn_peak(item[key][mask])
                base_peak = dyn_peak(base[key][base_mask])
                row[f"{scope}_{stem}_rms_{unit}"] = value
                row[f"{scope}_{stem}_rms_pct_vs_group_base"] = (value / (base_value + 1e-12) - 1.0) * 100.0
                row[f"{scope}_{stem}_dyn_peak_{unit}"] = peak
                row[f"{scope}_{stem}_dyn_peak_pct_vs_group_base"] = (peak / (base_peak + 1e-12) - 1.0) * 100.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_noirr_count(metrics: pd.DataFrame, out_dir: Path) -> Path:
    df = metrics[metrics["group"] == "noirr"].sort_values("void_count")
    x = df["void_count"].to_numpy()
    panels = [
        ("post2_carbody_az_rms_g", r"Car body $a_z$", "RMS (g)"),
        ("post2_bogie1_az_rms_g", r"Bogie 1 $a_z$", "RMS (g)"),
        ("post2_wheelset1_az_rms_g", r"Wheelset 1 $a_z$", "RMS (g)"),
        ("post2_axle1_fz_rms_kn", r"Axle 1 $F_z$", "RMS (kN)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.6), constrained_layout=True)
    for ax, (col, title, ylabel) in zip(axes.ravel(), panels):
        ax.plot(x, df[col], marker="o", ms=3.2, lw=1.15, color="#0072B2", markeredgewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Number of consecutive void sleepers")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        style_axes(ax)
    add_panel_labels(axes)
    fig.suptitle("No irregularity", fontsize=8, y=1.02)
    path = save_figure(fig, out_dir, "fig01_noirr_count_response")
    return path


def plot_nature_summary(metrics: pd.DataFrame, out_dir: Path) -> Path:
    """Create a compact main-text figure that emphasizes the grouped findings."""
    noirr = metrics[metrics["group"] == "noirr"].sort_values("void_count")
    x = noirr["void_count"].to_numpy()

    local_wheel = pivot_position_count(metrics, "void_local_wheelset1_az_rms_pct_vs_group_base")
    local_force = pivot_position_count(metrics, "void_local_axle1_fz_rms_pct_vs_group_base")
    contact = pivot_position_count(metrics, "void_contact_ratio") * 100.0

    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    subfigs = fig.subfigures(2, 2, wspace=0.05, hspace=0.08)
    ax_a = subfigs[0, 0].subplots()
    ax_b = subfigs[0, 1].subplots()
    ax_c = subfigs[1, 0].subplots()
    ax_d = subfigs[1, 1].subplots()

    ax_a.plot(x, noirr["post2_wheelset1_az_rms_g"], marker="o", ms=3.0, lw=1.05,
              color="#0072B2", label=r"Wheelset 1 $a_z$")
    ax_a.set_xlabel("Number of consecutive void sleepers")
    ax_a.set_ylabel(r"Wheelset RMS (g)", color="#0072B2")
    ax_a.tick_params(axis="y", colors="#0072B2")
    ax_a.set_xticks(x)
    style_axes(ax_a)
    ax_a2 = ax_a.twinx()
    ax_a2.plot(x, noirr["post2_axle1_fz_rms_kn"], marker="s", ms=2.8, lw=1.05,
               color="#D55E00", label=r"Axle 1 $F_z$")
    ax_a2.set_ylabel(r"Force RMS (kN)", color="#D55E00")
    ax_a2.tick_params(axis="y", colors="#D55E00", direction="out", pad=1.5, width=0.55, length=2.8)
    ax_a2.spines["top"].set_visible(False)
    ax_a2.spines["right"].set_linewidth(0.55)
    ax_a.set_title("No-irregularity baseline")

    draw_heatmap(
        ax_b,
        local_wheel,
        r"Local wheelset $a_z$ response",
        "RMS change (%)",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=max(20.0, float(np.nanmax(local_wheel.to_numpy(dtype=float)))),
    )
    draw_heatmap(
        ax_c,
        local_force,
        r"Local wheel-rail force response",
        "RMS change (%)",
        cmap="PuBuGn",
        vmin=min(0.0, float(np.nanmin(local_force.to_numpy(dtype=float)))),
        vmax=max(5.0, float(np.nanmax(local_force.to_numpy(dtype=float)))),
    )
    draw_heatmap(
        ax_d,
        contact,
        "Void closure ratio",
        "Contact ratio (%)",
        cmap="Greys",
        vmin=0.0,
        vmax=max(0.35, float(np.nanmax(contact.to_numpy(dtype=float)))),
        fmt=".2f",
    )

    for label, ax in zip("abcd", [ax_a, ax_b, ax_c, ax_d]):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=8,
                fontweight="bold", ha="left", va="top")
    path = save_figure(fig, out_dir, "fig00_nature_summary")
    return path


def plot_irr_position_by_count(metrics: pd.DataFrame, out_dir: Path, scope: str) -> Path:
    df = metrics[(metrics["group"] == "irr") & (metrics["void_count"] > 0)].copy()
    df["position_key"] = pd.Categorical(df["position_key"], categories=POSITION_ORDER, ordered=True)
    df = df.sort_values(["void_count", "position_key"])
    metric_cols = [
        (f"{scope}_carbody_az_rms_pct_vs_group_base", r"Car body $a_z$"),
        (f"{scope}_bogie1_az_rms_pct_vs_group_base", r"Bogie 1 $a_z$"),
        (f"{scope}_wheelset1_az_rms_pct_vs_group_base", r"Wheelset 1 $a_z$"),
        (f"{scope}_axle1_fz_rms_pct_vs_group_base", r"Axle 1 $F_z$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    positions = np.arange(len(POSITION_ORDER))
    width = 0.22
    for ax, (col, title) in zip(axes.ravel(), metric_cols):
        for offset, count in zip([-width, 0, width], [1, 2, 3]):
            sub = df[df["void_count"] == count].set_index("position_key").reindex(POSITION_ORDER)
            ax.bar(positions + offset, sub[col].to_numpy(dtype=float), width=width,
                   label=f"{count}", color=COUNT_COLORS[count], alpha=0.92, linewidth=0)
        ax.axhline(0.0, color="#222222", lw=0.55)
        ax.set_title(title)
        ax.set_ylabel("Change in RMS (%)")
        ax.set_xticks(positions)
        ax.set_xticklabels([POSITION_LABELS[p] for p in POSITION_ORDER], rotation=25, ha="right")
        style_axes(ax)
    axes[0, 0].legend(title="Void count", title_fontsize=6, loc="upper left")
    add_panel_labels(axes)
    label = "full record after 2 s" if scope == "post2" else "local passage window"
    fig.suptitle(f"Measured irregularity: {label}", fontsize=8, y=1.02)
    path = save_figure(fig, out_dir, f"fig02_irr_position_count_{scope}")
    return path


def plot_irr_count_trends(metrics: pd.DataFrame, out_dir: Path, scope: str) -> Path:
    df = metrics[metrics["group"] == "irr"].copy()
    metric_cols = [
        (f"{scope}_carbody_az_rms_pct_vs_group_base", r"Car body $a_z$"),
        (f"{scope}_bogie1_az_rms_pct_vs_group_base", r"Bogie 1 $a_z$"),
        (f"{scope}_wheelset1_az_rms_pct_vs_group_base", r"Wheelset 1 $a_z$"),
        (f"{scope}_axle1_fz_rms_pct_vs_group_base", r"Axle 1 $F_z$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.7), constrained_layout=True)
    for ax, (col, title) in zip(axes.ravel(), metric_cols):
        for pos in POSITION_ORDER:
            sub = df[(df["position_key"].isin(["", pos])) & ((df["position_key"] == pos) | (df["void_count"] == 0))]
            sub = sub.sort_values("void_count")
            ax.plot(sub["void_count"], sub[col], marker="o", ms=2.8, lw=0.95,
                    label=POSITION_LABELS[pos], color=POSITION_COLORS[pos], markeredgewidth=0)
        ax.axhline(0.0, color="#222222", lw=0.55)
        ax.set_title(title)
        ax.set_xlabel("Number of consecutive void sleepers")
        ax.set_ylabel("Change in RMS (%)")
        ax.set_xticks([0, 1, 2, 3])
        style_axes(ax)
    axes[0, 0].legend(loc="upper left", ncol=1)
    add_panel_labels(axes)
    label = "full record after 2 s" if scope == "post2" else "local passage window"
    fig.suptitle(f"Measured irregularity: void-count trend ({label})", fontsize=8, y=1.02)
    path = save_figure(fig, out_dir, f"fig03_irr_count_trends_{scope}")
    return path


def plot_grouped_delta_overlays(items: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    by_id = {item["case_id"]: item for item in items}
    groups = [
        ("noirr_count", ["case_01_noirr_no_void", "case_02_noirr_void_1_start_199p8m",
                         "case_03_noirr_void_2_start_199p8m", "case_04_noirr_void_3_start_199p8m"],
         "No irregularity: count comparison", "case_01_noirr_no_void"),
        ("irr_void1_positions", ["case_05_irr_no_void", "case_06_irr_void_1_high_peak",
                                 "case_07_irr_void_1_low_trough", "case_08_irr_void_1_abs_peak",
                                 "case_09_irr_void_1_max_gradient", "case_10_irr_void_1_near_flat"],
         "Measured irregularity: one void position comparison", "case_05_irr_no_void"),
        ("irr_void2_positions", ["case_05_irr_no_void", "case_11_irr_void_2_high_peak",
                                 "case_12_irr_void_2_low_trough", "case_13_irr_void_2_abs_peak",
                                 "case_14_irr_void_2_max_gradient", "case_15_irr_void_2_near_flat"],
         "Measured irregularity: two consecutive void position comparison", "case_05_irr_no_void"),
        ("irr_void3_positions", ["case_05_irr_no_void", "case_16_irr_void_3_high_peak",
                                 "case_17_irr_void_3_low_trough", "case_18_irr_void_3_abs_peak",
                                 "case_19_irr_void_3_max_gradient", "case_20_irr_void_3_near_flat"],
         "Measured irregularity: three consecutive void position comparison", "case_05_irr_no_void"),
    ]
    panels = [
        ("carbody_az_g", r"Car body $\Delta a_z$", "g"),
        ("bogie1_az_g", r"Bogie 1 $\Delta a_z$", "g"),
        ("wheelset1_az_g", r"Wheelset 1 $\Delta a_z$", "g"),
        ("axle1_fz_kn", r"Axle 1 $\Delta F_z$", "kN"),
    ]
    paths: list[Path] = []
    for group_id, case_ids, title, baseline_id in groups:
        base = by_id[baseline_id]
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
        color_cycle = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
        color_i = 0
        for ax, (key, panel_title, ylabel) in zip(axes.ravel(), panels):
            color_i = 0
            for case_id in case_ids:
                item = by_id[case_id]
                if item is base:
                    continue
                if item["mileage_m"] is None:
                    continue
                n = min(len(item["mileage_m"]), len(item[key]), len(base[key]))
                x = item["mileage_m"][:n]
                y = item[key][:n] - base[key][:n]
                x_ds, y_ds = downsample_xy(x, y)
                label = compact_case_label(item)
                ax.plot(x_ds, y_ds, lw=0.75, label=label, color=color_cycle[color_i % len(color_cycle)])
                color_i += 1
                mark_void_mileage(ax, item)
            ax.axhline(0.0, color="#222222", lw=0.55)
            ax.set_title(panel_title)
            ax.set_xlabel("Relative mileage (m)")
            ax.set_ylabel(ylabel)
            style_axes(ax)
        axes[0, 0].legend(loc="upper left", ncol=1)
        add_panel_labels(axes)
        fig.suptitle(title, fontsize=8, y=1.02)
        path = save_figure(fig, out_dir, f"fig04_delta_overlay_{group_id}")
        paths.append(path)
    return paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate grouped sleeper-void analysis plots.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Sweep manifest YAML.")
    parser.add_argument("--output-subdir", default="grouped_publication",
                        help="Subdirectory under results/<project>/_comparison for publication figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    apply_publication_style()
    manifest_path = Path(args.manifest).resolve()
    workspace = manifest_path.parent.parent.parent
    manifest = load_yaml(manifest_path)
    project_name = str((manifest.get("common", {}) or {}).get("project_name", manifest.get("manifest_name", "default_project")))
    out_dir = workspace / "results" / project_name / "_comparison" / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = manifest.get("cases", [])
    items = [load_case_series(find_case_result(workspace, manifest, str(case["case_id"])), case) for case in cases]
    metrics = compute_metrics(items)
    metrics_path = out_dir / "grouped_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    paths = [
        plot_experiment_comparison_master(metrics, out_dir, items),
        plot_nature_summary(metrics, out_dir),
        plot_noirr_count(metrics, out_dir),
        plot_irr_position_by_count(metrics, out_dir, "post2"),
        plot_irr_position_by_count(metrics, out_dir, "void_local"),
        plot_irr_count_trends(metrics, out_dir, "post2"),
        plot_irr_count_trends(metrics, out_dir, "void_local"),
    ]
    paths.extend(plot_grouped_delta_overlays(items, out_dir))

    print(f"Wrote grouped metrics: {metrics_path}")
    print("Wrote grouped figures:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
