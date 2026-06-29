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
DEFAULT_OUTPUT_SUBDIR = "grouped_publication"

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


def apply_publication_style() -> None:
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


def save_figure(fig, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        out_dir / f"{stem}.png",
        out_dir / f"{stem}.pdf",
        out_dir / f"{stem}.svg",
        out_dir / f"{stem}.tiff",
    ]
    fig.savefig(paths[0], dpi=600, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.025)
    fig.savefig(paths[2], bbox_inches="tight", pad_inches=0.025)
    fig.savefig(paths[3], dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return paths


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
        "group": group,
        "void_count": count,
        "void_start_m": float(start_m) if start_m != "" and not pd.isna(start_m) else np.nan,
        "position_key": position,
        "position_label": POSITION_LABELS.get(position, ""),
    }


def load_case_series(npz_path: Path, case: dict[str, Any]) -> dict[str, Any]:
    data = np.load(npz_path)
    a = data["A"]
    nt = a.shape[0]

    mileage_m = None
    if "Track_rel_mileage_m" in data.files:
        mileage_m = np.asarray(data["Track_rel_mileage_m"], dtype=float).reshape(-1)[:nt]

    return {
        **classify_case(case),
        "npz_path": npz_path,
        "args": load_json(npz_path.with_name("argparse_params.json")),
        "mileage_m": mileage_m,
        "carbody_az_g": a[:, 1] / 9.81 if a.shape[1] > 1 else np.zeros(nt),
    }


def load_items_for_panel_a(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = load_yaml(manifest_path)
    workspace = manifest_path.parent.parent.parent
    cases = manifest.get("cases", [])
    return [
        load_case_series(find_case_result(workspace, manifest, str(case["case_id"])), case)
        for case in cases
    ]


def default_output_dir(manifest_path: Path, output_subdir: str) -> Path:
    manifest = load_yaml(manifest_path)
    workspace = manifest_path.parent.parent.parent
    common = manifest.get("common", {}) or {}
    project_name = str(common.get("project_name", manifest.get("manifest_name", "default_project")))
    return workspace / "results" / project_name / "_comparison" / output_subdir


def pivot_position_count(metrics: pd.DataFrame, column: str) -> pd.DataFrame:
    df = metrics[(metrics["group"] == "irr") & (metrics["void_count"] > 0)].copy()
    pivot = df.pivot_table(index="position_label", columns="void_count", values=column, aggfunc="first")
    return pivot.reindex([POSITION_LABELS[key] for key in POSITION_ORDER]).reindex(columns=[1, 2, 3])


def draw_annotated_heatmap(
    ax,
    values: pd.DataFrame,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    fmt: str = ".1f",
    show_y: bool = True,
) -> None:
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

    cbar = plt.colorbar(image, ax=ax, fraction=0.034, pad=0.035)
    cbar.ax.tick_params(labelsize=5.4, width=0.45, length=2.2, pad=1.2)
    cbar.outline.set_linewidth(0.45)
    style_axes(ax, grid_axis="")


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
        if np.any(mask):
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


def add_position_trends(ax, df: pd.DataFrame, col: str) -> None:
    for pos in POSITION_ORDER:
        sub = df[(df["position_key"].isin(["", pos])) & ((df["position_key"] == pos) | (df["void_count"] == 0))]
        sub = sub.sort_values("void_count")
        if sub.empty:
            continue
        ax.plot(
            sub["void_count"].to_numpy(dtype=float),
            sub[col].to_numpy(dtype=float),
            marker="o",
            ms=2.4,
            lw=0.95,
            color=POSITION_COLORS[pos],
            markeredgewidth=0,
            label=POSITION_LABELS[pos],
        )


def plot_master(metrics: pd.DataFrame, items: list[dict[str, Any]], out_dir: Path) -> list[Path]:
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
    draw_annotated_heatmap(ax_c, local_wheel, r"Local wheelset $a_z$", "YlOrRd", 0.0, wheel_vmax)
    draw_annotated_heatmap(
        ax_d,
        local_bogie,
        r"Local bogie $a_z$",
        "PuBuGn",
        min(0.0, float(np.nanmin(local_bogie.to_numpy(dtype=float)))),
        bogie_vmax,
        show_y=False,
    )
    draw_annotated_heatmap(
        ax_e,
        contact,
        "Void closure",
        "Greys",
        0.0,
        max(0.35, float(np.nanmax(contact.to_numpy(dtype=float)))),
        fmt=".2f",
        show_y=False,
    )

    add_position_trends(ax_f, irr, "void_local_wheelset1_az_rms_pct_vs_group_base")
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

    return save_figure(fig, out_dir, "fig00_experiment_comparison_master")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate only Fig. 00 experiment-comparison master figure.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Sweep manifest YAML.")
    parser.add_argument("--metrics", default="", help="Path to grouped_metrics.csv. Defaults to output directory.")
    parser.add_argument("--output-dir", default="", help="Directory for exported figure files.")
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR,
                        help="Subdirectory under results/<project>/_comparison when --output-dir is omitted.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    apply_publication_style()

    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(manifest_path, args.output_subdir)
    metrics_path = Path(args.metrics).resolve() if args.metrics else out_dir / "grouped_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_path}")

    metrics = pd.read_csv(metrics_path)
    items = load_items_for_panel_a(manifest_path)
    paths = plot_master(metrics, items, out_dir)

    print("Wrote Fig. 00 files:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
