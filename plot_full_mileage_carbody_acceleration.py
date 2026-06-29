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
OUT_DIR = Path("results/transition_settlement_full_mileage_comparison")
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
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.2,
    "legend.frameon": False,
    "figure.dpi": 160,
})


COLORS = {
    "baseline": "#707478",
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
        az = np.asarray(data["A"][:, 1], dtype=float)
        cases.append({
            "scan": scan_name,
            "case_id": _case_id_from_path(path),
            "family": family,
            "amplitude_mm": amplitude_mm,
            "distance_m": distance_m,
            "carbody_az_mps2": az,
            "peak_full_mps2": float(np.nanmax(np.abs(az))),
            "rms_full_mps2": float(np.sqrt(np.nanmean(az ** 2))),
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
    ax.grid(True, color=COLORS["grid"], lw=0.45, alpha=0.75)
    ax.tick_params(length=2.5, width=0.7)
    ax.axvspan(SETTLEMENT_ZONE_M[0], SETTLEMENT_ZONE_M[1], color=COLORS["zone"], zorder=0)
    ax.axhline(0.0, color="#5A5A5A", lw=0.55)


def _plot_family(ax, cases: list[dict], family: str):
    baseline = next(row for row in cases if row["family"] == "baseline")
    family_cases = [row for row in cases if row["family"] == family]
    max_amp = max(row["amplitude_mm"] for row in family_cases)

    ax.plot(
        baseline["distance_m"],
        baseline["carbody_az_mps2"],
        color=COLORS["baseline"],
        lw=0.85,
        alpha=0.82,
        label="baseline",
        zorder=2,
    )
    for row in family_cases:
        alpha = 0.28 + 0.70 * row["amplitude_mm"] / max_amp
        lw = 0.75 + 0.45 * row["amplitude_mm"] / max_amp
        label = f"{row['amplitude_mm']:.0f} mm"
        ax.plot(
            row["distance_m"],
            row["carbody_az_mps2"],
            color=COLORS[family],
            lw=lw,
            alpha=alpha,
            label=label,
            zorder=3,
        )
    _style_axis(ax)


def make_figure(scans: dict[str, list[dict]]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8), sharex=False, sharey="row")
    scan_names = list(scans.keys())
    row_families = [("cosine", "Cosine settlement"), ("kink", "Kink settlement")]

    for col, scan_name in enumerate(scan_names):
        for row, (family, family_title) in enumerate(row_families):
            ax = axes[row, col]
            _plot_family(ax, scans[scan_name], family)
            title = f"{family_title}; {scan_name}"
            ax.set_title(title, loc="left", pad=4)
            ax.set_xlabel("Distance from valid start (m)")
            if col == 0:
                ax.set_ylabel("Carbody $a_z$ (m s$^{-2}$)")
            xmin = min(float(np.nanmin(item["distance_m"])) for item in scans[scan_name])
            xmax = max(float(np.nanmax(item["distance_m"])) for item in scans[scan_name])
            ax.set_xlim(xmin, xmax)
            if row == 0:
                ax.set_ylim(-0.95, 1.38)
            else:
                ax.set_ylim(-0.24, 0.30)
            if col == 1:
                ax.legend(loc="upper right", ncol=1, handlelength=1.5)
            else:
                ax.legend(loc="upper right", ncol=1, handlelength=1.5)

    for ax, letter in zip(axes.flat, "abcd"):
        ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")

    fig.suptitle("Full-mileage carbody acceleration response under transition-zone settlement", x=0.02, ha="left", fontsize=9.5, fontweight="bold")
    fig.text(0.02, 0.006, "Each trace uses the full saved mileage range; shaded band marks the imposed 20 m settlement zone.", fontsize=6.5, color="#444444")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.105, wspace=0.22, hspace=0.36)

    base = OUT_DIR / "full_mileage_carbody_acceleration_comparison"
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
                "peak_full_carbody_az_mps2": row["peak_full_mps2"],
                "rms_full_carbody_az_mps2": row["rms_full_mps2"],
                "result_npz": str(row["path"]),
            })
            for distance, az in zip(row["distance_m"], row["carbody_az_mps2"]):
                trace_rows.append({
                    "scan": row["scan"],
                    "case_id": row["case_id"],
                    "family": row["family"],
                    "amplitude_mm": row["amplitude_mm"],
                    "distance_m": distance,
                    "carbody_az_mps2": az,
                })
    pd.DataFrame(metric_rows).to_csv(OUT_DIR / "full_mileage_carbody_acceleration_metrics.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(OUT_DIR / "full_mileage_carbody_acceleration_traces.csv", index=False)


def main() -> None:
    scans = load_all()
    export_source_data(scans)
    figure = make_figure(scans)
    print(f"Saved figure: {figure}")
    print(f"Saved source data: {OUT_DIR}")


if __name__ == "__main__":
    main()
