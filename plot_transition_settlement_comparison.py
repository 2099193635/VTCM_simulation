from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_ROOT = Path("results/transition_settlement_scan")
OUT_DIR = RESULT_ROOT / "_publication_figures"
WINDOW_M = (50.0, 150.0)
SETTLEMENT_ZONE_M = (80.0, 100.0)
FIGURE_PREFIX = "transition_settlement_comparison"
FIGURE_TITLE = "Dynamic response signatures of transition-zone differential settlement"


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
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "figure.dpi": 160,
})


COLORS = {
    "baseline": "#6f7275",
    "cosine": "#0077A3",
    "kink": "#C24D2C",
    "grid": "#D8D8D8",
    "zone": "#E9EEF3",
}


def _case_id_from_path(path: Path) -> str:
    name = path.parents[1].name
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


def load_cases() -> list[dict]:
    cases = []
    for path in sorted(RESULT_ROOT.rglob("simulation_result.npz")):
        if "_publication_figures" in path.parts:
            continue
        data = np.load(path, allow_pickle=True)
        case_id = _case_id_from_path(path)
        meta = _metadata(data)
        if meta:
            spec = meta[0]
            family = str(spec.get("type", "")).lower()
            amplitude_mm = float(spec.get("amplitude_mm", abs(np.nanmin(data["Settlement_profile_mm"]))))
            label = str(spec.get("label", case_id))
        else:
            family = "baseline"
            amplitude_mm = 0.0
            label = "baseline"

        x = np.asarray(data["Settlement_distance_rel_m"], dtype=float)
        settlement = np.asarray(data["Settlement_profile_mm"], dtype=float)
        az = np.asarray(data["A"][:, 1], dtype=float)
        mask = (x >= WINDOW_M[0]) & (x <= WINDOW_M[1])
        peak = float(np.nanmax(np.abs(az[mask])))
        rms = float(np.sqrt(np.nanmean(az[mask] ** 2)))
        p95 = float(np.nanpercentile(np.abs(az[mask]), 95.0))
        cases.append({
            "case_id": case_id,
            "family": family,
            "amplitude_mm": amplitude_mm,
            "label": label,
            "distance_m": x,
            "settlement_mm": settlement,
            "az": az,
            "peak_abs_az": peak,
            "rms_az": rms,
            "p95_abs_az": p95,
            "path": path,
        })
    return sorted(cases, key=lambda d: ({"baseline": 0, "cosine": 1, "kink": 2}.get(d["family"], 9), d["amplitude_mm"]))


def _style_axis(ax):
    ax.grid(True, color=COLORS["grid"], lw=0.45, alpha=0.8)
    ax.tick_params(length=2.5, width=0.7)


def _shade_settlement_zone(ax):
    ax.axvspan(SETTLEMENT_ZONE_M[0], SETTLEMENT_ZONE_M[1], color=COLORS["zone"], zorder=0)


def _line_label(ax, x, y, text, color, x_at=146.0):
    y_at = float(np.interp(x_at, x, y))
    ax.text(x_at + 1.2, y_at, text, color=color, va="center", ha="left", fontsize=6)


def make_figure(cases: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = next(c for c in cases if c["family"] == "baseline")
    cosine = [c for c in cases if c["family"] == "cosine"]
    kink = [c for c in cases if c["family"] == "kink"]
    max_cosine = max(cosine, key=lambda c: c["amplitude_mm"])
    max_kink = max(kink, key=lambda c: c["amplitude_mm"])

    fig = plt.figure(figsize=(7.2, 5.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.18, 1.0], hspace=0.34, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
        ax.text(-0.14, 1.07, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")

    for group, color, linestyle in [(cosine, COLORS["cosine"], "-"), (kink, COLORS["kink"], "--")]:
        for c in group:
            x = c["distance_m"]
            y = c["settlement_mm"]
            mask = (x >= 60.0) & (x <= 145.0)
            alpha = 0.34 + 0.56 * c["amplitude_mm"] / max(max_cosine["amplitude_mm"], max_kink["amplitude_mm"])
            ax_a.plot(x[mask], y[mask], color=color, lw=1.0, alpha=alpha, ls=linestyle)

    _shade_settlement_zone(ax_a)
    ax_a.set_xlim(60, 145)
    ax_a.set_ylim(-38, 3)
    ax_a.set_xlabel("Distance from valid start (m)")
    ax_a.set_ylabel("Track settlement (mm)")
    ax_a.set_title("Parametric transition-zone settlement inputs", loc="left", pad=4)
    ax_a.text(101, -33.0, "cosine basin", color=COLORS["cosine"], fontsize=6.5)
    ax_a.text(101, -20.5, "kink ramp", color=COLORS["kink"], fontsize=6.5)
    _style_axis(ax_a)

    trace_cases = [
        (baseline, COLORS["baseline"], "baseline", 0.85, "-"),
        (max_cosine, COLORS["cosine"], "cosine, 35 mm", 1.0, "-"),
        (max_kink, COLORS["kink"], "kink, 1.0‰", 1.0, "-"),
    ]
    for c, color, label, alpha, linestyle in trace_cases:
        x = c["distance_m"]
        mask = (x >= WINDOW_M[0]) & (x <= WINDOW_M[1])
        ax_b.plot(x[mask], c["az"][mask], color=color, lw=1.05, alpha=alpha, ls=linestyle, label=label)
    _shade_settlement_zone(ax_b)
    ax_b.axhline(0, color="#5A5A5A", lw=0.55)
    ax_b.set_xlim(WINDOW_M)
    ax_b.set_xlabel("Distance from valid start (m)")
    ax_b.set_ylabel("Carbody $a_z$ (m s$^{-2}$)")
    ax_b.set_title("Carbody response near the settlement zone", loc="left", pad=4)
    ax_b.legend(loc="upper right", handlelength=1.8)
    _style_axis(ax_b)

    for group, color, name, marker in [(cosine, COLORS["cosine"], "cosine", "o"), (kink, COLORS["kink"], "kink", "s")]:
        amp = np.array([c["amplitude_mm"] for c in group], dtype=float)
        peak = np.array([c["peak_abs_az"] for c in group], dtype=float)
        slope = float(np.sum(amp * peak) / np.sum(amp * amp))
        ax_c.plot(amp, peak, color=color, lw=1.2, marker=marker, ms=4.0, label=f"{name}, {slope:.3f} m s$^{{-2}}$ mm$^{{-1}}$")
    ax_c.scatter([0], [baseline["peak_abs_az"]], s=18, color=COLORS["baseline"], zorder=3, label="baseline")
    ax_c.set_xlabel("Settlement amplitude (mm)")
    ax_c.set_ylabel("Peak |carbody $a_z$| (m s$^{-2}$)")
    ax_c.set_title("Peak acceleration scales with imposed settlement", loc="left", pad=4)
    ax_c.legend(loc="upper left")
    _style_axis(ax_c)

    for group, color, name, marker in [(cosine, COLORS["cosine"], "cosine", "o"), (kink, COLORS["kink"], "kink", "s")]:
        amp = np.array([c["amplitude_mm"] for c in group], dtype=float)
        rms = np.array([c["rms_az"] for c in group], dtype=float)
        slope = float(np.sum(amp * rms) / np.sum(amp * amp))
        ax_d.plot(amp, rms, color=color, lw=1.2, marker=marker, ms=4.0, label=f"{name}, {slope:.3f} m s$^{{-2}}$ mm$^{{-1}}$")
    ax_d.scatter([0], [baseline["rms_az"]], s=18, color=COLORS["baseline"], zorder=3, label="baseline")
    ax_d.set_xlabel("Settlement amplitude (mm)")
    ax_d.set_ylabel("RMS carbody $a_z$ (m s$^{-2}$)")
    ax_d.set_title("RMS response separates settlement forms", loc="left", pad=4)
    ax_d.legend(loc="upper left")
    _style_axis(ax_d)

    fig.suptitle(FIGURE_TITLE, x=0.02, ha="left", fontsize=9.5, fontweight="bold")
    fig.text(0.02, 0.005, "Metrics computed over 50-150 m; shaded band marks the imposed 20 m transition settlement zone.", fontsize=6.5, color="#444444")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.105)

    base = OUT_DIR / FIGURE_PREFIX
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return base.with_suffix(".png")


def export_source_data(cases: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    trace_rows = []
    for c in cases:
        metric_rows.append({
            "case_id": c["case_id"],
            "family": c["family"],
            "amplitude_mm": c["amplitude_mm"],
            "peak_abs_carbody_az_mps2": c["peak_abs_az"],
            "rms_carbody_az_mps2": c["rms_az"],
            "p95_abs_carbody_az_mps2": c["p95_abs_az"],
            "result_npz": str(c["path"]),
        })
        x = c["distance_m"]
        keep = (x >= WINDOW_M[0]) & (x <= WINDOW_M[1])
        for xi, si, ai in zip(x[keep], c["settlement_mm"][keep], c["az"][keep]):
            trace_rows.append({
                "case_id": c["case_id"],
                "family": c["family"],
                "amplitude_mm": c["amplitude_mm"],
                "distance_m": xi,
                "settlement_mm": si,
                "carbody_az_mps2": ai,
            })
    pd.DataFrame(metric_rows).to_csv(OUT_DIR / "transition_settlement_metrics.csv", index=False)
    pd.DataFrame(trace_rows).to_csv(OUT_DIR / "transition_settlement_traces_50_150m.csv", index=False)


def main() -> None:
    global RESULT_ROOT, OUT_DIR, FIGURE_PREFIX, FIGURE_TITLE
    parser = argparse.ArgumentParser(description="Plot transition settlement comparison figure.")
    parser.add_argument("--result-root", default=str(RESULT_ROOT), help="Directory containing settlement scan result folders.")
    parser.add_argument("--figure-prefix", default=FIGURE_PREFIX, help="Output figure filename prefix.")
    parser.add_argument("--title", default=FIGURE_TITLE, help="Figure title.")
    args = parser.parse_args()

    RESULT_ROOT = Path(args.result_root)
    OUT_DIR = RESULT_ROOT / "_publication_figures"
    FIGURE_PREFIX = args.figure_prefix
    FIGURE_TITLE = args.title

    cases = load_cases()
    if len(cases) != 9:
        raise RuntimeError(f"Expected 9 transition settlement cases, found {len(cases)}")
    export_source_data(cases)
    figure_path = make_figure(cases)
    print(f"Saved figure: {figure_path}")
    print(f"Saved source data: {OUT_DIR}")


if __name__ == "__main__":
    main()
