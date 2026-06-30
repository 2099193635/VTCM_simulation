from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


G = 9.80665
BASE_MC = 34000.0
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = PROJECT_ROOT / "results" / "ballast_condition_sync_eta_100m_scan"
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "sweeps" / "ballast_condition_sync_eta_100m_scan.yaml"
DEFAULT_OUT = DEFAULT_ROOT / "_comparison" / "publication_sync_eta_100m"


COLORS = {
    "baseline": "#4D4D4D",
    "loose": "#D55E00",
    "harden": "#0072B2",
    "neutral": "#777777",
    "unstable": "#B2182B",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.6,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.4,
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


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def case_metadata(manifest: dict) -> pd.DataFrame:
    rows = []
    common = manifest.get("common", {}) or {}
    default_speed = float(common.get("vx_set", 215.0))
    for case in manifest.get("cases", []):
        case_id = case["case_id"]
        args = case.get("case_args", {}) or {}
        defects = case.get("structure_defects", []) or []
        eta = np.nan
        count = 0
        start_m = np.nan
        if defects:
            defect = defects[0]
            eta = float(defect.get("stiffness_factor_eta_k", np.nan))
            count = int(defect.get("count", 0))
            start_m = float(defect.get("start_m", np.nan))
        speed = float(args.get("vx_set", default_speed))
        mass_factor = np.nan
        m = re.search(r"_w([0-9]p[0-9])_", case_id)
        if m:
            mass_factor = float(m.group(1).replace("p", "."))
        elif case_id in {"case_01_eta0p1_count167", "case_06_eta5_count167"}:
            mass_factor = 1.0
        elif case_id == "case_00_random_baseline":
            mass_factor = 1.0
        group = "baseline"
        if np.isfinite(eta):
            if eta < 1:
                group = "loose"
            elif eta > 1:
                group = "harden"
            else:
                group = "neutral"
        rows.append(
            {
                "case_id": case_id,
                "speed_kmh": speed,
                "mass_factor": mass_factor,
                "eta": eta,
                "eta_c": eta,
                "count": count,
                "start_m": start_m,
                "group": group,
                "note": case.get("note", ""),
            }
        )
    return pd.DataFrame(rows)


def enrich_summary(summary: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    df = summary.merge(meta, on=["case_id", "note"], how="left")
    df["Mc_base_effective"] = df["Mc_base"].fillna(BASE_MC)
    df["mass_factor"] = df["mass_factor"].fillna(df["Mc_base_effective"] / BASE_MC)
    metric_cols = [
        "carbody_az_rms_g",
        "carbody_ay_rms_g",
        "bogie1_az_rms_g",
        "wheelset1_az_rms_g",
        "wheelrail_fz_axle1_sum_rms_kn",
        "wheelrail_fz_axle1_sum_peak_kn",
        "structure_defect_ballast_condition_force_peak_kn",
    ]
    finite = np.ones(len(df), dtype=bool)
    for col in metric_cols:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy()
        finite &= np.isfinite(values)
    df["stable"] = finite & (df["carbody_az_rms_g"] < 1.0) & (df["wheelrail_fz_axle1_sum_peak_kn"] < 1.0e5)
    return df


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

    df["result_npz"] = df["result_npz"].map(relocate)
    missing = [path for path in df["result_npz"].map(Path) if not path.exists()]
    if missing:
        preview = "\n".join(str(path) for path in missing[:3])
        raise FileNotFoundError(f"Missing simulation_result.npz files after path relocation:\n{preview}")
    return df


def load_trace(npz_path: str | Path) -> pd.DataFrame:
    data = np.load(Path(npz_path), allow_pickle=True)
    distance = np.asarray(data.get("Irre_distance_m", data["Track_rel_mileage_m"]), dtype=float)
    accel_z_g = np.asarray(data["A"][:, 1], dtype=float) / G
    accel_y_g = np.asarray(data["A"][:, 0], dtype=float) / G
    force = np.asarray(data.get("Structure_defect_ballast_condition_FV_sum", np.zeros((len(distance), 2))), dtype=float)
    if force.ndim == 2:
        condition_force_kn = np.nanmax(np.abs(force), axis=1) / 1000.0
    else:
        condition_force_kn = np.abs(force) / 1000.0
    return pd.DataFrame(
        {
            "distance_m": distance,
            "carbody_az_g": accel_z_g,
            "carbody_ay_g": accel_y_g,
            "condition_force_kn": condition_force_kn,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.11, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom")


def shade_zone(ax: plt.Axes, start: float = 200.0, length: float = 100.2) -> None:
    ax.axvspan(start, start + length, color="#EFCB68", alpha=0.18, lw=0)
    ax.axvline(start, color="#9A6B00", lw=0.6, ls="--", alpha=0.8)


def plot_trace_panel(ax: plt.Axes, df: pd.DataFrame) -> pd.DataFrame:
    reps = [
        ("case_01_eta0p1_count167", r"loose, $\eta$=0.1", COLORS["loose"], 0.95, "-", 0.86, 2),
        ("case_06_eta5_count167", r"harden, $\eta$=5", COLORS["harden"], 0.95, "-", 0.86, 2),
        ("case_00_random_baseline", "baseline", "#111111", 1.25, "-.", 1.0, 5),
    ]
    source_rows = []
    for case_id, label, color, lw, ls, alpha, zorder in reps:
        row = df.loc[df["case_id"] == case_id].iloc[0]
        trace = load_trace(row["result_npz"])
        mask = trace["distance_m"].between(150, 330)
        ax.plot(
            trace.loc[mask, "distance_m"],
            trace.loc[mask, "carbody_az_g"],
            lw=lw,
            color=color,
            ls=ls,
            alpha=alpha,
            zorder=zorder,
            label=label,
        )
        out = trace.loc[mask].copy()
        out["case_id"] = case_id
        source_rows.append(out)
    shade_zone(ax)
    ax.axvline(160.0, color="#555555", lw=0.65, ls=":", alpha=0.9)
    ax.text(
        161.5,
        0.0128,
        "leading wheel\nreaches defect",
        fontsize=6.2,
        color="#444444",
        ha="left",
        va="top",
    )
    ax.axhline(0, color="#333333", lw=0.5, alpha=0.6)
    ax.set_xlim(150, 330)
    ax.set_ylabel(r"Carbody $a_z$ (g)")
    ax.set_title("Local carbody vertical acceleration around the ballast-condition window", loc="left")
    ax.legend(loc="upper right", frameon=False, ncol=3, handlelength=1.4, columnspacing=0.8)
    return pd.concat(source_rows, ignore_index=True)


def plot_eta_scan(ax: plt.Axes, df: pd.DataFrame) -> None:
    scan = df.loc[df["case_id"].str.match(r"case_0[1-7]_") & df["stable"]].sort_values("eta").copy()
    scan = scan.loc[scan["eta"] < 10].copy()
    baseline = df.loc[df["case_id"] == "case_00_random_baseline"].iloc[0]
    metrics = [
        ("carbody_az_rms_g", r"carbody $a_z$", "#34495E", "o"),
        ("bogie1_az_rms_g", r"bogie $a_z$", "#009E73", "s"),
        ("wheelset1_az_rms_g", r"wheelset $a_z$", "#7B3294", "^"),
    ]
    for col, label, color, marker in metrics:
        pct = (scan[col] / float(baseline[col]) - 1.0) * 100.0
        ax.plot(scan["eta"], pct, marker=marker, ms=3.7, lw=1.0, color=color, label=label)
    ax.axhline(0, color="#333333", lw=0.5, alpha=0.7)
    ax.set_ylim(-1.2, 9.6)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax.get_xaxis().set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax.set_xlabel(r"Synchronous multiplier $\eta_k=\eta_c$")
    ax.set_ylabel("RMS change vs baseline (%)")
    ax.set_title("Softening amplifies vibration in the main scan", loc="left")
    ax.legend(loc="upper right", frameon=False, ncol=1)


def plot_force_scan(ax: plt.Axes, df: pd.DataFrame) -> None:
    scan = df.loc[df["case_id"].str.match(r"case_0[1-7]_") & df["stable"]].sort_values("eta").copy()
    scan = scan.loc[scan["eta"] < 10].copy()
    ax.plot(
        scan["eta"],
        scan["structure_defect_ballast_condition_force_peak_kn"],
        marker="o",
        ms=4,
        lw=1.0,
        color="#2F4B7C",
    )
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax.get_xaxis().set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax.set_xlabel(r"Synchronous multiplier $\eta_k=\eta_c$")
    ax.set_ylabel("Peak ballast-state force (kN)")
    ax.set_title("Local force separates the softened regime", loc="left")


def heatmap_grid(df: pd.DataFrame, group: str, eta: float, metric: str) -> pd.DataFrame:
    part = df.loc[(df["group"] == group) & np.isclose(df["eta"], eta) & df["stable"]].copy()
    return part.pivot_table(index="mass_factor", columns="speed_kmh", values=metric, aggfunc="mean").sort_index(ascending=False)


def draw_heatmap(ax: plt.Axes, grid: pd.DataFrame, title: str, cmap: str, vmin: float, vmax: float, cbar_label: str) -> None:
    im = ax.imshow(grid.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels([f"{int(v)}" for v in grid.columns])
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in grid.index])
    ax.set_xlabel("Speed (km h$^{-1}$)")
    ax.set_ylabel("Carbody mass factor")
    ax.set_title(title, loc="left")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.4, color="#202020")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(labelsize=6.3, width=0.5, length=2)


def plot_speed_trends(ax: plt.Axes, df: pd.DataFrame) -> None:
    rows = []
    for group, eta, color, label in [
        ("loose", 0.1, COLORS["loose"], r"loose $\eta$=0.1"),
        ("harden", 5.0, COLORS["harden"], r"harden $\eta$=5"),
    ]:
        part = df.loc[(df["group"] == group) & np.isclose(df["eta"], eta) & df["stable"]].copy()
        part = part.sort_values(["mass_factor", "speed_kmh"])
        for mass, sub in part.groupby("mass_factor"):
            ls = {0.9: "--", 1.0: "-", 1.1: ":"}.get(round(float(mass), 1), "-")
            ax.plot(
                sub["speed_kmh"],
                sub["wheelset1_az_rms_g"],
                marker="o",
                ms=3.5,
                lw=1.0,
                ls=ls,
                color=color,
                alpha=0.9,
                label=f"{label}, {mass:.1f}x",
            )
            rows.append(sub)
    ax.set_xlabel("Speed (km h$^{-1}$)")
    ax.set_ylabel(r"Wheelset $a_z$ RMS (g)")
    ax.set_title("Speed dominates wheelset vibration across both structural states", loc="left")
    ax.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.5, columnspacing=0.8)


def welch_psd(signal: np.ndarray, fs: float, nperseg: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 16:
        return np.array([]), np.array([])
    nperseg = min(nperseg, x.size)
    step = max(1, nperseg // 2)
    window = np.hanning(nperseg)
    scale = fs * np.sum(window**2)
    spectra = []
    for start in range(0, x.size - nperseg + 1, step):
        seg = x[start : start + nperseg]
        seg = seg - np.mean(seg)
        fft = np.fft.rfft(seg * window)
        psd = (np.abs(fft) ** 2) / scale
        if psd.size > 2:
            psd[1:-1] *= 2.0
        spectra.append(psd)
    if not spectra:
        return np.array([]), np.array([])
    freq = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return freq, np.mean(np.vstack(spectra), axis=0)


def next_lower_power_of_two(n: int) -> int:
    if n < 2:
        return 1
    return 1 << (int(n).bit_length() - 1)


def resample_to_uniform_distance(distance_m: np.ndarray, signal: np.ndarray, dx_m: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    distance = np.asarray(distance_m, dtype=float)
    y = np.asarray(signal, dtype=float)
    n = min(distance.size, y.size)
    distance = distance[:n]
    y = y[:n]
    valid = np.isfinite(distance) & np.isfinite(y)
    if np.count_nonzero(valid) < 8:
        return np.array([]), np.array([])
    distance = distance[valid]
    y = y[valid]
    order = np.argsort(distance)
    distance = distance[order]
    y = y[order]
    distance_unique, idx = np.unique(distance, return_index=True)
    y_unique = y[idx]
    if distance_unique.size < 8:
        return np.array([]), np.array([])
    grid = np.arange(distance_unique[0], distance_unique[-1] + 0.5 * dx_m, dx_m)
    return grid, np.interp(grid, distance_unique, y_unique)


def welch_spatial_psd(signal: np.ndarray, dx_m: float, nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(signal, dtype=float)
    finite = np.isfinite(y)
    if np.count_nonzero(finite) < 8:
        return np.array([]), np.array([])
    y = y.copy()
    y[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), y[finite])
    y = y - np.mean(y)
    n = len(y)
    if nperseg is None:
        nperseg = min(4096, next_lower_power_of_two(n))
    nperseg = int(max(8, min(nperseg, n)))
    step = max(1, nperseg // 2)
    window = np.hanning(nperseg)
    scale = (1.0 / dx_m) * np.sum(window**2)
    spectra = []
    for start in range(0, n - nperseg + 1, step):
        seg = y[start : start + nperseg]
        seg = seg - np.mean(seg)
        fft = np.fft.rfft(seg * window)
        psd = (np.abs(fft) ** 2) / scale
        if psd.size > 2:
            psd[1:-1] *= 2.0
        spectra.append(psd)
    if not spectra:
        return np.array([]), np.array([])
    freq = np.fft.rfftfreq(nperseg, d=dx_m)
    return freq, np.mean(np.vstack(spectra), axis=0)


def carbody_az_spatial_psd_for_case(
    df: pd.DataFrame,
    case_id: str,
    dx_m: float = 0.25,
    buffer_time_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    row = df.loc[df["case_id"] == case_id].iloc[0]
    data = np.load(Path(row["result_npz"]), allow_pickle=True)
    distance = np.asarray(data.get("Irre_distance_m", data["Track_rel_mileage_m"]), dtype=float)
    sig = np.asarray(data["A"][:, 1], dtype=float) / G
    speed_kmh = float(row.get("speed_kmh", np.nan))
    if np.isfinite(speed_kmh) and speed_kmh > 0 and distance.size:
        buffer_m = (speed_kmh / 3.6) * buffer_time_s
        use = distance >= distance[0] + buffer_m
        distance = distance[use]
        sig = sig[use]
    else:
        buffer_m = np.nan
    _, sig_grid = resample_to_uniform_distance(distance, sig, dx_m=dx_m)
    freq, psd = welch_spatial_psd(sig_grid, dx_m=dx_m)
    keep = (freq >= 0.004) & (freq <= 2.0) & np.isfinite(psd) & (psd > 0)
    meta = {
        "case_id": case_id,
        "speed_kmh": speed_kmh,
        "mass_factor": float(row.get("mass_factor", np.nan)),
        "eta": float(row.get("eta", np.nan)),
        "spatial_resample_dx_m": dx_m,
        "removed_buffer_time_s": buffer_time_s,
        "removed_buffer_distance_m": buffer_m,
    }
    return freq[keep], psd[keep], meta


def plot_psd_panel(ax: plt.Axes, df: pd.DataFrame) -> pd.DataFrame:
    reps = [
        ("case_01_eta0p1_count167", r"loose, $\eta$=0.1", "#D55E00", 1.25, "-", 0.9, 5),
        ("case_06_eta5_count167", r"harden, $\eta$=5", "#0072B2", 1.05, (0, (2.2, 1.4)), 0.72, 4),
        ("case_00_random_baseline", "baseline", "#111111", 1.15, (0, (5.0, 2.4, 1.2, 2.4)), 0.8, 3),
    ]
    rows = []
    dx_m = 0.25
    buffer_time_s = 2.0
    for case_id, label, color, lw, ls, alpha, zorder in reps:
        freq, psd, meta = carbody_az_spatial_psd_for_case(df, case_id, dx_m=dx_m, buffer_time_s=buffer_time_s)
        ax.loglog(
            freq,
            psd,
            color=color,
            lw=lw,
            ls=ls,
            alpha=alpha,
            zorder=zorder,
            label=label,
        )
        rows.append(
            pd.DataFrame(
                {
                    "case_id": case_id,
                    "spatial_frequency_1pm": freq,
                    "carbody_az_spatial_psd_g2_per_1pm": psd,
                    **{k: v for k, v in meta.items() if k != "case_id"},
                }
            )
        )
    ax.set_xlim(0.004, 2.0)
    ax.set_xlabel("Spatial frequency (m$^{-1}$)")
    ax.set_ylabel(r"PSD of carbody $a_z$ (g$^2$/(m$^{-1}$))")
    ax.set_title("Carbody vertical acceleration spatial PSD", loc="left")
    ax.grid(which="both", axis="both", color="#D8D8D8", lw=0.28, alpha=0.65)
    ax.legend(loc="upper right", frameon=False, ncol=3, handlelength=1.8, columnspacing=0.8)
    return pd.concat(rows, ignore_index=True)


def format_spatial_psd_axis(ax: plt.Axes, title: str) -> None:
    ax.set_xlim(0.004, 2.0)
    ax.set_xlabel("Spatial frequency (m$^{-1}$)")
    ax.set_ylabel(r"PSD of carbody $a_z$ (g$^2$/(m$^{-1}$))")
    ax.set_title(title, loc="left")
    ax.grid(which="both", axis="both", color="#D8D8D8", lw=0.28, alpha=0.65)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, ncol=1, handlelength=1.8)


def plot_loose_psd_comparison(
    ax: plt.Axes,
    df: pd.DataFrame,
    reps: list[tuple[str, str, str, str, float]],
    title: str,
) -> pd.DataFrame:
    rows = []
    for case_id, label, color, ls, lw in reps:
        freq, psd, meta = carbody_az_spatial_psd_for_case(df, case_id)
        ax.loglog(freq, psd, color=color, ls=ls, lw=lw, alpha=0.92, label=label)
        rows.append(
            pd.DataFrame(
                {
                    "case_id": case_id,
                    "label": label,
                    "spatial_frequency_1pm": freq,
                    "carbody_az_spatial_psd_g2_per_1pm": psd,
                    **{k: v for k, v in meta.items() if k != "case_id"},
                }
            )
        )
    format_spatial_psd_axis(ax, title)
    return pd.concat(rows, ignore_index=True)


def plot_case_trace_comparison(
    ax: plt.Axes,
    df: pd.DataFrame,
    reps: list[tuple[str, str, str, str, float]],
    title: str,
) -> pd.DataFrame:
    rows = []
    for case_id, label, color, ls, lw in reps:
        row = df.loc[df["case_id"] == case_id].iloc[0]
        trace = load_trace(row["result_npz"])
        mask = trace["distance_m"].between(150, 330)
        ax.plot(
            trace.loc[mask, "distance_m"],
            trace.loc[mask, "carbody_az_g"],
            color=color,
            ls=ls,
            lw=lw,
            alpha=0.92,
            label=label,
        )
        out = trace.loc[mask].copy()
        out["case_id"] = case_id
        out["label"] = label
        rows.append(out)
    shade_zone(ax)
    ax.axhline(0, color="#333333", lw=0.5, alpha=0.6)
    ax.set_xlim(150, 330)
    ax.set_ylabel(r"Carbody $a_z$ (g)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.7)
    ax.grid(axis="y", color="#D8D8D8", lw=0.35, alpha=0.75)
    return pd.concat(rows, ignore_index=True)


def plot_loose_rms_metrics(
    ax: plt.Axes,
    df: pd.DataFrame,
    reps: list[tuple[str, str, str, str, float]],
    x_values: list[float],
    x_label: str,
    title: str,
) -> pd.DataFrame:
    rows = []
    for case_id, label, color, _, _ in reps:
        row = df.loc[df["case_id"] == case_id].iloc[0].copy()
        row["plot_label"] = label
        rows.append(row)
    part = pd.DataFrame(rows)
    ax.plot(x_values, part["carbody_az_rms_g"] * 1000, marker="o", ms=4, lw=1.1, color="#34495E", label=r"carbody $a_z$")
    ax.plot(x_values, part["bogie1_az_rms_g"] * 1000, marker="s", ms=3.8, lw=1.1, color="#009E73", label=r"bogie $a_z$")
    ax.set_xlabel(x_label)
    ax.set_ylabel("RMS (10$^{-3}$ g)")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", color="#D8D8D8", lw=0.35, alpha=0.75)
    ax.legend(loc="upper left", frameon=False, ncol=1)
    return part


def plot_loose_force_metrics(
    ax: plt.Axes,
    df: pd.DataFrame,
    reps: list[tuple[str, str, str, str, float]],
    x_values: list[float],
    x_label: str,
    title: str,
) -> pd.DataFrame:
    rows = []
    for case_id, label, color, _, _ in reps:
        row = df.loc[df["case_id"] == case_id].iloc[0].copy()
        row["plot_label"] = label
        rows.append(row)
    part = pd.DataFrame(rows)
    ax.plot(x_values, part["wheelset1_az_rms_g"], marker="^", ms=4, lw=1.1, color="#7B3294", label=r"wheelset $a_z$")
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"Wheelset $a_z$ RMS (g)")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", color="#D8D8D8", lw=0.35, alpha=0.75)
    ax2 = ax.twinx()
    ax2.plot(
        x_values,
        part["structure_defect_ballast_condition_force_peak_kn"],
        marker="o",
        ms=3.8,
        lw=1.0,
        color="#A6611A",
        label="peak ballast-state force",
    )
    ax2.set_ylabel("Peak ballast-state force (kN)")
    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="upper left", frameon=False, ncol=1)
    return part


def save_loose_analysis_figures(df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    speed_reps = [
        ("case_09_loose_eta0p1_v160_w1p0_count167", "160 km h$^{-1}$", "#2F4B7C", "-", 1.15),
        ("case_01_eta0p1_count167", "215 km h$^{-1}$", "#D55E00", "-", 1.25),
        ("case_14_loose_eta0p1_v250_w1p0_count167", "250 km h$^{-1}$", "#7B3294", "-", 1.15),
    ]
    mass_reps = [
        ("case_11_loose_eta0p1_v215_w0p9_count167", "0.9$\\times$ carbody mass", "#5AA6C8", "-", 1.15),
        ("case_01_eta0p1_count167", "1.0$\\times$ carbody mass", "#D55E00", "-", 1.25),
        ("case_12_loose_eta0p1_v215_w1p1_count167", "1.1$\\times$ carbody mass", "#8C2D04", "-", 1.15),
    ]

    speed_values = [160, 215, 250]
    mass_values = [0.9, 1.0, 1.1]

    fig_speed = plt.figure(figsize=(7.1, 5.9))
    gs_speed = fig_speed.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.34)
    ax_speed_a = fig_speed.add_subplot(gs_speed[0, 0])
    ax_speed_b = fig_speed.add_subplot(gs_speed[0, 1])
    ax_speed_c = fig_speed.add_subplot(gs_speed[1, 0])
    ax_speed_d = fig_speed.add_subplot(gs_speed[1, 1])
    speed_trace = plot_case_trace_comparison(
        ax_speed_a,
        df,
        speed_reps,
        "Local carbody response under loose ballast",
    )
    speed_source = plot_loose_psd_comparison(
        ax_speed_b,
        df,
        speed_reps,
        "Spatial PSD separates the speed cases",
    )
    speed_rms = plot_loose_rms_metrics(
        ax_speed_c,
        df,
        speed_reps,
        speed_values,
        "Speed (km h$^{-1}$)",
        "Carbody and bogie vibration increase with speed",
    )
    speed_force = plot_loose_force_metrics(
        ax_speed_d,
        df,
        speed_reps,
        speed_values,
        "Speed (km h$^{-1}$)",
        "Wheelset vibration and local force response",
    )
    for label, ax in zip(["a", "b", "c", "d"], [ax_speed_a, ax_speed_b, ax_speed_c, ax_speed_d]):
        panel_label(ax, label)
    fig_speed.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.09)
    save_outputs(fig_speed, out_dir, "loose_ballast_speed_analysis")
    plt.close(fig_speed)
    speed_trace.to_csv(out_dir / "source_data_loose_ballast_speed_traces.csv", index=False)
    speed_source.to_csv(out_dir / "source_data_loose_ballast_speed_spatial_psd.csv", index=False)
    pd.concat([speed_rms, speed_force]).drop_duplicates(subset=["case_id"]).to_csv(
        out_dir / "source_data_loose_ballast_speed_metrics.csv", index=False
    )

    fig_mass = plt.figure(figsize=(7.1, 5.9))
    gs_mass = fig_mass.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.34)
    ax_mass_a = fig_mass.add_subplot(gs_mass[0, 0])
    ax_mass_b = fig_mass.add_subplot(gs_mass[0, 1])
    ax_mass_c = fig_mass.add_subplot(gs_mass[1, 0])
    ax_mass_d = fig_mass.add_subplot(gs_mass[1, 1])
    mass_trace = plot_case_trace_comparison(
        ax_mass_a,
        df,
        mass_reps,
        "Local carbody response under loose ballast",
    )
    mass_source = plot_loose_psd_comparison(
        ax_mass_b,
        df,
        mass_reps,
        "Spatial PSD is weakly affected by carbody mass",
    )
    mass_rms = plot_loose_rms_metrics(
        ax_mass_c,
        df,
        mass_reps,
        mass_values,
        "Carbody mass factor",
        "Heavier carbody lowers carbody vibration",
    )
    mass_force = plot_loose_force_metrics(
        ax_mass_d,
        df,
        mass_reps,
        mass_values,
        "Carbody mass factor",
        "Mass shifts local force more than carbody PSD",
    )
    for label, ax in zip(["a", "b", "c", "d"], [ax_mass_a, ax_mass_b, ax_mass_c, ax_mass_d]):
        panel_label(ax, label)
    fig_mass.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.09)
    save_outputs(fig_mass, out_dir, "loose_ballast_mass_analysis")
    plt.close(fig_mass)
    mass_trace.to_csv(out_dir / "source_data_loose_ballast_mass_traces.csv", index=False)
    mass_source.to_csv(out_dir / "source_data_loose_ballast_mass_spatial_psd.csv", index=False)
    pd.concat([mass_rms, mass_force]).drop_duplicates(subset=["case_id"]).to_csv(
        out_dir / "source_data_loose_ballast_mass_metrics.csv", index=False
    )

    return out_dir / "loose_ballast_speed_analysis.png", out_dir / "loose_ballast_mass_analysis.png"


def save_outputs(fig: plt.Figure, out_dir: Path, base_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / base_name
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def make_figure(result_root: Path, manifest_path: Path, out_dir: Path) -> Path:
    set_style()
    manifest = load_manifest(manifest_path)
    meta = case_metadata(manifest)
    summary = pd.read_csv(result_root / "_comparison" / "sweep_response_summary.csv")
    summary = relocate_result_npz_paths(summary, result_root)
    df = enrich_summary(summary, meta)

    fig = plt.figure(figsize=(7.25, 9.65))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.02, 1.02, 0.92], hspace=0.53, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])
    ax_f = fig.add_subplot(gs[3, :])

    trace_source = plot_trace_panel(ax_a, df)
    plot_eta_scan(ax_b, df)
    plot_force_scan(ax_c, df)

    loose_grid = heatmap_grid(df, "loose", 0.1, "carbody_az_rms_g") * 1000
    harden_grid = heatmap_grid(df, "harden", 5.0, "carbody_az_rms_g") * 1000
    hmin = min(np.nanmin(loose_grid.to_numpy()), np.nanmin(harden_grid.to_numpy()))
    hmax = max(np.nanmax(loose_grid.to_numpy()), np.nanmax(harden_grid.to_numpy()))
    draw_heatmap(ax_d, loose_grid, r"Strong loose ballast: carbody $a_z$ RMS", "Oranges", hmin, hmax, "10$^{-3}$ g")
    draw_heatmap(ax_e, harden_grid, r"Hardened ballast: carbody $a_z$ RMS", "Blues", hmin, hmax, "10$^{-3}$ g")
    psd_source = plot_psd_panel(ax_f, df)

    for label, ax in zip(["a", "b", "c", "d", "e", "f"], [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]):
        panel_label(ax, label)
        if ax not in [ax_d, ax_e]:
            ax.grid(axis="y", color="#D8D8D8", lw=0.35, alpha=0.75)

    fig.text(
        0.02,
        0.01,
        "All stable panels use a fixed random-irregularity realization and a 100.2 m ballast-condition window starting at 200 m. "
        "The eta=10 case is excluded because it produced numerical divergence.",
        ha="left",
        va="bottom",
        fontsize=6.3,
        color="#333333",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.97, bottom=0.075)

    save_outputs(fig, out_dir, "ballast_sync_eta_100m_comparison")
    plt.close(fig)
    speed_png, mass_png = save_loose_analysis_figures(df, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    source = df.copy()
    source.to_csv(out_dir / "source_data_case_metrics.csv", index=False)
    trace_source.to_csv(out_dir / "source_data_representative_traces.csv", index=False)
    psd_source.to_csv(out_dir / "source_data_carbody_az_psd.csv", index=False)
    loose_grid.to_csv(out_dir / "source_data_loose_heatmap_carbody_az_milli_g.csv")
    harden_grid.to_csv(out_dir / "source_data_harden_heatmap_carbody_az_milli_g.csv")
    print(f"Saved loose speed analysis: {speed_png}")
    print(f"Saved loose mass analysis: {mass_png}")
    return out_dir / "ballast_sync_eta_100m_comparison.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publication figure for 100 m synchronized ballast eta scan.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    png = make_figure(args.result_root, args.manifest, args.out_dir)
    print(f"Saved figure preview: {png}")


if __name__ == "__main__":
    main()
