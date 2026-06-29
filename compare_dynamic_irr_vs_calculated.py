from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, filtfilt, welch

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


@dataclass
class Config:
    sim_npz_path: Path = Path(
        r"results\default_project\高速客车-外部导入-vehicle-standard-20260319_215206\files\simulation_result.npz"
    )
    sim_json_path: Path = Path(
        r"results\default_project\高速客车-外部导入-vehicle-standard-20260319_215206\files\argparse_params.json"
    )
    measured_csv_path: Path = Path(
        r"preprocessing\动检数据\呼局\20210416\处理后\动检上行20210416-238-363.processed.csv"
    )

    sim_channel: str = "wheelset_1_axle_z"
    measured_channel: str = "measured_left_irr_mm"

    lead_time_s: float = 2.0
    search_window_m: float = 400.0
    target_spacing_m: float = 0.25
    lowpass_cutoff_hz: float = 25.0
    max_start_candidates: int = 400

    figure_width_in: float = 7.2
    figure_height_in: float = 5.6
    figure_dpi: int = 140
    export_dpi: int = 300
    export_png: bool = True
    export_pdf: bool = True
    output_dir: Path = Path("results/figures")
    output_stem: str = "dynamic_irregularity_comparison"


def setup_publication_style() -> None:
    """Set matplotlib style for journal-grade figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "lines.markersize": 2.5,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.30,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def compute_metrics(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return correlation, RMSE, and NRMSE after mean removal."""
    n = min(len(x), len(y))
    x0 = np.asarray(x[:n], dtype=float) - np.mean(x[:n])
    y0 = np.asarray(y[:n], dtype=float) - np.mean(y[:n])
    corr = float(np.corrcoef(x0, y0)[0, 1])
    rmse = float(np.sqrt(np.mean((x0 - y0) ** 2)))
    nrmse = float(rmse / (np.std(y0) + 1e-12))
    return corr, rmse, nrmse


def lowpass_fft(x: np.ndarray, dt: float, fc: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    freq = np.fft.rfftfreq(len(x), d=dt)
    xf = np.fft.rfft(x)
    xf[freq > fc] = 0.0
    return np.fft.irfft(xf, n=len(x))


def lowpass_filter(x: np.ndarray, fs: float, dt: float, fc: float, order: int = 4) -> np.ndarray:
    if SCIPY_AVAILABLE:
        wn = fc / (0.5 * fs)
        wn = min(max(wn, 1e-6), 0.999999)
        b, a = butter(order, wn, btype="low")
        return filtfilt(b, a, x)
    return lowpass_fft(x, dt, fc)


def fft_power(x: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    x0 = np.asarray(x, dtype=float) - np.mean(x)
    freq = np.fft.rfftfreq(len(x0), d=dt)
    xf = np.fft.rfft(x0)
    power = (np.abs(xf) ** 2) / max(len(x0), 1)
    return freq, power


def build_sim_channels(x: np.ndarray, params_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Build simulation candidate channels using English channel keys."""
    z_cols = {
        "wheelset_1_axle_z": 16,
        "wheelset_2_axle_z": 21,
        "wheelset_3_axle_z": 26,
        "wheelset_4_axle_z": 31,
    }
    roll_cols = {
        "wheelset_1_roll": 17,
        "wheelset_2_roll": 22,
        "wheelset_3_roll": 27,
        "wheelset_4_roll": 32,
    }

    channels = {name: np.asarray(x[:, col], dtype=float) for name, col in z_cols.items()}

    dw = None
    try:
        if "veh_1" in params_df.columns:
            veh_cfg = params_df["veh_1"].iloc[0]
            if isinstance(veh_cfg, dict) and "dw" in veh_cfg:
                dw = float(veh_cfg["dw"])
    except Exception:
        dw = None

    if dw is not None:
        for i in range(1, 5):
            z_key = f"wheelset_{i}_axle_z"
            r_key = f"wheelset_{i}_roll"
            z_val = np.asarray(x[:, z_cols[z_key]], dtype=float)
            r_val = np.asarray(x[:, roll_cols[r_key]], dtype=float)
            channels[f"wheelset_{i}_left_equiv_z"] = z_val + dw * r_val
            channels[f"wheelset_{i}_right_equiv_z"] = z_val - dw * r_val

    return channels


def build_measured_channels(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Build measured candidate channels using English keys."""
    mileage_km = pd.to_numeric(df["里程"], errors="coerce").to_numpy(dtype=float)
    left_irr_mm = pd.to_numeric(df["左高低"], errors="coerce").to_numpy(dtype=float)
    right_irr_mm = pd.to_numeric(df["右高低"], errors="coerce").to_numpy(dtype=float)

    channels = {
        "measured_left_irr_mm": left_irr_mm,
        "measured_right_irr_mm": right_irr_mm,
    }
    return mileage_km, channels


def align_and_compare(cfg: Config) -> Tuple[dict, pd.DataFrame, dict]:
    data = np.load(cfg.sim_npz_path)
    params_df = pd.read_json(cfg.sim_json_path)
    measured_df = pd.read_csv(cfg.measured_csv_path)

    x = data["X"]
    dt = float(data["dt"])

    sim_channels = build_sim_channels(x, params_df)
    meas_mileage, meas_channels = build_measured_channels(measured_df)

    if cfg.sim_channel not in sim_channels:
        raise KeyError(f"Unknown simulation channel: {cfg.sim_channel}")
    if cfg.measured_channel not in meas_channels:
        raise KeyError(f"Unknown measured channel: {cfg.measured_channel}")

    sim_signal = np.asarray(sim_channels[cfg.sim_channel], dtype=float)
    measured_signal_m = np.asarray(meas_channels[cfg.measured_channel], dtype=float) / 1000.0

    start_mileage_km = float(params_df["start_mileage"].iloc[0])
    speed_kmh = float(params_df["vx_set"].iloc[0])
    speed_mps = speed_kmh / 3.6

    sim_t = np.arange(len(sim_signal), dtype=float) * dt
    valid_t = sim_t >= cfg.lead_time_s
    if np.count_nonzero(valid_t) < 10:
        raise ValueError("Too few simulation points remain after lead-time trimming.")

    sim_mileage_all = start_mileage_km + speed_kmh * sim_t / 3600.0
    sim_mileage_cut = sim_mileage_all[valid_t]
    sim_signal_cut = sim_signal[valid_t]

    spacing_km = cfg.target_spacing_m / 1000.0
    sim_mileage_ds = np.arange(sim_mileage_cut[0], sim_mileage_cut[-1] + 1e-12, spacing_km)
    sim_ds = np.interp(sim_mileage_ds, sim_mileage_cut, sim_signal_cut)
    sim_rel_km = sim_mileage_ds - sim_mileage_ds[0]

    dt_cmp = cfg.target_spacing_m / speed_mps
    fs_cmp = 1.0 / dt_cmp

    valid_meas = np.isfinite(meas_mileage) & np.isfinite(measured_signal_m)
    m_km = meas_mileage[valid_meas]
    y_m = measured_signal_m[valid_meas]

    order = np.argsort(m_km)
    m_km = m_km[order]
    y_m = y_m[order]

    m_unique, first_idx = np.unique(m_km, return_index=True)
    y_unique = y_m[first_idx]
    if len(m_unique) < 2:
        raise ValueError("Measured mileage has insufficient valid points.")

    center = float(sim_mileage_ds[0])
    candidate_starts = m_unique[
        (m_unique >= center - cfg.search_window_m / 1000.0)
        & (m_unique <= center + cfg.search_window_m / 1000.0)
    ]
    if len(candidate_starts) == 0:
        raise ValueError("No valid measured start candidates in the search window.")

    if len(candidate_starts) > cfg.max_start_candidates:
        step = int(np.ceil(len(candidate_starts) / cfg.max_start_candidates))
        candidate_starts = candidate_starts[::step]

    sim_raw = sim_ds - np.mean(sim_ds)
    sim_lp = lowpass_filter(sim_raw, fs_cmp, dt_cmp, cfg.lowpass_cutoff_hz)

    best = None
    records = []

    for s0 in candidate_starts:
        measured_axis = s0 + sim_rel_km
        if measured_axis[0] < m_unique[0] or measured_axis[-1] > m_unique[-1]:
            continue

        measured_raw = np.interp(measured_axis, m_unique, y_unique)
        measured_raw = measured_raw - np.mean(measured_raw)
        measured_lp = lowpass_filter(measured_raw, fs_cmp, dt_cmp, cfg.lowpass_cutoff_hz)

        corr_raw, rmse_raw, nrmse_raw = compute_metrics(sim_raw, measured_raw)
        corr_lp, rmse_lp, nrmse_lp = compute_metrics(sim_lp, measured_lp)
        score = corr_lp - 0.15 * nrmse_lp

        rec = {
            "sim_channel": cfg.sim_channel,
            "measured_channel": cfg.measured_channel,
            "start_km": float(s0),
            "shift_m": float((s0 - center) * 1000.0),
            "corr_raw": corr_raw,
            "rmse_raw_m": rmse_raw,
            "nrmse_raw": nrmse_raw,
            "corr_lp": corr_lp,
            "rmse_lp_m": rmse_lp,
            "nrmse_lp": nrmse_lp,
            "score": float(score),
        }
        records.append(rec)

        if (best is None) or (score > best["score"]):
            best = {
                **rec,
                "axis_km": measured_axis,
                "sim_raw": sim_raw,
                "measured_raw": measured_raw,
                "sim_lp": sim_lp,
                "measured_lp": measured_lp,
            }

    if best is None:
        raise RuntimeError("No valid alignment candidate produced a usable comparison.")

    scan_df = pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)

    analysis = {
        "dt_original_s": dt,
        "fs_original_hz": 1.0 / dt,
        "dt_compare_s": dt_cmp,
        "fs_compare_hz": fs_cmp,
        "speed_kmh": speed_kmh,
        "n_points_compare": len(sim_raw),
        "scipy_available": SCIPY_AVAILABLE,
    }

    return best, scan_df, analysis


def make_figure(best: dict, cfg: Config, analysis: dict) -> plt.Figure:
    """Create a 2x2 publication-style figure for the matched comparison."""
    axis_km = best["axis_km"]
    sim_raw_mm = best["sim_raw"] * 1000.0
    measured_raw_mm = best["measured_raw"] * 1000.0
    sim_lp_mm = best["sim_lp"] * 1000.0
    measured_lp_mm = best["measured_lp"] * 1000.0

    freq_s, power_s = fft_power(best["sim_raw"], analysis["dt_compare_s"])
    freq_m, power_m = fft_power(best["measured_raw"], analysis["dt_compare_s"])
    amp_s = np.sqrt(power_s)
    amp_m = np.sqrt(power_m)

    fig, axes = plt.subplots(2, 2, figsize=(cfg.figure_width_in, cfg.figure_height_in), dpi=cfg.figure_dpi)
    ax1, ax2, ax3, ax4 = axes.ravel()

    color_sim = "#1f77b4"
    color_mea = "#d62728"

    ax1.plot(axis_km, sim_raw_mm, color=color_sim, lw=1.0, label="Simulation (raw)")
    ax1.plot(axis_km, measured_raw_mm, color=color_mea, lw=1.0, alpha=0.9, label="Measured (raw)")
    ax1.set_xlabel("Mileage (km)")
    ax1.set_ylabel("Dynamic irregularity (mm)")
    ax1.set_title("(a) Raw signals", loc="left")
    ax1.grid(True)
    ax1.legend(frameon=False, loc="upper right")

    metric_text = (
        f"Raw: r = {best['corr_raw']:.3f}, NRMSE = {best['nrmse_raw']:.3f}\n"
        f"LP:   r = {best['corr_lp']:.3f}, NRMSE = {best['nrmse_lp']:.3f}\n"
        f"Shift = {best['shift_m']:.1f} m"
    )
    ax1.text(
        0.015,
        0.98,
        metric_text,
        transform=ax1.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.6", "boxstyle": "round,pad=0.2"},
    )

    ax2.plot(axis_km, sim_lp_mm, color=color_sim, lw=1.0, label="Simulation (low-pass)")
    ax2.plot(axis_km, measured_lp_mm, color=color_mea, lw=1.0, alpha=0.9, label="Measured (low-pass)")
    ax2.set_xlabel("Mileage (km)")
    ax2.set_ylabel("Dynamic irregularity (mm)")
    ax2.set_title(f"(b) Low-pass filtered (fc = {cfg.lowpass_cutoff_hz:.1f} Hz)", loc="left")
    ax2.grid(True)
    ax2.legend(frameon=False, loc="upper right")

    ax3.plot(freq_s, amp_s, color=color_sim, lw=1.0, label="Simulation")
    ax3.plot(freq_m, amp_m, color=color_mea, lw=1.0, alpha=0.9, label="Measured")
    ax3.axvline(cfg.lowpass_cutoff_hz, color="k", ls="--", lw=0.8, label="Cutoff")
    ax3.set_xlim(0.0, min(100.0, analysis["fs_compare_hz"] / 2.0))
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Amplitude (a.u.)")
    ax3.set_title("(c) FFT amplitude spectrum", loc="left")
    ax3.grid(True)
    ax3.legend(frameon=False, loc="upper right")

    if SCIPY_AVAILABLE:
        nperseg = min(4096, len(best["sim_raw"]))
        fw_s, psd_s = welch(best["sim_raw"], fs=analysis["fs_compare_hz"], nperseg=nperseg)
        fw_m, psd_m = welch(best["measured_raw"], fs=analysis["fs_compare_hz"], nperseg=nperseg)
        ax4.semilogy(fw_s, psd_s + 1e-20, color=color_sim, lw=1.0, label="Simulation")
        ax4.semilogy(fw_m, psd_m + 1e-20, color=color_mea, lw=1.0, alpha=0.9, label="Measured")
    else:
        ax4.semilogy(freq_s, power_s + 1e-20, color=color_sim, lw=1.0, label="Simulation")
        ax4.semilogy(freq_m, power_m + 1e-20, color=color_mea, lw=1.0, alpha=0.9, label="Measured")

    ax4.axvline(cfg.lowpass_cutoff_hz, color="k", ls="--", lw=0.8, label="Cutoff")
    ax4.set_xlim(0.0, min(100.0, analysis["fs_compare_hz"] / 2.0))
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("PSD")
    ax4.set_title("(d) Power spectral density", loc="left")
    ax4.grid(True)
    ax4.legend(frameon=False, loc="upper right")

    for ax in (ax1, ax2, ax3, ax4):
        ax.tick_params(direction="in", length=3.0, width=0.8)

    fig.tight_layout()
    return fig


def main() -> None:
    cfg = Config()
    setup_publication_style()

    best, scan_df, analysis = align_and_compare(cfg)

    print("=" * 72)
    print("Dynamic Irregularity Comparison Summary")
    print("=" * 72)
    print(f"Simulation channel      : {cfg.sim_channel}")
    print(f"Measured channel        : {cfg.measured_channel}")
    print(f"Lead-time trim          : {cfg.lead_time_s:.2f} s")
    print(f"Target spacing          : {cfg.target_spacing_m:.3f} m")
    print(f"Best aligned start      : {best['start_km']:.6f} km")
    print(f"Shift vs simulation     : {best['shift_m']:.2f} m")
    print(f"Raw correlation         : {best['corr_raw']:.4f}")
    print(f"Raw NRMSE               : {best['nrmse_raw']:.4f}")
    print(f"Low-pass correlation    : {best['corr_lp']:.4f}")
    print(f"Low-pass NRMSE          : {best['nrmse_lp']:.4f}")
    print(f"Scipy filter/PSD        : {analysis['scipy_available']}")
    print("=" * 72)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    scan_path = cfg.output_dir / f"{cfg.output_stem}_alignment_scan.csv"
    scan_df.to_csv(scan_path, index=False, encoding="utf-8-sig")
    print(f"Alignment scan table saved to: {scan_path}")

    fig = make_figure(best, cfg, analysis)

    if cfg.export_png:
        png_path = cfg.output_dir / f"{cfg.output_stem}.png"
        fig.savefig(png_path, dpi=cfg.export_dpi)
        print(f"Figure saved to: {png_path}")

    if cfg.export_pdf:
        pdf_path = cfg.output_dir / f"{cfg.output_stem}.pdf"
        fig.savefig(pdf_path)
        print(f"Figure saved to: {pdf_path}")

    plt.show()


if __name__ == "__main__":
    main()