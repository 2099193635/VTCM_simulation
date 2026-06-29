"""
Unified result analysis entry point.

This file combines:
- original result plotting from analyze_results.py
- dynamic irregularity and acceleration comparison from irre_analysis.py
- acceleration PSD plots from irr_psd_analysis.ipynb
"""

from pathlib import Path
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, detrend, filtfilt, welch

from utils.post_processing import ResultPlotter


mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "stix"


DEFAULT_ALIGNED_IRR_PATH = Path(
    "preprocessing/动检数据/呼局/20210416/处理后/动检上行20210416-238-363.aligned.csv"
)
DEFAULT_PROCESSED_IRR_PATH = Path(
    "preprocessing/动检数据/呼局/20210416/处理后/动检上行20210416-238-363.processed.csv"
)


def find_latest_result(results_dir="results"):
    """Compatibility fallback: return latest flat .npz directly under results/."""
    if not os.path.isdir(results_dir):
        return None

    npz_files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.lower().endswith(".npz")
    ]
    if not npz_files:
        return None
    return max(npz_files, key=os.path.getmtime)


def find_latest_run_folder(results_dir="results"):
    """Return latest run folder that contains files/*.npz, supporting nested project folders."""
    if not os.path.isdir(results_dir):
        return None

    candidates = []
    for root, _, _ in os.walk(results_dir):
        if os.path.basename(root).lower() != "files":
            continue
        npz_list = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith(".npz")
        ]
        if npz_list:
            latest_npz_time = max(os.path.getmtime(p) for p in npz_list)
            candidates.append((latest_npz_time, os.path.dirname(root)))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_latest_result_in_run(run_dir):
    """Return latest .npz in run_dir/files/."""
    files_dir = os.path.join(run_dir, "files")
    if not os.path.isdir(files_dir):
        return None

    npz_files = [
        os.path.join(files_dir, f)
        for f in os.listdir(files_dir)
        if f.lower().endswith(".npz")
    ]
    if not npz_files:
        return None
    return max(npz_files, key=os.path.getmtime)


def select_result_file(target_file="", target_run_folder="", results_dir="results"):
    selected_file = None
    selected_run_dir = None

    if target_file:
        selected_file = target_file
        abs_target = os.path.abspath(target_file)
        parent = os.path.dirname(abs_target)
        selected_run_dir = (
            os.path.dirname(parent) if os.path.basename(parent).lower() == "files" else None
        )
    else:
        if target_run_folder:
            selected_run_dir = (
                target_run_folder
                if os.path.isabs(target_run_folder)
                else os.path.join(results_dir, target_run_folder)
            )
        else:
            selected_run_dir = find_latest_run_folder(results_dir)

        if selected_run_dir and os.path.isdir(selected_run_dir):
            selected_file = find_latest_result_in_run(selected_run_dir)

    if not selected_file:
        selected_file = find_latest_result(results_dir)
        selected_run_dir = None

    return selected_file, selected_run_dir


def default_dynamic_irr_path():
    return DEFAULT_ALIGNED_IRR_PATH if DEFAULT_ALIGNED_IRR_PATH.exists() else DEFAULT_PROCESSED_IRR_PATH


def infer_argparse_params_path(result_file):
    params_path = Path(result_file).with_name("argparse_params.json")
    return params_path if params_path.exists() else None


def load_v_mps(input_path):
    if not input_path or not Path(input_path).exists():
        return None
    with open(input_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    vx_set = params.get("vx_set")
    if vx_set is None:
        return None
    if isinstance(vx_set, (list, tuple)):
        vx_kmh = float(vx_set[0])
    else:
        vx_kmh = float(vx_set)
    return vx_kmh / 3.6


def read_numeric_column(df, name):
    if name not in df.columns:
        raise KeyError(f"Column '{name}' not found. Available columns: {list(df.columns)}")
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)


def load_and_analyze(
    filepath,
    save_dir,
    show=False,
    plot_mileage_start=None,
    plot_mileage_end=None,
):
    """Load NPZ data and reconstruct the structure for core response plotting."""
    print(f" -> Loading result file: {filepath} ...")
    data = np.load(filepath)

    A = data["A"]
    dt = float(data["dt"])
    idx_car_start = int(data["idx_car_start"])
    Nt = A.shape[0]

    spy_dict = {}
    standard_keys = ["X", "V", "A", "dt", "idx_car_start"]
    for key in data.files:
        if key not in standard_keys:
            spy_dict[key] = data[key]

    print(f" -> Data loaded successfully. Steps: {Nt}, time step: {dt}s.")
    return ResultPlotter.plot_core_responses(
        Nt=Nt,
        dt=dt,
        A=A,
        spy_dict=spy_dict,
        idx_car_start=idx_car_start,
        save_dir=save_dir,
        show=show,
        plot_mileage_start=plot_mileage_start,
        plot_mileage_end=plot_mileage_end,
    )


def lp_butter(x, fs_loc, fc, order=4):
    wn = min(max(fc / (0.5 * fs_loc), 1e-6), 0.999999)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, np.asarray(x, dtype=float))


def bp_butter(x, fs_loc, f_low, f_high, order=4):
    nyq = 0.5 * fs_loc
    lo = max(float(f_low) / nyq, 1e-6)
    hi = min(float(f_high) / nyq, 0.999999)
    x = np.asarray(x, dtype=float)
    if hi <= lo:
        return detrend(x - np.nanmean(x), type="linear")
    b, a = butter(order, [lo, hi], btype="band")
    return filtfilt(b, a, x)


def accel_preprocess(x, fs_loc, f_low, f_high):
    x = np.asarray(x, dtype=float)
    fill = np.nanmean(x) if np.isfinite(np.nanmean(x)) else 0.0
    x = np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)
    x = detrend(x - np.mean(x), type="linear")
    return bp_butter(x, fs_loc, f_low, f_high)


def corr_nrmse_score(sim_seg, meas_seg):
    n = min(len(sim_seg), len(meas_seg))
    s = sim_seg[:n] - np.mean(sim_seg[:n])
    m = meas_seg[:n] - np.mean(meas_seg[:n])
    r = np.corrcoef(s, m)[0, 1]
    nrmse = np.sqrt(np.mean((s - m) ** 2)) / (np.std(m) + 1e-12)
    return r - 0.15 * nrmse, r, nrmse


def find_best_offset(
    irre_signal_m,
    irre_abs_km_axis,
    m_uniq_meas,
    meas_signal_mm,
    fs_cmp_irr,
    search_half_m=200,
    spacing_m=0.25,
    fc=25,
):
    ds_km = spacing_m / 1000.0
    sim_mileage_ds = np.arange(irre_abs_km_axis[0], irre_abs_km_axis[-1], ds_km)
    sim_ds = np.interp(sim_mileage_ds, irre_abs_km_axis, irre_signal_m) * 1000.0
    sim_rel_km = sim_mileage_ds - sim_mileage_ds[0]
    center_km = sim_mileage_ds[0]
    half_km = search_half_m / 1000.0
    cands = m_uniq_meas[
        (m_uniq_meas >= center_km - half_km) & (m_uniq_meas <= center_km + half_km)
    ]
    if len(cands) > 400:
        cands = cands[:: int(np.ceil(len(cands) / 400))]

    sim_raw = sim_ds - np.mean(sim_ds)
    sim_lp = lp_butter(sim_raw, fs_cmp_irr, fc)
    best = None
    for s0 in cands:
        ax = s0 + sim_rel_km
        if ax[0] < m_uniq_meas[0] or ax[-1] > m_uniq_meas[-1]:
            continue
        mr = np.interp(ax, m_uniq_meas, meas_signal_mm)
        mr = mr - np.mean(mr)
        mlp = lp_butter(mr, fs_cmp_irr, fc)
        score, r, nrmse = corr_nrmse_score(sim_lp, mlp)
        if best is None or score > best["score"]:
            best = dict(
                offset_m=(s0 - center_km) * 1000,
                score=score,
                corr=r,
                nrmse=nrmse,
                axis_km=ax,
                sim_ds_mm=sim_raw,
                meas_ds_mm=mr,
                sim_lp_mm=sim_lp,
                meas_lp_mm=mlp,
            )
    return best


find_best_offest = find_best_offset


def apply_fixed_offset(
    irre_signal_m,
    irre_abs_km_axis,
    m_uniq_meas,
    meas_signal_mm,
    fixed_offset_m,
    fs_cmp_irr,
    spacing_m=0.25,
    fc=25.0,
):
    ds_km = spacing_m / 1000.0
    sim_mileage_ds = np.arange(irre_abs_km_axis[0], irre_abs_km_axis[-1] + 1e-12, ds_km)
    sim_ds = np.interp(sim_mileage_ds, irre_abs_km_axis, irre_signal_m) * 1000.0
    sim_rel_km = sim_mileage_ds - sim_mileage_ds[0]

    center_km = sim_mileage_ds[0]
    s0 = center_km + fixed_offset_m / 1000.0
    ax = s0 + sim_rel_km
    if ax[0] < m_uniq_meas[0] or ax[-1] > m_uniq_meas[-1]:
        return None

    sim_raw = sim_ds - np.mean(sim_ds)
    sim_lp = lp_butter(sim_raw, fs_cmp_irr, fc)
    mr = np.interp(ax, m_uniq_meas, meas_signal_mm)
    mr = mr - np.mean(mr)
    mlp = lp_butter(mr, fs_cmp_irr, fc)
    _, r, nrmse = corr_nrmse_score(sim_lp, mlp)
    return dict(
        offset_m=fixed_offset_m,
        score=r - 0.15 * nrmse,
        corr=r,
        nrmse=nrmse,
        axis_km=ax,
        sim_ds_mm=sim_raw,
        meas_ds_mm=mr,
        sim_lp_mm=sim_lp,
        meas_lp_mm=mlp,
    )


def fft_amp(x, dt_loc):
    x = np.asarray(x, dtype=float) - np.mean(x)
    f = np.fft.rfftfreq(len(x), d=dt_loc)
    return f, np.abs(np.fft.rfft(x)) / max(len(x), 1)


def _next_lower_power_of_two(n):
    if n < 2:
        return 1
    return 1 << (int(n).bit_length() - 1)


def welch_spatial_psd(y, dx_m, nperseg=None):
    """Welch PSD for uniformly spaced spatial signal."""
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 8:
        return np.array([]), np.array([])
    y = y.copy()
    y[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), y[finite])
    y = y - np.mean(y)

    n = len(y)
    if nperseg is None:
        nperseg = min(4096, _next_lower_power_of_two(n))
    nperseg = int(max(8, min(nperseg, n)))
    if nperseg < 8:
        return np.array([]), np.array([])

    step = max(1, nperseg // 2)
    window = np.hanning(nperseg)
    win_power = float(np.sum(window**2))
    fs_space = 1.0 / float(dx_m)
    acc = []
    for start in range(0, n - nperseg + 1, step):
        seg = y[start : start + nperseg]
        seg = seg - np.mean(seg)
        spec = np.fft.rfft(seg * window)
        psd = (np.abs(spec) ** 2) / (fs_space * win_power)
        if len(psd) > 2:
            psd[1:-1] *= 2.0
        acc.append(psd)

    if not acc:
        return np.array([]), np.array([])
    freq = np.fft.rfftfreq(nperseg, d=dx_m)
    return freq, np.mean(np.vstack(acc), axis=0)


_welch_spatial_psd = welch_spatial_psd


def unique_sorted_xy(x_km, y):
    x_km = np.asarray(x_km, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x_km) & np.isfinite(y)
    x_m = x_km[valid] * 1000.0
    y = y[valid]
    order = np.argsort(x_m)
    x_m = x_m[order]
    y = y[order]
    x_unique, unique_idx = np.unique(x_m, return_index=True)
    return x_unique, y[unique_idx]


def resample_pair_to_common_grid(sim_km, sim_y, meas_km, meas_y, dx_m=None):
    sim_x, sim_y = unique_sorted_xy(sim_km, sim_y)
    meas_x, meas_y = unique_sorted_xy(meas_km, meas_y)

    if dx_m is None:
        meas_dx = np.diff(meas_x)
        meas_dx = meas_dx[np.isfinite(meas_dx) & (meas_dx > 0)]
        dx_m = float(np.median(meas_dx))

    x0 = max(sim_x[0], meas_x[0])
    x1 = min(sim_x[-1], meas_x[-1])
    if x1 <= x0:
        raise ValueError("仿真与实测里程没有重叠区间，无法比较 PSD。")

    x_grid = np.arange(x0, x1 + dx_m * 0.5, dx_m)
    sim_grid = np.interp(x_grid, sim_x, sim_y)
    meas_grid = np.interp(x_grid, meas_x, meas_y)
    return x_grid, sim_grid, meas_grid, dx_m


def welch_time_psd_no_preprocess(y, dx_m, v_mps, nperseg=None):
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 8:
        return np.array([]), np.array([]), np.nan
    y = y.copy()
    y[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), y[finite])
    y = y - np.mean(y)
    fs_time = float(v_mps) / float(dx_m)
    if nperseg is None:
        nperseg = min(4096, _next_lower_power_of_two(len(y)))
    nperseg = int(max(8, min(nperseg, len(y))))
    freq, psd = welch(
        y,
        fs=fs_time,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend=False,
    )
    return freq, psd, fs_time


def plot_irr_row_mileage(ax_row, best, ylabel, tag):
    if best is None:
        for ax in ax_row:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    ax = ax_row[0]
    ax.plot(best["axis_km"], best["sim_ds_mm"], lw=1.6, ls="-", color="#2A9D8F", label=f"Sim input ({tag})")
    ax.plot(best["axis_km"], best["meas_ds_mm"], lw=1.4, ls="--", color="#E76F51", alpha=0.95, label="Measured")
    ax.set_title(f'Mileage Domain Raw (offset={best["offset_m"]:.0f} m)')
    ax.set_xlabel("Mileage (km)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, ls="--")
    ax.legend(fontsize=8, frameon=False)

    ax = ax_row[1]
    ax.plot(best["axis_km"], best["sim_lp_mm"], lw=1.9, ls="-", color="#264653", label=f"Sim low-pass ({tag})")
    ax.plot(best["axis_km"], best["meas_lp_mm"], lw=1.7, ls="--", color="#F4A261", alpha=0.95, label="Measured low-pass")
    ax.set_title("Mileage Domain Low-pass")
    ax.set_xlabel("Mileage (km)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, ls="--")
    ax.legend(fontsize=8, frameon=False)


def plot_irr_row_freq(ax_row, best, ylabel, dt_cmp_irr, fs_cmp_irr, dx_m):
    if best is None:
        for ax in ax_row:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    fs, pa = fft_amp(best["sim_ds_mm"], dt_cmp_irr)
    fm, pm = fft_amp(best["meas_ds_mm"], dt_cmp_irr)
    ax = ax_row[0]
    ax.plot(fs, pa, lw=1.8, ls="-", color="#3A86FF", label="Sim FFT")
    ax.plot(fm, pm, lw=1.6, ls="--", color="#8338EC", alpha=0.95, label="Meas FFT")
    ax.set_xlim(0, min(100, fs_cmp_irr / 2))
    ax.set_title("FFT Amplitude")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (mm)")
    ax.grid(True, alpha=0.22, ls="--")
    ax.legend(fontsize=8, frameon=False)

    fws, ps = welch_spatial_psd(best["sim_ds_mm"], dx_m)
    fwm, pm2 = welch_spatial_psd(best["meas_ds_mm"], dx_m)
    ax = ax_row[1]
    keep_s = fws > 0
    keep_m = fwm > 0
    if np.count_nonzero(keep_s) == 0 or np.count_nonzero(keep_m) == 0:
        ax.text(0.5, 0.5, "PSD calculation failed", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.loglog(fws[keep_s], ps[keep_s] + 1e-30, lw=1.8, ls="-", color="#1D3557", label="Sim PSD")
        ax.loglog(fwm[keep_m], pm2[keep_m] + 1e-30, lw=1.6, ls="--", color="#E63946", alpha=0.95, label="Meas PSD")
    ax.set_title(f"Spatial PSD (Welch, dx={dx_m:.4g} m)")
    ax.set_xlabel("Spatial frequency (1/m)")
    ax.set_ylabel("PSD (mm^2/(1/m))")
    ax.grid(True, alpha=0.22, ls="--")
    ax.legend(fontsize=8, frameon=False)


def irre_analysis(
    file_path,
    input_path,
    dynamic_irr_path,
    lead_time_s_irr=2.0,
    f_cut_irr=25.0,
    meas_spacing=0.25,
    save_dir=None,
):
    data = np.load(file_path)
    required = ["X", "Irre_distance_m", "Track_abs_mileage_m", "Irre_by_L_ref", "Irre_by_R_ref"]
    missing = [key for key in required if key not in data.files]
    if missing:
        print(f"[Irre] Missing keys in result file, skipping irregularity analysis: {missing}")
        return None

    v_mps_loc = load_v_mps(input_path)
    if v_mps_loc is None or v_mps_loc <= 0:
        print("[Irre] vx_set not found or invalid, skipping irregularity analysis.")
        return None

    dynamic_irr = pd.read_csv(dynamic_irr_path, encoding="utf-8")
    X_full = data["X"]
    Nt_X = X_full.shape[0]
    irre_dist_full = data["Irre_distance_m"][:Nt_X]

    irre_bz_L_full = X_full[:, 16]
    irre_bz_R_full = X_full[:, 16]
    irre_by_L_full = data["Irre_by_L_ref"][:Nt_X]
    irre_by_R_full = data["Irre_by_R_ref"][:Nt_X]

    dt_irr = float(data["dt"])
    t_irre = np.arange(Nt_X) * dt_irr
    lead_idx = np.searchsorted(t_irre, lead_time_s_irr)
    irre_bz_L = irre_bz_L_full[lead_idx:]
    irre_bz_R = irre_bz_R_full[lead_idx:]
    irre_by_L = -irre_by_L_full[lead_idx:]
    irre_by_R = -irre_by_R_full[lead_idx:]
    irre_dist = irre_dist_full[lead_idx:]
    irre_abs_km = (float(data["Track_abs_mileage_m"][0]) + irre_dist) / 1000.0

    meas_mileage_km = read_numeric_column(dynamic_irr, "里程")
    meas_left_vert = read_numeric_column(dynamic_irr, "左高低")
    meas_left_lat = read_numeric_column(dynamic_irr, "左轨向")
    meas_right_vert = read_numeric_column(dynamic_irr, "右高低")
    meas_right_lat = read_numeric_column(dynamic_irr, "右轨向")

    valid = (
        np.isfinite(meas_mileage_km)
        & np.isfinite(meas_left_vert)
        & np.isfinite(meas_left_lat)
        & np.isfinite(meas_right_vert)
        & np.isfinite(meas_right_lat)
    )
    meas_mileage_km = meas_mileage_km[valid]
    meas_left_vert = meas_left_vert[valid]
    meas_left_lat = meas_left_lat[valid]
    meas_right_vert = meas_right_vert[valid]
    meas_right_lat = meas_right_lat[valid]

    ord_m = np.argsort(meas_mileage_km)
    meas_mileage_km = meas_mileage_km[ord_m]
    meas_left_vert = meas_left_vert[ord_m]
    meas_left_lat = meas_left_lat[ord_m]
    meas_right_vert = meas_right_vert[ord_m]
    meas_right_lat = meas_right_lat[ord_m]
    m_uniq, fi = np.unique(meas_mileage_km, return_index=True)
    ml_vert = meas_left_vert[fi]
    ml_lat = meas_left_lat[fi]
    mr_vert = meas_right_vert[fi]
    mr_lat = meas_right_lat[fi]

    print(
        f"Measured mileage: [{m_uniq[0]:.4f}, {m_uniq[-1]:.4f}] km | "
        f"left-vert std={np.nanstd(ml_vert):.3f} mm left-lat std={np.nanstd(ml_lat):.3f} mm"
    )

    dt_cmp_irr = meas_spacing / v_mps_loc
    fs_cmp_irr = 1.0 / dt_cmp_irr
    best_vert_irr = find_best_offset(
        irre_bz_L,
        irre_abs_km,
        m_uniq,
        ml_vert,
        fs_cmp_irr,
        spacing_m=meas_spacing,
        fc=f_cut_irr,
    )
    best_lat_irr = (
        apply_fixed_offset(
            irre_by_L,
            irre_abs_km,
            m_uniq,
            ml_lat,
            fixed_offset_m=best_vert_irr["offset_m"],
            fs_cmp_irr=fs_cmp_irr,
            spacing_m=meas_spacing,
            fc=f_cut_irr,
        )
        if best_vert_irr is not None
        else None
    )

    if best_vert_irr:
        print(
            f'[Vert-Irre] offset={best_vert_irr["offset_m"]:.1f} m '
            f'corr={best_vert_irr["corr"]:.4f} NRMSE={best_vert_irr["nrmse"]:.4f}'
        )
    if best_lat_irr:
        print(
            f'[Lat-Irre] offset={best_lat_irr["offset_m"]:.1f} m (locked to vertical offset) '
            f'corr={best_lat_irr["corr"]:.4f} NRMSE={best_lat_irr["nrmse"]:.4f}'
        )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    fig_m, axes_m = plt.subplots(2, 2, figsize=(30, 8.5))
    fig_m.suptitle(
        "Simulation Dynamic Irregularity vs Measured Dynamic Irregularity (Mileage Domain)\n"
        f"(lead-in {lead_time_s_irr:.0f} s trimmed, lateral sign-corrected)",
        fontsize=12,
    )
    plot_irr_row_mileage(axes_m[0], best_vert_irr, "Vertical Irregularity (mm)", "WS1_Z")
    plot_irr_row_mileage(axes_m[1], best_lat_irr, "Lateral Irregularity (mm)", "-Irre_by_L")
    plt.tight_layout()
    if save_dir:
        fig_m.savefig(os.path.join(save_dir, "irre_comparison.png"), dpi=300)
        fig_m.savefig(os.path.join(save_dir, "irre_comparison_mileage.png"), dpi=300)
    plt.close(fig_m)

    fig_f, axes_f = plt.subplots(2, 2, figsize=(16, 8.5))
    fig_f.suptitle(
        "Frequency / Spatial-Domain Comparison of Dynamic Irregularity (FFT + Spatial PSD)",
        fontsize=12,
    )
    plot_irr_row_freq(axes_f[0], best_vert_irr, "Vertical Irregularity (mm)", dt_cmp_irr, fs_cmp_irr, meas_spacing)
    plot_irr_row_freq(axes_f[1], best_lat_irr, "Lateral Irregularity (mm)", dt_cmp_irr, fs_cmp_irr, meas_spacing)
    plt.tight_layout()
    if save_dir:
        fig_f.savefig(os.path.join(save_dir, "irre_comparison_frequency.png"), dpi=300)
    plt.close(fig_f)

    return dict(
        irre_abs_km=irre_abs_km,
        irre_dist=irre_dist,
        best_offset_m=best_vert_irr["offset_m"] if best_vert_irr else 0.0,
        v_mps=v_mps_loc,
        right_vertical=irre_bz_R,
        right_lateral=irre_by_R,
        measured_right_vertical=mr_vert,
        measured_right_lateral=mr_lat,
    )


def accel_analysis(
    file_path,
    dynamic_irr_path,
    irre_abs_km,
    irre_dist,
    best_offset_m,
    lead_time_s_irr=2.0,
    f_cut_acc=200.0,
    meas_spacing=0.25,
    v_mps=None,
    save_dir=None,
    search_half_m_acc=200.0,
):
    """Generate mileage and frequency comparison plots for carbody, bogie and wheelset acceleration."""
    if v_mps is None or v_mps <= 0:
        print("[Accel] v_mps not provided, skipping.")
        return None

    data = np.load(file_path)
    dt = float(data["dt"])
    fs_sim = 1.0 / dt

    Nt_full = data["A"].shape[0]
    t_full = np.arange(Nt_full) * dt
    lead_idx = int(np.searchsorted(t_full, lead_time_s_irr))

    dt_meas = meas_spacing / v_mps
    fs_meas = 1.0 / dt_meas
    nyq_meas = 0.5 * fs_meas

    acc_f_low = 0.5
    acc_f_high = min(float(f_cut_acc), 0.80 * nyq_meas, 0.45 * fs_sim)
    if acc_f_high <= acc_f_low:
        acc_f_low = max(0.05, 0.10 * acc_f_high)
    align_f_low = acc_f_low
    align_f_high = min(10.0, acc_f_high)
    print(
        f"[Accel-Proc] fs_sim={fs_sim:.1f} Hz, fs_cmp={fs_meas:.1f} Hz, "
        f"band={acc_f_low:.2f}-{acc_f_high:.1f} Hz, "
        f"align_band={align_f_low:.2f}-{align_f_high:.1f} Hz"
    )

    A_trim = data["A"][lead_idx:]
    N = min(A_trim.shape[0], len(irre_dist))
    sim_abs_km_loc = irre_abs_km[:N]
    ds_km = meas_spacing / 1000.0
    sim_mil_ds = np.arange(sim_abs_km_loc[0], sim_abs_km_loc[-1], ds_km)

    dynamic_irr = pd.read_csv(dynamic_irr_path, encoding="utf-8")
    meas_mileage_km = read_numeric_column(dynamic_irr, "里程")
    meas_acc_vert = read_numeric_column(dynamic_irr, "垂向加速度(g)")
    meas_acc_lat = read_numeric_column(dynamic_irr, "横向加速度(g)")
    valid = np.isfinite(meas_mileage_km) & np.isfinite(meas_acc_vert) & np.isfinite(meas_acc_lat)
    meas_mileage_km = meas_mileage_km[valid]
    meas_acc_vert = meas_acc_vert[valid]
    meas_acc_lat = meas_acc_lat[valid]
    ord_m = np.argsort(meas_mileage_km)
    meas_mileage_km = meas_mileage_km[ord_m]
    meas_acc_vert = meas_acc_vert[ord_m]
    meas_acc_lat = meas_acc_lat[ord_m]
    m_uniq, fi = np.unique(meas_mileage_km, return_index=True)
    ma_vert = meas_acc_vert[fi]
    ma_lat = meas_acc_lat[fi]

    def sim_dof_to_axis(dof_col):
        sig_time = A_trim[:N, dof_col] / 9.81
        sig_time = detrend(sig_time - np.mean(sig_time), type="linear")
        sig_time = lp_butter(sig_time, fs_sim, acc_f_high)
        return np.interp(sim_mil_ds, sim_abs_km_loc, sig_time)

    def prep_for_align(sig):
        return accel_preprocess(sig, fs_meas, align_f_low, align_f_high)

    sim_bogie_vert_ds = sim_dof_to_axis(6)
    sim_bogie_lat_ds = sim_dof_to_axis(5)
    sim_bogie_vert_al = prep_for_align(sim_bogie_vert_ds)
    sim_bogie_lat_al = prep_for_align(sim_bogie_lat_ds)

    def search_best_meas_shift(
        sim_axis_km,
        sim_vert_align,
        sim_lat_align,
        meas_axis_km,
        meas_vert,
        meas_lat,
        half_m=200.0,
        step_m=0.25,
    ):
        cand_m = np.arange(-half_m, half_m + 0.5 * step_m, step_m)
        best = None
        for off_m in cand_m:
            shifted_axis = meas_axis_km + off_m / 1000.0
            if sim_axis_km[0] < shifted_axis[0] or sim_axis_km[-1] > shifted_axis[-1]:
                continue
            mv_raw = np.interp(sim_axis_km, shifted_axis, meas_vert)
            ml_raw = np.interp(sim_axis_km, shifted_axis, meas_lat)
            mv_al = prep_for_align(mv_raw)
            ml_al = prep_for_align(ml_raw)
            score_v, r_v, n_v = corr_nrmse_score(sim_vert_align, mv_al)
            score_l, r_l, n_l = corr_nrmse_score(sim_lat_align, ml_al)
            score = 0.5 * (score_v + score_l)
            if best is None or score > best["score"]:
                best = {
                    "offset_m": float(off_m),
                    "score": float(score),
                    "corr_v": float(r_v),
                    "nrmse_v": float(n_v),
                    "corr_l": float(r_l),
                    "nrmse_l": float(n_l),
                    "meas_v_raw": mv_raw,
                    "meas_l_raw": ml_raw,
                }
        return best

    best_acc = search_best_meas_shift(
        sim_axis_km=sim_mil_ds,
        sim_vert_align=sim_bogie_vert_al,
        sim_lat_align=sim_bogie_lat_al,
        meas_axis_km=m_uniq,
        meas_vert=ma_vert,
        meas_lat=ma_lat,
        half_m=search_half_m_acc,
        step_m=meas_spacing,
    )
    if best_acc is None:
        print("[Accel] Cannot align within search range. Increase search_half_m_acc.")
        return None

    accel_offset_m = best_acc["offset_m"]
    ax_km = sim_mil_ds
    meas_v_raw = best_acc["meas_v_raw"]
    meas_l_raw = best_acc["meas_l_raw"]

    print(
        f'[Accel-Align] bogie-based offset={accel_offset_m:.1f} m '
        f'Vert(corr={best_acc["corr_v"]:.4f}, NRMSE={best_acc["nrmse_v"]:.4f}) '
        f'Lat(corr={best_acc["corr_l"]:.4f}, NRMSE={best_acc["nrmse_l"]:.4f})'
    )

    def prep(sim_rs, meas_sig):
        s_raw = sim_rs - np.mean(sim_rs)
        m_raw = meas_sig - np.mean(meas_sig)
        s_cmp = accel_preprocess(sim_rs, fs_meas, acc_f_low, acc_f_high)
        m_cmp = accel_preprocess(meas_sig, fs_meas, acc_f_low, acc_f_high)
        _, r, nrmse = corr_nrmse_score(s_cmp, m_cmp)
        return dict(s_raw=s_raw, m_raw=m_raw, s_cmp=s_cmp, m_cmp=m_cmp, corr=r, nrmse=nrmse)

    def band_rms(freq, psd, f1, f2):
        mask = (freq >= f1) & (freq < f2)
        if np.count_nonzero(mask) < 2:
            return np.nan
        return float(np.sqrt(np.trapz(psd[mask], freq[mask])))

    def plot_acc_row_mileage(ax, res, ylabel, sim_tag):
        ax.plot(ax_km, res["s_cmp"], lw=1.6, ls="-", color="#2A9D8F", label=f"Sim ({sim_tag})")
        ax.plot(ax_km, res["m_cmp"], lw=1.4, ls="--", color="#E76F51", alpha=0.95, label="Meas")
        ax.set_title(
            f"Mileage Domain Detrend + Bandpass {acc_f_low:.1f}-{acc_f_high:.0f} Hz | "
            f'corr={res["corr"]:.3f} NRMSE={res["nrmse"]:.3f} '
            f"(accel offset={accel_offset_m:.0f} m)"
        )
        ax.set_xlabel("Mileage (km)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.22, ls="--")
        ax.legend(fontsize=8, frameon=False)

    def plot_acc_row_freq(ax_row, res):
        fs_s, pa_s = fft_amp(res["s_cmp"], dt_meas)
        fs_m, pa_m = fft_amp(res["m_cmp"], dt_meas)
        ax_row[0].plot(fs_s, pa_s, lw=1.8, ls="-", color="#3A86FF", label="Sim FFT")
        ax_row[0].plot(fs_m, pa_m, lw=1.6, ls="--", color="#8338EC", alpha=0.95, label="Meas FFT")
        ax_row[0].set_xlim(0, acc_f_high)
        ax_row[0].set_title("FFT Amplitude (processed)")
        ax_row[0].set_xlabel("Frequency (Hz)")
        ax_row[0].set_ylabel("Amplitude (g)")
        ax_row[0].grid(True, alpha=0.22, ls="--")
        ax_row[0].legend(fontsize=8, frameon=False)

        nperseg = min(max(256, int(round(4.0 * fs_meas))), len(res["s_cmp"]))
        noverlap = nperseg // 2 if nperseg >= 4 else None
        fw_s, ps = welch(res["s_cmp"], fs=fs_meas, window="hann", nperseg=nperseg, noverlap=noverlap, detrend=False)
        fw_m, pm = welch(res["m_cmp"], fs=fs_meas, window="hann", nperseg=nperseg, noverlap=noverlap, detrend=False)
        ax_row[1].semilogy(fw_s, ps + 1e-20, lw=1.8, ls="-", color="#1D3557", label="Sim PSD")
        ax_row[1].semilogy(fw_m, pm + 1e-20, lw=1.6, ls="--", color="#E63946", alpha=0.95, label="Meas PSD")
        ax_row[1].set_xlim(0, acc_f_high)
        ax_row[1].set_title(f"PSD (Welch, {nperseg / fs_meas:.1f}s Hann)")
        ax_row[1].set_xlabel("Frequency (Hz)")
        ax_row[1].set_ylabel("PSD (g^2/Hz)")
        ax_row[1].grid(True, alpha=0.22, ls="--")
        ax_row[1].legend(fontsize=8, frameon=False)

        bands = [(0.5, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, min(50.0, acc_f_high))]
        rms = []
        for f1, f2 in bands:
            if f2 > f1 and f2 <= acc_f_high + 1e-9:
                rms.append((f1, f2, band_rms(fw_s, ps, f1, f2), band_rms(fw_m, pm, f1, f2)))
        return rms

    components = [
        (1, 0, "Carbody", "Carbody_Z", "Carbody_Y", "accel_carbody"),
        (6, 5, "Bogie1", "Bogie1_Z", "Bogie1_Y", "accel_bogie"),
        (16, 15, "WS1", "WS1_Z", "WS1_Y", "accel_wheelset"),
    ]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for z_dof, y_dof, part_name, z_tag, y_tag, fname_base in components:
        sim_v = sim_dof_to_axis(z_dof)
        sim_l = sim_dof_to_axis(y_dof)
        res_v = prep(sim_v, meas_v_raw)
        res_l = prep(sim_l, meas_l_raw)

        print(
            f'[{part_name}] Vert: corr={res_v["corr"]:.4f} NRMSE={res_v["nrmse"]:.4f} | '
            f'Lat: corr={res_l["corr"]:.4f} NRMSE={res_l["nrmse"]:.4f}'
        )

        fig_m, axes_m = plt.subplots(2, 1, figsize=(16, 8.5))
        fig_m.suptitle(
            f"Sim {part_name} Acceleration vs Measured Acceleration (Mileage Domain)\n"
            f"Vertical: A[:,{z_dof}]/g | Lateral: A[:,{y_dof}]/g "
            f"(offset={accel_offset_m:.0f} m, fs_cmp={fs_meas:.1f} Hz)",
            fontsize=11,
        )
        plot_acc_row_mileage(axes_m[0], res_v, "Vertical Accel (g)", z_tag)
        plot_acc_row_mileage(axes_m[1], res_l, "Lateral Accel (g)", y_tag)
        plt.tight_layout()
        if save_dir:
            out_m = os.path.join(save_dir, f"{fname_base}_mileage.png")
            fig_m.savefig(out_m, dpi=300)
            print(f"    saved: {out_m}")
        plt.close(fig_m)

        fig_f, axes_f = plt.subplots(2, 2, figsize=(16, 8.5))
        fig_f.suptitle(
            f"Frequency-Domain Comparison of {part_name} Acceleration "
            f"(detrend + {acc_f_low:.1f}-{acc_f_high:.0f} Hz bandpass)",
            fontsize=11,
        )
        rms_v = plot_acc_row_freq(axes_f[0], res_v)
        rms_l = plot_acc_row_freq(axes_f[1], res_l)
        plt.tight_layout()
        if save_dir:
            out_f = os.path.join(save_dir, f"{fname_base}_frequency.png")
            fig_f.savefig(out_f, dpi=300)
            print(f"    saved: {out_f}")
        plt.close(fig_f)

        for label, rms_vals in [("Vert", rms_v), ("Lat", rms_l)]:
            if not rms_vals:
                continue
            parts = []
            for f1, f2, sim_rms, meas_rms in rms_vals:
                ratio = sim_rms / (meas_rms + 1e-12)
                parts.append(f"{f1:g}-{f2:g}Hz S/M={ratio:.2f}")
            print(f"    [{part_name} {label} band RMS ratio] " + " | ".join(parts))

    return dict(offset_m=accel_offset_m)


def accel_psd_no_preprocess(
    result_file,
    dynamic_irr_path,
    input_path=None,
    buffer_time_s=2.0,
    accel_offset_m=0.0,
    save_dir=None,
    show=False,
):
    """Notebook-derived acceleration PSD plots without filtering or detrending preprocessing."""
    result_file = Path(result_file)
    dynamic_irr_path = Path(dynamic_irr_path)
    data = np.load(result_file)
    required = ["A", "dt", "Irre_distance_m", "Track_abs_mileage_m"]
    missing = [key for key in required if key not in data.files]
    if missing:
        print(f"[PSD] Missing keys in result file, skipping no-preprocess PSD: {missing}")
        return []

    A = data["A"]
    dt = float(data["dt"])
    Nt = A.shape[0]
    t = np.arange(Nt) * dt
    buffer_idx = int(np.searchsorted(t, buffer_time_s))

    irre_dist = data["Irre_distance_m"][:Nt]
    A_use = A[buffer_idx:]
    buffer_distance_m = float(irre_dist[buffer_idx] - irre_dist[0]) if buffer_idx < len(irre_dist) else 0.0
    irre_dist_use = irre_dist[buffer_idx : buffer_idx + len(A_use)]
    N = min(len(A_use), len(irre_dist_use))
    A_use = A_use[:N]
    sim_mileage_km = (float(data["Track_abs_mileage_m"][0]) + irre_dist_use[:N]) / 1000.0

    dynamic_df = pd.read_csv(dynamic_irr_path, encoding="utf-8")
    meas_mileage_km = read_numeric_column(dynamic_df, "里程") + accel_offset_m / 1000.0
    meas_acc_vert = read_numeric_column(dynamic_df, "垂向加速度(g)")
    meas_acc_lat = read_numeric_column(dynamic_df, "横向加速度(g)")

    print(f"[PSD] 仿真文件: {result_file}")
    print(f"[PSD] 动检文件: {dynamic_irr_path}")
    print(f"[PSD] 已裁掉仿真无不平顺缓冲段: {buffer_time_s:g} s, 约 {buffer_distance_m:.2f} m")
    print(f"[PSD] 仿真里程范围(裁剪后): {sim_mileage_km[0]:.4f} - {sim_mileage_km[-1]:.4f} km")
    print(f"[PSD] 实测里程范围: {np.nanmin(meas_mileage_km):.4f} - {np.nanmax(meas_mileage_km):.4f} km")

    components = [
        (1, 0, "Carbody", "A[:,1]/g", "A[:,0]/g"),
        (6, 5, "Bogie1", "A[:,6]/g", "A[:,5]/g"),
        (16, 15, "WS1", "A[:,16]/g", "A[:,15]/g"),
    ]

    out_dir = Path(save_dir) if save_dir else result_file.parents[1] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    fig, axes = plt.subplots(len(components), 2, figsize=(14, 11), constrained_layout=True)
    dx_m_last = np.nan
    for row, (z_dof, y_dof, part_name, z_tag, y_tag) in enumerate(components):
        sim_vert_g = A_use[:, z_dof] / 9.81
        sim_lat_g = A_use[:, y_dof] / 9.81
        for col, (sim_sig, meas_sig, direction, tag) in enumerate(
            [
                (sim_vert_g, meas_acc_vert, "Vertical", z_tag),
                (sim_lat_g, meas_acc_lat, "Lateral", y_tag),
            ]
        ):
            _, sim_grid, meas_grid, dx_m = resample_pair_to_common_grid(
                sim_mileage_km, sim_sig, meas_mileage_km, meas_sig
            )
            dx_m_last = dx_m
            f_sim, psd_sim = welch_spatial_psd(sim_grid, dx_m)
            f_meas, psd_meas = welch_spatial_psd(meas_grid, dx_m)

            ax = axes[row, col]
            keep_sim = f_sim > 0
            keep_meas = f_meas > 0
            ax.loglog(f_sim[keep_sim], psd_sim[keep_sim] + 1e-30, lw=1.6, label=f"Sim {tag}")
            ax.loglog(f_meas[keep_meas], psd_meas[keep_meas] + 1e-30, lw=1.4, ls="--", label="Measured")
            ax.set_title(f"{part_name} {direction} acceleration spatial PSD")
            ax.set_xlabel("Spatial frequency (1/m)")
            ax.set_ylabel("PSD (g^2/(1/m))")
            ax.grid(True, which="both", alpha=0.28, ls="--")
            ax.legend(frameon=False)

    fig.suptitle(
        "Simulation vs measured acceleration spatial PSD, no acceleration preprocessing\n"
        f"dx={dx_m_last:.4g} m, removed buffer={buffer_time_s:g} s "
        f"({buffer_distance_m:.1f} m), meas offset={accel_offset_m:g} m",
        fontsize=13,
    )
    out_png = out_dir / "accel_spatial_psd_no_preprocess.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    saved.append(str(out_png))
    if show:
        plt.show()
    plt.close(fig)
    print(f"[PSD] saved: {out_png}")

    v_mps = load_v_mps(input_path)
    if v_mps is None or v_mps <= 0:
        print("[PSD] vx_set not found or invalid, skipping time PSD plots.")
        return saved
    vx_kmh = v_mps * 3.6

    fig, axes = plt.subplots(len(components), 2, figsize=(14, 11), constrained_layout=True)
    fs_time_last = np.nan
    dx_m_last = np.nan
    for row, (z_dof, y_dof, part_name, z_tag, y_tag) in enumerate(components):
        sim_vert_g = A_use[:, z_dof] / 9.81
        sim_lat_g = A_use[:, y_dof] / 9.81
        for col, (sim_sig, meas_sig, direction, tag) in enumerate(
            [
                (sim_vert_g, meas_acc_vert, "Vertical", z_tag),
                (sim_lat_g, meas_acc_lat, "Lateral", y_tag),
            ]
        ):
            _, sim_grid, meas_grid, dx_m = resample_pair_to_common_grid(
                sim_mileage_km, sim_sig, meas_mileage_km, meas_sig
            )
            f_sim, psd_sim, fs_time = welch_time_psd_no_preprocess(sim_grid, dx_m, v_mps)
            f_meas, psd_meas, _ = welch_time_psd_no_preprocess(meas_grid, dx_m, v_mps)
            fs_time_last = fs_time
            dx_m_last = dx_m

            ax = axes[row, col]
            keep_sim = f_sim > 0
            keep_meas = f_meas > 0
            ax.semilogy(f_sim[keep_sim], psd_sim[keep_sim] + 1e-30, lw=1.6, label=f"Sim {tag}")
            ax.semilogy(f_meas[keep_meas], psd_meas[keep_meas] + 1e-30, lw=1.4, ls="--", label="Measured")
            ax.set_title(f"{part_name} {direction} acceleration time PSD")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD (g^2/Hz)")
            ax.grid(True, which="both", alpha=0.28, ls="--")
            ax.legend(frameon=False)

    fig.suptitle(
        "Simulation vs measured acceleration time PSD, no acceleration preprocessing\n"
        f"v={vx_kmh:g} km/h ({v_mps:.3f} m/s), fs={fs_time_last:.2f} Hz, "
        f"dx={dx_m_last:.4g} m, removed buffer={buffer_time_s:g} s",
        fontsize=13,
    )
    out_png = out_dir / "accel_time_psd_no_preprocess.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    saved.append(str(out_png))
    if show:
        plt.show()
    plt.close(fig)
    print(f"[PSD] 速度: {vx_kmh:g} km/h = {v_mps:.3f} m/s")
    print(f"[PSD] 等效时间采样频率: {fs_time_last:.3f} Hz, dt={1 / fs_time_last:.6f} s")
    print(f"[PSD] saved: {out_png}")

    fig, axes = plt.subplots(len(components), 2, figsize=(14, 11), constrained_layout=True)
    fs_time_last = np.nan
    dx_m_last = np.nan
    for row, (z_dof, y_dof, part_name, z_tag, y_tag) in enumerate(components):
        sim_vert_g = A_use[:, z_dof] / 9.81
        sim_lat_g = A_use[:, y_dof] / 9.81
        for col, (sim_sig, meas_sig, direction, tag) in enumerate(
            [
                (sim_vert_g, meas_acc_vert, "Vertical", z_tag),
                (sim_lat_g, meas_acc_lat, "Lateral", y_tag),
            ]
        ):
            _, sim_grid, meas_grid, dx_m = resample_pair_to_common_grid(
                sim_mileage_km, sim_sig, meas_mileage_km, meas_sig
            )
            f_sim, psd_sim, fs_time = welch_time_psd_no_preprocess(sim_grid, dx_m, v_mps)
            f_meas, psd_meas, _ = welch_time_psd_no_preprocess(meas_grid, dx_m, v_mps)
            fs_time_last = fs_time
            dx_m_last = dx_m

            ax = axes[row, col]
            keep_sim = f_sim > 0
            keep_meas = f_meas > 0
            ax.plot(f_sim[keep_sim], psd_sim[keep_sim] + 1e-30, lw=1.6, label=f"Sim {tag}")
            ax.plot(f_meas[keep_meas], psd_meas[keep_meas] + 1e-30, lw=1.4, ls="--", label="Measured")
            ax.set_xscale("symlog", linthresh=1.0, linscale=1.0)
            ax.set_yscale("log")
            ax.set_xlim(left=0.0, right=0.5 * fs_time)
            ax.set_title(f"{part_name} {direction} Acceleration Time PSD")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD (g^2/Hz)")
            ax.grid(True, which="both", alpha=0.28, ls="--")
            ax.legend(frameon=False)

    fig.suptitle(
        "Simulation vs measured acceleration time PSD "
        "(x starts at 0 Hz with symlog scale, y log, no preprocessing)\n"
        f"v={vx_kmh:g} km/h ({v_mps:.3f} m/s), fs={fs_time_last:.2f} Hz, "
        f"dx={dx_m_last:.4g} m, removed buffer={buffer_time_s:g} s",
        fontsize=13,
    )
    out_png = out_dir / "accel_time_psd_x0_symlog_no_preprocess.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    saved.append(str(out_png))
    if show:
        plt.show()
    plt.close(fig)
    print(f"[PSD] saved: {out_png}")

    return saved


def run_all_analyses(
    target_file="",
    target_run_folder="",
    dynamic_irr_path=None,
    lead_time_s_irr=2.0,
    f_cut_irr=25.0,
    f_cut_acc=200.0,
    meas_spacing=0.25,
    plot_mileage_start=None,
    plot_mileage_end=None,
    run_core=True,
    run_irre=True,
    run_accel=True,
    run_psd=True,
    show=False,
):
    selected_file, selected_run_dir = select_result_file(target_file, target_run_folder)
    if not selected_file or not os.path.exists(selected_file):
        print("No valid result file found in 'results'. Please run generate_main.py first.")
        return None

    if selected_run_dir and os.path.isdir(selected_run_dir):
        figures_dir = os.path.join(selected_run_dir, "figures")
    else:
        figures_dir = os.path.join("results", "figures_legacy")
    os.makedirs(figures_dir, exist_ok=True)

    dynamic_irr_path = Path(dynamic_irr_path) if dynamic_irr_path else default_dynamic_irr_path()
    input_path = infer_argparse_params_path(selected_file)

    print(f" -> Selected result file: {selected_file}")
    print(f" -> Figures will be saved to: {figures_dir}")
    print(f" -> Dynamic inspection file: {dynamic_irr_path}")
    print(f" -> Params file: {input_path}")

    if run_core:
        load_and_analyze(
            selected_file,
            save_dir=figures_dir,
            show=show,
            plot_mileage_start=plot_mileage_start,
            plot_mileage_end=plot_mileage_end,
        )

    align_info = None
    if run_irre:
        align_info = irre_analysis(
            selected_file,
            input_path,
            dynamic_irr_path,
            lead_time_s_irr=lead_time_s_irr,
            f_cut_irr=f_cut_irr,
            meas_spacing=meas_spacing,
            save_dir=figures_dir,
        )

    if run_accel and align_info is not None:
        accel_analysis(
            file_path=selected_file,
            dynamic_irr_path=dynamic_irr_path,
            irre_abs_km=align_info["irre_abs_km"],
            irre_dist=align_info["irre_dist"],
            best_offset_m=align_info["best_offset_m"],
            lead_time_s_irr=lead_time_s_irr,
            f_cut_acc=f_cut_acc,
            meas_spacing=meas_spacing,
            v_mps=align_info["v_mps"],
            save_dir=figures_dir,
        )

    if run_psd:
        accel_psd_no_preprocess(
            result_file=selected_file,
            dynamic_irr_path=dynamic_irr_path,
            input_path=input_path,
            buffer_time_s=lead_time_s_irr,
            accel_offset_m=0.0,
            save_dir=figures_dir,
            show=show,
        )

    return dict(result_file=selected_file, run_dir=selected_run_dir, figures_dir=figures_dir)


if __name__ == "__main__":
    # Optional manual targets.
    # target_run_folder can be either a folder name under results/ or an absolute path.
    target_run_folder = "results\\long_distance\\高速客车-外部导入-vehicle-normal-10km_moving_window-20260629_133910"
    # target_file has highest priority. Leave empty to use the latest results/*/files/*.npz.
    target_file = "results\\long_distance\\高速客车-外部导入-vehicle-normal-10km_moving_window-20260629_133910\\files\\simulation_result.npz"

    # Mileage range for core response plots. Use None for the whole range.
    plot_mileage_start = None
    plot_mileage_end = None

    run_all_analyses(
        target_file=target_file,
        target_run_folder=target_run_folder,
        dynamic_irr_path=default_dynamic_irr_path(),
        lead_time_s_irr=2.0,
        f_cut_irr=25.0,
        f_cut_acc=200.0,
        meas_spacing=0.25,
        plot_mileage_start=plot_mileage_start,
        plot_mileage_end=plot_mileage_end,
        run_core=True,
        run_irre=True,
        run_accel=True,
        run_psd=True,
        show=False,
    )
