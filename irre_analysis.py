'''
Author: 2099193635 2099193635@qq.com
Date: 2026-06-01 06:39:07
LastEditors: Niscienc 60505912+2099193635@users.noreply.github.com
LastEditTime: 2026-06-22 19:56:38
FilePath: \VTCM_PYTHON\irre_analysis.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''
import numpy as np
import os
from utils.post_processing import ResultPlotter
import pandas as pd
from scipy.signal import butter, filtfilt, welch, detrend
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif']  = ['Liberation Serif', 'Nimbus Roman', 'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'stix'

def lp_butter(x, fs_loc, fc, order=4):
    wn = min(max(fc / (0.5 * fs_loc), 1e-6), 0.999999)
    b, a = butter(order, wn, btype='low')
    return filtfilt(b, a, np.asarray(x, dtype=float))


def bp_butter(x, fs_loc, f_low, f_high, order=4):
    nyq = 0.5 * fs_loc
    lo = max(float(f_low) / nyq, 1e-6)
    hi = min(float(f_high) / nyq, 0.999999)
    x = np.asarray(x, dtype=float)
    if hi <= lo:
        return detrend(x - np.nanmean(x), type='linear')
    b, a = butter(order, [lo, hi], btype='band')
    return filtfilt(b, a, x)


def accel_preprocess(x, fs_loc, f_low, f_high):
    x = np.asarray(x, dtype=float)
    fill = np.nanmean(x) if np.isfinite(np.nanmean(x)) else 0.0
    x = np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)
    x = detrend(x - np.mean(x), type='linear')
    return bp_butter(x, fs_loc, f_low, f_high)


def corr_nrmse_score(sim_seg, meas_seg):
    n = min(len(sim_seg), len(meas_seg))
    s = sim_seg[:n] - np.mean(sim_seg[:n])
    m = meas_seg[:n] - np.mean(meas_seg[:n])
    r = np.corrcoef(s, m)[0, 1]
    nrmse = np.sqrt(np.mean((s - m)**2)) / (np.std(m) + 1e-12)
    return r - 0.15 * nrmse, r, nrmse


def find_best_offest(irre_signal_m, 
                     irre_abs_km_axis, 
                     m_uniq_meas, 
                     meas_signal_mm,
                     fs_cmp_irr,
                     search_half_m = 200,
                     spacing_m = 0.25,
                     fc = 25):
    ds_km = spacing_m / 1000.0
    sim_mileage_ds = np.arange(irre_abs_km_axis[0], irre_abs_km_axis[-1], ds_km)
    sim_ds = np.interp(sim_mileage_ds, irre_abs_km_axis, irre_signal_m) * 1000.0
    sim_rel_km = sim_mileage_ds - sim_mileage_ds[0]
    center_km = sim_mileage_ds[0]
    half_km = search_half_m / 1000.0
    cands = m_uniq_meas[(m_uniq_meas >= center_km - half_km) & (m_uniq_meas <= center_km + half_km)]
    if len(cands) > 400:
        cands = cands[::int(np.ceil(len(cands) / 400))]
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
        if best is None or score > best['score']:
            best = dict(offset_m=(s0 - center_km)*1000, score=score, corr=r, nrmse=nrmse,
                        axis_km=ax, sim_ds_mm=sim_raw, meas_ds_mm=mr,
                        sim_lp_mm=sim_lp, meas_lp_mm=mlp)
    return best

def apply_fixed_offset(irre_signal_m, irre_abs_km_axis, m_uniq_meas, meas_signal_mm,
                       fixed_offset_m, fs_cmp_irr, spacing_m=0.25, fc=25.0):
    """Align lateral signal at the offset already determined from vertical."""
    ds_km = spacing_m / 1000.0
    sim_mileage_ds = np.arange(irre_abs_km_axis[0], irre_abs_km_axis[-1] + 1e-12, ds_km)
    sim_ds  = np.interp(sim_mileage_ds, irre_abs_km_axis, irre_signal_m) * 1000.0  # m → mm
    sim_rel_km = sim_mileage_ds - sim_mileage_ds[0]

    center_km = sim_mileage_ds[0]
    s0 = center_km + fixed_offset_m / 1000.0
    ax = s0 + sim_rel_km
    if ax[0] < m_uniq_meas[0] or ax[-1] > m_uniq_meas[-1]:
        return None

    sim_raw = sim_ds - np.mean(sim_ds)
    sim_lp  = lp_butter(sim_raw, fs_cmp_irr, fc)
    mr  = np.interp(ax, m_uniq_meas, meas_signal_mm)
    mr  = mr - np.mean(mr)
    mlp = lp_butter(mr, fs_cmp_irr, fc)
    _, r, nrmse = corr_nrmse_score(sim_lp, mlp)
    return dict(offset_m=fixed_offset_m, score=r - 0.15*nrmse, corr=r, nrmse=nrmse,
                axis_km=ax, sim_ds_mm=sim_raw, meas_ds_mm=mr,
                sim_lp_mm=sim_lp, meas_lp_mm=mlp)

def load_and_analyze(filepath, save_dir, show=False):
    print(f" -> Loading result file: {filepath} ...")
    data = np.load(filepath)
    A = data['A']
    dt = float(data['dt'])
    idx_car_start = int(data['idx_car_start'])
    Nt = A.shape[0]

    spy_dict = {}
    standard_keys = ['X', 'V', 'A', 'dt', 'idx_car_start']
    for key in data.files:
        if key not in standard_keys:
            spy_dict[key] = data[key]
    print(f" -> Data loaded successfully. Steps: {Nt}, time step: {dt}s. Preparing extended plots...")
    save_path = ResultPlotter.plot_core_responses(
        Nt=Nt, 
        dt=dt, 
        A=A, 
        spy_dict=spy_dict, 
        idx_car_start=idx_car_start,
        save_dir=save_dir,
        show=show
    )
    return save_path

def fft_amp(x, dt_loc):
    x = np.asarray(x, dtype=float) - np.mean(x)
    f = np.fft.rfftfreq(len(x), d=dt_loc)
    return f, np.abs(np.fft.rfft(x)) / max(len(x), 1)


def _next_lower_power_of_two(n: int) -> int:
    if n < 2:
        return 1
    return 1 << (int(n).bit_length() - 1)


def _welch_spatial_psd(y: np.ndarray, dx_m: float, nperseg: int | None = None):
    """Welch PSD for uniformly spaced spatial signal, matching preprocessing/dynamic_pre.py."""
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
    win_power = float(np.sum(window ** 2))
    fs_space = 1.0 / float(dx_m)
    acc = []
    for start in range(0, n - nperseg + 1, step):
        seg = y[start:start + nperseg]
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


def accel_analysis(file_path, dynamic_irr_path, irre_abs_km, irre_dist,
                   best_offset_m, lead_time_s_irr=2.0,
                   f_cut_acc=200.0, meas_spacing=0.25,
                   v_mps=None, save_dir=None,
                   search_half_m_acc=200.0):
    """
    分别生成车体、构架、轮对与实测轴箱加速度的比较图（共3张）。

    处理链路：仿真先按实测采样带宽抗混叠低通，再重采样到统一里程轴；
    实测与仿真随后采用一致的去趋势+带通处理，并用低频带通结果搜索里程偏移。
    """
    if v_mps is None or v_mps <= 0:
        print('[Accel] v_mps not provided — skipping.')
        return

    data  = np.load(file_path)
    dt    = float(data['dt'])
    fs_sim = 1.0 / dt

    # ── 去除前2s瞬态 ────────────────────────────────────────────────────
    Nt_full  = data['A'].shape[0]
    t_full   = np.arange(Nt_full) * dt
    lead_idx = int(np.searchsorted(t_full, lead_time_s_irr))

    # ── 测量等效时间参数：若没有实测速度列，则沿用恒速近似 ─────────────────
    dt_meas   = meas_spacing / v_mps
    fs_meas   = 1.0 / dt_meas
    nyq_meas  = 0.5 * fs_meas

    # 统一比较带宽。上限限制在实测奈奎斯特频率以下，避免降采样混叠。
    acc_f_low  = 0.5
    acc_f_high = min(float(f_cut_acc), 0.80 * nyq_meas, 0.45 * fs_sim)
    if acc_f_high <= acc_f_low:
        acc_f_low = max(0.05, 0.10 * acc_f_high)
    align_f_low  = acc_f_low
    align_f_high = min(10.0, acc_f_high)
    print(f'[Accel-Proc] fs_sim={fs_sim:.1f} Hz, fs_cmp={fs_meas:.1f} Hz, '
          f'band={acc_f_low:.2f}-{acc_f_high:.1f} Hz, align_band={align_f_low:.2f}-{align_f_high:.1f} Hz')

    # ── A 矩阵截断 ────────────────────────────────────────────────────────
    A_trim = data['A'][lead_idx:]              # (Nt-lead, 35)，单位 m/s²
    N      = min(A_trim.shape[0], len(irre_dist))

    # ── 构造仿真统一里程轴（后续以“平移实测里程”进行独立对齐）────────────────
    sim_abs_km_loc = irre_abs_km[:N]
    ds_km          = meas_spacing / 1000.0
    sim_mil_ds     = np.arange(sim_abs_km_loc[0], sim_abs_km_loc[-1], ds_km)

    # ── 读取测量加速度 ────────────────────────────────────────────────────
    dynamic_irr     = pd.read_csv(dynamic_irr_path)
    meas_mileage_km = pd.to_numeric(dynamic_irr['里程'],          errors='coerce').to_numpy(dtype=float)
    meas_acc_vert   = pd.to_numeric(dynamic_irr['垂向加速度(g)'], errors='coerce').to_numpy(dtype=float)
    meas_acc_lat    = pd.to_numeric(dynamic_irr['横向加速度(g)'], errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(meas_mileage_km) & np.isfinite(meas_acc_vert) & np.isfinite(meas_acc_lat)
    meas_mileage_km = meas_mileage_km[valid]
    meas_acc_vert   = meas_acc_vert[valid]
    meas_acc_lat    = meas_acc_lat[valid]
    ord_m           = np.argsort(meas_mileage_km)
    meas_mileage_km = meas_mileage_km[ord_m]
    meas_acc_vert   = meas_acc_vert[ord_m]
    meas_acc_lat    = meas_acc_lat[ord_m]
    m_uniq, fi      = np.unique(meas_mileage_km, return_index=True)
    ma_vert         = meas_acc_vert[fi]
    ma_lat          = meas_acc_lat[fi]

    def sim_dof_to_axis(dof_col):
        """仿真高采样率信号先抗混叠低通，再按统一里程轴重采样，单位换算为 g。"""
        sig_time = A_trim[:N, dof_col] / 9.81
        sig_time = detrend(sig_time - np.mean(sig_time), type='linear')
        sig_time = lp_butter(sig_time, fs_sim, acc_f_high)
        return np.interp(sim_mil_ds, sim_abs_km_loc, sig_time)

    def prep_for_align(sig):
        return accel_preprocess(sig, fs_meas, align_f_low, align_f_high)

    # 用构架(Bogie1)作为加速度对齐基准；当前实测通道更接近构架响应。
    sim_bogie_vert_ds = sim_dof_to_axis(6)
    sim_bogie_lat_ds  = sim_dof_to_axis(5)
    sim_bogie_vert_al = prep_for_align(sim_bogie_vert_ds)
    sim_bogie_lat_al  = prep_for_align(sim_bogie_lat_ds)

    def search_best_meas_shift(sim_axis_km, sim_vert_align, sim_lat_align,
                               meas_axis_km, meas_vert, meas_lat,
                               half_m=200.0, step_m=0.25):
        """用低频带通后的信号平移实测里程轴，减少高频相位对偏移搜索的干扰。"""
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
            if (best is None) or (score > best['score']):
                best = {
                    'offset_m': float(off_m),
                    'score': float(score),
                    'corr_v': float(r_v), 'nrmse_v': float(n_v),
                    'corr_l': float(r_l), 'nrmse_l': float(n_l),
                    'meas_v_raw': mv_raw, 'meas_l_raw': ml_raw,
                    'meas_v_al': mv_al, 'meas_l_al': ml_al,
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
        print('[Accel] 无法在给定偏移范围内完成对齐，请增大 search_half_m_acc。')
        return

    accel_offset_m = best_acc['offset_m']
    ax_km = sim_mil_ds
    meas_v_raw = best_acc['meas_v_raw']
    meas_l_raw = best_acc['meas_l_raw']

    print(f'[Accel-Align] bogie-based offset={accel_offset_m:.1f} m  '
          f'Vert(corr={best_acc["corr_v"]:.4f}, NRMSE={best_acc["nrmse_v"]:.4f})  '
          f'Lat(corr={best_acc["corr_l"]:.4f}, NRMSE={best_acc["nrmse_l"]:.4f})')

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
        ax.plot(ax_km, res['s_cmp'], lw=1.6, ls='-', color='#2A9D8F', label=f'Sim ({sim_tag})')
        ax.plot(ax_km, res['m_cmp'], lw=1.4, ls='--', color='#E76F51', alpha=0.95, label='Meas')
        ax.set_title(f'Mileage Domain – Detrend + Bandpass {acc_f_low:.1f}-{acc_f_high:.0f} Hz  |  '
                     f'corr={res["corr"]:.3f}  NRMSE={res["nrmse"]:.3f}  '
                     f'(accel offset={accel_offset_m:.0f} m)')
        ax.set_xlabel('Mileage (km)'); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.22, ls='--'); ax.legend(fontsize=8, frameon=False)

    def plot_acc_row_freq(ax_row, res):
        fs_s, Pa_s = fft_amp(res['s_cmp'], dt_meas)
        fs_m, Pa_m = fft_amp(res['m_cmp'], dt_meas)
        ax_row[0].plot(fs_s, Pa_s, lw=1.8, ls='-', color='#3A86FF', label='Sim FFT')
        ax_row[0].plot(fs_m, Pa_m, lw=1.6, ls='--', color='#8338EC', alpha=0.95, label='Meas FFT')
        ax_row[0].set_xlim(0, acc_f_high)
        ax_row[0].set_title('FFT Amplitude (processed)'); ax_row[0].set_xlabel('Frequency (Hz)')
        ax_row[0].set_ylabel('Amplitude (g)'); ax_row[0].grid(True, alpha=0.22, ls='--'); ax_row[0].legend(fontsize=8, frameon=False)

        nperseg = min(max(256, int(round(4.0 * fs_meas))), len(res['s_cmp']))
        noverlap = nperseg // 2 if nperseg >= 4 else None
        fw_s, Ps = welch(res['s_cmp'], fs=fs_meas, window='hann', nperseg=nperseg,
                         noverlap=noverlap, detrend=False)
        fw_m, Pm = welch(res['m_cmp'], fs=fs_meas, window='hann', nperseg=nperseg,
                         noverlap=noverlap, detrend=False)
        ax_row[1].semilogy(fw_s, Ps + 1e-20, lw=1.8, ls='-', color='#1D3557', label='Sim PSD')
        ax_row[1].semilogy(fw_m, Pm + 1e-20, lw=1.6, ls='--', color='#E63946', alpha=0.95, label='Meas PSD')
        ax_row[1].set_xlim(0, acc_f_high)
        ax_row[1].set_title(f'PSD (Welch, {nperseg / fs_meas:.1f}s Hann)')
        ax_row[1].set_xlabel('Frequency (Hz)')
        ax_row[1].set_ylabel('PSD (g²/Hz)'); ax_row[1].grid(True, alpha=0.22, ls='--'); ax_row[1].legend(fontsize=8, frameon=False)
        bands = [(0.5, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, min(50.0, acc_f_high))]
        rms = []
        for f1, f2 in bands:
            if f2 > f1 and f2 <= acc_f_high + 1e-9:
                rms.append((f1, f2, band_rms(fw_s, Ps, f1, f2), band_rms(fw_m, Pm, f1, f2)))
        return rms

    # ── 三个部件：(垂向DOF, 横向DOF, 部件名, 文件名后缀) ─────────────────
    components = [
        (1,  0,  '车体 (Carbody)', 'Carbody_Z', 'Carbody_Y', 'accel_carbody'),
        (6,  5,  '构架 (Bogie1)',  'Bogie1_Z',  'Bogie1_Y',  'accel_bogie'),
        (16, 15, '轮对 (WS1)',     'WS1_Z',     'WS1_Y',     'accel_wheelset'),
    ]

    for z_dof, y_dof, part_name, z_tag, y_tag, fname_base in components:
        sim_v = sim_dof_to_axis(z_dof)
        sim_l = sim_dof_to_axis(y_dof)
        res_v = prep(sim_v, meas_v_raw)
        res_l = prep(sim_l, meas_l_raw)

        print(f'[{part_name}]  '
              f'Vert: corr={res_v["corr"]:.4f} NRMSE={res_v["nrmse"]:.4f}  |  '
              f'Lat:  corr={res_l["corr"]:.4f} NRMSE={res_l["nrmse"]:.4f}')

        fig_m, axes_m = plt.subplots(2, 1, figsize=(16, 8.5))
        fig_m.suptitle(
            f'Sim {part_name} Acceleration vs Measured Acceleration (Mileage Domain)\n'
            f'Vertical: A[:,{z_dof}]/g  |  Lateral: A[:,{y_dof}]/g  '
            f'(offset={accel_offset_m:.0f} m, fs_cmp={fs_meas:.1f} Hz)',
            fontsize=11)
        plot_acc_row_mileage(axes_m[0], res_v, 'Vertical Accel (g)', z_tag)
        plot_acc_row_mileage(axes_m[1], res_l, 'Lateral Accel (g)',  y_tag)
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_m = os.path.join(save_dir, f'{fname_base}_mileage.png')
            plt.savefig(out_m, dpi=300)
            print(f'    → saved: {out_m}')
        plt.close(fig_m)

        fig_f, axes_f = plt.subplots(2, 2, figsize=(16, 8.5))
        fig_f.suptitle(
            f'Frequency-Domain Comparison of {part_name} Acceleration '
            f'(detrend + {acc_f_low:.1f}-{acc_f_high:.0f} Hz bandpass)',
            fontsize=11)
        rms_v = plot_acc_row_freq(axes_f[0], res_v)
        rms_l = plot_acc_row_freq(axes_f[1], res_l)
        plt.tight_layout()
        if save_dir:
            out_f = os.path.join(save_dir, f'{fname_base}_frequency.png')
            plt.savefig(out_f, dpi=300)
            print(f'    → saved: {out_f}')
        plt.close(fig_f)

        for label, rms_vals in [('Vert', rms_v), ('Lat', rms_l)]:
            if not rms_vals:
                continue
            parts = []
            for f1, f2, sim_rms, meas_rms in rms_vals:
                ratio = sim_rms / (meas_rms + 1e-12)
                parts.append(f'{f1:g}-{f2:g}Hz S/M={ratio:.2f}')
            print(f'    [{part_name} {label} band RMS ratio] ' + ' | '.join(parts))


def plot_irr_row_mileage(ax_row, best, ylabel, tag):
    if best is None:
        for ax in ax_row:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    ax = ax_row[0]
    ax.plot(best['axis_km'], best['sim_ds_mm'],  lw=1.6, ls='-',  color='#2A9D8F', label=f'Sim input ({tag})')
    ax.plot(best['axis_km'], best['meas_ds_mm'], lw=1.4, ls='--', color='#E76F51', alpha=0.95, label='Measured')
    ax.set_title(f'Mileage Domain – Raw  (offset={best["offset_m"]:.0f} m)')
    ax.set_xlabel('Mileage (km)'); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, ls='--'); ax.legend(fontsize=8, frameon=False)

    ax = ax_row[1]
    ax.plot(best['axis_km'], best['sim_lp_mm'],  lw=1.9, ls='-',  color='#264653', label=f'Sim low-pass ({tag})')
    ax.plot(best['axis_km'], best['meas_lp_mm'], lw=1.7, ls='--', color='#F4A261', alpha=0.95, label='Measured low-pass')
    ax.set_title('Mileage Domain – Low-pass')
    ax.set_xlabel('Mileage (km)'); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, ls='--'); ax.legend(fontsize=8, frameon=False)


def plot_irr_row_freq(ax_row, best, ylabel, dt_cmp_irr, fs_cmp_irr, dx_m):
    if best is None:
        for ax in ax_row:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return

    fs, Pa = fft_amp(best['sim_ds_mm'], dt_cmp_irr)
    fm, Pm = fft_amp(best['meas_ds_mm'], dt_cmp_irr)
    ax = ax_row[0]
    ax.plot(fs, Pa, lw=1.8, ls='-',  color='#3A86FF', label='Sim FFT')
    ax.plot(fm, Pm, lw=1.6, ls='--', color='#8338EC', alpha=0.95, label='Meas FFT')
    ax.set_xlim(0, min(100, fs_cmp_irr/2)); ax.set_title('FFT Amplitude')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Amplitude (mm)')
    ax.grid(True, alpha=0.22, ls='--'); ax.legend(fontsize=8, frameon=False)

    fws, Ps = _welch_spatial_psd(best['sim_ds_mm'], dx_m)
    fwm, Pm2 = _welch_spatial_psd(best['meas_ds_mm'], dx_m)
    ax = ax_row[1]
    keep_s = fws > 0
    keep_m = fwm > 0
    if np.count_nonzero(keep_s) == 0 or np.count_nonzero(keep_m) == 0:
        ax.text(0.5, 0.5, 'PSD calculation failed', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.loglog(fws[keep_s], Ps[keep_s] + 1e-30, lw=1.8, ls='-', color='#1D3557', label='Sim PSD')
        ax.loglog(fwm[keep_m], Pm2[keep_m] + 1e-30, lw=1.6, ls='--', color='#E63946', alpha=0.95, label='Meas PSD')
    ax.set_title(f'Spatial PSD (Welch, dx={dx_m:.4g} m)')
    ax.set_xlabel('Spatial frequency (1/m)'); ax.set_ylabel('PSD (mm²/(1/m))')
    ax.grid(True, alpha=0.22, ls='--'); ax.legend(fontsize=8, frameon=False)

    
def irre_analysis(file_path, input_path, dynamic_irr_path, lead_time_s_irr=2.0, f_cut_irr=25.0, meas_spacing=0.25, save_dir=None):
    data = np.load(file_path)
    jsion = pd.read_json(input_path)
    v = jsion['vx_set']
    dynamic_irr = pd.read_csv(dynamic_irr_path)

    # ── DOF indices (0-based) ──────────────────────────────────────────────
    # WS1: Z=16, Roll=17  |  WS2: Z=21  |  WS3: Z=26  |  WS4: Z=31
    # 动态垂向不平顺 = 车轮中心垂向位移 - 名义滚动圆半径 R0
    # 仿真 DOF 以静平衡为原点，故 Z_wc_abs = R0 + X[:,ws_Z]
    # ⟹ 动态不平顺 = X[:,ws_Z_dof]（R0 已隐式含于坐标零点，无需显式减去）
    X_full        = data['X']                   # (Nt, 35)
    Nt_X          = X_full.shape[0]
    irre_dist_full = data['Irre_distance_m']    # (Nt+1,) — 与 X 错一步，对齐截断
    irre_dist_full = irre_dist_full[:Nt_X]      # 对齐为 (Nt,)

    # 垂向：使用 WS1 车轮中心垂向位移（DOF 16）代表左/右轨垂向动态不平顺
    irre_bz_L_full = X_full[:, 16]             # (Nt,) m，相对静平衡
    irre_bz_R_full = X_full[:, 16]             # 轮心 Z 无左右之分；如需分轨可加 ±roll 贡献

    # 横向：仍使用仿真输入横向不平顺（符号修正保持原逻辑）
    irre_by_L_full = data['Irre_by_L_ref'][:Nt_X]
    irre_by_R_full = data['Irre_by_R_ref'][:Nt_X]

    dt_irr = float(data['dt'])
    t_irre = np.arange(Nt_X) * dt_irr
    lead_idx = np.searchsorted(t_irre, lead_time_s_irr)
    irre_bz_L = irre_bz_L_full[lead_idx:]
    irre_bz_R = irre_bz_R_full[lead_idx:]
    irre_by_L = -irre_by_L_full[lead_idx:]  # sign-flipped: sim lateral convention opposite to inspection
    irre_by_R = -irre_by_R_full[lead_idx:]
    irre_dist = irre_dist_full[lead_idx:]
    irre_abs_km = (float(data['Track_abs_mileage_m'][0]) + irre_dist) / 1000.0
    meas_mileage_km = pd.to_numeric(dynamic_irr['里程'],   errors='coerce').to_numpy(dtype=float)
    meas_left_vert  = pd.to_numeric(dynamic_irr['左高低'], errors='coerce').to_numpy(dtype=float)   # mm
    meas_left_lat   = pd.to_numeric(dynamic_irr['左轨向'], errors='coerce').to_numpy(dtype=float)   # mm
    meas_right_vert = pd.to_numeric(dynamic_irr['右高低'], errors='coerce').to_numpy(dtype=float)
    meas_right_lat  = pd.to_numeric(dynamic_irr['右轨向'], errors='coerce').to_numpy(dtype=float)
    ord_m = np.argsort(meas_mileage_km)
    meas_mileage_km = meas_mileage_km[ord_m]
    meas_left_vert  = meas_left_vert[ord_m];  meas_left_lat  = meas_left_lat[ord_m]
    meas_right_vert = meas_right_vert[ord_m]; meas_right_lat = meas_right_lat[ord_m]
    m_uniq, fi = np.unique(meas_mileage_km, return_index=True)
    ml_vert = meas_left_vert[fi];  ml_lat = meas_left_lat[fi]
    mr_vert = meas_right_vert[fi]; mr_lat = meas_right_lat[fi]
    print(f'Measured mileage: [{m_uniq[0]:.4f}, {m_uniq[-1]:.4f}] km  '
      f'| left-vert std={np.nanstd(ml_vert):.3f} mm  left-lat std={np.nanstd(ml_lat):.3f} mm')

    v_mps_loc   = float(v.iloc[0] if hasattr(v, 'iloc') else v[0]) / 3.6
    dt_cmp_irr  = meas_spacing / v_mps_loc
    fs_cmp_irr  = 1.0 / dt_cmp_irr
    best_vert_irr = find_best_offest(irre_bz_L, irre_abs_km, m_uniq, ml_vert, fs_cmp_irr)
    if best_vert_irr is not None:
        best_lat_irr = apply_fixed_offset(
            irre_by_L, irre_abs_km, m_uniq, ml_lat,
            fixed_offset_m=best_vert_irr['offset_m'], fs_cmp_irr=fs_cmp_irr)
    else:
        best_lat_irr = None
    if best_vert_irr:
        print(f'[Vert-Irre] offset={best_vert_irr["offset_m"]:.1f} m  '
            f'corr={best_vert_irr["corr"]:.4f}  NRMSE={best_vert_irr["nrmse"]:.4f}')
    if best_lat_irr:
        print(f'[Lat-Irre]  offset={best_lat_irr["offset_m"]:.1f} m  (locked to vertical offset)'
              f'\n            corr={best_lat_irr["corr"]:.4f}  NRMSE={best_lat_irr["nrmse"]:.4f}')
    # 图1：里程域对比（拉长x轴）
    fig_m, axes_m = plt.subplots(2, 2, figsize=(30, 8.5))
    fig_m.suptitle(
        'Simulation Dynamic Irregularity vs Measured Dynamic Irregularity (Mileage Domain)\n'
        f'(lead-in {lead_time_s_irr:.0f} s trimmed, lateral sign-corrected)',
        fontsize=12)
    plot_irr_row_mileage(axes_m[0], best_vert_irr, 'Vertical Irregularity (mm)', 'WS1_Z−R₀')
    plot_irr_row_mileage(axes_m[1], best_lat_irr,  'Lateral Irregularity (mm)',  '−Irre_by_L')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'irre_comparison.png'), dpi=300)
        plt.savefig(os.path.join(save_dir, 'irre_comparison_mileage.png'), dpi=300)
    plt.close(fig_m)

    # 图2：频域对比（单独绘图）
    fig_f, axes_f = plt.subplots(2, 2, figsize=(16, 8.5))
    fig_f.suptitle(
        'Frequency / Spatial-Domain Comparison of Dynamic Irregularity (FFT + Spatial PSD)',
        fontsize=12)
    plot_irr_row_freq(axes_f[0], best_vert_irr, 'Vertical Irregularity (mm)', dt_cmp_irr, fs_cmp_irr, meas_spacing)
    plot_irr_row_freq(axes_f[1], best_lat_irr,  'Lateral Irregularity (mm)',  dt_cmp_irr, fs_cmp_irr, meas_spacing)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'irre_comparison_frequency.png'), dpi=300)
    plt.close(fig_f)
    # 返回对齐信息供 accel_analysis 使用
    return dict(irre_abs_km=irre_abs_km, irre_dist=irre_dist,
                best_offset_m=best_vert_irr['offset_m'] if best_vert_irr else 0.0,
                v_mps=v_mps_loc)


if __name__ == "__main__":
    target_file = 'results\\long_distance\\高速客车-外部导入-vehicle-10km_moving_window-20260622_195458\\files\\simulation_result.npz'
    input_path = 'results\\long_distance\\高速客车-外部导入-vehicle-10km_moving_window-20260622_195458\\files\\argparse_params.json'
    dynamic_irr_path = 'preprocessing/动检数据/呼局/20210416/处理后/动检上行20210416-238-363.aligned.csv'
    selected_file = target_file
    abs_target = os.path.abspath(target_file)

    lead_time_s_irr = 2.0
    f_cut_irr       = 25.0
    meas_spacing    = 0.25


    selected_run_dir = os.path.dirname(os.path.dirname(abs_target)) if os.path.basename(os.path.dirname(abs_target)).lower() == 'files' else None
    figures_dir = os.path.join(selected_run_dir, 'figures')
    load_and_analyze(selected_file, save_dir=figures_dir, show=False)
    align_info = irre_analysis(selected_file, input_path, dynamic_irr_path,
                               lead_time_s_irr, f_cut_irr, meas_spacing, save_dir=figures_dir)
    if align_info is not None:
        accel_analysis(
            file_path        = selected_file,
            dynamic_irr_path = dynamic_irr_path,
            irre_abs_km      = align_info['irre_abs_km'],
            irre_dist        = align_info['irre_dist'],
            best_offset_m    = align_info['best_offset_m'],
            f_cut_acc        = 200.0,
            meas_spacing     = meas_spacing,
            v_mps            = align_info['v_mps'],
            save_dir         = figures_dir,
        )
    

    