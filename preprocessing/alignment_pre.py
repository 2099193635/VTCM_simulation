from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_io import ensure_mileage_sorted, save_csv


CHANNEL_MAP: Dict[str, Tuple[str, str]] = {
    "左高低": ("左高低", "实测左高低"),
    "右高低": ("右高低", "实测右高低"),
    "左轨向": ("左轨向", "实测左轨向"),
    "右轨向": ("右轨向", "实测右轨向"),
    "三角坑": ("三角坑", "实测三角坑"),
    "轨距": ("轨距", "实测轨距"),
    "超高": ("超高", "实测水平"),
}

CHANNEL_ENGLISH: Dict[str, str] = {
    "左高低": "Left Vertical",
    "右高低": "Right Vertical",
    "左轨向": "Left Alignment",
    "右轨向": "Right Alignment",
    "三角坑": "Cross-level",
    "轨距": "Gauge",
    "超高": "Superelevation",
}

_PALETTE = {
    "dynamic": "#2B6CB0",   # deep blue — inspection car
    "before":  "#C0392B",   # brick red  — before DTW
    "after":   "#1A7340",   # forest green — after DTW
}


def _interp_channel(df: pd.DataFrame, x_grid: np.ndarray, y_col: str) -> np.ndarray:
    x = pd.to_numeric(df["里程"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.full_like(x_grid, np.nan)
    x2 = x[valid]
    y2 = y[valid]
    order = np.argsort(x2)
    x2 = x2[order]
    y2 = y2[order]
    return np.interp(x_grid, x2, y2)


def _pick_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = [str(c).strip() for c in df.columns]
    for cand in candidates:
        if cand in cols:
            return cand
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in cols:
        if any(token in c for token in candidates):
            return c
    return None


def _robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.zeros_like(x)
    x2 = x.copy()
    med = np.nanmedian(x2[finite])
    mad = np.nanmedian(np.abs(x2[finite] - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        scale = np.nanstd(x2[finite])
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    x2[finite] = (x2[finite] - med) / scale
    x2[~finite] = 0.0
    return x2


def _interp_on_grid(df: pd.DataFrame, x_grid: np.ndarray, y_col: str) -> np.ndarray:
    x = pd.to_numeric(df["里程"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.full_like(x_grid, np.nan, dtype=float)
    x2 = x[valid]
    y2 = y[valid]
    order = np.argsort(x2)
    x2 = x2[order]
    y2 = y2[order]
    return np.interp(x_grid, x2, y2)


def _dtw_path_banded(x: np.ndarray, y: np.ndarray, band: int) -> tuple[list[tuple[int, int]], float]:
    """
    Sakoe-Chiba 带约束 DTW，支持多维特征（形状 (n, d) 或 (n,)）。
    返回 warp 路径（0 基索引对列表）和累计代价。
    """
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]
    n, m = x.shape[0], y.shape[0]
    band = max(int(band), abs(n - m))

    INF = float("inf")

    # direction: 0=对角(i-1,j-1) 1=上(i-1,j) 2=左(i,j-1) -1=起点
    # n×m int8 —— 7000×7000 ≈ 49 MB
    direction = np.full((n, m), -1, dtype=np.int8)

    prev = np.full(m, INF)
    cur  = np.full(m, INF)

    # ── 初始化第 0 行 ──────────────────────────────────────────────────
    j_hi_0 = min(m - 1, band)
    prev[0] = float(np.sum((x[0] - y[0]) ** 2))
    direction[0, 0] = -1
    for j in range(1, j_hi_0 + 1):
        prev[j] = prev[j - 1] + float(np.sum((x[0] - y[j]) ** 2))
        direction[0, j] = 2

    # ── 逐行填充 ──────────────────────────────────────────────────────
    for i in range(1, n):
        j_lo = max(0, i - band)
        j_hi = min(m - 1, i + band)
        cur[:] = INF
        xi = x[i]
        for j in range(j_lo, j_hi + 1):
            local = float(np.sum((xi - y[j]) ** 2))
            d_c = prev[j - 1] if j > 0 else INF
            u_c = prev[j]
            l_c = cur[j - 1]  if j > 0 else INF

            if d_c <= u_c and d_c <= l_c:
                best, bd = d_c, np.int8(0)
            elif u_c <= l_c:
                best, bd = u_c, np.int8(1)
            else:
                best, bd = l_c, np.int8(2)

            if best < INF:
                cur[j] = local + best
                direction[i, j] = bd

        prev, cur = cur, prev

    final_cost = float(prev[m - 1])
    if not np.isfinite(final_cost):
        raise ValueError(
            "DTW 受限带宽过窄，无法到达终点；请增大 band_ratio 或 grid_step_km。"
        )

    # ── 回溯路径 ──────────────────────────────────────────────────────
    path: list[tuple[int, int]] = []
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        path.append((i, j))
        d = int(direction[i, j])
        if d == 0:
            i -= 1; j -= 1
        elif d == 1:
            i -= 1
        else:
            j -= 1
    path.append((0, 0))
    path.reverse()
    return path, final_cost


def align_dynamic_to_static_by_dtw(
    dynamic_df: pd.DataFrame,
    static_df: pd.DataFrame,
    grid_step_km: float = 0.001,
    band_ratio: float = 0.04,
) -> dict:
    """
    以静检里程为基准修正动检里程。

    DTW 只用于估计动检相对静检的稳健里程偏移；最终只修改动检
    `里程` 列，不对任一通道的数值序列做重采样或非线性扭曲。
    """
    # 旧的粗对齐估计表示“加到静检上以贴近动检”的偏移；
    # 这里反向使用，把动检先粗略平移到静检基准。
    coarse_static_to_dynamic_km = estimate_shift_by_superelevation(dynamic_df, static_df)
    coarse_dynamic_to_static_km = -coarse_static_to_dynamic_km
    dynamic_shifted = apply_shift_to_dynamic(dynamic_df, coarse_dynamic_to_static_km)

    channel_specs = [
        ("左高低", "实测左高低", ["左高低"]),
        ("右高低", "实测右高低", ["右高低"]),
        ("左轨向", "实测左轨向", ["左轨向"]),
        ("右轨向", "实测右轨向", ["右轨向"]),
        ("轨距", "实测轨距", ["轨距"]),
        ("超高", "实测水平", ["超高", "水平"]),
    ]

    used_specs = []
    for dyn_hint, sta_col, dyn_aliases in channel_specs:
        dyn_col = _pick_first_existing_column(dynamic_shifted, dyn_aliases)
        if dyn_col is None or sta_col not in static_df.columns:
            continue
        used_specs.append((dyn_col, sta_col, dyn_hint))

    if not used_specs:
        raise ValueError("未找到可用于 DTW 的共同通道，请检查动检/静检字段名。")

    dmin = float(pd.to_numeric(dynamic_shifted["里程"], errors="coerce").min())
    dmax = float(pd.to_numeric(dynamic_shifted["里程"], errors="coerce").max())
    smin = float(pd.to_numeric(static_df["里程"], errors="coerce").min())
    smax = float(pd.to_numeric(static_df["里程"], errors="coerce").max())
    x0 = max(dmin, smin)
    x1 = min(dmax, smax)
    if not (np.isfinite(x0) and np.isfinite(x1) and x1 > x0):
        raise ValueError("动静检里程范围无有效交集，无法进行 DTW 对齐。")

    x_grid = np.arange(x0, x1 + grid_step_km * 0.5, grid_step_km)
    if len(x_grid) < 50:
        raise ValueError("用于 DTW 的采样点过少，请适当减小 grid_step_km 或检查里程范围。")

    dynamic_features = []
    static_features = []
    channel_names = []
    for dyn_col, sta_col, name in used_specs:
        dyn_seq = _interp_on_grid(dynamic_shifted, x_grid, dyn_col)
        sta_seq = _interp_on_grid(static_df, x_grid, sta_col)
        valid = np.isfinite(dyn_seq) & np.isfinite(sta_seq)
        if valid.sum() < 20:
            continue
        dyn_seq = _robust_zscore(np.where(valid, dyn_seq, np.nan))
        sta_seq = _robust_zscore(np.where(valid, sta_seq, np.nan))
        dynamic_features.append(dyn_seq)
        static_features.append(sta_seq)
        channel_names.append(name)

    if not dynamic_features:
        raise ValueError("DTW 特征通道有效点不足，无法对齐。")

    dyn_mat = np.column_stack(dynamic_features)
    sta_mat = np.column_stack(static_features)

    band = max(int(len(x_grid) * band_ratio), 8)
    path, dtw_cost = _dtw_path_banded(dyn_mat, sta_mat, band=band)
    path_arr = np.asarray(path, dtype=int)
    dyn_idx = path_arr[:, 0]
    sta_idx = path_arr[:, 1]

    dynamic_grid = x_grid[dyn_idx]
    static_grid = x_grid[sta_idx]
    additional_shift_samples_km = static_grid - dynamic_grid
    additional_shift_km = float(np.nanmedian(additional_shift_samples_km))
    total_dynamic_shift_km = float(coarse_dynamic_to_static_km + additional_shift_km)

    dynamic_aligned = apply_shift_to_dynamic(dynamic_df, total_dynamic_shift_km)
    static_aligned = ensure_mileage_sorted(static_df)

    summary = {
        "alignment_basis": "static",
        "correction_mode": "constant_mileage_shift",
        "coarse_static_to_dynamic_km": float(coarse_static_to_dynamic_km),
        "coarse_dynamic_to_static_km": float(coarse_dynamic_to_static_km),
        "additional_shift_km": additional_shift_km,
        "dynamic_shift_km": total_dynamic_shift_km,
        "median_shift_km": total_dynamic_shift_km,
        "mean_additional_shift_km": float(np.nanmean(additional_shift_samples_km)),
        "dtw_cost": float(dtw_cost),
        "band": int(band),
        "grid_step_km": float(grid_step_km),
        "used_channels": channel_names,
        "path_length": int(len(path)),
    }

    return {
        "dynamic_before": ensure_mileage_sorted(dynamic_df),
        "dynamic_shifted": dynamic_shifted,
        "dynamic_aligned": dynamic_aligned,
        "static_before": static_df,
        "static_aligned": static_aligned,
        "dtw_summary": summary,
    }


def align_static_by_dtw(
    dynamic_df: pd.DataFrame,
    static_df: pd.DataFrame,
    grid_step_km: float = 0.001,
    band_ratio: float = 0.04,
) -> dict:
    """兼容旧调用名；实际执行“以静检为基准修正动检里程”。"""
    return align_dynamic_to_static_by_dtw(
        dynamic_df=dynamic_df,
        static_df=static_df,
        grid_step_km=grid_step_km,
        band_ratio=band_ratio,
    )

def estimate_shift_by_superelevation(
    dynamic_df: pd.DataFrame,
    static_df: pd.DataFrame,
    max_shift_m: float = 0.8,
    grid_step_km: float = 0.0001,
) -> float:
    """基于超高/水平通道估计里程偏移量，返回单位 km（加到静检里程上）。"""
    if "超高" not in dynamic_df.columns or "实测水平" not in static_df.columns:
        return 0.0

    ddf = ensure_mileage_sorted(dynamic_df)
    sdf = ensure_mileage_sorted(static_df)

    x0 = max(ddf["里程"].min(), sdf["里程"].min())
    x1 = min(ddf["里程"].max(), sdf["里程"].max())
    if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0:
        return 0.0

    grid = np.arange(x0, x1, grid_step_km)
    if len(grid) < 100:
        return 0.0

    dyn = _interp_channel(ddf, grid, "超高")
    sta = _interp_channel(sdf, grid, "实测水平")
    valid = np.isfinite(dyn) & np.isfinite(sta)
    dyn = dyn[valid]
    sta = sta[valid]
    if len(dyn) < 200:
        return 0.0

    dyn = dyn - np.nanmean(dyn)
    sta = sta - np.nanmean(sta)

    corr = np.correlate(dyn, sta, mode="full")
    lags = np.arange(-len(sta) + 1, len(dyn))
    max_lag = int(max_shift_m / (grid_step_km * 1000.0))
    keep = np.abs(lags) <= max_lag
    if keep.sum() == 0:
        return 0.0

    best_lag = int(lags[keep][np.argmax(corr[keep])])
    return float(best_lag * grid_step_km)


def apply_shift_to_static(static_df: pd.DataFrame, shift_km: float) -> pd.DataFrame:
    out = static_df.copy()
    out["里程"] = pd.to_numeric(out["里程"], errors="coerce") + shift_km
    return ensure_mileage_sorted(out)




def apply_shift_to_dynamic(dynamic_df: pd.DataFrame, shift_km: float) -> pd.DataFrame:
    out = dynamic_df.copy()
    out["里程"] = pd.to_numeric(out["里程"], errors="coerce") + shift_km
    return ensure_mileage_sorted(out)


def flip_dynamic_alignment_irregularity(
    dynamic_df: pd.DataFrame,
    columns: tuple[str, ...] = ("左轨向", "右轨向"),
) -> tuple[pd.DataFrame, list[str]]:
    """仅将动检轨向不平顺反向；里程和其它通道保持不变。"""
    out = dynamic_df.copy()
    flipped: list[str] = []
    for col in columns:
        if col not in out.columns:
            continue
        out[col] = -pd.to_numeric(out[col], errors="coerce")
        flipped.append(col)
    return ensure_mileage_sorted(out), flipped

def get_alignment_range(
    dynamic_df: pd.DataFrame,
    static_df: pd.DataFrame,
    prefer_static: bool = True,
) -> tuple[float, float]:
    """
    获取对齐里程范围（单位 km）。
    - 默认以静检范围为主，再与动检取交集；
    - 若交集无效，退化为两者的严格交集检查。
    """
    dmin = float(pd.to_numeric(dynamic_df["里程"], errors="coerce").min())
    dmax = float(pd.to_numeric(dynamic_df["里程"], errors="coerce").max())
    smin = float(pd.to_numeric(static_df["里程"], errors="coerce").min())
    smax = float(pd.to_numeric(static_df["里程"], errors="coerce").max())

    if prefer_static:
        x0 = max(dmin, smin)
        x1 = min(dmax, smax)
    else:
        x0 = max(dmin, smin)
        x1 = min(dmax, smax)

    if not (np.isfinite(x0) and np.isfinite(x1) and x1 > x0):
        raise ValueError("动静检里程范围无有效交集，无法对齐。")
    return x0, x1


def clip_by_mileage_range(df: pd.DataFrame, x0: float, x1: float, mileage_col: str = "里程") -> pd.DataFrame:
    out = df.copy()
    out[mileage_col] = pd.to_numeric(out[mileage_col], errors="coerce")
    out = out[out[mileage_col].between(x0, x1, inclusive="both")]
    return ensure_mileage_sorted(out, mileage_col=mileage_col)


def _find_focus_window(
    df: pd.DataFrame,
    col: str,
    half_len_km: float = 3.0,
    gradient_quantile: float = 0.90,
) -> tuple[float, float] | None:
    """Return (x0, x1) window centred on the region with highest gradient."""
    x = pd.to_numeric(df["里程"], errors="coerce").to_numpy(dtype=float)
    if col not in df.columns:
        return None
    y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 20:
        return None
    xs, ys = x[valid], y[valid]
    grad = np.abs(np.gradient(ys, xs))
    finite_grad = grad[np.isfinite(grad)]
    if len(finite_grad) == 0:
        return None
    thr = np.quantile(finite_grad, gradient_quantile)
    hot = grad >= thr
    center = float(np.median(xs[hot])) if hot.any() else float(np.median(xs))
    x0 = max(float(xs.min()), center - half_len_km)
    x1 = min(float(xs.max()), center + half_len_km)
    if x1 - x0 < 0.5:
        return (float(xs.min()), float(xs.max()))
    return (x0, x1)


def plot_alignment_before_after(
    dynamic_before: pd.DataFrame,
    dynamic_after: pd.DataFrame,
    static_df: pd.DataFrame,
    out_dir: str | Path,
    focus_change_quantile: float = 0.90,
    focus_half_km: float = 3.0,
    mileage_start: float | None = None,
    mileage_end: float | None = None,
) -> list[Path]:
    """
    Save one figure per channel. Each figure has two stacked subplots:
      - top   : dynamic before correction vs static reference
      - bottom: dynamic after correction vs static reference
    The static mileage axis is kept fixed as the reference.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset":   "stix",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.8,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
    })

    xlim = None
    if mileage_start is not None and mileage_end is not None:
        x0 = float(mileage_start)
        x1 = float(mileage_end)
        if x1 < x0:
            x0, x1 = x1, x0
        if x1 > x0:
            xlim = (x0, x1)

    if xlim is None:
        xlim = _find_focus_window(
            static_df, "实测水平",
            half_len_km=focus_half_km,
            gradient_quantile=focus_change_quantile,
        )
    if xlim is None:
        sxs = pd.to_numeric(static_df["里程"], errors="coerce").dropna().to_numpy()
        if len(sxs) > 0:
            xlim = (float(sxs.min()), float(sxs.max()))

    saved: list[Path] = []

    for ch_name, (d_col, s_col) in CHANNEL_MAP.items():
        en = CHANNEL_ENGLISH.get(ch_name, ch_name)

        has_dyn_before = d_col in dynamic_before.columns
        has_dyn_after = d_col in dynamic_after.columns
        has_static = s_col in static_df.columns
        if not has_dyn_before and not has_dyn_after and not has_static:
            continue

        fig, (ax_bef, ax_aft) = plt.subplots(
            2, 1, figsize=(13, 6), sharex=True,
            gridspec_kw={"hspace": 0.38},
        )
        fig.patch.set_facecolor("#FAFAFA")

        def _plot_static_reference(ax):
            if has_static:
                ax.plot(
                    pd.to_numeric(static_df["里程"], errors="coerce"),
                    pd.to_numeric(static_df[s_col], errors="coerce"),
                    color=_PALETTE["after"], lw=1.3, ls="-", alpha=0.88,
                    label="Static reference", zorder=2,
                )

        if has_dyn_before:
            ax_bef.plot(
                pd.to_numeric(dynamic_before["里程"], errors="coerce"),
                pd.to_numeric(dynamic_before[d_col], errors="coerce"),
                color=_PALETTE["before"], lw=1.4, ls="--", alpha=0.90,
                label="Dynamic before correction", zorder=3,
            )
        _plot_static_reference(ax_bef)

        if has_dyn_after:
            ax_aft.plot(
                pd.to_numeric(dynamic_after["里程"], errors="coerce"),
                pd.to_numeric(dynamic_after[d_col], errors="coerce"),
                color=_PALETTE["dynamic"], lw=1.5, ls="-", alpha=0.95,
                label="Dynamic after correction", zorder=3,
            )
        _plot_static_reference(ax_aft)

        for ax, subtitle in ((ax_bef, "Before Dynamic Correction"), (ax_aft, "After Dynamic Correction")):
            ax.set_facecolor("#FFFFFF")
            ax.set_ylabel(f"{en} (mm)", fontsize=10)
            ax.set_title(subtitle, fontsize=10, color="#333333", pad=4)
            ax.legend(fontsize=8.5, frameon=False, loc="upper right")
            ax.grid(True, alpha=0.30, ls="--", lw=0.55, color="#888888")
            if xlim is not None:
                ax.set_xlim(*xlim)

        ax_aft.set_xlabel("Mileage (km)", fontsize=10)

        fig.suptitle(
            f"Dynamic Correction to Static Reference  |  {en}",
            fontsize=12, fontweight="bold", color="#1A1A2E", y=1.01,
        )

        slug = en.replace(" ", "_").replace("-", "").lower()
        out_fn = out_dir / f"alignment_{slug}.png"
        fig.savefig(out_fn, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.show()
        plt.close(fig)
        saved.append(out_fn)
        print(f"  [plot] saved -> {out_fn}")

    return saved

def save_aligned_results(
    dynamic_df: pd.DataFrame,
    static_aligned_df: pd.DataFrame,
    dynamic_out: str | Path,
    static_out: str | Path,
) -> tuple[Path, Path]:
    p1 = save_csv(dynamic_df, dynamic_out)
    p2 = save_csv(static_aligned_df, static_out)
    return p1, p2
