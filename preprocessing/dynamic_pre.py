from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_io import detect_text_encoding, ensure_mileage_sorted, save_csv


DYNAMIC_PSD_CHANNELS = (
    "左高低",
    "右高低",
    "左轨向",
    "右轨向",
    "三角坑",
    "轨距",
    "超高",
    "横向加速度(g)",
    "垂向加速度(g)",
)

DYNAMIC_CHANNEL_LABELS = {
    "左高低": "Left vertical irregularity",
    "右高低": "Right vertical irregularity",
    "左轨向": "Left alignment irregularity",
    "右轨向": "Right alignment irregularity",
    "三角坑": "Cross-level irregularity",
    "轨距": "Gauge irregularity",
    "超高": "Superelevation",
    "横向加速度(g)": "Carbody lateral acceleration",
    "垂向加速度(g)": "Carbody vertical acceleration",
}

DYNAMIC_CHANNEL_UNITS = {
    "横向加速度(g)": "g",
    "垂向加速度(g)": "g",
}


DYNAMIC_CHANNEL_MAP: Dict[str, list[str]] = {
    "左高低": ["左高低(mm)", "左高低", "高低左", "左高低值"],
    "右高低": ["右高低(mm)", "右高低", "高低右", "右高低值"],
    "左轨向": ["左轨向(mm)", "左轨向", "轨向左", "左轨向值"],
    "右轨向": ["右轨向(mm)", "右轨向", "轨向右", "右轨向值"],
    "三角坑": ["三角坑(mm)", "三角坑"],
    "轨距": ["轨距(mm)", "轨距"],
    "超高": ["超高(mm)", "超高"],
    "水平": ["水平(mm)", "水平"],
    "横向加速度(g)": ["横向加速度(g)", "横向加速度"],
    "垂向加速度(g)": ["垂向加速度(g)", "垂向加速度"],
}


def _pick_column(df: pd.DataFrame, aliases: list[str], fallback_index: int | None = None) -> str | None:
    cols = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in cols}

    for a in aliases:
        if a in cols:
            return a
        if a.lower() in lower_map:
            return lower_map[a.lower()]

    for c in cols:
        if any(k in c for k in aliases):
            return c

    if fallback_index is not None and 0 <= fallback_index < len(cols):
        return cols[fallback_index]
    return None


def read_dynamic_txt(file_path: str | Path) -> pd.DataFrame:
    p = Path(file_path)
    enc = detect_text_encoding(p)

    # 该文件通常是逗号分隔，第一行为标题
    df = pd.read_csv(p, encoding=enc, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def unify_dynamic_mileage(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    统一动检里程为绝对里程(km)：
    绝对里程 = 公里标 + Meters / 1000
    """
    cols = [str(c).strip() for c in raw_df.columns]

    col_km = _pick_column(raw_df, ["公里标", "Kilometer", "KM"], fallback_index=0)
    col_m = _pick_column(raw_df, ["Meters", "米", "里程偏移"], fallback_index=1)
    if col_km is None or col_m is None:
        raise ValueError("动检文件缺少公里标/米偏移列，无法统一里程。")

    km = pd.to_numeric(raw_df[col_km], errors="coerce")
    meters = pd.to_numeric(raw_df[col_m], errors="coerce")

    out = pd.DataFrame({"里程": km + meters / 1000.0})

    # 优先按别名匹配；若编码导致乱码，退化使用固定列位置
    fallback_idx = {
        "左高低": 4,
        "右高低": 5,
        "左轨向": 6,
        "右轨向": 7,
        "轨距": 8,
        "超高": 9,
        "水平": 10,
        "三角坑": 11,
    }

    for std_name, aliases in DYNAMIC_CHANNEL_MAP.items():
        c = _pick_column(raw_df, aliases, fallback_index=fallback_idx.get(std_name))
        if c is not None:
            out[std_name] = pd.to_numeric(raw_df[c], errors="coerce")

    out = ensure_mileage_sorted(out, "里程")
    return out



def _next_lower_power_of_two(n: int) -> int:
    if n < 2:
        return 1
    return 1 << (int(n).bit_length() - 1)


def _welch_spatial_psd(y: np.ndarray, dx_m: float, nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD for uniformly spaced spatial signal in mm."""
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


def plot_dynamic_irregularity_psd(
    dynamic_df: pd.DataFrame,
    out_dir: str | Path,
    file_prefix: str = "dynamic_irregularity",
    channels: tuple[str, ...] = DYNAMIC_PSD_CHANNELS,
    target_dx_m: float | None = None,
) -> dict:
    """Plot spatial PSD for dynamic irregularities and carbody acceleration channels."""
    if "里程" not in dynamic_df.columns:
        raise ValueError("dynamic_df is missing the mileage column required for PSD.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_km = pd.to_numeric(dynamic_df["里程"], errors="coerce").to_numpy(dtype=float)
    valid_x = np.isfinite(x_km)
    if valid_x.sum() < 8:
        raise ValueError("Insufficient valid dynamic mileage samples for PSD.")

    order = np.argsort(x_km[valid_x])
    x_m = x_km[valid_x][order] * 1000.0
    x_unique, unique_idx = np.unique(x_m, return_index=True)
    if len(x_unique) < 8:
        raise ValueError("Insufficient unique dynamic mileage samples for PSD.")

    dx_raw = np.diff(x_unique)
    dx_raw = dx_raw[np.isfinite(dx_raw) & (dx_raw > 0)]
    if len(dx_raw) == 0:
        raise ValueError("Invalid dynamic mileage spacing for PSD.")
    dx_m = float(target_dx_m) if target_dx_m is not None else float(np.median(dx_raw))
    if not np.isfinite(dx_m) or dx_m <= 0:
        raise ValueError("Invalid PSD resampling interval.")

    x_grid = np.arange(x_unique[0], x_unique[-1] + dx_m * 0.5, dx_m)
    if len(x_grid) < 8:
        raise ValueError("Insufficient resampled points for PSD.")

    available = [c for c in channels if c in dynamic_df.columns]
    if not available:
        raise ValueError("No dynamic channels are available for PSD.")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
    axes = axes.ravel()
    csv_paths: dict[str, str] = {}

    for i, col in enumerate(available):
        ax = axes[i]
        y_all = pd.to_numeric(dynamic_df.loc[valid_x, col], errors="coerce").to_numpy(dtype=float)[order]
        y_unique = y_all[unique_idx]
        finite = np.isfinite(y_unique)
        label = DYNAMIC_CHANNEL_LABELS.get(col, col)
        unit = DYNAMIC_CHANNEL_UNITS.get(col, "mm")
        if finite.sum() < 8:
            ax.text(0.5, 0.5, "Insufficient valid samples", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue

        y_grid = np.interp(x_grid, x_unique[finite], y_unique[finite])
        freq, psd = _welch_spatial_psd(y_grid, dx_m)
        if len(freq) == 0:
            ax.text(0.5, 0.5, "PSD calculation failed", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue

        keep = freq > 0
        ax.loglog(freq[keep], psd[keep] + 1e-30, lw=1.35, color="#1f77b4")
        ax.set_title(label)
        ax.set_xlabel("Spatial frequency (1/m)")
        ax.set_ylabel(f"PSD ({unit}^2/(1/m))")
        ax.grid(True, which="both", alpha=0.28, ls="--")

        csv_file = out_dir / f"{file_prefix}_{col}_psd.csv"
        pd.DataFrame({"spatial_frequency_1pm": freq, f"psd_{unit}2_per_1pm": psd}).to_csv(
            csv_file, index=False, encoding="utf-8-sig"
        )
        csv_paths[col] = str(csv_file)

    for j in range(len(available), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Dynamic irregularity and carbody acceleration spatial PSD (dx={dx_m:.4g} m, Welch)",
        fontsize=14,
    )
    png_path = out_dir / f"{file_prefix}_psd.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "plot": str(png_path),
        "csv": csv_paths,
        "channels": available,
        "dx_m": dx_m,
        "samples": int(len(x_grid)),
    }

def process_dynamic_file(raw_file: str | Path, save_file: str | Path | None = None) -> pd.DataFrame:
    raw_df = read_dynamic_txt(raw_file)
    dynamic_df = unify_dynamic_mileage(raw_df)
    if save_file is not None:
        save_csv(dynamic_df, save_file, encoding="utf-8-sig")
    return dynamic_df
