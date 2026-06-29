from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import win32com.client as win32


DEFAULT_ACCELERATION = Path(
    r"preprocessing\动检数据\呼局\20210416\处理后\动检上行20210416-238-363.acceleration.csv"
)
DEFAULT_RAW_DYNAMIC = Path(
    r"preprocessing\动检数据\呼局\20210416\原始文件\动检上行20210416-238-363.txt"
)
DEFAULT_LEDGER = Path(r"preprocessing\台账\京包客专.xls")


def pick_font() -> str:
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "Arial", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    return next((name for name in preferred if name in available), "DejaVu Sans")


def configure_style() -> None:
    font = pick_font()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font, "Arial", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.unicode_minus": False,
            "legend.frameon": False,
        }
    )


def read_xls_sheets_with_excel(path: Path) -> dict[str, pd.DataFrame]:
    excel = win32.DispatchEx("Excel.Application")
    excel.Visible = False
    excel.DisplayAlerts = False
    wb = excel.Workbooks.Open(str(path.resolve()), ReadOnly=True)
    try:
        sheets: dict[str, pd.DataFrame] = {}
        for ws in wb.Worksheets:
            used = ws.UsedRange
            rows = int(used.Rows.Count)
            cols = int(used.Columns.Count)
            if rows < 2 or cols < 1:
                continue
            values = ws.Range(ws.Cells(1, 1), ws.Cells(rows, cols)).Value
            if rows == 1:
                values = (values,)
            table = [list(row) for row in values]
            header = ["" if x is None else str(x).strip() for x in table[1]]
            data = table[2:]
            df = pd.DataFrame(data, columns=header).dropna(how="all").reset_index(drop=True)
            sheets[str(ws.Name)] = df
        return sheets
    finally:
        wb.Close(False)
        excel.Quit()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fix_mojibake(value: object) -> str:
    text = "" if value is None else str(value).strip()
    for source in ("latin1", "cp1252"):
        try:
            fixed = text.encode(source).decode("gbk")
        except UnicodeError:
            continue
        if any("\u4e00" <= ch <= "\u9fff" for ch in fixed):
            return fixed.strip()
    return text


def filter_up_direction(df: pd.DataFrame) -> pd.DataFrame:
    if "行别" not in df.columns:
        return df
    direction = df["行别"].map(fix_mojibake)
    up = direction.eq("上") | direction.str.contains("上", na=False)
    return df.loc[up].copy() if up.any() else df


def direction_short(value: object) -> str:
    text = fix_mojibake(value)
    if text.startswith("左") or text.lower().startswith("l"):
        return "L"
    if text.startswith("右") or text.lower().startswith("r"):
        return "R"
    return text[:1] or "-"


def read_acceleration(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["里程", "横向加速度(g)", "垂向加速度(g)"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[required].copy()
    for col in required:
        out[col] = numeric(out[col])
    out = out.dropna().sort_values("里程").drop_duplicates(subset="里程").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{path} has no valid mileage-acceleration rows.")
    return out


def read_speed_profile(path: Path, x0: float, x1: float) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["里程", "速度(km/h)"])
    usecols = ["里程", "Meters", "速度(km/h)"]
    raw = pd.read_csv(path, encoding="gbk", usecols=usecols)
    mileage = numeric(raw["里程"]) + numeric(raw["Meters"]) / 1000.0
    speed = numeric(raw["速度(km/h)"])
    out = pd.DataFrame({"里程": mileage, "速度(km/h)": speed}).dropna()
    out = out[(out["里程"] >= x0) & (out["里程"] <= x1)].sort_values("里程")
    return out.drop_duplicates(subset="里程").reset_index(drop=True)


def attach_speed(accel: pd.DataFrame, speed: pd.DataFrame) -> pd.DataFrame:
    out = accel.copy()
    if speed.empty:
        out["速度(km/h)"] = np.nan
        return out
    out["速度(km/h)"] = np.interp(out["里程"], speed["里程"], speed["速度(km/h)"])
    return out


def build_curve_table(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = filter_up_direction(sheets.get("曲线表", pd.DataFrame()).copy())
    if df.empty:
        return pd.DataFrame(columns=["start_km", "end_km", "direction", "radius_m", "cant_mm"])
    out = pd.DataFrame(
        {
            "start_km": numeric(df["起点里程"]),
            "end_km": numeric(df["终点里程"]),
            "direction": df["曲线方向"].map(fix_mojibake),
            "radius_m": numeric(df["曲线半径"]),
            "cant_mm": numeric(df["超高"]),
        }
    )
    return out.dropna(subset=["start_km", "end_km"]).query("end_km > start_km").reset_index(drop=True)


def build_bridge_table(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = filter_up_direction(sheets.get("桥梁简表", pd.DataFrame()).copy())
    if df.empty:
        return pd.DataFrame(columns=["start_km", "end_km", "name", "length_m", "center_km"])
    center = numeric(df["中心里程"])
    length = numeric(df["桥全长"])
    out = pd.DataFrame(
        {
            "start_km": center - length / 2000.0,
            "end_km": center + length / 2000.0,
            "center_km": center,
            "name": df["桥名"].map(fix_mojibake),
            "length_m": length,
        }
    )
    return out.dropna(subset=["start_km", "end_km"]).query("end_km > start_km").reset_index(drop=True)


def clip_intervals(df: pd.DataFrame, x0: float, x1: float) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df[(df["end_km"] >= x0) & (df["start_km"] <= x1)].copy()
    out["start_km"] = out["start_km"].clip(lower=x0, upper=x1)
    out["end_km"] = out["end_km"].clip(lower=x0, upper=x1)
    if "center_km" in out.columns:
        out["center_km"] = out["center_km"].clip(lower=x0, upper=x1)
    return out[out["end_km"] > out["start_km"]].reset_index(drop=True)


def uniform_signal(df: pd.DataFrame, col: str, dx_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_m = df["里程"].to_numpy(float) * 1000.0
    y = df[col].to_numpy(float)
    speed = df["速度(km/h)"].to_numpy(float)
    x_grid = np.arange(x_m.min(), x_m.max() + dx_m * 0.5, dx_m)
    y_grid = np.interp(x_grid, x_m, y)
    speed_grid = np.interp(x_grid, x_m, speed) if np.isfinite(speed).any() else np.full_like(x_grid, np.nan)
    return x_grid, y_grid, speed_grid


def local_spectrum(
    mileage_km: np.ndarray,
    signal: np.ndarray,
    speed_kmh: np.ndarray,
    dx_m: float,
    window_m: float,
    step_m: float,
    freq_hz_range: tuple[float, float],
) -> dict[str, np.ndarray]:
    x_m = mileage_km * 1000.0
    nper = int(round(window_m / dx_m))
    step = max(1, int(round(step_m / dx_m)))
    if nper < 64:
        raise ValueError("Window is too short for local spectrum analysis.")
    starts = np.arange(0, len(signal) - nper + 1, step)
    window = np.hanning(nper)
    win_power = np.sum(window**2)
    spatial_freq = np.fft.rfftfreq(nper, d=dx_m)
    centers: list[float] = []
    speeds: list[float] = []
    spectra: list[np.ndarray] = []
    peak_freq: list[float] = []
    peak_wavelength: list[float] = []
    peak_power: list[float] = []

    global_speed_mps = np.nanmedian(speed_kmh) / 3.6
    if not np.isfinite(global_speed_mps) or global_speed_mps <= 0:
        global_speed_mps = 1.0
    freq_hz_axis = spatial_freq * global_speed_mps
    keep_axis = (freq_hz_axis >= freq_hz_range[0]) & (freq_hz_axis <= freq_hz_range[1]) & (spatial_freq > 0)

    for start in starts:
        stop = start + nper
        seg = signal[start:stop]
        finite = np.isfinite(seg)
        if finite.sum() < nper * 0.8:
            continue
        if not finite.all():
            seg = seg.copy()
            seg[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), seg[finite])
        seg = seg - np.mean(seg)
        spec = np.fft.rfft(seg * window)
        psd = (np.abs(spec) ** 2) / ((1.0 / dx_m) * win_power)
        if len(psd) > 2:
            psd[1:-1] *= 2.0

        speed_mps = np.nanmedian(speed_kmh[start:stop]) / 3.6
        if not np.isfinite(speed_mps) or speed_mps <= 0:
            speed_mps = global_speed_mps
        local_freq_hz = spatial_freq * speed_mps
        keep_peak = (local_freq_hz >= freq_hz_range[0]) & (local_freq_hz <= freq_hz_range[1]) & (spatial_freq > 0)
        if not keep_peak.any():
            continue
        idx_candidates = np.flatnonzero(keep_peak)
        peak_idx = idx_candidates[int(np.argmax(psd[keep_peak]))]

        centers.append(float(np.mean(x_m[start:stop]) / 1000.0))
        speeds.append(float(speed_mps * 3.6))
        spectra.append(psd[keep_axis])
        peak_freq.append(float(local_freq_hz[peak_idx]))
        peak_wavelength.append(float(1.0 / spatial_freq[peak_idx]))
        peak_power.append(float(psd[peak_idx]))

    return {
        "centers_km": np.asarray(centers),
        "freq_hz_axis": freq_hz_axis[keep_axis],
        "spectra": np.asarray(spectra).T,
        "speed_kmh": np.asarray(speeds),
        "peak_freq_hz": np.asarray(peak_freq),
        "peak_wavelength_m": np.asarray(peak_wavelength),
        "peak_power": np.asarray(peak_power),
    }


def segment_label(x: float, bridges: pd.DataFrame, curves: pd.DataFrame) -> str:
    in_bridge = bridges[(bridges["start_km"] <= x) & (bridges["end_km"] >= x)]
    in_curve = curves[(curves["start_km"] <= x) & (curves["end_km"] >= x)]
    labels: list[str] = []
    if not in_bridge.empty:
        labels.append(f"Bridge-{int(in_bridge.index[0]) + 1}")
    if not in_curve.empty:
        row = in_curve.iloc[0]
        labels.append(f"Curve-{direction_short(row['direction'])}-R{row['radius_m']:.0f}")
    return "+".join(labels) if labels else "open track"


def add_infrastructure_background(ax, bridges: pd.DataFrame, curves: pd.DataFrame, colors: dict[str, str], centers: bool = False) -> None:
    for row in curves.itertuples(index=False):
        key = "curve_left" if direction_short(row.direction) == "L" else "curve_right"
        ax.axvspan(row.start_km, row.end_km, color=colors[key], alpha=0.08, lw=0)
    for row in bridges.itertuples(index=False):
        ax.axvspan(row.start_km, row.end_km, color=colors["bridge"], alpha=0.07, lw=0)
        if centers and hasattr(row, "center_km"):
            ax.axvline(row.center_km, color=colors["bridge"], lw=0.45, alpha=0.35, ls="--")


def plot_interval_bar(ax, row, y, height, color, alpha=0.85) -> None:
    ax.broken_barh(
        [(float(row.start_km), float(row.end_km - row.start_km))],
        (y - height / 2.0, height),
        facecolors=color,
        edgecolors="none",
        alpha=alpha,
    )


def make_figure(
    acceleration_csv: Path,
    raw_dynamic: Path,
    ledger_xls: Path,
    out_prefix: Path,
    mileage_range: tuple[float, float] | None = None,
    window_m: float = 512.0,
    step_m: float = 25.0,
    freq_min_hz: float = 0.5,
    freq_max_hz: float = 30.0,
) -> dict[str, Path]:
    configure_style()
    accel = read_acceleration(acceleration_csv)
    x0, x1 = mileage_range or (float(accel["里程"].min()), float(accel["里程"].max()))
    accel = accel[(accel["里程"] >= x0) & (accel["里程"] <= x1)].copy()
    speed = read_speed_profile(raw_dynamic, x0, x1)
    accel = attach_speed(accel, speed)

    sheets = read_xls_sheets_with_excel(ledger_xls)
    curves = clip_intervals(build_curve_table(sheets), x0, x1)
    bridges = clip_intervals(build_bridge_table(sheets), x0, x1).reset_index(drop=True)

    dx_m = float(np.nanmedian(np.diff(accel["里程"].to_numpy(float) * 1000.0)))
    x_grid_m, vertical, speed_grid = uniform_signal(accel, "垂向加速度(g)", dx_m)
    _, lateral, _ = uniform_signal(accel, "横向加速度(g)", dx_m)
    mileage_grid_km = x_grid_m / 1000.0
    freq_range = (freq_min_hz, freq_max_hz)
    vertical_spec = local_spectrum(mileage_grid_km, vertical, speed_grid, dx_m, window_m, step_m, freq_range)
    lateral_spec = local_spectrum(mileage_grid_km, lateral, speed_grid, dx_m, window_m, step_m, freq_range)

    colors = {
        "vertical": "#4C78A8",
        "lateral": "#E45756",
        "curve_left": "#59A14F",
        "curve_right": "#B07AA1",
        "bridge": "#7F7F7F",
        "mode": "#2F4858",
    }

    fig = plt.figure(figsize=(7.3, 7.1), constrained_layout=True)
    gs = fig.add_gridspec(5, 1, height_ratios=[1.45, 0.72, 1.45, 1.45, 1.2])
    ax_acc = fig.add_subplot(gs[0])
    ax_ledger = fig.add_subplot(gs[1], sharex=ax_acc)
    ax_v = fig.add_subplot(gs[2], sharex=ax_acc)
    ax_l = fig.add_subplot(gs[3], sharex=ax_acc)
    ax_peak = fig.add_subplot(gs[4], sharex=ax_acc)

    add_infrastructure_background(ax_acc, bridges, curves, colors, centers=True)
    ax_acc.plot(accel["里程"], accel["垂向加速度(g)"], lw=0.75, color=colors["vertical"], label="Vertical acceleration")
    ax_acc.plot(accel["里程"], accel["横向加速度(g)"], lw=0.75, color=colors["lateral"], label="Lateral acceleration")
    ax_acc.axhline(0, color="#6B7280", lw=0.6)
    ax_acc.set_ylabel("Acceleration (g)")
    ax_acc.grid(True, color="#D6DCE2", lw=0.4, alpha=0.8)
    ax_acc.legend(loc="upper right")
    ax_acc.text(0.01, 0.98, "a", transform=ax_acc.transAxes, ha="left", va="top", fontweight="bold", fontsize=9)

    ax_ledger.set_ylim(-0.65, 1.7)
    ax_ledger.set_yticks([0, 1])
    ax_ledger.set_yticklabels(["Bridges", "Curves"])
    ax_ledger.grid(axis="x", color="#D6DCE2", lw=0.4, alpha=0.8)
    for idx, row in enumerate(bridges.itertuples(index=False), start=1):
        plot_interval_bar(ax_ledger, row, 0, 0.38, colors["bridge"], alpha=0.62)
        mid = (row.start_km + row.end_km) / 2.0
        ax_ledger.text(mid, 0.28, f"Bridge-{idx}", rotation=35, ha="right", va="bottom", fontsize=5.5)
    for row in curves.itertuples(index=False):
        short = direction_short(row.direction)
        color = colors["curve_left"] if short == "L" else colors["curve_right"]
        plot_interval_bar(ax_ledger, row, 1, 0.38, color, alpha=0.85)
        mid = (row.start_km + row.end_km) / 2.0
        ax_ledger.text(mid, 1.32, short, ha="center", va="bottom", fontsize=6.5, fontweight="bold")
        ax_ledger.text(mid, 0.68, f"R={row.radius_m:.0f} m", ha="center", va="top", fontsize=5.8)
    ax_ledger.set_ylabel("Ledger")
    ax_ledger.text(0.01, 0.98, "b", transform=ax_ledger.transAxes, ha="left", va="top", fontweight="bold", fontsize=9)

    def draw_spectrogram(ax, spec: dict[str, np.ndarray], title: str, letter: str):
        z = 10.0 * np.log10(spec["spectra"] + 1e-18)
        vmin, vmax = np.nanpercentile(z, [5, 98])
        mesh = ax.pcolormesh(spec["centers_km"], spec["freq_hz_axis"], z, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
        add_infrastructure_background(ax, bridges, curves, colors, centers=True)
        ax.set_ylabel("Equivalent freq. (Hz)")
        ax.set_title(title, loc="left", fontsize=8, pad=2)
        ax.set_ylim(freq_min_hz, freq_max_hz)
        ax.text(0.01, 0.98, letter, transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=9, color="white")
        return mesh

    mesh_v = draw_spectrogram(ax_v, vertical_spec, "Vertical acceleration local spectrum", "c")
    mesh_l = draw_spectrogram(ax_l, lateral_spec, "Lateral acceleration local spectrum", "d")
    fig.colorbar(mesh_v, ax=ax_v, pad=0.01, aspect=22, label="PSD (dB)")
    fig.colorbar(mesh_l, ax=ax_l, pad=0.01, aspect=22, label="PSD (dB)")

    add_infrastructure_background(ax_peak, bridges, curves, colors, centers=True)
    ax_peak.plot(vertical_spec["centers_km"], vertical_spec["peak_freq_hz"], color=colors["vertical"], lw=1.1, label="Vertical peak")
    ax_peak.plot(lateral_spec["centers_km"], lateral_spec["peak_freq_hz"], color=colors["lateral"], lw=1.1, label="Lateral peak")
    ax_peak.set_ylabel("Dominant peak (Hz)")
    ax_peak.set_xlabel("Mileage (km)")
    ax_peak.set_ylim(freq_min_hz, freq_max_hz)
    ax_peak.grid(True, color="#D6DCE2", lw=0.4, alpha=0.8)
    ax_peak.legend(loc="upper right", ncols=2)
    ax_peak.text(0.01, 0.98, "e", transform=ax_peak.transAxes, ha="left", va="top", fontweight="bold", fontsize=9)

    median_speed = float(np.nanmedian(accel["速度(km/h)"])) if np.isfinite(accel["速度(km/h)"]).any() else float("nan")
    title = "Local spectrum and equivalent modal peaks aligned to bridge/curve ledger"
    if np.isfinite(median_speed):
        title += f" (median speed={median_speed:.1f} km/h)"
    fig.suptitle(title, fontsize=9.4)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": out_prefix.with_suffix(".png"),
        "svg": out_prefix.with_suffix(".svg"),
        "pdf": out_prefix.with_suffix(".pdf"),
        "tiff": out_prefix.with_suffix(".tiff"),
    }
    fig.savefig(outputs["png"], dpi=300, bbox_inches="tight")
    fig.savefig(outputs["svg"], bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    fig.savefig(outputs["tiff"], dpi=600, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(
        {
            "mileage_km": vertical_spec["centers_km"],
            "speed_kmh": vertical_spec["speed_kmh"],
            "vertical_peak_hz": vertical_spec["peak_freq_hz"],
            "vertical_peak_wavelength_m": vertical_spec["peak_wavelength_m"],
            "vertical_peak_power": vertical_spec["peak_power"],
            "lateral_peak_hz": np.interp(vertical_spec["centers_km"], lateral_spec["centers_km"], lateral_spec["peak_freq_hz"]),
            "lateral_peak_wavelength_m": np.interp(vertical_spec["centers_km"], lateral_spec["centers_km"], lateral_spec["peak_wavelength_m"]),
        }
    )
    summary["infrastructure"] = [segment_label(x, bridges, curves) for x in summary["mileage_km"]]
    modal_csv = out_prefix.with_name(out_prefix.name + "_modal_peaks.csv")
    summary.to_csv(modal_csv, index=False, encoding="utf-8-sig")
    outputs["modal_csv"] = modal_csv

    ledger_csv = out_prefix.with_name(out_prefix.name + "_ledger_intervals.csv")
    pd.concat([curves.assign(type="curve"), bridges.assign(type="bridge")], ignore_index=True, sort=False).to_csv(
        ledger_csv, index=False, encoding="utf-8-sig"
    )
    outputs["ledger_csv"] = ledger_csv
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceleration-csv", type=Path, default=DEFAULT_ACCELERATION)
    parser.add_argument("--raw-dynamic", type=Path, default=DEFAULT_RAW_DYNAMIC)
    parser.add_argument("--ledger-xls", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path(r"preprocessing\动检数据\呼局\20210416\处理后\carbody_acceleration_modal_ledger"),
    )
    parser.add_argument("--mileage-start", type=float, default=None)
    parser.add_argument("--mileage-end", type=float, default=None)
    parser.add_argument("--window-m", type=float, default=512.0)
    parser.add_argument("--step-m", type=float, default=25.0)
    parser.add_argument("--freq-min-hz", type=float, default=0.5)
    parser.add_argument("--freq-max-hz", type=float, default=30.0)
    args = parser.parse_args()

    mileage_range = None
    if args.mileage_start is not None or args.mileage_end is not None:
        if args.mileage_start is None or args.mileage_end is None or args.mileage_end <= args.mileage_start:
            raise ValueError("--mileage-start and --mileage-end must be supplied together with end > start.")
        mileage_range = (args.mileage_start, args.mileage_end)

    outputs = make_figure(
        acceleration_csv=args.acceleration_csv,
        raw_dynamic=args.raw_dynamic,
        ledger_xls=args.ledger_xls,
        out_prefix=args.out_prefix,
        mileage_range=mileage_range,
        window_m=args.window_m,
        step_m=args.step_m,
        freq_min_hz=args.freq_min_hz,
        freq_max_hz=args.freq_max_hz,
    )
    for kind, path in outputs.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
