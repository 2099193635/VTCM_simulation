from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import win32com.client as win32


DEFAULT_ACCELERATION = Path(
    r"preprocessing\动检数据\呼局\20210416\处理后\动检上行20210416-238-363.acceleration.csv"
)
DEFAULT_LEDGER = Path(r"preprocessing\台账\京包客专.xls")


def pick_font() -> str:
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "Arial",
        "DejaVu Sans",
    ]
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
    """Read old .xls workbooks through Excel COM when xlrd is unavailable."""
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
            df = pd.DataFrame(data, columns=header)
            df = df.dropna(how="all").reset_index(drop=True)
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
        return pd.DataFrame(columns=["start_km", "end_km", "name", "length_m"])
    center = numeric(df["中心里程"])
    length = numeric(df["桥全长"])
    out = pd.DataFrame(
        {
            "start_km": center - length / 2000.0,
            "end_km": center + length / 2000.0,
            "name": df["桥名"].map(fix_mojibake),
            "length_m": length,
        }
    )
    return out.dropna(subset=["start_km", "end_km"]).query("end_km > start_km").reset_index(drop=True)


def build_gradient_table(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = filter_up_direction(sheets.get("坡度表", pd.DataFrame()).copy())
    if df.empty:
        return pd.DataFrame(columns=["start_km", "end_km", "gradient_permille"])
    out = pd.DataFrame(
        {
            "start_km": numeric(df["起点里程"]),
            "end_km": numeric(df["终点里程"]),
            "gradient_permille": numeric(df["坡度"]),
        }
    )
    return (
        out.dropna(subset=["start_km", "end_km", "gradient_permille"])
        .query("end_km > start_km")
        .drop_duplicates()
        .reset_index(drop=True)
    )


def clip_intervals(df: pd.DataFrame, x0: float, x1: float) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df[(df["end_km"] >= x0) & (df["start_km"] <= x1)].copy()
    out["start_km"] = out["start_km"].clip(lower=x0, upper=x1)
    out["end_km"] = out["end_km"].clip(lower=x0, upper=x1)
    return out[out["end_km"] > out["start_km"]].reset_index(drop=True)


def direction_short(value: object) -> str:
    text = fix_mojibake(value)
    if text.startswith("左") or text.lower().startswith("l"):
        return "L"
    if text.startswith("右") or text.lower().startswith("r"):
        return "R"
    return text[:1] or "-"


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
    ledger_xls: Path,
    out_prefix: Path,
    mileage_range: tuple[float, float] | None = None,
) -> dict[str, Path]:
    configure_style()

    accel = read_acceleration(acceleration_csv)
    x0, x1 = mileage_range or (float(accel["里程"].min()), float(accel["里程"].max()))
    accel = accel[(accel["里程"] >= x0) & (accel["里程"] <= x1)].copy()

    sheets = read_xls_sheets_with_excel(ledger_xls)
    curves = clip_intervals(build_curve_table(sheets), x0, x1)
    bridges = clip_intervals(build_bridge_table(sheets), x0, x1)
    gradients = clip_intervals(build_gradient_table(sheets), x0, x1)

    fig = plt.figure(figsize=(7.2, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.45, 1.25, 1.25])
    ax_acc = fig.add_subplot(gs[0])
    ax_ledger = fig.add_subplot(gs[1], sharex=ax_acc)
    ax_grad = fig.add_subplot(gs[2], sharex=ax_acc)

    colors = {
        "vertical": "#4C78A8",
        "lateral": "#E45756",
        "curve_left": "#59A14F",
        "curve_right": "#B07AA1",
        "bridge": "#7F7F7F",
        "gradient": "#2F4858",
    }

    for row in bridges.itertuples(index=False):
        ax_acc.axvspan(row.start_km, row.end_km, color=colors["bridge"], alpha=0.08, lw=0)
    for row in curves.itertuples(index=False):
        color = colors["curve_left"] if direction_short(row.direction) == "L" else colors["curve_right"]
        ax_acc.axvspan(row.start_km, row.end_km, color=color, alpha=0.075, lw=0)

    ax_acc.plot(
        accel["里程"],
        accel["垂向加速度(g)"],
        lw=0.85,
        color=colors["vertical"],
        label="Vertical carbody acceleration",
    )
    ax_acc.plot(
        accel["里程"],
        accel["横向加速度(g)"],
        lw=0.85,
        color=colors["lateral"],
        label="Lateral carbody acceleration",
    )
    ax_acc.axhline(0, color="#6B7280", lw=0.65)
    ax_acc.set_ylabel("Acceleration (g)")
    ax_acc.grid(True, axis="both", color="#D6DCE2", lw=0.45, alpha=0.85)
    ax_acc.legend(loc="upper right", ncols=1)
    ax_acc.text(0.01, 0.98, "a", transform=ax_acc.transAxes, ha="left", va="top", fontweight="bold", fontsize=9)

    ax_ledger.set_xlim(x0, x1)
    ax_ledger.set_ylim(-0.65, 1.68)
    ax_ledger.set_yticks([0, 1])
    ax_ledger.set_yticklabels(["Bridges", "Curves"])
    ax_ledger.grid(axis="x", color="#D6DCE2", lw=0.45, alpha=0.85)

    for idx, row in enumerate(bridges.itertuples(index=False), start=1):
        plot_interval_bar(ax_ledger, row, 0, 0.38, colors["bridge"], alpha=0.62)
        mid = (row.start_km + row.end_km) / 2.0
        ax_ledger.text(mid, 0.28, f"Bridge-{idx}", rotation=35, ha="right", va="bottom", fontsize=5.7)

    for row in curves.itertuples(index=False):
        short = direction_short(row.direction)
        color = colors["curve_left"] if short == "L" else colors["curve_right"]
        plot_interval_bar(ax_ledger, row, 1, 0.38, color, alpha=0.85)
        mid = (row.start_km + row.end_km) / 2.0
        ax_ledger.text(mid, 1.32, short, ha="center", va="bottom", fontsize=6.5, fontweight="bold")
        ax_ledger.text(mid, 0.68, f"R={row.radius_m:.0f} m", ha="center", va="top", fontsize=5.8)

    ax_ledger.set_ylabel("Ledger")
    ax_ledger.text(0.01, 0.98, "b", transform=ax_ledger.transAxes, ha="left", va="top", fontweight="bold", fontsize=9)

    if not gradients.empty:
        xs: list[float] = []
        ys: list[float] = []
        for row in gradients.itertuples(index=False):
            xs.extend([row.start_km, row.end_km])
            ys.extend([row.gradient_permille, row.gradient_permille])
        ax_grad.plot(xs, ys, color=colors["gradient"], lw=1.15, drawstyle="steps-post")
        ax_grad.fill_between(xs, ys, 0, step="post", color=colors["gradient"], alpha=0.12)
    ax_grad.axhline(0, color="#6B7280", lw=0.65)
    ax_grad.grid(axis="both", color="#D6DCE2", lw=0.45, alpha=0.85)
    ax_grad.set_ylabel("Gradient (‰)")
    ax_grad.set_xlabel("Mileage (km)")
    ax_grad.text(0.01, 0.98, "c", transform=ax_grad.transAxes, ha="left", va="top", fontweight="bold", fontsize=9)

    fig.suptitle("Mileage-resolved carbody acceleration and infrastructure ledger", fontsize=9.5)

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

    extracted = out_prefix.with_name(out_prefix.name + "_ledger_intervals.csv")
    pd.concat(
        [
            curves.assign(type="curve"),
            bridges.assign(type="bridge"),
            gradients.assign(type="gradient"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(extracted, index=False, encoding="utf-8-sig")
    outputs["ledger_csv"] = extracted
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceleration-csv", type=Path, default=DEFAULT_ACCELERATION)
    parser.add_argument("--ledger-xls", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path(r"preprocessing\动检数据\呼局\20210416\处理后\carbody_acceleration_mileage_with_ledger"),
    )
    parser.add_argument("--mileage-start", type=float, default=None)
    parser.add_argument("--mileage-end", type=float, default=None)
    args = parser.parse_args()

    mileage_range = None
    if args.mileage_start is not None or args.mileage_end is not None:
        if args.mileage_start is None or args.mileage_end is None or args.mileage_end <= args.mileage_start:
            raise ValueError("--mileage-start and --mileage-end must be supplied together with end > start.")
        mileage_range = (args.mileage_start, args.mileage_end)

    outputs = make_figure(
        acceleration_csv=args.acceleration_csv,
        ledger_xls=args.ledger_xls,
        out_prefix=args.out_prefix,
        mileage_range=mileage_range,
    )
    for kind, path in outputs.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
