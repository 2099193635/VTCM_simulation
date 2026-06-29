from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from common_io import save_csv


CURVE_TARGET_COLUMNS = [
    "Start",
    "End",
    "Curve Direction",
    "Curve Radius",
    "Superelevation",
    "Initial Transition Length",
    "Final Transition Length",
    "Curve Length",
]

GRADIENT_TARGET_COLUMNS = [
    "Start",
    "End",
    "Gradient",
    "Slope Length",
]


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "")


def _find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    cols = [str(c).strip() for c in df.columns]
    norm_map = {_normalize_name(c): c for c in cols}
    for a in aliases:
        key = _normalize_name(a)
        if key in norm_map:
            return norm_map[key]
    for c in cols:
        c_norm = _normalize_name(c)
        if any(_normalize_name(a) in c_norm for a in aliases):
            return c
    return None


def _build_curve_df(df: pd.DataFrame) -> pd.DataFrame | None:
    mapping: Dict[str, str | None] = {
        "Start": _find_col(df, ["Start", "ZH", "起点", "起始里程", "曲线起点"]),
        "End": _find_col(df, ["End", "HZ", "终点", "终止里程", "曲线终点"]),
        "Curve Direction": _find_col(df, ["Curve Direction", "Direction", "方向", "左右", "转向"]),
        "Curve Radius": _find_col(df, ["Curve Radius", "Radius", "半径"]),
        "Superelevation": _find_col(df, ["Superelevation", "Cant", "超高"]),
        "Initial Transition Length": _find_col(df, ["Initial Transition Length", "L1", "前缓和", "前缓和曲线长"]),
        "Final Transition Length": _find_col(df, ["Final Transition Length", "L2", "后缓和", "后缓和曲线长"]),
        "Curve Length": _find_col(df, ["Curve Length", "圆曲线长", "圆曲线长度", "Lc"]),
    }

    required = ["Start", "End", "Curve Radius"]
    if any(mapping[k] is None for k in required):
        return None

    out = pd.DataFrame()
    for k in CURVE_TARGET_COLUMNS:
        src = mapping[k]
        if src is None:
            if k == "Curve Direction":
                out[k] = "right"
            else:
                out[k] = 0.0
        else:
            out[k] = df[src]

    out["Curve Direction"] = (
        out["Curve Direction"].astype(str).str.strip().str.lower()
        .replace({"r": "right", "l": "left", "右": "right", "左": "left"})
    )
    for c in ["Start", "End", "Curve Radius", "Superelevation", "Initial Transition Length", "Final Transition Length", "Curve Length"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["Start", "End", "Curve Radius"]).sort_values("Start").reset_index(drop=True)
    return out


def _build_gradient_df(df: pd.DataFrame) -> pd.DataFrame | None:
    mapping: Dict[str, str | None] = {
        "Start": _find_col(df, ["Start", "起点", "起始里程"]),
        "End": _find_col(df, ["End", "终点", "终止里程"]),
        "Gradient": _find_col(df, ["Gradient", "坡度", "坡率"]),
        "Slope Length": _find_col(df, ["Slope Length", "坡长", "长度"]),
    }

    required = ["Start", "End", "Gradient"]
    if any(mapping[k] is None for k in required):
        return None

    out = pd.DataFrame()
    for k in GRADIENT_TARGET_COLUMNS:
        src = mapping[k]
        if src is None:
            out[k] = 0.0
        else:
            out[k] = df[src]

    for c in GRADIENT_TARGET_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["Start", "End", "Gradient"]).sort_values("Start").reset_index(drop=True)
    return out


def read_ledger_xls(file_path: str | Path) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(file_path)
    out: Dict[str, pd.DataFrame] = {}
    for sheet in xls.sheet_names:
        # 台账格式通常第1行为表名，第2行为字段名
        df = xls.parse(sheet_name=sheet, header=1)
        df.columns = [str(c).strip() for c in df.columns]
        out[sheet] = df
    return out


def parse_ledger_to_curve_gradient(file_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sheets = read_ledger_xls(file_path)

    curve_df = None
    gradient_df = None
    for _, sdf in sheets.items():
        if curve_df is None:
            curve_df = _build_curve_df(sdf)
        if gradient_df is None:
            gradient_df = _build_gradient_df(sdf)

    if curve_df is None:
        curve_df = pd.DataFrame(columns=CURVE_TARGET_COLUMNS)
    if gradient_df is None:
        gradient_df = pd.DataFrame(columns=GRADIENT_TARGET_COLUMNS)

    return curve_df, gradient_df


def process_ledger_file(raw_file: str | Path, out_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_df, gradient_df = parse_ledger_to_curve_gradient(raw_file)

    out_dir = Path(out_dir)
    save_csv(curve_df, out_dir / "curve_parameters.csv")
    save_csv(gradient_df, out_dir / "gradient_parameters.csv")

    return curve_df, gradient_df


def clip_ledger_by_mileage_range(
    curve_df: pd.DataFrame,
    gradient_df: pd.DataFrame,
    x0: float,
    x1: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按里程范围（km）裁剪曲线/坡度台账。"""

    c = curve_df.copy()
    g = gradient_df.copy()

    if not c.empty:
        c["Start"] = pd.to_numeric(c["Start"], errors="coerce")
        c["End"] = pd.to_numeric(c["End"], errors="coerce")
        c = c[(c["End"] >= x0) & (c["Start"] <= x1)].copy()
        c["Start"] = c["Start"].clip(lower=x0, upper=x1)
        c["End"] = c["End"].clip(lower=x0, upper=x1)
        c = c[c["End"] > c["Start"]].sort_values("Start").reset_index(drop=True)
        if "Curve Length" in c.columns:
            c["Curve Length"] = (c["End"] - c["Start"]) * 1000.0

    if not g.empty:
        g["Start"] = pd.to_numeric(g["Start"], errors="coerce")
        g["End"] = pd.to_numeric(g["End"], errors="coerce")
        g = g[(g["End"] >= x0) & (g["Start"] <= x1)].copy()
        g["Start"] = g["Start"].clip(lower=x0, upper=x1)
        g["End"] = g["End"].clip(lower=x0, upper=x1)
        g = g[g["End"] > g["Start"]].sort_values("Start").reset_index(drop=True)
        if "Slope Length" in g.columns:
            g["Slope Length"] = (g["End"] - g["Start"]) * 1000.0

    return c, g
