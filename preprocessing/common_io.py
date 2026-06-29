from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def detect_text_encoding(file_path: str | Path, candidates: Optional[Iterable[str]] = None) -> str:
    """检测文本编码，优先中文常见编码。"""
    p = Path(file_path)
    raw = p.read_bytes()[:8192]
    candidates = list(candidates or ("gb18030", "gbk", "utf-8-sig", "utf-8"))
    for enc in candidates:
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "gb18030"


def to_numeric_if_possible(series: pd.Series) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    return series if converted.notna().sum() == 0 else converted


def ensure_mileage_sorted(df: pd.DataFrame, mileage_col: str = "里程") -> pd.DataFrame:
    out = df.copy()
    out[mileage_col] = pd.to_numeric(out[mileage_col], errors="coerce")
    out = out[np.isfinite(out[mileage_col].to_numpy(dtype=float))]
    out = out.sort_values(mileage_col, kind="mergesort").reset_index(drop=True)
    return out


def save_csv(df: pd.DataFrame, out_file: str | Path, encoding: str = "utf-8-sig") -> Path:
    p = Path(out_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding=encoding)
    return p
