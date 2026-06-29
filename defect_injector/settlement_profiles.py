from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


SUPPORTED_SETTLEMENT_TYPES = {'cosine', 'kink'}


def _as_float(value: Any, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ValueError('missing required float value')
        return float(default)
    return float(value)


def _normalize_type(value: Any) -> str:
    text = str(value or '').strip().lower()
    aliases = {
        'cos': 'cosine',
        'half_cosine': 'cosine',
        'cosine_settlement': 'cosine',
        'yu_xian': 'cosine',
        'kink_settlement': 'kink',
        'fold': 'kink',
        'angle': 'kink',
        'zhejiao': 'kink',
    }
    return aliases.get(text, text)


def cosine_settlement(distance_m: np.ndarray, start_m: float, length_m: float, amplitude_m: float) -> np.ndarray:
    """Full-wave cosine settlement basin. Downward settlement is negative."""
    s = np.asarray(distance_m, dtype=float)
    profile = np.zeros_like(s, dtype=float)
    length = float(length_m)
    if length <= 0:
        raise ValueError('cosine settlement length_m must be positive')
    local = s - float(start_m)
    mask = (local >= 0.0) & (local <= length)
    profile[mask] = -0.5 * float(amplitude_m) * (1.0 - np.cos(2.0 * np.pi * local[mask] / length))
    return profile


def kink_settlement(distance_m: np.ndarray, start_m: float, length_m: float, amplitude_m: float) -> np.ndarray:
    """Piecewise linear kink settlement. Downward settlement is negative."""
    s = np.asarray(distance_m, dtype=float)
    profile = np.zeros_like(s, dtype=float)
    length = float(length_m)
    if length <= 0:
        raise ValueError('kink settlement length_m must be positive')
    local = s - float(start_m)
    ramp = (local >= 0.0) & (local <= length)
    after = local > length
    profile[ramp] = -float(amplitude_m) * local[ramp] / length
    profile[after] = -float(amplitude_m)
    return profile


def normalize_settlement_specs(specs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(specs or [], start=1):
        if not isinstance(raw, dict):
            raise ValueError(f'settlement spec #{index} must be a mapping')
        kind = _normalize_type(raw.get('type', ''))
        if kind not in SUPPORTED_SETTLEMENT_TYPES:
            raise ValueError(f"unsupported settlement type '{raw.get('type')}', expected one of {sorted(SUPPORTED_SETTLEMENT_TYPES)}")

        start_m = _as_float(raw.get('start_m', raw.get('position_m')), 0.0)
        length_m = _as_float(raw.get('length_m'), 20.0)
        if 'amplitude_mm' in raw:
            amplitude_m = _as_float(raw.get('amplitude_mm')) / 1000.0
        elif 'amplitude_m' in raw:
            amplitude_m = _as_float(raw.get('amplitude_m'))
        elif kind == 'kink' and 'angle_permille' in raw:
            amplitude_m = _as_float(raw.get('angle_permille')) * length_m / 1000.0
        else:
            raise ValueError(f"settlement spec #{index} needs amplitude_mm/amplitude_m, or angle_permille for kink")

        if amplitude_m < 0:
            raise ValueError(f'settlement spec #{index} amplitude must be non-negative; downward sign is added internally')

        normalized.append({
            'type': kind,
            'start_m': start_m,
            'length_m': length_m,
            'amplitude_m': amplitude_m,
            'amplitude_mm': amplitude_m * 1000.0,
            'angle_permille': raw.get('angle_permille', None),
            'label': str(raw.get('label', f'{kind}_{index}')),
        })
    return normalized


def load_settlement_config(path: str | Path) -> list[dict[str, Any]]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f'settlement config not found: {config_path}')
    with config_path.open('r', encoding='utf-8') as file:
        data = yaml.safe_load(file) or {}
    if isinstance(data, list):
        specs = data
    elif isinstance(data, dict):
        specs = data.get('settlements', [])
    else:
        raise ValueError(f'settlement config root must be a mapping or list: {config_path}')
    if not isinstance(specs, list):
        raise ValueError(f'settlement config field "settlements" must be a list: {config_path}')
    return normalize_settlement_specs(specs)


def build_settlement_profile(distance_m: np.ndarray, specs: list[dict[str, Any]] | None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    normalized = normalize_settlement_specs(specs)
    distance = np.asarray(distance_m, dtype=float)
    profile = np.zeros_like(distance, dtype=float)
    for spec in normalized:
        if spec['type'] == 'cosine':
            profile += cosine_settlement(distance, spec['start_m'], spec['length_m'], spec['amplitude_m'])
        elif spec['type'] == 'kink':
            profile += kink_settlement(distance, spec['start_m'], spec['length_m'], spec['amplitude_m'])
    return profile, normalized
