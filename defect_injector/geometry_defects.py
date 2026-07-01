from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - handled at runtime when YAML is needed
    yaml = None


SHORT_WAVE_TYPES = {"weld_joint", "corrugation", "rail_spalling"}
PLACEHOLDER_TYPES = {"subgrade_settlement", "frost_heave"}
SUPPORTED_TYPES = SHORT_WAVE_TYPES | PLACEHOLDER_TYPES


def _norm_text(value: Any, default: str = "") -> str:
    text = str(value if value is not None else default).strip().lower()
    aliases = {
        "weld": "weld_joint",
        "weld_bead": "weld_joint",
        "joint": "weld_joint",
        "rail_corrugation": "corrugation",
        "wave_wear": "corrugation",
        "spalling": "rail_spalling",
        "squat": "rail_spalling",
        "block_drop": "rail_spalling",
        "settlement": "subgrade_settlement",
        "subgrade": "subgrade_settlement",
        "frost": "frost_heave",
    }
    return aliases.get(text, text or default)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return float(default)
    return float(value)


def _split_words(value: Any, default: str) -> set[str]:
    if value in (None, ""):
        value = default
    if isinstance(value, (list, tuple, set)):
        return {_norm_text(v) for v in value}
    return {_norm_text(v) for v in str(value).replace(";", ",").replace("|", ",").split(",") if str(v).strip()}


def load_geometry_defect_config(path: str | Path) -> list[dict[str, Any]]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"geometry defect config not found: {config_path}")
    if yaml is None:
        raise ImportError("PyYAML is required to read geometry defect YAML configs.")
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return normalize_geometry_defects(data)


def normalize_geometry_defects(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("defects", data.get("geometry_defects", []))
    else:
        raise ValueError("geometry defect config must be a list or a mapping with a defects list.")

    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(items or [], start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"geometry defect #{index} must be a mapping.")
        kind = _norm_text(raw.get("type", raw.get("kind", "")))
        if kind not in SUPPORTED_TYPES:
            raise ValueError(f"unsupported geometry defect type '{raw.get('type')}'")

        amplitude_mm = _as_float(raw.get("amplitude_mm", raw.get("depth_mm", 0.0)))
        spec = {
            "type": kind,
            "label": str(raw.get("label", raw.get("name", f"{kind}_{index}"))),
            "start_m": _as_float(raw.get("start_m", raw.get("relative_m", 0.0))),
            "length_m": _as_float(raw.get("length_m"), 1.0),
            "amplitude_mm": amplitude_mm,
            "wavelength_m": _as_float(raw.get("wavelength_m"), 0.1),
            "side": str(raw.get("side", "both")),
            "channel": str(raw.get("channel", "vertical")),
            "shape": str(raw.get("shape", "cosine")),
            "implemented": kind in SHORT_WAVE_TYPES,
        }
        if spec["length_m"] <= 0.0:
            raise ValueError(f"geometry defect #{index} length_m must be positive.")
        if spec["wavelength_m"] <= 0.0:
            raise ValueError(f"geometry defect #{index} wavelength_m must be positive.")
        normalized.append(spec)
    return normalized


def _defect_wave(distance_m: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    local = distance_m - float(spec["start_m"])
    length_m = float(spec["length_m"])
    mask = (local >= 0.0) & (local <= length_m)
    wave = np.zeros_like(distance_m, dtype=float)
    if not np.any(mask):
        return wave

    amp_m = float(spec["amplitude_mm"]) / 1000.0
    x = local[mask]
    kind = str(spec["type"])

    if kind == "weld_joint":
        # Local dipped weld: a broad cosine dip plus a smaller short-wave dip.
        wavelength = min(float(spec["wavelength_m"]), length_m)
        broad = -0.5 * amp_m * (1.0 - np.cos(2.0 * np.pi * x / length_m))
        short_center = 0.5 * length_m
        short_local = x - short_center + 0.5 * wavelength
        short_mask = (short_local >= 0.0) & (short_local <= wavelength)
        short = np.zeros_like(x)
        short[short_mask] = -0.25 * amp_m * (1.0 - np.cos(2.0 * np.pi * short_local[short_mask] / wavelength))
        wave[mask] = broad + short
    elif kind == "corrugation":
        taper = 0.5 - 0.5 * np.cos(2.0 * np.pi * x / length_m)
        wave[mask] = amp_m * taper * np.sin(2.0 * np.pi * x / float(spec["wavelength_m"]))
    elif kind == "rail_spalling":
        wave[mask] = -0.5 * amp_m * (1.0 - np.cos(2.0 * np.pi * x / length_m))
    return wave


def build_geometry_defect_profile(distance_m: np.ndarray, specs: list[dict[str, Any]] | None) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    distance = np.asarray(distance_m, dtype=float)
    profile = {
        "VL": np.zeros_like(distance, dtype=float),
        "VR": np.zeros_like(distance, dtype=float),
        "LL": np.zeros_like(distance, dtype=float),
        "LR": np.zeros_like(distance, dtype=float),
    }
    metadata: list[dict[str, Any]] = []
    for spec in specs or []:
        meta = dict(spec)
        if spec["type"] in PLACEHOLDER_TYPES:
            meta["note"] = "placeholder only; no waveform applied"
            metadata.append(meta)
            continue

        wave = _defect_wave(distance, spec)
        sides = _split_words(spec.get("side"), "both")
        channels = _split_words(spec.get("channel"), "vertical")
        left = bool(sides & {"left", "l", "both", "all"})
        right = bool(sides & {"right", "r", "both", "all"})
        vertical = bool(channels & {"vertical", "v", "z", "both", "all"})
        lateral = bool(channels & {"lateral", "h", "y", "horizontal", "both", "all"})
        if vertical and left:
            profile["VL"] += wave
        if vertical and right:
            profile["VR"] += wave
        if lateral and left:
            profile["LL"] += wave
        if lateral and right:
            profile["LR"] += wave
        meta["applied_peak_mm"] = float(np.max(np.abs(wave)) * 1000.0) if wave.size else 0.0
        metadata.append(meta)
    return profile, metadata
