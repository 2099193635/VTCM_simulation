import csv
import json
import os
from dataclasses import dataclass

import numpy as np

try:
    import yaml
except Exception:
    yaml = None


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("on", "true", "1", "yes", "y")


def _norm_text(value, default=""):
    text = str(value if value is not None else default).strip().lower()
    return text or default


def _split_words(value, default):
    if value is None:
        value = default
    if isinstance(value, (list, tuple, set)):
        return {_norm_text(v) for v in value}
    return {_norm_text(v) for v in str(value).replace(";", ",").replace("|", ",").split(",") if str(v).strip()}


@dataclass
class StructureDefectRecord:
    kind: str
    abs_start_m: float
    count: int = 1
    side: str = "both"
    directions: str = "both"
    stiffness_factor: float = 0.0
    damping_factor: float = 0.0
    delta_gap_m: float = 0.0
    label: str = ""


class StructureDefectManager:
    """Map absolute local track defects onto the current structural window."""

    def __init__(self, integration_params, structure_window=None, records=None, enabled=False):
        self.ip = integration_params
        self.structure_window = structure_window or {}
        self.enabled = bool(enabled)
        self.records = list(records or [])
        self.n_nodes = int(self.ip.Nsub) + 1
        self.node_spacing_m = float(self.ip.Lkj)
        self.node_local_m = np.asarray(self.ip.Cord_fastener, dtype=float)

    @classmethod
    def from_config(cls, config_path, integration_params, structure_window=None, enabled=False):
        records = []
        if enabled and config_path:
            data = cls._load_config(config_path)
            records = cls._parse_records(data, integration_params)
        return cls(
            integration_params=integration_params,
            structure_window=structure_window,
            records=records,
            enabled=enabled,
        )

    @staticmethod
    def _load_config(path):
        path = os.path.abspath(path)
        suffix = os.path.splitext(path)[1].lower()
        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        if suffix in (".yaml", ".yml"):
            if yaml is None:
                raise ImportError("PyYAML is required to read structure defect YAML configs.")
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        if suffix == ".csv":
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                return {"defects": list(csv.DictReader(f))}
        raise ValueError(f"Unsupported structure defect config format: {path}")

    @staticmethod
    def _parse_records(data, integration_params):
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("defects", data.get("structure_defects", []))
        else:
            raise ValueError("Structure defect config must be a list or a mapping with a defects list.")

        records = []
        base_abs_m = float(integration_params.S0_mileage)
        for idx, raw in enumerate(items):
            if not isinstance(raw, dict):
                raise ValueError(f"Structure defect #{idx} must be a mapping.")
            kind = _norm_text(raw.get("type", raw.get("kind", "fastener_failure")))
            if kind in ("fastener", "fastener_loss", "fastener_failed", "扣件失效", "扣件松脱"):
                kind = "fastener_failure"
            elif kind in ("void", "sleeper_void", "unsupported_sleeper", "轨枕空吊", "空吊"):
                kind = "sleeper_void"
            elif kind in (
                "ballast_condition",
                "ballast_stiffness",
                "substructure_stiffness",
                "subrail_stiffness",
                "hardened_ballast",
                "loose_ballast",
                "道床板结",
                "道床松散",
                "板结",
                "松散",
            ):
                kind = "ballast_condition"

            abs_start_m = StructureDefectManager._resolve_abs_start_m(raw, base_abs_m)
            count = max(1, int(float(raw.get("count", raw.get("node_count", 1)))))
            side = _norm_text(raw.get("side", "both"), "both")
            directions = _norm_text(raw.get("directions", raw.get("direction", "both")), "both")

            default_factor = 1.0 if kind == "ballast_condition" else 0.0
            stiffness_factor = float(raw.get(
                "stiffness_factor_eta_k",
                raw.get("eta_k", raw.get("stiffness_factor", raw.get("k_factor", default_factor))),
            ))
            damping_factor = float(raw.get(
                "damping_factor_eta_c",
                raw.get("eta_c", raw.get("damping_factor", raw.get("c_factor", default_factor))),
            ))
            delta_gap_m = float(raw.get("delta_gap_m", raw.get("gap_m", 0.0)))
            if "delta_gap_mm" in raw:
                delta_gap_m = float(raw["delta_gap_mm"]) / 1000.0

            records.append(StructureDefectRecord(
                kind=kind,
                abs_start_m=abs_start_m,
                count=count,
                side=side,
                directions=directions,
                stiffness_factor=stiffness_factor,
                damping_factor=damping_factor,
                delta_gap_m=delta_gap_m,
                label=str(raw.get("label", raw.get("name", f"{kind}_{idx + 1}"))),
            ))
        return records

    @staticmethod
    def _resolve_abs_start_m(raw, base_abs_m):
        if raw.get("abs_mileage_m") not in (None, ""):
            return float(raw["abs_mileage_m"])
        if raw.get("abs_mileage_km") not in (None, ""):
            return float(raw["abs_mileage_km"]) * 1000.0
        if raw.get("start_m") not in (None, ""):
            return base_abs_m + float(raw["start_m"])
        if raw.get("relative_m") not in (None, ""):
            return base_abs_m + float(raw["relative_m"])
        raise ValueError("Each structure defect requires start_m, relative_m, abs_mileage_m, or abs_mileage_km.")

    def is_active(self):
        return self.enabled and bool(self.records)

    def window_start_at(self, step_i):
        starts = self.structure_window.get("window_start_abs_m")
        if starts is not None:
            arr = np.asarray(starts, dtype=float)
            return float(arr[min(int(step_i), arr.size - 1)])
        return float(self.ip.S0_mileage)

    def node_abs_positions(self, step_i):
        return self.window_start_at(step_i) + self.node_local_m

    def get_step_defects(self, step_i):
        ones = np.ones(self.n_nodes, dtype=float)
        zeros = np.zeros(self.n_nodes, dtype=float)
        out = {
            "enabled": self.is_active(),
            "fastener_kv_factor_L": ones.copy(),
            "fastener_cv_factor_L": ones.copy(),
            "fastener_kh_factor_L": ones.copy(),
            "fastener_ch_factor_L": ones.copy(),
            "fastener_kv_factor_R": ones.copy(),
            "fastener_cv_factor_R": ones.copy(),
            "fastener_kh_factor_R": ones.copy(),
            "fastener_ch_factor_R": ones.copy(),
            "void_gap_L": zeros.copy(),
            "void_gap_R": zeros.copy(),
            "ballast_kv_factor_L": ones.copy(),
            "ballast_cv_factor_L": ones.copy(),
            "ballast_kv_factor_R": ones.copy(),
            "ballast_cv_factor_R": ones.copy(),
            "fastener_active_L": np.zeros(self.n_nodes, dtype=bool),
            "fastener_active_R": np.zeros(self.n_nodes, dtype=bool),
            "void_active_L": np.zeros(self.n_nodes, dtype=bool),
            "void_active_R": np.zeros(self.n_nodes, dtype=bool),
            "ballast_active_L": np.zeros(self.n_nodes, dtype=bool),
            "ballast_active_R": np.zeros(self.n_nodes, dtype=bool),
        }
        if not self.is_active():
            return out

        window_start = self.window_start_at(step_i)
        for record in self.records:
            start_idx = int(round((record.abs_start_m - window_start) / self.node_spacing_m))
            end_idx = start_idx + int(record.count)
            clip_start = max(0, start_idx)
            clip_end = min(self.n_nodes, end_idx)
            if clip_start >= clip_end:
                continue

            sides = _split_words(record.side, "both")
            left = bool(sides & {"left", "l", "both", "all"})
            right = bool(sides & {"right", "r", "both", "all"})
            directions = _split_words(record.directions, "both")
            vertical = bool(directions & {"vertical", "v", "z", "both", "all"})
            lateral = bool(directions & {"lateral", "h", "y", "horizontal", "both", "all"})
            sl = slice(clip_start, clip_end)

            if record.kind == "fastener_failure":
                if left:
                    out["fastener_active_L"][sl] = True
                    if vertical:
                        out["fastener_kv_factor_L"][sl] *= record.stiffness_factor
                        out["fastener_cv_factor_L"][sl] *= record.damping_factor
                    if lateral:
                        out["fastener_kh_factor_L"][sl] *= record.stiffness_factor
                        out["fastener_ch_factor_L"][sl] *= record.damping_factor
                if right:
                    out["fastener_active_R"][sl] = True
                    if vertical:
                        out["fastener_kv_factor_R"][sl] *= record.stiffness_factor
                        out["fastener_cv_factor_R"][sl] *= record.damping_factor
                    if lateral:
                        out["fastener_kh_factor_R"][sl] *= record.stiffness_factor
                        out["fastener_ch_factor_R"][sl] *= record.damping_factor
            elif record.kind == "sleeper_void":
                if left:
                    out["void_active_L"][sl] = True
                    out["void_gap_L"][sl] = np.maximum(out["void_gap_L"][sl], record.delta_gap_m)
                if right:
                    out["void_active_R"][sl] = True
                    out["void_gap_R"][sl] = np.maximum(out["void_gap_R"][sl], record.delta_gap_m)
            elif record.kind == "ballast_condition":
                if left and vertical:
                    out["ballast_active_L"][sl] = True
                    out["ballast_kv_factor_L"][sl] *= record.stiffness_factor
                    out["ballast_cv_factor_L"][sl] *= record.damping_factor
                if right and vertical:
                    out["ballast_active_R"][sl] = True
                    out["ballast_kv_factor_R"][sl] *= record.stiffness_factor
                    out["ballast_cv_factor_R"][sl] *= record.damping_factor

        return out

    def summary(self):
        rows = []
        for record in self.records:
            rows.append({
                "kind": record.kind,
                "label": record.label,
                "abs_start_m": float(record.abs_start_m),
                "count": int(record.count),
                "side": record.side,
                "directions": record.directions,
                "stiffness_factor": float(record.stiffness_factor),
                "damping_factor": float(record.damping_factor),
                "delta_gap_m": float(record.delta_gap_m),
            })
        return {
            "enabled": bool(self.enabled),
            "record_count": len(self.records),
            "records": rows,
        }

    def ballast_stiffness_field(self, abs_positions_m):
        """Return ballast vertical stiffness factors on arbitrary absolute positions.

        The field is a spatial label for inverse-learning datasets. It is
        independent from the current moving-window state, so the same absolute
        mileage always maps to the same eta_k value.
        """
        positions = np.asarray(abs_positions_m, dtype=float).reshape(-1)
        eta_l = np.ones(positions.size, dtype=np.float32)
        eta_r = np.ones(positions.size, dtype=np.float32)
        mask_l = np.zeros(positions.size, dtype=bool)
        mask_r = np.zeros(positions.size, dtype=bool)
        if not self.is_active():
            return {
                "eta_k_L": eta_l,
                "eta_k_R": eta_r,
                "mask_L": mask_l,
                "mask_R": mask_r,
            }

        for record in self.records:
            if record.kind != "ballast_condition":
                continue
            directions = _split_words(record.directions, "both")
            if not (directions & {"vertical", "v", "z", "both", "all"}):
                continue

            start_m = float(record.abs_start_m)
            end_m = start_m + float(record.count) * self.node_spacing_m
            tol = max(1e-9, 1e-9 * self.node_spacing_m)
            active = (positions >= start_m - tol) & (positions < end_m - tol)
            if not np.any(active):
                continue

            sides = _split_words(record.side, "both")
            left = bool(sides & {"left", "l", "both", "all"})
            right = bool(sides & {"right", "r", "both", "all"})
            factor = np.float32(record.stiffness_factor)
            if left:
                eta_l[active] *= factor
                mask_l[active] = True
            if right:
                eta_r[active] *= factor
                mask_r[active] = True

        return {
            "eta_k_L": eta_l,
            "eta_k_R": eta_r,
            "mask_L": mask_l,
            "mask_R": mask_r,
        }
