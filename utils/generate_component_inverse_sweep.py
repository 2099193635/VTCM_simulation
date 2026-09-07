from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SPEEDS_KMH = (160, 200, 215, 250)
SIDES = ("left", "right", "both")
FASTENER_FACTORS = (0.0, 0.1, 0.25, 0.5, 0.75)
MAX_DEFECT_END_M = 380.0
NODE_SPACING_M = 0.6


def _case_split(speed_index: int, seed_index_within_speed: int) -> str:
    if seed_index_within_speed < 17:
        return "train"
    val_count = 4 if speed_index < 2 else 3
    if seed_index_within_speed < 17 + val_count:
        return "val"
    return "test"


def _random_side(rng: np.random.Generator) -> str:
    return str(rng.choice(SIDES, p=(0.25, 0.25, 0.50)))


def _start_and_count(
    rng: np.random.Generator,
    min_count: int,
    max_count: int,
) -> tuple[float, int]:
    start_node = int(rng.integers(round(150.0 / NODE_SPACING_M), round(300.0 / NODE_SPACING_M) + 1))
    start_m = start_node * NODE_SPACING_M
    allowed = int(np.floor((MAX_DEFECT_END_M - start_m) / NODE_SPACING_M))
    upper = max(min_count, min(max_count, allowed))
    count = int(rng.integers(min_count, upper + 1))
    return round(start_m, 3), count


def _fastener_defect(rng: np.random.Generator) -> dict[str, Any]:
    start_m, count = _start_and_count(rng, 1, 10)
    return {
        "type": "fastener_failure",
        "label": "fastener_eta",
        "start_m": start_m,
        "count": count,
        "side": _random_side(rng),
        "directions": "both",
        "stiffness_factor_eta_k": float(rng.choice(FASTENER_FACTORS)),
        "damping_factor_eta_c": 1.0,
    }


def _void_defect(rng: np.random.Generator) -> dict[str, Any]:
    start_m, count = _start_and_count(rng, 1, 10)
    return {
        "type": "sleeper_void",
        "label": "sleeper_void",
        "start_m": start_m,
        "count": count,
        "side": _random_side(rng),
        "directions": "vertical",
        "delta_gap_mm": round(float(rng.uniform(0.5, 3.0)), 2),
    }


def _ballast_defect(rng: np.random.Generator) -> dict[str, Any]:
    start_m, count = _start_and_count(rng, 10, 167)
    if bool(rng.integers(0, 2)):
        factor = float(rng.uniform(0.1, 0.8))
        label = "ballast_loose"
    else:
        factor = float(rng.uniform(1.5, 8.0))
        label = "ballast_hardened"
    return {
        "type": "ballast_condition",
        "label": label,
        "start_m": start_m,
        "count": count,
        "side": _random_side(rng),
        "directions": "vertical",
        "stiffness_factor_eta_k": round(factor, 4),
        "damping_factor_eta_c": 1.0,
    }


def _subgrade_defect(rng: np.random.Generator) -> dict[str, Any]:
    start_m, count = _start_and_count(rng, 20, 167)
    return {
        "type": "subgrade_condition",
        "label": "subgrade_weakened",
        "start_m": start_m,
        "count": count,
        "side": _random_side(rng),
        "directions": "vertical",
        "stiffness_factor_eta_k": round(float(rng.uniform(0.1, 0.8)), 4),
        "damping_factor_eta_c": 1.0,
    }


DEFECT_BUILDERS = {
    "fastener": _fastener_defect,
    "sleeper_void": _void_defect,
    "ballast": _ballast_defect,
    "subgrade": _subgrade_defect,
}


def _case_entry(
    case_id: str,
    seed: int,
    speed_kmh: int,
    note: str,
    defects: list[dict[str, Any]],
) -> dict[str, Any]:
    case: dict[str, Any] = {
        "case_id": case_id,
        "note": note,
        "case_args": {
            "random_seed": int(seed),
            "vx_set": float(speed_kmh),
        },
        "updates": {},
    }
    if defects:
        case["structure_defects"] = defects
    return case


def generate_manifest(seed_count: int, master_seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if seed_count <= 0 or seed_count % len(SPEEDS_KMH) != 0:
        raise ValueError(f"seed_count must be a positive multiple of {len(SPEEDS_KMH)}")
    seeds_per_speed = seed_count // len(SPEEDS_KMH)
    if seed_count == 96 and seeds_per_speed != 24:
        raise AssertionError("The 576-case design requires 24 seeds per speed.")

    rng = np.random.default_rng(master_seed)
    seeds = rng.choice(np.arange(10_000_000, 99_999_999, dtype=np.int64), size=seed_count, replace=False)
    cases: list[dict[str, Any]] = []
    catalog_cases: list[dict[str, Any]] = []
    seed_splits: dict[str, str] = {}

    seed_global_index = 0
    for speed_index, speed in enumerate(SPEEDS_KMH):
        for local_index in range(seeds_per_speed):
            seed = int(seeds[seed_global_index])
            seed_global_index += 1
            split = _case_split(speed_index, local_index) if seed_count == 96 else (
                "train" if local_index < max(1, round(seeds_per_speed * 0.71))
                else ("val" if local_index < max(2, round(seeds_per_speed * 0.86)) else "test")
            )
            seed_splits[str(seed)] = split
            prefix = f"s{seed_global_index:03d}_seed{seed}_v{speed}"
            definitions = [
                ("baseline", []),
                ("fastener", [_fastener_defect(rng)]),
                ("sleeper_void", [_void_defect(rng)]),
                ("ballast", [_ballast_defect(rng)]),
                ("subgrade", [_subgrade_defect(rng)]),
            ]
            coupled_parts = list(rng.choice(list(DEFECT_BUILDERS), size=2, replace=False))
            coupled = [DEFECT_BUILDERS[part](rng) for part in coupled_parts]
            definitions.append(("coupled_" + "_".join(coupled_parts), coupled))

            for case_kind, defects in definitions:
                case_id = f"{prefix}_{case_kind}"
                note = f"component inverse {case_kind}; split={split}"
                cases.append(_case_entry(case_id, seed, speed, note, defects))
                catalog_cases.append(
                    {
                        "case_id": case_id,
                        "random_seed": seed,
                        "speed_kmh": speed,
                        "split": split,
                        "case_kind": case_kind,
                        "defects": defects,
                    }
                )

    manifest = {
        "manifest_name": "component_inverse_576",
        "base_profile_dir": "configs/standard",
        "output_root": "configs/trials/component_inverse_576",
        "common": {
            "main_script": "generate_main.py",
            "project_name": "component_inverse_576",
            "vehicle_type": "高速客车",
            "irr_type": "随机不平顺",
            "defect_switch": "off",
            "tz": 10.0,
            "save_dof_mode": "vehicle",
            "save_spy_level": "core",
            "save_stride": 10,
            "plot_figs": "Off",
            "note_prefix": "component_inverse",
            "description": "Counterfactual spectrum-irregularity and component-stiffness inverse dataset.",
        },
        "cases": cases,
    }
    catalog = {
        "schema_version": "1.0",
        "master_seed": master_seed,
        "seed_count": seed_count,
        "case_count": len(cases),
        "speeds_kmh": list(SPEEDS_KMH),
        "seed_splits": seed_splits,
        "cases": catalog_cases,
    }
    return manifest, catalog


def _write_catalog_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "random_seed", "speed_kmh", "split", "case_kind", "defects_json"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "random_seed": case["random_seed"],
                    "speed_kmh": case["speed_kmh"],
                    "split": case["split"],
                    "case_kind": case["case_kind"],
                    "defects_json": json.dumps(case["defects"], ensure_ascii=False),
                }
            )


def validate_design(manifest: dict[str, Any], catalog: dict[str, Any]) -> None:
    cases = manifest["cases"]
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case_id values were generated.")
    if catalog["seed_count"] == 96 and len(cases) != 576:
        raise ValueError(f"Expected 576 cases, got {len(cases)}")
    speed_counts = {speed: 0 for speed in SPEEDS_KMH}
    seed_case_counts: dict[int, int] = {}
    for case in catalog["cases"]:
        speed_counts[case["speed_kmh"]] += 1
        seed_case_counts[case["random_seed"]] = seed_case_counts.get(case["random_seed"], 0) + 1
        for defect in case["defects"]:
            end_m = float(defect["start_m"]) + int(defect["count"]) * NODE_SPACING_M
            if end_m > MAX_DEFECT_END_M + 1e-9:
                raise ValueError(f"Defect exceeds {MAX_DEFECT_END_M} m: {case['case_id']}")
    if any(count != 6 for count in seed_case_counts.values()):
        raise ValueError("Each spectrum seed must have exactly six counterfactual cases.")
    if catalog["seed_count"] == 96 and any(count != 144 for count in speed_counts.values()):
        raise ValueError(f"Expected 144 cases per speed, got {speed_counts}")
    split_seed_counts = {"train": 0, "val": 0, "test": 0}
    for split in catalog["seed_splits"].values():
        split_seed_counts[split] += 1
    if catalog["seed_count"] == 96 and split_seed_counts != {"train": 68, "val": 14, "test": 14}:
        raise ValueError(f"Unexpected seed split counts: {split_seed_counts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the component inverse sweep manifest and case catalog.")
    parser.add_argument("--output", default="configs/sweeps/component_inverse_576.yaml")
    parser.add_argument("--catalog-json", default="configs/sweeps/component_inverse_576_catalog.json")
    parser.add_argument("--catalog-csv", default="configs/sweeps/component_inverse_576_catalog.csv")
    parser.add_argument("--seed-count", type=int, default=96)
    parser.add_argument("--master-seed", type=int, default=20260715)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, catalog = generate_manifest(args.seed_count, args.master_seed)
    validate_design(manifest, catalog)
    if args.check_only:
        print(f"Validated {catalog['seed_count']} seeds and {catalog['case_count']} cases.")
        return

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, allow_unicode=True, sort_keys=False)
    catalog_json = Path(args.catalog_json)
    catalog_json.parent.mkdir(parents=True, exist_ok=True)
    catalog_json.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_catalog_csv(Path(args.catalog_csv), catalog["cases"])
    print(f"Saved manifest: {output}")
    print(f"Saved catalog JSON: {catalog_json}")
    print(f"Saved catalog CSV: {args.catalog_csv}")
    print(f"Generated {catalog['seed_count']} seeds and {catalog['case_count']} cases.")


if __name__ == "__main__":
    main()