from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SPEEDS_KMH = (160, 200, 215, 250)
SIDES = ("left", "right", "both")
NODE_SPACING_M = 0.6
MAX_DEFECT_END_M = 380.0
LEGACY_SEED_COUNT = 96
STAGE1_SEED_COUNT = 150
FULL_SEED_COUNT = 300

SINGLE_KINDS = ("fastener", "sleeper_void", "ballast", "subgrade")
COUPLED_PAIRS = (
    ("fastener", "ballast"),
    ("fastener", "sleeper_void"),
    ("fastener", "subgrade"),
    ("ballast", "sleeper_void"),
    ("ballast", "subgrade"),
    ("subgrade", "sleeper_void"),
)
RELATIONS = ("overlap", "partial_overlap", "separate")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须为字典: {path}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层必须为字典: {path}")
    return data


def _target_split_counts(seed_count: int) -> dict[str, int]:
    train = int(round(seed_count * 0.70))
    val = int(round(seed_count * 0.15))
    return {"train": train, "val": val, "test": seed_count - train - val}


def _new_split_sequence(
    target_seed_count: int,
    legacy_counts: dict[str, int],
    master_seed: int,
) -> list[str]:
    desired = _target_split_counts(target_seed_count)
    new_counts = {name: desired[name] - legacy_counts[name] for name in desired}
    if min(new_counts.values()) < 0:
        raise ValueError("目标 seed 数过小，无法保留现有 train/val/test 划分")

    # Keep the first 54 assignments stable so stage 1 is a strict prefix of the full run.
    first_size = min(target_seed_count - LEGACY_SEED_COUNT, STAGE1_SEED_COUNT - LEGACY_SEED_COUNT)
    stage1_total = _target_split_counts(STAGE1_SEED_COUNT)
    first_counts = {
        name: min(new_counts[name], stage1_total[name] - legacy_counts[name])
        for name in new_counts
    }
    if sum(first_counts.values()) != first_size:
        first_counts = _target_split_counts(LEGACY_SEED_COUNT + first_size)
        first_counts = {name: first_counts[name] - legacy_counts[name] for name in first_counts}

    split_rng = np.random.default_rng(master_seed + 17)

    def shuffled_block(counts: dict[str, int]) -> list[str]:
        block = [name for name in ("train", "val", "test") for _ in range(counts[name])]
        split_rng.shuffle(block)
        return block

    remainder = {name: new_counts[name] - first_counts[name] for name in new_counts}
    return shuffled_block(first_counts) + shuffled_block(remainder)


def _new_seeds(existing: set[int], count: int, master_seed: int) -> list[int]:
    rng = np.random.default_rng(master_seed)
    values: list[int] = []
    used = set(existing)
    while len(values) < count:
        value = int(rng.integers(10_000_000, 100_000_000))
        if value not in used:
            used.add(value)
            values.append(value)
    return values


def _side(design_index: int, offset: int = 0) -> str:
    return SIDES[(design_index + offset) % len(SIDES)]


def _start_m(rng: np.random.Generator, count: int) -> float:
    max_start = min(300.0, MAX_DEFECT_END_M - count * NODE_SPACING_M)
    low_node = round(150.0 / NODE_SPACING_M)
    high_node = int(np.floor(max_start / NODE_SPACING_M))
    return round(int(rng.integers(low_node, high_node + 1)) * NODE_SPACING_M, 3)


def _stiffness_factor(rng: np.random.Generator, design_index: int) -> float:
    bands = (
        (0.10, 0.30),
        (0.30, 0.70),
        (0.70, 0.95),
        (1.05, 1.30),
        (1.30, 3.00),
        (3.00, 8.00),
    )
    low, high = bands[design_index % len(bands)]
    return round(float(rng.uniform(low, high)), 4)


def _single_defect(kind: str, rng: np.random.Generator, design_index: int) -> dict[str, Any]:
    if kind == "fastener":
        counts = (1, 2, 3, 5, 8, 12, 20)
        factors = (0.0, 0.1, 0.25, 0.5, 0.75, 0.85, 0.90, 0.95, 1.05, 1.10, 1.25)
        count = counts[design_index % len(counts)]
        return {
            "type": "fastener_failure",
            "label": "fastener_eta",
            "start_m": _start_m(rng, count),
            "count": count,
            "side": _side(design_index),
            "directions": "both",
            "stiffness_factor_eta_k": factors[design_index % len(factors)],
            "damping_factor_eta_c": 1.0,
        }
    if kind == "sleeper_void":
        counts = (1, 2, 3, 5, 8, 12, 20)
        gap_bands = ((0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0))
        count = counts[design_index % len(counts)]
        low, high = gap_bands[design_index % len(gap_bands)]
        return {
            "type": "sleeper_void",
            "label": "sleeper_void",
            "start_m": _start_m(rng, count),
            "count": count,
            "side": _side(design_index, 1),
            "directions": "vertical",
            "delta_gap_mm": round(float(rng.uniform(low, high)), 3),
        }

    counts = (3, 5, 8, 12, 20, 40, 80)
    count = counts[design_index % len(counts)]
    factor = _stiffness_factor(rng, design_index)
    is_ballast = kind == "ballast"
    return {
        "type": "ballast_condition" if is_ballast else "subgrade_condition",
        "label": ("ballast_softened" if factor < 1.0 else "ballast_hardened") if is_ballast else ("subgrade_softened" if factor < 1.0 else "subgrade_hardened"),
        "start_m": _start_m(rng, count),
        "count": count,
        "side": _side(design_index, 2 if is_ballast else 0),
        "directions": "vertical",
        "stiffness_factor_eta_k": factor,
        "damping_factor_eta_c": 1.0,
    }


def _coupled_defects(
    pair: tuple[str, str],
    relation: str,
    rng: np.random.Generator,
    design_index: int,
) -> list[dict[str, Any]]:
    first = _single_defect(pair[0], rng, design_index + 3)
    second = _single_defect(pair[1], rng, design_index + 7)
    base_start = round((180.0 + int(rng.integers(0, 101)) * NODE_SPACING_M), 3)
    first["start_m"] = base_start
    if relation == "overlap":
        second_start = base_start
    elif relation == "partial_overlap":
        shift_count = max(1, min(int(first["count"]), int(second["count"])) // 2)
        second_start = base_start + shift_count * NODE_SPACING_M
    else:
        second_start = base_start + max(int(first["count"]), int(second["count"])) * NODE_SPACING_M + 12.0
    max_second_start = MAX_DEFECT_END_M - int(second["count"]) * NODE_SPACING_M
    second["start_m"] = round(min(second_start, max_second_start), 3)
    return [first, second]


def _case(
    case_id: str,
    seed: int,
    speed: int,
    split: str,
    kind: str,
    defects: list[dict[str, Any]],
    pair_group: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_case: dict[str, Any] = {
        "case_id": case_id,
        "note": f"component inverse v2 {kind}; split={split}; pair_group={pair_group}",
        "case_args": {"random_seed": seed, "vx_set": float(speed)},
        "updates": {},
    }
    if defects:
        manifest_case["structure_defects"] = deepcopy(defects)
    catalog_case = {
        "case_id": case_id,
        "random_seed": seed,
        "speed_kmh": speed,
        "split": split,
        "case_kind": kind,
        "pair_group": pair_group,
        "defects": deepcopy(defects),
    }
    return manifest_case, catalog_case


def _append_case(
    manifest_cases: list[dict[str, Any]],
    catalog_cases: list[dict[str, Any]],
    item: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    manifest_case, catalog_case = item
    manifest_cases.append(manifest_case)
    catalog_cases.append(catalog_case)


def generate_v2(
    legacy_manifest: dict[str, Any],
    legacy_catalog: dict[str, Any],
    target_seed_count: int,
    master_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if target_seed_count < STAGE1_SEED_COUNT or target_seed_count > FULL_SEED_COUNT:
        raise ValueError(f"target_seed_count must be between {STAGE1_SEED_COUNT} and {FULL_SEED_COUNT}")

    legacy_splits = {int(seed): str(split) for seed, split in legacy_catalog["seed_splits"].items()}
    if len(legacy_splits) != LEGACY_SEED_COUNT:
        raise ValueError(f"Expected {LEGACY_SEED_COUNT} legacy seeds, got {len(legacy_splits)}")
    legacy_counts = {name: list(legacy_splits.values()).count(name) for name in ("train", "val", "test")}

    manifest_cases = deepcopy(legacy_manifest["cases"])
    catalog_cases = deepcopy(legacy_catalog["cases"])
    seed_splits = {str(seed): split for seed, split in legacy_splits.items()}
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for item in legacy_catalog["cases"]:
        by_seed.setdefault(int(item["random_seed"]), []).append(item)

    ordered_legacy_seeds = list(legacy_splits)
    for index, seed in enumerate(ordered_legacy_seeds):
        rows = by_seed[seed]
        original_speed = int(rows[0]["speed_kmh"])
        second_speed = SPEEDS_KMH[(SPEEDS_KMH.index(original_speed) + 1 + index % 2) % len(SPEEDS_KMH)]
        single_kind = SINGLE_KINDS[index % len(SINGLE_KINDS)]
        source = next(row for row in rows if row["case_kind"] == single_kind)
        pair_group = f"legacy_seed{seed}_cross_speed"
        prefix = f"v2l{index + 1:03d}_seed{seed}_v{second_speed}"
        _append_case(manifest_cases, catalog_cases, _case(
            f"{prefix}_baseline", seed, second_speed, legacy_splits[seed], "baseline", [], pair_group
        ))
        _append_case(manifest_cases, catalog_cases, _case(
            f"{prefix}_{single_kind}", seed, second_speed, legacy_splits[seed], single_kind,
            deepcopy(source.get("defects", [])), pair_group
        ))

    new_count = target_seed_count - LEGACY_SEED_COUNT
    new_seeds = _new_seeds(set(legacy_splits), new_count, master_seed)
    new_splits = _new_split_sequence(target_seed_count, legacy_counts, master_seed)
    for index, (seed, split) in enumerate(zip(new_seeds, new_splits)):
        seed_splits[str(seed)] = split
        rng = np.random.default_rng(master_seed ^ seed)
        primary_speed = SPEEDS_KMH[index % len(SPEEDS_KMH)]
        secondary_speed = SPEEDS_KMH[(index + 1 + index % 2) % len(SPEEDS_KMH)]
        pair_group = f"v2_seed{seed}"
        prefix_primary = f"v2n{index + 1:03d}_seed{seed}_v{primary_speed}"
        prefix_secondary = f"v2n{index + 1:03d}_seed{seed}_v{secondary_speed}"

        singles = {
            kind: _single_defect(kind, rng, index + offset)
            for offset, kind in enumerate(SINGLE_KINDS)
        }
        pair_a = COUPLED_PAIRS[(2 * index) % len(COUPLED_PAIRS)]
        pair_b = COUPLED_PAIRS[(2 * index + 1) % len(COUPLED_PAIRS)]
        relation_a = RELATIONS[index % len(RELATIONS)]
        relation_b = RELATIONS[(index + 1) % len(RELATIONS)]
        coupled_a = _coupled_defects(pair_a, relation_a, rng, index)
        coupled_b = _coupled_defects(pair_b, relation_b, rng, index + 1)
        coupled_a_kind = f"coupled_{pair_a[0]}_{pair_a[1]}_{relation_a}"
        coupled_b_kind = f"coupled_{pair_b[0]}_{pair_b[1]}_{relation_b}"

        primary_definitions = [("baseline", [])]
        primary_definitions.extend((kind, [defect]) for kind, defect in singles.items())
        primary_definitions.extend(((coupled_a_kind, coupled_a), (coupled_b_kind, coupled_b)))
        for kind, defects in primary_definitions:
            _append_case(manifest_cases, catalog_cases, _case(
                f"{prefix_primary}_{kind}", seed, primary_speed, split, kind, defects, pair_group
            ))

        repeated_kind = SINGLE_KINDS[index % len(SINGLE_KINDS)]
        secondary_definitions = (
            ("baseline", []),
            (repeated_kind, [singles[repeated_kind]]),
            (coupled_a_kind, coupled_a),
        )
        for kind, defects in secondary_definitions:
            _append_case(manifest_cases, catalog_cases, _case(
                f"{prefix_secondary}_{kind}", seed, secondary_speed, split, kind, defects, pair_group
            ))

    common = deepcopy(legacy_manifest.get("common", {}))
    common.update({
        "project_name": "component_inverse_v2",
        "note_prefix": "component_inverse_v2",
        "plot_figs": "Off",
        "description": "Compute-aware counterfactual component inverse dataset v2.",
    })
    manifest = {
        "manifest_name": "component_inverse_v2",
        "base_profile_dir": legacy_manifest.get("base_profile_dir", "configs/standard"),
        "output_root": "configs/trials/component_inverse_v2",
        "common": common,
        "cases": manifest_cases,
    }
    catalog = {
        "schema_version": "2.0",
        "master_seed": master_seed,
        "seed_count": target_seed_count,
        "case_count": len(catalog_cases),
        "speeds_kmh": list(SPEEDS_KMH),
        "seed_splits": seed_splits,
        "cases": catalog_cases,
    }
    validate_v2(manifest, catalog, target_seed_count)
    return manifest, catalog


def validate_v2(manifest: dict[str, Any], catalog: dict[str, Any], target_seed_count: int) -> None:
    expected_cases = 576 + LEGACY_SEED_COUNT * 2 + (target_seed_count - LEGACY_SEED_COUNT) * 10
    if len(manifest["cases"]) != expected_cases or len(catalog["cases"]) != expected_cases:
        raise ValueError(f"Expected {expected_cases} cases, got {len(manifest['cases'])}")
    case_ids = [str(case["case_id"]) for case in manifest["cases"]]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case_id values were generated")
    if len(catalog["seed_splits"]) != target_seed_count:
        raise ValueError("Unexpected unique seed count")

    desired_splits = _target_split_counts(target_seed_count)
    actual_splits = {
        name: list(catalog["seed_splits"].values()).count(name)
        for name in ("train", "val", "test")
    }
    if actual_splits != desired_splits:
        raise ValueError(f"Unexpected split counts: {actual_splits}, expected {desired_splits}")

    speeds_by_seed: dict[int, set[int]] = {}
    counts_by_seed: dict[int, int] = {}
    for case in catalog["cases"]:
        seed = int(case["random_seed"])
        speeds_by_seed.setdefault(seed, set()).add(int(case["speed_kmh"]))
        counts_by_seed[seed] = counts_by_seed.get(seed, 0) + 1
        for defect in case.get("defects", []):
            end_m = float(defect["start_m"]) + int(defect["count"]) * NODE_SPACING_M
            if end_m > MAX_DEFECT_END_M + 1e-9:
                raise ValueError(f"Defect exceeds {MAX_DEFECT_END_M} m: {case['case_id']}")
    if any(len(speeds) < 2 for speeds in speeds_by_seed.values()):
        raise ValueError("Every seed must cover at least two speeds")
    legacy_seeds = set(int(seed) for seed in list(catalog["seed_splits"])[:LEGACY_SEED_COUNT])
    for seed, count in counts_by_seed.items():
        expected = 8 if seed in legacy_seeds else 10
        if count != expected:
            raise ValueError(f"Seed {seed} has {count} cases, expected {expected}")


def _write_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    fields = ["case_id", "random_seed", "speed_kmh", "split", "case_kind", "pair_group", "defects_json"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            row = {name: case.get(name, "") for name in fields[:-1]}
            row["defects_json"] = json.dumps(case.get("defects", []), ensure_ascii=False)
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the staged component inverse v2 dataset manifest")
    parser.add_argument("--legacy-manifest", default="configs/sweeps/component_inverse_576.yaml")
    parser.add_argument("--legacy-catalog", default="configs/sweeps/component_inverse_576_catalog.json")
    parser.add_argument("--target-seed-count", type=int, choices=(150, 300), default=150)
    parser.add_argument("--master-seed", type=int, default=20260907)
    parser.add_argument("--output", default="configs/sweeps/component_inverse_v2_stage1.yaml")
    parser.add_argument("--catalog-json", default="configs/sweeps/component_inverse_v2_stage1_catalog.json")
    parser.add_argument("--catalog-csv", default="configs/sweeps/component_inverse_v2_stage1_catalog.csv")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, catalog = generate_v2(
        _load_yaml(Path(args.legacy_manifest)),
        _load_json(Path(args.legacy_catalog)),
        args.target_seed_count,
        args.master_seed,
    )
    if args.check_only:
        print(f"Validated {catalog['seed_count']} seeds and {catalog['case_count']} cases")
        return
    output = Path(args.output)
    catalog_json = Path(args.catalog_json)
    catalog_csv = Path(args.catalog_csv)
    for path in (output, catalog_json, catalog_csv):
        path.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, allow_unicode=True, sort_keys=False)
    catalog_json.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(catalog_csv, catalog["cases"])
    print(f"Saved manifest: {output}")
    print(f"Saved catalog JSON: {catalog_json}")
    print(f"Saved catalog CSV: {catalog_csv}")
    print(f"Generated {catalog['seed_count']} seeds and {catalog['case_count']} cases")


if __name__ == "__main__":
    main()
