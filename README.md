# VTCM Simulation

This project contains the vehicle-track coupled dynamics simulation code, research trial configurations, result analysis scripts, and generated datasets.

## Entry Point

`generate_main.py` is the stable CLI entry point. The former `generate_main_copy.py` has been promoted to this file. The old root `generate_main.py` is kept only for reference at `legacy/generate_main_legacy.py`.

```powershell
python generate_main.py
```

Background run helper:

```powershell
.\run_generate_main_background.ps1
```

## Sweeps

Sweep manifests live in `configs/sweeps/` and now point to `generate_main.py`.

Example dry run:

```powershell
python utils\run_param_sweep.py `
  --manifest configs\sweeps\ballast_condition_stiffness_only_100m_scan.yaml `
  --cases case_00_seed20260627_baseline_v215 `
  --dry-run
```

Example sweep with comparison:

```powershell
python run_sweep_and_compare.py `
  --manifest configs\sweeps\structure_defect_fastener_void_scan.yaml `
  --cases baseline sleeper_void_1mm sleeper_void_2mm sleeper_void_5mm `
  --skip-single-analysis
```

## Main Directories

- `physics_modules/`, `solver/`, `track_geometry/`, `infrastructure/`: simulation core.
- `configs/`: standard parameters, trial folders, and sweep manifests.
- `data/`, `Profile_file/`: input data and wheel/rail profiles.
- `results/`: simulation outputs and comparison figures.
- `pipeline/`, `inverse_model/`, `pino_model/`: dataset and inverse-model workflows.
- `tests/`: regression and unit tests.
- `legacy/`: historical entry points and old material kept for traceability.

## Validation

```powershell
python -m unittest tests.test_structure_defects
```

Generated results are expected under `results/<project_or_sweep_name>/...`.
