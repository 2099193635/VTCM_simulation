import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from pipeline.build_component_inverse_hdf5_dataset import (
    build_windows,
    support_eta_from_components,
    write_hdf5,
)
from pipeline.dataset_generator import JointInverseHDF5Dataset
from utils.generate_component_inverse_sweep import generate_manifest, validate_design
from utils.generate_component_inverse_v2 import generate_v2, validate_v2


class ComponentInverseDatasetTests(unittest.TestCase):
    def _write_run(self, root: Path, run_id: str, seed: int, defect: bool) -> None:
        files = root / run_id / "files"
        files.mkdir(parents=True)
        n = 96
        rel = np.arange(n, dtype=np.float64) * 0.25
        abs_s = 1000.0 + rel
        accel = np.zeros((n, 35), dtype=np.float32)
        accel[:, 1] = np.sin(rel).astype(np.float32)
        geom = (0.001 * np.sin(rel / 3.0)).astype(np.float32)
        fast_l = np.ones(n, dtype=np.float32)
        ballast_l = np.ones(n, dtype=np.float32)
        subgrade_l = np.ones(n, dtype=np.float32)
        gap_l = np.zeros(n, dtype=np.float32)
        void_l = np.zeros(n, dtype=np.int8)
        if defect:
            fast_l[24:40] = 0.25
            gap_l[48:56] = 0.002
            void_l[48:56] = 1
            ballast_l[56:72] = 0.4
            subgrade_l[72:88] = 0.5

        np.savez(
            files / "simulation_result.npz",
            A=accel,
            dt=np.asarray(0.01),
            Track_rel_mileage_m=rel,
            Track_abs_mileage_m=abs_s,
            Irre_bz_L_ref=geom,
            Irre_bz_R_ref=geom,
            Irre_by_L_ref=np.zeros(n, dtype=np.float32),
            Irre_by_R_ref=np.zeros(n, dtype=np.float32),
            Fastener_eta_k_L_ref=fast_l,
            Fastener_eta_k_R_ref=np.ones(n, dtype=np.float32),
            Ballast_eta_k_L_ref=ballast_l,
            Ballast_eta_k_R_ref=np.ones(n, dtype=np.float32),
            Subgrade_eta_k_L_ref=subgrade_l,
            Subgrade_eta_k_R_ref=np.ones(n, dtype=np.float32),
            Sleeper_void_gap_L_m_ref=gap_l,
            Sleeper_void_gap_R_m_ref=np.zeros(n, dtype=np.float32),
            Sleeper_void_active_L_ref=void_l,
            Sleeper_void_active_R_ref=np.zeros(n, dtype=np.int8),
        )
        (files / "argparse_params.json").write_text(
            json.dumps({"vx_set": 215.0, "random_seed": seed}),
            encoding="utf-8",
        )

    def test_support_eta_normal_and_open_void(self):
        eta = np.ones((4, 3, 2), dtype=np.float32)
        void = np.zeros((4, 2), dtype=bool)
        normal = support_eta_from_components(eta, void)
        np.testing.assert_allclose(normal, 1.0, rtol=1e-6)
        void[1, 0] = True
        weakened = support_eta_from_components(eta, void)
        self.assertEqual(float(weakened[1, 0]), 0.0)
        self.assertAlmostEqual(float(weakened[1, 1]), 1.0, places=6)

    def test_build_and_read_hdf5(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            self._write_run(root, "run_train", 101, defect=True)
            self._write_run(root, "run_val", 102, defect=False)
            self._write_run(root, "run_test", 103, defect=False)
            catalog = Path(tmp) / "catalog.json"
            catalog.write_text(
                json.dumps({"seed_splits": {"101": "train", "102": "val", "103": "test"}}),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                result_roots=[str(root)],
                output=str(Path(tmp) / "dataset.h5"),
                case_catalog=str(catalog),
                ds_m=0.25,
                window_m=8.0,
                stride_m=4.0,
                min_valid_fraction=0.0,
                min_eta=0.02,
                max_eta=50.0,
                split_seed=7,
            )
            arrays = build_windows(args)
            self.assertEqual(arrays["obs"].shape[1:], (32, 10))
            self.assertEqual(arrays["geometry_abs"].shape[1:], (32, 4))
            self.assertEqual(arrays["component_eta"].shape[1:], (32, 3, 2))
            self.assertTrue(np.any(arrays["void_active"] > 0.5))
            self.assertTrue(np.any(arrays["support_eta"] == 0.0))

            write_hdf5(args.output, arrays, args)
            with h5py.File(args.output, "r") as h5:
                self.assertEqual(set(h5.keys()), {"train", "val", "test"})
                self.assertEqual(h5.attrs["schema_version"], "1.0")
                self.assertGreater(h5["train"]["obs"].shape[0], 0)

            dataset = JointInverseHDF5Dataset(args.output, split="train", obs_channels=[0, 1, 3])
            try:
                sample = dataset[0]
                self.assertEqual(sample["obs_in"].shape, (32, 3))
                self.assertEqual(sample["component_eta_tgt"].shape, (32, 3, 2))
            finally:
                dataset.close()

    def test_576_case_manifest_design(self):
        manifest, catalog = generate_manifest(seed_count=96, master_seed=20260715)
        validate_design(manifest, catalog)
        self.assertEqual(len(manifest["cases"]), 576)
        split_counts = {"train": 0, "val": 0, "test": 0}
        for split in catalog["seed_splits"].values():
            split_counts[split] += 1
        self.assertEqual(split_counts, {"train": 68, "val": 14, "test": 14})
        self.assertFalse(any("geometry_defects" in case for case in manifest["cases"]))

    def test_v2_staged_manifest_design(self):
        root = Path(__file__).resolve().parents[1]
        with (root / "configs/sweeps/component_inverse_576.yaml").open("r", encoding="utf-8") as handle:
            import yaml

            legacy_manifest = yaml.safe_load(handle)
        legacy_catalog = json.loads(
            (root / "configs/sweeps/component_inverse_576_catalog.json").read_text(encoding="utf-8")
        )

        stage1_manifest, stage1_catalog = generate_v2(
            legacy_manifest, legacy_catalog, target_seed_count=150, master_seed=20260907
        )
        full_manifest, full_catalog = generate_v2(
            legacy_manifest, legacy_catalog, target_seed_count=300, master_seed=20260907
        )
        validate_v2(stage1_manifest, stage1_catalog, 150)
        validate_v2(full_manifest, full_catalog, 300)
        self.assertEqual(len(stage1_manifest["cases"]), 1308)
        self.assertEqual(len(full_manifest["cases"]), 2808)
        self.assertEqual(
            [case["case_id"] for case in stage1_manifest["cases"]],
            [case["case_id"] for case in full_manifest["cases"][:1308]],
        )
        split_counts = {
            split: list(full_catalog["seed_splits"].values()).count(split)
            for split in ("train", "val", "test")
        }
        self.assertEqual(split_counts, {"train": 210, "val": 45, "test": 45})


if __name__ == "__main__":
    unittest.main()
