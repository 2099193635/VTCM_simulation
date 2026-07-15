import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.joint_inverse_baseline import (
    JointInverseRidge,
    aggregate_window_predictions,
    load_joint_inverse_dataset,
    predict_dataset,
    save_predictions,
    train_from_dataset,
)


class JointInverseBaselineTests(unittest.TestCase):
    def make_dataset(self, n: int = 24, w: int = 16) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(1234)
        geom = rng.normal(0.0, 0.002, size=(n, w, 4)).astype(np.float32)
        eta = np.ones((n, w, 2), dtype=np.float32)
        for i in range(n):
            if i % 3 == 0:
                eta[i, 5:11, :] = 0.3
            elif i % 3 == 1:
                eta[i, 3:8, :] = 2.0
        log_eta = np.log(eta).astype(np.float32)

        obs = np.zeros((n, w, 6), dtype=np.float32)
        obs[:, :, 0] = 3.0 * geom[:, :, 0] + 0.2 * log_eta[:, :, 0]
        obs[:, :, 1] = -2.0 * geom[:, :, 1] + 0.1 * log_eta[:, :, 1]
        obs[:, :, 2] = geom[:, :, 2] - geom[:, :, 3]
        obs[:, :, 3] = log_eta[:, :, 0] + log_eta[:, :, 1]
        obs[:, :, 4] = geom[:, :, 0] + geom[:, :, 1]
        obs[:, :, 5] = 0.5 * log_eta[:, :, 0]

        context = np.column_stack(
            [
                np.full(n, 215.0, dtype=np.float32),
                np.arange(n, dtype=np.float32),
                np.full(n, 0.5, dtype=np.float32),
                np.arange(n, dtype=np.float32),
            ]
        )
        mask = np.ones((n, w, 3), dtype=np.float32)
        meta_json = np.asarray(
            [
                (
                    '{"run_name":"run_%02d","window_start_rel_m":%.1f,'
                    '"window_start_abs_m":%.1f}'
                )
                % (i // 2, float((i % 2) * 4), 1000.0 + float((i % 2) * 4))
                for i in range(n)
            ]
        )
        return {
            "obs": obs,
            "geom_irregularity": geom,
            "stiffness_irregularity": eta,
            "stiffness_log_eta_k": log_eta,
            "mask": mask,
            "context": context,
            "meta_json": meta_json,
            "ds_m": np.asarray(0.5, dtype=np.float32),
        }

    def test_ridge_trains_predicts_and_reloads(self):
        dataset = self.make_dataset(n=160, w=8)
        model, report = train_from_dataset(dataset, alpha=1e-6, test_fraction=0.25, seed=5)

        self.assertEqual(report["train_samples"], 120)
        self.assertEqual(report["test_samples"], 40)
        self.assertLess(report["train_metrics"]["stiffness_eta_rmse"], 0.08)

        pred = model.predict(dataset["obs"], dataset["context"])
        self.assertEqual(pred["geom_irregularity"].shape, dataset["geom_irregularity"].shape)
        self.assertEqual(pred["stiffness_irregularity"].shape, dataset["stiffness_irregularity"].shape)

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.npz"
            model.save(model_path)
            loaded = JointInverseRidge.load(model_path)
            pred_loaded = loaded.predict(dataset["obs"], dataset["context"])
            np.testing.assert_allclose(pred["stiffness_irregularity"], pred_loaded["stiffness_irregularity"], rtol=1e-6, atol=1e-6)

    def test_prediction_save_load_and_profile_aggregation(self):
        dataset = self.make_dataset(n=6, w=8)
        model, _ = train_from_dataset(dataset, alpha=1e-4, test_fraction=0.0)
        pred = predict_dataset(model, dataset)
        profiles = aggregate_window_predictions(pred, dataset["meta_json"], float(dataset["ds_m"]))

        self.assertTrue(profiles)
        first = next(iter(profiles.values()))
        self.assertIn("rel_m", first)
        self.assertIn("geom_irregularity", first)
        self.assertIn("stiffness_irregularity", first)
        self.assertEqual(first["geom_irregularity"].shape[1], 4)
        self.assertEqual(first["stiffness_irregularity"].shape[1], 2)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "predictions.npz"
            save_predictions(out, pred, profiles)
            loaded = load_joint_inverse_dataset(_write_dataset_npz(Path(tmp) / "dataset.npz", dataset))
            self.assertEqual(loaded["obs"].shape, dataset["obs"].shape)
            self.assertTrue(out.exists())
            with np.load(out, allow_pickle=True) as saved:
                self.assertIn("stiffness_defect_spans_json", saved.files)


def _write_dataset_npz(path: Path, dataset: dict[str, np.ndarray]) -> Path:
    np.savez_compressed(path, **dataset)
    return path


if __name__ == "__main__":
    unittest.main()
