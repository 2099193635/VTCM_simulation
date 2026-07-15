from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TARGET_GEOM_KEY = "geom_irregularity"
TARGET_STIFFNESS_ETA_KEY = "stiffness_irregularity"
TARGET_STIFFNESS_LOG_KEY = "stiffness_log_eta_k"


@dataclass
class JointInverseRidge:
    """Closed-form ridge baseline for joint geometry/stiffness inversion.

    The model maps one response window to the co-located geometry irregularity
    and stiffness profile window. It is intentionally dependency-light so the
    inverse pipeline can be tested before a neural sequence model is introduced.
    """

    alpha: float = 1.0
    use_context: bool = True
    stiffness_target: str = "log_eta"
    x_mean: np.ndarray | None = None
    x_scale: np.ndarray | None = None
    y_mean: np.ndarray | None = None
    y_scale: np.ndarray | None = None
    coef: np.ndarray | None = None
    intercept: np.ndarray | None = None
    window_points: int | None = None
    obs_channels: int | None = None
    geom_channels: int | None = None
    stiffness_channels: int | None = None
    context_channels: int = 0

    def fit(
        self,
        obs: np.ndarray,
        geom: np.ndarray,
        stiffness: np.ndarray,
        context: np.ndarray | None = None,
        sample_mask: np.ndarray | None = None,
    ) -> "JointInverseRidge":
        obs = _as_3d(obs, "obs")
        geom = _as_3d(geom, "geom")
        stiffness = _as_3d(stiffness, "stiffness")
        _validate_compatible(obs, geom, stiffness)

        keep = _sample_keep_mask(obs, geom, stiffness, sample_mask)
        if np.count_nonzero(keep) < 1:
            raise ValueError("No valid samples remain for training.")

        self.window_points = int(obs.shape[1])
        self.obs_channels = int(obs.shape[2])
        self.geom_channels = int(geom.shape[2])
        self.stiffness_channels = int(stiffness.shape[2])

        x = _flatten_obs_context(obs[keep], context[keep] if self.use_context and context is not None else None)
        # Keep target blocks explicit: [all geometry samples, all stiffness samples].
        # This matches predict(), which splits the output vector by target family.
        y = np.concatenate(
            [
                geom[keep].reshape(np.count_nonzero(keep), -1),
                stiffness[keep].reshape(np.count_nonzero(keep), -1),
            ],
            axis=1,
        )

        x_norm, self.x_mean, self.x_scale = _standardize_fit(x)
        y_norm, self.y_mean, self.y_scale = _standardize_fit(y)
        self.context_channels = 0 if context is None or not self.use_context else int(context.shape[1])

        xtx = x_norm.T @ x_norm
        reg = float(self.alpha) * np.eye(xtx.shape[0], dtype=np.float64)
        self.coef = np.linalg.solve(xtx + reg, x_norm.T @ y_norm).astype(np.float32)
        self.intercept = np.zeros(y_norm.shape[1], dtype=np.float32)
        return self

    def predict(self, obs: np.ndarray, context: np.ndarray | None = None) -> dict[str, np.ndarray]:
        if self.coef is None or self.x_mean is None or self.x_scale is None:
            raise RuntimeError("Model is not fitted.")
        obs = _as_3d(obs, "obs")
        if self.window_points is not None and obs.shape[1] != self.window_points:
            raise ValueError(f"Expected {self.window_points} window points, got {obs.shape[1]}.")
        if self.obs_channels is not None and obs.shape[2] != self.obs_channels:
            raise ValueError(f"Expected {self.obs_channels} obs channels, got {obs.shape[2]}.")

        x = _flatten_obs_context(obs, context if self.use_context else None)
        x_norm = (x - self.x_mean) / self.x_scale
        y_norm = x_norm @ self.coef
        if self.intercept is not None:
            y_norm = y_norm + self.intercept
        y = y_norm * self.y_scale + self.y_mean

        geom_count = int(self.window_points or obs.shape[1]) * int(self.geom_channels or 0)
        geom = y[:, :geom_count].reshape(obs.shape[0], int(self.window_points or obs.shape[1]), int(self.geom_channels or 0))
        stiffness_raw = y[:, geom_count:].reshape(
            obs.shape[0],
            int(self.window_points or obs.shape[1]),
            int(self.stiffness_channels or 0),
        )
        if self.stiffness_target == "eta":
            stiffness_eta = np.clip(stiffness_raw, 1e-4, None)
            stiffness_log_eta = np.log(stiffness_eta)
        else:
            stiffness_log_eta = stiffness_raw
            stiffness_eta = np.exp(stiffness_log_eta)
        return {
            TARGET_GEOM_KEY: geom.astype(np.float32),
            TARGET_STIFFNESS_LOG_KEY: stiffness_log_eta.astype(np.float32),
            TARGET_STIFFNESS_ETA_KEY: stiffness_eta.astype(np.float32),
        }

    def evaluate(
        self,
        obs: np.ndarray,
        geom: np.ndarray,
        stiffness_eta: np.ndarray,
        context: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        defect_eta_tol: float = 0.05,
    ) -> dict[str, float]:
        pred = self.predict(obs, context=context)
        valid = _valid_point_mask(mask, obs.shape[:2])
        geom_true = _as_3d(geom, "geom")
        eta_true = _as_3d(stiffness_eta, "stiffness_eta")
        geom_pred = pred[TARGET_GEOM_KEY]
        eta_pred = pred[TARGET_STIFFNESS_ETA_KEY]

        metrics = {
            "geom_rmse": _masked_rmse(geom_pred, geom_true, valid),
            "geom_mae": _masked_mae(geom_pred, geom_true, valid),
            "stiffness_eta_rmse": _masked_rmse(eta_pred, eta_true, valid),
            "stiffness_eta_mae": _masked_mae(eta_pred, eta_true, valid),
        }
        true_defect = np.any(np.abs(eta_true - 1.0) > defect_eta_tol, axis=2)
        pred_defect = np.any(np.abs(eta_pred - 1.0) > defect_eta_tol, axis=2)
        metrics.update(_binary_metrics(pred_defect, true_defect, valid))
        return metrics

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        if self.coef is None:
            raise RuntimeError("Cannot save an unfitted model.")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            alpha=np.asarray(self.alpha, dtype=np.float32),
            use_context=np.asarray(self.use_context),
            stiffness_target=np.asarray(self.stiffness_target),
            x_mean=self.x_mean,
            x_scale=self.x_scale,
            y_mean=self.y_mean,
            y_scale=self.y_scale,
            coef=self.coef,
            intercept=self.intercept,
            window_points=np.asarray(self.window_points, dtype=np.int32),
            obs_channels=np.asarray(self.obs_channels, dtype=np.int32),
            geom_channels=np.asarray(self.geom_channels, dtype=np.int32),
            stiffness_channels=np.asarray(self.stiffness_channels, dtype=np.int32),
            context_channels=np.asarray(self.context_channels, dtype=np.int32),
            metadata_json=np.asarray(json.dumps(metadata or {}, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "JointInverseRidge":
        data = np.load(path, allow_pickle=True)
        model = cls(
            alpha=float(data["alpha"]),
            use_context=bool(data["use_context"]),
            stiffness_target=str(data["stiffness_target"]),
        )
        model.x_mean = np.asarray(data["x_mean"], dtype=np.float32)
        model.x_scale = np.asarray(data["x_scale"], dtype=np.float32)
        model.y_mean = np.asarray(data["y_mean"], dtype=np.float32)
        model.y_scale = np.asarray(data["y_scale"], dtype=np.float32)
        model.coef = np.asarray(data["coef"], dtype=np.float32)
        model.intercept = np.asarray(data["intercept"], dtype=np.float32)
        model.window_points = int(data["window_points"])
        model.obs_channels = int(data["obs_channels"])
        model.geom_channels = int(data["geom_channels"])
        model.stiffness_channels = int(data["stiffness_channels"])
        model.context_channels = int(data["context_channels"])
        return model


def load_joint_inverse_dataset(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = [TARGET_GEOM_KEY, TARGET_STIFFNESS_ETA_KEY, TARGET_STIFFNESS_LOG_KEY, "obs"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Dataset is missing required arrays: {missing}")
    out = {key: np.asarray(data[key]) for key in data.files}
    return out


def train_test_split_indices(n: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2 or test_fraction <= 0.0:
        idx = np.arange(n)
        return idx, idx[:0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = min(n - 1, max(1, int(round(n * test_fraction))))
    test = np.sort(perm[:n_test])
    train = np.sort(perm[n_test:])
    return train, test


def train_from_dataset(
    dataset: dict[str, np.ndarray],
    alpha: float = 1.0,
    stiffness_target: str = "log_eta",
    use_context: bool = True,
    test_fraction: float = 0.2,
    seed: int = 20260701,
) -> tuple[JointInverseRidge, dict[str, Any]]:
    obs = np.asarray(dataset["obs"], dtype=np.float32)
    geom = np.asarray(dataset[TARGET_GEOM_KEY], dtype=np.float32)
    eta = np.asarray(dataset[TARGET_STIFFNESS_ETA_KEY], dtype=np.float32)
    stiffness = eta if stiffness_target == "eta" else np.asarray(dataset[TARGET_STIFFNESS_LOG_KEY], dtype=np.float32)
    context = np.asarray(dataset["context"], dtype=np.float32) if use_context and "context" in dataset else None
    mask = np.asarray(dataset["mask"], dtype=np.float32) if "mask" in dataset else None

    train_idx, test_idx = train_test_split_indices(obs.shape[0], test_fraction, seed)
    model = JointInverseRidge(alpha=alpha, use_context=use_context, stiffness_target=stiffness_target)
    model.fit(
        obs[train_idx],
        geom[train_idx],
        stiffness[train_idx],
        context=context[train_idx] if context is not None else None,
        sample_mask=mask[train_idx] if mask is not None else None,
    )

    report: dict[str, Any] = {
        "train_samples": int(train_idx.size),
        "test_samples": int(test_idx.size),
        "alpha": float(alpha),
        "stiffness_target": stiffness_target,
        "use_context": bool(use_context),
    }
    report["train_metrics"] = model.evaluate(
        obs[train_idx],
        geom[train_idx],
        eta[train_idx],
        context=context[train_idx] if context is not None else None,
        mask=mask[train_idx] if mask is not None else None,
    )
    if test_idx.size:
        report["test_metrics"] = model.evaluate(
            obs[test_idx],
            geom[test_idx],
            eta[test_idx],
            context=context[test_idx] if context is not None else None,
            mask=mask[test_idx] if mask is not None else None,
        )
    return model, report


def predict_dataset(model: JointInverseRidge, dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    context = np.asarray(dataset["context"], dtype=np.float32) if model.use_context and "context" in dataset else None
    pred = model.predict(np.asarray(dataset["obs"], dtype=np.float32), context=context)
    if "meta_json" in dataset:
        pred["meta_json"] = np.asarray(dataset["meta_json"])
    return pred


def aggregate_window_predictions(
    predictions: dict[str, np.ndarray],
    meta_json: np.ndarray,
    ds_m: float,
) -> dict[str, dict[str, np.ndarray]]:
    """Average overlapping window predictions back to run-level mileage profiles."""
    geom = np.asarray(predictions[TARGET_GEOM_KEY], dtype=np.float32)
    eta = np.asarray(predictions[TARGET_STIFFNESS_ETA_KEY], dtype=np.float32)
    meta = [json.loads(str(item)) for item in meta_json]
    grouped: dict[str, dict[str, Any]] = {}

    for i, item in enumerate(meta):
        run_name = str(item.get("run_name") or item.get("npz_path") or f"run_{i}")
        start_rel = float(item.get("window_start_rel_m", 0.0))
        start_abs = float(item.get("window_start_abs_m", start_rel))
        group = grouped.setdefault(run_name, {"rel": [], "abs": [], "geom": [], "eta": []})
        w = geom.shape[1]
        rel = start_rel + np.arange(w, dtype=np.float64) * ds_m
        abs_s = start_abs + np.arange(w, dtype=np.float64) * ds_m
        group["rel"].append(rel)
        group["abs"].append(abs_s)
        group["geom"].append(geom[i])
        group["eta"].append(eta[i])

    out: dict[str, dict[str, np.ndarray]] = {}
    for run_name, group in grouped.items():
        rel_all = np.concatenate(group["rel"])
        abs_all = np.concatenate(group["abs"])
        geom_all = np.concatenate(group["geom"], axis=0)
        eta_all = np.concatenate(group["eta"], axis=0)
        rel_unique = np.unique(np.round(rel_all / ds_m).astype(np.int64))
        rel_grid = rel_unique.astype(np.float64) * ds_m
        geom_sum = np.zeros((rel_grid.size, geom_all.shape[1]), dtype=np.float64)
        eta_sum = np.zeros((rel_grid.size, eta_all.shape[1]), dtype=np.float64)
        abs_sum = np.zeros(rel_grid.size, dtype=np.float64)
        count = np.zeros(rel_grid.size, dtype=np.float64)
        index = {key: j for j, key in enumerate(rel_unique)}
        for rel, abs_s, g, e in zip(np.round(rel_all / ds_m).astype(np.int64), abs_all, geom_all, eta_all):
            j = index[int(rel)]
            geom_sum[j] += g
            eta_sum[j] += e
            abs_sum[j] += abs_s
            count[j] += 1.0
        count_safe = np.maximum(count[:, None], 1.0)
        out[run_name] = {
            "rel_m": rel_grid.astype(np.float32),
            "abs_m": (abs_sum / np.maximum(count, 1.0)).astype(np.float32),
            TARGET_GEOM_KEY: (geom_sum / count_safe).astype(np.float32),
            TARGET_STIFFNESS_ETA_KEY: (eta_sum / count_safe).astype(np.float32),
            "support_count": count.astype(np.float32),
        }
    return out


def save_predictions(path: str | Path, predictions: dict[str, np.ndarray], profiles: dict[str, dict[str, np.ndarray]] | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {key: value for key, value in predictions.items()}
    if profiles:
        run_names = []
        spans = []
        for idx, (run_name, profile) in enumerate(profiles.items()):
            prefix = f"profile_{idx:03d}"
            run_names.append(run_name)
            for key, value in profile.items():
                arrays[f"{prefix}_{key}"] = value
            spans.extend(detect_stiffness_defect_spans(profile, run_name=run_name))
        arrays["profile_run_names"] = np.asarray(run_names)
        arrays["stiffness_defect_spans_json"] = np.asarray(json.dumps(spans, ensure_ascii=False, indent=2))
    np.savez_compressed(output, **arrays)


def detect_stiffness_defect_spans(
    profile: dict[str, np.ndarray],
    run_name: str = "",
    eta_tol: float = 0.15,
    min_length_m: float = 2.0,
) -> list[dict[str, Any]]:
    rel = np.asarray(profile["rel_m"], dtype=np.float64)
    abs_s = np.asarray(profile["abs_m"], dtype=np.float64)
    eta = np.asarray(profile[TARGET_STIFFNESS_ETA_KEY], dtype=np.float64)
    if eta.ndim != 2 or eta.shape[0] != rel.size:
        raise ValueError("profile stiffness eta must be [N, 2] and align with rel_m")
    active = np.any(np.abs(eta - 1.0) > eta_tol, axis=1)
    spans: list[dict[str, Any]] = []
    if not np.any(active):
        return spans
    edges = np.diff(np.concatenate([[False], active, [False]]).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    for start, end in zip(starts, ends):
        end_idx = end - 1
        length = float(rel[end_idx] - rel[start])
        if length < min_length_m:
            continue
        eta_seg = eta[start:end]
        spans.append(
            {
                "run_name": run_name,
                "start_rel_m": float(rel[start]),
                "end_rel_m": float(rel[end_idx]),
                "start_abs_m": float(abs_s[start]),
                "end_abs_m": float(abs_s[end_idx]),
                "length_m": length,
                "eta_k_L_min": float(np.nanmin(eta_seg[:, 0])),
                "eta_k_R_min": float(np.nanmin(eta_seg[:, 1])),
                "eta_k_L_max": float(np.nanmax(eta_seg[:, 0])),
                "eta_k_R_max": float(np.nanmax(eta_seg[:, 1])),
                "defect_score": float(np.nanmax(np.abs(eta_seg - 1.0))),
            }
        )
    return spans


def _as_3d(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be [N, W, C], got {arr.shape}")
    return arr


def _validate_compatible(obs: np.ndarray, geom: np.ndarray, stiffness: np.ndarray) -> None:
    if obs.shape[:2] != geom.shape[:2] or obs.shape[:2] != stiffness.shape[:2]:
        raise ValueError(f"obs, geom and stiffness windows must align, got {obs.shape}, {geom.shape}, {stiffness.shape}")


def _flatten_obs_context(obs: np.ndarray, context: np.ndarray | None) -> np.ndarray:
    x = obs.reshape(obs.shape[0], -1).astype(np.float64)
    if context is not None:
        ctx = np.asarray(context, dtype=np.float64)
        if ctx.ndim == 1:
            ctx = ctx[:, None]
        if ctx.shape[0] != obs.shape[0]:
            raise ValueError("context sample count does not match obs")
        x = np.concatenate([x, ctx], axis=1)
    return x


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    scale = np.nanstd(x, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (x - mean) / scale, mean.astype(np.float32), scale.astype(np.float32)


def _sample_keep_mask(obs: np.ndarray, geom: np.ndarray, stiffness: np.ndarray, sample_mask: np.ndarray | None) -> np.ndarray:
    keep = np.isfinite(obs).all(axis=(1, 2)) & np.isfinite(geom).all(axis=(1, 2)) & np.isfinite(stiffness).all(axis=(1, 2))
    if sample_mask is not None:
        valid = _valid_point_mask(sample_mask, obs.shape[:2])
        keep &= np.mean(valid, axis=1) > 0.5
    return keep


def _valid_point_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        return arr[:, :, 0] > 0.5
    if arr.ndim == 2:
        return arr > 0.5
    raise ValueError(f"mask must be [N,W] or [N,W,C], got {arr.shape}")


def _masked_rmse(pred: np.ndarray, true: np.ndarray, valid: np.ndarray) -> float:
    diff = pred - true
    mask = valid[:, :, None]
    if np.count_nonzero(mask) == 0:
        return float("nan")
    return float(np.sqrt(np.sum((diff ** 2) * mask) / (np.count_nonzero(mask) * pred.shape[2])))


def _masked_mae(pred: np.ndarray, true: np.ndarray, valid: np.ndarray) -> float:
    diff = np.abs(pred - true)
    mask = valid[:, :, None]
    if np.count_nonzero(mask) == 0:
        return float("nan")
    return float(np.sum(diff * mask) / (np.count_nonzero(mask) * pred.shape[2]))


def _binary_metrics(pred: np.ndarray, true: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    pred = pred & valid
    true = true & valid
    tp = float(np.count_nonzero(pred & true))
    fp = float(np.count_nonzero(pred & ~true))
    fn = float(np.count_nonzero(~pred & true))
    tn = float(np.count_nonzero(~pred & ~true & valid))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    return {
        "stiffness_defect_precision": precision,
        "stiffness_defect_recall": recall,
        "stiffness_defect_iou": iou,
        "stiffness_defect_accuracy": accuracy,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or apply a joint irregularity/stiffness inverse baseline.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train the ridge inverse model from a joint inverse dataset.")
    train.add_argument("--dataset", required=True, help="Input dataset npz from pipeline/build_joint_inverse_dataset.py.")
    train.add_argument("--model-output", default="results/joint_inverse_dataset/joint_inverse_ridge_model.npz")
    train.add_argument("--report-output", default="results/joint_inverse_dataset/joint_inverse_ridge_report.json")
    train.add_argument("--prediction-output", default="", help="Optional npz with train/test predictions.")
    train.add_argument("--alpha", type=float, default=1.0)
    train.add_argument("--stiffness-target", choices=["log_eta", "eta"], default="log_eta")
    train.add_argument("--test-fraction", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=20260701)
    train.add_argument("--no-context", action="store_true")

    pred = sub.add_parser("predict", help="Apply a saved ridge inverse model.")
    pred.add_argument("--dataset", required=True)
    pred.add_argument("--model", required=True)
    pred.add_argument("--output", required=True)
    pred.add_argument("--aggregate-profiles", action="store_true")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "train":
        dataset = load_joint_inverse_dataset(args.dataset)
        model, report = train_from_dataset(
            dataset,
            alpha=args.alpha,
            stiffness_target=args.stiffness_target,
            use_context=not args.no_context,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        metadata = {
            "dataset": args.dataset,
            "report": report,
        }
        model.save(args.model_output, metadata=metadata)
        report_path = Path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved model: {args.model_output}")
        print(f"Saved report: {args.report_output}")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.prediction_output:
            pred = predict_dataset(model, dataset)
            profiles = None
            if "meta_json" in dataset:
                ds_m = float(np.asarray(dataset.get("ds_m", 1.0)))
                profiles = aggregate_window_predictions(pred, np.asarray(dataset["meta_json"]), ds_m)
            save_predictions(args.prediction_output, pred, profiles)
            print(f"Saved predictions: {args.prediction_output}")
    elif args.command == "predict":
        dataset = load_joint_inverse_dataset(args.dataset)
        model = JointInverseRidge.load(args.model)
        pred = predict_dataset(model, dataset)
        profiles = None
        if args.aggregate_profiles:
            if "meta_json" not in dataset:
                raise KeyError("Dataset has no meta_json; cannot aggregate profiles.")
            ds_m = float(np.asarray(dataset.get("ds_m", 1.0)))
            profiles = aggregate_window_predictions(pred, np.asarray(dataset["meta_json"]), ds_m)
        save_predictions(args.output, pred, profiles)
        print(f"Saved predictions: {args.output}")


if __name__ == "__main__":
    main()
