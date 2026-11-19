"""Classical baselines under the exact D3-HM five-block spatial protocol.

This program evaluates same-time spatial interpolation when an entire station
block is unavailable.  It deliberately reuses ``spatial_split_used.json``
written by the D3-HM/D3HM-A10 experiment instead of generating a new split.

Default baselines
-----------------
* SpatialMean
* NearestNeighbor
* IDW (power selected only on visible-station validation data)
* KNN (k selected only on visible-station validation data)
* OrdinaryKriging (optional; skipped if pykrige is unavailable)

For every outer fold, all stations in the held-out block are excluded from
the inputs and from hyperparameter selection.  At test time the baselines use
only same-time concentrations from stations in the other four blocks.  Target
truth is accessed only for evaluation.

The default output is placed directly below the existing spatial-block folder:

  /etc/cj/boundary/gfsc_unet_d3hm_a10_hybrid_mask_v1/
      spatial_block/CLASSICAL_BASELINES/

The script also looks for existing D3-HM and D3HM-A10 predictions and, when
found, creates a common paper table and stationwise paired tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from pykrige.ok import OrdinaryKriging
except Exception:
    OrdinaryKriging = None


POLLUTANTS = ("no2", "o3", "pm25")
DEFAULT_MODELS = (
    "SpatialMean",
    "NearestNeighbor",
    "IDW",
    "KNN",
    "OrdinaryKriging",
)
IDW_POWER_CANDIDATES = (0.5, 0.75, 1.0, 1.5, 2.0)
KNN_K_CANDIDATES = (4, 6, 8, 12, 16, 24)
VERSION = "d3hm_exact_five_block_classical_baselines_v1"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def stable_hash(payload: object) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size < 2 or float(np.var(y_true)) <= 1e-12:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": safe_r2(y_true, y_pred),
    }


@dataclass
class PollutantData:
    pollutant: str
    data_dir: Path
    concentration: np.ndarray
    legacy_scaler: object | None
    station_positions: np.ndarray
    station_coordinates: np.ndarray
    time_splits: dict
    original_time_indices: np.ndarray

    @classmethod
    def load(cls, data_root: Path, pollutant: str) -> "PollutantData":
        data_dir = data_root / pollutant
        raw_path = data_dir / "conc_img_raw.npy"
        if raw_path.exists():
            concentration = np.load(raw_path, mmap_mode="r")
            legacy_scaler = None
        else:
            concentration = np.load(data_dir / "conc_img_full.npy", mmap_mode="r")
            legacy_scaler = joblib.load(data_dir / "conc_scaler.pkl")

        station_positions = np.load(data_dir / "station_positions.npy").astype(np.int64)
        coord_img = np.load(data_dir / "coord_img.npy").astype(np.float64)
        station_coordinates = coord_img[
            station_positions[:, 0], station_positions[:, 1]
        ].astype(np.float64)
        time_splits = np.load(
            data_dir / "time_splits.npy", allow_pickle=True
        ).item()
        original_path = data_dir / "valid_time_indices.npy"
        if original_path.exists():
            original_time_indices = np.load(original_path).astype(np.int64)
        else:
            original_time_indices = np.arange(len(concentration), dtype=np.int64)

        if len(station_positions) != 55:
            raise ValueError(
                f"{pollutant}: expected 55 stations, found {len(station_positions)}"
            )
        return cls(
            pollutant=pollutant,
            data_dir=data_dir,
            concentration=concentration,
            legacy_scaler=legacy_scaler,
            station_positions=station_positions,
            station_coordinates=station_coordinates,
            time_splits=time_splits,
            original_time_indices=original_time_indices,
        )

    def raw_image(self, array_idx: int) -> np.ndarray:
        stored = np.asarray(self.concentration[array_idx], dtype=np.float64)
        if self.legacy_scaler is None:
            return stored.copy()
        return (
            stored * float(self.legacy_scaler.scale_[0])
            + float(self.legacy_scaler.mean_[0])
        )

    def station_values(self, array_idx: int) -> np.ndarray:
        image = self.raw_image(array_idx)
        return image[
            self.station_positions[:, 0], self.station_positions[:, 1]
        ].astype(np.float64)


def locate_split(spatial_root: Path, explicit_path: str | None) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            spatial_root / "D3HMA10" / "spatial_split_used.json",
            spatial_root / "D3HM" / "spatial_split_used.json",
            spatial_root / "spatial_split_used.json",
            Path(
                "/etc/cj/boundary/gfsc_unet_d3_hybrid_mask_v1/"
                "spatial_block/D3HM/spatial_split_used.json"
            ),
        ]
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    joined = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find the fixed D3-HM spatial split. Checked:\n  - " + joined
    )


def load_and_validate_split(
    split_path: Path, reference: PollutantData
) -> tuple[list[list[int]], dict]:
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    blocks = [sorted(int(value) for value in block) for block in payload["heldout_by_fold"]]
    if len(blocks) != 5:
        raise ValueError(f"Expected exactly 5 spatial blocks, found {len(blocks)}")

    flattened = [station for block in blocks for station in block]
    expected = list(range(len(reference.station_positions)))
    if sorted(flattened) != expected or len(set(flattened)) != len(expected):
        raise ValueError(
            "The spatial split must contain every station exactly once across the 5 blocks"
        )

    if "station_positions" in payload:
        saved = np.asarray(payload["station_positions"], dtype=np.int64)
        if saved.shape != reference.station_positions.shape or not np.array_equal(
            saved, reference.station_positions
        ):
            raise ValueError("station_positions do not match spatial_split_used.json")
    if "coordinates" in payload:
        saved_coordinates = np.asarray(payload["coordinates"], dtype=np.float64)
        if saved_coordinates.shape != reference.station_coordinates.shape or not np.allclose(
            saved_coordinates, reference.station_coordinates, atol=1e-7, rtol=0.0
        ):
            raise ValueError("station coordinates do not match spatial_split_used.json")
    return blocks, payload


def distance_matrix(target: np.ndarray, known: np.ndarray) -> np.ndarray:
    return np.linalg.norm(target[:, None, :] - known[None, :, :], axis=2)


def rowwise_spatial_mean(values: np.ndarray, n_targets: int) -> np.ndarray:
    return np.full(n_targets, float(np.mean(values)), dtype=np.float64)


def rowwise_nearest(distances: np.ndarray, values: np.ndarray) -> np.ndarray:
    return values[np.argmin(distances, axis=1)].astype(np.float64)


def rowwise_idw(
    distances: np.ndarray, values: np.ndarray, power: float
) -> np.ndarray:
    predictions = np.empty(distances.shape[0], dtype=np.float64)
    for row_idx, row in enumerate(distances):
        zero = np.flatnonzero(row < 1e-12)
        if zero.size:
            predictions[row_idx] = float(values[int(zero[0])])
            continue
        weights = 1.0 / np.power(np.maximum(row, 1e-12), power)
        predictions[row_idx] = float(np.sum(weights * values) / np.sum(weights))
    return predictions


def rowwise_knn(
    distances: np.ndarray, values: np.ndarray, k: int
) -> np.ndarray:
    predictions = np.empty(distances.shape[0], dtype=np.float64)
    for row_idx, row in enumerate(distances):
        order = np.argsort(row)[: min(k, len(row))]
        selected_distances = row[order]
        selected_values = values[order]
        zero = np.flatnonzero(selected_distances < 1e-12)
        if zero.size:
            predictions[row_idx] = float(selected_values[int(zero[0])])
            continue
        weights = 1.0 / np.maximum(selected_distances, 1e-12)
        predictions[row_idx] = float(
            np.sum(weights * selected_values) / np.sum(weights)
        )
    return predictions


def ordinary_kriging_predictions(
    known_coordinates: np.ndarray,
    known_values: np.ndarray,
    target_coordinates: np.ndarray,
) -> np.ndarray:
    fallback = rowwise_spatial_mean(known_values, len(target_coordinates))
    if OrdinaryKriging is None:
        return fallback
    try:
        model = OrdinaryKriging(
            known_coordinates[:, 1],
            known_coordinates[:, 0],
            known_values,
            variogram_model="spherical",
            verbose=False,
            enable_plotting=False,
        )
        prediction, _ = model.execute(
            "points", target_coordinates[:, 1], target_coordinates[:, 0]
        )
        result = np.asarray(prediction, dtype=np.float64).reshape(-1)
        result[~np.isfinite(result)] = fallback[~np.isfinite(result)]
        return result
    except Exception:
        return fallback


def validation_truth_and_predictions(
    data: PollutantData,
    visible_stations: np.ndarray,
    idw_power: float | None = None,
    knn_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    truths: list[float] = []
    predictions: list[float] = []
    for array_idx in np.asarray(data.time_splits["val"], dtype=np.int64):
        values = data.station_values(int(array_idx))
        for target_station in visible_stations:
            truth = float(values[target_station])
            if not np.isfinite(truth):
                continue
            known_ids = visible_stations[visible_stations != target_station]
            finite = np.isfinite(values[known_ids])
            known_ids = known_ids[finite]
            if len(known_ids) < 2:
                continue
            distances = distance_matrix(
                data.station_coordinates[[target_station]],
                data.station_coordinates[known_ids],
            )
            known_values = values[known_ids]
            if idw_power is not None:
                prediction = rowwise_idw(distances, known_values, idw_power)[0]
            elif knn_k is not None:
                prediction = rowwise_knn(distances, known_values, knn_k)[0]
            else:
                raise ValueError("Specify idw_power or knn_k")
            truths.append(truth)
            predictions.append(float(prediction))
    return np.asarray(truths), np.asarray(predictions)


def select_hyperparameters(
    data: PollutantData, visible_stations: np.ndarray
) -> tuple[float, int, dict]:
    idw_scores: dict[str, float] = {}
    for power in IDW_POWER_CANDIDATES:
        truth, prediction = validation_truth_and_predictions(
            data, visible_stations, idw_power=power
        )
        idw_scores[str(power)] = float(mean_absolute_error(truth, prediction))

    knn_scores: dict[str, float] = {}
    for k in KNN_K_CANDIDATES:
        truth, prediction = validation_truth_and_predictions(
            data, visible_stations, knn_k=k
        )
        knn_scores[str(k)] = float(mean_absolute_error(truth, prediction))

    best_power = float(min(idw_scores, key=idw_scores.get))
    best_k = int(min(knn_scores, key=knn_scores.get))
    return best_power, best_k, {
        "idw_validation_MAE": idw_scores,
        "knn_validation_MAE": knn_scores,
    }


def fold_paths(output_root: Path, pollutant: str, fold: int) -> tuple[Path, Path]:
    fold_dir = output_root / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    return (
        fold_dir / f"{pollutant}_block_{fold:02d}_predictions.csv",
        fold_dir / f"{pollutant}_block_{fold:02d}.complete.json",
    )


def reusable_fold(
    output_root: Path,
    pollutant: str,
    fold: int,
    models: tuple[str, ...],
    split_signature: str,
    force: bool,
) -> pd.DataFrame | None:
    prediction_path, marker_path = fold_paths(output_root, pollutant, fold)
    if force or not prediction_path.exists() or not marker_path.exists():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        expected = {
            "version": VERSION,
            "pollutant": pollutant,
            "fold": fold,
            "models": list(models),
            "split_signature": split_signature,
            "status": "completed",
        }
        if any(marker.get(key) != value for key, value in expected.items()):
            return None
        frame = pd.read_csv(prediction_path)
        if sorted(frame["model"].unique()) != sorted(models):
            return None
        return frame
    except Exception:
        return None


def run_fold(
    data: PollutantData,
    fold: int,
    heldout: list[int],
    models: tuple[str, ...],
    output_root: Path,
    split_signature: str,
    force: bool,
) -> tuple[pd.DataFrame, dict]:
    reused = reusable_fold(
        output_root, data.pollutant, fold, models, split_signature, force
    )
    prediction_path, marker_path = fold_paths(output_root, data.pollutant, fold)
    if reused is not None:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        print(f"  {data.pollutant} block {fold}: reuse completed predictions")
        return reused, marker["hyperparameters"]

    heldout_ids = np.asarray(sorted(heldout), dtype=np.int64)
    visible_ids = np.asarray(
        [idx for idx in range(len(data.station_positions)) if idx not in set(heldout)],
        dtype=np.int64,
    )
    best_power, best_k, validation_scores = select_hyperparameters(data, visible_ids)
    print(
        f"  {data.pollutant} block {fold}: heldout={len(heldout_ids)}, "
        f"visible={len(visible_ids)}, IDW p={best_power}, KNN k={best_k}"
    )

    rows: list[dict] = []
    started = time.time()
    test_indices = np.asarray(data.time_splits["test"], dtype=np.int64)
    for array_idx in test_indices:
        values = data.station_values(int(array_idx))
        finite_known = np.isfinite(values[visible_ids])
        known_ids = visible_ids[finite_known]
        known_values = values[known_ids]
        finite_target = np.isfinite(values[heldout_ids])
        target_ids = heldout_ids[finite_target]
        target_values = values[target_ids]
        if len(known_ids) < 2 or len(target_ids) == 0:
            continue

        known_coordinates = data.station_coordinates[known_ids]
        target_coordinates = data.station_coordinates[target_ids]
        distances = distance_matrix(target_coordinates, known_coordinates)
        prediction_by_model: dict[str, np.ndarray] = {}
        if "SpatialMean" in models:
            prediction_by_model["SpatialMean"] = rowwise_spatial_mean(
                known_values, len(target_ids)
            )
        if "NearestNeighbor" in models:
            prediction_by_model["NearestNeighbor"] = rowwise_nearest(
                distances, known_values
            )
        if "IDW" in models:
            prediction_by_model["IDW"] = rowwise_idw(
                distances, known_values, best_power
            )
        if "KNN" in models:
            prediction_by_model["KNN"] = rowwise_knn(distances, known_values, best_k)
        if "OrdinaryKriging" in models:
            prediction_by_model["OrdinaryKriging"] = ordinary_kriging_predictions(
                known_coordinates, known_values, target_coordinates
            )

        original_time = int(data.original_time_indices[int(array_idx)])
        for model, predictions in prediction_by_model.items():
            for station, truth, prediction in zip(
                target_ids, target_values, predictions
            ):
                if np.isfinite(truth) and np.isfinite(prediction):
                    rows.append(
                        {
                            "variant": model,
                            "model": model,
                            "pollutant": data.pollutant,
                            "fold": fold,
                            "station": int(station),
                            "time_idx": original_time,
                            "true": float(truth),
                            "prediction": float(prediction),
                            "error": float(prediction - truth),
                        }
                    )

    frame = pd.DataFrame(rows)
    expected_rows_per_model = len(heldout_ids) * len(test_indices)
    counts = frame.groupby("model").size().to_dict()
    for model in models:
        if counts.get(model, 0) != expected_rows_per_model:
            raise RuntimeError(
                f"{data.pollutant} block {fold} {model}: expected "
                f"{expected_rows_per_model} predictions, found {counts.get(model, 0)}"
            )
    frame.to_csv(prediction_path, index=False)
    hyperparameters = {
        "selected_idw_power": best_power,
        "selected_knn_k": best_k,
        **validation_scores,
    }
    marker = {
        "status": "completed",
        "version": VERSION,
        "pollutant": data.pollutant,
        "fold": fold,
        "heldout_stations": heldout,
        "models": list(models),
        "split_signature": split_signature,
        "n_test_times": int(len(test_indices)),
        "n_rows_per_model": expected_rows_per_model,
        "elapsed_seconds": time.time() - started,
        "hyperparameters": hyperparameters,
    }
    atomic_json(marker_path, marker)
    return frame, hyperparameters


def station_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    output = []
    for (pollutant, model, station), group in predictions.groupby(
        ["pollutant", "model", "station"], sort=True
    ):
        values = metric_dict(group["true"].to_numpy(), group["prediction"].to_numpy())
        output.append(
            {
                "pollutant": pollutant,
                "model": model,
                "station": int(station),
                "fold": int(group["fold"].iloc[0]),
                "n": len(group),
                **values,
            }
        )
    return pd.DataFrame(output)


def fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    output = []
    for (pollutant, model, fold), group in predictions.groupby(
        ["pollutant", "model", "fold"], sort=True
    ):
        output.append(
            {
                "pollutant": pollutant,
                "model": model,
                "fold": int(fold),
                "n_stations": int(group["station"].nunique()),
                "n": len(group),
                **metric_dict(
                    group["true"].to_numpy(), group["prediction"].to_numpy()
                ),
            }
        )
    return pd.DataFrame(output)


def aggregate_metrics(
    predictions: pd.DataFrame, station_table: pd.DataFrame
) -> pd.DataFrame:
    output = []
    for (pollutant, model), group in predictions.groupby(
        ["pollutant", "model"], sort=True
    ):
        current_stations = station_table[
            (station_table["pollutant"] == pollutant)
            & (station_table["model"] == model)
        ]
        pooled = metric_dict(
            group["true"].to_numpy(), group["prediction"].to_numpy()
        )
        output.append(
            {
                "pollutant": pollutant,
                "model": model,
                "n": len(group),
                "n_stations": int(group["station"].nunique()),
                "n_folds": int(group["fold"].nunique()),
                "pooled_MAE": pooled["MAE"],
                "pooled_RMSE": pooled["RMSE"],
                "pooled_R2": pooled["R2"],
                "mean_station_MAE": float(current_stations["MAE"].mean()),
                "mean_station_RMSE": float(current_stations["RMSE"].mean()),
                "mean_station_R2": float(current_stations["R2"].mean()),
                "median_station_R2": float(current_stations["R2"].median()),
            }
        )
    return pd.DataFrame(output)


def normalize_external_predictions(frame: pd.DataFrame, default_model: str) -> pd.DataFrame:
    result = frame.copy()
    if "model" not in result.columns:
        if "variant" in result.columns:
            result["model"] = result["variant"]
        else:
            result["model"] = default_model
    result["model"] = result["model"].fillna(default_model).astype(str)
    if "variant" not in result.columns:
        result["variant"] = result["model"]
    if "error" not in result.columns:
        result["error"] = result["prediction"] - result["true"]
    required = [
        "variant", "model", "pollutant", "fold", "station", "time_idx",
        "true", "prediction", "error",
    ]
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise ValueError(f"External prediction file is missing columns: {missing}")
    return result[required]


def external_prediction_candidates(spatial_root: Path) -> dict[str, list[Path]]:
    return {
        "D3HM": [
            Path(os.environ.get("D3HM_PREDICTIONS", "")),
            spatial_root / "D3HM" / "predictions" / "test_predictions.csv",
            Path(
                "/etc/cj/boundary/gfsc_unet_d3_hybrid_mask_v1/"
                "spatial_block/D3HM/predictions/test_predictions.csv"
            ),
        ],
        "D3HMA10": [
            Path(os.environ.get("D3HMA10_PREDICTIONS", "")),
            spatial_root / "D3HMA10" / "predictions" / "test_predictions.csv",
        ],
    }


def load_external_predictions(spatial_root: Path) -> list[pd.DataFrame]:
    loaded: list[pd.DataFrame] = []
    for model, candidates in external_prediction_candidates(spatial_root).items():
        unique_candidates = []
        for path in candidates:
            if str(path) and path not in unique_candidates:
                unique_candidates.append(path)
        chosen = next((path for path in unique_candidates if path.is_file()), None)
        if chosen is None:
            print(f"[warning] {model} predictions not found; combined table will omit it")
            continue
        frame = normalize_external_predictions(pd.read_csv(chosen), model)
        # Force the formal display name even if an old internal label is present.
        frame["model"] = model
        frame["variant"] = model
        loaded.append(frame)
        print(f"Loaded {model} predictions: {chosen}")
    return loaded


def assert_same_test_keys(reference: pd.DataFrame, candidate: pd.DataFrame) -> None:
    key = ["pollutant", "fold", "station", "time_idx"]
    ref = reference[key + ["true"]].sort_values(key).reset_index(drop=True)
    cur = candidate[key + ["true"]].sort_values(key).reset_index(drop=True)
    if not ref[key].equals(cur[key]):
        raise ValueError("Prediction files do not contain identical spatial-block test keys")
    if not np.allclose(ref["true"], cur["true"], atol=1e-4, rtol=0.0):
        maximum = float(np.max(np.abs(ref["true"].to_numpy() - cur["true"].to_numpy())))
        raise ValueError(f"Prediction files disagree on test truth; max difference={maximum}")


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=np.float64)
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    finite_idx = np.flatnonzero(np.isfinite(values))
    if finite_idx.size == 0:
        return adjusted
    finite_values = values[finite_idx]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    m = len(ranked)
    q_ranked = ranked * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.minimum(q_ranked, 1.0)
    restored = np.empty(m, dtype=np.float64)
    restored[order] = q_ranked
    adjusted[finite_idx] = restored
    return adjusted


def paired_station_tests(
    combined_station_metrics: pd.DataFrame, reference_model: str = "D3HM"
) -> pd.DataFrame:
    reference = combined_station_metrics[
        combined_station_metrics["model"] == reference_model
    ]
    if reference.empty:
        return pd.DataFrame()
    output = []
    for pollutant in POLLUTANTS:
        ref_pollutant = reference[reference["pollutant"] == pollutant]
        for model in sorted(combined_station_metrics["model"].unique()):
            if model == reference_model:
                continue
            candidate = combined_station_metrics[
                (combined_station_metrics["pollutant"] == pollutant)
                & (combined_station_metrics["model"] == model)
            ]
            paired = ref_pollutant.merge(
                candidate,
                on=["pollutant", "station"],
                suffixes=("_reference", "_candidate"),
            )
            for metric in ("MAE", "RMSE", "R2"):
                ref_values = paired[f"{metric}_reference"].to_numpy(dtype=np.float64)
                cand_values = paired[f"{metric}_candidate"].to_numpy(dtype=np.float64)
                finite = np.isfinite(ref_values) & np.isfinite(cand_values)
                ref_values = ref_values[finite]
                cand_values = cand_values[finite]
                difference = cand_values - ref_values
                if len(difference) == 0 or np.allclose(difference, 0.0):
                    statistic, p_value = 0.0, 1.0
                else:
                    statistic, p_value = wilcoxon(
                        difference, alternative="two-sided", zero_method="wilcox"
                    )
                lower_is_better = metric in {"MAE", "RMSE"}
                candidate_better = int(
                    np.sum(difference < 0) if lower_is_better else np.sum(difference > 0)
                )
                reference_better = int(
                    np.sum(difference > 0) if lower_is_better else np.sum(difference < 0)
                )
                output.append(
                    {
                        "pollutant": pollutant,
                        "reference": reference_model,
                        "candidate": model,
                        "metric": metric,
                        "n_stations": len(difference),
                        "mean_candidate_minus_reference": float(np.mean(difference)),
                        "median_candidate_minus_reference": float(np.median(difference)),
                        "candidate_better_stations": candidate_better,
                        "reference_better_stations": reference_better,
                        "ties": int(len(difference) - candidate_better - reference_better),
                        "wilcoxon_statistic": float(statistic),
                        "p_value": float(p_value),
                    }
                )
    result = pd.DataFrame(output)
    if not result.empty:
        result["p_BH_all_tests"] = benjamini_hochberg(result["p_value"])
        result["significant_BH_0.05"] = result["p_BH_all_tests"] < 0.05
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=os.environ.get("SCUNET_DATA_ROOT", "/etc/cj/boundary/duoyuan/data"),
    )
    parser.add_argument(
        "--spatial-root",
        default=os.environ.get(
            "D3HM_SPATIAL_ROOT",
            "/etc/cj/boundary/gfsc_unet_d3hm_a10_hybrid_mask_v1/spatial_block",
        ),
        help="Existing folder shown in the screenshot; outputs are placed below it.",
    )
    parser.add_argument(
        "--split-json",
        default=os.environ.get("SPATIAL_SPLIT_JSON"),
        help="Optional explicit path to spatial_split_used.json.",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated baselines.",
    )
    parser.add_argument(
        "--pollutants",
        default=",".join(POLLUTANTS),
        help="Comma-separated pollutants.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore completed folds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    spatial_root = Path(args.spatial_root).resolve()
    output_root = spatial_root / "CLASSICAL_BASELINES"
    metrics_root = output_root / "metrics"
    predictions_root = output_root / "predictions"
    metrics_root.mkdir(parents=True, exist_ok=True)
    predictions_root.mkdir(parents=True, exist_ok=True)

    requested_models = tuple(
        value.strip() for value in args.models.split(",") if value.strip()
    )
    unknown = sorted(set(requested_models) - set(DEFAULT_MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    models = requested_models
    if "OrdinaryKriging" in models and OrdinaryKriging is None:
        print(
            "[warning] pykrige is unavailable; OrdinaryKriging is skipped. "
            "Install it with: pip install pykrige"
        )
        models = tuple(model for model in models if model != "OrdinaryKriging")
    if not models:
        raise ValueError("No runnable baseline remains")

    pollutants = tuple(
        value.strip().lower() for value in args.pollutants.split(",") if value.strip()
    )
    invalid_pollutants = sorted(set(pollutants) - set(POLLUTANTS))
    if invalid_pollutants:
        raise ValueError(f"Unknown pollutants: {invalid_pollutants}")

    reference_data = PollutantData.load(data_root, pollutants[0])
    split_path = locate_split(spatial_root, args.split_json)
    blocks, split_payload = load_and_validate_split(split_path, reference_data)
    split_signature = str(
        split_payload.get("split_signature", stable_hash(split_payload["heldout_by_fold"]))
    )

    protocol = {
        "version": VERSION,
        "data_root": str(data_root),
        "spatial_root": str(spatial_root),
        "output_root": str(output_root),
        "split_json": str(split_path),
        "split_signature": split_signature,
        "outer_protocol": "the exact fixed five spatial blocks used by D3-HM",
        "heldout_by_fold": blocks,
        "models": list(models),
        "pollutants": list(pollutants),
        "test_inputs": "same-time concentrations from stations in the other four blocks only",
        "target_history_used": False,
        "prediction_clipping": False,
        "selection": {
            "IDW_power_candidates": list(IDW_POWER_CANDIDATES),
            "KNN_k_candidates": list(KNN_K_CANDIDATES),
            "criterion": "MAE on validation times and visible stations only",
            "heldout_block_used_for_selection": False,
        },
        "ordinary_kriging": {
            "variogram_model": "spherical",
            "fallback_on_numerical_failure": "SpatialMean",
        },
    }
    atomic_json(output_root / "protocol.json", protocol)

    all_predictions: list[pd.DataFrame] = []
    hyperparameter_rows: list[dict] = []
    for pollutant in pollutants:
        data = reference_data if pollutant == pollutants[0] else PollutantData.load(
            data_root, pollutant
        )
        # All pollutants must use the exact same station geometry and split.
        load_and_validate_split(split_path, data)
        for fold, heldout in enumerate(blocks):
            frame, hyperparameters = run_fold(
                data=data,
                fold=fold,
                heldout=heldout,
                models=models,
                output_root=output_root,
                split_signature=split_signature,
                force=args.force,
            )
            all_predictions.append(frame)
            hyperparameter_rows.append(
                {
                    "pollutant": pollutant,
                    "fold": fold,
                    "heldout_station_count": len(heldout),
                    "selected_idw_power": hyperparameters["selected_idw_power"],
                    "selected_knn_k": hyperparameters["selected_knn_k"],
                    "idw_validation_MAE_json": json.dumps(
                        hyperparameters["idw_validation_MAE"], sort_keys=True
                    ),
                    "knn_validation_MAE_json": json.dumps(
                        hyperparameters["knn_validation_MAE"], sort_keys=True
                    ),
                }
            )

    predictions = pd.concat(all_predictions, ignore_index=True)
    expected_total_per_model = 55 * len(
        np.asarray(reference_data.time_splits["test"], dtype=np.int64)
    ) * len(pollutants)
    counts = predictions.groupby("model").size()
    if not (counts == expected_total_per_model).all():
        raise RuntimeError(
            f"Incomplete pooled predictions: expected {expected_total_per_model} per model, "
            f"found {counts.to_dict()}"
        )

    station_table = station_metrics(predictions)
    fold_table = fold_metrics(predictions)
    aggregate_table = aggregate_metrics(predictions, station_table)
    predictions.to_csv(predictions_root / "test_predictions.csv", index=False)
    station_table.to_csv(metrics_root / "station_metrics.csv", index=False)
    fold_table.to_csv(metrics_root / "fold_metrics.csv", index=False)
    aggregate_table.to_csv(metrics_root / "aggregate_metrics.csv", index=False)
    pd.DataFrame(hyperparameter_rows).to_csv(
        metrics_root / "selected_hyperparameters.csv", index=False
    )

    external = load_external_predictions(spatial_root)
    if external:
        reference_external = external[0].drop_duplicates(
            ["pollutant", "fold", "station", "time_idx"]
        )
        baseline_reference = predictions[predictions["model"] == models[0]]
        assert_same_test_keys(reference_external, baseline_reference)
        for frame in external[1:]:
            assert_same_test_keys(reference_external, frame)
        combined = pd.concat([*external, predictions], ignore_index=True)
        combined_station = station_metrics(combined)
        combined_aggregate = aggregate_metrics(combined, combined_station)
        combined_aggregate.to_csv(
            metrics_root / "paper_table_spatial_block_all_models.csv", index=False
        )
        tests = paired_station_tests(combined_station, reference_model="D3HM")
        if not tests.empty:
            tests.to_csv(
                metrics_root / "paired_station_tests_vs_D3HM.csv", index=False
            )

    completion = {
        "status": "completed",
        "version": VERSION,
        "split_signature": split_signature,
        "pollutants": list(pollutants),
        "models": list(models),
        "n_spatial_folds": 5,
        "n_stations": 55,
        "n_test_times": int(len(reference_data.time_splits["test"])),
        "expected_records_per_pollutant_model": int(
            55 * len(reference_data.time_splits["test"])
        ),
    }
    atomic_json(metrics_root / "completion_status.json", completion)

    print("\nFive-block pooled results")
    print(
        aggregate_table[
            ["pollutant", "model", "pooled_MAE", "pooled_RMSE", "pooled_R2"]
        ].to_string(index=False)
    )
    print(f"\nResults saved to: {output_root}")


if __name__ == "__main__":
    main()
