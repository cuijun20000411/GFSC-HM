"""Final paper-aligned preprocessing for station-to-grid image inpainting.

The concentration target is read only from ``*_label.npy``.  Dense
``*_ss_label.npy`` files are intentionally never loaded because they are not
ground truth away from monitoring stations.

The script keeps the original 32 x 32 raster convention: grid id ``g`` is
mapped to ``(g // 32, g % 32)``.  This convention must match the data creator.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import random
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
warnings.filterwarnings("ignore")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 300,
    }
)

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)


class Config:
    """Edit these paths, or override them with environment variables."""

    ROOT_DATA_PATH = os.environ.get(
        "SCUNET_RAW_ROOT", "/etc/cj/boundary/boundary"
    )
    OUTPUT_DIR = os.environ.get(
        "SCUNET_PREPROCESS_ROOT", "/etc/cj/boundary/duoyuan"
    )

    IMG_SHAPE = (32, 32)
    NUM_GRIDS = 999
    TOTAL_GRIDS = IMG_SHAPE[0] * IMG_SHAPE[1]
    EXPECTED_FEATURE_DIMS = {"no2": 39, "o3": 39, "pm25": 40}
    POLLUTANTS = [
        x.strip()
        for x in os.environ.get("SCUNET_POLLUTANTS", "no2,o3,pm25").split(",")
        if x.strip()
    ]

    POLLUTANT_PATH = {
        "no2": {
            "label": os.path.join(ROOT_DATA_PATH, "no2", "no2_label.npy"),
            "features": os.path.join(ROOT_DATA_PATH, "no2", "no2_features.npy"),
        },
        "o3": {
            "label": os.path.join(ROOT_DATA_PATH, "o3", "o3_label.npy"),
            "features": os.path.join(ROOT_DATA_PATH, "o3", "o3_features.npy"),
        },
        "pm25": {
            "label": os.path.join(ROOT_DATA_PATH, "pm2_5", "pm2_5_label.npy"),
            "features": os.path.join(
                ROOT_DATA_PATH, "pm2_5", "pm2_5_features.npy"
            ),
        },
    }

    GRID_LABEL_PATH = os.path.join(ROOT_DATA_PATH, "grid_label.csv")
    ADJ_MATRIX_PATH = os.path.join(ROOT_DATA_PATH, "adj.npy")


def check_all_paths(config: Config) -> None:
    if config.NUM_GRIDS > config.TOTAL_GRIDS:
        raise ValueError("NUM_GRIDS cannot exceed the 32x32 raster size")

    required = [config.GRID_LABEL_PATH]
    for pollutant in config.POLLUTANTS:
        if pollutant not in config.POLLUTANT_PATH:
            raise KeyError(f"Unsupported pollutant: {pollutant}")
        required.extend(config.POLLUTANT_PATH[pollutant].values())

    missing = [str(path) for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

    for name in ("data", "base_info", "visualization"):
        Path(config.OUTPUT_DIR, name).mkdir(parents=True, exist_ok=True)


def load_station_info(config: Config):
    grid_label = pd.read_csv(config.GRID_LABEL_PATH)
    if grid_label.shape[1] < 2:
        raise ValueError("grid_label.csv must contain grid id in its second column")

    station_indices = pd.to_numeric(grid_label.iloc[:, 1], errors="coerce")
    station_indices = station_indices.dropna().to_numpy(dtype=np.int64)
    station_indices = np.unique(
        station_indices[
            (station_indices >= 0) & (station_indices < config.NUM_GRIDS)
        ]
    )
    if station_indices.size < 2:
        raise ValueError("At least two monitoring stations are required")

    station_positions = np.column_stack(
        (station_indices // config.IMG_SHAPE[1], station_indices % config.IMG_SHAPE[1])
    ).astype(np.int64)
    non_station_grid_idx = np.setdiff1d(
        np.arange(config.NUM_GRIDS, dtype=np.int64), station_indices
    )

    global_mask = np.zeros(config.IMG_SHAPE, dtype=np.float32)
    global_mask[station_positions[:, 0], station_positions[:, 1]] = 1.0

    # Only the first NUM_GRIDS pixels are part of the study domain.  The final
    # 25 pixels of a 32x32 array are padding when NUM_GRIDS == 999.
    valid_grid_mask = np.zeros(config.TOTAL_GRIDS, dtype=np.float32)
    valid_grid_mask[: config.NUM_GRIDS] = 1.0
    valid_grid_mask = valid_grid_mask.reshape(config.IMG_SHAPE)

    base_dir = Path(config.OUTPUT_DIR, "base_info")
    np.save(base_dir / "station_indices.npy", station_indices)
    np.save(base_dir / "station_positions.npy", station_positions)
    np.save(base_dir / "station_grid_idx.npy", station_indices)
    np.save(base_dir / "non_station_grid_idx.npy", non_station_grid_idx)
    np.save(base_dir / "global_mask.npy", global_mask)
    np.save(base_dir / "valid_grid_mask.npy", valid_grid_mask)
    if os.path.exists(config.ADJ_MATRIX_PATH):
        np.save(base_dir / "adj_matrix.npy", np.load(config.ADJ_MATRIX_PATH))

    logging.info("Loaded %d monitoring stations", len(station_indices))
    return station_indices, station_positions, global_mask, valid_grid_mask


def _validate_arrays(
    pollutant: str,
    label: np.ndarray,
    features: np.ndarray,
    station_grid_idx: np.ndarray,
    config: Config,
) -> None:
    if label.ndim != 2:
        raise ValueError(f"{pollutant}: label must have shape (T, G)")
    if features.ndim != 3:
        raise ValueError(f"{pollutant}: features must have shape (T, G, F)")
    if label.shape[:2] != features.shape[:2]:
        raise ValueError(
            f"{pollutant}: label/features mismatch: {label.shape} vs {features.shape}"
        )
    if label.shape[1] != config.NUM_GRIDS:
        raise ValueError(
            f"{pollutant}: expected {config.NUM_GRIDS} grids, got {label.shape[1]}"
        )
    if not np.isfinite(features).all():
        raise ValueError(f"{pollutant}: features contain NaN or infinity")
    expected_dim = config.EXPECTED_FEATURE_DIMS.get(pollutant)
    if expected_dim is not None and features.shape[-1] != expected_dim:
        raise ValueError(
            f"{pollutant}: expected {expected_dim} feature channels, "
            f"got {features.shape[-1]}"
        )

    # The only supervised concentration values must be the station pixels.
    finite_any = np.where(np.isfinite(label).any(axis=0))[0]
    if not np.array_equal(finite_any, station_grid_idx):
        missing = np.setdiff1d(station_grid_idx, finite_any)
        extra = np.setdiff1d(finite_any, station_grid_idx)
        raise ValueError(
            f"{pollutant}: label/station mismatch; missing={missing.tolist()}, "
            f"extra={extra.tolist()}. Do not substitute ss_label as ground truth."
        )


def process_pollutant_data_loocv(
    config: Config,
    pollutant: str,
    station_grid_idx: np.ndarray,
    station_positions: np.ndarray,
    global_mask: np.ndarray,
    valid_grid_mask: np.ndarray,
) -> None:
    logging.info("Processing %s", pollutant.upper())
    label = np.load(config.POLLUTANT_PATH[pollutant]["label"], mmap_mode="r")
    features = np.load(
        config.POLLUTANT_PATH[pollutant]["features"], mmap_mode="r"
    )
    _validate_arrays(pollutant, label, features, station_grid_idx, config)

    total_times, _, feature_dim = features.shape
    # Integer arithmetic avoids int(720 * 0.70) becoming 503 because of
    # floating-point representation.  For T=720 this gives 504/108/108.
    train_end = total_times * 70 // 100
    val_end = total_times * 85 // 100
    raw_splits = {
        "train": np.arange(0, train_end),
        "val": np.arange(train_end, val_end),
        "test": np.arange(val_end, total_times),
    }

    valid_original = {}
    for mode, indices in raw_splits.items():
        station_values = np.asarray(label[indices][:, station_grid_idx])
        valid_original[mode] = indices[np.isfinite(station_values).all(axis=1)]
        if len(valid_original[mode]) == 0:
            raise ValueError(f"{pollutant}: no valid time steps in {mode} split")

    valid_time_indices = np.concatenate(list(valid_original.values()))
    time_splits = {}
    start = 0
    for mode in ("train", "val", "test"):
        stop = start + len(valid_original[mode])
        time_splits[mode] = np.arange(start, stop, dtype=np.int64)
        start = stop

    # This global scaler is retained for backward compatibility and quick
    # visualization.  rh4_final_paper.py reads conc_img_raw.npy and re-fits
    # concentration statistics inside each LOSO fold using training times and
    # non-held-out stations only.
    train_station_values = np.asarray(
        label[valid_original["train"]][:, station_grid_idx]
    ).reshape(-1, 1)
    conc_scaler = StandardScaler().fit(train_station_values)
    conc_mean = float(conc_scaler.mean_[0])
    conc_scale = float(conc_scaler.scale_[0])

    feat_scaler = StandardScaler()
    train_features = np.asarray(features[valid_original["train"]])
    feat_scaler.fit(train_features.reshape(-1, feature_dim))

    num_valid_times = len(valid_time_indices)
    conc_img_full = np.full(
        (num_valid_times, *config.IMG_SHAPE), np.nan, dtype=np.float32
    )
    conc_img_raw = np.full(
        (num_valid_times, *config.IMG_SHAPE), np.nan, dtype=np.float32
    )
    cov_img_full = np.zeros(
        (num_valid_times, feature_dim, *config.IMG_SHAPE), dtype=np.float32
    )

    flat_station = station_grid_idx
    for out_t, original_t in enumerate(valid_time_indices):
        conc_flat = conc_img_full[out_t].reshape(-1)
        raw_flat = conc_img_raw[out_t].reshape(-1)
        values = np.asarray(label[original_t, flat_station], dtype=np.float32)
        raw_flat[flat_station] = values
        conc_flat[flat_station] = (values - conc_mean) / (conc_scale + 1e-8)

        feat_t = np.asarray(features[original_t])
        feat_norm = feat_scaler.transform(feat_t).astype(np.float32, copy=False)
        cov_flat = cov_img_full[out_t].reshape(feature_dim, -1)
        cov_flat[:, : config.NUM_GRIDS] = feat_norm.T

    coord_img = np.stack(
        np.meshgrid(
            np.arange(config.IMG_SHAPE[0], dtype=np.float32)
            / max(config.IMG_SHAPE[0] - 1, 1),
            np.arange(config.IMG_SHAPE[1], dtype=np.float32)
            / max(config.IMG_SHAPE[1] - 1, 1),
            indexing="ij",
        ),
        axis=-1,
    )

    save_dir = Path(config.OUTPUT_DIR, "data", pollutant)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "conc_img_raw.npy", conc_img_raw)
    np.save(save_dir / "conc_img_full.npy", conc_img_full)
    np.save(save_dir / "cov_img_full.npy", cov_img_full)
    np.save(save_dir / "coord_img.npy", coord_img)
    np.save(save_dir / "time_splits.npy", time_splits)
    np.save(save_dir / "valid_time_indices.npy", valid_time_indices)
    np.save(save_dir / "fixed_global_stats.npy", np.array([0.0, 1.0]))
    np.save(save_dir / "station_positions.npy", station_positions)
    np.save(save_dir / "station_grid_idx.npy", station_grid_idx)
    np.save(save_dir / "global_mask.npy", global_mask)
    np.save(save_dir / "valid_grid_mask.npy", valid_grid_mask)
    joblib.dump(conc_scaler, save_dir / "conc_scaler.pkl")
    joblib.dump(feat_scaler, save_dir / "feat_scaler.pkl")
    metadata = {
        "pollutant": pollutant,
        "image_shape": list(config.IMG_SHAPE),
        "num_valid_grids": config.NUM_GRIDS,
        "num_stations": int(len(station_grid_idx)),
        "feature_dim": int(feature_dim),
        "split_rule": "chronological 70/15/15",
        "split_counts": {k: int(len(v)) for k, v in time_splits.items()},
        "target_source": "standard monitoring-station label only",
        "dense_ss_label_used": False,
        "grid_mapping": "row=grid_id//32, col=grid_id%32",
    }
    with open(save_dir / "preprocess_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(conc_img_full[0], cmap="viridis")
    ax.scatter(
        station_positions[:, 1],
        station_positions[:, 0],
        color="red",
        s=14,
        label="Monitoring stations",
    )
    ax.set_title(f"{pollutant.upper()} sparse observations")
    ax.legend(loc="best")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(
        Path(config.OUTPUT_DIR, "visualization", f"{pollutant}_station_vis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    logging.info(
        "%s saved: train=%d, val=%d, test=%d, features=%d",
        pollutant.upper(),
        len(time_splits["train"]),
        len(time_splits["val"]),
        len(time_splits["test"]),
        feature_dim,
    )
    del conc_img_raw, conc_img_full, cov_img_full, train_features
    gc.collect()


def main() -> None:
    config = Config()
    check_all_paths(config)
    station_indices, station_positions, global_mask, valid_grid_mask = (
        load_station_info(config)
    )
    for pollutant in config.POLLUTANTS:
        process_pollutant_data_loocv(
            config,
            pollutant,
            station_indices,
            station_positions,
            global_mask,
            valid_grid_mask,
        )
    logging.info("All preprocessing outputs saved to %s", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
