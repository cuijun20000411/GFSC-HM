"""GFSC-UNet D3最终主程序。

默认读取既有严格LOSO预测生成定量图，并为每种污染物训练/复用一个全站D3模型，
输出仅用于空间形态展示的完整浓度场。该文件不依赖任何消融实验脚本。

模式：
  diagnostics  读取既有D3留出站预测，生成散点图和逐站误差图，不训练。
  full-field   训练/复用3个全站D3模型，保存完整场与代表性场图。
  all          依次执行上述两项（默认），不会重跑55折LOSO。
  loso         从头复现D3严格LOSO；仅用于独立复现，不是画图所必需。
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.cuda.amp import GradScaler, autocast

import rh4_final_strict_light as core
import fusion_final_strict as fusion


POLLUTANT_LABELS = {"no2": r"NO$_2$", "o3": r"O$_3$", "pm25": r"PM$_{2.5}$"}
GLOBAL_SEED = 42
DEVICE = core.DEVICE
USE_AMP = core.USE_AMP


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def stable_hash(payload: dict) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) < 2 or np.var(y_true) <= 1e-12:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": safe_r2(y_true, y_pred),
    }


class D3Config(fusion.Config):
    """固定后的论文主模型配置；不再从消融选择器读取结构。"""

    MODEL_VERSION = "gfsc_unet_d3_final_main_v1"
    USE_SCB_ATTENTION = False
    GROUP_WIDTH = 8
    GROUP_DROPOUT = 0.25
    GATE_BIAS = -3.0
    IMPORTANCE_BETA = 0.0
    START_MASK_RATIO = 0.05
    FINAL_MASK_RATIO = 0.25
    CURRICULUM_EPOCH = 50
    FIXED_NEIGHBOR_RADIUS = 2
    HUBER_WEIGHT = 0.8
    MSE_WEIGHT = 0.2
    DIFFUSION_COEFF = 0.1
    TV_WEIGHTS = {"no2": 0.01, "o3": 0.10, "pm25": 0.05}
    LAPLACIAN_WEIGHTS = {"no2": 0.01, "o3": 0.01, "pm25": 0.01}
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 150
    PATIENCE = 15
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    LOG_INTERVAL = 10
    REUSE_CHECKPOINTS = True


class GFSCUNetD3(fusion.FusionSCUNet):
    """最终D3：分组协变量编码、多尺度门控残差融合、无SCB通道注意力。"""


def d3_variant(config: D3Config) -> fusion.FusionVariant:
    return fusion.FusionVariant(
        code="D3",
        description="Final GFSC-UNet without SCB channel attention",
        use_covariates=True,
        bottleneck_fusion=True,
        decoder_fusion=True,
        group_dropout=float(config.GROUP_DROPOUT),
        ablate_group=None,
    )


def resolve_reference_root(path: str | Path) -> Path:
    root = Path(path)
    return root if root.name == "D3" else root / "D3"


def configure(args: argparse.Namespace) -> D3Config:
    config = D3Config()
    config.DATA_ROOT_DIR = args.data_root
    config.EPOCHS = args.epochs
    config.PATIENCE = args.patience
    config.BATCH_SIZE = args.batch_size
    config.NUM_WORKERS = args.num_workers
    config.LOG_INTERVAL = args.log_interval
    config.REUSE_CHECKPOINTS = True
    core.GLOBAL_SEED = GLOBAL_SEED
    fusion.GLOBAL_SEED = GLOBAL_SEED
    core.Config = D3Config
    fusion.Config = D3Config
    seed_everything(GLOBAL_SEED)
    return config


def load_groups(reference_root: Path, pollutant: str, feature_dim: int, config: D3Config):
    protocol_path = reference_root / f"protocol_{pollutant}.json"
    if not protocol_path.exists():
        raise FileNotFoundError(f"缺少D3协议文件: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    observed_ablation = protocol.get("ablation", {})
    checks = {
        "variant": observed_ablation.get("code") == "D3",
        "seed": protocol.get("seed") == GLOBAL_SEED,
        "scb_channel_attention": protocol.get("scb_channel_attention") is False,
        "bottleneck_fusion": protocol.get("bottleneck_fusion") is True,
        "decoder_fusion": protocol.get("decoder_fusion") is True,
        "group_width": protocol.get("group_width") == config.GROUP_WIDTH,
        "group_dropout": np.isclose(protocol.get("group_dropout", np.nan), config.GROUP_DROPOUT),
        "importance_beta": np.isclose(protocol.get("importance_beta", np.nan), 0.0),
        "tv_weights": protocol.get("tv_weights") == config.TV_WEIGHTS,
        "laplacian_weights": protocol.get("laplacian_weights") == config.LAPLACIAN_WEIGHTS,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"既有D3协议与最终主模型不一致: {failed}")
    payload = protocol.get("resolved_groups")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{protocol_path}没有resolved_groups")
    groups = [
        fusion.GroupSpec(str(item["name"]), tuple(int(i) for i in item["indices"]), str(item["kind"]))
        for item in payload
    ]
    flat = [index for group in groups for index in group.indices]
    if sorted(flat) != list(range(feature_dim)) or len(flat) != len(set(flat)):
        raise ValueError(f"{pollutant}: 协变量分组没有恰好覆盖全部{feature_dim}个通道")
    return groups, protocol_path, protocol


def validate_data_files(data_root: Path, pollutant: str) -> None:
    folder = data_root / pollutant
    required = [
        "cov_img_full.npy", "coord_img.npy", "station_positions.npy",
        "valid_time_indices.npy", "time_splits.npy", "valid_grid_mask.npy",
    ]
    if not (folder / "conc_img_raw.npy").exists():
        required.extend(["conc_img_full.npy", "conc_scaler.pkl"])
    missing = [name for name in required if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(f"{pollutant}缺少数据文件: {missing}")
    valid = np.load(folder / "valid_grid_mask.npy")
    if valid.shape != (32, 32) or int(np.count_nonzero(valid)) != 999:
        raise ValueError(f"{pollutant}: valid_grid_mask必须对应32×32中的999个有效网格")


def raw_concentrations(folder: Path) -> np.ndarray:
    raw_path = folder / "conc_img_raw.npy"
    if raw_path.exists():
        return np.load(raw_path, mmap_mode="r")
    values = np.load(folder / "conc_img_full.npy", mmap_mode="r")
    scaler = joblib.load(folder / "conc_scaler.pkl")
    return np.asarray(values) * float(scaler.scale_[0]) + float(scaler.mean_[0])


def validate_reference_predictions(args, pollutant: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    data_folder = Path(args.data_root) / pollutant
    positions = np.load(data_folder / "station_positions.npy").astype(int)
    valid_mask = np.load(data_folder / "valid_grid_mask.npy").astype(bool)
    prediction_path = resolve_reference_root(args.d3_reference_root) / "predictions" / f"{pollutant}_test_predictions.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"缺少既有D3留出站预测: {prediction_path}")
    frame = pd.read_csv(prediction_path)
    required = {"station", "time_idx", "true", "prediction"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{prediction_path}缺少列: {sorted(required-set(frame.columns))}")
    if "variant" in frame and set(frame["variant"].astype(str)) != {"D3"}:
        raise ValueError(f"{prediction_path}并非纯D3结果")
    if frame.duplicated(["station", "time_idx"]).any():
        raise ValueError(f"{pollutant}: 存在重复站点—时刻预测")
    if not np.isfinite(frame[["true", "prediction"]].to_numpy()).all():
        raise ValueError(f"{pollutant}: 预测文件包含非有限值")
    station_ids = set(frame["station"].astype(int))
    if station_ids != set(range(len(positions))):
        raise ValueError(f"{pollutant}: D3预测未覆盖全部{len(positions)}个站点")

    original_times = np.load(data_folder / "valid_time_indices.npy").astype(int)
    splits = np.load(data_folder / "time_splits.npy", allow_pickle=True).item()
    test_array_ids = np.asarray(splits["test"], dtype=int)
    raw = raw_concentrations(data_folder)
    expected = []
    for station, (row, col) in enumerate(positions):
        for array_idx in test_array_ids:
            truth = float(raw[array_idx, row, col])
            if np.isfinite(truth):
                expected.append((station, int(original_times[array_idx]), truth))
    expected_df = pd.DataFrame(expected, columns=["station", "time_idx", "expected_true"])
    merged = expected_df.merge(frame, on=["station", "time_idx"], how="outer", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError(f"{pollutant}: D3预测与当前测试站点—时刻集合不一致")
    if not np.allclose(merged["expected_true"], merged["true"], rtol=2e-6, atol=2e-5):
        raise ValueError(f"{pollutant}: 预测文件中的真值与当前数据不一致")
    frame = frame.copy()
    frame["station"] = frame["station"].astype(int)
    frame["error"] = frame["prediction"] - frame["true"]
    return frame, positions, valid_mask


def setup_figure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "savefig.facecolor": "white",
        "axes.unicode_minus": False,
    })


def save_figure(fig, stem: Path, formats: list[str], dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(stem.with_suffix("." + fmt), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_diagnostics(args) -> None:
    setup_figure_style()
    figure_dir = Path(args.output) / "figures"
    metric_dir = Path(args.output) / "metrics"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    formats = [part.strip() for part in args.figure_formats.split(",") if part.strip()]
    payload = {}
    station_frames = []
    aggregate_rows = []
    for pollutant in args.pollutants:
        frame, positions, valid = validate_reference_predictions(args, pollutant)
        station_rows = []
        for station, part in frame.groupby("station", sort=True):
            values = metric_dict(part["true"].to_numpy(), part["prediction"].to_numpy())
            row, col = positions[int(station)]
            station_rows.append({"pollutant": pollutant, "station": int(station), "row": int(row), "col": int(col), "n": len(part), **values})
        station_df = pd.DataFrame(station_rows)
        station_frames.append(station_df)
        values = metric_dict(frame["true"].to_numpy(), frame["prediction"].to_numpy())
        aggregate_rows.append({"pollutant": pollutant, "n_stations": len(station_df), "n_test_observations": len(frame), **values})
        payload[pollutant] = {"samples": frame, "stations": station_df, "positions": positions, "valid": valid}
    pd.concat(station_frames, ignore_index=True).to_csv(metric_dir / "D3_station_metrics_for_figures.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(metric_dir / "D3_aggregate_metrics_recomputed.csv", index=False)

    fig, axes = plt.subplots(1, len(args.pollutants), figsize=(3.65 * len(args.pollutants), 3.35), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for label_index, (ax, pollutant) in enumerate(zip(axes, args.pollutants)):
        frame = payload[pollutant]["samples"]
        true = frame["true"].to_numpy(); prediction = frame["prediction"].to_numpy()
        limits = [float(min(true.min(), prediction.min())), float(max(true.max(), prediction.max()))]
        margin = 0.03 * max(limits[1] - limits[0], 1.0)
        limits = [limits[0] - margin, limits[1] + margin]
        ax.scatter(true, prediction, s=7, alpha=0.22, color="#315f8c", edgecolors="none", rasterized=True)
        ax.plot(limits, limits, color="#b04a3f", lw=1.1, ls="--")
        ax.set(xlim=limits, ylim=limits, xlabel=r"Observed concentration ($\mu$g m$^{-3}$)")
        if label_index == 0:
            ax.set_ylabel(r"Predicted concentration ($\mu$g m$^{-3}$)")
        metric = metric_dict(true, prediction)
        ax.text(0.04, 0.96, f"MAE = {metric['MAE']:.2f}\nRMSE = {metric['RMSE']:.2f}\n$R^2$ = {metric['R2']:.3f}",
                transform=ax.transAxes, va="top", ha="left", linespacing=1.35)
        ax.set_title(f"({chr(97+label_index)}) {POLLUTANT_LABELS[pollutant]}", loc="left")
        ax.grid(color="#d9d9d9", lw=0.45, alpha=0.65)
    save_figure(fig, figure_dir / "Fig_D3_observed_vs_predicted", formats, args.dpi)

    fig, axes = plt.subplots(1, len(args.pollutants), figsize=(3.55 * len(args.pollutants), 3.25), constrained_layout=True)
    axes = np.atleast_1d(axes)
    background = ListedColormap(["#f1f1ed"])
    for label_index, (ax, pollutant) in enumerate(zip(axes, args.pollutants)):
        item = payload[pollutant]; station_df = item["stations"]
        ax.imshow(np.ma.masked_where(~item["valid"], item["valid"].astype(float)), cmap=background, vmin=0, vmax=1, origin="upper")
        scatter = ax.scatter(station_df["col"], station_df["row"], c=station_df["RMSE"], cmap="magma", s=38,
                             edgecolors="black", linewidths=0.35)
        ax.set_title(f"({chr(97+label_index)}) {POLLUTANT_LABELS[pollutant]}", loc="left")
        ax.set_xlabel("Grid column")
        if label_index == 0:
            ax.set_ylabel("Grid row")
        ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.set_aspect("equal")
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.047, pad=0.03)
        colorbar.set_label(r"Station RMSE ($\mu$g m$^{-3}$)")
    save_figure(fig, figure_dir / "Fig_D3_station_error_maps", formats, args.dpi)

    fig, axes = plt.subplots(1, len(args.pollutants), figsize=(3.55 * len(args.pollutants), 3.05), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for label_index, (ax, pollutant) in enumerate(zip(axes, args.pollutants)):
        station_df = payload[pollutant]["stations"].sort_values("RMSE").reset_index(drop=True)
        ax.plot(np.arange(1, len(station_df)+1), station_df["RMSE"], color="#315f8c", lw=1.2)
        ax.scatter(np.arange(1, len(station_df)+1), station_df["RMSE"], s=10, color="#315f8c")
        ax.axhline(station_df["RMSE"].median(), color="#b04a3f", lw=0.9, ls="--", label="Median")
        ax.set_title(f"({chr(97+label_index)}) {POLLUTANT_LABELS[pollutant]}", loc="left")
        ax.set_xlabel("Stations ranked by RMSE")
        if label_index == 0:
            ax.set_ylabel(r"RMSE ($\mu$g m$^{-3}$)")
        ax.grid(axis="y", color="#d9d9d9", lw=0.45)
    save_figure(fig, figure_dir / "Fig_D3_stationwise_RMSE", formats, args.dpi)
    print(f"D3定量图已生成: {figure_dir}", flush=True)


def model_fingerprint(config, pollutant, station_idx, groups, protocol_path) -> str:
    data_folder = Path(config.DATA_ROOT_DIR) / pollutant
    input_files = [
        data_folder / name for name in (
            "conc_img_raw.npy", "conc_img_full.npy", "conc_scaler.pkl",
            "cov_img_full.npy", "coord_img.npy", "station_positions.npy",
            "valid_time_indices.npy", "time_splits.npy", "valid_grid_mask.npy",
        ) if (data_folder / name).exists()
    ]
    return stable_hash({
        "model_version": config.MODEL_VERSION,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": GLOBAL_SEED,
        "pollutant": pollutant,
        "station_idx": station_idx,
        "groups": [asdict(group) for group in groups],
        "source_protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "data_files": {item.name: {"size": item.stat().st_size, "mtime_ns": item.stat().st_mtime_ns} for item in input_files},
        "epochs": config.EPOCHS,
        "patience": config.PATIENCE,
        "batch_size": config.BATCH_SIZE,
        "lr": config.LR,
        "weight_decay": config.WEIGHT_DECAY,
        "masking": [config.START_MASK_RATIO, config.FINAL_MASK_RATIO, config.CURRICULUM_EPOCH],
        "tv": config.TV_WEIGHTS,
        "laplacian": config.LAPLACIAN_WEIGHTS,
    })


def train_or_load(config, pollutant, station_idx, groups, protocol_path, checkpoint):
    variant = d3_variant(config)
    train_dataset = fusion.FusionDataset(config, pollutant, station_idx, mode="train", variant=variant)
    val_dataset = fusion.FusionDataset(config, pollutant, station_idx, mode="val", variant=variant)
    fingerprint = model_fingerprint(config, pollutant, station_idx, groups, protocol_path)
    marker_path = Path(str(checkpoint) + ".complete.json")
    seed = fusion.fold_seed(pollutant, station_idx)
    seed_everything(seed)
    model = GFSCUNetD3(train_dataset.base_channels, train_dataset.F_dim, groups, variant, config).to(DEVICE)
    if checkpoint.exists() and marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") == "completed" and marker.get("fingerprint") == fingerprint:
            model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
            print(f"复用D3检查点: {pollutant}, fold={station_idx}", flush=True)
            return model.eval(), train_dataset, marker
        raise RuntimeError(f"检查点配置与当前主程序不同，请更换输出目录: {checkpoint}")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    train_loader = core.make_loader(train_dataset, config, shuffle=True)
    val_loader = core.make_loader(val_dataset, config, shuffle=False)
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.LR,
                                               steps_per_epoch=max(1, len(train_loader)), epochs=config.EPOCHS)
    scaler = GradScaler(enabled=USE_AMP)
    best_loss = float("inf"); best_epoch = -1; waiting = 0; started = time.time()
    for epoch in range(config.EPOCHS):
        ratio = config.START_MASK_RATIO if epoch < config.CURRICULUM_EPOCH else config.FINAL_MASK_RATIO
        train_dataset.set_mask_ratio(ratio); val_dataset.set_mask_ratio(config.FINAL_MASK_RATIO)
        model.train(); train_losses = []
        for batch in train_loader:
            inputs, observed, known, targets, valid = core.move_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=USE_AMP):
                prediction = model(inputs, observed, known)
                loss, _ = fusion.fusion_loss(config, pollutant, prediction, targets, valid)
            if torch.isfinite(loss):
                train_losses.append(float(loss.item()))
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            scheduler.step()
        model.eval(); validation_losses = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, observed, known, targets, valid = core.move_batch(batch)
                prediction = model(inputs, observed, known)
                loss, _ = fusion.fusion_loss(config, pollutant, prediction, targets, valid)
                if torch.isfinite(loss): validation_losses.append(float(loss.item()))
        current = float(np.mean(validation_losses)) if validation_losses else float("inf")
        if current < best_loss:
            best_loss = current; best_epoch = epoch + 1; waiting = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            waiting += 1
        if epoch == 0 or (epoch + 1) % config.LOG_INTERVAL == 0:
            train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
            print(f"  {pollutant.upper()} fold={station_idx} epoch={epoch+1:03d} train={train_mean:.6f} val={current:.6f} best={best_loss:.6f}", flush=True)
        if waiting >= config.PATIENCE: break
    if not checkpoint.exists(): raise RuntimeError("训练未产生有限的验证检查点")
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    marker = {"status": "completed", "fingerprint": fingerprint, "model_version": config.MODEL_VERSION,
              "pollutant": pollutant, "station_idx": station_idx, "seed": GLOBAL_SEED, "fold_seed": seed,
              "best_epoch": best_epoch, "best_validation_loss": best_loss,
              "train_seconds": time.time()-started, "parameter_count": sum(p.numel() for p in model.parameters())}
    atomic_json(marker_path, marker)
    return model.eval(), train_dataset, marker


def evaluate_loso(config, pollutant, station_idx, model):
    dataset = fusion.FusionDataset(config, pollutant, station_idx, mode="test", variant=d3_variant(config))
    loader = core.make_loader(dataset, config, shuffle=False, batch_size=1)
    row, col = dataset.station_positions[station_idx]
    records = []
    with torch.no_grad():
        for batch in loader:
            inputs, observed, known, target, _ = core.move_batch(batch)
            prediction = model(inputs, observed, known)
            true_z = float(target[0, 0, row, col].cpu()); pred_z = float(prediction[0, 0, row, col].cpu())
            if np.isfinite(true_z) and np.isfinite(pred_z):
                true = true_z*dataset.fold_std+dataset.fold_mean; pred = pred_z*dataset.fold_std+dataset.fold_mean
                records.append({"variant":"D3", "pollutant":pollutant, "station":station_idx,
                                "time_idx":int(batch["time_idx"].item()), "true":true, "prediction":pred, "error":pred-true})
    frame = pd.DataFrame(records)
    if frame.empty: raise RuntimeError(f"{pollutant} station {station_idx}: 无有效测试样本")
    return frame, {"variant":"D3", "pollutant":pollutant, "station":station_idx, "n_test":len(frame),
                   **metric_dict(frame.true, frame.prediction), "fold_train_mean":dataset.fold_mean, "fold_train_std":dataset.fold_std}


def run_loso(args, config):
    root = Path(args.output) / "loso" / "D3"
    for pollutant in args.pollutants:
        probe = fusion.FusionDataset(config, pollutant, 0, mode="test", variant=d3_variant(config))
        groups, protocol_path, _ = load_groups(resolve_reference_root(args.d3_reference_root), pollutant, probe.F_dim, config)
        total = probe.total_stations; del probe
        stop = total if args.max_folds <= 0 else min(total, args.fold_start + args.max_folds)
        fold_dir = root / "folds"; fold_dir.mkdir(parents=True, exist_ok=True)
        for station_idx in range(args.fold_start, stop):
            checkpoint = root / "models" / f"{pollutant}_station_{station_idx:02d}_D3.pth"
            model, _, info = train_or_load(config, pollutant, station_idx, groups, protocol_path, checkpoint)
            frame, summary = evaluate_loso(config, pollutant, station_idx, model)
            summary.update({k:v for k,v in info.items() if k in {"best_epoch","best_validation_loss","train_seconds","parameter_count"}})
            frame.to_csv(fold_dir / f"{pollutant}_station_{station_idx:02d}_predictions.csv", index=False)
            atomic_json(fold_dir / f"{pollutant}_station_{station_idx:02d}_metrics.json", summary)
            del model; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        # 从磁盘重新收集所有与当前配置相容的已完成折，支持分段运行和恢复。
        pollutant_frames = []; pollutant_rows = []
        for station_idx in range(total):
            checkpoint = root / "models" / f"{pollutant}_station_{station_idx:02d}_D3.pth"
            marker_path = Path(str(checkpoint) + ".complete.json")
            prediction_path = fold_dir / f"{pollutant}_station_{station_idx:02d}_predictions.csv"
            metric_path = fold_dir / f"{pollutant}_station_{station_idx:02d}_metrics.json"
            if not (marker_path.exists() and prediction_path.exists() and metric_path.exists()): continue
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            expected = model_fingerprint(config, pollutant, station_idx, groups, protocol_path)
            if marker.get("status") != "completed" or marker.get("fingerprint") != expected: continue
            frame = pd.read_csv(prediction_path)
            if frame.empty or set(frame["station"].astype(int)) != {station_idx}: continue
            pollutant_frames.append(frame)
            pollutant_rows.append(json.loads(metric_path.read_text(encoding="utf-8")))
        if pollutant_frames:
            prediction_df = pd.concat(pollutant_frames, ignore_index=True); station_df = pd.DataFrame(pollutant_rows)
            (root/"predictions").mkdir(parents=True, exist_ok=True); (root/"metrics").mkdir(parents=True, exist_ok=True)
            prediction_df.to_csv(root/"predictions"/f"{pollutant}_test_predictions.csv", index=False)
            station_df.to_csv(root/"metrics"/f"{pollutant}_station_metrics.csv", index=False)
            overall = metric_dict(prediction_df.true, prediction_df.prediction)
            pd.DataFrame([{ "variant":"D3", "pollutant":pollutant, "n_stations":station_df.station.nunique(),
                            "n_test_observations":len(prediction_df), **overall,
                            "mean_station_MAE":station_df.MAE.mean(), "mean_station_RMSE":station_df.RMSE.mean(),
                            "mean_station_R2":station_df.R2.mean(), "median_station_R2":station_df.R2.median()}]).to_csv(
                                root/"metrics"/f"{pollutant}_aggregate_metrics.csv", index=False)
    print(f"独立D3 LOSO输出: {root}", flush=True)


def reconstruct_full_fields(args, config):
    root = Path(args.output); result_dir = root / "full_field"; model_dir = root / "models"
    result_dir.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for pollutant in args.pollutants:
        probe = fusion.FusionDataset(config, pollutant, -1, mode="test", variant=d3_variant(config))
        groups, protocol_path, protocol = load_groups(resolve_reference_root(args.d3_reference_root), pollutant, probe.F_dim, config)
        del probe
        checkpoint = model_dir / f"{pollutant}_full_region_D3.pth"
        model, _, training_info = train_or_load(config, pollutant, -1, groups, protocol_path, checkpoint)
        dataset = fusion.FusionDataset(config, pollutant, -1, mode="test", variant=d3_variant(config))
        loader = core.make_loader(dataset, config, shuffle=False, batch_size=1)
        predictions=[]; observations=[]; times=[]; preservation=[]
        with torch.no_grad():
            for batch in loader:
                inputs, observed, known, _, _ = core.move_batch(batch)
                output = model(inputs, observed, known)
                difference = torch.abs(output[observed.bool()] - known[observed.bool()])
                preservation.append(float(difference.max().cpu()) if difference.numel() else 0.0)
                field = output[0,0].cpu().numpy()*dataset.fold_std+dataset.fold_mean
                known_raw = known[0,0].cpu().numpy()*dataset.fold_std+dataset.fold_mean
                visible = observed[0,0].cpu().numpy().astype(bool)
                field[dataset.valid_grid_mask<=0]=np.nan; known_raw[~visible]=np.nan; known_raw[dataset.valid_grid_mask<=0]=np.nan
                predictions.append(field.astype(np.float32)); observations.append(known_raw.astype(np.float32)); times.append(int(batch["time_idx"].item()))
        if max(preservation, default=0.0) > 1e-5: raise RuntimeError(f"{pollutant}: 可见站点硬保持检查失败")
        predictions=np.stack(predictions); observations=np.stack(observations); times=np.asarray(times,dtype=np.int64)
        archive=result_dir/f"{pollutant}_D3_full_region_test_fields.npz"
        np.savez_compressed(archive, predictions=predictions, known_observations=observations,
                            original_time_indices=times, station_positions=dataset.station_positions,
                            valid_grid_mask=dataset.valid_grid_mask, train_mean=dataset.fold_mean, train_std=dataset.fold_std)
        station_means=np.nanmean(observations,axis=(1,2)); target=float(np.nanquantile(station_means,args.representative_quantile))
        selected=int(np.nanargmin(np.abs(station_means-target)))
        records.append({"pollutant":pollutant,"array_index_in_test_output":selected,"original_time_idx":int(times[selected]),
                        "selection_rule":f"visible-station mean nearest quantile {args.representative_quantile}",
                        "visible_station_mean":float(station_means[selected]),"archive":str(archive),
                        "checkpoint":str(checkpoint),"best_epoch":training_info.get("best_epoch"),
                        "best_validation_loss":training_info.get("best_validation_loss")})
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    pd.DataFrame(records).to_csv(result_dir/"representative_times.csv",index=False)
    plot_representative_fields(args, records)


def plot_representative_fields(args, records):
    setup_figure_style(); formats=[x.strip() for x in args.figure_formats.split(",") if x.strip()]
    fig,axes=plt.subplots(1,len(records),figsize=(3.7*len(records),3.35),constrained_layout=True);axes=np.atleast_1d(axes)
    for index,(ax,record) in enumerate(zip(axes,records)):
        data=np.load(record["archive"]);selected=int(record["array_index_in_test_output"])
        field=data["predictions"][selected];known=data["known_observations"][selected];positions=data["station_positions"]
        finite=field[np.isfinite(field)];vmin=float(finite.min());vmax=float(finite.max())
        image=ax.imshow(field,cmap="viridis",vmin=vmin,vmax=vmax,origin="upper")
        rr,cc=positions.T;station_values=known[rr,cc]
        ax.scatter(cc,rr,c=station_values,cmap="viridis",vmin=vmin,vmax=vmax,s=20,edgecolors="white",linewidths=.35)
        pollutant=record["pollutant"]
        ax.set_title(f"({chr(97+index)}) {POLLUTANT_LABELS[pollutant]}, time {record['original_time_idx']}",loc="left")
        ax.set_xlabel("Grid column")
        if index==0:ax.set_ylabel("Grid row")
        ax.set_xlim(-.5,31.5);ax.set_ylim(31.5,-.5);ax.set_aspect("equal")
        bar=fig.colorbar(image,ax=ax,fraction=.047,pad=.03);bar.set_label(r"Concentration ($\mu$g m$^{-3}$)")
    save_figure(fig,Path(args.output)/"figures"/"Fig_D3_representative_fields",formats,args.dpi)
    print(f"完整浓度场与代表性场图已生成: {Path(args.output)/'full_field'}",flush=True)


def write_run_protocol(args, config):
    payload={"program":"GFSC-UNet D3 final main","model_version":config.MODEL_VERSION,"mode":args.mode,
             "seed":GLOBAL_SEED,"device":str(DEVICE),"data_root":str(Path(args.data_root).resolve()),
             "D3_reference_root":str(resolve_reference_root(args.d3_reference_root).resolve()),"output":str(Path(args.output).resolve()),
             "pollutants":args.pollutants,"architecture":{"SCB_channel_attention":False,"group_width":8,
             "group_dropout":.25,"bottleneck_gated_adapter":True,"decoder_gated_adapters":True},
             "training":{"epochs":config.EPOCHS,"patience":config.PATIENCE,"batch_size":config.BATCH_SIZE,
             "learning_rate":config.LR,"weight_decay":config.WEIGHT_DECAY,"mask_ratio":[.05,.25],"switch_epoch":50},
             "spatial_weights":{"TV":config.TV_WEIGHTS,"Laplacian":config.LAPLACIAN_WEIGHTS},
             "interpretation":{"LOSO_diagnostics":"quantitative held-out-station evidence",
             "full_field":"qualitative spatial reconstruction; unmonitored grids have no reference truth"}}
    atomic_json(Path(args.output)/"run_protocol.json",payload)


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["diagnostics","full-field","all","loso"],default="all")
    parser.add_argument("--data-root",default="/etc/cj/boundary/duoyuan/data")
    parser.add_argument("--d3-reference-root",default="/etc/cj/boundary/f3_direct_ablation_final_strict_v1/D3")
    parser.add_argument("--output",default="/etc/cj/boundary/gfsc_unet_d3_main_v1")
    parser.add_argument("--pollutants",default="no2,o3,pm25")
    parser.add_argument("--epochs",type=int,default=150);parser.add_argument("--patience",type=int,default=15)
    parser.add_argument("--batch-size",type=int,default=32);parser.add_argument("--num-workers",type=int,default=4)
    parser.add_argument("--log-interval",type=int,default=10);parser.add_argument("--fold-start",type=int,default=0)
    parser.add_argument("--max-folds",type=int,default=0)
    parser.add_argument("--representative-quantile",type=float,default=.5)
    parser.add_argument("--figure-formats",default="png,pdf,svg");parser.add_argument("--dpi",type=int,default=400)
    args=parser.parse_args();args.pollutants=[x.strip().lower() for x in args.pollutants.split(",") if x.strip()]
    if not args.pollutants or set(args.pollutants)-set(POLLUTANT_LABELS):raise ValueError("pollutants只能从no2,o3,pm25选择")
    if not 0<=args.representative_quantile<=1:raise ValueError("representative-quantile必须在0到1之间")
    if min(args.epochs,args.patience,args.batch_size,args.dpi)<=0 or args.num_workers<0:raise ValueError("训练/输出参数无效")
    return args


def main():
    args=parse_args();config=configure(args);Path(args.output).mkdir(parents=True,exist_ok=True)
    for pollutant in args.pollutants:
        validate_data_files(Path(args.data_root),pollutant)
        probe=fusion.FusionDataset(config,pollutant,0,mode="test",variant=d3_variant(config))
        load_groups(resolve_reference_root(args.d3_reference_root),pollutant,probe.F_dim,config);del probe
    write_run_protocol(args,config)
    print(f"device={DEVICE} | mode={args.mode} | pollutants={args.pollutants}",flush=True)
    if args.mode in {"diagnostics","all"}:build_diagnostics(args)
    if args.mode in {"full-field","all"}:reconstruct_full_fields(args,config)
    if args.mode=="loso":run_loso(args,config)


if __name__=="__main__":
    main()
