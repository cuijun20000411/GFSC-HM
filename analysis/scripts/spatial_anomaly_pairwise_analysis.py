#!/usr/bin/env python3
"""
独立的空间异常技能与站点对浓度差分析。

本脚本只读取严格LOSO测试预测和站点坐标，不训练模型，也不读取checkpoint。

输出：
1. spatial_anomaly_summary.csv / table_spatial_anomaly.tex
2. pairwise_contrast_summary.csv
3. table_pairwise_overall_high_contrast.tex
4. table_pairwise_distance_quartiles.tex
5. anomaly_bootstrap_replicates.csv.gz
6. pairwise_bootstrap_replicates.csv.gz
7. pair_definitions.csv
8. analysis_manifest.txt
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


POLLUTANTS = ["no2", "o3", "pm25"]
MODELS = ["SpatialMean", "IDW", "GFSC-UNet-HM"]
POLLUTANT_TEX = {
    "no2": r"$\mathrm{NO}_2$",
    "o3": r"$\mathrm{O}_3$",
    "pm25": r"$\mathrm{PM}_{2.5}$",
}


def normalise_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def canonical_pollutant(value: object) -> str:
    token = normalise_token(value)
    mapping = {"no2": "no2", "o3": "o3", "pm25": "pm25", "pm25ugm3": "pm25"}
    if token not in mapping:
        raise ValueError(f"无法识别污染物名称：{value!r}")
    return mapping[token]


def canonical_model(value: object) -> str:
    token = normalise_token(value)
    mapping = {
        "d3hm": "GFSC-UNet-HM",
        "gfscunethm": "GFSC-UNet-HM",
        "hybridmask": "GFSC-UNet-HM",
        "idw": "IDW",
        "spatialmean": "SpatialMean",
    }
    return mapping.get(token, str(value))


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    raise ValueError(f"缺少字段；候选={list(candidates)}，实际={list(columns)}")


def standardise_prediction_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    pollutant_col = first_existing(df.columns, ["pollutant", "pollutant_name"])
    station_col = first_existing(df.columns, ["station", "station_id", "site", "site_id"])
    time_col = first_existing(df.columns, ["time_idx", "timestamp", "datetime", "time"])
    true_col = first_existing(df.columns, ["true", "y_true", "observed", "target"])
    pred_col = first_existing(df.columns, ["prediction", "y_pred", "predicted", "pred"])
    model_col = first_existing(df.columns, ["model", "variant", "method"])

    if "hide_ratio" in df.columns:
        df = df.loc[np.isclose(pd.to_numeric(df["hide_ratio"]), 0.0)].copy()
    if "repeat" in df.columns:
        repeats = pd.to_numeric(df["repeat"], errors="coerce")
        if (repeats == 0).any():
            df = df.loc[repeats == 0].copy()

    result = pd.DataFrame(
        {
            "pollutant": df[pollutant_col].map(canonical_pollutant),
            "station": pd.to_numeric(df[station_col], errors="raise").astype(int),
            "time": df[time_col],
            "model": df[model_col].map(canonical_model),
            "true": pd.to_numeric(df[true_col], errors="raise").astype(float),
            "prediction": pd.to_numeric(df[pred_col], errors="raise").astype(float),
        }
    )
    numeric_time = pd.to_numeric(result["time"], errors="coerce")
    if numeric_time.notna().all():
        result["time"] = numeric_time.astype(int)
    else:
        result["time"] = pd.to_datetime(result["time"], errors="raise")
    return result


def load_loso_predictions(path: Path) -> pd.DataFrame:
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.glob("*predictions*.csv"))
        if not files:
            files = sorted(path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"{path} 下没有预测CSV")

    frames = [standardise_prediction_file(file) for file in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.loc[df["model"].isin(MODELS)].copy()
    found = set(df["model"].unique())
    missing = set(MODELS) - found
    if missing:
        raise ValueError(f"LOSO预测缺少模型：{sorted(missing)}；现有={sorted(found)}")

    key = ["pollutant", "model", "station", "time"]
    duplicated = df.duplicated(key, keep=False)
    if duplicated.any():
        check = df.loc[duplicated].groupby(key).agg(
            n_true=("true", "nunique"), n_prediction=("prediction", "nunique")
        )
        if (check[["n_true", "n_prediction"]] > 1).any().any():
            raise ValueError("预测文件中存在键相同但数值冲突的记录")
        df = df.drop_duplicates(key)

    for pollutant, part in df.groupby("pollutant"):
        references: pd.Series | None = None
        reference_model = ""
        for model, current in part.groupby("model"):
            values = current.set_index(["time", "station"])["true"].sort_index()
            if references is None:
                references = values
                reference_model = model
            else:
                if not values.index.equals(references.index):
                    raise ValueError(f"{pollutant}: {model}与{reference_model}测试键不同")
                if not np.allclose(values.to_numpy(), references.to_numpy(), atol=1e-6, rtol=0):
                    raise ValueError(f"{pollutant}: {model}与{reference_model}真实值不同")

    return df.sort_values(["pollutant", "model", "time", "station"]).reset_index(drop=True)


def load_coordinates(path: Path, stations: list[int]) -> pd.DataFrame:
    df = pd.read_csv(path)
    station_col = first_existing(df.columns, ["station", "station_id", "site", "site_id"])
    if {"x", "y"}.issubset(df.columns):
        x_col, y_col = "x", "y"
    elif {"grid_col", "grid_row"}.issubset(df.columns):
        x_col, y_col = "grid_col", "grid_row"
    elif {"lon", "lat"}.issubset(df.columns):
        x_col, y_col = "lon", "lat"
    elif {"longitude", "latitude"}.issubset(df.columns):
        x_col, y_col = "longitude", "latitude"
    else:
        raise ValueError("站点坐标文件需要x/y、grid_col/grid_row或经纬度字段")

    coords = pd.DataFrame(
        {
            "station": pd.to_numeric(df[station_col], errors="raise").astype(int),
            "x": pd.to_numeric(df[x_col], errors="raise").astype(float),
            "y": pd.to_numeric(df[y_col], errors="raise").astype(float),
        }
    ).drop_duplicates("station")
    missing = sorted(set(stations) - set(coords["station"]))
    if missing:
        raise ValueError(f"坐标文件缺少站点：{missing}")
    return coords.set_index("station").loc[stations].reset_index()


def circular_moving_block_indices(
    n_times: int, block_length: int, n_bootstrap: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n_times / block_length)
    offsets = np.arange(block_length)
    result = []
    for _ in range(n_bootstrap):
        starts = rng.integers(0, n_times, size=n_blocks)
        sampled = np.concatenate([(start + offsets) % n_times for start in starts])
        result.append(sampled[:n_times])
    return result


def percentile_ci(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return np.nan, np.nan
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def pearson_from_sums(stats: np.ndarray) -> float:
    n, sx, sy, sxx, syy, sxy, _ = stats
    if n < 3:
        return np.nan
    var_x = sxx - sx * sx / n
    var_y = syy - sy * sy / n
    denominator = math.sqrt(max(var_x, 0.0) * max(var_y, 0.0))
    if denominator <= 1e-15:
        return np.nan
    return float((sxy - sx * sy / n) / denominator)


def aggregate_pair_stats(
    observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """逐时刻返回[n,sumx,sumy,sumxx,sumyy,sumxy,sse]。"""
    n_times = observed.shape[0]
    rows = np.zeros((n_times, 7), dtype=float)
    if mask.ndim == 1:
        mask = np.broadcast_to(mask, observed.shape)
    for index in range(n_times):
        keep = mask[index]
        x = observed[index, keep]
        y = predicted[index, keep]
        rows[index, 0] = len(x)
        if len(x) == 0:
            continue
        rows[index, 1] = x.sum()
        rows[index, 2] = y.sum()
        rows[index, 3] = np.square(x).sum()
        rows[index, 4] = np.square(y).sum()
        rows[index, 5] = (x * y).sum()
        rows[index, 6] = np.square(y - x).sum()
    return rows


def analyse_pollutant(
    part: pd.DataFrame,
    coords: pd.DataFrame,
    n_bootstrap: int,
    block_hours: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], pd.DataFrame, dict[str, object]]:
    pollutant = str(part["pollutant"].iloc[0])
    reference = part.loc[part["model"] == "SpatialMean"]
    truth = reference.pivot(index="time", columns="station", values="true").sort_index()
    if truth.isna().any().any():
        raise ValueError(f"{pollutant}: 站点-时刻真实值矩阵不完整")
    times = truth.index.to_list()
    stations = [int(value) for value in truth.columns]
    coords = coords.set_index("station").loc[stations].reset_index()
    n_times, n_stations = truth.shape
    bootstrap_indices = circular_moving_block_indices(
        n_times, block_hours, n_bootstrap, seed
    )

    observed = truth.to_numpy(float)
    observed_anomaly = observed - observed.mean(axis=1, keepdims=True)
    observed_anomaly_sse = np.square(observed_anomaly).sum(axis=1)
    anomaly_count = np.full(n_times, n_stations, dtype=float)

    predictions: dict[str, np.ndarray] = {}
    for model in ["IDW", "GFSC-UNet-HM"]:
        matrix = (
            part.loc[part["model"] == model]
            .pivot(index="time", columns="station", values="prediction")
            .reindex(index=times, columns=stations)
        )
        if matrix.isna().any().any():
            raise ValueError(f"{pollutant}/{model}: 预测矩阵不完整")
        predictions[model] = matrix.to_numpy(float)

    anomaly_rows: list[dict[str, object]] = []
    anomaly_boot_rows: list[dict[str, object]] = []
    for model in MODELS:
        if model == "SpatialMean":
            predicted_anomaly = np.zeros_like(observed_anomaly)
        else:
            prediction = predictions[model]
            predicted_anomaly = prediction - prediction.mean(axis=1, keepdims=True)
        anomaly_sse = np.square(predicted_anomaly - observed_anomaly).sum(axis=1)
        rmse = math.sqrt(float(anomaly_sse.sum() / anomaly_count.sum()))
        skill = 1.0 - float(anomaly_sse.sum() / observed_anomaly_sse.sum())

        rmse_reps: list[float] = []
        skill_reps: list[float] = []
        for replicate, indices in enumerate(bootstrap_indices):
            rmse_rep = math.sqrt(
                float(anomaly_sse[indices].sum() / anomaly_count[indices].sum())
            )
            skill_rep = 1.0 - float(
                anomaly_sse[indices].sum() / observed_anomaly_sse[indices].sum()
            )
            rmse_reps.append(rmse_rep)
            skill_reps.append(skill_rep)
            anomaly_boot_rows.append(
                {
                    "replicate": replicate,
                    "pollutant": pollutant,
                    "model": model,
                    "spatial_anomaly_rmse": rmse_rep,
                    "spatial_anomaly_skill": skill_rep,
                }
            )
        rmse_low, rmse_high = percentile_ci(rmse_reps)
        skill_low, skill_high = percentile_ci(skill_reps)
        anomaly_rows.append(
            {
                "pollutant": pollutant,
                "model": model,
                "n_stations": n_stations,
                "n_timestamps": n_times,
                "spatial_anomaly_rmse": rmse,
                "spatial_anomaly_rmse_ci_low": rmse_low,
                "spatial_anomaly_rmse_ci_high": rmse_high,
                "spatial_anomaly_skill": skill,
                "spatial_anomaly_skill_ci_low": skill_low,
                "spatial_anomaly_skill_ci_high": skill_high,
            }
        )

    x = coords["x"].to_numpy(float)
    y = coords["y"].to_numpy(float)
    pair_i, pair_j = np.triu_indices(n_stations, k=1)
    distance = np.sqrt(np.square(x[pair_i] - x[pair_j]) + np.square(y[pair_i] - y[pair_j]))
    q25, q50, q75 = np.quantile(distance, [0.25, 0.50, 0.75])
    distance_group = np.digitize(distance, [q25, q50, q75], right=True) + 1
    pair_definitions = pd.DataFrame(
        {
            "pollutant": pollutant,
            "station_i": np.asarray(stations)[pair_i],
            "station_j": np.asarray(stations)[pair_j],
            "distance": distance,
            "distance_quartile": [f"Distance Q{value}" for value in distance_group],
        }
    )

    observed_delta = observed[:, pair_i] - observed[:, pair_j]
    high_contrast_threshold = float(np.quantile(np.abs(observed_delta), 0.75))
    strata: list[tuple[str, np.ndarray]] = [("All pairs", np.ones(len(pair_i), dtype=bool))]
    for quartile in range(1, 5):
        strata.append((f"Distance Q{quartile}", distance_group == quartile))
    strata.append(("Top 25% $|\\Delta y|$", np.abs(observed_delta) >= high_contrast_threshold))

    pairwise_rows: list[dict[str, object]] = []
    pair_boot_rows: list[dict[str, object]] = []
    for model in MODELS:
        if model == "SpatialMean":
            predicted_delta = np.zeros_like(observed_delta)
        else:
            prediction = predictions[model]
            predicted_delta = prediction[:, pair_i] - prediction[:, pair_j]

        for stratum_name, mask in strata:
            per_time = aggregate_pair_stats(observed_delta, predicted_delta, mask)
            total = per_time.sum(axis=0)
            rmse = math.sqrt(float(total[6] / total[0]))
            correlation = pearson_from_sums(total)
            skill = 1.0 - float(total[6] / total[3])

            rmse_reps: list[float] = []
            correlation_reps: list[float] = []
            skill_reps: list[float] = []
            for replicate, indices in enumerate(bootstrap_indices):
                current = per_time[indices].sum(axis=0)
                rmse_rep = math.sqrt(float(current[6] / current[0]))
                correlation_rep = pearson_from_sums(current)
                skill_rep = 1.0 - float(current[6] / current[3])
                rmse_reps.append(rmse_rep)
                correlation_reps.append(correlation_rep)
                skill_reps.append(skill_rep)
                pair_boot_rows.append(
                    {
                        "replicate": replicate,
                        "pollutant": pollutant,
                        "model": model,
                        "stratum": stratum_name,
                        "pairwise_contrast_rmse": rmse_rep,
                        "pairwise_pearson_r": correlation_rep,
                        "pairwise_skill_vs_zero": skill_rep,
                    }
                )
            rmse_low, rmse_high = percentile_ci(rmse_reps)
            r_low, r_high = percentile_ci(correlation_reps)
            skill_low, skill_high = percentile_ci(skill_reps)
            pairwise_rows.append(
                {
                    "pollutant": pollutant,
                    "model": model,
                    "stratum": stratum_name,
                    "n_unique_pairs": int(mask.sum()) if mask.ndim == 1 else len(pair_i),
                    "n_pair_timestamp_records": int(total[0]),
                    "pairwise_contrast_rmse": rmse,
                    "pairwise_contrast_rmse_ci_low": rmse_low,
                    "pairwise_contrast_rmse_ci_high": rmse_high,
                    "pairwise_pearson_r": correlation,
                    "pairwise_pearson_ci_low": r_low,
                    "pairwise_pearson_ci_high": r_high,
                    "pairwise_skill_vs_zero": skill,
                    "pairwise_skill_ci_low": skill_low,
                    "pairwise_skill_ci_high": skill_high,
                }
            )

    metadata = {
        "pollutant": pollutant,
        "n_stations": n_stations,
        "n_timestamps": n_times,
        "n_unique_pairs": len(pair_i),
        "distance_q25": float(q25),
        "distance_q50": float(q50),
        "distance_q75": float(q75),
        "high_contrast_abs_delta_threshold": high_contrast_threshold,
    }
    return (
        anomaly_rows,
        anomaly_boot_rows,
        pairwise_rows,
        pair_boot_rows,
        pair_definitions,
        metadata,
    )


def tex_number(value: float, digits: int = 3) -> str:
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}"


def interval_text(point: float, low: float, high: float, digits: int = 3) -> str:
    if not np.isfinite(point):
        return "--"
    return f"{point:.{digits}f} ({low:.{digits}f}, {high:.{digits}f})"


def write_anomaly_table(summary: pd.DataFrame, path: Path, n_bootstrap: int, block_hours: int) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Spatial-anomaly reconstruction under strict LOSO-CV. At each timestamp, the cross-station mean was removed from both observations and predictions. Skill is relative to a zero-spatial-anomaly field; positive values indicate improvement over the contemporaneous spatial-mean baseline.}",
        r"\label{tab:spatial_anomaly_skill}",
        r"\small",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Pollutant & Method & Spatial-anomaly RMSE (95\% CI) & Spatial-anomaly skill (\%, 95\% CI) \\",
        r"\midrule",
    ]
    for pollutant in POLLUTANTS:
        for model in MODELS:
            row = summary.loc[
                (summary["pollutant"] == pollutant) & (summary["model"] == model)
            ].iloc[0]
            rmse = interval_text(
                row["spatial_anomaly_rmse"],
                row["spatial_anomaly_rmse_ci_low"],
                row["spatial_anomaly_rmse_ci_high"],
            )
            skill = interval_text(
                100 * row["spatial_anomaly_skill"],
                100 * row["spatial_anomaly_skill_ci_low"],
                100 * row["spatial_anomaly_skill_ci_high"],
                digits=2,
            )
            table_row = f"{POLLUTANT_TEX[pollutant]} & {model} & {rmse} & {skill} "
            lines.append(table_row + r"\\")
        lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            rf"\parbox{{0.98\textwidth}}{{\footnotesize Values in parentheses are 95\% percentile intervals from {n_bootstrap:,} paired circular moving-block bootstrap replicates with a block length of {block_hours} h. SpatialMean is represented by the zero-spatial-anomaly field; its skill is therefore zero by definition.}}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def pairwise_table_lines(
    summary: pd.DataFrame,
    strata: list[str],
    caption: str,
    label: str,
    n_bootstrap: int,
    block_hours: int,
) -> list[str]:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Pollutant & Stratum & Method & Pairwise RMSE (95\% CI) & Pearson $r$ (95\% CI) & Skill (\%, 95\% CI) \\",
        r"\midrule",
    ]
    for pollutant in POLLUTANTS:
        for stratum in strata:
            for model in MODELS:
                row = summary.loc[
                    (summary["pollutant"] == pollutant)
                    & (summary["model"] == model)
                    & (summary["stratum"] == stratum)
                ].iloc[0]
                rmse = interval_text(
                    row["pairwise_contrast_rmse"],
                    row["pairwise_contrast_rmse_ci_low"],
                    row["pairwise_contrast_rmse_ci_high"],
                )
                correlation = interval_text(
                    row["pairwise_pearson_r"],
                    row["pairwise_pearson_ci_low"],
                    row["pairwise_pearson_ci_high"],
                )
                skill = interval_text(
                    100 * row["pairwise_skill_vs_zero"],
                    100 * row["pairwise_skill_ci_low"],
                    100 * row["pairwise_skill_ci_high"],
                    digits=2,
                )
                stratum_tex = stratum.replace("Distance ", "Dist. ").replace("%", r"\%")
                lines.append(
                    f"{POLLUTANT_TEX[pollutant]} & {stratum_tex} & {model} & "
                    f"{rmse} & {correlation} & {skill} \\\\"
                )
            lines.append(r"\addlinespace")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend(
        [
            r"\end{tabular}",
            r"\vspace{2pt}",
            rf"\parbox{{0.98\textwidth}}{{\footnotesize Values in parentheses are 95\% percentile intervals from {n_bootstrap:,} paired circular moving-block bootstrap replicates with a block length of {block_hours} h. The same timestamp blocks and station pairs were used for all methods. SpatialMean has zero pairwise contrast, so its Pearson correlation is undefined and its skill is zero by definition. The target-specific leave-one-station-out means were not correlated across assembled folds because doing so induces a deterministic inverse perturbation.}}",
            r"\end{table*}",
            "",
        ]
    )
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从既有LOSO预测计算空间异常技能和站点对空间差值；不重新训练。"
    )
    parser.add_argument("--loso", type=Path, required=True, help="LOSO预测CSV或预测目录。")
    parser.add_argument("--coordinates", type=Path, required=True, help="站点坐标CSV。")
    parser.add_argument("--output", type=Path, required=True, help="输出目录。")
    parser.add_argument("--bootstrap", type=int, default=5000, help="bootstrap次数。")
    parser.add_argument("--block-hours", type=int, default=24, help="移动块长度，默认24小时。")
    parser.add_argument("--seed", type=int, default=20260923, help="随机种子。")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap < 1000:
        raise ValueError("正式结果建议至少使用1000次bootstrap")
    if args.block_hours < 2:
        raise ValueError("块长度至少为2小时")
    args.output.mkdir(parents=True, exist_ok=True)

    predictions = load_loso_predictions(args.loso)
    all_stations = sorted(int(value) for value in predictions["station"].unique())
    coordinates = load_coordinates(args.coordinates, all_stations)

    anomaly_rows: list[dict[str, object]] = []
    anomaly_boot_rows: list[dict[str, object]] = []
    pairwise_rows: list[dict[str, object]] = []
    pair_boot_rows: list[dict[str, object]] = []
    pair_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []

    for index, pollutant in enumerate(POLLUTANTS):
        part = predictions.loc[predictions["pollutant"] == pollutant].copy()
        if part.empty:
            raise ValueError(f"缺少{pollutant}预测")
        result = analyse_pollutant(
            part,
            coordinates,
            args.bootstrap,
            args.block_hours,
            args.seed + index,
        )
        anomaly_rows.extend(result[0])
        anomaly_boot_rows.extend(result[1])
        pairwise_rows.extend(result[2])
        pair_boot_rows.extend(result[3])
        pair_frames.append(result[4])
        metadata_rows.append(result[5])

    anomaly = pd.DataFrame(anomaly_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    anomaly.to_csv(args.output / "spatial_anomaly_summary.csv", index=False)
    pairwise.to_csv(args.output / "pairwise_contrast_summary.csv", index=False)
    pd.concat(pair_frames, ignore_index=True).to_csv(
        args.output / "pair_definitions.csv", index=False
    )
    pd.DataFrame(anomaly_boot_rows).to_csv(
        args.output / "anomaly_bootstrap_replicates.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(pair_boot_rows).to_csv(
        args.output / "pairwise_bootstrap_replicates.csv.gz",
        index=False,
        compression="gzip",
    )

    write_anomaly_table(
        anomaly,
        args.output / "table_spatial_anomaly.tex",
        args.bootstrap,
        args.block_hours,
    )
    overall_lines = pairwise_table_lines(
        pairwise,
        ["All pairs", "Top 25% $|\\Delta y|$"],
        "Station-pair concentration-contrast reconstruction under strict LOSO-CV. The high-contrast stratum comprises the pollutant-specific upper quartile of the absolute observed station-pair difference.",
        "tab:pairwise_contrast",
        args.bootstrap,
        args.block_hours,
    )
    (args.output / "table_pairwise_overall_high_contrast.tex").write_text(
        "\n".join(overall_lines), encoding="utf-8"
    )
    distance_lines = pairwise_table_lines(
        pairwise,
        ["Distance Q1", "Distance Q2", "Distance Q3", "Distance Q4"],
        "Distance-stratified station-pair concentration-contrast reconstruction. Distance quartiles were fixed from the 1,485 unique station pairs and applied identically to all pollutants and methods.",
        "tab:pairwise_distance",
        args.bootstrap,
        args.block_hours,
    )
    (args.output / "table_pairwise_distance_quartiles.tex").write_text(
        "\n".join(distance_lines), encoding="utf-8"
    )

    manifest = [
        "Spatial-anomaly and pairwise-contrast analysis completed.",
        f"LOSO input: {args.loso}",
        f"Coordinate input: {args.coordinates}",
        f"Bootstrap replicates: {args.bootstrap}",
        f"Circular moving-block length: {args.block_hours} h",
        "SpatialMean definition for these spatial-pattern diagnostics: zero spatial anomaly / zero pairwise contrast.",
        "Stored target-specific LOSO SpatialMean predictions are not used for cross-station correlation because they induce a deterministic inverse perturbation.",
    ]
    for item in metadata_rows:
        manifest.append(
            f"{item['pollutant']}: stations={item['n_stations']}, timestamps={item['n_timestamps']}, "
            f"pairs={item['n_unique_pairs']}, distance quartiles="
            f"({item['distance_q25']:.6f}, {item['distance_q50']:.6f}, {item['distance_q75']:.6f}), "
            f"top-25% |delta y| threshold={item['high_contrast_abs_delta_threshold']:.6f}"
        )
    (args.output / "analysis_manifest.txt").write_text(
        "\n".join(manifest) + "\n", encoding="utf-8"
    )

    print(f"完成：{args.output.resolve()}")
    print("空间异常：spatial_anomaly_summary.csv / table_spatial_anomaly.tex")
    print("站点对差值：pairwise_contrast_summary.csv / 两个LaTeX表")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
