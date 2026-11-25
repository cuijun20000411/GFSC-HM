# Experiments

D3 denotes GFSC-UNet-RM and D3HM denotes GFSC-UNet-HM in the archived filenames and outputs. The original model code is kept together under `code/` so its imports and file fingerprint checks resolve.

| Experiment | Entry points |
|---|---|
| GFSC-UNet-HM strict LOSO, spatial-block holdout and field visualization | [gfsc_unet_d3_hybrid_mask_main.py](code/gfsc_unet_d3_hybrid_mask_main.py) |
| Strict LOSO baselines | [baseline_final_strict.py](code/baseline_final_strict.py), [pconv_covariate_final_strict.py](code/pconv_covariate_final_strict.py) |
| Model evolution and component ablations | [f3_direct_ablation_final_strict.py](code/f3_direct_ablation_final_strict.py), [d3_direct_ablation_final_strict.py](code/d3_direct_ablation_final_strict.py) |
| Spatial-block baselines and fusion ablations | [d3hm_spatial_block_classical_baselines.py](code/d3hm_spatial_block_classical_baselines.py), [gfsc_unet_d3hm_a10_spatial_block.py](code/gfsc_unet_d3hm_a10_spatial_block.py), [gfsc_unet_hm_spatial_structure_ablation.py](code/gfsc_unet_hm_spatial_structure_ablation.py) |
| HM and RM multi-seed experiment launchers | [HM launcher](runners/01_d3hm_multiseed_spatial_block.py), [RM launcher](runners/run_rm_multiseed_spatial.py) |
| Test-time station reduction | [d3hm_density_stress_final_strict.py](code/d3hm_density_stress_final_strict.py), [density_stress_final_strict.py](code/density_stress_final_strict.py) |
| GPU forward-pass benchmark | [03_d3hm_complexity_efficiency.py](../tools/benchmark/03_d3hm_complexity_efficiency.py) |

## Prepare the model inputs

Open the repository root in Bash on Linux or WSL. Use Python 3.10 or later and install the packages in [requirements.txt](../requirements.txt) in your experiment environment.

```bash
python -m pip install -r requirements.txt
```

The AirFusion data are described by [Xu et al.](https://doi.org/10.1016/j.ipm.2025.104474). The source arrays and prepared inputs are obtained separately from the code release. Organize the source files as shown below. The PM2.5 source folder is named `pm2_5`, while its prepared folder is named `pm25`.

```text
AirFusion/
├── grid_label.csv
├── no2/
│   ├── no2_label.npy
│   └── no2_features.npy
├── o3/
│   ├── o3_label.npy
│   └── o3_features.npy
└── pm2_5/
    ├── pm2_5_label.npy
    └── pm2_5_features.npy
```

Replace `/absolute/path/to/AirFusion` with the directory that contains `grid_label.csv`. Run the [paper preprocessing script](../tools/preprocessing/zh_final_paper.py).

```bash
repo="$(pwd -P)"
export SCUNET_RAW_ROOT="/absolute/path/to/AirFusion"
export SCUNET_PREPROCESS_ROOT="$repo/experiments/data/airfusion_prepared"

SCUNET_POLLUTANTS=no2,o3,pm25 python "$repo/tools/preprocessing/zh_final_paper.py"
```

The script writes the three pollutant folders under `$SCUNET_PREPROCESS_ROOT/data/`. It reads the sparse station labels and feature arrays. See [Model input data](../protocol/DATA_FORMAT.md) for the output files and array dimensions.

## Main model

After preprocessing completes, set the model input directory in the same shell.

```bash
data_root="$SCUNET_PREPROCESS_ROOT/data"
```

If you already have the prepared files, set `repo="$(pwd -P)"` from the repository root and set `data_root` to the absolute path of the directory containing the `no2`, `o3` and `pm25` subdirectories.

Check the inputs and fixed spatial partition before training.

```bash
python "$repo/experiments/code/gfsc_unet_d3_hybrid_mask_main.py" \
  --mode spatial-block \
  --data-root "$data_root" \
  --d3-reference-root "$repo/protocol/reference/D3" \
  --spatial-split-json "$repo/protocol/spatial_split.json" \
  --output "$repo/experiments/results/runs/hm_seed42" \
  --preflight
```

Once the preflight passes, remove `--preflight` to train. Set `--mode loso` to run leave-one-station-out evaluation. The structural ablation runner accepts the same fixed split via `--spatial-split-json`. The [optional requirements](../tools/requirements-optional.txt) include PyKrige for ordinary kriging.

## Matched seeds

Run the original RM block training on Linux or WSL because it uses `fcntl` file locks. The launchers start their child processes in the model source directory. Use the absolute `repo` and `data_root` paths from the preceding sections.

```bash
python "$repo/experiments/runners/01_d3hm_multiseed_spatial_block.py" \
  --main-script "$repo/experiments/code/gfsc_unet_d3_hybrid_mask_main.py" \
  --data-root "$data_root" \
  --d3-reference-root "$repo/protocol/reference/D3" \
  --spatial-split-json "$repo/protocol/spatial_split.json" \
  --reference-seed-root "$repo/experiments/results/spatial_block/hm_seed42" \
  --output "$repo/experiments/results/runs/hm_multiseed" \
  --preflight

python "$repo/experiments/runners/run_rm_multiseed_spatial.py" \
  --script "$repo/experiments/code/d3_spatial_block_final_strict.py" \
  --data-root "$data_root" \
  --output-root "$repo/experiments/results/runs/rm_multiseed" \
  --preflight-only
```

The HM launcher uses the packaged seed-42 reference results. The RM launcher checks the original source hashes and then starts its model-specific preflight. Remove the respective preflight flag to run seeds 43 and 44.

## Reading the baselines and timing

The manuscript's PConv-UNet-Cov records are in `results/loso/baselines/pconv_covariate/` under this directory. The standard-baseline results also retain an earlier PartialConvUNet run. For the benchmark, pass the local main script with `--d3hm-main` and supply model and checkpoint paths with `--baseline`. The archived timing settings and measurements are in `results/efficiency/`.
