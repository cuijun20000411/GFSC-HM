# Model input data

The training scripts use 720 aligned hours for each pollutant, with 504 training, 108 validation and 108 test samples. Each pollutant uses the same 55 candidate monitoring locations on a 32 by 32 grid, of which 999 cells are valid.

## Source files and preprocessing

The [experiment guide](../experiments/README.md#prepare-the-model-inputs) shows the required source tree and the commands to prepare it. Set `SCUNET_RAW_ROOT` to the directory containing `grid_label.csv` and the `no2`, `o3` and `pm2_5` source folders. Each pollutant folder supplies its own sparse label array (`*_label.npy`) and feature array (`*_features.npy`).

The [paper preprocessing script](../tools/preprocessing/zh_final_paper.py) writes the model inputs to `$SCUNET_PREPROCESS_ROOT/data/{no2,o3,pm25}/`. Pass `$SCUNET_PREPROCESS_ROOT/data` as `--data-root`. If you already have the prepared arrays in a different location, pass the directory that contains all three pollutant subdirectories. The legacy [zh.py](../tools/preprocessing/zh.py) records an earlier preprocessing version.

## Files in each prepared pollutant folder

| File | Meaning |
|---|---|
| `conc_img_raw.npy` | Sparse observed concentrations in physical units, shape `(T, 32, 32)`. |
| `cov_img_full.npy` | Full-grid covariates, shape `(T, F, 32, 32)`. The feature count is 39 for NO2 and O3 and 40 for PM2.5. |
| `coord_img.npy` | Grid coordinate channels, shape `(32, 32, 2)`. |
| `station_positions.npy` | Row and column indices in the fixed station order, shape `(55, 2)`. |
| `valid_time_indices.npy` | Original hourly indices. |
| `time_splits.npy` | Dictionary of array indices for `train`, `val` and `test`. |
| `valid_grid_mask.npy` | Binary valid-grid mask, shape `(32, 32)`. |

The loader also accepts `conc_img_full.npy` with its corresponding `conc_scaler.pkl`. It restores raw concentration units before fitting the scaler for each outer fold. The original hourly identifiers and the row indices in the aligned arrays serve different purposes.

The [reference protocols](reference/D3) record the original resolved covariate groups and run settings. The [fixed station partition](spatial_split.json) and [station list](station_blocks.csv) are in this directory. The [main README](../README.md) gives the data citation and the coverage of the public release.
