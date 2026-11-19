# Model input data

The training scripts use 720 aligned hours for each pollutant, with 504 training, 108 validation and 108 test samples. Each pollutant uses the same 55 candidate monitoring locations on a 32 by 32 grid, of which 999 cells are valid.

Provide a directory for each pollutant under experiments/data/processed (no2, o3 and pm25). The input files are as follows.

| File | Meaning |
| --- | --- |
| conc_img_raw.npy | Sparse observed concentrations in physical units (T, 32, 32) |
| cov_img_full.npy | Full-grid covariates (T, F, 32, 32), with F equal to 39 for NO2 and O3 and 40 for PM2.5 |
| coord_img.npy | Grid coordinate channels (32, 32, 2) |
| station_positions.npy | Row and column indices in the fixed station order (55, 2) |
| valid_time_indices.npy | Original hourly indices |
| time_splits.npy | Dictionary of array indices for train, val and test |
| valid_grid_mask.npy | Binary valid-grid mask (32, 32) |

The loader also accepts conc_img_full.npy with its corresponding conc_scaler.pkl. It restores raw concentration units before fitting the scaler for each outer fold. The original hourly identifiers and the row indices in the aligned arrays serve different purposes.

tools/preprocessing/zh_final_paper.py prepares the aligned data from the source files. Set SCUNET_RAW_ROOT to the input root containing the no2, o3 and pm2_5 source folders, then set SCUNET_PREPROCESS_ROOT to the output root. The script reads the sparse label arrays and corresponding feature arrays. See the Config class for the input file names. The legacy tools/preprocessing/zh.py script records an earlier preprocessing version.

The reference protocols in protocol/reference/D3 record the original resolved covariate groups and run settings. The fixed station partition and station list are in protocol. The data citation and package coverage are given in the main README.
