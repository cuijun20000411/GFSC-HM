# GFSC-UNet-HM

Code and evaluation records for *Single-timestamp reconstruction of air pollutant concentration fields under sparse monitoring using hybrid spatial masking and multiscale covariate fusion*.

## Repository layout

| Directory | Contents |
| --- | --- |
| [analysis/](analysis/) | Statistical tests, matched-seed summaries, spatial diagnostics and plotting |
| [experiments/](experiments/) | Training, baselines, ablations, experiment launchers and released predictions |
| [protocol/](protocol/) | Fixed station folds, original reference settings, data format and source hashes |
| [tools/](tools/) | Data preprocessing, inference benchmarking and optional dependencies |

## Check the results

Use Python 3.10 or later. The following checks use NumPy and pandas. For model training, install the dependencies in [requirements.txt](requirements.txt).

Run from the repository root.

    python experiments/code/test_hybrid_masking_logic.py

    python analysis/scripts/summarize_hm_rm_matched_spatial.py \
      --rm-seed42-root experiments/results/spatial_block/rm_seed42 \
      --rm-multiseed-root experiments/results/rm_multiseed \
      --hm-seed42-root experiments/results/spatial_block/hm_seed42 \
      --hm-multiseed-root experiments/results/hm_multiseed \
      --output-dir experiments/results/runs/matched_seed_summary

    python analysis/scripts/spatial_anomaly_pairwise_analysis.py \
      --loso analysis/results/spatial_pattern/loso_inputs \
      --coordinates protocol/station_blocks.csv \
      --output experiments/results/runs/spatial_pattern

- [Fusion comparison](experiments/results/spatial_block/structure/summary/paper_table_spatial_structure_ablation.csv). Pooled NO2 RMSE is 4.0619 for multiscale fusion and 4.6372 for input concatenation.
- [Matched-seed comparison](experiments/results/hm_multiseed/metrics/paper_matched_seed_rmse_means.csv). Mean pooled NO2 RMSE is 4.3507 for RM and 4.1725 for HM. The manuscript reports the relative change between the two means (4.09% lower). The summary script also reports the mean of the per-seed percentage changes.
- [Spatial diagnostics](analysis/results/spatial_pattern/). The supplied zero-additional-hiding LOSO predictions support regeneration of the spatial-anomaly and station-pair summaries.

## Train a model

See [Experiments](experiments/README.md) for training entry points and [Model input data](protocol/DATA_FORMAT.md) for file formats. The source dataset is described by Xu et al. (doi:10.1016/j.ipm.2025.104474). Training uses separately obtained input arrays. The release provides selected predictions and metrics. The density-stress experiments are represented by their protocols, scenarios and metrics. Retraining requires the input arrays. Recreating the full-field visualizations requires the trained model or its numerical field outputs. Recreating Fig. 9 requires recovery of the original permutation script.

Code is distributed under the [MIT License](LICENSE). The source dataset is subject to its own terms.
