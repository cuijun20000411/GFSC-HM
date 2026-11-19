# Experiments

D3 denotes GFSC-UNet-RM and D3HM denotes GFSC-UNet-HM in the archived filenames and outputs. The original model code is kept together under code/ so its imports and file fingerprint checks resolve.

| Experiment | Entry points |
| --- | --- |
| GFSC-UNet-HM strict LOSO, spatial-block holdout and field visualization | [gfsc_unet_d3_hybrid_mask_main.py](code/gfsc_unet_d3_hybrid_mask_main.py) |
| Strict LOSO baselines | [baseline_final_strict.py](code/baseline_final_strict.py), [pconv_covariate_final_strict.py](code/pconv_covariate_final_strict.py) |
| Model evolution and component ablations | [f3_direct_ablation_final_strict.py](code/f3_direct_ablation_final_strict.py), [d3_direct_ablation_final_strict.py](code/d3_direct_ablation_final_strict.py) |
| Spatial-block baselines and fusion ablations | [d3hm_spatial_block_classical_baselines.py](code/d3hm_spatial_block_classical_baselines.py), [gfsc_unet_d3hm_a10_spatial_block.py](code/gfsc_unet_d3hm_a10_spatial_block.py), [gfsc_unet_hm_spatial_structure_ablation.py](code/gfsc_unet_hm_spatial_structure_ablation.py) |
| HM and RM multi-seed experiment launchers | [HM launcher](runners/01_d3hm_multiseed_spatial_block.py), [RM launcher](runners/run_rm_multiseed_spatial.py) |
| Test-time station reduction | [d3hm_density_stress_final_strict.py](code/d3hm_density_stress_final_strict.py), [density_stress_final_strict.py](code/density_stress_final_strict.py) |
| GPU forward-pass benchmark | [03_d3hm_complexity_efficiency.py](../tools/benchmark/03_d3hm_complexity_efficiency.py) |

## Main model

Place the arrays described in [Model input data](../protocol/DATA_FORMAT.md) under experiments/data/processed/. From the repository root, check the config and inputs with:

    python experiments/code/gfsc_unet_d3_hybrid_mask_main.py \
      --mode spatial-block \
      --data-root experiments/data/processed \
      --d3-reference-root protocol/reference/D3 \
      --spatial-split-json protocol/spatial_split.json \
      --output experiments/results/runs/hm_seed42 \
      --preflight

Remove the preflight flag to fit the model. Set mode to loso for leave-one-station-out evaluation. The structural ablation runner also accepts the fixed split via the spatial-split-json option. The [optional requirements](../tools/requirements-optional.txt) include PyKrige for ordinary kriging.

## Matched seeds

Run the original RM block training on Linux or WSL because it uses fcntl file locks. The launchers start their child processes in the model source directory. Pass absolute paths for the data, protocol and output directories. With Bash, start from the repository root:

    repo="$PWD"
    python "$repo/experiments/runners/01_d3hm_multiseed_spatial_block.py" \
      --main-script "$repo/experiments/code/gfsc_unet_d3_hybrid_mask_main.py" \
      --data-root "$repo/experiments/data/processed" \
      --d3-reference-root "$repo/protocol/reference/D3" \
      --spatial-split-json "$repo/protocol/spatial_split.json" \
      --reference-seed-root "$repo/experiments/results/spatial_block/hm_seed42" \
      --output "$repo/experiments/results/runs/hm_multiseed" \
      --preflight

    python "$repo/experiments/runners/run_rm_multiseed_spatial.py" \
      --script "$repo/experiments/code/d3_spatial_block_final_strict.py" \
      --data-root "$repo/experiments/data/processed" \
      --output-root "$repo/experiments/results/runs/rm_multiseed" \
      --preflight-only

The HM launcher uses the packaged seed-42 reference results. The RM launcher checks the original source hashes and then starts its model-specific preflight. Remove the respective preflight flag to run seeds 43 and 44.

## Reading the baselines and timing

The manuscript's PConv-UNet-Cov records are in results/loso/baselines/pconv_covariate/. The standard-baseline results also retain an earlier PartialConvUNet run. For the benchmark, pass the local main script with d3hm-main and supply model/checkpoint paths with baseline. The archived timing settings and measurements are in results/efficiency/.
