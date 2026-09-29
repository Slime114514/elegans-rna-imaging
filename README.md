# Analysis code: RNA-triggered fluorescence imaging in C. elegans

Companion code for the manuscript:

> **RNA-triggered fluorescence enables live spatial mapping of endogenous RNA-associated dynamics in C. elegans**

The repository contains the image-analysis and statistics pipeline used for confocal recordings of fluorescent reporters in *C. elegans* (skn-1, daf-16, col-10), covering continuous-stimulation response analysis, long-term imaging series, negative-control handling, and vitality assays (body length, egg shape, brood size, lifespan).

## Contents

| Path | Purpose |
| --- | --- |
| `vsi_to_pixel_data.py` | Converts Olympus `.vsi` recordings to per-instance pixel CSV files (interactive segmentation tool with a napari GUI). |
| `negative_control_calculator.py` | Computes the mean brightness of the negative-control group from pixel CSV files. |
| `h_net_adder.py` | Adds the background-corrected `h_net` column to pixel CSV files. The negative-control mean is entered at runtime. |
| `continuous/` | Shared settings in `00_config_utils.py`, analysis modules `01`-`08` (binning, response statistics, robustness filtering, figures), and `run_all_pipeline.py` to run the whole chain. |
| `longterm/` | Shared settings in `longterm_00_config_utils.py`, modules `01`-`02` for the day 1-16 series, and `longterm_run_all.py` to run both steps. |
| `negative/` | Negative-control figure and summary statistics (3x3 t-stack). |
| `vitality/` | Combined figure for body length, egg shape, brood size, and lifespan (Kaplan-Meier curves with log-rank tests). |

## Requirements

Two conda environments are provided because the extraction tool and the analysis pipeline need different environments:

- `environment.yml`: main analysis environment (Python 3.8). Covers everything under `continuous/`, `longterm/`, `negative/`, and `vitality/`, plus `negative_control_calculator.py` and `h_net_adder.py`.
- `environment-vsi.yml`: environment for `vsi_to_pixel_data.py` (Python 3.9). The versions are pinned exactly because the Bioformats and napari stack is sensitive to package conflicts. Do not upgrade individual packages.

```bash
# Main analysis environment
conda env create -f environment.yml
conda activate longevity_plot

# Extraction tool environment
conda env create -f environment-vsi.yml
conda activate Py_39_vsi
```

The pipeline was developed and tested on Windows.

## Configuring paths

Each script has a marked settings block ("User-adjustable settings" or "User paths"). Replace the `path/to/your/...` placeholders before running.

| Script | What to set |
| --- | --- |
| `vsi_to_pixel_data.py` | `input_dir` (folder with `.vsi` files) and `output_dir` (folder for CSV output). |
| `negative_control_calculator.py` | `input_folder`: pixel CSV files of the negative-control strain. |
| `h_net_adder.py` | `input_folder`: pixel CSV files to correct. The negative-control value is requested at runtime. |
| `continuous/00_config_utils.py` | `INPUT_GROUPS`: one folder of `*_pixels_h_net.csv` files per gene and condition; `OUTPUT_DIR` for results. |
| `longterm/longterm_00_config_utils.py` | `DAY_FOLDERS`: `h_net` folders for days 1, 4, 7, 10, 13, and 16; `OUTPUT_DIR` for results. |
| `negative/negative_3x3_tstack.py` | `NEG_FOLDER`, `EXP_FOLDER`, `OUTPUT_PATH`. |
| `vitality/vitality_plot.py` | Four measurement workbooks and `OUTPUT_PATH`. |

## Running the pipeline

Pixel extraction from raw recordings (optional if pixel CSV files already exist), in the `Py_39_vsi` environment:

```bash
python vsi_to_pixel_data.py
```

The tool opens a napari window, asks for an experiment name and processing mode, and writes one CSV per instance.

The remaining steps run in the `longevity_plot` environment:

```bash
# 1. Negative-control mean brightness (keep the printed value for the next step)
python negative_control_calculator.py

# 2. Background correction, adds the h_net column
python h_net_adder.py

# 3. Continuous-stimulation analysis, modules 01-08
python continuous/run_all_pipeline.py

# 4. Day 1-16 series, modules 01-02
python longterm/longterm_run_all.py

# 5. Standalone figures
python negative/negative_3x3_tstack.py
python vitality/vitality_plot.py
```

The two `run_all` scripts run their modules in order and skip steps whose outputs already exist. A stop file can interrupt a run; modules also run individually (see `clear_pipeline_stop.py` and `request_pipeline_stop.py` in each pipeline folder).

## Data

Raw imaging recordings and the measurement workbooks are not included in this repository. They are available from the corresponding author upon reasonable request.

The pipeline expects pixel CSV files that contain, for each pixel, the binned coordinates `s` and `r`, the measured signal `h`, and an `is_signal` flag. `h_net_adder.py` adds the background-corrected `h_net` column used by the downstream modules.

## Citation

If you use this code, please cite the manuscript above.

## License

MIT. See `LICENSE`.

## Contact

Please open an issue in this repository for questions about running the code.
