# README

## Overview

The script, `prep_limit.py` performs Gaussian Process Regression (GPR) on data with signal extraction features. It is steered by the config file `fit_config.yml`.

The steering config `fit_config.yml` allows one to specify various things including those pertaining to data loading (histogram names, ranges and rebinning), signal features (mass, cross section and integrated luminosity), signal injection and signal extraction (background-only, template or parametric fit to signal). It also allows for the specification of a mean function to be passed to the GP as a prior provided it is already declared in `prep_limit.py`. Kernel and function initial parameters and bounds are declared in the config with any additional hardcoding done in the running script.

## Requirements

- Python 3.x
- ROOT
- NumPy
- SciPy
- Matplotlib
- mplhep
- iminuit
- argparse

Requirements can also be installed through conda with the `environment.yml` file as follows
```
conda env create -f environment.yml
```

## Usage

The script can be run with the following command:

```bash
python3 gp4hep/prep_limit.py --input_file /home/pb4918/Physics/Projects/L1Scouting/GPR/GPRJul25/Data/2025/processed_data_2025_1gev_50ls.root --input_file_sig /home/pb4918/Physics/Projects/L1Scouting/GPR/GPRJul25/Data/Signal/processed_output_ztoqq_mlm_350.root --output test --config gp4hep/fit_config.yml
```

### Options
- `--input_file`: Path to the input ROOT file with data (default: resolved2016_reg2.root).
- `--input_file_sig`: Path to the input ROOT file with signal (default: signal.root).
- `--output`: Output directory where plots are saved (default: test)
- `--config`: Path to the steering config (default: fit_config.yaml)
